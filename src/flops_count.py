import torch
import torch.nn as nn
import inspect
from fvcore.nn import FlopCountAnalysis
from adaptive_full_rank import HeadGater
# Safe-import jit handles (not all exist in every fvcore version)
try:
    from fvcore.nn.jit_handles import addmm_flop_jit
except Exception:
    addmm_flop_jit = None
try:
    from fvcore.nn.jit_handles import bmm_flop_jit
except Exception:
    bmm_flop_jit = None
try:
    from fvcore.nn.jit_handles import matmul_flop_jit
except Exception:
    matmul_flop_jit = None

from DiT import create_model
from low_rank_compression import low_rank_layer_replacement, TimestepConditionedWrapper
from config import TrainingConfig, print_config
import warnings
import time
try:
    # diffusers >=0.21
    from diffusers import DDIMScheduler
except Exception:
    # explicit submodule import fallback
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _maybe_get_num_classes(model):
    """Try common attribute names to detect class-conditional setups."""
    for attr in ("num_classes", "n_classes", "classes", "cfg_num_classes"):
        if hasattr(model, attr):
            try:
                val = int(getattr(model, attr))
                if val > 0:
                    return val
            except Exception:
                pass
    # Try pulling from diffusers-style config if present
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("num_classes", "num_class_embeds", "num_labels", "class_labels"):
            if hasattr(cfg, attr):
                try:
                    val = int(getattr(cfg, attr))
                    if val > 0:
                        return val
                except Exception:
                    pass
    # As a fallback, scan for LabelEmbedding modules to infer num_embeddings
    try:
        for m in model.modules():
            if type(m).__name__ == "LabelEmbedding":
                table = getattr(m, "embedding_table", None)
                if table is not None and hasattr(table, "num_embeddings"):
                    num = int(getattr(table, "num_embeddings", 0))
                    if num > 0:
                        return num
    except Exception:
        pass
    return None


class DiTForwardWrapper(nn.Module):
    """
    Wraps a DiT-like model so FlopCountAnalysis can be run with a single Tensor input.
    - Stores timesteps and (optional) class labels as buffers.
    - Calls the wrapped model with the right argument names if present.
    - Returns a Tensor if the model returns an object/dict with .sample.
    """
    def __init__(self, model: nn.Module, t: torch.Tensor, y: torch.Tensor | None):
        super().__init__()
        self.model = model
        self.register_buffer("t", t)
        if y is not None:
            self.register_buffer("y", y)
        else:
            self.y = None
        self._sig = inspect.signature(model.forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build kwargs based on the model's forward signature
        kwargs = {}
        # common timestep arg names in diffusion/DiT codebases
        if "t" in self._sig.parameters:
            kwargs["t"] = self.t
        elif "timesteps" in self._sig.parameters:
            kwargs["timesteps"] = self.t
        elif "timestep" in self._sig.parameters:
            kwargs["timestep"] = self.t

        # class labels (optional)
        if self.y is not None:
            if "y" in self._sig.parameters:
                kwargs["y"] = self.y
            elif "class_labels" in self._sig.parameters:
                kwargs["class_labels"] = self.y
            elif "labels" in self._sig.parameters:
                kwargs["labels"] = self.y
        else:
            # If the model accepts class labels but none were provided, default to zeros
            # to satisfy modules that require label embeddings during forward.
            if "y" in self._sig.parameters:
                kwargs["y"] = torch.zeros_like(self.t, dtype=torch.long)
            elif "class_labels" in self._sig.parameters:
                kwargs["class_labels"] = torch.zeros_like(self.t, dtype=torch.long)
            elif "labels" in self._sig.parameters:
                kwargs["labels"] = torch.zeros_like(self.t, dtype=torch.long)

        out = self.model(x, **kwargs)
        # Normalize outputs to a Tensor for fvcore
        if hasattr(out, "sample"):     # diffusers-style output
            return out.sample
        if isinstance(out, dict) and "sample" in out:
            return out["sample"]
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        return out  # hope it's a Tensor


def _forward_with_auto_args(model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
    """Call model.forward using detected arg names; normalize output to Tensor."""
    sig = inspect.signature(model.forward)
    kwargs = {}
    if "t" in sig.parameters:
        kwargs["t"] = t
    elif "timesteps" in sig.parameters:
        kwargs["timesteps"] = t
    elif "timestep" in sig.parameters:
        kwargs["timestep"] = t
    if y is not None:
        if "y" in sig.parameters:
            kwargs["y"] = y
        elif "class_labels" in sig.parameters:
            kwargs["class_labels"] = y
        elif "labels" in sig.parameters:
            kwargs["labels"] = y
    else:
        if "y" in sig.parameters:
            kwargs["y"] = torch.zeros_like(t, dtype=torch.long)
        elif "class_labels" in sig.parameters:
            kwargs["class_labels"] = torch.zeros_like(t, dtype=torch.long)
        elif "labels" in sig.parameters:
            kwargs["labels"] = torch.zeros_like(t, dtype=torch.long)

    out = model(x, **kwargs)
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, dict) and "sample" in out:
        return out["sample"]
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    return out


def _register_common_op_handles(flops: FlopCountAnalysis):
    """
    Register flop handles that exist across fvcore versions.
    Map mm -> matmul handler when mm-specific handle is unavailable.
    """
    if addmm_flop_jit is not None:
        flops.set_op_handle("aten::addmm", addmm_flop_jit)
    if bmm_flop_jit is not None:
        flops.set_op_handle("aten::bmm", bmm_flop_jit)
    if matmul_flop_jit is not None:
        # cover both mm and matmul
        flops.set_op_handle("aten::matmul", matmul_flop_jit)
        flops.set_op_handle("aten::mm", matmul_flop_jit)


def _sdpa_flop_handle(inputs, outputs):
    """
    Approx FLOP count for PyTorch 2.x scaled_dot_product_attention kernels.

    Assumes inputs like:
      q: (B, H, L, D)
      k: (B, H, S, D)
      v: (B, H, S, D)
    Ignores masking/causal specifics beyond dimensions.
    Counts:
      - QK^T  : 2 * B * H * L * S * D
      - softmax (exp + norm approx): B * H * L * S
      - (softmax @ V): 2 * B * H * L * S * D
    Total ≈ 4 * B * H * L * S * D + B * H * L * S
    """
    try:
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]
        # Some backends pass view tensors; we only need shapes
        B, H, L, D = q.shape
        Bk, Hk, S, Dk = k.shape
        # Basic sanity
        if (Bk != B) or (Hk != H) or (Dk != D):
            # fall back cautiously
            L = q.shape[-2]
            S = k.shape[-2]
            H = q.shape[-3]
            B = int(q.numel() // (H * L * D))
        # FLOPs (float)
        flops = 4.0 * B * H * L * S * D + (B * H * L * S)
        return flops
    except Exception:
        # If shapes not as expected, return 0 to avoid crashing
        return 0.0


def _register_sdpa_handles(flops: FlopCountAnalysis):
    """
    Register handles for the various SDPA operator names used across PyTorch builds.
    If the op isn't present in your run, set_op_handle is harmless.
    """
    sdpa_ops = [
        "aten::scaled_dot_product_attention",
        "aten::scaled_dot_product_attention_math",
        "aten::scaled_dot_product_attention_default",
        "aten::scaled_dot_product_attention_efficient_attention",
        "aten::scaled_dot_product_attention_flash_attention",
    ]
    for op in sdpa_ops:
        try:
            flops.set_op_handle(op, _sdpa_flop_handle)
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------------------

def calculate_model_flops(model, input_shape, device="cpu", verbose=True, max_timesteps=1000):
    """
    Calculate FLOPs for a given model using fvcore.Fl opCountAnalysis.

    Args:
        model: PyTorch model
        input_shape: dict with keys {'channels','height','width'}
        device: 'cpu' or 'cuda'
        verbose: print human-friendly summary
        max_timesteps: upper bound for random timestep (exclusive)

    Returns:
        dict with FLOP totals and breakdowns.
    """
    model.eval().to(device)

    # Dummy inputs
    B = 1
    C, H, W = input_shape["channels"], input_shape["height"], input_shape["width"]
    x = torch.randn(B, C, H, W, device=device)

    # t = torch.randint(0, max_timesteps, (B,), dtype=torch.long, device=device)
    t = torch.randint(999, 1000, (B,), dtype=torch.long, device=device)

    num_classes = _maybe_get_num_classes(model)
    y = None
    if num_classes is not None and num_classes > 1:
        y = torch.randint(0, num_classes, (B,), dtype=torch.long, device=device)

    wrapped = DiTForwardWrapper(model, t, y).to(device)

    with torch.no_grad():
        flops = FlopCountAnalysis(wrapped, (x,))
        _register_common_op_handles(flops)
        _register_sdpa_handles(flops)

        total = flops.total()
        by_op = flops.by_operator()
        by_module = flops.by_module()

    if verbose:
        print(f"Total FLOPs: {total:,}")
        print(f"Total FLOPs (M): {total / 1e6:.2f}")
        print(f"Total FLOPs (G): {total / 1e9:.2f}")
        unsupported = flops.unsupported_ops()
        if unsupported:
            print("\n[Warning] Unsupported ops encountered (not counted):")
            for k, v in unsupported.items():
                print(f"  {k}: {v}x")

    return {
        "total_flops": int(total),
        "flops_M": total / 1e6,
        "flops_G": total / 1e9,
        "by_operator": by_op,     # dict: op -> flops
        "by_module": by_module,   # dict: module_name -> flops
    }


def _build_ddim_timesteps(step_stride: int = 10, include_999: bool = True) -> torch.Tensor:
    """Create descending timesteps like [999, 990, ..., 0] for DDIM."""
    steps = list(range(0, 1000, step_stride))
    if include_999 and 999 not in steps:
        steps.append(999)
    steps = sorted(set(s for s in steps if 0 <= s <= 999))
    return torch.tensor(list(reversed(steps)), dtype=torch.long)


def simulate_ddim_flops_and_time(model: nn.Module, input_shape: dict, device: torch.device, step_stride: int = 10, batch_size: int = 1, verbose: bool = True):
    """
    Simulate DDIM inference over timesteps [0, step_stride, ..., 999] (descending order for scheduler),
    measure wall-clock time, and estimate total FLOPs as per-step FLOPs × num_steps.
    """
    model.eval().to(device)
    C, H, W = input_shape["channels"], input_shape["height"], input_shape["width"]
    B = batch_size
    x = torch.randn(B, C, H, W, device=device)

    # Class labels if model is conditional
    num_classes = _maybe_get_num_classes(model)
    y = None
    if num_classes is not None and num_classes > 1:
        y = torch.zeros(B, dtype=torch.long, device=device)

    # Build custom timesteps and scheduler
    timesteps = _build_ddim_timesteps(step_stride=step_stride, include_999=True).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=len(timesteps), device=device)
    # Override with our exact list
    scheduler.timesteps = timesteps

    # FLOPs per step (compute for each DDIM timestep)
    per_step_flops_list = []  # list[(t_int, flops_int)]
    with torch.no_grad():
        for t in timesteps:
            tt = torch.full((B,), int(t.item()), dtype=torch.long, device=device)
            wrapped = DiTForwardWrapper(model, t=tt, y=y).to(device)
            flops = FlopCountAnalysis(wrapped, (x,))
            _register_common_op_handles(flops)
            _register_sdpa_handles(flops)
            per_step_flops_list.append((int(t.item()), int(flops.total())))

    # Time the DDIM loop (forward + scheduler step)
    latents = torch.randn_like(x)
    start = time.time()
    with torch.no_grad():
        for t in timesteps:
            tt = torch.full((B,), int(t.item()), dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            pred = _forward_with_auto_args(model, latents, tt, y)
            # Some models return channels-first prediction matching latents
            latents = scheduler.step(model_output=pred, timestep=t, sample=latents).prev_sample
            if device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.time() - start

    total_steps = len(timesteps)
    total_flops = sum(fl for _, fl in per_step_flops_list)
    if verbose:
        print("\nDDIM Simulation (stride={}):".format(step_stride))
        print(f"Timesteps (desc): {timesteps.tolist()}")
        for t_int, fl in per_step_flops_list:
            print(f"t={t_int:3d} : {fl / 1e9:.3f} G FLOPs")
        print(f"Total steps: {total_steps}")
        print(f"Total FLOPs: {total_flops / 1e9:.3f} G")
        print(f"Wall-clock time: {elapsed:.3f} s  (B={B})")

    return {
        "per_step_flops": per_step_flops_list,
        "total_steps": total_steps,
        "total_flops": total_flops,
        "elapsed_s": elapsed,
        "timesteps": timesteps.tolist(),
    }


def count_parameters(model, verbose=True):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters (M): {total_params / 1e6:.2f}")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_M": total_params / 1e6,
    }


def analyze_layer_types(model, verbose=True):
    """
    Analyze the types of layers in the model (without double-counting params).
    Uses recurse=False so each module reports its own parameters only.
    """
    layer_counts = {}
    param_counts = {}

    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1

        # Count only parameters owned by this module
        own_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        param_counts[module_type] = param_counts.get(module_type, 0) + own_params

    if verbose:
        print("\nLayer Type Analysis:")
        print("-" * 50)
        for layer_type, count in sorted(layer_counts.items()):
            params = param_counts.get(layer_type, 0)
            print(f"{layer_type}: {count} layers, {params:,} parameters ({params/1e6:.2f}M)")

    return layer_counts, param_counts


# --------------------------------------------------------------------------------------
# Main comparison
# --------------------------------------------------------------------------------------

def main():
    """Compare FLOPs between full rank and low rank DiT models."""
    print("=" * 80)
    print("DiT Model FLOP Analysis")
    print("=" * 80)

    # Create configuration
    config = TrainingConfig()
    config.train_batch_size = 1  # FLOPs typically reported at batch=1

    # Define input shape (e.g., CIFAR-10: 3x32x32)
    input_shape = {
        "channels": getattr(config, "pixel_channels", 3),
        "height": getattr(config, "image_size", 32),
        "width": getattr(config, "image_size", 32),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input shape: {input_shape}\n")

    # ============================================
    # Full Rank Model Analysis
    # ============================================
    print("1. FULL RANK DiT MODEL")
    print("-" * 40)

    full_rank_model = create_model(config)
    full_rank_model = HeadGater(full_rank_model,     
                      max_timestep=config.num_training_steps,
                      min_ratio=config.rank_min_ratio,
                      schedule=config.rank_schedule,
                      debug=False,)

    print("Parameter Analysis:")
    full_rank_params = count_parameters(full_rank_model, verbose=True)

    _ = analyze_layer_types(full_rank_model, verbose=True)

    print("\nFLOP Analysis:")
    full_rank_flops = calculate_model_flops(full_rank_model, input_shape, device, verbose=True)

    print("\n" + "=" * 80)

    # ============================================
    # Low Rank Model Analysis
    # ============================================
    print("2. LOW RANK DiT MODEL")
    print("-" * 40)

    low_rank_model = create_model(config)

    compression_percentage = 1.0
    print(f"Applying low rank compression with {compression_percentage*100:.0f}% rank...")
    low_rank_model = low_rank_layer_replacement(low_rank_model, percentage=compression_percentage)

    print("\nParameter Analysis:")
    low_rank_params = count_parameters(low_rank_model, verbose=True)

    _ = analyze_layer_types(low_rank_model, verbose=True)

    print("\nFLOP Analysis:")
    low_rank_flops = calculate_model_flops(low_rank_model, input_shape, device, verbose=True)

    print("\n" + "=" * 80)

    # ============================================
    # Low Rank Model with Timestep Conditioning
    # ============================================
    print("3. LOW RANK DiT MODEL WITH TIMESTEP CONDITIONING")
    print("-" * 50)

    adaptive_config = TrainingConfig()
    adaptive_config.timestep_conditioning = True

    adaptive_model = create_model(adaptive_config)
    adaptive_model = low_rank_layer_replacement(adaptive_model, percentage=compression_percentage, config=adaptive_config)
    adaptive_model = TimestepConditionedWrapper(adaptive_model, adaptive_config)

    print("Parameter Analysis:")
    adaptive_params = count_parameters(adaptive_model, verbose=True)

    print("\nFLOP Analysis:")
    adaptive_flops = calculate_model_flops(adaptive_model, input_shape, device, verbose=True)

    print("\n" + "=" * 80)

    # ============================================
    # DDIM Simulation
    # ============================================
    print("4. DDIM FLOP SIMULATION")
    print("-" * 40)

    print("Full Rank DDIM:")
    full_rank_ddim_sim = simulate_ddim_flops_and_time(full_rank_model, input_shape, device, verbose=True)

    print("\nLow Rank DDIM:")
    low_rank_ddim_sim = simulate_ddim_flops_and_time(low_rank_model, input_shape, device, verbose=True)

    print("\nAdaptive DDIM:")
    adaptive_ddim_sim = simulate_ddim_flops_and_time(adaptive_model, input_shape, device, verbose=True)

    print("\n" + "=" * 80)

    # ============================================
    # Comparison and Summary
    # ============================================
    print("5. COMPARISON SUMMARY")
    print("-" * 40)

    print("Parameter Comparison:")
    print(f"Full Rank:     {full_rank_params['params_M']:8.2f}M parameters")
    print(f"Low Rank:      {low_rank_params['params_M']:8.2f}M parameters")
    print(f"Adaptive:      {adaptive_params['params_M']:8.2f}M parameters")

    param_reduction = (1 - low_rank_params["total_params"] / full_rank_params["total_params"]) * 100
    adaptive_param_reduction = (1 - adaptive_params["total_params"] / full_rank_params["total_params"]) * 100

    print(f"\nParameter Reduction:")
    print(f"Low Rank:      {param_reduction:8.2f}% reduction")
    print(f"Adaptive:      {adaptive_param_reduction:8.2f}% reduction")

    print(f"\nFLOP Comparison:")
    print(f"Full Rank:     {full_rank_flops['flops_G']:8.2f}G FLOPs")
    print(f"Low Rank:      {low_rank_flops['flops_G']:8.2f}G FLOPs")
    print(f"Adaptive:      {adaptive_flops['flops_G']:8.2f}G FLOPs")

    flop_reduction = (1 - low_rank_flops["total_flops"] / full_rank_flops["total_flops"]) * 100
    adaptive_flop_reduction = (1 - adaptive_flops["total_flops"] / full_rank_flops["total_flops"]) * 100

    print(f"\nFLOP Reduction:")
    print(f"Low Rank:      {flop_reduction:8.2f}% reduction")
    print(f"Adaptive:      {adaptive_flop_reduction:8.2f}% reduction")

    print(f"\nEfficiency Metrics (FLOPs / Param):")
    print(f"Full Rank:     {full_rank_flops['total_flops'] / full_rank_params['total_params']:8.2f}")
    print(f"Low Rank:      {low_rank_flops['total_flops'] / low_rank_params['total_params']:8.2f}")
    print(f"Adaptive:      {adaptive_flops['total_flops'] / adaptive_params['total_params']:8.2f}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")

    return {
        "full_rank": {"params": full_rank_params, "flops": full_rank_flops},
        "low_rank": {"params": low_rank_params, "flops": low_rank_flops},
        "adaptive": {"params": adaptive_params, "flops": adaptive_flops},
    }


if __name__ == "__main__":
    results = main()
