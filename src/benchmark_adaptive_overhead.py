import time
import torch
import torch.nn as nn
import inspect

from DiT import create_model
from config import TrainingConfig
from low_rank_compression import low_rank_layer_replacement, TimestepConditionedWrapper

try:
    # diffusers >=0.21
    from diffusers import DDIMScheduler
except Exception:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler


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


def _build_ddim_timesteps(step_stride: int = 10, include_999: bool = True) -> torch.Tensor:
    steps = list(range(0, 1000, step_stride))
    if include_999 and 999 not in steps:
        steps.append(999)
    steps = sorted(set(s for s in steps if 0 <= s <= 999))
    return torch.tensor(list(reversed(steps)), dtype=torch.long)


def _count_low_rank_linear_modules(model: nn.Module) -> int:
    from low_rank_compression import LowRankLinear
    return sum(1 for m in model.modules() if isinstance(m, LowRankLinear))


@torch.no_grad()
def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base config and input shape
    config = TrainingConfig()
    B = 1
    C = getattr(config, "pixel_channels", 3)
    H = getattr(config, "image_size", 32)
    W = getattr(config, "image_size", 32)
    x = torch.randn(B, C, H, W, device=device)

    # DDIM setup (we time only model forward/injection, not scheduler overhead)
    timesteps = _build_ddim_timesteps(step_stride=10, include_999=True).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=len(timesteps), device=device)
    scheduler.timesteps = timesteps

    # Helper for optional labels
    def maybe_labels(model: nn.Module):
        # Try to infer num classes
        for attr in ("num_classes", "n_classes", "classes", "cfg_num_classes"):
            if hasattr(model, attr):
                try:
                    val = int(getattr(model, attr))
                    if val > 1:
                        return torch.zeros(B, dtype=torch.long, device=device)
                except Exception:
                    pass
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("num_classes", "num_class_embeds", "num_labels", "class_labels"):
                if hasattr(cfg, attr):
                    try:
                        val = int(getattr(cfg, attr))
                        if val > 1:
                            return torch.zeros(B, dtype=torch.long, device=device)
                    except Exception:
                        pass
        return None

    # --------------------------------------
    # 1) Full rank model
    # --------------------------------------
    full_model = create_model(config).eval().to(device)
    y_full = maybe_labels(full_model)
    latents = torch.randn_like(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    fwd_ms_full = []
    for t in timesteps:
        tt = torch.full((B,), int(t.item()), dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        pred = _forward_with_auto_args(full_model, latents, tt, y_full)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_ms_full.append((time.time() - start) * 1000.0)
        latents = scheduler.step(model_output=pred, timestep=t, sample=latents).prev_sample
    total_full_s = time.time() - t0
    print(f"Full Rank: steps={len(timesteps)} total={total_full_s:.3f}s avg_fwd={sum(fwd_ms_full)/len(fwd_ms_full):.3f}ms")

    # --------------------------------------
    # 2) Low rank model
    # --------------------------------------
    low_model = create_model(config)
    compression_percentage = 0.5
    low_model = low_rank_layer_replacement(low_model, percentage=compression_percentage)
    low_model.eval().to(device)
    y_low = maybe_labels(low_model)
    latents = torch.randn_like(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    fwd_ms_low = []
    for t in timesteps:
        tt = torch.full((B,), int(t.item()), dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        pred = _forward_with_auto_args(low_model, latents, tt, y_low)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_ms_low.append((time.time() - start) * 1000.0)
        latents = scheduler.step(model_output=pred, timestep=t, sample=latents).prev_sample
    total_low_s = time.time() - t0
    print(f"Low Rank:  steps={len(timesteps)} total={total_low_s:.3f}s avg_fwd={sum(fwd_ms_low)/len(fwd_ms_low):.3f}ms  LRLayers={_count_low_rank_linear_modules(low_model)}")

    # --------------------------------------
    # 3) Adaptive (Low rank + wrapper) with explicit timing of injection vs forward
    # --------------------------------------
    adaptive_cfg = TrainingConfig()
    adaptive_cfg.timestep_conditioning = True
    adaptive_model = create_model(adaptive_cfg)
    adaptive_model = low_rank_layer_replacement(adaptive_model, percentage=compression_percentage, config=adaptive_cfg)
    adaptive_wrapper = TimestepConditionedWrapper(adaptive_model, adaptive_cfg).eval().to(device)
    y_adapt = maybe_labels(adaptive_wrapper)
    latents = torch.randn_like(x)

    inject_ms_list = []
    fwd_ms_adapt = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for t in timesteps:
        tt = torch.full((B,), int(t.item()), dtype=torch.long, device=device)

        # Explicitly perform the wrapper's injection work and time it
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_inj = time.time()
        if adaptive_wrapper.training_config.timestep_conditioning and tt is not None:
            for i, layer in enumerate(adaptive_wrapper.low_rank_layers):
                if i in adaptive_wrapper.conditioning_enabled_layers:
                    layer._wrapper_timesteps = tt
                    layer._wrapper_T = adaptive_wrapper.training_config.num_training_steps
                    layer._wrapper_config = adaptive_wrapper.training_config
                else:
                    layer._wrapper_timesteps = None
                    layer._wrapper_T = None
                    layer._wrapper_config = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        inject_ms_list.append((time.time() - start_inj) * 1000.0)

        # Now measure just the base model forward
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_fwd = time.time()
        pred = _forward_with_auto_args(adaptive_wrapper.base_model, latents, tt, y_adapt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_ms_adapt.append((time.time() - start_fwd) * 1000.0)

        latents = scheduler.step(model_output=pred, timestep=t, sample=latents).prev_sample
    total_adapt_s = time.time() - t0

    print(
        "Adaptive: steps={} total={:.3f}s avg_inject={:.3f}ms avg_fwd={:.3f}ms  LRLayers={}".format(
            len(timesteps),
            total_adapt_s,
            sum(inject_ms_list) / len(inject_ms_list),
            sum(fwd_ms_adapt) / len(fwd_ms_adapt),
            _count_low_rank_linear_modules(adaptive_wrapper),
        )
    )

    # Optional: simple ratios
    print("\nRatios (Adaptive vs Low Rank):")
    print("  total:  {:.2f}x".format(total_adapt_s / max(total_low_s, 1e-6)))
    print("  fwd:    {:.2f}x (avg per step)".format((sum(fwd_ms_adapt) / len(fwd_ms_adapt)) / max(sum(fwd_ms_low) / len(fwd_ms_low), 1e-6)))
    print("  inject: {:.2f}ms/step (~overhead)".format(sum(inject_ms_list) / len(inject_ms_list)))


if __name__ == "__main__":
    benchmark()


