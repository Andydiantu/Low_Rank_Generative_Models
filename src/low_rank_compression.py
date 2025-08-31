import torch
import torch.nn as nn
import math


class TimestepConditionedWrapper(nn.Module):
    """
    Wrapper for DiT models to enable timestep-conditioned rank scheduling.
    Stores current timesteps and config, then injects them into LowRankLinear layers.
    """
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.training_config = config  # Rename to avoid confusion
        self.current_timesteps = None
        
        # Store reference to all LowRankLinear layers for timestep injection
        self.low_rank_layers = []
        self.low_rank_layer_names = []  # Store corresponding names
        self._collect_low_rank_layers(base_model)
        self.timestep_lower_bound = None # When equals None, no curriculum learning slicing is applied
        
        # Determine which layers should use timestep conditioning
        self.conditioning_enabled_layers = self._determine_conditioning_layers()
    
    @property
    def config(self):
        """Forward config attribute to base model."""
        return self.base_model.config
    
    @property
    def dtype(self):
        """Forward dtype attribute to base model."""
        return self.base_model.dtype
    
    @property
    def device(self):
        """Forward device attribute to base model."""
        return next(self.base_model.parameters()).device
    
    def _collect_low_rank_layers(self, module, prefix=""):
        """Recursively collect all LowRankLinear layers with their names."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LowRankLinear):
                self.low_rank_layers.append(child)
                self.low_rank_layer_names.append(full_name)
            else:
                self._collect_low_rank_layers(child, prefix=full_name)
    
    def _determine_conditioning_layers(self):
        """Determine which layers should use timestep conditioning based on config."""
        if (self.training_config.timestep_conditioning_first_n_blocks == 0 and 
            self.training_config.timestep_conditioning_last_n_blocks == 0):
            # Apply to all layers (original behavior)
            return set(range(len(self.low_rank_layers)))
        
        # Count total transformer blocks by looking at layer names
        transformer_blocks = set()
        for name in self.low_rank_layer_names:
            if "transformer_blocks." in name:
                # Extract block number from name like "transformer_blocks.5.attn.to_q"
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "transformer_blocks":
                    try:
                        block_idx = int(parts[1])
                        transformer_blocks.add(block_idx)
                    except ValueError:
                        continue
        
        total_blocks = len(transformer_blocks)
        if total_blocks == 0:
            # No transformer blocks found, apply to all layers
            return set(range(len(self.low_rank_layers)))
        
        # Determine which blocks should have conditioning
        first_n = self.training_config.timestep_conditioning_first_n_blocks
        last_n = self.training_config.timestep_conditioning_last_n_blocks
        
        target_blocks = set()
        if first_n > 0:
            target_blocks.update(range(min(first_n, total_blocks)))
        if last_n > 0:
            start_idx = max(0, total_blocks - last_n)
            target_blocks.update(range(start_idx, total_blocks))
        
        # Find layer indices that belong to target blocks
        conditioning_layers = set()
        for layer_idx, name in enumerate(self.low_rank_layer_names):
            if "transformer_blocks." in name:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "transformer_blocks":
                    try:
                        block_idx = int(parts[1])
                        if block_idx in target_blocks:
                            conditioning_layers.add(layer_idx)
                    except ValueError:
                        continue
            elif not target_blocks:
                # If no transformer blocks are targeted, include non-transformer layers
                conditioning_layers.add(layer_idx)
        
        print(f"Timestep conditioning enabled for {len(conditioning_layers)}/{len(self.low_rank_layers)} layers")
        print(f"Target transformer blocks: {sorted(target_blocks)} out of {total_blocks} total blocks")
        
        return conditioning_layers
    
    def set_timestep_lower_bound(self, timestep_lower_bound):
        self.timestep_lower_bound = timestep_lower_bound
    
    def forward(self, hidden_states, timestep=None, class_labels=None, **kwargs):
        # Store timesteps for use in LowRankLinear layers
        self.current_timesteps = timestep

        # Check if timestep lower bound is met
        if self.timestep_lower_bound is not None:
           if self.current_timesteps.min() < self.timestep_lower_bound:
               raise ValueError(f"Timestep lower bound {self.timestep_lower_bound} is not met, minimum timestep is {self.current_timesteps.min()}")

        # Inject timestep conditioning function into each LowRankLinear layer
        if self.training_config.timestep_conditioning and timestep is not None:
            for i, layer in enumerate(self.low_rank_layers):
                if i in self.conditioning_enabled_layers:
                    layer._wrapper_timesteps = timestep
                    layer._wrapper_T = self.training_config.num_training_steps
                    layer._wrapper_config = self.training_config
                    layer._wrapper_timestep_lower_bound = self.timestep_lower_bound

                else:
                    # Clear timestep conditioning for layers not in the selected set
                    layer._wrapper_timesteps = None
                    layer._wrapper_T = None
                    layer._wrapper_config = None
                    layer._wrapper_timestep_lower_bound = None
        
        # Call the base model
        return self.base_model(hidden_states, timestep, class_labels, **kwargs)


class LowRankLinear(nn.Module):
    """
    Low-rank parameterization W ≈ U @ V with optional timestep-conditioned rank gating.
    - Per-sample gating is done by masking the rank activations (x @ V.T) -> [B, r].
    - If all items share the same active rank (common at inference), we optionally slice to save FLOPs.
    """
    def __init__(self, in_features, out_features, rank, initialise=True):
        super().__init__()
        # Limit rank to the maximum possible rank for this layer
        self.rank = in_features if rank is None else min(rank, min(in_features, out_features))
        self.in_features = in_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(out_features, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.rank, in_features))
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Attributes for wrapper integration
        self._wrapper_timesteps = None
        self._wrapper_T = None
        self._wrapper_config = None
        self._wrapper_timestep_lower_bound = None

        if initialise:
            # Initialize using standard linear init strategy
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # SVD init on a Xavier weight; splits singular values as sqrt(S) across U and V
        W_full = torch.empty(self.out_features, self.in_features, device=self.U.device, dtype=self.U.dtype)
        nn.init.xavier_uniform_(W_full)
        U_, S_, Vh_ = torch.linalg.svd(W_full, full_matrices=False)
        k = self.rank
        root_S = S_[:k].sqrt()
        self.U.copy_(U_[:, :k] * root_S)                # [out, k]
        self.V.copy_(Vh_[:k, :] * root_S.view(-1, 1))   # [k, in]
        nn.init.constant_(self.bias, 0)

    def _active_ranks(self, t, T, r_min_ratio=0.5, schedule="decreasing", logistic_k=8.0, logistic_m=0.6, warmup_ratio=0.05):
        """
        Map per-sample timestep t -> active rank r(t).
        t: [B] int/float tensor; T: scalar max timestep.
        rank: int, maximum rank value
        schedule: 'decreasing' | 'increasing' | 'midpeak' | 'logistic_decreasing' | 'logistic_increasing'
        logistic_k: sharpness parameter for logistic schedules (>0)
        logistic_m: midpoint parameter for logistic schedules in [0,1]
        warmup_ratio: fraction of timesteps to use full rank before decreasing (for decreasing schedules only)
        """
        r = self.rank
        r_min = max(1, int(round(r_min_ratio * r)))
        t = t.to(self.U.device)
        T = torch.as_tensor(T, device=self.U.device, dtype=t.dtype)

        if schedule == "decreasing":
            # At t=0: r_full, at t=T: r_min
            # With warmup: full rank for first warmup_ratio * T timesteps
            warmup_t = warmup_ratio * T
            
            # Create mask for warmup period
            is_warmup = t < warmup_t
            
            # For warmup period: use full rank
            # For decreasing period: linear decrease over remaining time
            remaining_T = T - warmup_t
            adjusted_t = torch.clamp(t - warmup_t, min=0)  # Time since warmup ended
            
            r_t_decreasing = (r_min + (r - r_min) * (remaining_T - adjusted_t) / (remaining_T + 1e-8)).floor().clamp(min=r_min, max=r)
            r_t = torch.where(is_warmup, r, r_t_decreasing)

        elif schedule == "increasing":
            # At t=0: r_min, at t=T: r_full
            r_t = (r_min + (r - r_min) * (t / T)).floor().clamp(min=r_min, max=r)

        elif schedule == "midpeak":
            # simple tent peaking at T/2
            mid  = 0.5 * T
            left  = (t <= mid).to(t.dtype) * (t / mid)
            right = (t >  mid).to(t.dtype) * ((T - t) / (T - mid + 1e-8))
            frac = torch.maximum(left, right)
            r_t = (r_min + (r - r_min) * frac).floor().clamp(min=r_min, max=r)

        elif schedule == "logistic_decreasing":
            # S-curve: start at r, end at r_min
            # With warmup: full rank for first warmup_ratio * T timesteps
            k = float(logistic_k)   # sharpness (>0)
            m = float(logistic_m)   # midpoint in [0,1]
            warmup_t = warmup_ratio * T
            
            # Create mask for warmup period
            is_warmup = t < warmup_t
            
            # For logistic decrease over remaining time
            remaining_T = T - warmup_t
            adjusted_t = torch.clamp(t - warmup_t, min=0)  # Time since warmup ended
            frac = (adjusted_t / (remaining_T + 1e-8)).clamp(0, 1)  # Fraction of remaining time
            
            # Normalize sigmoid to actually go from 1 to 0
            raw_s = torch.sigmoid(-k * (frac - m))
            s_max = torch.sigmoid(torch.tensor(-k * (0 - m), device=self.U.device, dtype=frac.dtype))    # value at frac=0 (start of decrease)
            s_min = torch.sigmoid(torch.tensor(-k * (1 - m), device=self.U.device, dtype=frac.dtype))    # value at frac=1 (end of decrease)
            s = (raw_s - s_min) / (s_max - s_min + 1e-8)  # normalize to [0,1], then 1 early, 0 late
            
            r_t_decreasing = (r_min + (r - r_min) * s).floor().clamp(min=r_min, max=r)
            r_t = torch.where(is_warmup, r, r_t_decreasing)

        elif schedule == "logistic_increasing":
            # S-curve: start at r_min, end at r
            k = float(logistic_k)   # sharpness (>0)
            m = float(logistic_m)   # midpoint in [0,1]
            frac = (t / T).clamp(0, 1)
            
            # Normalize sigmoid to actually go from 0 to 1
            raw_s = torch.sigmoid(k * (frac - m))
            s_min = torch.sigmoid(torch.tensor(k * (0 - m), device=self.U.device, dtype=frac.dtype))     # value at frac=0 (t=0)
            s_max = torch.sigmoid(torch.tensor(k * (1 - m), device=self.U.device, dtype=frac.dtype))     # value at frac=1 (t=T)
            s = (raw_s - s_min) / (s_max - s_min)  # normalize to [0,1], then 0 early, 1 late
            
            r_t = (r_min + (r - r_min) * s).floor().clamp(min=r_min, max=r)

        else:
            raise ValueError("Unknown schedule")

        return r_t.to(torch.long)  # [B]
    # def _active_ranks(self, t, T, r_min_ratio=0.5, schedule="decreasing"):
    #     """
    #     Map per-sample timestep t -> active rank r(t).
    #     t: [B] int/float tensor; T: scalar max timestep.
    #     schedule: 'decreasing' | 'increasing' | 'midpeak' | 'logistic_decreasing' | 'logistic_increasing'
    #     """
    #     r = self.rank
    #     r_min = max(1, int(round(r_min_ratio * r)))
    #     t = t.to(self.U.device)
    #     T = torch.as_tensor(T, device=self.U.device, dtype=t.dtype)

    #     if schedule == "decreasing":
    #         # At t=0: r_full, at t=T: r_min
    #         r_t = (r_min + 1 + (r - r_min) * (T - t) / T).floor().clamp(min=r_min, max=r)

    #     elif schedule == "increasing":
    #         # At t=0: r_min, at t=T: r_full
    #         r_t = (r_min + 1 + (r - r_min) * (t / T)).floor().clamp(min=r_min, max=r)

    #     elif schedule == "midpeak":
    #         # simple tent peaking at T/2
    #         mid  = 0.5 * T
    #         left  = (t <= mid).to(t.dtype) * (t / mid)
    #         right = (t >  mid).to(t.dtype) * ((T - t) / (T - mid + 1e-8))
    #         frac = torch.maximum(left, right)
    #         r_t = (r_min + (r - r_min) * frac).floor().clamp(min=r_min, max=r)

    #     elif schedule == "logistic_decreasing":
    #         # S-curve: start near r, end near r_min
    #         k = float(getattr(self, "logistic_k", 8.0))   # sharpness (>0)
    #         m = float(getattr(self, "logistic_m", 0.6))   # midpoint in [0,1]
    #         frac = (t / T).clamp(0, 1)
    #         s = torch.sigmoid(-k * (frac - m))            # ~1 early, ~0 late
    #         r_t = (r_min + (r - r_min) * s).floor().clamp(min=r_min, max=r)

    #     elif schedule == "logistic_increasing":
    #         # S-curve: start near r_min, end near r
    #         k = float(getattr(self, "logistic_k", 8.0))   # sharpness (>0)
    #         m = float(getattr(self, "logistic_m", 0.6))   # midpoint in [0,1]
    #         frac = (t / T).clamp(0, 1)
    #         s = torch.sigmoid(k * (frac - m))             # ~0 early, ~1 late
    #         r_t = (r_min + (r - r_min) * s).floor().clamp(min=r_min, max=r)

    #     else:
    #         raise ValueError("Unknown schedule")

    #     return r_t.to(torch.long)  # [B]


    def forward(
        self,
        x,
        t: torch.Tensor = None,      # [B] timesteps (optional). If None -> no gating.
        T: int | float = None,       # max timestep (required if t is given)
        *,
        r_min_ratio: float = 0.5,
        schedule: str = "decreasing",
        slice_if_uniform: bool = True,   # if all r_t equal, slice U/V to save FLOPs
        return_mask: bool = False
    ):
        """
        x: [B, in_features]
        Returns: y [B, out_features] (and optionally the boolean mask [B, r])
        """
        
        # Check if timestep conditioning is enabled via wrapper
        if (self._wrapper_timesteps is not None and 
            self._wrapper_config is not None and 
            self._wrapper_config.timestep_conditioning):
            t = self._wrapper_timesteps
            T = self._wrapper_T
            r_min_ratio = self._wrapper_config.rank_min_ratio
            schedule = self._wrapper_config.rank_schedule
        if t is None:
            # Baseline path: full rank, standard low-rank multiply
            # (x @ V.T) -> [B, r], then -> [B, out]
            Ax = nn.functional.linear(x, self.V)          # [B, r]
            y = nn.functional.linear(Ax, self.U, self.bias)
            return (y, None) if return_mask else y

        # Handle batch size mismatch between patches and timesteps
        B_x = x.shape[0]  # Actual batch size (could be patches)
        B_t = t.shape[0]  # Original timestep batch size
        
        # Smart timestep handling for different processing levels
        if B_x == B_t:
            # Same batch size - direct use (image-level processing)
            t_expanded = t
        elif B_x % B_t == 0:
            # Batch size is a multiple - likely patch-level processing
            patches_per_image = B_x // B_t
            t_expanded = t.repeat_interleave(patches_per_image)  # [B_x]
        else:
            # Unexpected batch relationship: expand by nearest-repeat and trim/pad
            repeat = math.ceil(B_x / B_t)
            t_expanded = t.repeat_interleave(repeat)
            if t_expanded.shape[0] > B_x:
                t_expanded = t_expanded[:B_x]
            elif t_expanded.shape[0] < B_x:
                pad = B_x - t_expanded.shape[0]
                t_expanded = torch.cat([t_expanded, t_expanded[-1:].expand(pad)], dim=0)

        # Per-sample active ranks
        r_t = self._active_ranks(t_expanded, T, r_min_ratio=r_min_ratio, schedule=schedule)  # [B_x]
        r = self.rank
        # print(f"r_t: {r_t}")
        idx = torch.arange(r, device=x.device)
        mask = (idx.unsqueeze(0) < r_t.unsqueeze(1))  # [B_x, r] bool

        if slice_if_uniform and bool(torch.all(r_t == r_t[0])):  # whole batch shares same rank -> slice
            r_active = int(r_t[0].item())
            Ax = nn.functional.linear(x, self.V[:r_active, :])              # [B_x, r_active] or [B_x, patches, r_active]
            y = nn.functional.linear(Ax, self.U[:, :r_active], self.bias)   # [B_x, out] or [B_x, patches, out]
            return (y, mask) if return_mask else y
        
       
        if self._wrapper_timestep_lower_bound is not None and self._wrapper_timestep_lower_bound <= t_expanded.min():
            r_bound = int(self._active_ranks(torch.tensor([self._wrapper_timestep_lower_bound], device=x.device), T, r_min_ratio=r_min_ratio, schedule=schedule)[0].item())
            Ax = nn.functional.linear(x, self.V[:r_bound, :])              # [B_x, r_active] or [B_x, patches, r_active]
            mask = mask[:, :r_bound]
            
            if Ax.dim() == 3:
                # 3D: [batch, patches, rank] - expand mask to [batch, 1, rank] for broadcasting
                mask = mask.unsqueeze(1)  # [B_x, 1, r] - broadcasts with [B_x, patches, r]
            # else: 2D case [batch, rank] - mask already correct shape [B_x, r]
            Ax = Ax * mask

            y = nn.functional.linear(Ax, self.U[:, :r_bound], self.bias)   # [B_x, out] or [B_x, patches, out]
            return (y, mask) if return_mask else y

        # General case: mixed timesteps in batch -> mask activations (one vectorized forward)
        Ax = nn.functional.linear(x, self.V)                # [B_x, r] or [B_x, patches, r]
        
        # Handle both 2D and 3D tensor formats
        if Ax.dim() == 3:
            # 3D: [batch, patches, rank] - expand mask to [batch, 1, rank] for broadcasting
            mask = mask.unsqueeze(1)  # [B_x, 1, r] - broadcasts with [B_x, patches, r]
        # else: 2D case [batch, rank] - mask already correct shape [B_x, r]
        
        Ax = Ax * mask

        y  = nn.functional.linear(Ax, self.U, self.bias)    # [B_x, out]
        return (y, mask) if return_mask else y
    
    def svd_decomposition(self, base_linear, rank=None, threshold=None):
        if(rank is None and threshold is None):
            raise ValueError("Either rank or threshold must be specified")
        
        weight = base_linear.weight.data
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        if rank is None: 
            self.threshold = threshold
            S_squared = S**2
            total_energy = S_squared.sum()
            cumulative_energy = torch.cumsum(S_squared / total_energy, dim=0)
            self.rank = int((cumulative_energy < self.threshold).sum()) + 1
        else: 
            self.rank = min(rank, min(self.out_features, self.in_features))

        root_S = S[: self.rank].sqrt()   

        self.U = nn.Parameter(U[:, : self.rank] * root_S)
        self.V = nn.Parameter(Vh[: self.rank, :] * root_S.view(-1, 1))

        self.bias = base_linear.bias if base_linear.bias is not None else 0

    def orthogonality_loss(self, rho: float = 1.0):
        """
        Double-Soft-Orthogonality (DSO) penalty:
        R_d(U) = ρ / Φ² [‖UᵀU−I‖²_F + ‖UUᵀ−I‖²_F]
        applied to both Uk and Vhk.
        """
        def _dso(mat: torch.Tensor):
            m, n = mat.shape
            eye_m = torch.eye(m, device=mat.device, dtype=mat.dtype)
            eye_n = torch.eye(n, device=mat.device, dtype=mat.dtype)
            return (torch.norm(mat.T @ mat - eye_n, p='fro') ** 2 +
                    torch.norm(mat @ mat.T - eye_m, p='fro') ** 2) / (mat.shape[0] ** 2)

        return rho * (_dso(self.U) + _dso(self.V))

    def frobenius_loss(self) -> torch.Tensor:
        """
        0.5 * ‖UVᵀ‖_F²   —>  add   λ * frobenius_loss()
        to your main loss for Frobenius-decay.
        """
        return 0.5 * (self.U @ self.V).pow(2).sum()


def apply_low_rank_compression(module, rank=None, threshold=None):
    """
    Apply low-rank compression to a module's linear layers.
    
    Args:
        module: The PyTorch module to compress
        rank: Fixed rank to use for all linear layers (optional)
        threshold: Energy threshold for adaptive rank determination (optional)
    
    Note: Either rank or threshold must be provided, but not both.
    """
    if rank is not None and threshold is not None:
        raise ValueError("Provide either rank or threshold, not both")
    if rank is None and threshold is None:
        raise ValueError("Either rank or threshold must be specified")
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            low_rank_layer = LowRankLinear(child.in_features, child.out_features, rank = None, initialise = False)

            if rank is not None:
                low_rank_layer.svd_decomposition(child, rank=rank)
                setattr(module, name, low_rank_layer)
            else:
                low_rank_layer.svd_decomposition(child, threshold=threshold)
                setattr(module, name, low_rank_layer)
        else:
            apply_low_rank_compression(child, rank=rank, threshold=threshold)

    return module

# def low_rank_layer_replacement(module, rank):

#     # Replace all Linear layers with LowRankLinear
#     for name, child in module.named_children():
#         if isinstance(child, nn.Linear):
#             new_layer = LowRankLinear(
#                 child.in_features, 
#                 child.out_features, 
#                 rank=rank,
#                 initialise = True
#             )

#             setattr(module, name, new_layer)

#         else:
#             low_rank_layer_replacement(child, rank)
    
#     return module

def _should_apply_timestep_conditioning(full_name, config, total_blocks=None):
    """
    Determine if timestep conditioning should be applied to a layer based on its name and config.
    
    Args:
        full_name: Full path name of the layer (e.g., "transformer_blocks.2.attn.to_q")
        config: Training configuration with timestep conditioning settings
        total_blocks: Total number of transformer blocks (if None, will default to 6)
    
    Returns:
        bool: True if timestep conditioning should be applied to this layer
    """
    if config is None or not config.timestep_conditioning:
        return False
    
    # If both first_n and last_n are 0, apply to all layers (original behavior)
    first_n = config.timestep_conditioning_first_n_blocks
    last_n = config.timestep_conditioning_last_n_blocks
    
    if first_n == 0 and last_n == 0:
        return True
    
    # Check if this is a transformer block layer
    if "transformer_blocks." not in full_name:
        return False
    
    # Extract block index from name like "transformer_blocks.5.attn.to_q"
    parts = full_name.split(".")
    if len(parts) < 2 or parts[0] != "transformer_blocks":
        return False
    
    try:
        block_idx = int(parts[1])
    except ValueError:
        return False
    
    # Use provided total_blocks or default to 6
    if total_blocks is None:
        total_blocks = 6  # Default based on DiT config
    
    # Check if block is in first_n blocks
    if first_n > 0 and block_idx < first_n:
        return True
    
    # Check if block is in last_n blocks
    if last_n > 0 and block_idx >= (total_blocks - last_n):
        return True
    
    return False


def _get_total_transformer_blocks(module):
    """
    Dynamically determine the total number of transformer blocks by scanning the module.
    
    Args:
        module: The model module to scan
        
    Returns:
        int: Total number of transformer blocks found
    """
    block_indices = set()
    
    def scan_module(mod, prefix=""):
        for name, child in mod.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if "transformer_blocks." in full_name:
                parts = full_name.split(".")
                if len(parts) >= 2 and parts[0] == "transformer_blocks":
                    try:
                        block_idx = int(parts[1])
                        block_indices.add(block_idx)
                    except ValueError:
                        pass
            scan_module(child, full_name)
    
    scan_module(module)
    return len(block_indices) if block_indices else 6  # Default to 6 if no blocks found


def low_rank_layer_replacement(module, percentage, prefix="", config = None, enable_timestep_conditioning=False, _total_blocks=None):
    """
    Replace Linear layers with LowRankLinear layers, with optional selective timestep conditioning.
    
    Args:
        module: The model module to process
        percentage: Percentage for rank calculation
        prefix: Internal parameter for recursive calls
        config: Training configuration containing timestep conditioning settings
        enable_timestep_conditioning: Legacy parameter (now uses config.timestep_conditioning)
        _total_blocks: Internal parameter for passing total blocks through recursion
        
    Timestep Conditioning Configuration:
        - timestep_conditioning: bool = Enable timestep conditioning
        - timestep_conditioning_first_n_blocks: int = Apply to first n blocks (0 = disabled)
        - timestep_conditioning_last_n_blocks: int = Apply to last n blocks (0 = disabled)
        - timestep_conditioning_total_blocks: int = Override total blocks (0 = auto-detect)
        - timestep_conditioning_match_type: str = "activated" or "total"
        
    Examples:
        # Apply to first 2 blocks only:
        config.timestep_conditioning = True
        config.timestep_conditioning_first_n_blocks = 2
        config.timestep_conditioning_last_n_blocks = 0
        
        # Apply to last 3 blocks only:
        config.timestep_conditioning = True  
        config.timestep_conditioning_first_n_blocks = 0
        config.timestep_conditioning_last_n_blocks = 3
        
        # Apply to first 2 and last 2 blocks:
        config.timestep_conditioning = True
        config.timestep_conditioning_first_n_blocks = 2
        config.timestep_conditioning_last_n_blocks = 2
        
        # Apply to all blocks (original behavior):
        config.timestep_conditioning = True
        config.timestep_conditioning_first_n_blocks = 0  
        config.timestep_conditioning_last_n_blocks = 0
    """
    # Replace all Linear layers with LowRankLinear
    
    # Dynamically determine total number of transformer blocks (only do this at the top level)
    if prefix == "" and _total_blocks is None:
        # Check if total blocks is overridden in config
        if config is not None and hasattr(config, 'timestep_conditioning_total_blocks') and config.timestep_conditioning_total_blocks > 0:
            total_blocks = config.timestep_conditioning_total_blocks
            print(f"Using configured total blocks: {total_blocks}")
        else:
            total_blocks = _get_total_transformer_blocks(module)
            print(f"Auto-detected {total_blocks} transformer blocks")
        
        if config is not None and config.timestep_conditioning:
            if config.timestep_conditioning_first_n_blocks > 0:
                print(f"Applying timestep conditioning to first {config.timestep_conditioning_first_n_blocks} blocks")
            if config.timestep_conditioning_last_n_blocks > 0:
                print(f"Applying timestep conditioning to last {config.timestep_conditioning_last_n_blocks} blocks")
    else:
        total_blocks = _total_blocks  # Use provided value from recursive call
    
    for name, child in module.named_children():
        # Build the full path name
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear) and "transformer_blocks.0.ff.net" not in full_name and "proj_out" not in full_name:
            print(f"Replacing {full_name} with LowRankLinear")
            # Calculate original rank of the weight matrix
            original_rank = min(child.in_features, child.out_features)
            # Calculate new rank as percentage of original rank
            new_rank = int((child.in_features * child.out_features * percentage) / (child.in_features + child.out_features))
            
            # Apply timestep conditioning rank adjustment only for specified blocks
            if (config is not None and 
                _should_apply_timestep_conditioning(full_name, config, total_blocks) and 
                config.timestep_conditioning_match_type == "activated"):
                actual_rank = calculate_rank_for_expected_rank(new_rank, 
                                                            config.num_training_steps, 
                                                            config.rank_min_ratio, 
                                                            config.rank_schedule)['found_rank']
                expected_rank = calculate_rank_for_expected_rank(new_rank, 
                                                            config.num_training_steps, 
                                                            config.rank_min_ratio, 
                                                            config.rank_schedule)['found_expected_rank']
                print(f"Layer {full_name}: timestep conditioning applied - actual rank: {actual_rank} was {new_rank} expected rank: {expected_rank}")
                new_rank = actual_rank
            elif config is not None and config.timestep_conditioning:
                print(f"Layer {full_name}: timestep conditioning skipped (not in specified blocks)")

            new_rank = max(1, new_rank)
            print(f"original rank: {original_rank}, new rank: {new_rank}")
            new_layer = LowRankLinear(
                child.in_features, 
                child.out_features, 
                rank=new_rank,
                initialise = True
            )

            setattr(module, name, new_layer)

        else:
            low_rank_layer_replacement(child, percentage, prefix=full_name, config=config, enable_timestep_conditioning=enable_timestep_conditioning, _total_blocks=total_blocks)
    
    return module


def label_low_rank_gradient_layers(model):

    # label layers for galore optimizer

    galore_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # TODO: Possible layer selection here

        # print('enable GaLore for weights in module: ', module_name)
        galore_params.append(module.weight)

    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

    # count_parameter_groups(galore_params, regular_params)
    return galore_params, regular_params

def count_parameter_groups(galore_params, regular_params):
    """
    Count and print the number of parameters in each parameter group.
    
    Args:
        galore_params: List of parameters to be optimized with GaLore
        regular_params: List of parameters to be optimized with standard optimizer
    """
    galore_count = sum(p.numel() for p in galore_params)
    regular_count = sum(p.numel() for p in regular_params)
    total_count = galore_count + regular_count
    
    print(f"GaLore parameters: {galore_count:,} ({galore_count/total_count:.2%} of total)")
    print(f"Regular parameters: {regular_count:,} ({regular_count/total_count:.2%} of total)")
    print(f"Total parameters: {total_count:,}")
    
    # Calculate estimated memory usage
    bytes_per_param = 12  # AdamW stores 3 copies (param, momentum, variance) at 4 bytes each
    galore_bytes = galore_count * (4 + 8 * (2/galore_count))  # GaLore uses less memory per parameter
    regular_bytes = regular_count * bytes_per_param
    
    print(f"\nEstimated optimizer memory usage:")
    print(f"GaLore parameters: {galore_bytes/1024**2:.2f} MB")
    print(f"Regular parameters: {regular_bytes/1024**2:.2f} MB")
    print(f"Total: {(galore_bytes + regular_bytes)/1024**2:.2f} MB")
    
    return galore_count, regular_count, total_count

def nuclear_norm(model) -> torch.Tensor:
    """
    Sum of the nuclear norms (σ₁ + σ₂ + …) of every linear-layer weight
    in the model, including LowRankLinear layers.

    Returns:
        A scalar tensor on the same device as the model parameters.
    """
    device = next(model.parameters()).device
    total = torch.tensor(0.0, device=device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            total = total + torch.linalg.matrix_norm(module.weight, ord='nuc')
        elif isinstance(module, LowRankLinear):
            total = total + torch.linalg.matrix_norm(module.U @ module.V, ord='nuc')

    return total

def frobenius_norm(model) -> torch.Tensor:
    """
    Sum of the frobenius norms (||W||_F) of every linear-layer weight
    in the model, including LowRankLinear layers.

    Returns:
        A scalar tensor on the same device as the model parameters.
    """
    device = next(model.parameters()).device
    total = torch.tensor(0.0, device=device)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            total = total + torch.linalg.matrix_norm(module.weight, ord='fro')
        elif isinstance(module, LowRankLinear):
            total = total + torch.linalg.matrix_norm(module.U @ module.V, ord='fro')

    return total


def calculate_rank_for_expected_rank(
    target_expected_rank: float,
    T: int | float,
    r_min_ratio: float = 0.5,
    schedule: str = "decreasing",
    timestep_distribution: str = "uniform",
    num_samples: int = 1000,
    max_rank: int = 1000,
    tolerance: float = 0.1,
    **schedule_kwargs
) -> dict:
    """
    Reverse function: Given a target expected rank, find the actual rank needed.
    
    Args:
        target_expected_rank: Desired expected active rank
        T: Maximum timestep 
        r_min_ratio: Minimum rank ratio
        schedule: Rank schedule type
        timestep_distribution: Distribution of timesteps
        num_samples: Number of timestep samples for numerical integration
        max_rank: Maximum rank to search up to
        tolerance: Tolerance for the binary search
        **schedule_kwargs: Additional parameters for logistic schedules
    
    Returns:
        Dictionary with the found rank and statistics
    """
    
    def get_expected_rank_for_rank(rank):
        """Helper function to get expected rank for a given max rank."""
        try:
            stats = calculate_expected_rank_single_layer(
                rank=rank, T=T, r_min_ratio=r_min_ratio, schedule=schedule,
                timestep_distribution=timestep_distribution, num_samples=num_samples,
                **schedule_kwargs
            )
            return stats['expected_rank']
        except:
            return float('inf')  # Return a large value if calculation fails
    
    # Check if target is feasible
    min_possible_rank = max(1, int(round(r_min_ratio * 1)))  # Minimum possible rank
    max_expected_at_max = get_expected_rank_for_rank(max_rank)
    min_expected_at_min = get_expected_rank_for_rank(target_expected_rank + 1)  # Start search slightly above target
    
    if target_expected_rank < min_possible_rank:
        return {
            'error': f'Target expected rank {target_expected_rank:.2f} is too low. Minimum possible: {min_possible_rank}',
            'target_expected_rank': target_expected_rank,
            'min_possible': min_possible_rank
        }
    
    if target_expected_rank > max_expected_at_max:
        return {
            'error': f'Target expected rank {target_expected_rank:.2f} is too high. Maximum achievable with rank {max_rank}: {max_expected_at_max:.2f}',
            'target_expected_rank': target_expected_rank,
            'max_achievable': max_expected_at_max,
            'max_rank_searched': max_rank
        }
    
    # Binary search for the correct rank
    low = 1
    high = max_rank
    best_rank = None
    best_expected = None
    
    while high - low > 1:
        mid = (low + high) // 2
        expected = get_expected_rank_for_rank(mid)
        
        if abs(expected - target_expected_rank) <= tolerance:
            best_rank = mid
            best_expected = expected
            break
        elif expected < target_expected_rank:
            low = mid
        else:
            high = mid
    
    # If we didn't find an exact match, check the final candidates
    if best_rank is None:
        for candidate in [low, high]:
            expected = get_expected_rank_for_rank(candidate)
            if best_rank is None or abs(expected - target_expected_rank) < abs(best_expected - target_expected_rank):
                best_rank = candidate
                best_expected = expected
    
    # Get full statistics for the best rank found
    final_stats = calculate_expected_rank_single_layer(
        rank=best_rank, T=T, r_min_ratio=r_min_ratio, schedule=schedule,
        timestep_distribution=timestep_distribution, num_samples=num_samples,
        **schedule_kwargs
    )
    
    # Add reverse calculation specific info
    final_stats.update({
        'target_expected_rank': target_expected_rank,
        'found_rank': best_rank,
        'found_expected_rank': best_expected,
        'error_from_target': abs(best_expected - target_expected_rank),
        'relative_error_percent': abs(best_expected - target_expected_rank) / target_expected_rank * 100
    })
    
    return final_stats

def calculate_expected_rank_single_layer(
    rank: int,
    T: int | float,
    r_min_ratio: float = 0.5,
    schedule: str = "decreasing",
    timestep_distribution: str = "uniform",
    num_samples: int = 1000,
    **schedule_kwargs
) -> dict:
    """
    Calculate expected active rank for a single layer across a distribution of timesteps.
    
    Args:
        rank: Maximum rank of the layer
        T: Maximum timestep 
        r_min_ratio: Minimum rank ratio (r_min = max(1, int(r_min_ratio * rank)))
        schedule: Rank schedule type ('decreasing', 'increasing', 'midpeak', 'logistic_decreasing', 'logistic_increasing')
        timestep_distribution: Distribution of timesteps ('uniform', 'early_heavy', 'late_heavy')
        num_samples: Number of timestep samples for numerical integration
        **schedule_kwargs: Additional parameters for logistic schedules (logistic_k, logistic_m)
    
    Returns:
        Dictionary with expected rank, min rank, max rank, and other statistics
    """
    r_min = max(1, int(round(r_min_ratio * rank)))
    
    # Generate timestep samples based on distribution
    if timestep_distribution == "uniform":
        t_samples = torch.linspace(0, T, num_samples)
    elif timestep_distribution == "early_heavy":
        # More samples in early timesteps (quadratic bias towards 0)
        uniform_samples = torch.linspace(0, 1, num_samples)
        t_samples = T * (uniform_samples ** 2)
    elif timestep_distribution == "late_heavy":
        # More samples in late timesteps (quadratic bias towards T)
        uniform_samples = torch.linspace(0, 1, num_samples)
        t_samples = T * (1 - (1 - uniform_samples) ** 2)
    else:
        raise ValueError(f"Unknown timestep_distribution: {timestep_distribution}")
    
    # Create a dummy layer to use its _active_ranks method
    dummy_layer = LowRankLinear(rank, rank, rank, initialise=False)
    
    # Set logistic parameters if provided
    for key, value in schedule_kwargs.items():
        if key in ['logistic_k', 'logistic_m']:
            setattr(dummy_layer, key, value)
    
    # Calculate active ranks for all timestep samples
    with torch.no_grad():
        active_ranks = dummy_layer._active_ranks(
            t_samples, T, r_min_ratio=r_min_ratio, schedule=schedule
        ).float()
    
    # Calculate statistics
    expected_rank = active_ranks.mean().item()
    min_active_rank = active_ranks.min().item()
    max_active_rank = active_ranks.max().item()
    std_rank = active_ranks.std().item()
    
    # Calculate rank utilization (as percentage of max rank)
    rank_utilization = expected_rank / rank * 100
    
    return {
        'expected_rank': expected_rank,
        'min_rank': min_active_rank,
        'max_rank': max_active_rank,
        'std_rank': std_rank,
        'max_possible_rank': rank,
        'min_possible_rank': r_min,
        'rank_utilization_percent': rank_utilization,
        'schedule': schedule,
        'r_min_ratio': r_min_ratio,
        'timestep_distribution': timestep_distribution
    }
