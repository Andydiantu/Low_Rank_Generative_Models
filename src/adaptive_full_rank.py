import math
from typing import Optional

import torch
import torch.nn as nn


def compute_active_heads(
    t: torch.Tensor,
    T: int | float,
    num_heads: int,
    min_ratio: float = 0.5,
    schedule: str = "decreasing",
    logistic_k: float = 8.0,
    logistic_m: float = 0.6,
) -> torch.Tensor:
    """
    Map per-sample timestep t -> active heads H(t).
    t: [B] tensor; T: scalar max timestep.
    Returns: [B] long tensor with 1..num_heads active heads per sample.
    """
    h_full = int(num_heads)
    h_min = max(1, int(round(min_ratio * h_full)))

    t = t.to(dtype=torch.float32)
    T_t = torch.as_tensor(T, device=t.device, dtype=t.dtype)
    frac = (t / T_t).clamp(0, 1)

    if schedule == "decreasing":
        # t=0 -> h_full, t=T -> h_min
        s = 1.0 - frac
        h_t = (h_min + (h_full - h_min) * s).floor()
    elif schedule == "increasing":
        # t=0 -> h_min, t=T -> h_full
        s = frac
        h_t = (h_min + (h_full - h_min) * s).floor()
    elif schedule == "midpeak":
        mid = 0.5
        left = (frac <= mid).to(t.dtype) * (frac / mid)
        right = (frac > mid).to(t.dtype) * ((1.0 - frac) / (1.0 - mid + 1e-8))
        s = torch.maximum(left, right)
        h_t = (h_min + (h_full - h_min) * s).floor()
    elif schedule == "logistic_decreasing":
        s = torch.sigmoid(-logistic_k * (frac - logistic_m))
        h_t = (h_min + (h_full - h_min) * s).floor()
    elif schedule == "logistic_increasing":
        s = torch.sigmoid(logistic_k * (frac - logistic_m))
        h_t = (h_min + (h_full - h_min) * s).floor()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return h_t.clamp(min=h_min, max=h_full).to(torch.long)


# Improved HeadGater (override) with robust hooks and debug info
import math
import torch
import torch.nn as nn
from typing import Optional


# New: compute active features for FFN hidden dimension masking

def compute_active_features(
    t: torch.Tensor,
    T: int | float,
    total_dim: int,
    min_ratio: float = 0.5,
    schedule: str = "decreasing",
    logistic_k: float = 8.0,
    logistic_m: float = 0.6,
) -> torch.Tensor:
    """
    Map per-sample timestep t -> active hidden features C(t).
    At t=0 -> total_dim, at t=T -> floor(min_ratio * total_dim) for 'decreasing' schedule.
    Returns: [B] long tensor with 1..total_dim active features per sample.
    """
    c_full = int(total_dim)
    c_min = max(1, int(round(min_ratio * c_full)))

    t = t.to(dtype=torch.float32)
    T_t = torch.as_tensor(T, device=t.device, dtype=t.dtype)
    frac = (t / T_t).clamp(0, 1)

    if schedule == "decreasing":
        s = 1.0 - frac
        c_t = (c_min + (c_full - c_min) * s).floor()
    elif schedule == "increasing":
        s = frac
        c_t = (c_min + (c_full - c_min) * s).floor()
    elif schedule == "midpeak":
        mid = 0.5
        left = (frac <= mid).to(t.dtype) * (frac / mid)
        right = (frac > mid).to(t.dtype) * ((1.0 - frac) / (1.0 - mid + 1e-8))
        s = torch.maximum(left, right)
        c_t = (c_min + (c_full - c_min) * s).floor()
    elif schedule == "logistic_decreasing":
        s = torch.sigmoid(-logistic_k * (frac - logistic_m))
        c_t = (c_min + (c_full - c_min) * s).floor()
    elif schedule == "logistic_increasing":
        s = torch.sigmoid(logistic_k * (frac - logistic_m))
        c_t = (c_min + (c_full - c_min) * s).floor()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return c_t.clamp(min=c_min, max=c_full).to(torch.long)


class HeadGater:
    def __init__(
        self,
        model: nn.Module,
        max_timestep: int | float = 1000,
        min_ratio: float = 0.5,
        schedule: str = "decreasing",
        logistic_k: float = 8.0,
        logistic_m: float = 0.6,
        debug: bool = True,
        ffn_min_ratio: Optional[float] = 0.5,
        ffn_schedule: Optional[str] = None,
    ) -> None:
        self.model = model
        self.max_timestep = max_timestep
        self.min_ratio = min_ratio
        self.schedule = schedule
        self.logistic_k = logistic_k
        self.logistic_m = logistic_m
        self.debug = debug
        # FFN-specific behavior (defaults to half at max timestep)
        self.ffn_min_ratio = ffn_min_ratio if ffn_min_ratio is not None else min_ratio
        self.ffn_schedule = ffn_schedule if ffn_schedule is not None else schedule

        self.current_timesteps: Optional[torch.Tensor] = None
        self.last_active_heads: Optional[torch.Tensor] = None  # [B]
        self.last_num_heads: Optional[int] = None
        self._handles: list = []

        self._handles.append(
            self.model.register_forward_pre_hook(self._capture_timesteps, with_kwargs=True)
        )
        self._register_attention_output_hooks()
        # New: also register FFN hidden-dimension hooks
        self._register_ffn_hidden_hooks()

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _capture_timesteps(self, module, args, kwargs):
        t = kwargs.get("timestep", None)
        if t is None:
            t = kwargs.get("timesteps", None)
        if t is None:
            self.current_timesteps = None
            return
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=self._device())
        t = t.to(dtype=torch.float32)
        if t.ndim == 0:
            t = t[None]
        self.current_timesteps = t

    def _register_attention_output_hooks(self) -> None:
        self.global_num_heads = getattr(getattr(self.model, "config", None), "num_attention_heads", None)
        self.global_head_dim = getattr(getattr(self.model, "config", None), "attention_head_dim", None)

        def to_out_pre_hook(linear: nn.Module, inputs):
            if self.current_timesteps is None:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return

            # x is [B, S, C] or [B, C]
            B = x.shape[0]
            C = x.shape[-1]

            num_heads = self.global_num_heads
            head_dim = self.global_head_dim
            if num_heads is None or head_dim is None or (num_heads * head_dim) != C:
                if head_dim is not None and C % int(head_dim) == 0:
                    num_heads = C // int(head_dim)
                elif num_heads is not None and num_heads > 0:
                    head_dim = C // int(num_heads)
                else:
                    return

            t = self.current_timesteps
            B_t = t.shape[0]
            if B == B_t:
                t_exp = t
            elif B % B_t == 0:
                t_exp = t.repeat_interleave(B // B_t)
            else:
                repeat = math.ceil(B / B_t)
                t_exp = t.repeat_interleave(repeat)[:B]

            H_t = compute_active_heads(
                t=t_exp,
                T=self.max_timestep,
                num_heads=int(num_heads),
                min_ratio=self.min_ratio,
                schedule=self.schedule,
                logistic_k=self.logistic_k,
                logistic_m=self.logistic_m,
            )  # [B]

            head_idx = torch.arange(int(num_heads), device=x.device)
            head_mask = (head_idx.unsqueeze(0) < H_t.unsqueeze(1))  # [B, heads]
            feat_mask = head_mask.repeat_interleave(int(head_dim), dim=1)  # [B, C]
            if x.ndim == 3:
                feat_mask = feat_mask.unsqueeze(1)  # [B, 1, C]

            # Debug bookkeeping
            self.last_active_heads = H_t.detach().cpu()
            self.last_num_heads = int(num_heads)

            return (x * feat_mask,)

        for module in self.model.modules():
            to_out = getattr(module, "to_out", None)
            if isinstance(to_out, (nn.Sequential, nn.ModuleList)) and len(to_out) >= 1:
                first = to_out[0]
                if isinstance(first, nn.Linear):
                    h = first.register_forward_pre_hook(to_out_pre_hook, with_kwargs=False)
                    self._handles.append(h)
            elif isinstance(to_out, nn.Linear):  # handle direct Linear
                h = to_out.register_forward_pre_hook(to_out_pre_hook, with_kwargs=False)
                self._handles.append(h)

    # New: Register hooks to adaptively mask FFN hidden dimensions (mask input to fc2)
    def _register_ffn_hidden_hooks(self) -> None:
        def ffn_fc2_pre_hook(linear: nn.Module, inputs):
            if self.current_timesteps is None:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            # x is [B, S, C_hidden] or [B, C_hidden]
            B = x.shape[0]
            C_hidden = x.shape[-1]

            t = self.current_timesteps
            B_t = t.shape[0]
            if B == B_t:
                t_exp = t
            elif B % B_t == 0:
                t_exp = t.repeat_interleave(B // B_t)
            else:
                repeat = math.ceil(B / B_t)
                t_exp = t.repeat_interleave(repeat)[:B]

            C_t = compute_active_features(
                t=t_exp,
                T=self.max_timestep,
                total_dim=int(C_hidden),
                min_ratio=self.ffn_min_ratio,
                schedule=self.ffn_schedule,
                logistic_k=self.logistic_k,
                logistic_m=self.logistic_m,
            )  # [B]

            ch_idx = torch.arange(int(C_hidden), device=x.device)
            ch_mask = (ch_idx.unsqueeze(0) < C_t.unsqueeze(1))  # [B, C_hidden]
            if x.ndim == 3:
                ch_mask = ch_mask.unsqueeze(1)  # [B, 1, C_hidden]
            return (x * ch_mask,)

        # Attach only to FFN fc2 layers where in_features > out_features and module path suggests FFN/MLP
        suspect_tokens = ("ff", "ffn", "mlp", "feedforward", "feed_forward")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                lname = name.lower()
                if any(tok in lname for tok in suspect_tokens):
                    try:
                        in_f = int(module.in_features)
                        out_f = int(module.out_features)
                    except Exception:
                        continue
                    # Likely the second FFN projection (hidden -> model_dim)
                    if in_f > out_f:
                        h = module.register_forward_pre_hook(ffn_fc2_pre_hook, with_kwargs=False)
                        self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
