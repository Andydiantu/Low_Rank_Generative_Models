import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, base_linear, rank=None, threshold=None):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        if rank is None:
            self.adaptive_rank = True
            self.threshold = threshold

        else:
            self.adaptive_rank = False
            self.rank = rank

        # Perform SVD
        weight = base_linear.weight.data
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        if self.adaptive_rank:
            S_squared = S**2
            total_energy = S_squared.sum()
            cumulative_energy = torch.cumsum(S_squared / total_energy, dim=0)
            self.rank = int((cumulative_energy < self.threshold).sum()) + 1
        
        U_k = U[:, : self.rank]
        S_k = S[: self.rank]
        Vh_k = Vh[: self.rank, :]

        # Split into low-rank matrices
        self.A = nn.Parameter(U_k @ torch.diag(S_k))
        self.B = nn.Parameter(Vh_k)

        # Optional: Freeze bias if present
        self.bias = base_linear.bias if base_linear.bias is not None else None

    def forward(self, x):
        return x @ self.B.T @ self.A.T + (self.bias if self.bias is not None else 0)


def apply_structural_low_rank(module, rank=64):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LowRankLinear(child, rank=rank))
        else:
            apply_structural_low_rank(child, rank)


def apply_structural_low_rank_adaptive(module, threshold=0.99):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LowRankLinear(child, threshold=threshold))
        else:
            apply_structural_low_rank_adaptive(child, threshold=threshold)
