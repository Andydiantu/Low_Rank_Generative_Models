import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, initialise=True):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        self.Uk = nn.Parameter(torch.Tensor(out_features, rank))
        self.Sk = nn.Parameter(torch.Tensor(rank))  # Diagonal singular values
        self.Vhk = nn.Parameter(torch.Tensor(rank, in_features))
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        if initialise:
            # Initialize using standard linear init strategy
            self.reset_parameters()

    def reset_parameters(self):
        # Initialize Uk and Vhk with orthogonal matrices
        nn.init.orthogonal_(self.Uk)
        nn.init.orthogonal_(self.Vhk)
        
        # Initialize Sk with 1/sqrt(rank) scaling for stable gradients
        nn.init.normal_(self.Sk, mean=1.0, std=1.0 / self.rank**0.5)
        
        # Initialize bias
        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        low_rank_weight = self.Uk @ torch.diag(self.Sk) @ self.Vhk
        return nn.functional.linear(x, low_rank_weight, self.bias)
    
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
            self.rank = rank

        self.Uk = nn.Parameter(U[:, : self.rank])
        self.Sk = nn.Parameter(S[: self.rank])
        self.Vhk = nn.Parameter(Vh[: self.rank, :])

        self.bias = base_linear.bias if base_linear.bias is not None else 0
            


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
            low_rank_layer = LowRankLinear(child.in_features, child.out_features, rank = 1, initialise = False)

            if rank is not None:
                low_rank_layer.svd_decomposition(child, rank=rank)
                setattr(module, name, low_rank_layer)
            else:
                low_rank_layer.svd_decomposition(child, threshold=threshold)
                setattr(module, name, low_rank_layer)
        else:
            apply_low_rank_compression(child, rank=rank, threshold=threshold)

def low_rank_layer_replacement(module, rank):

    # Replace all Linear layers with LowRankLinear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            new_layer = LowRankLinear(
                module.in_features, 
                module.out_features, 
                rank=rank,
                initialise = True
            )

            #TODO: check do this or setattr(module, name, low_rank_layer)
            # Set parent module's attribute to the new layer
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_layer)
    
    return model
