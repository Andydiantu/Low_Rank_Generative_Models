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

        return rho * (_dso(self.Uk) + _dso(self.Vhk))
            


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
            low_rank_layer = LowRankLinear(child.in_features, child.out_features, rank = rank, initialise = False)

            if rank is not None:
                low_rank_layer.svd_decomposition(child, rank=rank)
                setattr(module, name, low_rank_layer)
            else:
                low_rank_layer.svd_decomposition(child, threshold=threshold)
                setattr(module, name, low_rank_layer)
        else:
            apply_low_rank_compression(child, rank=rank, threshold=threshold)

    return module

def low_rank_layer_replacement(module, rank):

    # Replace all Linear layers with LowRankLinear
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = LowRankLinear(
                child.in_features, 
                child.out_features, 
                rank=rank,
                initialise = True
            )

            setattr(module, name, new_layer)

        else:
            low_rank_layer_replacement(child, rank)
    
    return module


def label_low_rank_gradient_layers(model):

    # label layers for galore optimizer

    galore_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # TODO: Possible layer selection here

        print('enable GaLore for weights in module: ', module_name)
        galore_params.append(module.weight)

    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

    count_parameter_groups(galore_params, regular_params)
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
