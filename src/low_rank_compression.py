import torch
import torch.nn as nn



class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, initialise=True):
        super().__init__()
        # Limit rank to the maximum possible rank for this layer
        self.rank = in_features if rank is None else min(rank, min(in_features, out_features))
        self.in_features = in_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(out_features, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.rank, in_features))
        
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        if initialise:
            # Initialize using standard linear init strategy
            self.reset_parameters()
    @torch.no_grad()
    def reset_parameters(self):

        W_full = torch.empty(self.out_features, self.in_features)
        nn.init.xavier_uniform_(W_full)

        U_, S_, Vh_ = torch.linalg.svd(W_full, full_matrices=False)
        root_S = S_[: self.rank].sqrt()   

        self.U.copy_(U_[:, : self.rank] * root_S)
        self.V.copy_(Vh_[: self.rank, :] * root_S.view(-1, 1))        
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        return nn.functional.linear(x, self.U @ self.V, self.bias)
    
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

def low_rank_layer_replacement(module, percentage, prefix=""):
    # Replace all Linear layers with LowRankLinear
    for name, child in module.named_children():
        # Build the full path name
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear) and "transformer_blocks.0.ff.net" not in full_name and "proj_out" not in full_name:
            print(f"Replacing {full_name} with LowRankLinear")
            # Calculate original rank of the weight matrix
            print(f"original weight shape: {child.weight.shape}")
            original_rank = min(child.in_features, child.out_features)
            # Calculate new rank as percentage of original rank
            new_rank = int((child.in_features * child.out_features * percentage) / (child.in_features + child.out_features))
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
            low_rank_layer_replacement(child, percentage, prefix=full_name)
    
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
