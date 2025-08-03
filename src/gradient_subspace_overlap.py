import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from config import TrainingConfig
from DiT import create_model
from preprocessing import create_dataloader
from DiT import create_noise_scheduler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


def print_gpu_memory_usage(stage: str = ""):
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def projection_loss(G1, G2, energy_threshold: float = 0.99):
    # Assume G1 and G2 are (d × n) gradient matrices
    U, _, _ = torch.linalg.svd(G1, full_matrices=False)
    P = U[:, :effective_rank(G1, energy_threshold)]
    G2_proj = P @ (P.T @ G2)
    loss = torch.norm(G2 - G2_proj, p='fro') / torch.norm(G2, p='fro')
    return loss.item()



def effective_rank(w: torch.Tensor, energy_threshold: float = 0.99) -> int:
    """Compute the effective rank (number of singular values capturing `energy_threshold` of spectral energy)."""
    # Keep tensor on original device (GPU if available)
    device = w.device
    
    if w.ndim > 2:
        w = w.flatten(1)  # (out_channels, in_channels * kernel_size)
    # Ensure smaller dimension is along rows
    if w.shape[0] < w.shape[1]:
        w = w.t()
    with torch.no_grad():
        s = torch.linalg.svdvals(w)  # This stays on GPU
        e = s.square()                 # s**2
        cum = torch.cumsum(e, dim=0)
        total = cum[-1]
        if total == 0:
            return 0
        ratio = cum / total
        k = torch.searchsorted(ratio, energy_threshold).item() + 1
        return k



def subspace_overlap(G1, G2, energy_threshold: float = 0.99):
    """Compute subspace overlap between two gradient matrices on GPU."""
    # Ensure both tensors are on the same device (preferably GPU)
    device = G1.device
    if G2.device != device:
        G2 = G2.to(device)
    
    with torch.no_grad():  # Save memory during SVD operations
        # Compute top-k singular vectors (all operations stay on GPU)
        U1, _, _ = torch.linalg.svd(G1, full_matrices=False)
        U2, _, _ = torch.linalg.svd(G2, full_matrices=False)

        effective_rank_1 = effective_rank(G1, energy_threshold)
        effective_rank_2 = effective_rank(G2, energy_threshold)

        k = max(effective_rank_1, effective_rank_2)

        # print(f"Effective rank of U1: {effective_rank_1}")
        # print(f"Effective rank of U2: {effective_rank_2}")

        U1_k = U1[:, :k]
        U2_k = U2[:, :k]

        # Compute cosine of principal angles (GPU operations)
        M = U1_k.T @ U2_k
        singular_values = torch.linalg.svdvals(M)

        # all singular values are in the range [0, 1]
        singular_values = singular_values.clamp(0, 1)

        principal_angles = torch.acos(singular_values)

        # Compute similarity and distance measures
        similarity = (singular_values ** 2).mean().item()
        
    return principal_angles, similarity

def get_gradient_subspace_overlap(grads, energy_threshold: float = 0.99):
    """
    Calculate subspace overlap for the same layers between different timestep groups on GPU.
    
    Args:
        grads: List of gradient dictionaries, one for each timestep group
        energy_threshold: Threshold for effective rank calculation
    
    Returns:
        Dictionary containing overlap results for each layer
    """
    if len(grads) < 2:
        print("Need at least 2 timestep groups to calculate overlap")
        return {}
    
    # Get all layer names (should be consistent across timestep groups)
    layer_names = list(grads[0].keys())
    # print(f"Found {len(layer_names)} layers to analyze")
    
    # Check device of first gradient for reference
    if layer_names and grads[0]:
        first_grad = next(iter(grads[0].values()))
        device = first_grad.device
        print(f"Performing subspace overlap analysis on device: {device}")
    
    overlap_results = {}
    
    for layer_name in tqdm(layer_names, desc="Analyzing layers"):
        # print(f"\n=== Analyzing layer: {layer_name} ===")
        
        # Check if this layer exists in all timestep groups
        layer_grads = []
        for i, grad_dict in enumerate(grads):
            if layer_name in grad_dict:
                layer_grads.append(grad_dict[layer_name])
            else:
                print(f"Warning: Layer {layer_name} not found in timestep group {i}")
                break
        
        if len(layer_grads) != len(grads):
            print(f"Skipping layer {layer_name} due to missing data")
            continue
        
        # Calculate pairwise overlaps between all timestep groups for this layer
        layer_overlaps = {}
        layer_similarities = {}
        layer_projection_losses = {}
        
        for i in range(len(layer_grads)):
            for j in range(i + 1, len(layer_grads)):
                # print(f"Comparing timestep group {i} vs {j} for {layer_name}")
                
                grad_i = layer_grads[i]
                grad_j = layer_grads[j]
                
                # Ensure gradients are on the same device and 2D for SVD
                device = grad_i.device
                if grad_j.device != device:
                    grad_j = grad_j.to(device)
                    
                if grad_i.ndim > 2:
                    grad_i = grad_i.flatten(1)  # Keep on GPU
                    print("FLATTENED")
                if grad_j.ndim > 2:
                    grad_j = grad_j.flatten(1)  # Keep on GPU
                    print("FLATTENED")
                
                try:
                    principal_angles, similarity = subspace_overlap(grad_i, grad_j, energy_threshold)
                    proj_loss = projection_loss(grad_i, grad_j, energy_threshold)
                    
                    pair_key = f"timestep_{i}_vs_{j}"
                    layer_overlaps[pair_key] = principal_angles
                    layer_similarities[pair_key] = similarity
                    layer_projection_losses[pair_key] = proj_loss
                    
                    # print(f"  Similarity: {similarity:.4f}, Projection Loss: {proj_loss:.4f}")
                    
                except Exception as e:
                    print(f"Error calculating overlap for {layer_name} between groups {i} and {j}: {e}")
                    continue
        
        if layer_overlaps:
            overlap_results[layer_name] = {
                'principal_angles': layer_overlaps,
                'similarities': layer_similarities,
                'projection_losses': layer_projection_losses
            }
            
            # Calculate and print average similarity and projection loss for this layer
            total_similarity = 0
            total_projection_loss = 0
            for key, value in layer_similarities.items():
                total_similarity += value
            for key, value in layer_projection_losses.items():
                total_projection_loss += value
            average_similarity = total_similarity / len(layer_similarities)
            average_projection_loss = total_projection_loss / len(layer_projection_losses)
            print(f"Average similarity for {layer_name}: {average_similarity:.4f}")
            print(f"Average projection loss for {layer_name}: {average_projection_loss:.4f}")
    
    return overlap_results


def create_similarity_matrix(similarities, num_timestep_groups):
    """Create a symmetric similarity matrix from pairwise similarities."""
    matrix = np.eye(num_timestep_groups)  # Identity matrix (diagonal = 1)
    
    for pair_key, similarity in similarities.items():
        # Parse timestep indices from key like "timestep_0_vs_1"
        parts = pair_key.split('_')
        i, j = int(parts[1]), int(parts[3])
        matrix[i, j] = similarity
        matrix[j, i] = similarity  # Make symmetric
    
    return matrix


def visualize_layer_similarities(overlap_results, num_timestep_groups, checkpoint_name, save_dir):
    """Create heatmaps showing similarity between timestep groups for each layer."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    layer_names = list(overlap_results.keys())
    if not layer_names:
        print("No overlap results to visualize")
        return
    
    # Create subplot grid
    n_layers = len(layer_names)
    cols = min(3, n_layers)  # Reduce columns for better spacing
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))  # Increased size significantly
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Timestep Group Similarities by Layer - Checkpoint {checkpoint_name}', fontsize=20)
    
    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        similarities = overlap_results[layer_name]['similarities']
        matrix = create_similarity_matrix(similarities, num_timestep_groups)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{layer_name}', fontsize=12, pad=10)
        ax.set_xlabel('Timestep Group', fontsize=10)
        ax.set_ylabel('Timestep Group', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations with better formatting
        for i in range(num_timestep_groups):
            for j in range(num_timestep_groups):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}', 
                             ha="center", va="center", 
                             color="white" if matrix[i, j] < 0.5 else "black",
                             fontsize=9, weight='bold')
    
    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout(pad=3.0)  # Added padding to prevent text overlap
    plt.savefig(save_dir / f'layer_similarities_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_projection_loss_matrix(projection_losses, num_timestep_groups):
    """Create a symmetric projection loss matrix from pairwise projection losses."""
    matrix = np.zeros((num_timestep_groups, num_timestep_groups))  # Zero matrix (diagonal = 0)
    
    for pair_key, proj_loss in projection_losses.items():
        # Parse timestep indices from key like "timestep_0_vs_1"
        parts = pair_key.split('_')
        i, j = int(parts[1]), int(parts[3])
        matrix[i, j] = proj_loss
        matrix[j, i] = proj_loss  # Make symmetric
    
    return matrix


def visualize_layer_projection_losses(overlap_results, num_timestep_groups, checkpoint_name, save_dir):
    """Create heatmaps showing projection losses between timestep groups for each layer."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    layer_names = list(overlap_results.keys())
    if not layer_names:
        print("No overlap results to visualize")
        return
    
    # Create subplot grid
    n_layers = len(layer_names)
    cols = min(3, n_layers)  # Reduce columns for better spacing
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))  # Increased size significantly
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Timestep Group Projection Losses by Layer - Checkpoint {checkpoint_name}', fontsize=20)
    
    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        projection_losses = overlap_results[layer_name]['projection_losses']
        matrix = create_projection_loss_matrix(projection_losses, num_timestep_groups)
        
        # Create heatmap (use 'Reds' colormap since higher projection loss = worse)
        vmax = matrix.max() if matrix.max() > 0 else 1
        im = ax.imshow(matrix, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'{layer_name}', fontsize=12, pad=10)
        ax.set_xlabel('Timestep Group', fontsize=10)
        ax.set_ylabel('Timestep Group', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations with better formatting
        for i in range(num_timestep_groups):
            for j in range(num_timestep_groups):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}', 
                             ha="center", va="center", 
                             color="white" if matrix[i, j] > vmax/2 else "black",
                             fontsize=9, weight='bold')
    
    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout(pad=3.0)  # Added padding to prevent text overlap
    plt.savefig(save_dir / f'layer_projection_losses_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_principal_angles(overlap_results, checkpoint_name, save_dir):
    """Create histograms of principal angles for each layer."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    layer_names = list(overlap_results.keys())
    if not layer_names:
        return
    
    n_layers = len(layer_names)
    cols = min(3, n_layers)  # Reduce columns for better spacing
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))  # Increased size significantly
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Principal Angles Distribution - Checkpoint {checkpoint_name}', fontsize=18)
    
    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Collect all principal angles for this layer
        all_angles = []
        principal_angles_dict = overlap_results[layer_name]['principal_angles']
        
        for pair_key, angles in principal_angles_dict.items():
            all_angles.extend(angles.cpu().numpy().flatten())
        
        if all_angles:
            ax.hist(all_angles, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{layer_name}', fontsize=12, pad=10)
            ax.set_xlabel('Principal Angles (radians)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.axvline(np.pi/2, color='red', linestyle='--', alpha=0.7, label='π/2 (orthogonal)')
            ax.legend(fontsize=9)
    
    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout(pad=3.0)  # Added padding to prevent text overlap
    plt.savefig(save_dir / f'principal_angles_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_similarity_summary(overlap_results, checkpoint_name, save_dir):
    """Create summary visualizations of similarities across layers."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if not overlap_results:
        return
    
    # Calculate average similarity and projection loss for each layer
    layer_avg_similarities = {}
    layer_avg_projection_losses = {}
    layer_names = []
    
    for layer_name, results in overlap_results.items():
        similarities = list(results['similarities'].values())
        projection_losses = list(results['projection_losses'].values())
        if similarities and projection_losses:
            layer_avg_similarities[layer_name] = np.mean(similarities)
            layer_avg_projection_losses[layer_name] = np.mean(projection_losses)
            layer_names.append(layer_name.split('.')[-1])  # Use short name
    
    if not layer_avg_similarities:
        return
    
    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))  # 2x2 layout for more plots
    
    # Plot 1: Bar plot of average similarities by layer
    avg_sims = list(layer_avg_similarities.values())
    ax1.bar(range(len(layer_names)), avg_sims, alpha=0.7, edgecolor='black', color='blue')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Average Similarity', fontsize=12)
    ax1.set_title(f'Average Timestep Similarity by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(avg_sims):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Bar plot of average projection losses by layer
    avg_proj_losses = list(layer_avg_projection_losses.values())
    ax2.bar(range(len(layer_names)), avg_proj_losses, alpha=0.7, edgecolor='black', color='red')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Average Projection Loss', fontsize=12)
    ax2.set_title(f'Average Projection Loss by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(avg_proj_losses):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Box plot of similarity distributions
    all_similarities_by_layer = []
    for layer_name, results in overlap_results.items():
        similarities = list(results['similarities'].values())
        all_similarities_by_layer.append(similarities)
    
    ax3.boxplot(all_similarities_by_layer, labels=layer_names)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Similarity', fontsize=12)
    ax3.set_title(f'Similarity Distribution by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of projection loss distributions
    all_projection_losses_by_layer = []
    for layer_name, results in overlap_results.items():
        projection_losses = list(results['projection_losses'].values())
        all_projection_losses_by_layer.append(projection_losses)
    
    ax4.boxplot(all_projection_losses_by_layer, labels=layer_names)
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Projection Loss', fontsize=12)
    ax4.set_title(f'Projection Loss Distribution by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)  # Added padding to prevent text overlap
    plt.savefig(save_dir / f'similarity_summary_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_cross_checkpoint_analysis(all_checkpoint_results, save_dir):
    """Create visualizations comparing results across different checkpoints."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if not all_checkpoint_results:
        return
    
    checkpoints = list(all_checkpoint_results.keys())
    
    # Get all unique layer names across checkpoints
    all_layer_names = set()
    for results in all_checkpoint_results.values():
        all_layer_names.update(results.keys())
    all_layer_names = sorted(list(all_layer_names))
    
    # Create evolution plot with 2x3 layout for 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(30, 16))  # Increased size significantly
    
    # Plot 1: Average similarity evolution across checkpoints
    ax1 = axes[0, 0]
    for layer_name in all_layer_names[:10]:  # Limit to first 10 layers for readability
        layer_short_name = layer_name
        avg_sims = []
        for checkpoint in checkpoints:
            if layer_name in all_checkpoint_results[checkpoint]:
                similarities = list(all_checkpoint_results[checkpoint][layer_name]['similarities'].values())
                avg_sims.append(np.mean(similarities) if similarities else 0)
            else:
                avg_sims.append(0)
        ax1.plot(checkpoints, avg_sims, marker='o', label=layer_short_name, alpha=0.7)
    
    ax1.set_xlabel('Checkpoint', fontsize=12)
    ax1.set_ylabel('Average Similarity', fontsize=12)
    ax1.set_title('Similarity Evolution Across Checkpoints', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Plot 2: Heatmap of average similarities by checkpoint and layer
    ax2 = axes[0, 1]
    similarity_matrix = np.zeros((len(all_layer_names), len(checkpoints)))
    
    for i, layer_name in enumerate(all_layer_names):
        for j, checkpoint in enumerate(checkpoints):
            if layer_name in all_checkpoint_results[checkpoint]:
                similarities = list(all_checkpoint_results[checkpoint][layer_name]['similarities'].values())
                similarity_matrix[i, j] = np.mean(similarities) if similarities else 0
    
    im = ax2.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Checkpoint', fontsize=12)
    ax2.set_ylabel('Layer', fontsize=12)
    ax2.set_title('Similarity Heatmap Across Checkpoints', fontsize=14)
    ax2.set_xticks(range(len(checkpoints)))
    ax2.set_xticklabels(checkpoints, rotation=45, fontsize=10)
    ax2.set_yticks(range(len(all_layer_names)))
    ax2.set_yticklabels([name.split('.')[-1] for name in all_layer_names], fontsize=8)
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Average similarities between timestep groups heatmap
    ax3 = axes[0, 2]
    
    # Extract all unique timestep pairs
    all_timestep_pairs = set()
    for results in all_checkpoint_results.values():
        for layer_results in results.values():
            all_timestep_pairs.update(layer_results['similarities'].keys())
    all_timestep_pairs = sorted(list(all_timestep_pairs))
    
    # Create heatmap data: timestep pairs vs checkpoints
    timestep_similarity_matrix = np.zeros((len(all_timestep_pairs), len(checkpoints)))
    
    for i, timestep_pair in enumerate(all_timestep_pairs):
        for j, checkpoint in enumerate(checkpoints):
            # Calculate average similarity across all layers for this timestep pair and checkpoint
            similarities_for_pair = []
            for layer_results in all_checkpoint_results[checkpoint].values():
                if timestep_pair in layer_results['similarities']:
                    similarities_for_pair.append(layer_results['similarities'][timestep_pair])
            
            if similarities_for_pair:
                timestep_similarity_matrix[i, j] = np.mean(similarities_for_pair)
    
    im3 = ax3.imshow(timestep_similarity_matrix, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Checkpoint', fontsize=12)
    ax3.set_ylabel('Timestep Pairs', fontsize=12)
    ax3.set_title('Average Similarities Between Timestep Groups', fontsize=14)
    ax3.set_xticks(range(len(checkpoints)))
    ax3.set_xticklabels(checkpoints, rotation=45, fontsize=10)
    ax3.set_yticks(range(len(all_timestep_pairs)))
    # Simplify timestep pair labels
    pair_labels = [pair.replace('timestep_', '').replace('_vs_', ' vs ') for pair in all_timestep_pairs]
    ax3.set_yticklabels(pair_labels, fontsize=10)
    plt.colorbar(im3, ax=ax3)
    
    # Plot 4: Overall similarity statistics
    ax4 = axes[1, 0]
    overall_stats = []
    for checkpoint in checkpoints:
        all_sims = []
        for layer_results in all_checkpoint_results[checkpoint].values():
            all_sims.extend(list(layer_results['similarities'].values()))
        if all_sims:
            overall_stats.append({
                'mean': np.mean(all_sims),
                'std': np.std(all_sims),
                'min': np.min(all_sims),
                'max': np.max(all_sims)
            })
        else:
            overall_stats.append({'mean': 0, 'std': 0, 'min': 0, 'max': 0})
    
    means = [stat['mean'] for stat in overall_stats]
    stds = [stat['std'] for stat in overall_stats]
    
    ax4.errorbar(checkpoints, means, yerr=stds, marker='o', capsize=5)
    ax4.set_xlabel('Checkpoint', fontsize=12)
    ax4.set_ylabel('Overall Similarity', fontsize=12)
    ax4.set_title('Overall Similarity Statistics', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Plot 5: Similarity variance across layers
    ax5 = axes[1, 1]
    layer_variances = []
    for layer_name in all_layer_names:
        layer_sims = []
        for checkpoint in checkpoints:
            if layer_name in all_checkpoint_results[checkpoint]:
                similarities = list(all_checkpoint_results[checkpoint][layer_name]['similarities'].values())
                layer_sims.extend(similarities)
        if layer_sims:
            layer_variances.append(np.var(layer_sims))
        else:
            layer_variances.append(0)
    
    ax5.bar(range(len(all_layer_names)), layer_variances, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Layer', fontsize=12)
    ax5.set_ylabel('Similarity Variance', fontsize=12)
    ax5.set_title('Similarity Variance by Layer (Across All Checkpoints)', fontsize=14)
    ax5.set_xticks(range(len(all_layer_names)))
    ax5.set_xticklabels([name.split('.')[-1] for name in all_layer_names], rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Average projection losses between timestep groups heatmap
    ax6 = axes[1, 2]
    
    # Extract all unique timestep pairs for projection losses
    all_timestep_pairs_proj = set()
    for results in all_checkpoint_results.values():
        for layer_results in results.values():
            all_timestep_pairs_proj.update(layer_results['projection_losses'].keys())
    all_timestep_pairs_proj = sorted(list(all_timestep_pairs_proj))
    
    # Create heatmap data: timestep pairs vs checkpoints for projection losses
    timestep_projection_matrix = np.zeros((len(all_timestep_pairs_proj), len(checkpoints)))
    
    for i, timestep_pair in enumerate(all_timestep_pairs_proj):
        for j, checkpoint in enumerate(checkpoints):
            # Calculate average projection loss across all layers for this timestep pair and checkpoint
            projection_losses_for_pair = []
            for layer_results in all_checkpoint_results[checkpoint].values():
                if timestep_pair in layer_results['projection_losses']:
                    projection_losses_for_pair.append(layer_results['projection_losses'][timestep_pair])
            
            if projection_losses_for_pair:
                timestep_projection_matrix[i, j] = np.mean(projection_losses_for_pair)
    
    im6 = ax6.imshow(timestep_projection_matrix, cmap='Reds', aspect='auto')
    ax6.set_xlabel('Checkpoint', fontsize=12)
    ax6.set_ylabel('Timestep Pairs', fontsize=12)
    ax6.set_title('Average Projection Losses Between Timestep Groups', fontsize=14)
    ax6.set_xticks(range(len(checkpoints)))
    ax6.set_xticklabels(checkpoints, rotation=45, fontsize=10)
    ax6.set_yticks(range(len(all_timestep_pairs_proj)))
    # Simplify timestep pair labels
    pair_labels_proj = [pair.replace('timestep_', '').replace('_vs_', ' vs ') for pair in all_timestep_pairs_proj]
    ax6.set_yticklabels(pair_labels_proj, fontsize=10)
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout(pad=3.0)  # Added padding to prevent text overlap
    plt.savefig(save_dir / 'cross_checkpoint_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_gradient_timestep_group(max_timestep: int, min_timestep: int, model, dataloader, noise_scheduler, device, num_gradient_samples: int = 1, fixed_noise = None, fixed_batch=None, seed=None):

    model.train()
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    grads = []
    
    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Use fixed batch if provided, otherwise use dataloader
    if fixed_batch is not None:
        batch = fixed_batch
    else:
        dataloader_iter = iter(dataloader)

    for _ in range(num_gradient_samples):
        if fixed_batch is None:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

        clean_images = batch["img"].to(device)
        labels = batch.get("label", None)
        if labels is not None:
            labels = labels.to(device)

        batch_size = clean_images.size(0)
        timesteps = torch.randint(min_timestep, max_timestep, (batch_size,), device=device)

        if fixed_noise is None:
            noise = torch.randn_like(clean_images, device=device)
        else:
            noise = fixed_noise

        model.zero_grad(set_to_none=True)

        class_labels = None
        if "label" in batch:
            class_labels = batch["label"].to(device)
        else:
            class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise, reduction="none")
        loss = loss.mean()
        loss.backward()

        # ----- store gradients in original shapes ---------------------------
        grad_dict = {}
        for name, p in named_params:
            if p.grad is not None and len(p.grad.shape) == 2:
                grad_dict[name] = p.grad.detach().clone()

        grads.append(grad_dict)

        print(len(grad_dict.keys()))


    return grads[0]

def main():
    config = TrainingConfig()
    config.train_batch_sie = 128
    config.low_rank_gradient = True
    num_timestep_groups = 10

    checkpoint_list = ["0099", "0199", "0299", "0399", "0499"]
    
    # Create visualization directory
    viz_dir = Path(__file__).parent.parent / "visualizations_new" / "gradient_subspace_overlap"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visualizations will be saved to: {viz_dir}")
    
    # Store results for cross-checkpoint analysis
    all_checkpoint_results = {}

    # Pre-load a fixed batch for gradient collection
    fixed_dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.6)
    fixed_batch = next(iter(fixed_dataloader))

    for checkpoint_idx, checkpoint in enumerate(checkpoint_list):
        print(f"\n=== Processing Checkpoint {checkpoint} ({checkpoint_idx + 1}/{len(checkpoint_list)}) ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print_gpu_memory_usage("start of checkpoint")
        
        # low rank counter part
        new_folder = "DiT20250626_213046"
        
        load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250529_232857" / f"model_{checkpoint}.pt"  

        model = create_model(config)
        model.to(device)
        noise_scheduler = create_noise_scheduler(config)
        model.load_state_dict(torch.load(load_pretrain_model_path))
        print(f"Loaded model from {load_pretrain_model_path}")
        print_gpu_memory_usage("after loading model")

        # dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.6) # This line is no longer needed

        # Collect gradients for all timestep groups
        grads = []
        print_gpu_memory_usage("before gradient collection")
        
        for timestep_group in tqdm(range(num_timestep_groups), desc=f"Collecting gradients for checkpoint {checkpoint}"):
            min_timestep = config.num_training_steps // num_timestep_groups * timestep_group
            max_timestep = min_timestep + config.num_training_steps // num_timestep_groups            

            fixed_noise = torch.randn_like(fixed_batch["img"], device=device)
            grad = get_gradient_timestep_group(max_timestep, min_timestep, model, None, noise_scheduler, device, num_gradient_samples=1, fixed_noise=fixed_noise, fixed_batch=fixed_batch, seed=42)

            grads.append(grad)
            
            # Clear cache periodically during gradient collection
            if (timestep_group + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Collected gradients for {len(grads)} timestep groups")
        print_gpu_memory_usage("after gradient collection")
        
        # Calculate subspace overlap for this checkpoint
        print(f"\n=== Calculating subspace overlap for checkpoint {checkpoint} ===")
        print_gpu_memory_usage("before subspace analysis")
        overlap_results = get_gradient_subspace_overlap(grads, energy_threshold=0.99)
        print_gpu_memory_usage("after subspace analysis")
        
        # Clear GPU cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Store results for cross-checkpoint analysis
        all_checkpoint_results[checkpoint] = overlap_results
        
        # Print summary of results
        print(f"\n--- Summary for checkpoint {checkpoint} ---")
        for layer_name, results in overlap_results.items():
            print(f"Layer name: {layer_name}")
            similarities = results['similarities']
            projection_losses = results['projection_losses']
            avg_similarity = sum(similarities.values()) / len(similarities) if similarities else 0
            avg_projection_loss = sum(projection_losses.values()) / len(projection_losses) if projection_losses else 0
            print(f"Layer {layer_name}: Average similarity = {avg_similarity:.4f}, Average projection loss = {avg_projection_loss:.4f}")
        
        # Create visualizations for this checkpoint
        print(f"\n=== Creating visualizations for checkpoint {checkpoint} ===")
        try:
            visualize_layer_similarities(overlap_results, num_timestep_groups, checkpoint, viz_dir)
            print(f"✓ Created layer similarity heatmaps for checkpoint {checkpoint}")
            
            visualize_layer_projection_losses(overlap_results, num_timestep_groups, checkpoint, viz_dir)
            print(f"✓ Created layer projection loss heatmaps for checkpoint {checkpoint}")
            
            visualize_principal_angles(overlap_results, checkpoint, viz_dir)
            print(f"✓ Created principal angles plots for checkpoint {checkpoint}")
            
            visualize_similarity_summary(overlap_results, checkpoint, viz_dir)
            print(f"✓ Created similarity summary plots for checkpoint {checkpoint}")
            
        except Exception as e:
            print(f"Error creating visualizations for checkpoint {checkpoint}: {e}")
            
        print(f"Completed analysis for checkpoint {checkpoint}")

    # Create cross-checkpoint analysis
    print(f"\n=== Creating cross-checkpoint analysis ===")
    try:
        visualize_cross_checkpoint_analysis(all_checkpoint_results, viz_dir)
        print(f"✓ Created cross-checkpoint analysis")
    except Exception as e:
        print(f"Error creating cross-checkpoint analysis: {e}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"All visualizations saved to: {viz_dir}")
    print(f"Generated files:")
    for file in viz_dir.glob("*.png"):
        print(f"  - {file.name}")

    return all_checkpoint_results

if __name__ == "__main__":
    main()
