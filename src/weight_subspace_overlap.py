import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def print_gpu_memory_usage(stage: str = ""):
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def effective_rank(w: torch.Tensor, energy_threshold: float = 0.95) -> int:
    """Compute the effective rank (number of singular values capturing `energy_threshold` of spectral energy)."""
    if w.ndim > 2:
        w = w.flatten(1)
    if w.shape[0] < w.shape[1]:
        w = w.t()
    with torch.no_grad():
        s = torch.linalg.svdvals(w)
        e = s.square()
        cum = torch.cumsum(e, dim=0)
        total = cum[-1]
        if total == 0:
            return 0
        ratio = cum / total
        k = torch.searchsorted(ratio, energy_threshold).item() + 1
        return k


def projection_loss(W1: torch.Tensor, W2: torch.Tensor, energy_threshold: float = 0.95) -> float:
    """Projection loss of W2 onto the principal subspace of W1 up to effective rank."""
    if W1.ndim > 2:
        W1 = W1.flatten(1)
    if W2.ndim > 2:
        W2 = W2.flatten(1)
    U, _, _ = torch.linalg.svd(W1, full_matrices=False)
    P = U[:, :effective_rank(W1, energy_threshold)]
    W2_proj = P @ (P.T @ W2)
    loss = torch.norm(W2 - W2_proj, p='fro') / torch.norm(W2, p='fro')
    return loss.item()


def subspace_overlap(W1: torch.Tensor, W2: torch.Tensor, energy_threshold: float = 0.95):
    """Compute subspace overlap (principal angles + mean singular value) between two weight matrices."""
    device = W1.device
    if W2.device != device:
        W2 = W2.to(device)

    with torch.no_grad():
        if W1.ndim > 2:
            W1 = W1.flatten(1)
        if W2.ndim > 2:
            W2 = W2.flatten(1)

        U1, _, _ = torch.linalg.svd(W1, full_matrices=False)
        U2, _, _ = torch.linalg.svd(W2, full_matrices=False)

        k = max(effective_rank(W1, energy_threshold), effective_rank(W2, energy_threshold))

        U1_k = U1[:, :k]
        U2_k = U2[:, :k]

        M = U1_k.T @ U2_k
        s = torch.linalg.svdvals(M).clamp(0, 1)
        principal_angles = torch.acos(s)
        similarity = s.mean().item()

    return principal_angles, similarity


def create_similarity_matrix(similarities: Dict[str, float], num_groups: int):
    matrix = np.eye(num_groups)
    for pair_key, similarity in similarities.items():
        parts = pair_key.split('_')
        i, j = int(parts[1]), int(parts[3])
        matrix[i, j] = similarity
        matrix[j, i] = similarity
    return matrix


def create_projection_loss_matrix(projection_losses: Dict[str, float], num_groups: int):
    matrix = np.zeros((num_groups, num_groups))
    for pair_key, proj_loss in projection_losses.items():
        parts = pair_key.split('_')
        i, j = int(parts[1]), int(parts[3])
        matrix[i, j] = proj_loss
        matrix[j, i] = proj_loss
    return matrix


def visualize_layer_similarities(overlap_results, num_groups: int, checkpoint_name: str, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    layer_names = list(overlap_results.keys())
    if not layer_names:
        print("No overlap results to visualize")
        return

    n_layers = len(layer_names)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Weight Timestep-Group Similarities by Layer - Checkpoint {checkpoint_name}', fontsize=20)

    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        similarities = overlap_results[layer_name]['similarities']
        matrix = create_similarity_matrix(similarities, num_groups)

        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{layer_name}', fontsize=12, pad=10)
        ax.set_xlabel('Timestep Group', fontsize=10)
        ax.set_ylabel('Timestep Group', fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(num_groups):
            for j in range(num_groups):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if matrix[i, j] < 0.5 else "black",
                        fontsize=9, weight='bold')

    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_dir / f'weight_layer_similarities_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_layer_projection_losses(overlap_results, num_groups: int, checkpoint_name: str, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    layer_names = list(overlap_results.keys())
    if not layer_names:
        print("No overlap results to visualize")
        return

    n_layers = len(layer_names)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Weight Timestep-Group Projection Losses by Layer - Checkpoint {checkpoint_name}', fontsize=20)

    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        projection_losses = overlap_results[layer_name]['projection_losses']
        matrix = create_projection_loss_matrix(projection_losses, num_groups)

        vmax = matrix.max() if matrix.max() > 0 else 1
        im = ax.imshow(matrix, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'{layer_name}', fontsize=12, pad=10)
        ax.set_xlabel('Timestep Group', fontsize=10)
        ax.set_ylabel('Timestep Group', fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(num_groups):
            for j in range(num_groups):
                ax.text(j, i, f'{matrix[i, j]:.3f}',
                        ha="center", va="center",
                        color="white" if matrix[i, j] > vmax / 2 else "black",
                        fontsize=9, weight='bold')

    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_dir / f'weight_layer_projection_losses_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_principal_angles(overlap_results, checkpoint_name: str, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    layer_names = list(overlap_results.keys())
    if not layer_names:
        return

    n_layers = len(layer_names)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Weight Principal Angles Distribution - Checkpoint {checkpoint_name}', fontsize=18)

    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        all_angles = []
        principal_angles_dict = overlap_results[layer_name]['principal_angles']

        for _, angles in principal_angles_dict.items():
            all_angles.extend(angles.cpu().numpy().flatten())

        if all_angles:
            ax.hist(all_angles, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{layer_name}', fontsize=12, pad=10)
            ax.set_xlabel('Principal Angles (radians)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.axvline(np.pi / 2, color='red', linestyle='--', alpha=0.7, label='π/2 (orthogonal)')
            ax.legend(fontsize=9)

    for idx in range(n_layers, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_dir / f'weight_principal_angles_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_similarity_summary(overlap_results, checkpoint_name: str, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if not overlap_results:
        return

    layer_avg_similarities = {}
    layer_avg_projection_losses = {}
    layer_names = []

    for layer_name, results in overlap_results.items():
        similarities = list(results['similarities'].values())
        projection_losses = list(results['projection_losses'].values())
        if similarities and projection_losses:
            layer_avg_similarities[layer_name] = np.mean(similarities)
            layer_avg_projection_losses[layer_name] = np.mean(projection_losses)
            layer_names.append(layer_name.split('.')[-1])

    if not layer_avg_similarities:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    avg_sims = list(layer_avg_similarities.values())
    ax1.bar(range(len(layer_names)), avg_sims, alpha=0.7, edgecolor='black', color='blue')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Average Similarity', fontsize=12)
    ax1.set_title(f'Average Weight Timestep-Group Similarity by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    for i, v in enumerate(avg_sims):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    avg_proj_losses = list(layer_avg_projection_losses.values())
    ax2.bar(range(len(layer_names)), avg_proj_losses, alpha=0.7, edgecolor='black', color='red')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Average Projection Loss', fontsize=12)
    ax2.set_title(f'Average Weight Projection Loss by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    for i, v in enumerate(avg_proj_losses):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    all_similarities_by_layer = []
    for layer_name, results in overlap_results.items():
        similarities = list(results['similarities'].values())
        all_similarities_by_layer.append(similarities)

    ax3.boxplot(all_similarities_by_layer, labels=layer_names)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Similarity', fontsize=12)
    ax3.set_title(f'Weight Similarity Distribution by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.grid(True, alpha=0.3)

    all_projection_losses_by_layer = []
    for layer_name, results in overlap_results.items():
        projection_losses = list(results['projection_losses'].values())
        all_projection_losses_by_layer.append(projection_losses)

    ax4.boxplot(all_projection_losses_by_layer, labels=layer_names)
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Projection Loss', fontsize=12)
    ax4.set_title(f'Weight Projection Loss Distribution by Layer\nCheckpoint {checkpoint_name}', fontsize=14)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_dir / f'weight_similarity_summary_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_overall_timestep_group_similarity(overlap_results, num_groups: int, checkpoint_name: str, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if not overlap_results:
        return

    pair_sums = {}
    pair_counts = {}
    for layer_results in overlap_results.values():
        for pair_key, sim in layer_results['similarities'].items():
            pair_sums[pair_key] = pair_sums.get(pair_key, 0.0) + sim
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    matrix = np.eye(num_groups)
    for pair_key, total in pair_sums.items():
        parts = pair_key.split('_')
        i, j = int(parts[1]), int(parts[3])
        avg_sim = total / max(pair_counts[pair_key], 1)
        matrix[i, j] = avg_sim
        matrix[j, i] = avg_sim

    plt.figure(figsize=(6, 5))
    plt.title(f'Overall Weight Timestep-Group Similarity\nCheckpoint {checkpoint_name}')
    im = plt.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Timestep Group')
    plt.ylabel('Timestep Group')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    for i in range(num_groups):
        for j in range(num_groups):
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha="center", va="center",
                     color="white" if matrix[i, j] < 0.5 else "black", fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / f'weight_overall_timestep_similarity_checkpoint_{checkpoint_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_cross_checkpoint_analysis(all_checkpoint_results: Dict[str, dict], save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if not all_checkpoint_results:
        return

    checkpoints = list(all_checkpoint_results.keys())

    all_layer_names = set()
    for results in all_checkpoint_results.values():
        all_layer_names.update(results.keys())
    all_layer_names = sorted(list(all_layer_names))

    fig, axes = plt.subplots(2, 3, figsize=(30, 16))

    ax1 = axes[0, 0]
    for layer_name in all_layer_names[:10]:
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
    ax1.set_title('Weight Similarity Evolution Across Checkpoints', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)

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
    ax2.set_title('Weight Similarity Heatmap Across Checkpoints', fontsize=14)
    ax2.set_xticks(range(len(checkpoints)))
    ax2.set_xticklabels(checkpoints, rotation=45, fontsize=10)
    ax2.set_yticks(range(len(all_layer_names)))
    ax2.set_yticklabels([name.split('.')[-1] for name in all_layer_names], fontsize=8)
    plt.colorbar(im, ax=ax2)

    ax3 = axes[0, 2]
    all_timestep_pairs = set()
    for results in all_checkpoint_results.values():
        for layer_results in results.values():
            all_timestep_pairs.update(layer_results['similarities'].keys())
    all_timestep_pairs = sorted(list(all_timestep_pairs))

    timestep_similarity_matrix = np.zeros((len(all_timestep_pairs), len(checkpoints)))
    for i, timestep_pair in enumerate(all_timestep_pairs):
        for j, checkpoint in enumerate(checkpoints):
            similarities_for_pair = []
            for layer_results in all_checkpoint_results[checkpoint].values():
                if timestep_pair in layer_results['similarities']:
                    similarities_for_pair.append(layer_results['similarities'][timestep_pair])
            if similarities_for_pair:
                timestep_similarity_matrix[i, j] = np.mean(similarities_for_pair)
    im3 = ax3.imshow(timestep_similarity_matrix, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Checkpoint', fontsize=12)
    ax3.set_ylabel('Timestep Pairs', fontsize=12)
    ax3.set_title('Average Weight Similarities Between Timestep Groups', fontsize=14)
    ax3.set_xticks(range(len(checkpoints)))
    ax3.set_xticklabels(checkpoints, rotation=45, fontsize=10)
    ax3.set_yticks(range(len(all_timestep_pairs)))
    pair_labels = [pair.replace('timestep_', '').replace('_vs_', ' vs ') for pair in all_timestep_pairs]
    ax3.set_yticklabels(pair_labels, fontsize=10)
    plt.colorbar(im3, ax=ax3)

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
    ax4.set_title('Overall Weight Similarity Statistics', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45, labelsize=10)

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
    ax5.set_title('Weight Similarity Variance by Layer (Across All Checkpoints)', fontsize=14)
    ax5.set_xticks(range(len(all_layer_names)))
    ax5.set_xticklabels([name.split('.')[-1] for name in all_layer_names], rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    all_timestep_pairs_proj = set()
    for results in all_checkpoint_results.values():
        for layer_results in results.values():
            all_timestep_pairs_proj.update(layer_results['projection_losses'].keys())
    all_timestep_pairs_proj = sorted(list(all_timestep_pairs_proj))

    timestep_projection_matrix = np.zeros((len(all_timestep_pairs_proj), len(checkpoints)))
    for i, timestep_pair in enumerate(all_timestep_pairs_proj):
        for j, checkpoint in enumerate(checkpoints):
            projection_losses_for_pair = []
            for layer_results in all_checkpoint_results[checkpoint].values():
                if timestep_pair in layer_results['projection_losses']:
                    projection_losses_for_pair.append(layer_results['projection_losses'][timestep_pair])
            if projection_losses_for_pair:
                timestep_projection_matrix[i, j] = np.mean(projection_losses_for_pair)
    im6 = ax6.imshow(timestep_projection_matrix, cmap='Reds', aspect='auto')
    ax6.set_xlabel('Checkpoint', fontsize=12)
    ax6.set_ylabel('Timestep Pairs', fontsize=12)
    ax6.set_title('Average Weight Projection Losses Between Timestep Groups', fontsize=14)
    ax6.set_xticks(range(len(checkpoints)))
    ax6.set_xticklabels(checkpoints, rotation=45, fontsize=10)
    ax6.set_yticks(range(len(all_timestep_pairs_proj)))
    pair_labels_proj = [pair.replace('timestep_', '').replace('_vs_', ' vs ') for pair in all_timestep_pairs_proj]
    ax6.set_yticklabels(pair_labels_proj, fontsize=10)
    plt.colorbar(im6, ax=ax6)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_dir / 'weight_cross_checkpoint_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_weight_subspace_overlap(weights_by_group: List[Dict[str, torch.Tensor]], energy_threshold: float = 0.95):
    """Compute subspace overlaps for same layers across different timestep-group models."""
    if len(weights_by_group) < 2:
        print("Need at least 2 groups to calculate overlap")
        return {}

    ref_keys = [k for k, v in weights_by_group[0].items() if v.ndim >= 2]
    if not ref_keys:
        print("No eligible weight matrices (ndim >= 2) found in first group")
        return {}

    overlap_results = {}

    for layer_name in tqdm(ref_keys, desc="Analyzing layers"):
        # Check layer exists in all groups and is matrix-like
        layer_weights = []
        valid = True
        for gi, wdict in enumerate(weights_by_group):
            if layer_name not in wdict or wdict[layer_name].ndim < 2:
                print(f"Skipping {layer_name}: missing or not matrix in group {gi}")
                valid = False
                break
            layer_weights.append(wdict[layer_name])
        if not valid:
            continue

        layer_overlaps = {}
        layer_similarities = {}
        layer_projection_losses = {}

        for i in range(len(layer_weights)):
            for j in range(i + 1, len(layer_weights)):
                Wi = layer_weights[i]
                Wj = layer_weights[j]

                device = Wi.device
                if Wj.device != device:
                    Wj = Wj.to(device)

                try:
                    principal_angles, similarity = subspace_overlap(Wi, Wj, energy_threshold)
                    proj_loss = projection_loss(Wi, Wj, energy_threshold)

                    pair_key = f"timestep_{i}_vs_{j}"
                    layer_overlaps[pair_key] = principal_angles
                    layer_similarities[pair_key] = similarity
                    layer_projection_losses[pair_key] = proj_loss
                except Exception as e:
                    print(f"Error on {layer_name} between groups {i} and {j}: {e}")
                    continue

        if layer_overlaps:
            overlap_results[layer_name] = {
                'principal_angles': layer_overlaps,
                'similarities': layer_similarities,
                'projection_losses': layer_projection_losses
            }

            avg_sim = sum(layer_similarities.values()) / len(layer_similarities)
            avg_proj = sum(layer_projection_losses.values()) / len(layer_projection_losses)
            print(f"Average similarity for {layer_name}: {avg_sim:.4f}")
            print(f"Average projection loss for {layer_name}: {avg_proj:.4f}")

    return overlap_results


def load_group_weights_for_checkpoint(group_dirs: List[Path], checkpoint: str, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Load weight tensors (ndim >= 2) from each group directory for the given checkpoint."""
    weights_list: List[Dict[str, torch.Tensor]] = []
    for gidx, gdir in enumerate(group_dirs):
        ckpt_path = Path(gdir) / f"model_{checkpoint}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        # Filter to matrix-like weights only, prefer '.weight' params
        weight_dict: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim >= 2 and key.endswith('.weight'):
                weight_dict[key] = tensor.detach().to(device)
        if not weight_dict:
            print(f"Warning: No eligible weight matrices found in {ckpt_path}")
        weights_list.append(weight_dict)
    return weights_list


def main():
    parser = argparse.ArgumentParser(description="Compute weight subspace overlaps across timestep-group-trained models.")
    parser.add_argument('--group_dirs', type=str, action='append', required=True,
                        help='Paths to the five timestep-group model directories (one per group). Provide this flag 5 times.')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='List of checkpoint identifiers, e.g., 1999 2099 2199')
    parser.add_argument('--energy_threshold', type=float, default=0.95,
                        help='Energy threshold for effective rank (default: 0.95)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations (default: visualizations_new/weight_subspace_overlap)')

    args = parser.parse_args()

    group_dirs = [Path(p) for p in args.group_dirs]
    if len(group_dirs) < 2:
        raise ValueError("At least two group directories are required")

    num_groups = len(group_dirs)
    checkpoints = args.checkpoints
    energy_threshold = args.energy_threshold

    root_save = Path(args.save_dir) if args.save_dir else (Path(__file__).parent.parent / "visualizations_new" / "weight_subspace_overlap")
    root_save.mkdir(parents=True, exist_ok=True)
    print(f"Visualizations will be saved to: {root_save}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_gpu_memory_usage("start")

    all_checkpoint_results: Dict[str, dict] = {}

    for cp_idx, checkpoint in enumerate(checkpoints):
        print(f"\n=== Processing Checkpoint {checkpoint} ({cp_idx + 1}/{len(checkpoints)}) ===")
        print_gpu_memory_usage("before loading weights")

        weights_by_group = load_group_weights_for_checkpoint(group_dirs, checkpoint, device)
        print("Loaded weights for all groups")
        print_gpu_memory_usage("after loading weights")

        print(f"\n=== Calculating weight subspace overlap for checkpoint {checkpoint} ===")
        overlap_results = get_weight_subspace_overlap(weights_by_group, energy_threshold=energy_threshold)
        all_checkpoint_results[checkpoint] = overlap_results
        print_gpu_memory_usage("after subspace analysis")

        # Per-checkpoint visualizations
        try:
            visualize_layer_similarities(overlap_results, num_groups, checkpoint, root_save)
            print(f"\u2713 Created weight layer similarity heatmaps for checkpoint {checkpoint}")

            visualize_layer_projection_losses(overlap_results, num_groups, checkpoint, root_save)
            print(f"\u2713 Created weight layer projection loss heatmaps for checkpoint {checkpoint}")

            visualize_principal_angles(overlap_results, checkpoint, root_save)
            print(f"\u2713 Created weight principal angles plots for checkpoint {checkpoint}")

            visualize_similarity_summary(overlap_results, checkpoint, root_save)
            print(f"\u2713 Created weight similarity summary plots for checkpoint {checkpoint}")

            visualize_overall_timestep_group_similarity(overlap_results, num_groups, checkpoint, root_save)
            print(f"\u2713 Created overall weight timestep-group similarity heatmap for checkpoint {checkpoint}")
        except Exception as e:
            print(f"Error creating visualizations for checkpoint {checkpoint}: {e}")

        print(f"Completed analysis for checkpoint {checkpoint}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-checkpoint analysis
    print("\n=== Creating cross-checkpoint analysis ===")
    try:
        visualize_cross_checkpoint_analysis(all_checkpoint_results, root_save)
        print("\u2713 Created weight cross-checkpoint analysis")
    except Exception as e:
        print(f"Error creating cross-checkpoint analysis: {e}")

    print("\n=== Analysis Complete ===")
    print(f"All visualizations saved to: {root_save}")
    print("Generated files:")
    for file in root_save.glob("*.png"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()











