from DiT import create_noise_scheduler, create_model
from preprocessing import create_dataloader
from config import TrainingConfig
import torch
from tqdm import tqdm
from pathlib import Path
from torch import nn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import json
from datetime import datetime


def _effective_rank(G: torch.Tensor) -> float:
    """Frobenius-norm-based effective rank of a matrix G (float32 on CPU/GPU)."""
    # singular values in descending order
    S = torch.linalg.svdvals(G)           # shape [min(P,K)]

    # The largest singular value S[0] is the operator norm.
    # If it's zero, the matrix is a zero matrix.
    if S.numel() == 0 or S[0] == 0:
        print(f"its a zero matrix: {S}")
        return 0.0

    # The effective rank is the ratio of the squared Frobenius norm to the
    # squared operator norm.
    # Squared Frobenius norm = sum of squared singular values
    # Squared operator norm = max singular value squared
    erank = torch.sum(S**2) / (S[0]**2)

    return float(erank)    


 

def eval_gradient_timestep_group(
    max_timestep: int,
    min_timestep: int,
    model: nn.Module,
    dataloader,
    noise_scheduler,
    device,
    num_gradient_samples: int = 1,       # ❶ how many mini‑batches to stack
) -> Tuple[float, float]:
    """
    Return (effective_rank, frobenius_norm) of the **stacked** parameter‑gradient
    matrix obtained from `num_gradient_samples` random mini‑batches whose
    timesteps are drawn uniformly from [min_timestep, max_timestep).
    """
    model.train()                         # keep BN / dropout behaviour
    param_list = [p for p in model.parameters() if p.requires_grad]

    grads = []                            # will hold K gradient dictionaries
    dataloader_iter = iter(dataloader)

    for _ in range(num_gradient_samples):
        # ----- sample one mini‑batch ----------------------------------------
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
        # print(f"batch_size: {batch_size}")
        timesteps = torch.randint(min_timestep, max_timestep, (batch_size,),
                                  device=device)

        noise = torch.randn_like(clean_images, device=device)

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
        for i, p in enumerate(param_list):
            if p.grad is not None:
                grad_dict[i] = p.grad.detach().clone()
            else:
                # If parameter received no grad (rare), store zero tensor with same shape
                grad_dict[i] = torch.zeros_like(p)

        grads.append(grad_dict)

    # ----- calculate effective rank and Frobenius norm for each 2D gradient matrix -----
    all_eranks = []
    all_frobs = []
    
    for sample_idx in range(num_gradient_samples):
        grad_dict = grads[sample_idx]
        
        for param_idx in sorted(grad_dict.keys()):
            grad_tensor = grad_dict[param_idx]
            
            # Only process 2D gradient matrices
            if len(grad_tensor.shape) == 2:
                # print(f"Processing 2D gradient matrix: {grad_tensor.shape}")
                
                # Calculate effective rank for this 2D matrix
                erank = _effective_rank(grad_tensor)
                # print(f"erank: {erank}")
                all_eranks.append(erank)
                
                # Calculate Frobenius norm for this 2D matrix
                frob = torch.linalg.norm(grad_tensor).item()
                all_frobs.append(frob)
                
                # print(f"  Effective rank: {erank:.4f}, Frobenius norm: {frob:.4f}")


    # Calculate averages
    if all_eranks:
        avg_erank = sum(all_eranks) / len(all_eranks)
        avg_frob = sum(all_frobs) / len(all_frobs)
        print(f"Average effective rank across {len(all_eranks)} 2D matrices: {avg_erank:.4f}")
        print(f"Average Frobenius norm across {len(all_frobs)} 2D matrices: {avg_frob:.4f}")
    else:
        avg_erank = 0.0
        avg_frob = 0.0
        print("No 2D gradient matrices found!")

    return avg_erank, avg_frob


def plot_gradient_analysis(timestep_group_averages, step_wise_data, checkpoint_list, config, boundaries, output_dir=None):
    """
    Create visualizations for effective rank and Frobenius norm across timestep groups.
    
    Args:
        timestep_group_averages: Dict containing average effective ranks for each timestep group
        step_wise_data: Dict containing detailed data for each timestep group and checkpoint
        checkpoint_list: List of checkpoint names
        config: Training configuration object
        output_dir: Optional directory to save plots (if None, plots are displayed)
    """
    # Only regular timestep groups from provided boundaries
    num_timestep_groups = len(boundaries) - 1
    total_groups = num_timestep_groups
    
    # Prepare data for plotting
    timestep_group_labels = []
    avg_effective_ranks = []
    avg_frobenius_norms = []
    
    # Data for individual checkpoints
    effective_rank_data = []  # List of lists for each checkpoint
    frobenius_norm_data = []  # List of lists for each checkpoint
    
    # Regular timestep groups (positions 0, 1, ..., num_timestep_groups-1)
    for timestep_group in range(num_timestep_groups):
        min_timestep = boundaries[timestep_group]
        max_timestep = boundaries[timestep_group + 1]
        timestep_group_labels.append(f"[{min_timestep},{max_timestep})")
        
        # Calculate average effective rank per group
        if timestep_group_averages[timestep_group]:
            avg_effective_ranks.append(np.mean(timestep_group_averages[timestep_group]))
        else:
            avg_effective_ranks.append(0)
        
        # Calculate average Frobenius norms per group
        frobenius_values = [step_wise_data[timestep_group][ckpt]['frobenius_norm'] 
                           for ckpt in checkpoint_list 
                           if ckpt in step_wise_data[timestep_group]]
        if frobenius_values:
            avg_frobenius_norms.append(np.mean(frobenius_values))
        else:
            avg_frobenius_norms.append(0)
    
    # Prepare individual checkpoint data for detailed plots
    for checkpoint in checkpoint_list:
        eff_ranks = []
        frob_norms = []
        
        # Regular timestep groups data (positions 0, 1, ..., num_timestep_groups-1)
        for timestep_group in range(num_timestep_groups):
            if checkpoint in step_wise_data[timestep_group]:
                eff_ranks.append(step_wise_data[timestep_group][checkpoint]['effective_rank'])
                frob_norms.append(step_wise_data[timestep_group][checkpoint]['frobenius_norm'])
            else:
                eff_ranks.append(0)
                frob_norms.append(0)
                
        effective_rank_data.append(eff_ranks)
        frobenius_norm_data.append(frob_norms)
    
    # Create the plots (2x2 layout)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gradient Analysis Across Timestep Groups', fontsize=16, fontweight='bold')
    
    # x_pos should include all groups (regular only)
    x_pos = np.arange(total_groups)
    
    # Plot 1: Average Effective Rank (Frobenius-based)
    ax1.bar(x_pos, avg_effective_ranks, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Timestep Groups')
    ax1.set_ylabel('Effective Rank (Frobenius)')
    ax1.set_title('Average Effective Rank (Frobenius) by Timestep Group')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(timestep_group_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(avg_effective_ranks):
        ax1.text(i, v + (max(avg_effective_ranks) if avg_effective_ranks else 0) * 0.01, f'{v:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Average Frobenius Norm
    ax2.bar(x_pos, avg_frobenius_norms, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Timestep Groups')
    ax2.set_ylabel('Frobenius Norm')
    ax2.set_title('Average Frobenius Norm by Timestep Group')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(timestep_group_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(avg_frobenius_norms):
        ax2.text(i, v + (max(avg_frobenius_norms) if avg_frobenius_norms else 0) * 0.01, f'{v:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Effective Rank Trends Across Checkpoints
    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoint_list)))
    for i, (checkpoint, eff_ranks) in enumerate(zip(checkpoint_list, effective_rank_data)):
        ax3.plot(x_pos, eff_ranks, marker='o', label=f'Epoch {checkpoint}', 
                color=colors[i], linewidth=2, markersize=4)
    
    ax3.set_xlabel('Timestep Groups')
    ax3.set_ylabel('Effective Rank (Frobenius)')
    ax3.set_title('Effective Rank (Frobenius) Trends Across All Checkpoints')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(timestep_group_labels, rotation=45, ha='right')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 5: Frobenius Norm Trends Across Checkpoints
    for i, (checkpoint, frob_norms) in enumerate(zip(checkpoint_list, frobenius_norm_data)):
        ax4.plot(x_pos, frob_norms, marker='s', label=f'Ckpt {checkpoint}', 
                color=colors[i], linewidth=2, markersize=4)
    
    ax4.set_xlabel('Timestep Groups')
    ax4.set_ylabel('Frobenius Norm')
    ax4.set_title('Frobenius Norm Trends Across All Checkpoints')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(timestep_group_labels, rotation=45, ha='right')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_filename = output_path / "gradient_analysis_plots.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_filename}")
    else:
        plt.show()
    
    plt.close()
    
    # Create a separate detailed heatmap
    create_heatmap_visualization(step_wise_data, checkpoint_list, config, boundaries, output_dir)


def create_heatmap_visualization(step_wise_data, checkpoint_list, config, boundaries, output_dir=None):
    """
    Create heatmap visualizations for effective rank and Frobenius norm.
    """
    num_timestep_groups = len(boundaries) - 1
    num_total_groups = num_timestep_groups
    
    # Prepare data matrices
    effective_rank_matrix = np.zeros((len(checkpoint_list), num_total_groups))
    frobenius_norm_matrix = np.zeros((len(checkpoint_list), num_total_groups))
    
    for i, checkpoint in enumerate(checkpoint_list):
        # Regular timestep groups only
        for j in range(num_timestep_groups):
            if j in step_wise_data and checkpoint in step_wise_data[j]:
                effective_rank_matrix[i, j] = step_wise_data[j][checkpoint]['effective_rank']
                frobenius_norm_matrix[i, j] = step_wise_data[j][checkpoint]['frobenius_norm']
    
    
    # Create timestep group labels (following the same order as the main plot)
    timestep_labels = []
    
    # Regular timestep group labels (positions 0, 1, ..., num_timestep_groups-1)
    for group in range(num_timestep_groups):
        min_t = boundaries[group]
        max_t = boundaries[group + 1]
        timestep_labels.append(f"[{min_t},{max_t})")
    
    # Create the heatmaps (1x2 layout)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Gradient Metrics Heatmaps: Checkpoints vs Timestep Groups', fontsize=16, fontweight='bold')
    
    # Effective Rank Heatmap
    im1 = ax1.imshow(effective_rank_matrix, cmap='viridis', aspect='auto')
    ax1.set_title('Effective Rank (Frobenius)')
    ax1.set_xlabel('Timestep Groups')
    ax1.set_ylabel('Checkpoints')
    ax1.set_xticks(range(num_total_groups))
    ax1.set_xticklabels(timestep_labels, rotation=45, ha='right')
    ax1.set_yticks(range(len(checkpoint_list)))
    ax1.set_yticklabels(checkpoint_list)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Effective Rank (Frobenius)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(checkpoint_list)):
        for j in range(num_total_groups):
            ax1.text(j, i, f'{effective_rank_matrix[i, j]:.4f}', 
                     ha="center", va="center", color="white", fontweight='bold')
    
    # Frobenius Norm Heatmap
    im2 = ax2.imshow(frobenius_norm_matrix, cmap='plasma', aspect='auto')
    ax2.set_title('Frobenius Norm')
    ax2.set_xlabel('Timestep Groups')
    ax2.set_ylabel('Checkpoints')
    ax2.set_xticks(range(num_total_groups))
    ax2.set_xticklabels(timestep_labels, rotation=45, ha='right')
    ax2.set_yticks(range(len(checkpoint_list)))
    ax2.set_yticklabels(checkpoint_list)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Frobenius Norm', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(checkpoint_list)):
        for j in range(num_total_groups):
            ax2.text(j, i, f'{frobenius_norm_matrix[i, j]:.4f}', 
                     ha="center", va="center", color="white", fontweight='bold')
    
    # (Energy heatmap removed)
    
    plt.tight_layout()
    
    # Save or show the heatmap
    if output_dir:
        output_path = Path(output_dir)
        heatmap_filename = output_path / "gradient_analysis_heatmaps.png"
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to: {heatmap_filename}")
    else:
        plt.show()
    
    plt.close()


def save_gradient_analysis_data(output_dir,
                                boundaries,
                                checkpoint_list,
                                config,
                                timestep_group_averages,
                                step_wise_data,
                                filename: str = "gradient_analysis_data.json"):
    """
    Persist computed analysis to JSON for later loading and plotting.
    """
    # Convert int keys to strings for JSON compatibility
    tga_serializable = {str(k): v for k, v in timestep_group_averages.items()}
    swd_serializable = {str(k): v for k, v in step_wise_data.items()}

    payload = {
        "meta": {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "num_training_steps": getattr(config, "num_training_steps", None),
            "train_batch_size": getattr(config, "train_batch_size", None),
            "low_rank_gradient": getattr(config, "low_rank_gradient", None),
            "boundaries": boundaries,
            "checkpoint_list": checkpoint_list,
            "output_dir": output_dir,
        },
        "timestep_group_averages": tga_serializable,
        "step_wise_data": swd_serializable,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / filename
    with open(json_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved analysis JSON to: {json_path}")


def replot_from_saved(json_path: str, output_dir: str = None):
    """
    Load a previously saved analysis JSON and regenerate plots/heatmaps.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    boundaries = meta.get("boundaries", [])
    checkpoint_list = meta.get("checkpoint_list", [])
    num_training_steps = meta.get("num_training_steps", None)

    # Reconstruct structures with int keys
    tga = {int(k): v for k, v in data.get("timestep_group_averages", {}).items()}
    swd = {int(k): v for k, v in data.get("step_wise_data", {}).items()}

    cfg = TrainingConfig()
    if num_training_steps is not None:
        cfg.num_training_steps = num_training_steps

    plot_dir = output_dir if output_dir is not None else meta.get("output_dir", None)
    plot_gradient_analysis(tga, swd, checkpoint_list, cfg, boundaries, plot_dir)


def main():
    config = TrainingConfig()
    config.train_batch_size = 64
    config.low_rank_gradient = True
    # Preset timestep boundaries for regular groups
    boundaries = [0, 133, 372, 653, 881, 1000]
    num_timestep_groups = len(boundaries) - 1
    
    checkpoint_list = ["0099", "0199", "0299", "0399", "0499", "0599", "0699", "0799", "0899", "0999", "1099", "1199", "1299", "1399", "1499", "1599", "1699", "1799", "1899", "1999"]
    # checkpoint_list = ["0099", "0199"]
    timestep_group_averages = {i: [] for i in range(num_timestep_groups)}
    step_wise_data = {i: {} for i in range(num_timestep_groups)}

    for checkpoint_idx, checkpoint in enumerate(checkpoint_list):
        print(f"\n=== Processing Checkpoint {checkpoint} ({checkpoint_idx + 1}/{len(checkpoint_list)}) ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250715_215849" / f"model_{checkpoint}.pt"  
        load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250726_235632" / f"model_{checkpoint}.pt"  

        model = create_model(config)
        model.to(device)
        noise_scheduler = create_noise_scheduler(config)
        model.load_state_dict(torch.load(load_pretrain_model_path))
        print(f"Loaded model from {load_pretrain_model_path}")
        print(f"Using device: {device}")

        dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.6)

        for timestep_group in tqdm(range(num_timestep_groups)):
            # Regular timestep groups via preset boundaries
            min_timestep = boundaries[timestep_group]
            max_timestep = boundaries[timestep_group + 1]
            
            print(f"Processing timestep group {timestep_group}: [{min_timestep}, {max_timestep})")
            gradient_effective_rank, gradient_magnitude = eval_gradient_timestep_group(max_timestep, min_timestep, model, dataloader, noise_scheduler, device)
            
            print(f"  Effective Rank: {gradient_effective_rank:.4f}")
            print(f"  Frobenius Norm: {gradient_magnitude:.4f}")
            
            # Store the results
            timestep_group_averages[timestep_group].append(gradient_effective_rank)
            step_wise_data[timestep_group][checkpoint] = {
                'effective_rank': gradient_effective_rank,
                'frobenius_norm': gradient_magnitude
            }
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for timestep_group in range(num_timestep_groups):
        if timestep_group_averages[timestep_group]:
            avg_effective_rank = sum(timestep_group_averages[timestep_group]) / len(timestep_group_averages[timestep_group])
            
            # Regular timestep groups via preset boundaries
            min_timestep = boundaries[timestep_group]
            max_timestep = boundaries[timestep_group + 1]
            
            group_name = f"Timestep Group {timestep_group} [{min_timestep}, {max_timestep})"
            print(f"{group_name}:")
            print(f"  Average Effective Rank: {avg_effective_rank:.4f}")
            
            # Show effective rank for each checkpoint
            effective_ranks = [step_wise_data[timestep_group][ckpt]['effective_rank'] for ckpt in checkpoint_list if ckpt in step_wise_data[timestep_group]]
            frobenius_norms = [step_wise_data[timestep_group][ckpt]['frobenius_norm'] for ckpt in checkpoint_list if ckpt in step_wise_data[timestep_group]]
            if effective_ranks:
                print(f"  Effective Rank Range: {min(effective_ranks):.4f} - {max(effective_ranks):.4f}")
                print(f"  Frobenius Norm Range: {min(frobenius_norms):.4f} - {max(frobenius_norms):.4f}")
            print()

    # Plot the analysis
    output_dir = "heat_maps/test_maps"
    plot_gradient_analysis(timestep_group_averages, step_wise_data, checkpoint_list, config, boundaries, output_dir)

    # Persist computed data for later replots
    save_gradient_analysis_data(
        output_dir=output_dir,
        boundaries=boundaries,
        checkpoint_list=checkpoint_list,
        config=config,
        timestep_group_averages=timestep_group_averages,
        step_wise_data=step_wise_data,
        filename="gradient_analysis_data.json",
    )


if __name__ == "__main__":
    main()