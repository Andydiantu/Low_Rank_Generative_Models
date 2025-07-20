import os
from collections import defaultdict  # Needed for per‑bucket accumulation

import numpy as np

from DiT import create_noise_scheduler, create_model
from preprocessing import create_dataloader
from config import TrainingConfig
import torch
import torch.nn.functional as F
from galore_torch import GaLoreEvalAdamW
from tqdm import tqdm
from pathlib import Path
from low_rank_compression import label_low_rank_gradient_layers


def _classify_param(param_name: str) -> str:
    """Bucket a parameter into *attention*, *mlp*, or *others* based on its name."""
    lower = param_name.lower()
    if "attn" in lower or "attention" in lower:
        return "attention"
    if "mlp" in lower or "ff" in lower or "ffn" in lower:
        return "mlp"
    return "others"

def eval_galore_projection_loss_timestep(
    timestep_group, model, dataloader, noise_scheduler, device
):
    
    step_loss_dict = defaultdict(dict)
    avg_f_norm_per_step = []  # New list to store average F_norm_mean per step

    galore_params, regular_params = label_low_rank_gradient_layers(model)
    param_to_name = {param: name for name, param in model.named_parameters()}

    regular_param_names = [param_to_name[p] for p in regular_params]
    galore_param_names = [param_to_name[p] for p in galore_params]

    dict_rank = {
        "total": 96,
        "first": 64,
        "second": 16,
        "third": 16,
        "first_total": 96,
        "second_total": 96,
        "third_total": 96,
    }

    rank = dict_rank[timestep_group]

    param_groups = [
        {"params": regular_params, "param_names": regular_param_names},
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": 200,
            "scale": 1.0,
            "proj_type": "std",
            "param_names": galore_param_names,
        },
    ]
    optimizer = GaLoreEvalAdamW(param_groups, lr=0.0001, weight_decay=0.0)

    # Inner progress bar for batches within current timestep
    # Limit to 200 iterations or the dataloader length, whichever is smaller
    max_iterations = min(200, len(dataloader))
    # Track individual layers instead of buckets
    layer_f_norm = defaultdict(list)
    layer_cos_sim = defaultdict(list)
    batch_pbar = tqdm(
        total=max_iterations,
        desc=f"Timestep {timestep_group}" if timestep_group is not None else "All Timesteps",
        disable="SLURM_JOB_ID" in os.environ,
        leave=False,
        position=1,
    )

    

    for step, batch in enumerate(dataloader):
        # Limit to 200 iterations
        if step >= 200:
            break
            
        clean_images = batch["img"].to(device)
        noise = torch.randn(clean_images.shape, device=device)
        batch_size = clean_images.shape[0]

        if timestep_group == "total" or timestep_group == "low_total":
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
            ).long()

        elif timestep_group == "first":
            timesteps = torch.randint(
                0,
                200,
                (batch_size,),
                device=clean_images.device,
            ).long()

        elif timestep_group == "second":
            timesteps = torch.randint(
                200,
                500,
                (batch_size,),
                device=clean_images.device,
            ).long()
        elif timestep_group == "third":
            timesteps = torch.randint(
                500,
                1000,
                (batch_size,),
                device=clean_images.device,
            ).long()
        elif timestep_group == "first_total" or timestep_group == "second_total" or timestep_group == "third_total":
            if step == 0:
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=clean_images.device,
                ).long()
            else:
                if timestep_group == "first_total":
                    timesteps = torch.randint(
                        0,
                        200,
                        (batch_size,),
                        device=clean_images.device,
                    ).long()
                elif timestep_group == "second_total":
                    timesteps = torch.randint(
                        200,
                        500,
                        (batch_size,),
                        device=clean_images.device,
                    ).long()
                elif timestep_group == "third_total":
                    timesteps = torch.randint(
                        500,
                        1000,
                        (batch_size,),
                        device=clean_images.device,
                    ).long()

        # print(f"range of timesteps: {timesteps.min()} to {timesteps.max()}")

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
        _, projection_loss_dict = optimizer.step()
        optimizer.zero_grad()

        # Collect data per individual layer
        for param_name, (err_F, err_cos) in projection_loss_dict.items():
            layer_f_norm[param_name].append(err_F)
            layer_cos_sim[param_name].append(err_cos)

        # Update batch progress bar with current loss info
        batch_pbar.set_postfix(
            {
                "batch_loss": f"{loss.item():.4f}",
            }
        )
        batch_pbar.update(1)

        # Store stats for each individual layer
        step_loss_dict[step] = {}
        for param_name in projection_loss_dict.keys():
            f_arr = torch.tensor(layer_f_norm[param_name], dtype=torch.float64)
            c_arr = torch.tensor(layer_cos_sim[param_name], dtype=torch.float64)
            step_loss_dict[step][param_name] = {
                "F_norm_mean": float(f_arr.mean()),
                "F_norm_std": float(f_arr.std()),
                "cos_sim_mean": float(c_arr.mean()),
                "cos_sim_std": float(c_arr.std()),
            }

        # Calculate average F_norm_mean across all layers for this step
        f_norm_means = [step_loss_dict[step][param_name]["F_norm_mean"] 
                       for param_name in step_loss_dict[step].keys()]
        avg_f_norm = np.mean(f_norm_means) if f_norm_means else 0.0
        avg_f_norm_per_step.append(avg_f_norm)

        

    batch_pbar.close()
    return step_loss_dict, avg_f_norm_per_step


import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_step_loss(
    step_loss_dicts: list[dict],
    metric: str = "F_norm",                    # or "cos_sim"
    figsize: tuple[int, int] = (10, 6),
    save_dir: str | os.PathLike = "plots_per_layer",
    cmap_name: str = "turbo",                  # perceptually-uniform, 256 colours
):
    """
    Draw one figure per individual layer. Each shows one coloured line
    per timestep. Existing PNGs are overwritten on every call.

    Parameters
    ----------
    step_loss_dicts : list[dict]
        The growing list of eval_galore_projection_loss_timestep outputs.
    metric : {"F_norm", "cos_sim"}
        Statistic to draw (plots <metric>_mean).
    figsize : tuple[int, int]
        Size of each figure.
    save_dir : str | Path
        Output directory for PNGs.
    cmap_name : str
        Matplotlib colormap name used to colour the lines.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_ts = len(step_loss_dicts)
    cmap = cm.get_cmap(cmap_name, num_ts)            # discretise into N colours
    norm = mcolors.Normalize(vmin=0, vmax=num_ts - 1)

    # Collect all unique layer names from all timesteps
    all_layer_names = set()
    for step_loss_dict in step_loss_dicts:
        for step_data in step_loss_dict.values():
            all_layer_names.update(step_data.keys())

    # Create one plot per layer
    for layer_name in sorted(all_layer_names):
        fig, ax = plt.subplots(figsize=figsize)

        for t_idx, step_loss_dict in enumerate(step_loss_dicts):
            steps_sorted = np.array(sorted(step_loss_dict.keys()))
            means = np.array(
                [step_loss_dict[s].get(layer_name, {}).get(f"{metric}_mean", np.nan)
                 for s in steps_sorted],
                dtype=float,
            )
            valid = ~np.isnan(means)
            if not valid.any():
                continue   # layer missing for this timestep

            ax.plot(
                steps_sorted[valid],
                means[valid],
                label=f"t={t_idx}",
                color=cmap(norm(t_idx)),
                linewidth=1.1,
                alpha=0.85,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{layer_name} – {metric.replace('_', ' ')} vs Step")
        ax.grid(True, ls="--", lw=0.4)
        # show max 12 legend entries, else colourbar is clearer
        if num_ts <= 12:
            ax.legend(fontsize=8, ncol=2, loc="best")
        else:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label("timestep index")

        fig.tight_layout()
        
        # Create filename with group prefix
        group = _classify_param(layer_name)
        # Clean layer name for filename (replace dots and other special chars)
        clean_layer_name = layer_name.replace(".", "_").replace("/", "_")
        out_path = f"galore_eval_{group}_{clean_layer_name}_{metric}.png"
        fig.savefig(save_dir + "/" + out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

def plot_all_layers_per_timestep(
    step_loss_dict: dict,
    timestep: int,
    metric: str = "F_norm",
    figsize: tuple[int, int] = (12, 8),
    save_dir: str | os.PathLike = "plots_per_layer",
    cmap_name: str = "tab20",
):
    """
    Create a plot showing all layers together for a single timestep.
    
    Parameters
    ----------
    step_loss_dict : dict
        The output from eval_galore_projection_loss_timestep for one timestep.
    timestep : int
        The timestep number.
    metric : {"F_norm", "cos_sim"}
        Statistic to draw (plots <metric>_mean).
    figsize : tuple[int, int]
        Size of the figure.
    save_dir : str | Path
        Output directory for PNGs.
    cmap_name : str
        Matplotlib colormap name used to colour the lines.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect all unique layer names
    all_layer_names = set()
    for step_data in step_loss_dict.values():
        all_layer_names.update(step_data.keys())
    
    all_layer_names = sorted(all_layer_names)
    
    if not all_layer_names:
        return  # No data to plot
    
    # Define colors for each group
    group_colors = {
        "attention": "#1f77b4",  # blue
        "mlp": "#ff7f0e",        # orange
        "others": "#2ca02c"      # green
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each layer
    for layer_name in all_layer_names:
        steps_sorted = np.array(sorted(step_loss_dict.keys()))
        means = np.array(
            [step_loss_dict[s].get(layer_name, {}).get(f"{metric}_mean", np.nan)
             for s in steps_sorted],
            dtype=float,
        )
        valid = ~np.isnan(means)
        if not valid.any():
            continue  # layer missing for this timestep
        
        # Get group for styling and coloring
        group = _classify_param(layer_name)
        linestyle = "-" if group == "attention" else "--" if group == "mlp" else ":"
        color = group_colors[group]
        
        ax.plot(
            steps_sorted[valid],
            means[valid],
            label=f"{layer_name}",
            color=color,
            linewidth=1.2,
            alpha=0.7,
            linestyle=linestyle,
        )
    
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"All Layers – {metric.replace('_', ' ')} vs Step (Timestep {timestep})")
    ax.grid(True, ls="--", lw=0.4, alpha=0.7)
    
    # Legend handling - if too many layers, don't show legend
    num_layers = len(all_layer_names)
    if num_layers <= 20:
        ax.legend(fontsize=6, ncol=2, loc="best", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
    else:
        plt.tight_layout()
    
    # Save plot
    out_path = f"galore_eval_timestep_{timestep:04d}_all_layers_{metric}.png"
    fig.savefig(save_dir + "/" + out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_timestep_group_averages(timestep_group_averages, checkpoint_list, save_dir="plots_per_layer"):
    """
    Plot average performance for each timestep_group across all checkpoints.
    Each timestep_group is a line in the plot.
    
    Parameters
    ----------
    timestep_group_averages : dict
        Dictionary with timestep_group as keys and list of averages per checkpoint as values
    checkpoint_list : list
        List of checkpoint names
    save_dir : str
        Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Define color and style mapping for timestep groups
    style_map = {
        'total': {'color': '#000000', 'linestyle': '-', 'marker': 'o'},
        'first': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        'first_total': {'color': '#1f77b4', 'linestyle': '--', 'marker': 's'},
        'second': {'color': '#ff7f0e', 'linestyle': '-', 'marker': 'o'},
        'second_total': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'},
        'third': {'color': '#2ca02c', 'linestyle': '-', 'marker': 'o'},
        'third_total': {'color': '#2ca02c', 'linestyle': '--', 'marker': 's'},
    }
    
    for timestep_group, averages in timestep_group_averages.items():
        # Skip if no data for this group
        if not averages:
            continue
            
        # Convert checkpoint names to integers for x-axis
        checkpoint_nums = [int(ckpt) for ckpt in checkpoint_list]
        
        # Get style for this timestep group
        style = style_map.get(timestep_group, {'color': '#d62728', 'linestyle': '-', 'marker': 'o'})
        
        plt.plot(checkpoint_nums, averages, 
                label=f'{timestep_group}', 
                color=style['color'], 
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=2, 
                markersize=6)
    
    plt.xlabel('Checkpoint')
    plt.ylabel('Average F_norm Performance')
    plt.title('Average Performance per Timestep Group Across Checkpoints')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'timestep_group_averages.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved timestep group averages plot to {save_path}")

def plot_step_performance_per_checkpoint(step_wise_data, checkpoint_list, save_dir="plots_per_layer"):
    """
    For each timestep_group, create a plot showing step performance for each checkpoint.
    Each checkpoint is a line, and there's one plot per timestep_group.
    
    Parameters
    ----------
    step_wise_data : dict
        Dictionary with timestep_group as keys, then checkpoint as keys, then list of step performances
    checkpoint_list : list
        List of checkpoint names
    save_dir : str
        Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Color map for checkpoints
    cmap = plt.cm.get_cmap('viridis', len(checkpoint_list))
    
    for timestep_group, checkpoint_data in step_wise_data.items():
        plt.figure(figsize=(14, 8))
        
        for i, checkpoint in enumerate(checkpoint_list):
            if checkpoint in checkpoint_data:
                step_values = checkpoint_data[checkpoint]
                steps = range(len(step_values))
                
                plt.plot(steps, step_values, 
                        label=f'Checkpoint {checkpoint}', 
                        color=cmap(i), 
                        linewidth=1.5, 
                        alpha=0.8)
        
        plt.xlabel('Step')
        plt.ylabel('Average F_norm Performance')
        plt.title(f'Step Performance per Checkpoint - Timestep Group: {timestep_group}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'step_performance_{timestep_group}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved step performance plot for {timestep_group} to {save_path}")

def main():
    config = TrainingConfig()
    config.train_batch_size = 128
    config.low_rank_gradient = True
    
    checkpoint_list = ["0009", "0029", "0049", "0069", "0089", "0109", "0129", "0149", "0169", "0189", "0209", "0229", "0249"]
    
    # Data structures to store results for plotting
    timestep_group_averages = {
        "total": [],
        "first": [],
        "second": [],
        "third": [],
        "first_total": [],
        "second_total": [],
        "third_total": []
    }
    
    step_wise_data = {
        "total": {},
        "first": {},
        "second": {},
        "third": {},
        "first_total": {},
        "second_total": {},
        "third_total": {}
    }

    for checkpoint_idx, checkpoint in enumerate(checkpoint_list):
        print(f"\n=== Processing Checkpoint {checkpoint} ({checkpoint_idx + 1}/{len(checkpoint_list)}) ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250715_215849" / f"model_{checkpoint}.pt"  

        model = create_model(config)
        model.to(device)
        noise_scheduler = create_noise_scheduler(config)
        model.load_state_dict(torch.load(load_pretrain_model_path))
        print(f"Loaded model from {load_pretrain_model_path}")
        print(f"Using device: {device}")

        dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.6)

        for timestep_group in ["total", "first", "second", "third", "first_total", "second_total", "third_total"]:
            step_loss_dict, avg_f_norm_per_step = eval_galore_projection_loss_timestep(timestep_group, model, dataloader, noise_scheduler, device)
            print(f"Checkpoint {checkpoint}, Timestep group {timestep_group}:")
            print(f"  Shape of avg_f_norm_per_step: {len(avg_f_norm_per_step)}")
            print(f"  Average of all timesteps: {np.mean(avg_f_norm_per_step):.6f}")
            
            # Store data for plotting
            overall_average = np.mean(avg_f_norm_per_step)
            timestep_group_averages[timestep_group].append(overall_average)
            step_wise_data[timestep_group][checkpoint] = avg_f_norm_per_step
        
        # Update plots after each checkpoint is processed
        print(f"Updating plots with data from checkpoint {checkpoint}...")
        
        # Get the checkpoints processed so far
        checkpoints_so_far = checkpoint_list[:checkpoint_idx + 1]
        
        # Create/update the plots with current data
        plot_timestep_group_averages(timestep_group_averages, checkpoints_so_far, save_dir="performance_plot/adaptive_rank")
        plot_step_performance_per_checkpoint(step_wise_data, checkpoints_so_far, save_dir="performance_plot/adaptive_rank")

        torch.save(timestep_group_averages, "timestep_group_averages.pt")
        torch.save(step_wise_data, "step_wise_data.pt")
        
        print(f"Plots updated! Processed {checkpoint_idx + 1}/{len(checkpoint_list)} checkpoints so far.")
    
    print("\n=== All checkpoints processed and plots finalized! ===")

if __name__ == "__main__":
    main()