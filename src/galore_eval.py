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
    timestep, model, dataloader, noise_scheduler, device
):
    
    step_loss_dict = defaultdict(dict)

    galore_params, regular_params = label_low_rank_gradient_layers(model)
    param_to_name = {param: name for name, param in model.named_parameters()}

    regular_param_names = [param_to_name[p] for p in regular_params]
    galore_param_names = [param_to_name[p] for p in galore_params]

    param_groups = [
        {"params": regular_params, "param_names": regular_param_names},
        {
            "params": galore_params,
            "rank": 32,
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
    bucket_f_norm = defaultdict(list)
    bucket_cos_sim = defaultdict(list)
    bucket_param_name = defaultdict(list)
    batch_pbar = tqdm(
        total=max_iterations,
        desc=f"Timestep {timestep:4d}",
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

        timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

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

        # print(projection_loss_dict)
        # print(len(projection_loss_dict))


        for param_name, (err_F, err_cos) in projection_loss_dict.items():
            bucket = _classify_param(param_name)
            bucket_f_norm[bucket].append(err_F)
            bucket_cos_sim[bucket].append(err_cos)

        # for param_name, (err_F, err_cos) in projection_loss_dict.items():
        #     print(f"{param_name}: F_norm_error={err_F:.4f}, cos_sim={err_cos:.4f}")

        f_norm_errors = [v[0] for v in projection_loss_dict.values()]
        cos_sims = [v[1] for v in projection_loss_dict.values()]

        f_norm_mean =  torch.mean(torch.tensor(f_norm_errors))
        f_norm_std = torch.std(torch.tensor(f_norm_errors))
        cos_sim_mean = torch.mean(torch.tensor(cos_sims))
        cos_sim_std = torch.std(torch.tensor(cos_sims))

        step_loss_dict[step] = {
            "F_norm_mean": f_norm_mean,
            "F_norm_std": f_norm_std,
            "cos_sim_mean": cos_sim_mean,
            "cos_sim_std": cos_sim_std,
        }

        # Update batch progress bar with current loss info
        batch_pbar.set_postfix(
            {
                "batch_loss": f"{loss.item():.4f}",
            }
        )
        batch_pbar.update(1)

        step_loss_dict[step] = {}
        for bucket in ("attention", "mlp", "others"):
            f_arr = torch.tensor(bucket_f_norm[bucket], dtype=torch.float64)
            c_arr = torch.tensor(bucket_cos_sim[bucket], dtype=torch.float64)
            if f_arr.size == 0:
                step_loss_dict[step][bucket] = None  # bucket empty for this model
                continue
            step_loss_dict[step][bucket] = {
                "F_norm_mean": float(f_arr.mean()),
                "F_norm_std": float(f_arr.std()),
                "cos_sim_mean": float(c_arr.mean()),
                "cos_sim_std": float(c_arr.std()),
            }

    return step_loss_dict


import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_step_loss(
    step_loss_dicts: list[dict],
    metric: str = "F_norm",                    # or "cos_sim"
    buckets: tuple[str, ...] = ("attention", "mlp", "others"),
    figsize: tuple[int, int] = (10, 6),
    save_dir: str | os.PathLike = "plots",
    cmap_name: str = "turbo",                  # perceptually-uniform, 256 colours
):
    """
    Draw three figures (MLP, attention, others).  Each shows one coloured line
    per timestep.  Existing PNGs are overwritten on every call.

    Parameters
    ----------
    step_loss_dicts : list[dict]
        The growing list of eval_galore_projection_loss_timestep outputs.
    metric : {"F_norm", "cos_sim"}
        Statistic to draw (plots <metric>_mean).
    buckets : iterable of str
        Which model buckets to create separate figures for.
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

    for bucket in buckets:
        fig, ax = plt.subplots(figsize=figsize)

        for t_idx, step_loss_dict in enumerate(step_loss_dicts):
            steps_sorted = np.array(sorted(step_loss_dict.keys()))
            means = np.array(
                [step_loss_dict[s].get(bucket, {}).get(f"{metric}_mean", np.nan)
                 for s in steps_sorted],
                dtype=float,
            )
            valid = ~np.isnan(means)
            if not valid.any():
                continue   # bucket missing for this timestep

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
        ax.set_title(f"{bucket.upper()} – {metric.replace('_', ' ')} vs Step")
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
        out_path = f"galore_eval_{bucket}_{metric}.png"
        fig.savefig(save_dir + "/" + out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

def main():
    config = TrainingConfig()
    config.train_batch_size = 64
    step_loss_dicts = []

    step_loss_dicts = torch.load("plots/step_loss_dicts.pt", weights_only=False)

    for timestep in range(116, 1000,2):
        torch.manual_seed(42)
        np.random.seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250621_005915" / "model_1249.pt"  

        model = create_model(config)
        model.to(device)
        noise_scheduler = create_noise_scheduler(config)
        model.load_state_dict(torch.load(load_pretrain_model_path))
        print(f"Loaded model from {load_pretrain_model_path}")
        print(f"Using device: {device}")

        dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.6)


        step_loss_dict = eval_galore_projection_loss_timestep(timestep, model, dataloader, noise_scheduler, device)
        # print(step_loss_dict)
        # print(len(step_loss_dict))
        step_loss_dicts.append(step_loss_dict)


        plot_step_loss(step_loss_dicts, metric="F_norm")
        plot_step_loss(step_loss_dicts, metric="cos_sim")

        torch.save(step_loss_dicts, "plots/step_loss_dicts.pt")

        print(f"timestep {timestep} done")

if __name__ == "__main__":
    main()