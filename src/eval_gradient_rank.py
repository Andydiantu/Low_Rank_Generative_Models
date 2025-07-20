import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from DiT import (
    create_model,
    create_noise_scheduler,
    print_model_settings,
    print_noise_scheduler_settings,
)
from low_rank_compression import low_rank_layer_replacement
from preprocessing import create_dataloader
from config import TrainingConfig


def _svdvals_cpu(mat: torch.Tensor) -> torch.Tensor:
    """Utility: flatten to 2-D if needed and return singular values on CPU."""
    if mat.ndim != 2:
        mat = mat.flatten(1)
    return torch.linalg.svdvals(mat.detach().cpu())


def _rank_k_approx(mat: torch.Tensor, rank: int) -> torch.Tensor:
    """Return best rank-k approximation of `mat` (on CPU) via truncated SVD."""
    if rank <= 0:
        raise ValueError("rank must be positive")

    if mat.ndim != 2:
        mat = mat.flatten(1)
    u, s, v = torch.linalg.svd(mat.detach().cpu(), full_matrices=False)

    k = min(rank, s.numel())
    u_k = u[:, :k]
    s_k = s[:k]
    v_k = v[:k, :]
    return (u_k * s_k) @ v_k  # (out, k) * (k,) broadcast then @ (k, in)


def projection_information_loss(mat: torch.Tensor, rank: int) -> float:
    """Relative information loss when projecting `mat` onto its top-`rank` singular vectors.

    Equivalent to `(||mat||_F^2 − ||mat_r||_F^2)/||mat||_F^2` where `mat_r` is the
    best rank-`rank` approximation (SVD truncation). 0 → no loss (rank covers all),
    1 → full loss.
    """
    s = _svdvals_cpu(mat)

    if rank >= s.numel():
        return 0.0  # full energy retained

    s2 = s.pow(2)
    total_energy = s2.sum().item()
    retained_energy = s2[:rank].sum().item()
    loss = (total_energy - retained_energy) / total_energy
    return loss


def projection_cosine_similarity(mat: torch.Tensor, rank: int, eps: float = 1e-8) -> float:
    """Cosine similarity between original matrix and its rank-k approximation."""
    approx = _rank_k_approx(mat, rank)
    dot = (mat * approx.to(mat.device)).sum().item()
    norm_orig = torch.norm(mat).item()
    norm_approx = torch.norm(approx).item()
    return dot / (norm_orig * norm_approx + eps)


def eval_projection_metrics(
    model: nn.Module,
    noise_scheduler,
    dataloader,
    device: torch.device,
    rank: int = 32,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """Compute projection information loss and cosine similarity for each linear layer per diffusion timestep.

    Returns a tuple of two nested dicts: `(info_loss_dict, cos_sim_dict)` where *info_loss* ∈ [0,1]."""
    info_loss_dict: Dict[int, Dict[str, float]] = {}
    cos_sim_dict: Dict[int, Dict[str, float]] = {}
    num_timesteps = noise_scheduler.config.num_train_timesteps

    # Outer progress bar over timesteps
    timestep_pbar = tqdm(
        total=num_timesteps,
        desc="Timesteps",
        disable="SLURM_JOB_ID" in os.environ,
        position=0,
    )

    for timestep in range(num_timesteps):
        # Create zeroed buffers for accumulating gradients of linear layers
        grad_buffers: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.ndim == 2  # Only linear-layer weights
        }

        batch_pbar = tqdm(
            total=len(dataloader),
            desc=f"t = {timestep:4d}",
            disable="SLURM_JOB_ID" in os.environ,
            leave=False,
            position=1,
        )

        for batch in dataloader:
            clean_images = batch["img"].to(device)
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]
            timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

            if "label" in batch and batch["label"] is not None:
                class_labels = batch["label"].to(device)
            else:
                class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            model.zero_grad(set_to_none=True)
            noise_pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # Accumulate gradients
            for name, param in model.named_parameters():
                if name in grad_buffers and param.grad is not None:
                    # accumulate detached gradients (safe, doesn't interfere with autograd)
                    grad_buffers[name] += param.grad.detach()

            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            batch_pbar.update(1)

        batch_pbar.close()

        # Average accumulated gradients over number of batches for scale invariance
        for name in grad_buffers:
            grad_buffers[name] = grad_buffers[name] / len(dataloader)

        # Compute projection information loss and cosine similarity for this timestep
        loss_dict: Dict[str, float] = {}
        cos_dict: Dict[str, float] = {}
        for name, grad in grad_buffers.items():
            loss_dict[name] = projection_information_loss(grad, rank)
            cos_dict[name] = projection_cosine_similarity(grad, rank)

        info_loss_dict[timestep] = loss_dict
        cos_sim_dict[timestep] = cos_dict

        # Progress bar update & show mean info-loss across layers
        mean_loss = sum(loss_dict.values()) / len(loss_dict) if loss_dict else 0.0
        timestep_pbar.set_postfix({"mean_loss": f"{mean_loss:.3f}"})
        timestep_pbar.update(1)

    timestep_pbar.close()
    return info_loss_dict, cos_sim_dict


def plot_info_loss(
    info_loss_dict: Dict[int, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """Plot projection information loss vs timestep for each linear layer."""
    if not info_loss_dict:
        print("[WARN] Empty gradient-rank dict; skipping plot.")
        return

    timesteps = sorted(info_loss_dict.keys())
    layer_names: List[str] = list(next(iter(info_loss_dict.values())).keys())

    plt.figure(figsize=(12, 6))
    for name in layer_names:
        losses = [info_loss_dict[t].get(name, 0.0) for t in timesteps]
        plt.plot(timesteps, losses, label=name, alpha=0.5)

    plt.xlabel("Timestep")
    plt.ylabel("Projection Information Loss (ratio)")
    plt.title("GaLore Projection Information Loss vs Timestep")
    plt.legend(fontsize=6, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# ----------------------------
# Utility: grouped average plot
# ----------------------------


def _classify_layer(name: str) -> str:
    """Return 'attention', 'mlp', or 'other' according to keywords in parameter name."""
    lname = name.lower()
    if any(kw in lname for kw in ["attn", "attention", "q_proj", "k_proj", "v_proj", "qkv", "out_proj"]):
        return "attention"
    if any(kw in lname for kw in ["mlp", "fc", "dense", "feed_forward", "ff"]):
        return "mlp"
    print(f"other: {name}")
    return "other"


def plot_grouped_average_info_loss(
    info_loss_dict_path: str | Path,
    save_path: Optional[str] = "grouped_projection_info_loss.png",
):
    """Load saved info-loss dictionary and plot average loss for attention/mlp/other groups.

    The resulting plot contains exactly three lines (one per group)."""

    info_loss_dict: Dict[int, Dict[str, float]] = torch.load(info_loss_dict_path, map_location="cpu")
    if not info_loss_dict:
        raise ValueError("Loaded info-loss dictionary is empty.")

    timesteps = sorted(info_loss_dict.keys())

    # Initialise accumulators
    groups = {"attention": [], "mlp": [], "other": []}
    for t in timesteps:
        loss_by_layer = info_loss_dict[t]
        # create mapping for each group this timestep
        group_sums = {"attention": 0.0, "mlp": 0.0, "other": 0.0}
        group_counts = {"attention": 0, "mlp": 0, "other": 0}
        for name, loss in loss_by_layer.items():
            cat = _classify_layer(name)
            group_sums[cat] += loss
            group_counts[cat] += 1
        # average for each category (handle zero count)
        for cat in group_sums:
            avg_loss = group_sums[cat] / group_counts[cat] if group_counts[cat] > 0 else float("nan")
            groups[cat].append(avg_loss)

    # Plot
    plt.figure(figsize=(10, 5))
    for cat, series in groups.items():
        plt.plot(timesteps, series, label=cat.capitalize())

    plt.xlabel("Timestep")
    plt.ylabel("Average Projection Information Loss")
    plt.title("Average Information Loss per Layer Group vs Timestep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grouped plot saved to {save_path}")
    else:
        plt.show()


def plot_grouped_average_cos_similarity(
    cos_sim_dict_path: str | Path,
    save_path: Optional[str] = "grouped_projection_cos_similarity.png",
):
    """Load cosine-similarity dictionary & plot grouped averages (3 lines)."""

    cos_dict: Dict[int, Dict[str, float]] = torch.load(cos_sim_dict_path, map_location="cpu")
    if not cos_dict:
        raise ValueError("Cosine-similarity dictionary empty.")

    timesteps = sorted(cos_dict.keys())
    groups = {"attention": [], "mlp": [], "other": []}

    for t in timesteps:
        per_layer = cos_dict[t]
        sums = {g: 0.0 for g in groups}
        counts = {g: 0 for g in groups}
        for name, val in per_layer.items():
            cat = _classify_layer(name)
            sums[cat] += val
            counts[cat] += 1
        for cat in groups:
            groups[cat].append(sums[cat] / counts[cat] if counts[cat] else float("nan"))

    plt.figure(figsize=(10, 5))
    for cat, series in groups.items():
        plt.plot(timesteps, series, label=cat.capitalize())

    plt.xlabel("Timestep")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Average Cosine Similarity (Projection vs Original) per Group")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grouped cosine plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to pretrained weights (modify as needed)
    load_pretrain_model_path = Path(__file__).parent.parent / "logs" / "DiT20250603_001620"/"model_1899.pt"

    # Config & model
    config = TrainingConfig()
    model = create_model(config)

    # Optionally replace Linear layers with LowRank variants before loading weights
    if config.low_rank_pretraining:
        model = low_rank_layer_replacement(model, rank=config.low_rank_rank)
        print(
            f"Number of parameters after low-rank replacement: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    state_dict = torch.load(load_pretrain_model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {load_pretrain_model_path}")

    model.to(device)
    model.train()  # Enable grad computation; no dropout layers present in DiT

    noise_scheduler = create_noise_scheduler(config)

    print_model_settings(model)
    print_noise_scheduler_settings(noise_scheduler)

    # Dataloader (use small subset for speed)
    dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.02)

    # Compute info-loss
    rank_k = config.low_rank_gradient_rank  # GaLore rank
    info_loss_dict, cos_sim_dict = eval_projection_metrics(
        model,
        noise_scheduler,
        dataloader,
        device,
        rank=rank_k,
    )

    # Persist and plot
    torch.save(info_loss_dict, "projection_info_loss_by_timestep.pt")
    torch.save(cos_sim_dict, "projection_cos_sim_by_timestep.pt")
    print("Saved info-loss & cosine dictionaries.")

    plot_info_loss(info_loss_dict, save_path="projection_info_loss_vs_timestep.png")
    plot_grouped_average_cos_similarity("projection_cos_sim_by_timestep.pt")

    # In addition to the previous analysis, also demonstrate grouped plot if the file exists
    info_path = Path("projection_info_loss_by_timestep.pt")
    cos_path = Path("projection_cos_sim_by_timestep.pt")
    if info_path.exists():
        plot_grouped_average_info_loss(info_path)
    if cos_path.exists():
        plot_grouped_average_cos_similarity(cos_path)
