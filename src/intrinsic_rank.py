import argparse
import os
from typing import List, Dict, Tuple

import torch
import matplotlib.pyplot as plt
import math  # for numerical stability

from diffusers import DiTTransformer2DModel
from DiT_trainer import create_model
from config import TrainingConfig
from pathlib import Path


def load_dit_model(model_path: str) -> DiTTransformer2DModel:
    """Load a DiTTransformer2DModel from a local directory or HF hub name."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

    try:
        model = DiTTransformer2DModel.from_pretrained(model_path)
    except (EnvironmentError, ValueError):
        # Fallback: assume it's a PyTorch state_dict file along with a config.json in the same dir
        config_dir = os.path.dirname(model_path)
        model = DiTTransformer2DModel.from_pretrained(config_dir, state_dict=torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def effective_rank(w: torch.Tensor, energy_threshold: float = 0.99) -> int:
    """Compute the effective rank (number of singular values capturing `energy_threshold` of spectral energy)."""
    if w.ndim > 2:
        w = w.flatten(1)  # (out_channels, in_channels * kernel_size)
    # Ensure smaller dimension is along rows
    if w.shape[0] < w.shape[1]:
        w = w.t()
    with torch.no_grad():
        s = torch.linalg.svdvals(w)
        cumulative_energy = torch.cumsum(s, dim=0) / s.sum()
        return int((cumulative_energy < energy_threshold).sum().item() + 1)


def analyse_model(model: DiTTransformer2DModel, energy_threshold: float = 0.99) -> List[Dict]:
    """Return list of dictionaries with rank information for every weight matrix."""
    results = []
    for name, param in model.named_parameters():
        if "weight" not in name or param.ndim == 1:
            continue  # Skip biases and LayerNorm weights (1D)
        w = param.detach().cpu()
        min_dim = min(w.shape[0], int(torch.prod(torch.tensor(w.shape[1:])).item()))
        eff_rank = effective_rank(w, energy_threshold)
        full_rank = torch.linalg.matrix_rank(w.flatten(1)).item()
        # Classify layer type for separate analysis
        lname = name.lower()
        if any(kw in lname for kw in ["attn", "attention", "q_proj", "k_proj", "v_proj", "qkv", "out_proj"]):
            category = "attention"
        elif any(kw in lname for kw in ["mlp", "fc", "dense", "feed_forward", "ff"]):
            category = "mlp"
        else:
            category = "other"

        results.append({
            "name": name,
            "shape": tuple(w.shape),
            "min_dim": min_dim,
            "effective_rank": eff_rank,
            "full_rank": full_rank,
            "ratio": eff_rank / min_dim,
            "category": category,
        })
    return results


def _plot_single_category(res: List[Dict], title_prefix: str, output_path: str):
    """Internal helper to create bar+hist figure for a single category."""
    if not res:
        print(f"[WARN] No layers found for category '{title_prefix}'. Skipping plot.")
        return

    names = [r["name"] for r in res]
    ratios = [r["ratio"] for r in res]
    eff_ranks = [r["effective_rank"] for r in res]

    num_layers = len(res)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, max(4, num_layers * 0.25)), constrained_layout=True)

    # Bar chart
    ax1.barh(range(num_layers), ratios, color="steelblue")
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("Effective Rank / Min Dimension")
    ax1.set_title(f"{title_prefix}: Intrinsic Rank Ratio")

    # Histogram
    ax2.hist(eff_ranks, bins=30, color="tomato", edgecolor="black")
    ax2.set_xlabel("Effective Rank")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{title_prefix}: Rank Distribution")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {title_prefix} visualisation to {output_path}")


def plot_results_by_category(results: List[Dict], output_stem: str):
    """Produce separate plots for attention and mlp layers.

    output_stem is the base file name (with or without .png). We will append
    _attention.png and _mlp.png.
    """
    if output_stem.lower().endswith(".png"):
        output_stem = output_stem[:-4]

    categories = {"attention": [], "mlp": [], "other": []}
    for r in results:
        categories[r["category"]].append(r)

    _plot_single_category(categories["attention"], "Attention Layers", f"{output_stem}_attention.png")
    _plot_single_category(categories["mlp"], "MLP / Fully-Connected Layers", f"{output_stem}_mlp.png")
    if categories["other"]:
        _plot_single_category(categories["other"], "Other Layers", f"{output_stem}_other.png")


def plot_residual_energy(model: DiTTransformer2DModel, energy_threshold: float, output_path: str, max_rank: int = 512):
    """Plot residual (1 - cumulative) energy curves for all weight matrices.

    Each curve corresponds to a single weight matrix. The y-axis is on a
    logarithmic scale, showing how quickly the residual energy decays as more
    singular directions are added.

    Parameters
    ----------
    model : DiTTransformer2DModel
        The model whose weight matrices are analysed.
    energy_threshold : float
        The energy threshold used elsewhere (e.g., 0.95). A horizontal line
        is drawn at (1−threshold).
    output_path : str
        Destination PNG path (e.g., "spectra.png").
    max_rank : int, optional
        Truncate spectra to this length for readability/performance.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, param in model.named_parameters():
        if "weight" not in name or param.ndim == 1:
            continue  # skip biases/LN

        w = param.detach().cpu()
        if w.ndim > 2:
            w = w.flatten(1)

        # Singular values (descending)
        s = torch.linalg.svdvals(w)
        # Limit to max_rank to keep plot uncluttered
        if s.numel() > max_rank:
            s = s[:max_rank]

        # Residual energy after k components
        energy = torch.cumsum(s ** 2, dim=0) / (s ** 2).sum()
        residual = (1.0 - energy).clamp(min=1e-8)  # avoid log(0)

        ax.plot(
            range(1, residual.numel() + 1),
            residual.numpy(),
            alpha=0.3,
        )

    # Horizontal line showing chosen threshold (residual = 1 - threshold)
    ax.axhline(1.0 - energy_threshold, color="red", linestyle="--", linewidth=1, label=f"{energy_threshold:.2f} energy")

    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Residual energy 1 - E(k)")
    ax.set_title("Residual Energy vs Rank across DiT Weight Matrices")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(loc="upper right")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved residual-energy plot to {output_path}")


# ----------------------
# Spectra-by-category utilities
# ----------------------


def _collect_params_by_category(model: DiTTransformer2DModel) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
    """Return dict mapping category -> list of (name, tensor) tuples."""
    buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = {"attention": [], "mlp": [], "other": []}
    for name, param in model.named_parameters():
        if "weight" not in name or param.ndim == 1:
            continue

        lname = name.lower()
        entry = (name, param.detach().cpu())
        if any(kw in lname for kw in ["attn", "attention", "q_proj", "k_proj", "v_proj", "qkv", "out_proj"]):
            buckets["attention"].append(entry)
        elif any(kw in lname for kw in ["mlp", "fc", "dense", "feed_forward", "ff"]):
            buckets["mlp"].append(entry)
        else:
            buckets["other"].append(entry)
    return buckets


def _plot_residual_energy_from_params(params: List[Tuple[str, torch.Tensor]], energy_threshold: float, output_path: str, title_prefix: str, max_rank: int = 512, legend_limit: int = 20):
    """Helper that plots residual-energy curves for a given list of (name, tensor) pairs with a legend.

    legend_limit caps the number of curve labels shown to avoid overcrowding. All
    curves are still drawn, but only the first *legend_limit* are labelled.
    """
    if not params:
        print(f"[WARN] No parameters provided for {title_prefix}. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, w) in enumerate(params):
        if w.ndim > 2:
            w = w.flatten(1)
        s = torch.linalg.svdvals(w)
        if s.numel() > max_rank:
            s = s[:max_rank]
        energy = torch.cumsum(s ** 2, dim=0) / (s ** 2).sum()
        residual = (1.0 - energy).clamp(min=1e-8)

        label = name if idx < legend_limit else None
        ax.plot(range(1, residual.numel() + 1), residual.numpy(), alpha=0.3, label=label)

    # Horizontal line at chosen threshold
    ax.axhline(1.0 - energy_threshold, color="red", linestyle="--", linewidth=1, label=f"{energy_threshold:.2f} energy")

    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Residual energy 1 - E(k)")
    ax.set_title(f"{title_prefix}: Residual Energy vs Rank")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(loc="upper right", fontsize=6, ncol=1)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {title_prefix} spectra to {output_path}")


def plot_residual_energy_by_category(model: DiTTransformer2DModel, energy_threshold: float, output_stem: str):
    """Generate residual-energy overlay plots per layer category."""
    if output_stem.lower().endswith(".png"):
        output_stem = output_stem[:-4]

    buckets = _collect_params_by_category(model)

    _plot_residual_energy_from_params(
        buckets["attention"],
        energy_threshold,
        f"{output_stem}_attention_spectra.png",
        "Attention Layers",
    )

    _plot_residual_energy_from_params(
        buckets["mlp"],
        energy_threshold,
        f"{output_stem}_mlp_spectra.png",
        "MLP / Fully-Connected Layers",
    )

    if buckets["other"]:
        _plot_residual_energy_from_params(
            buckets["other"],
            energy_threshold,
            f"{output_stem}_other_spectra.png",
            "Other Layers",
        )

# ----------------------
# main
# ----------------------


def main():
    parser = argparse.ArgumentParser(description="Analyse intrinsic rank of DiT weights and plot results.")
    parser.add_argument("--energy", type=float, default=0.95, help="Energy threshold for effective rank computation.")
    parser.add_argument("--out", type=str, default="intrinsic_rank.png", help="Output path for the plot.")
    args = parser.parse_args()

    config = TrainingConfig()
    model = create_model(config)
    model.load_state_dict(torch.load(Path(Path(__file__).parent.parent, config.pretrained_model_path)))
    model.eval()

    results = analyse_model(model, args.energy)

    # Sort by ratio descending for clearer plot ordering
    results.sort(key=lambda x: x["ratio"], reverse=True)

    plot_results_by_category(results, args.out)

    # Per-category spectra overlays
    plot_residual_energy_by_category(model, args.energy, args.out)


if __name__ == "__main__":
    main() 