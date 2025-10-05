from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, List, Dict

import torch
from torch import nn
import torch.nn.functional as F  # noqa: F401  (kept for parity, not used)
import numpy as np
import matplotlib.pyplot as plt

from config import TrainingConfig
from DiT import create_model


def _effective_rank(G: torch.Tensor) -> float:
    """Frobenius-norm-based effective rank of a 2D matrix G."""
    S = torch.linalg.svdvals(G)
    if S.numel() == 0 or S[0] == 0:
        return 0.0
    erank = torch.sum(S**2) / (S[0]**2)
    return float(erank)


def _flatten_weight_to_2d(w: torch.Tensor) -> torch.Tensor:
    """Return a 2D view of weight tensor for rank/norm analysis."""
    if w.ndim > 2:
        w = w.flatten(1)
    if w.shape[0] < w.shape[1]:
        w = w.t()
    return w


def _effective_rank_energy(w: torch.Tensor, energy_threshold: float = 0.99) -> int:
    """Number of singular values needed to capture `energy_threshold` spectral energy."""
    with torch.no_grad():
        s = torch.linalg.svdvals(w)
        if s.numel() == 0:
            return 0
        cumulative = torch.cumsum(s, dim=0) / s.sum()
        return int((cumulative < energy_threshold).sum().item() + 1)


def _transform_checkpoint_display(checkpoint: str) -> str:
    """Transform checkpoint number for display: subtract 1000 and format as 4-digit string."""
    try:
        ckpt_num = int(checkpoint)
        transformed = ckpt_num - 1000
        return f"{transformed:04d}"
    except ValueError:
        return checkpoint  # Return original if conversion fails


def compute_weight_metrics(model: nn.Module) -> Tuple[float, float, float]:
    """
    Compute average effective rank (Frobenius-based),
    average effective rank (energy-based, 99%), and Frobenius norm
    across all 2D weight matrices. Returns raw values (not normalized).
    Returns (avg_erank_frobenius, avg_erank_energy, avg_frobenius_norm).
    """
    eranks_frobenius: List[float] = []
    eranks_energy: List[float] = []
    frobs: List[float] = []

    with torch.no_grad():
        for param in model.parameters():
            if param.data is None:
                continue
            w = param.data
            # Skip biases and 1D tensors
            if w.ndim < 2:
                continue
            w2d = _flatten_weight_to_2d(w)
            if w2d.numel() == 0 or w2d.ndim != 2:
                continue
            try:
                # Frobenius-based effective rank (raw)
                erank = _effective_rank(w2d)
                # Energy-based effective rank (99%) raw count
                erank_energy_val = _effective_rank_energy(w2d)

                frob = torch.linalg.norm(w2d).item()
                eranks_frobenius.append(float(erank))
                eranks_energy.append(float(erank_energy_val))
                frobs.append(frob)
            except RuntimeError:
                # SVD may fail for some degenerate tensors; skip safely
                continue

    if len(eranks_frobenius) == 0:
        return 0.0, 0.0, 0.0

    return (
        float(sum(eranks_frobenius) / len(eranks_frobenius)),
        float(sum(eranks_energy) / len(eranks_energy)),
        float(sum(frobs) / len(frobs)),
    )


def plot_weight_analysis(step_wise_data: Dict[int, Dict[str, Dict[str, float]]],
                         group_labels: List[str],
                         checkpoint_list: List[str],
                         output_dir: str = None) -> None:
    """
    step_wise_data[group_index][checkpoint] = {
        'effective_rank': float,            # raw Frobenius-based effective rank
        'effective_rank_energy': float,     # raw energy-based effective rank (count)
        'frobenius_norm': float
    }
    group_labels ordered to match boundaries groups.
    """
    num_groups = len(group_labels)
    
    # Transform checkpoint labels for display (subtract 1000, format as 4-digit)
    checkpoint_display_labels = [_transform_checkpoint_display(ckpt) for ckpt in checkpoint_list]

    # 1x2 layout: effective rank and frobenius norm trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Weight Matrix Analysis: Checkpoint Trends Across Groups', fontsize=16, fontweight='bold')

    # Plot 1: Per-checkpoint Effective Rank trends (x-axis = groups, lines = checkpoints)
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(checkpoint_list))))
    x_idx = np.arange(num_groups)
    for ci, ckpt in enumerate(checkpoint_list):
        series = [step_wise_data.get(gi, {}).get(ckpt, {}).get('effective_rank', np.nan) for gi in range(num_groups)]
        display_label = checkpoint_display_labels[ci]
        ax1.plot(x_idx, series, marker='o', label=display_label, color=colors[ci % len(colors)], linewidth=2, markersize=4)
    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Effective Rank (Frobenius)')
    ax1.set_title('Effective Rank by Checkpoint Across Groups')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(group_labels, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Per-checkpoint Frobenius Norm trends (x-axis = groups, lines = checkpoints)
    for ci, ckpt in enumerate(checkpoint_list):
        series = [step_wise_data.get(gi, {}).get(ckpt, {}).get('frobenius_norm', np.nan) for gi in range(num_groups)]
        display_label = checkpoint_display_labels[ci]
        ax2.plot(x_idx, series, marker='s', label=display_label, color=colors[ci % len(colors)], linewidth=2, markersize=4)
    ax2.set_xlabel('Groups')
    ax2.set_ylabel('Frobenius Norm')
    ax2.set_title('Frobenius Norm by Checkpoint Across Groups')
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(group_labels, rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_filename = output_path / "weight_matrix_analysis_plots.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_filename}")
    else:
        plt.show()

    plt.close()


def save_weight_analysis_data(output_dir: str,
                              boundaries: List[int],
                              group_dirs: List[str],
                              group_labels: List[str],
                              checkpoint_list: List[str],
                              step_wise_data: Dict[int, Dict[str, Dict[str, float]]],
                              filename: str = "weight_matrix_analysis_data.json") -> None:
    """Persist computed weight analysis to JSON for later replotting."""
    swd_serializable = {str(k): v for k, v in step_wise_data.items()}
    payload = {
        "meta": {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "boundaries": boundaries,
            "group_dirs": group_dirs,
            "group_labels": group_labels,
            "checkpoint_list": checkpoint_list,
            "output_dir": output_dir,
        },
        "step_wise_data": swd_serializable,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / filename
    with open(json_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved weight analysis JSON to: {json_path}")


def replot_weight_from_saved(json_path: str, output_dir: str = None) -> None:
    """Replot weight analysis from saved JSON data with transformed checkpoint labels."""
    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    group_labels = meta.get("group_labels", [])
    checkpoint_list = meta.get("checkpoint_list", [])

    swd = {int(k): v for k, v in data.get("step_wise_data", {}).items()}
    plot_dir = output_dir if output_dir is not None else meta.get("output_dir", None)
    plot_weight_analysis(swd, group_labels, checkpoint_list, plot_dir)


def main():
    config = TrainingConfig()
    boundaries = [0, 133, 372, 653, 881, 1000]

    # Provide your five group-specific model directories here (each contains model_XXXX.pt checkpoints)
    # Example placeholders below; replace with your actual directories.
    group_model_dirs: List[str] = [
        str(Path(__file__).parent.parent / "logs" / "DiT20250903_190529"),
        str(Path(__file__).parent.parent / "logs" / "DiT20250903_191216"),
        str(Path(__file__).parent.parent / "logs" / "DiT20250903_190040"),
        str(Path(__file__).parent.parent / "logs" / "DiT20250903_184628"),
        str(Path(__file__).parent.parent / "logs" / "DiT20250903_184433"),
    ]

    group_labels = [
        f"[{boundaries[i]},{boundaries[i+1]})" for i in range(len(boundaries) - 1)
    ]

    checkpoint_list = [
        "1099", "1199", "1299", "1399", "1499", "1599",
         "1699", "1799", "1899", "1999", "2099", "2199", "2299", "2399", "2499", "2599", "2699", "2799", "2899", "2999"
    ]
    # checkpoint_list = [
    #     "0899", "0999", "1099", "1199", "1299", "1399", "1499", "1599",
    #      "1699", "1799", "1899", "1999", "2099", "2199"
        
    # ]

    assert len(group_model_dirs) == 5, "Expected five model directories (one per group)."

    # Data structure: step_wise_data[group_index][checkpoint] = metrics
    step_wise_data: Dict[int, Dict[str, Dict[str, float]]] = {i: {} for i in range(len(group_model_dirs))}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for gi, model_dir in enumerate(group_model_dirs):
        print(f"\n=== Processing Group {gi} {group_labels[gi]} from {model_dir} ===")
        for ckpt in checkpoint_list:
            ckpt_path = Path(model_dir) / f"model_{ckpt}.pt"
            if not ckpt_path.exists():
                # silently skip missing checkpoints
                continue

            # Load model and state
            model = create_model(config)
            model.to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()

            # Compute metrics over weights
            avg_erank_normed, avg_erank_energy_normed, avg_frob = compute_weight_metrics(model)

            step_wise_data[gi][ckpt] = {
                'effective_rank': avg_erank_normed,
                'effective_rank_energy': avg_erank_energy_normed,
                'frobenius_norm': avg_frob,
            }

            print(
                f"Group {gi} Ckpt {ckpt}: ERank(F)={avg_erank_normed:.4f} "
                f"ERank(E)={avg_erank_energy_normed:.4f} Frobenius={avg_frob:.4f}"
            )

    # Plot and save
    out_dir = str(Path(__file__).parent / "heat_maps" / "test_maps")
    plot_weight_analysis(step_wise_data, group_labels, checkpoint_list, out_dir)
    save_weight_analysis_data(out_dir, boundaries, group_model_dirs, group_labels, checkpoint_list, step_wise_data)


if __name__ == "__main__":
    main()


