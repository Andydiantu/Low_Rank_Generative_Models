import argparse
from pathlib import Path
from typing import Dict, Literal

import torch
import matplotlib.pyplot as plt


def load_loss_dict(path: Path) -> Dict[int, float]:
    """
    Load a loss_dict saved via torch.save from eval_timestamp_loss.py.
    Ensures keys are ints and values are floats.
    """
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}, got {type(data)}")

    result: Dict[int, float] = {}
    for key, value in data.items():
        try:
            timestep = int(key)
        except Exception as exc:
            raise ValueError(f"Timestep key {key!r} in {path} is not convertible to int") from exc
        try:
            result[timestep] = float(value)
        except Exception as exc:
            raise ValueError(
                f"Loss value for timestep {key!r} in {path} is not convertible to float"
            ) from exc
    return result


def compute_normalized_differences(
    loss_a: Dict[int, float],
    loss_b: Dict[int, float],
    normalize: Literal["mean", "first", "second", "none"] = "mean",
    eps: float = 1e-12,
) -> Dict[int, float]:
    """
    Compute differences per timestep for timesteps present in both dicts.

    When `normalize` is one of {"mean", "first", "second"}:
      diff_t = (loss_a[t] - loss_b[t]) / denom
      where denom is chosen by `normalize`:
        - "mean": (abs(a) + abs(b)) / 2  [symmetric percent difference]
        - "first": abs(a)
        - "second": abs(b)

    When `normalize` == "none":
      diff_t = (loss_a[t] - loss_b[t])  [raw difference]
    """
    common_timesteps = sorted(set(loss_a.keys()) & set(loss_b.keys()))
    diffs: Dict[int, float] = {}
    for t in common_timesteps:
        a = float(loss_a[t])
        b = float(loss_b[t])
        if normalize == "none":
            diffs[t] = a - b
            continue
        if normalize == "mean":
            denom = abs((abs(a) - abs(b))) / 2.0
        elif normalize == "first":
            denom = abs(a)
        elif normalize == "second":
            denom = abs(b)
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}")
        denom = max(denom, eps)
        diffs[t] = (a - b) / denom
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two loss_dict .pt files and report timesteps with the largest "
            "normalized differences."
        )
    )
    parser.add_argument(
        "loss_a",
        type=Path,
        help="Path to first loss_dict .pt file (e.g., Full_rank_timestep_loss_*.pt)",
    )
    parser.add_argument(
        "loss_b",
        type=Path,
        help="Path to second loss_dict .pt file (e.g., Full_rank_timestep_loss_*.pt)",
    )
    parser.add_argument(
        "--normalize",
        choices=["mean", "first", "second", "none"],
        default="mean",
        help="Difference mode: mean/first/second normalize; 'none' = raw (a - b)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of timesteps to display with largest |normalized diff|",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all timesteps instead of only top-k",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save a CSV with columns: timestep,loss_a,loss_b,norm_diff,abs_norm_diff",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a line plot of normalized difference vs timestep",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path to save the plot (e.g., .png or .pdf)",
    )

    args = parser.parse_args()

    loss_a = load_loss_dict(args.loss_a)
    loss_b = load_loss_dict(args.loss_b)

    common = sorted(set(loss_a.keys()) & set(loss_b.keys()))
    missing_in_b = sorted(set(loss_a.keys()) - set(loss_b.keys()))
    missing_in_a = sorted(set(loss_b.keys()) - set(loss_a.keys()))

    if not common:
        raise SystemExit("No overlapping timesteps between the two loss_dict files.")

    diffs = compute_normalized_differences(loss_a, loss_b, normalize=args.normalize)

    # Prepare rows for display and optional CSV
    rows = [
        (
            t,
            float(loss_a[t]),
            float(loss_b[t]),
            float(diffs[t]),
            abs(float(diffs[t])),
        )
        for t in common
    ]
    # Sort by absolute normalized difference descending
    rows.sort(key=lambda r: r[4], reverse=True)

    print(
        f"Compared {len(common)} common timesteps | normalize={args.normalize} | "
        f"missing_in_a={len(missing_in_a)} missing_in_b={len(missing_in_b)}"
    )
    if missing_in_a:
        print(f"Timesteps present only in second file (missing in first): {missing_in_a[:10]}{'...' if len(missing_in_a) > 10 else ''}")
    if missing_in_b:
        print(f"Timesteps present only in first file (missing in second): {missing_in_b[:10]}{'...' if len(missing_in_b) > 10 else ''}")

    if args.normalize == "none":
        header = ("timestep", "loss_a", "loss_b", "diff", "abs_diff")
    else:
        header = ("timestep", "loss_a", "loss_b", "norm_diff", "abs_norm_diff")
    to_show = rows if args.show_all else rows[: args.top_k]
    if args.normalize == "none":
        print("\nTop timesteps by |difference|:")
    else:
        print("\nTop timesteps by |normalized difference|:")
    print(" ".join(h.rjust(14) for h in header))
    for t, la, lb, nd, andiff in to_show:
        print(
            f"{t:14d} {la:14.6f} {lb:14.6f} {nd:14.6f} {andiff:14.6f}"
        )

    if args.save_csv is not None:
        try:
            import csv
        except Exception:  # pragma: no cover - csv should always be available in stdlib
            raise
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nSaved CSV to: {args.save_csv}")

    # Plot difference vs timestep (in ascending timestep order)
    if args.plot or args.save_plot is not None:
        timesteps_sorted = sorted(common)
        y_vals = [diffs[t] for t in timesteps_sorted]
        if args.normalize == "none":
            title_label = "Loss Difference per Timestep"
            ylabel = "Loss Difference"
        else:
            title_label = "Normalized Loss Difference per Timestep"
            ylabel = "Normalized Difference"

        # Publication-ready styling
        with plt.rc_context({
            "font.family": ["Times New Roman", "Liberation Serif", "serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
        }):
            # Create figure with publication aspect ratio
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use academic color palette - professional blue
            main_color = "#1f77b4"  # Professional blue
            grid_color = "#E5E5E5"  # Light gray for grid
            
            # Main plot line with better styling
            ax.plot(timesteps_sorted, y_vals, color=main_color, linewidth=2.5, alpha=0.9)
            
            # Zero reference line
            ax.axhline(0.0, color="#2F2F2F", linewidth=1.2, linestyle="--", alpha=0.7)
            
            # Title and labels with better formatting
            ax.set_title(title_label, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Timestep", fontsize=14, fontweight='medium')
            ax.set_ylabel(ylabel, fontsize=14, fontweight='medium')
            
            # Improved grid
            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color=grid_color)
            ax.set_axisbelow(True)  # Put grid behind data
            
            # Clean academic-style spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color("#2F2F2F")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # Professional tick styling
            ax.tick_params(
                axis="both", 
                which="major", 
                direction="out", 
                length=6, 
                width=1.2,
                colors="#2F2F2F",
                labelcolor="#2F2F2F"
            )
            ax.tick_params(axis="both", which="minor", length=3, width=0.8)

            # Enhanced model information box
            model_info = (
                "Full Rank Model\n"
                "• 39.80M parameters\n"
                "• 5.46 GFLOPs\n\n"
                "Low Rank Model (76% compression)\n"
                "• 9.46M parameters\n"
                "• 0.82 GFLOPs"
            )
            
            # Position box in upper right with larger font
            ax.text(
                0.98,
                0.98,
                model_info,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="white",
                    alpha=0.98,
                    edgecolor=main_color,
                    linewidth=1.5,
                ),
            )
            
            # Improve layout with better margins
            fig.tight_layout(pad=2.0)

        if args.save_plot is not None:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_plot, dpi=300, bbox_inches="tight")
            print(f"Saved plot to: {args.save_plot}")

        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()


