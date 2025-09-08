import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_latent_statistics():
    """
    Check if your saved latents have reasonable statistics for diffusion training.
    """
    
    # Load saved latents
    latents_path = Path(Path(__file__).parent.parent, "data", "imagenet-1k-128x128_latents.pt")
    
    if not latents_path.exists():
        print(f"ERROR: Latents file not found at {latents_path}")
        return
    
    print(f"Loading latents from {latents_path}")
    saved = torch.load(latents_path, map_location='cpu')
    
    if isinstance(saved, dict):
        latents = saved.get("latents", None)
        labels = saved.get("labels", None)
    else:
        latents = saved
        labels = None
    
    print(f"Latents shape: {latents.shape}")
    print(f"Latents dtype: {latents.dtype}")
    
    # Sample a subset for analysis (to avoid memory issues)
    sample_size = min(10000, latents.shape[0])
    indices = torch.randperm(latents.shape[0])[:sample_size]
    sample_latents = latents[indices]
    
    print(f"\nAnalyzing {sample_size} samples...")
    
    # Global statistics
    mean = sample_latents.mean()
    std = sample_latents.std()
    min_val = sample_latents.min()
    max_val = sample_latents.max()
    
    print(f"\nGlobal Statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Min:  {min_val:.6f}")
    print(f"  Max:  {max_val:.6f}")
    
    # Per-channel statistics
    print(f"\nPer-Channel Statistics:")
    for c in range(sample_latents.shape[1]):
        channel_data = sample_latents[:, c]
        print(f"  Channel {c}: mean={channel_data.mean():.6f}, std={channel_data.std():.6f}, "
              f"min={channel_data.min():.6f}, max={channel_data.max():.6f}")
    
    # Check for common issues
    print(f"\nDiagnostic Checks:")
    
    # 1. Check if latents are roughly zero-centered
    if abs(mean.item()) > 0.2:
        print(f"⚠️  WARNING: Latents are not well-centered (mean={mean:.6f})")
    else:
        print(f"✅ Latents are reasonably centered (mean={mean:.6f})")
    
    # 2. Check if variance is reasonable for diffusion
    if std.item() < 0.3:
        print(f"⚠️  WARNING: Low variance (std={std:.6f}) - latents might be under-scaled")
    elif std.item() > 3.0:
        print(f"⚠️  WARNING: High variance (std={std:.6f}) - latents might be over-scaled")
    else:
        print(f"✅ Reasonable variance (std={std:.6f})")
    
    # 3. Check for extreme outliers
    q99 = torch.quantile(sample_latents, 0.99)
    q01 = torch.quantile(sample_latents, 0.01)
    if q99 > 5.0 or q01 < -5.0:
        print(f"⚠️  WARNING: Extreme outliers detected (1%ile={q01:.3f}, 99%ile={q99:.3f})")
    else:
        print(f"✅ No extreme outliers (1%ile={q01:.3f}, 99%ile={q99:.3f})")
    
    # 4. Check for NaN or Inf values
    if torch.isnan(sample_latents).any():
        print(f"❌ ERROR: NaN values detected!")
    elif torch.isinf(sample_latents).any():
        print(f"❌ ERROR: Inf values detected!")
    else:
        print(f"✅ No NaN or Inf values")
    
    # 5. Check channel balance
    channel_stds = [sample_latents[:, c].std().item() for c in range(sample_latents.shape[1])]
    max_std_ratio = max(channel_stds) / min(channel_stds)
    if max_std_ratio > 3.0:
        print(f"⚠️  WARNING: Unbalanced channels (max/min std ratio: {max_std_ratio:.2f})")
    else:
        print(f"✅ Balanced channels (max/min std ratio: {max_std_ratio:.2f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Overall histogram
    flat_latents = sample_latents.flatten().numpy()
    axes[0, 0].hist(flat_latents, bins=100, density=True, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mean.item(), color='red', linestyle='--', label=f'Mean: {mean:.3f}')
    axes[0, 0].axvline(mean.item() + std.item(), color='orange', linestyle='--', label=f'+1σ: {mean.item() + std.item():.3f}')
    axes[0, 0].axvline(mean.item() - std.item(), color='orange', linestyle='--', label=f'-1σ: {mean.item() - std.item():.3f}')
    axes[0, 0].set_xlabel('Latent Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overall Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Per-channel histograms
    colors = ['red', 'green', 'blue', 'orange']
    for c in range(sample_latents.shape[1]):
        channel_data = sample_latents[:, c].flatten().numpy()
        axes[0, 1].hist(channel_data, bins=50, density=True, alpha=0.5, 
                       label=f'Channel {c}', color=colors[c % len(colors)])
    axes[0, 1].set_xlabel('Latent Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Per-Channel Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot against normal distribution
    from scipy import stats
    stats.probplot(flat_latents[:10000], dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot vs Normal Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Channel statistics
    channel_means = [sample_latents[:, c].mean().item() for c in range(sample_latents.shape[1])]
    channel_stds = [sample_latents[:, c].std().item() for c in range(sample_latents.shape[1])]
    
    x = range(sample_latents.shape[1])
    axes[1, 0].bar([i-0.2 for i in x], channel_means, 0.4, label='Mean', alpha=0.7)
    axes[1, 0].bar([i+0.2 for i in x], channel_stds, 0.4, label='Std', alpha=0.7)
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Channel Statistics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Spatial mean across all samples
    spatial_mean = sample_latents.mean(dim=0).mean(dim=0).numpy()
    im1 = axes[1, 1].imshow(spatial_mean, cmap='RdBu_r', interpolation='nearest')
    axes[1, 1].set_title('Spatial Mean Pattern')
    plt.colorbar(im1, ax=axes[1, 1])
    
    # Plot 6: Spatial std across all samples  
    spatial_std = sample_latents.std(dim=0).mean(dim=0).numpy()
    im2 = axes[1, 2].imshow(spatial_std, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title('Spatial Std Pattern')
    plt.colorbar(im2, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(__file__).parent / "latent_statistics_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if abs(mean.item()) < 0.1 and 0.5 < std.item() < 2.0:
        print("✅ Your latents look good for diffusion training!")
    else:
        print("⚠️  Consider these potential improvements:")
        if abs(mean.item()) > 0.1:
            print(f"   - Latents are not well-centered (mean={mean:.6f})")
        if std.item() < 0.5:
            print(f"   - Low variance may make learning difficult (std={std:.6f})")
        if std.item() > 2.0:
            print(f"   - High variance may cause training instability (std={std:.6f})")
    
    print(f"\nWith clip_sample=False fix, your model should converge to more natural latent regions.")
    
    return {
        'mean': mean.item(),
        'std': std.item(),
        'min': min_val.item(),
        'max': max_val.item(),
        'channel_stats': {
            'means': channel_means,
            'stds': channel_stds,
        }
    }

if __name__ == "__main__":
    results = check_latent_statistics()
