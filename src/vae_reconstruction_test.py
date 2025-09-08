import torch
import numpy as np
from pathlib import Path
from vae import SD_VAE
import matplotlib.pyplot as plt

def test_vae_reconstruction():
    """
    Test VAE reconstruction consistency by:
    1. Loading saved latents
    2. Decoding them to images
    3. Re-encoding images back to latents
    4. Computing MSE between original and reconstructed latents
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load VAE (same as used for encoding)
    vae = SD_VAE(device=device)
    
    # Load saved latents
    latents_path = Path(Path(__file__).parent.parent, "data", "imagenet-1k-128x128_latents.pt")
    
    if not latents_path.exists():
        print(f"ERROR: Latents file not found at {latents_path}")
        print("Please run encode_imagenet_128.py first to generate the latents file.")
        return
    
    print(f"Loading latents from {latents_path}")
    saved = torch.load(latents_path, map_location=device)
    
    # Support both dict format {"latents", "labels"} and raw tensor format
    if isinstance(saved, dict):
        original_latents = saved.get("latents", None)
        if original_latents is None:
            raise ValueError("Saved file is a dict but does not contain 'latents' key")
    else:
        original_latents = saved
    
    print(f"Original latents shape: {original_latents.shape}")
    print(f"Original latents dtype: {original_latents.dtype}")
    print(f"Original latents mean: {original_latents.mean():.6f}")
    print(f"Original latents std: {original_latents.std():.6f}")
    print(f"Original latents min: {original_latents.min():.6f}")
    print(f"Original latents max: {original_latents.max():.6f}")
    
    # Test with a batch of latents (use first 16 for speed)
    batch_size = min(16, original_latents.shape[0])
    original_batch = original_latents[:batch_size].to(device)
    
    print(f"\nTesting reconstruction with batch size: {batch_size}")
    
    with torch.no_grad():
        # Step 1: Decode latents to images
        print("Step 1: Decoding latents to images...")
        decoded_images = vae.decode(original_batch)
        
        print(f"Decoded images shape: {decoded_images.shape}")
        print(f"Decoded images dtype: {decoded_images.dtype}")
        print(f"Decoded images mean: {decoded_images.mean():.6f}")
        print(f"Decoded images std: {decoded_images.std():.6f}")
        print(f"Decoded images min: {decoded_images.min():.6f}")
        print(f"Decoded images max: {decoded_images.max():.6f}")
        
        # Step 2: Re-encode images back to latents
        print("Step 2: Re-encoding images to latents...")
        reconstructed_latents = vae.encode(decoded_images)
        
        print(f"Reconstructed latents shape: {reconstructed_latents.shape}")
        print(f"Reconstructed latents dtype: {reconstructed_latents.dtype}")
        print(f"Reconstructed latents mean: {reconstructed_latents.mean():.6f}")
        print(f"Reconstructed latents std: {reconstructed_latents.std():.6f}")
        print(f"Reconstructed latents min: {reconstructed_latents.min():.6f}")
        print(f"Reconstructed latents max: {reconstructed_latents.max():.6f}")
    
    # Step 3: Compute reconstruction metrics
    print("\nStep 3: Computing reconstruction metrics...")
    
    # MSE between original and reconstructed latents
    mse = torch.nn.functional.mse_loss(original_batch, reconstructed_latents)
    print(f"MSE between original and reconstructed latents: {mse.item():.8f}")
    
    # Mean Absolute Error
    mae = torch.nn.functional.l1_loss(original_batch, reconstructed_latents)
    print(f"MAE between original and reconstructed latents: {mae.item():.8f}")
    
    # Relative error (percentage)
    relative_error = (mse / torch.var(original_batch)).item() * 100
    print(f"Relative MSE (as % of original variance): {relative_error:.4f}%")
    
    # Per-channel statistics
    print("\nPer-channel reconstruction errors:")
    for c in range(original_batch.shape[1]):
        channel_mse = torch.nn.functional.mse_loss(
            original_batch[:, c], reconstructed_latents[:, c]
        )
        print(f"  Channel {c}: MSE = {channel_mse.item():.8f}")
    
    # Correlation coefficient
    original_flat = original_batch.flatten()
    reconstructed_flat = reconstructed_latents.flatten()
    correlation = torch.corrcoef(torch.stack([original_flat, reconstructed_flat]))[0, 1]
    print(f"Correlation coefficient: {correlation.item():.6f}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Original vs Reconstructed scatter plot
    axes[0, 0].scatter(original_flat.cpu().numpy(), reconstructed_flat.cpu().numpy(), 
                      alpha=0.1, s=0.5)
    axes[0, 0].plot([original_flat.min().item(), original_flat.max().item()], 
                    [original_flat.min().item(), original_flat.max().item()], 'r--')
    axes[0, 0].set_xlabel('Original Latents')
    axes[0, 0].set_ylabel('Reconstructed Latents')
    axes[0, 0].set_title('Original vs Reconstructed Scatter')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of differences
    diff = (original_batch - reconstructed_latents).flatten().cpu().numpy()
    axes[0, 1].hist(diff, bins=50, alpha=0.7, density=True)
    axes[0, 1].set_xlabel('Difference (Original - Reconstructed)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Reconstruction Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Per-sample MSE
    per_sample_mse = torch.nn.functional.mse_loss(
        original_batch, reconstructed_latents, reduction='none'
    ).mean(dim=[1, 2, 3]).cpu().numpy()
    axes[0, 2].bar(range(len(per_sample_mse)), per_sample_mse)
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('MSE')
    axes[0, 2].set_title('Per-Sample Reconstruction MSE')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Original latent statistics
    axes[1, 0].hist(original_flat.cpu().numpy(), bins=50, alpha=0.7, density=True, label='Original')
    axes[1, 0].hist(reconstructed_flat.cpu().numpy(), bins=50, alpha=0.7, density=True, label='Reconstructed')
    axes[1, 0].set_xlabel('Latent Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Channel-wise MSE
    channel_mses = []
    for c in range(original_batch.shape[1]):
        channel_mse = torch.nn.functional.mse_loss(
            original_batch[:, c], reconstructed_latents[:, c]
        ).item()
        channel_mses.append(channel_mse)
    
    axes[1, 1].bar(range(len(channel_mses)), channel_mses)
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Per-Channel Reconstruction MSE')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Relative error per spatial location
    spatial_mse = torch.nn.functional.mse_loss(
        original_batch, reconstructed_latents, reduction='none'
    ).mean(dim=[0, 1]).cpu().numpy()
    
    im = axes[1, 2].imshow(spatial_mse, cmap='hot', interpolation='nearest')
    axes[1, 2].set_title('Spatial Reconstruction Error')
    axes[1, 2].set_xlabel('Width')
    axes[1, 2].set_ylabel('Height')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(__file__).parent / "vae_reconstruction_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()
    
    # Print interpretation
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    
    if mse.item() < 1e-3:
        print("✅ EXCELLENT: Very low reconstruction error. VAE is consistent.")
    elif mse.item() < 1e-2:
        print("✅ GOOD: Low reconstruction error. VAE is mostly consistent.")
    elif mse.item() < 1e-1:
        print("⚠️  MODERATE: Some reconstruction error. Check for VAE version mismatch.")
    else:
        print("❌ HIGH: High reconstruction error. Likely VAE mismatch or scaling issue.")
    
    if relative_error < 1.0:
        print("✅ Reconstruction error is much smaller than data variance.")
    elif relative_error < 10.0:
        print("⚠️  Reconstruction error is noticeable compared to data variance.")
    else:
        print("❌ Reconstruction error is significant compared to data variance.")
    
    if correlation.item() > 0.99:
        print("✅ Very high correlation between original and reconstructed latents.")
    elif correlation.item() > 0.95:
        print("⚠️  Good correlation, but some systematic differences.")
    else:
        print("❌ Low correlation suggests major reconstruction issues.")
    
    # Check for specific issues
    print("\nPOSSIBLE ISSUES TO CHECK:")
    if abs(original_batch.mean().item() - reconstructed_latents.mean().item()) > 0.1:
        print("- Mean shift detected: possible scaling factor mismatch")
    
    if abs(original_batch.std().item() - reconstructed_latents.std().item()) > 0.1:
        print("- Standard deviation change: possible variance scaling issue")
    
    print(f"- Check that the VAE used for encoding ({vae.vae.config._name_or_path if hasattr(vae.vae.config, '_name_or_path') else 'unknown'}) matches training")
    print(f"- VAE scaling factor: {vae.vae.config.scaling_factor}")
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'relative_error': relative_error,
        'correlation': correlation.item(),
        'original_stats': {
            'mean': original_batch.mean().item(),
            'std': original_batch.std().item(),
            'min': original_batch.min().item(),
            'max': original_batch.max().item(),
        },
        'reconstructed_stats': {
            'mean': reconstructed_latents.mean().item(),
            'std': reconstructed_latents.std().item(),
            'min': reconstructed_latents.min().item(),
            'max': reconstructed_latents.max().item(),
        }
    }

if __name__ == "__main__":
    results = test_vae_reconstruction()

