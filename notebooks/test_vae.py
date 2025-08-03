import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.models import AutoencoderKL
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import time
import heapq

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Stable Diffusion AutoencoderKL
print("Loading Stable Diffusion AutoencoderKL...")
vae = AutoencoderKL.from_pretrained("tpremoli/MLD-CelebA-128-80k", subfolder="vae")

vae = vae.to(device)
vae.eval()

# Load CelebA dataset
print("Loading CelebA dataset...")
dataset = load_dataset("nielsr/CelebA-faces", split="train")

# Define transforms for 64x64 images
transform = transforms.Compose([
    transforms.CenterCrop(178),     # from 178×218 → 178×178
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Custom dataset class
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return image

# Create dataset and dataloader
celeba_dataset = CelebADataset(dataset, transform=transform)
dataloader = DataLoader(celeba_dataset, batch_size=128, shuffle=True, num_workers=2)

print(f"Dataset loaded with {len(celeba_dataset)} images")
print(f"Batch size: 16")

# --- BENCHMARKING ---
def benchmark_encoding_time(vae, dataloader, num_batches=100, device="cuda"):
    """
    Benchmarks the VAE encoding time for a given number of batches.
    """
    print("\n--- Benchmarking VAE Encoding Time ---")
    vae.eval()
    times = []
    
    # Warm-up to handle initial CUDA setup costs
    print("Running warm-up batches...")
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 5 warm-up batches
            break
        with torch.no_grad():
            _ = vae.encode(batch.to(device))
            
    # Benchmark loop
    print("Starting benchmark...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Benchmarking", total=num_batches)):
            if i >= num_batches:
                break
            
            images_to_encode = batch.to(device)
            
            # Synchronize before timing for accurate measurement of GPU execution
            torch.cuda.synchronize()
            start_time = time.time()
            
            _ = vae.encode(images_to_encode)
            
            # Synchronize again to ensure the encoding operation is complete
            torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
            
    avg_time = np.mean(times)
    std_dev = np.std(times)
    images_per_sec = dataloader.batch_size / avg_time
    
    print(f"\nBenchmark Results ({num_batches} batches):")
    print(f"  - Average time per batch: {avg_time * 1000:.2f} ms")
    print(f"  - Standard deviation:      {std_dev * 1000:.2f} ms")
    print(f"  - Throughput:              {images_per_sec:.2f} images/sec")
    print("------------------------------------\n")

# Run the benchmark before other calculations
benchmark_encoding_time(vae, dataloader, num_batches=100, device=device)
# --- END BENCHMARKING ---

# Function to calculate reconstruction errors
def calculate_reconstruction_errors(vae, dataloader, num_batches=1000):
    """
    Calculate various reconstruction error metrics
    """
    mse_errors = []
    mae_errors = []
    lpips_errors = []  # We'll use a simple proxy for LPIPS
    ssim_scores = []
    
    vae.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Computing reconstruction errors")):
            if i >= num_batches:
                break
                
            # Move batch to device
            original_images = batch.to(device)
            
            # Encode and decode with AutoencoderKL
            encoded_output = vae.encode(original_images)
            latents = encoded_output.latent_dist.sample()
            reconstructed_images = vae.decode(latents).sample
            
            # Calculate MSE (Mean Squared Error)
            mse = F.mse_loss(reconstructed_images, original_images, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)  # Per-image MSE
            mse_errors.extend(mse.cpu().numpy())
            
            # Calculate MAE (Mean Absolute Error)
            mae = F.l1_loss(reconstructed_images, original_images, reduction='none')
            mae = mae.view(mae.size(0), -1).mean(dim=1)  # Per-image MAE
            mae_errors.extend(mae.cpu().numpy())
            
            # Simple perceptual loss proxy (high-frequency content difference)
            original_gray = torch.mean(original_images, dim=1, keepdim=True)
            reconstructed_gray = torch.mean(reconstructed_images, dim=1, keepdim=True)
            
            # Calculate gradient differences as a proxy for perceptual differences
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
            
            orig_grad_x = F.conv2d(original_gray, sobel_x, padding=1)
            orig_grad_y = F.conv2d(original_gray, sobel_y, padding=1)
            recon_grad_x = F.conv2d(reconstructed_gray, sobel_x, padding=1)
            recon_grad_y = F.conv2d(reconstructed_gray, sobel_y, padding=1)
            
            grad_diff = F.mse_loss(orig_grad_x, recon_grad_x, reduction='none') + \
                       F.mse_loss(orig_grad_y, recon_grad_y, reduction='none')
            grad_diff = grad_diff.view(grad_diff.size(0), -1).mean(dim=1)
            lpips_errors.extend(grad_diff.cpu().numpy())
    
    return {
        'mse': np.array(mse_errors),
        'mae': np.array(mae_errors),
        'perceptual_proxy': np.array(lpips_errors)
    }

# Calculate reconstruction errors
print("Calculating reconstruction errors...")
errors = calculate_reconstruction_errors(vae, dataloader, num_batches=1000)

# Print statistics
print("\n=== Reconstruction Error Statistics ===")
print(f"MSE - Mean: {errors['mse'].mean():.6f}, Std: {errors['mse'].std():.6f}")
print(f"MAE - Mean: {errors['mae'].mean():.6f}, Std: {errors['mae'].std():.6f}")
print(f"Perceptual Proxy - Mean: {errors['perceptual_proxy'].mean():.6f}, Std: {errors['perceptual_proxy'].std():.6f}")

# Convert MSE to PSNR (Peak Signal-to-Noise Ratio)
psnr_values = -10 * np.log10(errors['mse'] + 1e-8)  # Add small epsilon to avoid log(0)
print(f"PSNR - Mean: {psnr_values.mean():.2f} dB, Std: {psnr_values.std():.2f} dB")


# Visualize reconstruction errors with histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MSE histogram
axes[0, 0].hist(errors['mse'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title('MSE Distribution')
axes[0, 0].set_xlabel('MSE')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(errors['mse'].mean(), color='red', linestyle='--', label=f'Mean: {errors["mse"].mean():.6f}')
axes[0, 0].legend()

# MAE histogram
axes[0, 1].hist(errors['mae'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title('MAE Distribution')
axes[0, 1].set_xlabel('MAE')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(errors['mae'].mean(), color='red', linestyle='--', label=f'Mean: {errors["mae"].mean():.6f}')
axes[0, 1].legend()

# PSNR histogram
axes[1, 0].hist(psnr_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_title('PSNR Distribution')
axes[1, 0].set_xlabel('PSNR (dB)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(psnr_values.mean(), color='red', linestyle='--', label=f'Mean: {psnr_values.mean():.2f} dB')
axes[1, 0].legend()

# Perceptual proxy histogram
axes[1, 1].hist(errors['perceptual_proxy'], bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].set_title('Perceptual Proxy Distribution')
axes[1, 1].set_xlabel('Perceptual Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(errors['perceptual_proxy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {errors["perceptual_proxy"].mean():.6f}')
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# Visualize sample reconstructions
def visualize_reconstructions(vae, dataloader, num_samples=8):
    """
    Visualize original vs reconstructed images
    """
    vae.eval()
    with torch.no_grad():
        # Get a batch of images
        batch = next(iter(dataloader))
        original_images = batch[:num_samples].to(device)
        
        # Encode and decode with AutoencoderKL
        encoded_output = vae.encode(original_images)
        latents = encoded_output.latent_dist.sample()
        reconstructed_images = vae.decode(latents).sample
        
        # Move to CPU for visualization
        original_images = original_images.cpu()
        reconstructed_images = reconstructed_images.cpu()
        
        # Create visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))
        
        for i in range(num_samples):
            # Convert from [-1, 1] to [0, 1] for display
            orig_img = (original_images[i] + 1) / 2
            recon_img = (reconstructed_images[i] + 1) / 2
            diff_img = torch.abs(orig_img - recon_img)
            
            # Original images
            axes[0, i].imshow(orig_img.permute(1, 2, 0).clamp(0, 1))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed images
            axes[1, i].imshow(recon_img.permute(1, 2, 0).clamp(0, 1))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
            
            # Difference images
            diff_display = diff_img.permute(1, 2, 0)
            # Enhance differences for better visibility
            diff_display = diff_display / (diff_display.max() + 1e-8)
            axes[2, i].imshow(diff_display.clamp(0, 1))
            
            # Calculate individual MSE for this image
            mse_individual = F.mse_loss(original_images[i], reconstructed_images[i]).item()
            axes[2, i].set_title(f'Diff (MSE: {mse_individual:.4f})')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstructions.png')
        plt.show()
        
        return original_images, reconstructed_images

# Visualize reconstructions
original_imgs, reconstructed_imgs = visualize_reconstructions(vae, dataloader, num_samples=8)


# Additional analysis: Error vs Image complexity
def analyze_error_vs_complexity(vae, dataloader, num_batches=20):
    """
    Analyze how reconstruction error correlates with image complexity
    """
    complexities = []
    mse_errors = []
    
    vae.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Analyzing complexity vs error")):
            if i >= num_batches:
                break
                
            original_images = batch.to(device)
            
            # Calculate image complexity as variance in gradients
            gray_images = torch.mean(original_images, dim=1, keepdim=True)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
            
            grad_x = F.conv2d(gray_images, sobel_x, padding=1)
            grad_y = F.conv2d(gray_images, sobel_y, padding=1)
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            
            # Use gradient variance as complexity measure
            complexity = gradient_magnitude.view(gradient_magnitude.size(0), -1).var(dim=1)
            complexities.extend(complexity.cpu().numpy())
            
            # Calculate reconstruction error with AutoencoderKL
            encoded_output = vae.encode(original_images)
            latents = encoded_output.latent_dist.sample()
            reconstructed_images = vae.decode(latents).sample
            
            mse = F.mse_loss(reconstructed_images, original_images, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            mse_errors.extend(mse.cpu().numpy())
    
    return np.array(complexities), np.array(mse_errors)

# Analyze complexity vs error
complexities, complexity_errors = analyze_error_vs_complexity(vae, dataloader, num_batches=50)

# Plot complexity vs error
plt.figure(figsize=(10, 6))
plt.scatter(complexities, complexity_errors, alpha=0.6, s=20)
plt.xlabel('Image Complexity (Gradient Variance)')
plt.ylabel('Reconstruction MSE')
plt.title('Reconstruction Error vs Image Complexity')

# Add trend line
z = np.polyfit(complexities, complexity_errors, 1)
p = np.poly1d(z)
plt.plot(complexities, p(complexities), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2e}x + {z[1]:.2e}')

# Calculate correlation
correlation = np.corrcoef(complexities, complexity_errors)[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Correlation between image complexity and reconstruction error: {correlation:.4f}")

# -----------------------------------------------------------------------------
# Visualise the worst-reconstructed samples (highest MSE)
# -----------------------------------------------------------------------------

def visualize_worst_reconstructions(vae, dataloader, top_k=8, max_batches=200):
    """Find the *top_k* images with the largest reconstruction MSE and plot them.

    Args:
        vae (AutoencoderKL): trained/loaded VAE.
        dataloader (DataLoader): dataloader that yields the *original* images.
        top_k (int): how many worst samples to show.
        max_batches (int): safety limit so we do not iterate over the whole
            dataset if it is very large. Set to None to disable.
    """
    # Each entry: (error, orig_cpu_tensor, recon_cpu_tensor)
    worst_samples = []  # min-heap of fixed size *top_k*

    vae.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Finding worst reconstructions")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            original_images = batch.to(device)
            latents = vae.encode(original_images).latent_dist.sample()
            reconstructed_images = vae.decode(latents).sample

            # Per-image MSE
            mse = F.mse_loss(reconstructed_images, original_images, reduction="none")
            mse = mse.view(mse.size(0), -1).mean(dim=1)

            for img_idx in range(original_images.size(0)):
                err_val = mse[img_idx].item()
                sample_tuple = (err_val, original_images[img_idx].cpu(), reconstructed_images[img_idx].cpu())

                if len(worst_samples) < top_k:
                    heapq.heappush(worst_samples, sample_tuple)
                else:
                    # heap top is smallest error; replace if current bigger
                    if err_val > worst_samples[0][0]:
                        heapq.heapreplace(worst_samples, sample_tuple)

    # Sort descending by error for nicer visualisation
    worst_samples = sorted(worst_samples, key=lambda x: x[0], reverse=True)

    # Plot
    num_samples = len(worst_samples)
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 6))
    for i, (err_val, orig_img, recon_img) in enumerate(worst_samples):
        # Convert to [0,1]
        orig_disp = (orig_img + 1) / 2
        recon_disp = (recon_img + 1) / 2
        diff_disp = torch.abs(orig_disp - recon_disp)
        diff_disp = diff_disp / (diff_disp.max() + 1e-8)

        axes[0, i].imshow(orig_disp.permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_disp.permute(1, 2, 0).clamp(0, 1))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

        axes[2, i].imshow(diff_disp.permute(1, 2, 0).clamp(0, 1))
        axes[2, i].set_title(f"Diff\nMSE: {err_val:.4f}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("worst_reconstructions.png")
    plt.show()


# -----------------------------------------------------------------------------
# Run the above helper
# -----------------------------------------------------------------------------

visualize_worst_reconstructions(vae, dataloader, top_k=8, max_batches=300)
