from diffusers.models import AutoencoderKL
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 16


vae = AutoencoderKL.from_pretrained("tpremoli/MLD-CelebA-128-80k", subfolder="vae")
vae = vae.to(device)
vae.eval()


latents = torch.load(Path(Path(__file__).parent.parent, "data", "celebA_latents.pt"))
# Calculate mean and std
# Reshape to combine all spatial dimensions and channels
# From [202599, 4, 16, 16] to [202599, 1024] (4*16*16 = 1024)
flattened_latents = latents.reshape(latents.shape[0], -1)

# Calculate mean and std across all pixels
mean = flattened_latents.mean(dim=0).mean(dim=0)  # Average across all samples
std = flattened_latents.std(dim=0).mean(dim=0)    # Std across all samples


# Print the results
print(f"Mean shape: {mean.shape}")
print(f"Mean: {mean}")
print(f"Std shape: {std.shape}")
print(f"Std: {std}")

# If you want to see the statistics per channel
if len(mean.shape) > 1:  # If the latents have multiple channels
    for i in range(mean.shape[0]):
        print(f"\nChannel {i}:")
        print(f"Mean: {mean[i].item():.4f}")
        print(f"Std: {std[i].item():.4f}")
print(latents.shape)




dataset = TensorDataset(latents)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for batch in dataloader:
    with torch.no_grad():
        batch = batch[0].to(device)  # Extract tensor from tuple
        images = vae.decode(batch).sample
        print(f"Decoded images shape: {images.shape}")
        
        # Move to CPU and convert to numpy for visualization
        images_cpu = images.cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        images_normalized = (images_cpu + 1.0) / 2.0
        images_normalized = torch.clamp(images_normalized, 0, 1)
        
        # Create a grid visualization
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Decoded Images from Latents', fontsize=16)
        
        for i in range(min(16, images_normalized.shape[0])):
            row = i // 4
            col = i % 4
            
            # Convert from CHW to HWC format for matplotlib
            img = images_normalized[i].permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Image {i+1}')
            axes[row, col].axis('off')
        
        # Hide any unused subplots
        for i in range(images_normalized.shape[0], 16):
            row = i // 4
            col = i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('decoded_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print some statistics
        print(f"Image value range: [{images_normalized.min():.3f}, {images_normalized.max():.3f}]")
        print(f"Latent value range: [{batch.min():.3f}, {batch.max():.3f}]")
        
        break