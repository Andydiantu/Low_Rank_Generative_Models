from diffusers.models import AutoencoderKL, VQModel
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 16


vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
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
dataloader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


final_dataset = []

for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    with torch.no_grad():
        batch = batch.to(device)
        latents = vae.encode(batch).latents

        final_dataset.append(latents)

final_dataset = torch.cat(final_dataset, dim=0)
print(final_dataset.shape)

torch.save(final_dataset, Path(Path(__file__).parent.parent, "data", "celebA_latents_128.pt"))

print(final_dataset.shape)