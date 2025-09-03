from vae import SD_VAE
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128


# Load VAE consistent with trainer (facebook/DiT-XL-2-256 VAE via SD_VAE wrapper)
vae = SD_VAE(device="cuda" if torch.cuda.is_available() else "cpu")


# Load ImageNet-128 dataset
print("Loading ImageNet-128 dataset...")
dataset = load_dataset("benjamin-paine/imagenet-1k-128x128", split="train")

# Define transforms for 128x128 images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])


# Custom dataset class
class ImageNet128Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label'] if 'label' in item else None
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}


# Create dataset and dataloader
imagenet_dataset = ImageNet128Dataset(dataset, transform=transform)
dataloader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


latents_list = []
labels_list = []

for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    with torch.no_grad():
        images = batch["image"].to(device)
        labels = batch["label"]
        latents = vae.encode(images)
        latents_list.append(latents.cpu())
        labels_list.append(labels.cpu())


final_latents = torch.cat(latents_list, dim=0)
final_labels = torch.cat(labels_list, dim=0)
print(final_latents.shape, final_labels.shape)

# Ensure data directory exists at repo root
data_dir = Path(Path(__file__).parent.parent, "data")
data_dir.mkdir(parents=True, exist_ok=True)

torch.save({"latents": final_latents, "labels": final_labels}, data_dir / "imagenet-1k-128x128_latents.pt")

print(final_latents.shape, final_labels.shape)



# --------------------------------------------------------------
# Simple decode check: load saved latents, decode, and save image
# --------------------------------------------------------------
try:
    print("Decoding a sample from saved latents...")
    latents_path = Path(Path(__file__).parent.parent, "data", "imagenet-1k-128x128_latents.pt")
    saved = torch.load(latents_path, map_location=device)

    # Support both dict format {"latents", "labels"} and raw tensor format
    if isinstance(saved, dict):
        latents = saved.get("latents", None)
        if latents is None:
            raise ValueError("Saved file is a dict but does not contain 'latents' key")
    else:
        latents = saved

    # Select up to 16 latents for a small grid
    num_samples = min(16, latents.shape[0])
    z = latents[:num_samples].to(device)

    with torch.no_grad():
        # SD_VAE.decode handles scaling factor and returns images in [0,1]
        decoded = vae.decode(z).cpu()

    out_path = Path(__file__).parent / "imagenet-1k-128x128_decode_check.png"
    save_image(decoded, out_path, nrow=4)
    print(f"Saved decoded sample grid to {out_path}")
except Exception as e:
    print(f"Decode check failed: {e}")