import argparse
from diffusers.models import AutoencoderKL
import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="ImageNet split to encode")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--vae_repo", type=str, default="tpremoli/MLD-CelebA-128-80k")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = AutoencoderKL.from_pretrained(args.vae_repo, subfolder=args.vae_subfolder)
    vae = vae.to(device)
    vae.eval()

    print(f"Loading ImageNet-1k split='{args.split}'...")
    dataset = load_dataset("imagenet-1k", split=args.split)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    class ImageNetWithLabels(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            image = sample["image"]
            label = int(sample["label"]) if "label" in sample else -1
            if self.transform:
                image = self.transform(image)
            return image, label

    imagenet_ds = ImageNetWithLabels(dataset, transform=transform)
    dataloader = DataLoader(imagenet_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    latents_list = []
    labels_list = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            latents_list.append(z.cpu())
            labels_list.append(labels.cpu())

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    out_dir = Path(Path(__file__).parent.parent, "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    latents_path = out_dir / f"imagenet_{args.split}_latents.pt"
    labels_path = out_dir / f"imagenet_{args.split}_labels.pt"

    torch.save(latents, latents_path)
    torch.save(labels, labels_path)

    print(latents.shape, labels.shape)
    print(f"Saved latents to: {latents_path}")
    print(f"Saved labels  to: {labels_path}")


if __name__ == "__main__":
    main()


