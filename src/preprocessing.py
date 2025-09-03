from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Subset
from datasets import load_dataset, Dataset
from torchvision import transforms
from config import TrainingConfig
import numpy as np

def load_dataset_from_hf(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset {dataset_name} {split} split with {len(dataset)} images")
    return dataset

class LatentsTorchDataset(torch.utils.data.Dataset):
    """Memory-efficient wrapper around tensors of latents (and optional labels)."""
    def __init__(self, latents: torch.Tensor, labels=None):
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        sample = {"img": self.latents[idx]}
        if self.labels is not None:
            sample["label"] = self.labels[idx] if isinstance(self.labels, torch.Tensor) else self.labels[idx]
        return sample


def load_pre_encoded_latents(dataset_name, split):
    print(f"Loading pre-encoded latents from {dataset_name} {split} split")
    loaded = torch.load(
        Path(Path(__file__).parent.parent, "data", f"{dataset_name}_latents.pt")
    )
    if isinstance(loaded, dict) and "latents" in loaded:
        latents = loaded["latents"]
        labels = loaded.get("labels", None)
    else:
        latents = loaded
        labels = None

    print(f"Latents shape: {tuple(latents.shape)}")
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            print(f"Labels shape: {tuple(labels.shape)}")
        else:
            print(f"Labels length: {len(labels)}")

    return LatentsTorchDataset(latents, labels)


def preprocess_dataset(dataset, config, split, dataset_name, eval=False, latents=False):
    # TODO: Parameterise the proprocessing transformations
    
    # Base transforms
    base_transforms = []
    
    # Add CelebA-specific transforms
    if "CelebA" in dataset_name or "celeba" in dataset_name.lower() and not latents:
        base_transforms.extend([
            transforms.CenterCrop(178),     # from 178×218 → 178×178
            transforms.Resize(128),          # resize to 128×128
        ])
    
    if not eval and not latents:
        base_transforms.extend([
            # transforms.RandomCrop(config.image_size, padding=2),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(size=32, scale=(0.95, 1.0), ratio=(0.95, 1.05), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        tfm = transforms.Compose(base_transforms)
    else:
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        tfm = transforms.Compose(base_transforms)

    # TODO: fix it so it works for both cifar10 and celebA
    dataset.set_transform(
        lambda examples: {
            "img": [tfm(image.convert("RGB")) for image in examples["img"]] if "img" in examples else [tfm(image.convert("RGB")) for image in examples["image"]],
            "label": examples["label"] if "label" in examples else None
        }
    )

    return dataset


def create_dataloader(dataset_name, split, config, eval=False, latents=False, subset_size = None):
    if latents:
        dataset = load_pre_encoded_latents(dataset_name, split)
    else:
        dataset = load_dataset_from_hf(dataset_name, split=split)
        dataset = preprocess_dataset(dataset, config, split, dataset_name, eval)
    
    if subset_size is not None:
        # Check if dataset has train_test_split method (HuggingFace dataset)
        if hasattr(dataset, 'train_test_split'):
            # HuggingFace dataset
            subset_dataset = dataset.train_test_split(test_size=subset_size, seed=42)
            train_dataset = subset_dataset['train']
            val_dataset = subset_dataset['test']
            train_labels = get_labels_from_dataset(train_dataset)
            val_labels = get_labels_from_dataset(val_dataset)

            print("Train:", Counter(train_labels))
            print("Val:", Counter(val_labels))
            dataset = subset_dataset['test']
            print(f"Subset size: {len(dataset)}")
        else:
            # PyTorch dataset (e.g., LatentsTorchDataset)
            total_size = len(dataset)
            subset_size_count = int(total_size * subset_size)
            train_size = total_size - subset_size_count
            
            # Create random indices
            np.random.seed(42)
            indices = np.random.permutation(total_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            # Create subsets
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            # Get labels for counting (subset of indices for efficiency)
            sample_indices = val_indices[:min(100, len(val_indices))]  # Sample for label counting
            val_labels = []
            for idx in sample_indices:
                item = dataset[idx]
                if isinstance(item, dict) and 'label' in item:
                    val_labels.append(item['label'])
            
            if val_labels:
                print("Val labels sample:", Counter(val_labels))
            dataset = val_dataset
            print(f"Subset size: {len(dataset)}")

    print(f"Creating dataloader for {dataset_name} {split} split")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.train_batch_size if not eval else config.eval_batch_size, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=False,  
        persistent_workers=False, 
        prefetch_factor=1,  
    )
    print(f"Dataloader created for {dataset_name} {split} split")
    return dataloader

def create_lantent_dataloader_celebA(config):
    dataset = load_pre_encoded_latents("celebA", "train")
    
    # Handle PyTorch dataset splitting
    total_size = len(dataset)
    val_size = int(total_size * 0.05)
    train_size = total_size - val_size
    
    # Create random indices
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return train_dataloader, val_dataloader


def get_labels_from_dataset(dataset):
    """Extract labels from any dataset type"""
    labels = []
    
    # Try different approaches
    for i in range(len(dataset)):  # Check first 10 items
        item = dataset[i]
        
        if isinstance(item, dict):
            if 'label' in item:
                labels.append(item['label'])
            elif 'labels' in item:
                labels.append(item['labels'])
            elif 'class' in item:
                labels.append(item['class'])
        elif isinstance(item, (list, tuple)) and len(item) > 1:
            # Assuming (data, label) format
            labels.append(item[1])
        
        if i == 0:  # Print structure of first item
            print(f"Dataset item structure: {type(item)}")
            if isinstance(item, dict):
                print(f"Available keys: {list(item.keys())}")
    
    return labels


if __name__ == "__main__":
    create_lantent_dataloader_celebA(TrainingConfig())