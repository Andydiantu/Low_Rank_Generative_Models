from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Subset
from datasets import load_dataset, Dataset
from torchvision import transforms
from config import TrainingConfig
import numpy as np

def print_class_distribution(dataset, dataset_name, is_latent_dataset=False, max_classes_to_show=20):
    """Print per-class counts for a dataset"""
    try:
        labels = []
        
        if is_latent_dataset:
            # For LatentsTorchDataset
            if dataset.labels is not None:
                if isinstance(dataset.labels, torch.Tensor):
                    labels = dataset.labels.tolist()
                else:
                    labels = list(dataset.labels)
        else:
            # For HuggingFace datasets - sample a subset for efficiency on large datasets
            sample_size = min(len(dataset), 10000)  # Sample max 10k items for class counting
            indices = np.random.choice(len(dataset), sample_size, replace=False) if len(dataset) > sample_size else range(len(dataset))
            
            for idx in indices:
                item = dataset[idx]
                if isinstance(item, dict):
                    if "label" in item and item["label"] is not None:
                        labels.append(item["label"])
                    elif "labels" in item and item["labels"] is not None:
                        labels.append(item["labels"])
        
        if labels:
            class_counts = Counter(labels)
            total_samples = len(labels)
            num_classes = len(class_counts)
            
            print(f"\n=== Class Distribution for {dataset_name} ===")
            print(f"Total samples: {total_samples}")
            print(f"Number of classes: {num_classes}")
            
            # Sort by class label
            sorted_classes = sorted(class_counts.items())
            
            # Show first max_classes_to_show classes
            classes_to_show = sorted_classes[:max_classes_to_show]
            for class_id, count in classes_to_show:
                percentage = (count / total_samples) * 100
                print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
            
            if len(sorted_classes) > max_classes_to_show:
                remaining = len(sorted_classes) - max_classes_to_show
                print(f"  ... and {remaining} more classes")
            
            # Show statistics
            counts = list(class_counts.values())
            print(f"Min samples per class: {min(counts)}")
            print(f"Max samples per class: {max(counts)}")
            print(f"Avg samples per class: {np.mean(counts):.1f}")
            print("=" * 50)
        else:
            print(f"No labels found for {dataset_name}")
            
    except Exception as e:
        print(f"Error computing class distribution for {dataset_name}: {e}")

def filter_latent_dataset_by_class(dataset, n_classes):
    """Filter LatentsTorchDataset to keep only first n_classes"""
    if dataset.labels is None:
        print("Warning: No labels found in latent dataset, cannot filter by class")
        return dataset
    
    # Convert labels to tensor if needed
    labels = dataset.labels
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # Create mask for first n classes
    mask = labels < n_classes
    
    # Filter latents and labels
    filtered_latents = dataset.latents[mask]
    filtered_labels = labels[mask]
    
    return LatentsTorchDataset(filtered_latents, filtered_labels)

def load_dataset_from_hf(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset {dataset_name} {split} split with {len(dataset)} images")
    
    # Print per-class counts
    print_class_distribution(dataset, dataset_name)
    
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

    dataset = LatentsTorchDataset(latents, labels)
    
    # Print per-class counts for latent dataset
    print_class_distribution(dataset, dataset_name, is_latent_dataset=True)
    
    return dataset


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
        # Optionally restrict ImageNet latents to first N classes by label id
        if ("imagenet" in dataset_name.lower() or "imagenet-1k" in dataset_name.lower()) and getattr(config, "imagenet_first_n_classes", None):
            n = int(config.imagenet_first_n_classes)
            dataset = filter_latent_dataset_by_class(dataset, n)
            print(f"Filtered ImageNet latents to first {n} classes → {len(dataset)} samples")
            # Print class distribution after filtering
            print_class_distribution(dataset, f"{dataset_name} latents (filtered to {n} classes)", is_latent_dataset=True)
    else:
        dataset = load_dataset_from_hf(dataset_name, split=split)
        # Optionally restrict ImageNet to first N classes by label id
        
        if ("imagenet" in dataset_name.lower() or "imagenet-1k" in dataset_name.lower()) and getattr(config, "imagenet_first_n_classes", None):
            n = int(config.imagenet_first_n_classes)
            # HF dataset returns dicts with 'label'
            dataset = dataset.filter(lambda x: ("label" in x) and (x["label"] is not None) and (int(x["label"]) < n))
            print(f"Filtered ImageNet to first {n} classes → {len(dataset)} samples")
            # Print class distribution after filtering
            print_class_distribution(dataset, f"{dataset_name} (filtered to {n} classes)")
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