from collections import Counter
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from torchvision import transforms
from config import TrainingConfig

def load_dataset_from_hf(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset {dataset_name} {split} split with {len(dataset)} images")
    return dataset

def load_pre_encoded_latents(dataset_name, split):
    # (N, latent_dim, …)
    latents = torch.load(
        Path(Path(__file__).parent.parent, "data",
             f"{dataset_name}_latents_128.pt")
    )

    data_dict = {"img": [t for t in latents]}        
    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="torch", columns=["img"])


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
    else:
        dataset = load_dataset_from_hf(dataset_name, split=split)
        dataset = preprocess_dataset(dataset, config, split, dataset_name, eval)
    
    if subset_size is not None:
        subset_dataset = dataset.train_test_split(test_size=subset_size, seed=42)
        train_dataset = subset_dataset['train']
        val_dataset = subset_dataset['test']
        train_labels = get_labels_from_dataset(train_dataset)
        val_labels = get_labels_from_dataset(val_dataset)

        print("Train:", Counter(train_labels))
        print("Val:", Counter(val_labels))
        dataset = subset_dataset['test']
        print(f"Subset size: {len(dataset)}")

    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.train_batch_size if not eval else config.eval_batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,  
        persistent_workers=True, 
        prefetch_factor=2,  
    )
    return dataloader

def create_lantent_dataloader_celebA(config):
    dataset = load_pre_encoded_latents("celebA", "train")
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

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