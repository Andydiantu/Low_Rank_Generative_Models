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
             f"{dataset_name}_latents.pt")
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
            # "label": examples["label"] if "label" in examples else None
        }
    )

    return dataset


def create_dataloader(dataset_name, split, config, eval=False, latents=False):
    if latents:
        dataset = load_pre_encoded_latents(dataset_name, split)
    else:
        dataset = load_dataset_from_hf(dataset_name, split=split)
        dataset = preprocess_dataset(dataset, config, split, dataset_name, eval)
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


if __name__ == "__main__":
    create_lantent_dataloader_celebA(TrainingConfig())