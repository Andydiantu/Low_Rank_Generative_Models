import torch
from datasets import load_dataset
from torchvision import transforms


def load_dataset_from_hf(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset {dataset_name} {split} split with {len(dataset)} images")
    return dataset


def preprocess_dataset(dataset, config, split):
    # TODO: Parameterise the proprocessing transformations
    if split == "train":
        tfm = transforms.Compose([
            # transforms.RandomCrop(config.image_size, padding=2),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(size=32, scale=(0.95, 1.0), ratio=(0.95, 1.05), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])

    dataset.set_transform(
        lambda examples: {
            "img": [tfm(image.convert("RGB")) for image in examples["img"]],
            "label": examples["label"]
        }
    )

    return dataset


def create_dataloader(dataset_name, split, config):
    dataset = load_dataset_from_hf(dataset_name, split=split)
    dataset = preprocess_dataset(dataset, config, split)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,  
        persistent_workers=True, 
        prefetch_factor=2,  
    )
    return dataloader
