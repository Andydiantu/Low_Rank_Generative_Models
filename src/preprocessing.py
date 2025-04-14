import torch
from datasets import load_dataset
from torchvision import transforms


def load_dataset_from_hf(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset {dataset_name} {split} split with {len(dataset)} images")
    return dataset


def preprocess_dataset(dataset, config):
    # TODO: Parameterise the proprocessing transformations
    # TODO: Add other transformations
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset.set_transform(
        lambda examples: {
            "img": [preprocess(image.convert("RGB")) for image in examples["img"]]
        }
    )

    return dataset


def create_dataloader(dataset_name, split, config):
    dataset = load_dataset_from_hf(dataset_name, split=split)
    dataset = preprocess_dataset(dataset, config)
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
