from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    image_size = 32      
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 100
    latent_channels = 4
    pixel_channels = 3
    gradient_accumulation_steps = 2
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10  # for testing
    save_model_epochs = 50
    eval_dataset_size = 1000
    # mixed_precision = (
    #     "fp16"
    # )
    output_dir = (
        Path(__file__).parent.parent
        / "logs"
        / ("DiT" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    vae = False
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
