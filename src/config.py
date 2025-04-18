from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    image_size = 64
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 50
    latent_channels = 4
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1  # for testing
    save_model_epochs = 30
    # mixed_precision = (
    #     "fp16"
    # )
    output_dir = (
        Path(__file__).parent.parent
        / "logs"
        / ("DiT" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )

    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
