from datetime import datetime
from dataclasses import dataclass, field # Import field for Path with default_factory
from pathlib import Path
from typing import Optional # For potentially optional pretrained_model_path

@dataclass
class TrainingConfig:
    image_size: int = 32
    train_batch_size: int = 128
    eval_batch_size: int = 64
    num_epochs: int = 2000
    latent_channels: int = 4
    pixel_channels: int = 3
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 2000
    save_image_epochs: int = 50
    save_model_epochs: int = 25
    evaluate_fid_epochs: int = 400
    # save_image_epochs: int = 1 # for testing
    # save_model_epochs: int = 1 # for testing
    # evaluate_fid_epochs: int = 1 # for testing
    eval_dataset_size: int = 1000
    num_training_steps: int = 1000
    num_inference_steps: int = 1000
    cfg_enabled: bool = True
    unconditional_prob: float = 0.1
    guidance_scale: float = 4.5
    low_rank_pretraining: bool = False
    low_rank_rank: int = 64
    low_rank_compression: bool = False
    low_rank_gradient: bool = True
    low_rank_gradient_rank: int = 128
    load_pretrained_model: bool = False
    pretrained_model_path: Optional[str] = "logs/DiT20250511_044825/model.pt" # Or Path if you prefer
    # mixed_precision: str = "fp16" # Uncomment and type if used
    vae: bool = False
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs" / ("DiT" + datetime.now().strftime("%Y%m%d_%H%M%S")))
