from datetime import datetime
from dataclasses import dataclass, field # Import field for Path with default_factory
from pathlib import Path
from typing import Optional # For potentially optional pretrained_model_path

@dataclass
class TrainingConfig:
    image_size: int = 32
    train_batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 3000
    prediction_type: str = "epsilon"
    latent_channels: int = 4
    pixel_channels: int = 3
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_warmup_steps: int = 3000
    validation_epochs: int = 15
    save_image_epochs: int = 15
    save_model_epochs: int = 15
    evaluate_fid_epochs: int = 150
    # validation_epochs: int = 1 # for testing
    # save_image_epochs: int = 1 # for testing
    # save_model_epochs: int = 1 # for testing
    # evaluate_fid_epochs: int = 10 # for testing
    eval_dataset_size: int = 1024
    noise_scheduler: str = "DDIM"
    num_training_steps: int = 1000
    num_inference_steps: int = 100
    cfg_enabled: bool = True
    unconditional_prob: float = 0.1
    guidance_scale: float = 2
    low_rank_pretraining: bool = False
    ortho_loss_weight: float = 1e-1 
    frobenius_loss_weight: float = 1e-5
    nuclear_norm_loss: bool = False
    nuclear_norm_loss_weight: float = 1e-5
    frobenius_norm_loss: bool = False
    frobenius_norm_loss_weight: float = 1e-6
    low_rank_rank: float = 0.25
    low_rank_compression: bool = False
    low_rank_gradient: bool = False
    low_rank_gradient_rank: int = 32
    curriculum_learning: bool = False
    curriculum_learning_patience: int = 5
    curriculum_learning_timestep_num_groups: int = 5
    curriculum_learning_current_group_portion: float = 0.2
    curriculum_learning_ema_alpha: float = 0.1
    curriculum_learning_ema_warmup: int = 3
    curriculum_learning_start_from_low: bool = True
    curriculum_learning_start_from_middle: bool = False
    curriculum_learning_middle_group_index: int = 7
    curriculum_learning_full_finetune_count: int = 200
    curriculum_learning_gradual: bool = True
    curriculum_learning_gradual_patience: int = 5
    curriculum_learning_gradual_start: int = 844
    curriculum_learning_gradual_step_size: int = 50
    curriculum_learning_gradual_start_alpha: float = 0.05
    curriculum_learning_gradual_end_alpha: float = 0.3
    gradual_curriculum_learning_num_epochs: int = 100
    curriculum_learning_gradual_alpha_schedule: str = "bell"
    real_features_path: str = "data/fid_features/CIFAR10_train_features_fp64_0.3.pt"
    load_pretrained_model: bool = False
    pretrained_model_path: Optional[str] = "logs/DiT20250823_110454/model_0044.pt"
    load_pretrained_ema: bool = False
    pretrained_ema_path: Optional[str] = "logs/DiT20250817_152007/EMA_model_0899.pt"
    # mixed_precision: str = "fp16" # Uncomment and type if used
    vae: bool = False
    use_latents: bool = False
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs" / ("DiT" + datetime.now().strftime("%Y%m%d_%H%M%S")))


@dataclass
class LDConfig:
    image_size: int = 128
    train_batch_size: int = 256
    eval_batch_size: int = 16
    num_epochs: int = 1500
    latent_channels: int = 4
    pixel_channels: int = 3
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    lr_warmup_steps: int = 1500
    validation_epochs: int = 15
    save_image_epochs: int = 15
    save_model_epochs: int = 15
    evaluate_fid_epochs: int = 300
    # validation_epochs: int = 1 # for testing
    # save_image_epochs: int = 1 # for testing
    # save_model_epochs: int = 1 # for testing
    # evaluate_fid_epochs: int = 2 # for testing
    eval_dataset_size: int = 1024
    noise_scheduler: str = "DDIM"
    num_training_steps: int = 1000
    num_inference_steps: int = 100
    cfg_enabled: bool = False
    unconditional_prob: float = 0.1
    guidance_scale: float = 2
    low_rank_pretraining: bool = False
    ortho_loss_weight: float = 1e-1 
    frobenius_loss_weight: float = 1e-5
    nuclear_norm_loss: bool = False
    nuclear_norm_loss_weight: float = 1e-5
    low_rank_rank: int = 32
    low_rank_compression: bool = False
    low_rank_gradient: bool = True
    low_rank_gradient_rank: int = 32
    real_features_path: str = "data/fid_features/CIFAR10_train_features.pt"
    load_pretrained_model: bool = False
    pretrained_model_path: Optional[str] = "logs/DiT20250612_203153/model_0149.pt"
    load_pretrained_ema: bool = False
    pretrained_ema_path: Optional[str] = "logs/DiT20250612_203153/EMA_model_0149.pt"
    # mixed_precision: str = "fp16" # Uncomment and type if used
    vae: bool = True
    use_latents: bool = True
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs" / ("DiT" + datetime.now().strftime("%Y%m%d_%H%M%S")))

def print_config(config):
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")

