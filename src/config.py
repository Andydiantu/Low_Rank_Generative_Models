from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    image_size = 32      
    train_batch_size = 128
    eval_batch_size = 16
    num_epochs = 1000
    latent_channels = 4
    pixel_channels = 3
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    weight_decay = 0.01
    lr_warmup_steps = 2000
    save_image_epochs = 10 
    save_model_epochs = 50
    evaluate_fid_epochs = 100
    eval_dataset_size = 10000
    num_inference_steps = 1000
    cfg_enabled = True
    unconditional_prob = 0.1
    guidance_scale = 7.5
    low_rank_pretraining = False
    low_rank_rank = 64
    low_rank_compression = False
    load_pretrained_model = False
    pretrained_model_path = "put a path here"
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
