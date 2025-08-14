from diffusers import DDPMScheduler, DDIMScheduler, DiTTransformer2DModel

# TODO: Parameterise the model config
def create_model(config):
    model = DiTTransformer2DModel(
        sample_size=config.image_size if not config.vae else 32,  # 16 for 128px images with 8x downsampling
        in_channels=config.latent_channels if config.vae else config.pixel_channels,
        out_channels=config.latent_channels if config.vae else config.pixel_channels,
        num_layers=12,  # DiT-S/2 uses 12 layers
        num_attention_heads=6,  # DiT-S/2 uses 6 heads
        attention_head_dim=64,  # This gives 384 total dim (6*64)
        patch_size=2,
    )
    return model



def create_noise_scheduler(config):
    if config.noise_scheduler == "DDPM":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_training_steps, 
            beta_schedule="squaredcos_cap_v2", 
            clip_sample=True,
            prediction_type=config.prediction_type,
        )
    elif config.noise_scheduler == "DDIM":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_training_steps, 
            beta_schedule="squaredcos_cap_v2", 
            clip_sample=True,
            prediction_type=config.prediction_type,
        )

    return noise_scheduler


def print_model_settings(model):
    """Print all settings of the DiT model."""
    print("DiT Model Settings:")
    print(f"  Sample Size: {model.config.sample_size}")
    print(f"  In Channels: {model.config.in_channels}")
    print(f"  Out Channels: {model.config.out_channels}")
    print(f"  Num Layers: {model.config.num_layers}")
    print(f"  Num Attention Heads: {model.config.num_attention_heads}")
    print(f"  Attention Head Dim: {model.config.attention_head_dim}")
    print(f"  Patch Size: {model.config.patch_size}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")


def print_noise_scheduler_settings(noise_scheduler):
    """Print all settings of the noise scheduler."""
    print("Noise Scheduler Settings:")
    print(f"  Num Train Timesteps: {noise_scheduler.config.num_train_timesteps}")
    print(f"  Beta Schedule: {noise_scheduler.config.beta_schedule}")
    print(f"  Clip Sample: {noise_scheduler.config.clip_sample}")
    print(f"  Prediction Type: {noise_scheduler.config.prediction_type}")
    print(f"  Beta Start: {noise_scheduler.config.beta_start}")
    print(f"  Beta End: {noise_scheduler.config.beta_end}")

