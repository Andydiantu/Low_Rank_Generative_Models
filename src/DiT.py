from diffusers import DDPMScheduler, DiTTransformer2DModel

# TODO: Parameterise the model config
def create_model(config):
    model = DiTTransformer2DModel(
        sample_size=config.image_size,
        in_channels=config.latent_channels if config.vae else config.pixel_channels,
        out_channels=config.latent_channels if config.vae else config.pixel_channels,
        num_layers=8,  
        num_attention_heads=8,
        attention_head_dim=32,
    )
    return model



def create_noise_scheduler(config):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        beta_schedule="scaled_linear", 
        clip_sample=False,
        prediction_type="epsilon",
    )
    return noise_scheduler

