from diffusers import DDPMScheduler, DiTTransformer2DModel

# TODO: Parameterise the model config
def create_model(config):
    model = DiTTransformer2DModel(
        sample_size=config.image_size,
        in_channels=config.latent_channels if config.vae else config.pixel_channels,
        out_channels=config.latent_channels if config.vae else config.pixel_channels,
        num_layers=12,  
        num_attention_heads=6,
        attention_head_dim=32,
        patch_size = 4,
    )
    return model



def create_noise_scheduler(config):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_inference_steps, 
        beta_schedule="squaredcos_cap_v2", 
        clip_sample=True,
        prediction_type="epsilon",
    )
    return noise_scheduler

