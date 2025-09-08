import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DiTPipeline
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings

# Add the src directory to Python path so we can import custom modules
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from vae import SD_VAE, DummyAutoencoderKL
from DiT import create_noise_scheduler
from config import TrainingConfig


def save_images(images: List["Image.Image"], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(images):
        img.save(os.path.join(output_dir, f"{idx:04d}.png"))


def load_pipeline_from_checkpoint(checkpoint_path: str, device: torch.device) -> DiTPipeline:
    """Recreate pipeline from scratch and load weights, just like in training"""
    
    print("Recreating model components from scratch...")
    
    # Create components from scratch using default config (same as training)
    config = TrainingConfig()
    print(f"Config: vae={config.vae}, use_latents={config.use_latents}")
    
    # Create model, scheduler, and VAE exactly like in training
    model = create_model(config)
    noise_scheduler = create_noise_scheduler(config)
    
    # Create VAE based on config flags, matching the training logic exactly
    if not config.vae and config.use_latents:
        # Model works in latent space but needs VAE for decoding (default case)
        print("Creating SD_VAE for latent decoding (config.vae=False, config.use_latents=True)")
        vae_wrapper = SD_VAE()
        vae = vae_wrapper.vae
    elif config.vae:
        # VAE is used during training 
        print("Using SD_VAE as configured (config.vae=True)")
        vae_wrapper = SD_VAE()
        vae = vae_wrapper.vae
    else:
        # No VAE, works directly with images
        print("Using DummyAutoencoderKL (config.vae=False, config.use_latents=False)")
        vae_wrapper = DummyAutoencoderKL()
        vae = vae_wrapper
    
    # Load weights if provided
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt"):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("✅ Successfully loaded model weights")
    elif os.path.isdir(checkpoint_path):
        print("Directory provided but no weights file specified. Using randomly initialized model.")
    else:
        raise ValueError("checkpoint_path must be a .pt weights file or a directory")
    
    # Create the pipeline
    pipeline = DiTPipeline(
        transformer=model,
        scheduler=noise_scheduler,
        vae=vae,
    )
    
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    print("✅ Pipeline created successfully")
    return pipeline


def generate_images(
    checkpoint_path: str,
    num_images: int,
    output_dir: str,
    seed: Optional[int],
    num_inference_steps: int,
    guidance_scale: float,
    cfg_enabled: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Set default output directory based on checkpoint path
    if not output_dir:
        if os.path.isfile(checkpoint_path):
            output_dir = os.path.join(str(Path(checkpoint_path).parent), "samples")
        else:
            output_dir = os.path.join(checkpoint_path, "samples")
    else:
        output_dir = os.path.abspath(output_dir)

    # Load the pipeline using our robust loading approach
    pipeline = load_pipeline_from_checkpoint(checkpoint_path, device)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    # Create class labels for generation
    if cfg_enabled:
        class_labels = list(range(num_images))
        print(f"Generating {num_images} images with class labels 0-{num_images-1} and CFG scale {guidance_scale}")
    else:
        class_labels = torch.zeros(num_images, dtype=torch.long, device=device)
        print(f"Generating {num_images} images unconditionally")

    print(f"Using {num_inference_steps} inference steps with seed {seed}")

    with torch.inference_mode():
        images = pipeline(
            class_labels=class_labels,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale if cfg_enabled else 1.0,
        ).images

    save_images(images, output_dir)
    print(f"✅ Saved {len(images)} images to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a trained DiT checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to saved pipeline directory or .pt weights file inside that directory")
    parser.add_argument("--num-images", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save images. Defaults to <checkpoint>/samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="CFG guidance scale (used only if cfg-enabled)")
    parser.add_argument("--cfg-enabled", action="store_true", help="Enable classifier-free guidance with class labels 0..N-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_images(
        checkpoint_path=args.checkpoint_path,
        num_images=args.num_images,
        output_dir=args.output_dir if args.output_dir else "",
        seed=args.seed,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        cfg_enabled=args.cfg_enabled,
    )


if __name__ == "__main__":
    main()


