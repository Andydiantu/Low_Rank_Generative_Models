import os
from pathlib import Path
from PIL import Image

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import create_repo, upload_folder
from preprocessing import create_dataloader
from torch.nn import functional as F
from tqdm.auto import tqdm

from DiT import create_model, create_noise_scheduler


class DiTTrainer:
    def __init__(self, model, noise_scheduler, train_dataloader, config):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )

    def train_loop(self):
        logging_dir = os.path.join(self.config.output_dir, "logs")
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.output_dir, logging_dir=logging_dir
        )
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        global_step = 0

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["img"]
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                batch_size = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.num_train_timesteps,
                    (batch_size,),
                    device=clean_images.device,
                ).long()
                
                # For DiT models, class conditioning is important
                class_labels = None
                if "label" in batch:
                    class_labels = batch["label"]
                else:
                    class_labels = torch.zeros(
                        batch_size, dtype=torch.long, device=clean_images.device
                    )

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = self.noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                )

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(
                        noisy_images, timesteps, class_labels, return_dict=False
                    )[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDIMPipeline(  
                    unet=accelerator.unwrap_model(model), 
                    scheduler=self.noise_scheduler
                )

                if (
                    (epoch + 1) % self.config.save_image_epochs == 0
                    or epoch == self.config.num_epochs - 1
                ):
                    self.evaluate(self.config, epoch, pipeline)

                if (
                    (epoch + 1) % self.config.save_model_epochs == 0
                    or epoch == self.config.num_epochs - 1
                ):
                    pipeline.save_pretrained(self.config.output_dir)

    # Code to visualise the current epoch generated images
    def make_grid(self, images, rows, cols):
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(self, config, epoch, pipeline):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=200,
        ).images

        image_grid = self.make_grid(images, rows=4, cols=4)

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")


def main():
    # TODO: using import args to parameterise the model config

    from dataclasses import dataclass

    @dataclass
    class TrainingConfig:
        image_size = 64 
        train_batch_size = 16
        eval_batch_size = 16 
        num_epochs = 50
        gradient_accumulation_steps = 1
        learning_rate = 1e-4
        lr_warmup_steps = 500
        save_image_epochs = 1 # for testing
        save_model_epochs = 30
        mixed_precision = (
            "fp16"  
        )
        output_dir = "DiT"  

        push_to_hub = False
        hub_private_repo = False
        overwrite_output_dir = (
            True
        )
        seed = 0

    config = TrainingConfig()

    train_loader = create_dataloader("uoft-cs/cifar10", "train", config)

    model = create_model(config)
    noise_scheduler = create_noise_scheduler(config)

    trainer = DiTTrainer(model, noise_scheduler, train_loader, config)
    trainer.train_loop()


if __name__ == "__main__":
    main()
