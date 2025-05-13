import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DiTPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from PIL import Image
from torch.nn import functional as F
from tqdm.auto import tqdm
from galore_torch import GaLoreAdamW

from config import TrainingConfig
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings
from eval import Eval
from preprocessing import create_dataloader
from vae import SD_VAE, DummyAutoencoderKL
from low_rank_compression import label_low_rank_gradient_layers,apply_low_rank_compression, low_rank_layer_replacement


class DiTTrainer:
    def __init__(self, model, noise_scheduler, train_dataloader, config):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.config = config

        if not config.low_rank_gradient:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            galore_params, regular_params = label_low_rank_gradient_layers(self.model)
            param_groups = [{'params': regular_params}, 
                            {'params': galore_params, 'rank': self.config.low_rank_gradient_rank, 'update_proj_gap': 200, 'scale': 1, 'proj_type': 'std'}]
            self.optimizer = GaLoreAdamW(param_groups, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )
        self.ema_model = EMAModel(
            parameters=self.model.parameters(),
            decay=0.9999,
            use_ema_warmup=True,
            power = 0.75, 
        )

        if config.vae:
            self.vae = SD_VAE()
        else:
            self.vae = DummyAutoencoderKL()

    def train_loop(self):
        logging_dir = os.path.join(self.config.output_dir, "logs")
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.output_dir, logging_dir=logging_dir
        )
        accelerator = Accelerator(
            # mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        model, optimizer, train_dataloader, lr_scheduler, vae, ema_model = accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.vae,
            self.ema_model,
        )  

        # Manually move ema_model to the correct device
        ema_model.to(accelerator.device) 

        global_step = 0

        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not accelerator.is_local_main_process,
                dynamic_ncols=True,
                leave=False  
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["img"]
                if self.config.vae:
                    latents = self.vae.encode(clean_images)
                else:
                    latents = clean_images
                # Sample noise to add to the images
                noise = torch.randn(latents.shape).to(latents.device)
                batch_size = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.num_train_timesteps,
                    (batch_size,),
                    device=clean_images.device,
                ).long()

                # Add dummy class labels if doing unconditional generation
                class_labels = None
                if "label" in batch:
                    class_labels = batch["label"].to(latents.device)
                else:
                    class_labels = torch.zeros(batch_size, dtype=torch.long, device=latents.device)

                # ---- Classifier-Free Guidance: random label drop ----
                if self.config.cfg_enabled:
                    # boolean mask: True → keep label, False → drop
                    keep_mask = torch.rand(batch_size, device=latents.device) > self.config.unconditional_prob
                    # clone so we don't modify the original
                    class_labels_input = class_labels.clone()
                    class_labels_input = torch.where(
                        keep_mask,
                        class_labels_input,
                        torch.full_like(class_labels_input, fill_value=1000)   # 1000 triggers "no label", surprise!
                    )
                else:
                    class_labels_input = class_labels
                    
                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = self.noise_scheduler.add_noise(latents, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(
                        noisy_images, timesteps, class_labels_input, return_dict=False
                    )[0]
                    
                    alphas = self.noise_scheduler.alphas_cumprod[timesteps].to(latents.device)
                    alphas = alphas.view(-1, 1, 1, 1)
                    snr = alphas / (1 - alphas)  # SNR = alpha/(1-alpha)
                    snr_weight = (snr / (snr + 1)).detach() 
                    
                    loss = F.mse_loss(noise_pred, noise, reduction="none")
                    loss = loss * snr_weight
                    loss = loss.mean()

                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    ema_model.step(model.parameters())
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
                # TODO: Seems very memory intensive here, could i reduce it?

                if (
                    (epoch + 1) % self.config.save_image_epochs == 0
                    or (epoch + 1) % self.config.save_model_epochs == 0
                    or epoch == self.config.num_epochs - 1
                ):
                    # Create pipeline with memory optimizations with autocast
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                        model.eval()
                        pipeline = DiTPipeline(
                            transformer=accelerator.unwrap_model(model),
                            scheduler=self.noise_scheduler,
                            vae=self.vae.vae if self.config.vae else self.vae,
                        )

                        pipeline.enable_attention_slicing()

                        # Evaluation
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
                            torch.save(
                                model.state_dict(),
                                os.path.join(self.config.output_dir, f"model_{epoch:04d}.pt"),
                            )
                            
                        if (
                            (epoch + 1) % self.config.evaluate_fid_epochs == 0
                            or epoch == self.config.num_epochs - 1
                        ): 
                            self.evaluate_fid(self.config, epoch, pipeline)

                    # Explicit cleanup
                    ema_model.restore(model.parameters())
                    model.train()
                    del pipeline
                    torch.cuda.empty_cache()

    # Code to visualise the current epoch generated images
    def make_grid(self, images, rows, cols):
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(self, config, epoch, pipeline):

        # Sample some images from random noise (this is the backward diffusion process).
        images = pipeline(
            class_labels=torch.tensor([i % 10 for i in range(config.eval_batch_size)], dtype=torch.long),
            generator=torch.manual_seed(config.seed),
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale if config.cfg_enabled else None,
        ).images

        image_grid = self.make_grid(images, rows=4, cols=4)

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def evaluate_fid(self, config, epoch, pipeline):
        test_dataloader = create_dataloader("uoft-cs/cifar10", "test", config)
        eval = Eval(test_dataloader, config)
        self.ema_model.copy_to(self.model.parameters())
        fid_score = eval.compute_metrics(pipeline)
        print(f"FID Score: {fid_score}")
        del pipeline




def main():

    config = TrainingConfig()
    print(config)

    train_loader = create_dataloader("uoft-cs/cifar10", "train", config)

    model = create_model(config)
    noise_scheduler = create_noise_scheduler(config)

    print_model_settings(model)
    print_noise_scheduler_settings(noise_scheduler)

    if config.load_pretrained_model:
        path = Path(__file__).parent.parent / config.pretrained_model_path
        print(f"Loading pretrained model from {path}")
        model.load_state_dict(torch.load(path))
        print("Loading complete")


    if config.low_rank_pretraining:
        model = low_rank_layer_replacement(model, rank=config.low_rank_rank)
        print(f"number of parameters in model after compression is: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = DiTTrainer(model, noise_scheduler, train_loader, config)
    trainer.train_loop()

    

    if config.low_rank_compression:

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        
        print(f"number of parameters in model: {count_parameters(model)}")
        model = apply_low_rank_compression(model, threshold=0.6)
        print(f"number of parameters in model after compression is: {count_parameters(model)}")
        config.num_epochs = 5 # finetune for 5 epoch TODO: parameterise this.
        finetune_trainer = DiTTrainer(model, noise_scheduler, train_loader, config)
        finetune_trainer.train_loop()

        compressed_pipeline = DiTPipeline(
            transformer=model,
            scheduler=noise_scheduler,
            vae=trainer.vae.vae if config.vae else trainer.vae,
        )

        compressed_pipeline.enable_attention_slicing()
        fid_score = eval.compute_metrics(compressed_pipeline)
        print(f"FID Score: {fid_score}")


if __name__ == "__main__":
    main()
