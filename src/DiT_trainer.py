import os
from pathlib import Path
import time

import torch
import torch.profiler
from diffusers import DiTPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from PIL import Image
from torch.nn import functional as F
from tqdm.auto import tqdm
from galore_torch import GaLoreAdamW, GaLoreEvalAdamW
import matplotlib.pyplot as plt
import numpy as np
from training_monitor import TrainingMonitor
from config import TrainingConfig, LDConfig, print_config
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings
from eval import Eval, plot_loss_curves
from preprocessing import create_dataloader, create_lantent_dataloader_celebA
from vae import SD_VAE, DummyAutoencoderKL
from low_rank_compression import label_low_rank_gradient_layers,apply_low_rank_compression, low_rank_layer_replacement, LowRankLinear, nuclear_norm, frobenius_norm



class DiTTrainer:
    def __init__(self, model, noise_scheduler, train_dataloader, validation_dataloader, config):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.config = config
        self.train_loss_history = []
        self.val_loss_history = []
        self.ema_val_loss_history = []
        self.eval = Eval(train_dataloader , config)

        self.projection_loss_history = []
        
        # Add gradient variance tracking - per epoch approach
        self.epoch_gradient_norms = []  # Store gradient norms for current epoch
        self.epoch_avg_gradient_norms = []  # Store average gradient norm per epoch
        self.epoch_gradient_variances = []  # Store gradient variance per epoch

        if config.curriculum_learning:
            self.training_monitor = TrainingMonitor(patience=config.curriculum_learning_patience, num_timestep_groups=config.curriculum_learning_timestep_num_groups+1)

        if not config.low_rank_gradient:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            print("Using GaLoreEvalAdamW")
            galore_params, regular_params = label_low_rank_gradient_layers(self.model)
            param_to_name = {param: name for name, param in self.model.named_parameters()}
            regular_param_names = [param_to_name[p] for p in regular_params]
            galore_param_names = [param_to_name[p] for p in galore_params]

            param_groups = [{'params': regular_params, 'param_names': regular_param_names}, 
                            {'params': galore_params, 'rank': self.config.low_rank_gradient_rank, 'update_proj_gap': 200, 'scale': 1, 'proj_type': 'std', 'param_names': galore_param_names}]
            self.optimizer = GaLoreEvalAdamW(param_groups, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

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
            # Freeze VAE parameters explicitly
            for param in self.vae.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = DummyAutoencoderKL()

    def train_loop(self):
        logging_dir = os.path.join(self.config.output_dir, "logs")
        # accelerator_project_config = ProjectConfiguration(
        #     project_dir=self.config.output_dir, logging_dir=logging_dir
        # )
        # accelerator = Accelerator(
        #     # mixed_precision=self.config.mixed_precision,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        #     log_with="tensorboard",
        #     project_config=accelerator_project_config,
        # )
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if torch.cuda.is_available():
            os.makedirs(self.config.output_dir, exist_ok=True)
            # accelerator.init_trackers("train_example")

        model, optimizer, train_dataloader, lr_scheduler, vae, ema_model, validation_dataloader = (
            self.model.to(device),
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.vae.to(device),
            self.ema_model,
            self.validation_dataloader
        )

        # Manually move ema_model to the correct device
        ema_model.to(device) 

        global_step = 0

        for epoch in range(self.config.num_epochs):
            # Reset epoch gradient norms for the new epoch
            self.epoch_gradient_norms = []
            
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable= "SLURM_JOB_ID" in os.environ, 
                dynamic_ncols=True,
                leave=False  
            )
            progress_bar.set_description(f"Epoch {epoch}")

            epoch_train_loss = 0.0
            epoch_ortho_loss = 0.0
            epoch_frobenius_loss = 0.0
            epoch_nuclear_norm_loss = 0.0
            epoch_frobenius_norm_loss = 0.0
            epoch_projection_loss = 0.0
            epoch_current_timestep_group_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["img"].to(device)
                if self.config.vae and not self.config.use_latents:
                    with torch.no_grad():
                        latents = self.vae.encode(clean_images)
                else:
                    latents = clean_images
                # Sample noise to add to the images
                noise = torch.randn(latents.shape).to(latents.device)
                batch_size = latents.shape[0]

                # Sample a random timestep for each image
                if self.config.curriculum_learning and not self.training_monitor.get_if_curriculum_learning_is_done():

                    trained_boundaries = self.training_monitor.get_trained_timesteps_boundaries()
                    trained_low_bound = trained_boundaries[0]
                    trained_high_bound = trained_boundaries[1]


                    # Split batch in half: first half samples from [low_bound, high_bound], 
                    # second half samples from [high_bound, num_train_timesteps]
                    current_group_boundaries = self.training_monitor.get_current_group_range()
                    current_low_bound = current_group_boundaries[0]
                    current_high_bound = current_group_boundaries[1]

                    if not trained_high_bound == trained_low_bound:

                        
                        first_batch = int(batch_size * self.config.curriculum_learning_current_group_portion)
                        second_batch = batch_size - first_batch
                        
                        # First half: sample from [low_bound, high_bound]
                        timesteps_first_half = torch.randint(
                            current_low_bound,
                            current_high_bound,
                            (first_batch,),
                            device=clean_images.device,
                        ).long()

                        timesteps_second_half = torch.randint(
                            trained_low_bound,
                            trained_high_bound,
                            (second_batch,),
                            device=clean_images.device,
                        ).long()

                        timesteps = torch.cat([timesteps_first_half, timesteps_second_half], dim=0)

                        current_timestep_group_batch_size = first_batch

                        
                    else:

                        timesteps = torch.randint(
                            current_low_bound,
                            current_high_bound,
                            (batch_size,),
                            device=clean_images.device,
                        ).long()

                        current_timestep_group_batch_size = batch_size



                else:
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,),
                        device=clean_images.device,
                    ).long()

                # print(f"sampling timesteps from {timesteps.min()} to {timesteps.max()}")

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

                optimizer.zero_grad()
                # with torch.profiler.profile(
                #     activities=[torch.profiler.ProfilerActivity.CPU,
                #                 torch.profiler.ProfilerActivity.CUDA],
                #     profile_memory=True,
                #     record_shapes=True
                # ) as prof:

                #     # Predict the noise residual
                pred = model(
                    noisy_images, timesteps, class_labels_input, return_dict=False
                )[0]
                if self.config.prediction_type == "epsilon":
                    target = noise
                elif self.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                        

                    
                # alphas = self.noise_scheduler.alphas_cumprod[timesteps].to(latents.device)
                # alphas = alphas.view(-1, 1, 1, 1)
                # # snr = alphas**2 / (1 - alphas**2)  # SNR = alpha/(1-alpha)
                # # print(snr)
                # snr = alphas / (1 - alphas)  # SNR = alpha/(1-alpha)
                # gamma  = 5.0
                # loss_weight = torch.minimum(gamma / snr, torch.ones_like(snr))  # Eq. (1)
                # print(loss_weight)
             
                loss = F.mse_loss(pred, target, reduction="none")


                if self.config.curriculum_learning:
                    loss_current_timestep_group = loss[:current_timestep_group_batch_size]

                    loss_current_timestep_group = loss_current_timestep_group.mean()

                    epoch_current_timestep_group_loss += loss_current_timestep_group.detach().item()

                # loss = loss * loss_weight
                loss = loss.mean()
                    
                # if self.config.low_rank_pretraining:
                #     ortho_loss = self.config.ortho_loss_weight * sum(
                #         m.orthogonality_loss(rho=0.01)
                #         for m in model.modules() if isinstance(m, LowRankLinear)
                #     )
                #     epoch_ortho_loss += ortho_loss.detach().item()

                #     frobenius_loss = self.config.frobenius_loss_weight * sum(
                #         m.frobenius_loss()
                #         for m in model.modules() if isinstance(m, LowRankLinear)
                #     )
                #     epoch_frobenius_loss += frobenius_loss.detach().item()

                if self.config.nuclear_norm_loss:
                    nuclear_norm_loss = self.config.nuclear_norm_loss_weight * nuclear_norm(model)
                    loss = loss + nuclear_norm_loss
                    epoch_nuclear_norm_loss += nuclear_norm_loss.detach().item()

                if self.config.frobenius_norm_loss:
                    frobenius_norm_loss = self.config.frobenius_norm_loss_weight * (frobenius_norm(model) ** 2) 
                    loss = loss + frobenius_norm_loss
                    epoch_frobenius_norm_loss += frobenius_norm_loss.detach().item()

                epoch_train_loss += loss.detach().item()

                # if self.config.low_rank_pretraining:
                #     loss = loss +  frobenius_loss 

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                # Calculate gradient norm for variance tracking (always, not just for curriculum learning)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                
                total_norm = total_norm ** 0.5  # Convert to L2 norm
                
                # Update gradient norms sliding window
                self.epoch_gradient_norms.append(total_norm)

                if self.config.low_rank_gradient:
                    _, projection_loss_dict = optimizer.step()
                    total_projection_loss = 0.0
                    num_layer_count = 0
                    for param_name, (err_F, err_cos) in projection_loss_dict.items():
                        total_projection_loss += err_F
                        num_layer_count += 1

                    # print(f"average projection loss: {total_projection_loss.detach().cpu() / num_layer_count:.6f}")
                    epoch_projection_loss += total_projection_loss / num_layer_count
                else:
                    optimizer.step()

                
                lr_scheduler.step()
                ema_model.step(model.parameters())

                # total_projection_loss = 0.0
                # for param_name, (err_F, err_cos) in projection_loss_dict.items():
                #     total_projection_loss += err_F
                # print(f"total projection loss: {total_projection_loss:.6f}")



                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    # "nuclear_norm_loss": nuclear_norm_loss.detach().item() if self.config.nuclear_norm_loss else 0,
                    # "frobenius_norm_loss": f"{frobenius_norm_loss.detach().item():.6f}" if self.config.frobenius_norm_loss else 0,
                    "grad_norm": f"{total_norm:.6f}",
                }
                
                progress_bar.set_postfix(**logs)
                # accelerator.log(logs, step=global_step)
                global_step += 1

            avg_epoch_train_loss = epoch_train_loss / len(train_dataloader)
            avg_epoch_ortho_loss = epoch_ortho_loss / len(train_dataloader)
            avg_epoch_frobenius_loss = epoch_frobenius_loss / len(train_dataloader)
            avg_epoch_nuclear_norm_loss = epoch_nuclear_norm_loss / len(train_dataloader)
            avg_epoch_frobenius_norm_loss = epoch_frobenius_norm_loss / len(train_dataloader)
            avg_epoch_current_timestep_group_loss = epoch_current_timestep_group_loss / len(train_dataloader) if self.config.curriculum_learning else 0.0
            self.train_loss_history.append(avg_epoch_train_loss)
            if self.config.low_rank_gradient:
                self.projection_loss_history.append(epoch_projection_loss.detach().cpu()/len(train_dataloader))
            torch.save(self.projection_loss_history, os.path.join(self.config.output_dir, "projection_loss_history.pt"))
            self.plot_projection_loss()
            
            # Save and plot gradient variance history
            # Calculate average gradient norm for the epoch
            if self.epoch_gradient_norms:
                avg_gradient_norm = np.mean(self.epoch_gradient_norms)
                self.epoch_avg_gradient_norms.append(avg_gradient_norm)
                # Calculate variance for the epoch
                if len(self.epoch_gradient_norms) > 1:
                    current_gradient_variance = np.var(self.epoch_gradient_norms)
                    self.epoch_gradient_variances.append(current_gradient_variance)
                else:
                    self.epoch_gradient_variances.append(0.0) # No variance if only one sample

            torch.save(self.epoch_gradient_variances, os.path.join(self.config.output_dir, "gradient_variance_history.pt"))
            torch.save(self.epoch_avg_gradient_norms, os.path.join(self.config.output_dir, "gradient_norms_history.pt"))
            self.plot_gradient_statistics()

            
            print(f"avg_epoch_current_timestep_group_loss: {avg_epoch_current_timestep_group_loss}")
            # if self.config.curriculum_learning and epoch % 2 == 0 and not self.training_monitor.get_if_curriculum_learning_is_done():
            #     boundaries = self.training_monitor.get_current_group_range()
            #     val_loss = self.validation_loss(model, ema_model, validation_dataloader, self.config, epoch, global_step, EMA = False, timestep_lower_bound = boundaries[0], timestep_upper_bound = boundaries[1])
                # print(f"Validation loss: {val_loss}")
            if self.training_monitor.call_simple_compare_best(avg_epoch_current_timestep_group_loss):
                if self.config.low_rank_gradient:
                    self.optimizer.reset_projection_matrices()
                    print("Resetting projection matrices")
                print(f"Updating curriculum learning timestep num groups to {self.training_monitor.current_timestep_groups}")
                if self.training_monitor.get_if_curriculum_learning_is_done():
                    print("Curriculum learning is done")
                else:
                    boundaries = self.training_monitor.get_current_group_range()
                    print(f"Current timestep groups low bound: {boundaries[0]}") 
                    print(f"Current timestep groups high bound: {boundaries[1]}")


            # Print the loss, lr, and step to the log file if running on a SLURM job
            if "SLURM_JOB_ID" in os.environ:
                print(f"Epoch {epoch} completed | loss: {avg_epoch_train_loss:.4f} | lr: {lr_scheduler.get_last_lr()[0]:.6f} | step: {global_step}" + 
                    #   (f" | ortho_loss: {avg_epoch_ortho_loss:.4f} | frobenius_loss: {avg_epoch_frobenius_loss:.4f}" if self.config.low_rank_pretraining else "") + 
                    #   (f" | nuclear_norm_loss: {avg_epoch_nuclear_norm_loss:.4f}" if self.config.nuclear_norm_loss else "") +
                    #   (f" | frobenius_norm_loss: {avg_epoch_frobenius_norm_loss:.4f}" if self.config.frobenius_norm_loss else "") +
                      (f" | grad_var: {self.epoch_gradient_variances[-1]:.6f}" if self.epoch_gradient_variances else "") +
                      (f" | grad_norm: {self.epoch_avg_gradient_norms[-1]:.6f}" if self.epoch_avg_gradient_norms else ""))                
            
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if torch.cuda.is_available():

                if (epoch + 1) % self.config.validation_epochs == 0:
                    EMA_val_loss = self.validation_loss(model, ema_model, validation_dataloader, self.config, epoch, global_step, EMA = True)
                    self.ema_val_loss_history.append(EMA_val_loss)

                    val_loss = self.validation_loss(model, ema_model, validation_dataloader, self.config, epoch, global_step, EMA = False)
                    self.val_loss_history.append(val_loss)

                    plot_loss_curves(self.config.validation_epochs, self.train_loss_history, self.val_loss_history, self.ema_val_loss_history, save_path=os.path.join(self.config.output_dir, "loss_curves.pdf"))

                if (
                    (epoch + 1) % self.config.save_image_epochs == 0
                    or (epoch + 1) % self.config.save_model_epochs == 0
                    or epoch == self.config.num_epochs - 1
                ):
                    # Create pipeline with memory optimizations with autocast
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        # Store original model parameters before applying EMA weights
                        original_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        
                        # Copy EMA parameters to model for evaluation
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                        model.eval()
                        
                        pipeline = DiTPipeline(
                            transformer=model,
                            scheduler=self.noise_scheduler,
                            vae=self.vae.vae if self.config.vae else self.vae,
                        )
                        
                        pipeline.set_progress_bar_config(disable=True)

                        # Evaluation
                        if (
                            (epoch + 1) % self.config.save_image_epochs == 0
                        ):
                            self.evaluate(self.config, epoch, pipeline)

                        if (
                            (epoch + 1) % self.config.evaluate_fid_epochs == 0
                        ): 
                            self.evaluate_fid(self.config, pipeline)

                        if (
                            (epoch + 1) % self.config.save_model_epochs == 0
                        ):
                            pipeline.save_pretrained(self.config.output_dir)
                            # Save EMA model (which is currently in the model)
                            torch.save(
                                model.state_dict(),
                                os.path.join(self.config.output_dir, f"EMA_model_{epoch:04d}.pt"),
                            )
                            
                            # Save original model from our saved state
                            torch.save(
                                original_model_state,
                                os.path.join(self.config.output_dir, f"model_{epoch:04d}.pt"),
                            )

                    # Restore original model parameters and set back to training mode
                    model.load_state_dict(original_model_state)
                    model.train()
                    del pipeline
                    del original_model_state  # Clean up the temporary state dict
                    torch.cuda.empty_cache()

    # Code to visualise the current epoch generated images
    def make_grid(self, images, rows, cols):
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def plot_projection_loss(self):
        """Plot and save the projection loss progression."""
        if not self.projection_loss_history:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.projection_loss_history)
        plt.title('Projection Loss Progression')
        plt.xlabel('Training Step')
        plt.ylabel('Projection Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.config.output_dir, "projection_loss_plot.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    def plot_gradient_statistics(self):
        """Plot gradient norms and variance in a combined figure."""
        if not self.epoch_gradient_variances or not self.epoch_avg_gradient_norms:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot average gradient norms per epoch
        ax1.plot(self.epoch_avg_gradient_norms)
        ax1.set_title('Average Gradient Norm Per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Gradient Norm')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot gradient variance per epoch
        ax2.plot(self.epoch_gradient_variances)
        ax2.set_title('Gradient Variance Per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gradient Variance')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.output_dir, "gradient_statistics_plot.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    def evaluate(self, config, epoch, pipeline):

        # Sample some images from random noise (this is the backward diffusion process).
        images = pipeline(
            class_labels=[i % 10 for i in range(config.eval_batch_size)] if config.cfg_enabled else torch.zeros(config.eval_batch_size, dtype=torch.long),
            generator = torch.Generator(device=self.model.device).manual_seed(config.seed),
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale if config.cfg_enabled else 1,
        ).images

        image_grid = self.make_grid(images, rows=4, cols=4)

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def evaluate_fid(self, config, pipeline, num_samples = 5000 ):
        self.ema_model.copy_to(self.model.parameters())
        fid_score = self.eval.compute_metrics(pipeline, num_samples)
        print(f"FID Score: {fid_score} \n")
        del pipeline

    def validation_loss(self, model, ema_model, validation_dataloader, config, epoch, global_step, EMA = False, timestep_lower_bound = 0, timestep_upper_bound = 1000):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        if EMA:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
        model.eval()

        val_loss_epoch = 0
        val_progress_bar = tqdm(
                total=len(validation_dataloader),
                disable="SLURM_JOB_ID" in os.environ,
                dynamic_ncols=True,
                leave=False
            )
        val_progress_bar.set_description(f"Epoch {epoch} - Validation")

        with torch.no_grad():
            for val_step, val_batch in enumerate(validation_dataloader):
                clean_images = val_batch["img"].to(device)
                if self.config.vae and not self.config.use_latents:
                    with torch.no_grad():
                        latents = self.vae.encode(clean_images)
                else:
                    latents = clean_images
                # Sample noise to add to the images
                noise = torch.randn(latents.shape).to(latents.device)
                batch_size = latents.shape[0]

                timesteps = torch.randint(
                    timestep_lower_bound,
                    timestep_upper_bound,
                    (batch_size,),
                    device=clean_images.device,
                ).long()

                class_labels = None
                if "label" in val_batch:
                    class_labels = val_batch["label"].to(latents.device)
                else:
                    class_labels = torch.zeros(batch_size, dtype=torch.long, device=latents.device)
                    
                if self.config.cfg_enabled:
                    keep_mask = torch.rand(batch_size, device=latents.device) > self.config.unconditional_prob
                    class_labels_input = class_labels.clone()
                    class_labels_input = torch.where(
                        keep_mask,
                        class_labels_input,
                        torch.full_like(class_labels_input, fill_value=1000)
                    )
                else: 
                    class_labels_input = class_labels

                noisy_images = self.noise_scheduler.add_noise(latents, noise, timesteps)

                pred = model(
                    noisy_images, timesteps, class_labels_input, return_dict=False
                )[0]
                if self.config.prediction_type == "epsilon":
                    target = noise
                elif self.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps) 

                val_loss = F.mse_loss(pred, target, reduction="none")
                val_loss = val_loss.mean()

                val_progress_bar.update(1)
                logs = {
                    "loss": val_loss.detach().item(),
                }
                val_progress_bar.set_postfix(**logs)
                val_loss_epoch += val_loss.detach().item()
        avg_val_loss = val_loss_epoch / len(validation_dataloader)
        # accelerator.log({"val_loss": avg_val_loss}, step=global_step)
        val_progress_bar.close()
        if EMA:
            ema_model.restore(model.parameters())
        model.train()


        if "SLURM_JOB_ID" in os.environ:
            print(f"Epoch {epoch} completed | val_loss: {avg_val_loss:.4f}\n")

        return avg_val_loss


def main():

    config = TrainingConfig()
    # config = LDConfig()
    print_config(config)
    
    train_loader = create_dataloader("uoft-cs/cifar10", "train", config)
    validation_loader = create_dataloader("uoft-cs/cifar10", "test", config, eval=True)

    # train_loader = create_dataloader("nielsr/CelebA-faces", "train", config)
    # validation_loader = create_dataloader("nielsr/CelebA-faces", "train", config, eval=True)

    # train_loader, validation_loader = create_lantent_dataloader_celebA(config)

    model = create_model(config)
    noise_scheduler = create_noise_scheduler(config)

    print_model_settings(model)
    print_noise_scheduler_settings(noise_scheduler)

    if config.low_rank_pretraining:
        model = low_rank_layer_replacement(model, percentage=config.low_rank_rank)
        print(f"number of parameters in model after compression is: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if config.load_pretrained_model:
        path = Path(__file__).parent.parent / config.pretrained_model_path
        print(f"Loading pretrained model from {path}")
        model.load_state_dict(torch.load(path))
        print("Loading complete")


 
    trainer = DiTTrainer(model, noise_scheduler, train_loader, validation_loader, config)
    trainer.train_loop()

    

    if config.low_rank_compression:

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        
        print(f"number of parameters in model: {count_parameters(model)}")
        model = apply_low_rank_compression(model, threshold=0.9)
        print(f"number of parameters in model after compression is: {count_parameters(model)}")
        config.num_epochs = 1000 # finetune for 5 epoch TODO: parameterise this.
        finetune_trainer = DiTTrainer(model, noise_scheduler, train_loader, validation_loader, config)
        finetune_trainer.train_loop()

        compressed_pipeline = DiTPipeline(
            transformer=model,
            scheduler=noise_scheduler,
            vae=trainer.vae.vae if config.vae else trainer.vae,
        )

        compressed_pipeline.enable_attention_slicing()
        fid_score = finetune_trainer.eval.compute_metrics(compressed_pipeline)
        print(f"FID Score: {fid_score}")


if __name__ == "__main__":
    main()
