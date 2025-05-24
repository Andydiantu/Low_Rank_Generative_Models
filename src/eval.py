import random
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os


class Eval: 
    def __init__(self, val_dataloader, config):
        self.val_dataloader = val_dataloader
        self.eval_dataset_size = config.eval_dataset_size
        self.eval_batch_size = config.eval_batch_size
        self.num_inference_steps = config.num_inference_steps
        self.guidance_scale = config.guidance_scale
        self.cfg_enabled = config.cfg_enabled
        self.setup_metrics()

    def setup_metrics(self):
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False)
        
        # Precompute real image features
        for batch in tqdm(self.val_dataloader, desc="Computing real features", disable= "SLURM_JOB_ID" in os.environ):
            real_images = batch["img"]
            # Convert from [-1, 1] to [0, 1] range for FID calculation
            real_images = (real_images + 1.0) / 2.0
            self.fid.update(real_images, real=True)

        print("Real features computed")

    def compute_metrics(self, pipeline, num_samples = 5000):
        # TODO: Make this conditional and parameterise the number of classes
        per_class = num_samples // 10
        labels = torch.arange(10).repeat(per_class)  
        labels = labels[torch.randperm(num_samples)]


        batch_num = num_samples // self.eval_batch_size
        for i in range(batch_num):
            batch_labels = labels[i*self.eval_batch_size:(i+1)*self.eval_batch_size]
            images = pipeline(
                class_labels = batch_labels.tolist(),
                num_inference_steps=self.num_inference_steps,
                output_type="numpy",
                guidance_scale=self.guidance_scale if self.cfg_enabled else None,
            ).images
            
            generated_images = torch.tensor(images)
            generated_images = generated_images.permute(0, 3, 1, 2)
            self.fid.update(generated_images, real=False)
            
        fid_score = self.fid.compute()
        self.fid.reset()
        return {"fid": fid_score.item()}

def plot_loss_curves(validation_epochs, train_loss, val_loss, ema_val_loss, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        validation_epochs: How often validation loss is calculated (every n epochs)
        train_loss: List of training losses (one per epoch)
        val_loss: List of validation losses (one every validation_epochs)
        ema_val_loss: List of EMA validation losses (one every validation_epochs)
        save_path: Optional path to save the plot
    """
        
    # Create x-axis values
    train_epochs = list(range(1, len(train_loss) + 1))
    val_epochs = list(range(validation_epochs, validation_epochs * len(val_loss) + 1, validation_epochs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, label='Training Loss')
    plt.plot(val_epochs, val_loss, label='Validation Loss')
    plt.plot(val_epochs, ema_val_loss, label='EMA Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=500)

    plt.close()
    