from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch.nn.functional as F


class Eval: 
    def __init__(self, val_dataloader, config):
        self.val_dataloader = val_dataloader
        self.eval_dataset_size = config.eval_dataset_size
        self.eval_batch_size = config.eval_batch_size
        self.num_inference_steps = config.num_inference_steps
        self.guidance_scale = config.guidance_scale
        self.cfg_enabled = config.cfg_enabled
        self.setup_metrics(config.real_features_path)
    
    def resize_for_fid(self, images):
        """
        Resize images to 299x299 for FID calculation using bilinear interpolation.
        This is the standard size for Inception-v3 network used in FID.
        
        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 1]
        Returns:
            Resized images of shape (B, C, 299, 299)
        """
        return F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False, antialias=True)

    def setup_metrics(self, real_features_path):
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False)
        self.fid.set_dtype(torch.float64)
        # Move FID metric to GPU if available
        if torch.cuda.is_available():
            self.fid = self.fid.cuda()
            self.fid.set_dtype(torch.float64)
        
        real_features_path = Path(__file__).parent.parent / real_features_path

        if real_features_path.exists():
            self.fid = torch.load(real_features_path, weights_only=False, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            print(self.fid.real_features_num_samples)
            print("Real features loaded")
            self.fid = self.fid.cuda()
            self.fid.set_dtype(torch.float64)

        else:
            print(self.val_dataloader)
            # Precompute real image features
            for batch in tqdm(self.val_dataloader, desc="Computing real features", disable= "SLURM_JOB_ID" in os.environ):
                real_images = batch["img"]
                real_images = real_images.to(self.fid.device)
                
                # Convert from [-1, 1] to [0, 1] range for FID calculation
                real_images = (real_images + 1.0) / 2.0
                
                # Resize to 299x299 for proper FID calculation
                real_images = self.resize_for_fid(real_images)
                
                self.fid.update(real_images, real=True)
            
            real_features_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.fid, real_features_path)
            print(self.fid.real_features_num_samples)
            print("Real features computed")

        

    def _extract_inception_features_from_real(self, num_samples: int) -> torch.Tensor:
        """
        Extract Inception-v3 features for a subset of real images for use in precision/recall.
        Returns tensor of shape (N, 2048) on the same device as the FID inception network.
        """
        device = next(self.fid.inception.parameters()).device
        collected_features = []
        collected = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                real_images = batch["img"]
                # Real images come in [-1, 1]; convert to [0, 255] uint8 as expected by inception wrapper
                real_images_uint8 = (((real_images + 1.0) / 2.0) * 255.0).clamp(0, 255).to(torch.uint8).to(device)
                feats = self.fid.inception(real_images_uint8)
                collected_features.append(feats)
                collected += feats.shape[0]
                if collected >= num_samples:
                    break
        return torch.cat(collected_features, dim=0)[:num_samples]

    def _generate_images_and_extract_features(self, pipeline, labels: torch.Tensor) -> torch.Tensor:
        """
        Use the diffusion pipeline to generate images and extract Inception-v3 features.
        Returns tensor of shape (N, 2048) on the same device as the FID inception network.
        """
        device = next(self.fid.inception.parameters()).device
        generated_features = []
        batch_num = labels.shape[0] // self.eval_batch_size
        with torch.no_grad():
            for i in range(batch_num):
                batch_labels = labels[i * self.eval_batch_size : (i + 1) * self.eval_batch_size]
                images = pipeline(
                    class_labels=batch_labels.tolist(),
                    num_inference_steps=self.num_inference_steps,
                    output_type="numpy",
                    guidance_scale=self.guidance_scale if self.cfg_enabled else 1,
                ).images

                generated_images = torch.tensor(images)
                generated_images = generated_images.permute(0, 3, 1, 2)
                # Convert [0,1] float to [0,255] uint8 for inception wrapper
                gen_uint8 = (generated_images * 255.0).clamp(0, 255).to(torch.uint8).to(device)
                feats = self.fid.inception(gen_uint8)
                generated_features.append(feats)
        return torch.cat(generated_features, dim=0)

    def _compute_precision_recall_from_features(self, real_feats: torch.Tensor, gen_feats: torch.Tensor, k: int = 3):
        """
        Compute Improved Precision and Recall (Kynkäänniemi et al.) from feature tensors.
        Returns (precision, recall).
        """
        # Use float32 for distance computations
        real_feats = real_feats.float()
        gen_feats = gen_feats.float()

        # Compute k-NN radii within each set (exclude self by setting diag to +inf)
        real_real = torch.cdist(real_feats, real_feats, p=2)
        real_real.fill_diagonal_(float("inf"))
        real_radii = torch.kthvalue(real_real, k=k, dim=1).values  # (R,)

        gen_gen = torch.cdist(gen_feats, gen_feats, p=2)
        gen_gen.fill_diagonal_(float("inf"))
        gen_radii = torch.kthvalue(gen_gen, k=k, dim=1).values  # (G,)

        # Distances between sets
        d_gr = torch.cdist(gen_feats, real_feats, p=2)  # (G, R)

        # Precision: fraction of generated samples inside real manifold
        dmin_gr, nn_real_idx = torch.min(d_gr, dim=1)  # (G,), (G,)
        precision_mask = dmin_gr <= real_radii[nn_real_idx]
        precision = precision_mask.float().mean().item()

        # Recall: fraction of real samples inside generated manifold
        d_rg = d_gr.T  # (R, G)
        dmin_rg, nn_gen_idx = torch.min(d_rg, dim=1)  # (R,), (R,)
        recall_mask = dmin_rg <= gen_radii[nn_gen_idx]
        recall = recall_mask.float().mean().item()

        return precision, recall

    def compute_metrics(self, pipeline, num_samples = 5000):
        # TODO: Make this conditional and parameterise the number of classes
        if self.cfg_enabled:
            per_class = num_samples // 10
            labels = torch.arange(10).repeat(per_class)  
            labels = labels[torch.randperm(num_samples)]
        else:
            labels = torch.zeros(num_samples, dtype=torch.long, device=self.fid.device)

        # 1) FID using cached real stats
        batch_num = num_samples // self.eval_batch_size
        for i in range(batch_num):
            batch_labels = labels[i * self.eval_batch_size : (i + 1) * self.eval_batch_size]
            images = pipeline(
                class_labels=batch_labels.tolist(),
                num_inference_steps=self.num_inference_steps,
                output_type="numpy",
                guidance_scale=self.guidance_scale if self.cfg_enabled else 1,
            ).images

            generated_images = torch.tensor(images)
            generated_images = generated_images.permute(0, 3, 1, 2)
            generated_images = generated_images.to(device=self.fid.device, dtype=next(self.fid.inception.parameters()).dtype)

            # Resize to 299x299 for proper FID calculation
            generated_images = self.resize_for_fid(generated_images)

            self.fid.update(generated_images, real=False)

        fid_score = self.fid.compute().item()
        self.fid.reset()

        # 2) Precision/Recall using kNN over features
        # Recreate labels (since we consumed above) and extract features
        if self.cfg_enabled:
            per_class = num_samples // 10
            labels_pr = torch.arange(10).repeat(per_class)
            labels_pr = labels_pr[torch.randperm(num_samples)].to(self.fid.device)
        else:
            labels_pr = torch.zeros(num_samples, dtype=torch.long, device=self.fid.device)

        real_feats = self._extract_inception_features_from_real(num_samples)
        gen_feats = self._generate_images_and_extract_features(pipeline, labels_pr)
        precision, recall = self._compute_precision_recall_from_features(real_feats, gen_feats, k=3)

        return {"fid": fid_score, "precision": precision, "recall": recall}

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
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=500)

    plt.close()
    