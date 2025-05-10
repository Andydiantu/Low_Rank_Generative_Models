import random
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset



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
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
        
        indices = random.sample(range(len(self.val_dataloader.dataset)), self.eval_dataset_size)
        subset_dataset = Subset(self.val_dataloader.dataset, indices)
        self.eval_dataloader = DataLoader(subset_dataset, batch_size=self.val_dataloader.batch_size, shuffle=False)

        # Precompute real image features
        for batch in tqdm(self.eval_dataloader, desc="Computing real features"):
            real_images = batch["img"]
            # Convert from [-1, 1] to [0, 1] range for FID calculation
            # real_images = (real_images + 1.0) / 2.0
            self.fid.update(real_images, real=True)

    def compute_metrics(self, pipeline,):

        batch_num = self.eval_dataset_size // self.eval_batch_size
        for i in range(batch_num):
            images = pipeline(
                class_labels = torch.randint(0, 10, (self.eval_batch_size,), dtype=torch.long),
                num_inference_steps=self.num_inference_steps,
                output_type="numpy",
                guidance_scale=self.guidance_scale if self.cfg_enabled else None,
            ).images
            
            generated_images = torch.tensor(images)
            generated_images = generated_images.permute(0, 3, 1, 2)
            self.fid.update(generated_images, real=False)
            
        fid_score = self.fid.compute()
        return {"fid": fid_score.item()}