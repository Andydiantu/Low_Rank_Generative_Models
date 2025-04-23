import random
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset



class Eval: 
    def __init__(self, val_dataloader, eval_dataset_size = 10000):
        self.val_dataloader = val_dataloader
        self.eval_dataset_size = eval_dataset_size
        self.setup_metrics()

    def setup_metrics(self):
        self.fid = FrechetInceptionDistance(feature=64, normalize=True, input_img_size=(3, 32, 32))
        
        indices = random.sample(range(len(self.val_dataloader.dataset)), self.eval_dataset_size)
        subset_dataset = Subset(self.val_dataloader.dataset, indices)
        self.eval_dataloader = DataLoader(subset_dataset, batch_size=self.val_dataloader.batch_size, shuffle=False)

        # Precompute real image features
        for batch in tqdm(self.eval_dataloader, desc="Computing real features"):
            real_images = batch["img"]
            # Convert from [-1, 1] to [0, 1] range for FID calculation
            real_images = (real_images + 1.0) / 2.0
            self.fid.update(real_images, real=True)

    def compute_metrics(self, pipeline):

        batch_num = self.eval_dataset_size // 64
        for i in range(batch_num):
            images = pipeline(
                class_labels=torch.zeros(64, dtype=torch.long),
                num_inference_steps=1000,
                output_type="numpy"
            ).images
            generated_images = torch.tensor(images)
            generated_images = generated_images.permute(0, 3, 1, 2)

        self.fid.update(generated_images, real=False)
        fid_score = self.fid.compute()
        return {"fid": fid_score.item()}