from torchmetrics.image.fid import FrechetInceptionDistance

def setup_metrics(self):
    self.fid = FrechetInceptionDistance(feature=64)
    
    # Precompute real image features
    real_features = []
    for batch in tqdm(self.val_dataloader, desc="Computing real features"):
        real_images = batch["img"]
        # Convert from [-1, 1] to [0, 1] range for FID calculation
        real_images = (real_images + 1.0) / 2.0
        self.fid.update(real_images, real=True)

def compute_metrics(self, generated_images):
    # Convert from PIL to tensor
    gen_tensors = torch.stack([transforms.ToTensor()(img) for img in generated_images])
    self.fid.update(gen_tensors, real=False)
    fid_score = self.fid.compute()
    return {"fid": fid_score.item()}