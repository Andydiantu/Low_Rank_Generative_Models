from diffusers import AutoencoderKL
import torch

#a wrapper for Stable Diffusion's VAEs 
class SD_VAE: 
    def __init__(self, device="cuda") -> None:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema"
        )
        vae = vae.to(device)
        self.vae = vae
        self.device = device

    def __call__(self, x):
        # assumes normalized image
        single_image = False
        if len(x.size()) == 3: 
            single_image = True
            x = x.unsqueeze(dim=0)
        
        # Images from preprocessing are already normalized to [-1,1]
        # Don't apply double normalization

        with torch.no_grad():
            encode = self.vae.encode(x.to(self.device))
            batch = encode.latent_dist.sample() * self.vae.config.scaling_factor
        if single_image: 
            batch = batch.squeeze(dim=0)
        return batch
    
    def encode(self, x):
        # This needs to match the interface expected in DiT_trainer.py
        with torch.no_grad():
            return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
    
    def decode(self, z): 
        """
        returns: decode image"""
        if len(z.size()) == 3: 
            z = z.unsqueeze(dim=0)
        with torch.no_grad():
            x = self.vae.decode(z / self.vae.config.scaling_factor, return_dict=False)[0]
            #x = self.vae.decode(z , return_dict=False)[0]

        x = ((1 + x) * 0.5).clip(0, 1)
        return x
    
    
class IdentityVAE(torch.nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z

    def forward(self, x):
        return x


# A dummy implementation of a identity VAE, as DIT pipeline from diffusers package require a placeholder of VAE.
class DummyAutoencoderKL(AutoencoderKL):
    def __init__(self):
        super().__init__()
        self.encoder = IdentityVAE()
        self.decoder = IdentityVAE()
        self.config.scaling_factor = 1.0

    def encode(self, x):
        return type(
            "DummyOutput",
            (object,),
            {"latent_dist": type("DummyLatentDist", (object,), {"sample": lambda: x})},
        )()

    def decode(self, z):
        return type("DummyOutput", (object,), {"sample": z})

    def forward(self, x):
        return x