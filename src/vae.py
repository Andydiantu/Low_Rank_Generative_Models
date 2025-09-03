from diffusers import AutoencoderKL, DiTPipeline
import torch

#a wrapper for Stable Diffusion's VAEs 
class SD_VAE: 
    def __init__(self, device="cuda") -> None:
        self.device = device
        
        # Method 1: Load VAE directly (most memory efficient)
        try:
            # Try to load VAE component directly
            vae = AutoencoderKL.from_pretrained(
                "facebook/DiT-XL-2-256", 
                subfolder="vae",
                torch_dtype=torch.float32
            )
            print("✅ Loaded VAE directly from DiT-XL-2-256")
        except:
            # Method 2: Fallback - load from pipeline but clean up immediately
            print("⚠️  Direct VAE loading failed, using pipeline extraction...")
            pipe = DiTPipeline.from_pretrained(
                "facebook/DiT-XL-2-256", 
                torch_dtype=torch.float32
            )
            vae = pipe.vae
            # Delete the pipeline to free transformer memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("✅ Extracted VAE and cleaned up pipeline")
        
        vae.eval()
        vae = vae.to(device)
        self.vae = vae

    def __call__(self, x):
        # assumes 0-1 normalized image
        single_image = False
        if len(x.size()) == 3: 
            single_image = True
            x = x.unsqueeze(dim=0)
        
        if x.min() >= 0: 
            x = x * 2 - 1 

        # Ensure input is on the same device as the VAE
        x = x.to(self.device)

        with torch.no_grad():
            encode = self.vae.encode(x)
            batch = encode.latent_dist.sample() * self.vae.config.scaling_factor
        if single_image: 
            batch = batch.squeeze(dim=0)
        return batch

    def encode(self, x): 
        return self(x)
    
    def decode(self, z): 
        """
        returns: decode image"""
        if len(z.size()) == 3: 
            z = z.unsqueeze(dim=0)
        
        # Ensure latents are on the same device as the VAE  
        z = z.to(self.device)
        
        with torch.no_grad():
            x = self.vae.decode(z / self.vae.config.scaling_factor, return_dict=False)[0]

        x = ((1 + x) * 0.5).clip(0, 1)
        return x
    
    def to(self, device):
        """Move the VAE to the specified device"""
        self.device = device
        self.vae = self.vae.to(device)
        return self


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