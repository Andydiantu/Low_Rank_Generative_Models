from diffusers import AutoencoderKL, VQModel
import torch

#a wrapper for Stable Diffusion's VAEs 
class SD_VAE: 
    def __init__(self, device="cuda") -> None:
        vae = AutoencoderKL.from_pretrained("tpremoli/MLD-CelebA-128-80k", subfolder="vae")
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

        with torch.no_grad():
            encode = self.vae.encode(x.cuda())
            batch = encode.latent_dist.sample() *  self.vae.config.scaling_factor
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
        with torch.no_grad():
            x = self.vae.decode(z / self.vae.config.scaling_factor, return_dict=False)[0]

        x = ((1 + x) * 0.5).clip(0, 1)
        return x
    
    def to(self, device):
        """Move the VAE to the specified device"""
        self.vae = self.vae.to(device)
        return self


# a wrapper for VQModel to work with DiTPipeline
class VQ_VAE:
    def __init__(self, model_id="CompVis/ldm-celebahq-256", device="cuda") -> None:
        vae = VQModel.from_pretrained(model_id, subfolder="vqvae")
        vae.eval()
        vae = vae.to(device)
        self.vae = vae
        
        # VQModel doesn't have a built-in scaling factor, so we need to check
        # Check if VQModel has a config with scaling_factor
        if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'scaling_factor'):
            self.scaling_factor = self.vae.config.scaling_factor
        else:
            # Default scaling factor commonly used for VQ models
            # You may need to tune this based on your specific model
            self.scaling_factor = 1.0
            print(f"Warning: VQModel doesn't have scaling_factor, using default: {self.scaling_factor}")

    def __call__(self, x):
        # assumes 0-1 normalized image
        single_image = False
        if len(x.size()) == 3: 
            single_image = True
            x = x.unsqueeze(dim=0)
        
        if x.min() >= 0: 
            x = x * 2 - 1 

        with torch.no_grad():
            encoder_output = self.vae.encode(x.cuda())
            batch = encoder_output.latents * self.scaling_factor  # Apply scaling factor
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
        with torch.no_grad():
            decoder_output = self.vae.decode(z / self.scaling_factor)  # Apply inverse scaling
            x = decoder_output.sample  # VQModel returns DecoderOutput with .sample

        x = ((1 + x) * 0.5).clip(0, 1)
        return x
    
    def to(self, device):
        """Move the VAE to the specified device"""
        self.vae = self.vae.to(device)
        return self
    
    
class IdentityVAE(torch.nn.Module):
    def encode(self, x):
        return x

    def decode(self, x):
        return x

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