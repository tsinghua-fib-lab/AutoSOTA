import os
from abc import ABC, abstractmethod

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents


class VAEWrapper(ABC):
    """Abstract base class for VAE wrappers."""
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space.
        
        Args:
            x: Input image tensor [B, C, H, W] in [0, 1] range
            
        Returns:
            z: Latent tensor
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space.
        
        Args:
            z: Latent tensor
            
        Returns:
            x: Reconstructed image [B, C, H, W] in [0, 1] range
        """
        pass


class TransformWrapper:
    def __init__(self, vae, preprocess_fn, undo_preprocess_fn):
        self.vae = vae
        self.preprocess_fn = preprocess_fn
        self.undo_preprocess_fn = undo_preprocess_fn

    def encode(self, x):
        return self.vae.encode(self.undo_preprocess_fn(x))

    def decode(self, z):
        return self.preprocess_fn(self.vae.decode(z))


class MARVAEWrapper(VAEWrapper):
    """Wrapper for MAR VAE from cleanig.mar_vae."""
    
    def __init__(self, vae_model, device):
        """
        Args:
            vae_model: MAR VAE model instance
            use_mean: If True, use mean of posterior distribution (no sampling)
        """
        self.vae = vae_model
        self.device = device
        
        # MAR VAE normalization parameters (maps [0,1] to [-1,1])
        self.vae_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.vae_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using MAR VAE.
        
        MAR VAE returns a DiagonalGaussianDistribution, so we sample or use mean.
        """
        # MAR VAE expects input normalized with mean=0.5, std=0.5 (maps [0,1] to [-1,1])
        x_norm = (x - self.vae_mean) / self.vae_std
        posterior = self.vae.encode(x_norm)
        z = posterior.mean       
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode using MAR VAE."""
        
        # You can use torch.set_grad_enabled to respect z.requires_grad dynamically
        with torch.set_grad_enabled(z.requires_grad):
            x_norm = self.vae.decode(z)
            
        # Denormalize from [-1,1] to [0,1]
        x = x_norm * self.vae_std + self.vae_mean
        return x


class StableDiffusionVAEWrapper(VAEWrapper):
    """Wrapper for Stable Diffusion VAE from diffusers."""

    def __init__(self, vae_model, device):
        """
        Args:
            vae_model: MAR VAE model instance
            use_mean: If True, use mean of posterior distribution (no sampling)
        """
        self.vae = vae_model
        self.device = device
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using SD VAE.
        
        SD VAE expects input in [-1, 1] range.
        """
        # Convert from [0, 1] to [-1, 1]
        x = x.to(self.device, dtype=self.vae.dtype)
        x_norm = 2.0 * x - 1.0
        latent = self.vae.encode(x_norm).latent_dist.mean            
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode using SD VAE."""
        with torch.set_grad_enabled(z.requires_grad):
            x = self.vae.decode(z).sample
            
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        return x


class KandinskyVAEWrapper(VAEWrapper):
    def __init__(self, vae_model, device):
        """
        Args:
            vae_model: MAR VAE model instance
            use_mean: If True, use mean of posterior distribution (no sampling)
        """
        self.vae = vae_model
        self.device = device
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using SD VAE.
        
        SD VAE expects input in [-1, 1] range.
        """
        x = x.to(self.device, dtype=self.vae.dtype)
        x_norm = 2.0 * x - 1.0
        latents = retrieve_latents(self.vae.encode(x_norm))
        return latents

    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        with torch.set_grad_enabled(z.requires_grad):
            recon = self.vae.decode(
                z.to(self.vae.dtype), force_not_quantize=True, return_dict=False
            )[0]
        x = (recon + 1.0) / 2.0
        return x


def create_vae(vae_type, preprocess_fn, undo_preprocess_fn, device):
    """Factory function to create VAE wrapper.
    
    Args:
        vae_type: Type of VAE ('mar', 'sd', 'hf', or HuggingFace repo ID)
        **kwargs: Additional arguments for the wrapper
        
    Returns:
        VAEWrapper instance
    """
    if vae_type == "mar":

        from cleanig.mar_vae.vae import AutoencoderKL

        vae_model = AutoencoderKL(
            embed_dim=16,
            ch_mult=[1, 1, 2, 2, 4],
        ).to(device)
        vae_model.eval()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = torch.load(os.path.join(base_dir, "mar_vae/kl16.ckpt"), map_location="cpu")
        vae_model.load_state_dict(checkpoint['model'])
        # Assume vae_model is passed in kwargs
        vae = MARVAEWrapper(vae_model, device)
    elif vae_type in ["sd1", "sd2"]:

        from diffusers import AutoencoderKL

        if vae_type == "sd1":
            repo_id = "CompVis/stable-diffusion-v1-1"
        elif vae_type == "sd2":
            repo_id = "lzyvegetable/stable-diffusion-2-1"
        vae_model = AutoencoderKL.from_pretrained(
            repo_id,
            subfolder="vae",
            torch_dtype=torch.float32
        ).to(device)
        vae_model.eval()
        vae = StableDiffusionVAEWrapper(vae_model, device)
    elif vae_type == "kd":

        from diffusers import AutoPipelineForImage2Image

        repo_id = "kandinsky-community/kandinsky-2-1"

        pipe = AutoPipelineForImage2Image.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
        )
        vae_model = pipe.movq
        vae_model.to(device)
        vae = KandinskyVAEWrapper(vae_model, device)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    return TransformWrapper(vae, preprocess_fn, undo_preprocess_fn)