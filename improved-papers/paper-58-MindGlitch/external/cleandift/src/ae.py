import diffusers
import torch
from torch import nn

class AutoencoderKL(nn.Module):
    def __init__(self, scale: float = 0.18215, shift: float = 0.0, repo="Charles-Elena/stable-diffusion-2-1"):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.ae = diffusers.AutoencoderKL.from_pretrained(repo, subfolder="vae")
        self.ae.eval()
        self.ae.compile()
        self.ae.requires_grad_(False)

    def forward(self, img):
        return self.encode(img)

    @torch.no_grad()
    def encode(self, img):
        latent = self.ae.encode(img, return_dict=False)[0].sample()
        return (latent - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, latent):
        rec = self.ae.decode(latent / self.scale + self.shift, return_dict=False)[0]
        return rec