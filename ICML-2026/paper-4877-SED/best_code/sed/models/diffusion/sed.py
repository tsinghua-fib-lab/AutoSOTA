import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import Adam

from sed.models.diffusion.diffusion import Diffusion
from sed.models.vae.svae import SVAE


class SED(L.LightningModule):
    def __init__(
        self,
        diffusion_model_config: dict,  # Configuration dict for diffusion model creation
        vae_dir: str                  # Path to a trained VAE checkpoint file
    ):
        super().__init__()
        self.save_hyperparameters()
        # Load pretrained VAE from checkpoint for encoding and decoding
        self.vae = SVAE.load_from_checkpoint(vae_dir)
        self.vae.freeze()  # Freeze VAE parameters to prevent their training

        # Build the diffusion model, matching latent dimension to VAE output
        self.diffusion_model = Diffusion(
            **diffusion_model_config, image_size=self.vae.d_model)

    def training_step(self, input, batch_idx):
        # Unpack input batch: in_positions, in_values may correspond to sequence features
        in_positions, in_values = input
        # Pass input through VAE, get encoded latent 'mu' (plus optionally other outputs)
        _, mu, _, tgt_y_positions, tgt_y_values = self.vae(
            in_positions, in_values)
        # Pass latent encoding through the diffusion model for prediction
        pred = self.diffusion_model(mu)
        # Compute mean squared error between prediction and latent representation
        loss = F.mse_loss(pred, mu)
        # Log training loss to Lightning progress bar/outputs
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer only updates diffusion model (UNet) parameters, not the VAE
        return Adam(self.diffusion_model.unet_model.parameters(),
                    lr=1e-4, betas=(0.9, 0.99))

    @torch.no_grad()
    def sample(self, batch_size=25):
        # Generation: sample latent vectors using the diffusion model
        sampled_z = self.diffusion_model.sample(batch_size)
        # Pass generated latents through VAE decoder to get output data (positions, values)
        out_positions, out_values = self.vae.sample(batch_size, sampled_z)
        return out_positions, out_values
