import torch
import torch.nn as nn
import torch.nn.functional as F

from diffig.mar_vae.discriminator import NLayerDiscriminator, weights_init
from diffig.mar_vae.lpips import LPIPS


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class KLVAELoss(nn.Module):
    def __init__(self, 
                 disc_start=50001,
                 kl_weight=1.0e-06,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3, 
                 disc_factor=1.0,
                 disc_weight=0.5,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_ndf=64,
                 disc_loss="hinge"):
        super().__init__()
        
        assert disc_loss in ["hinge", "vanilla"]
        
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize LPIPS
        self.perceptual_loss = LPIPS().eval()
        
        # Initialize discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)
        
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        
        print(f"KLVAELoss running with {disc_loss} loss.")
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]
        
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def kl_loss(self, posterior):
        """
        Calculate KL divergence loss
        posterior: DiagonalGaussianDistribution object from VAE encoder
        """
        kl = posterior.kl()
        kl = torch.sum(kl) / kl.shape[0]
        return kl
    
    def forward(self, inputs, reconstructions, posterior, optimizer_idx,
                global_step, last_layer=None, split="train"):
        """
        inputs: original images
        reconstructions: reconstructed images from VAE
        posterior: DiagonalGaussianDistribution from encoder
        optimizer_idx: 0 for VAE, 1 for discriminator
        global_step: current training step
        last_layer: last layer of decoder for adaptive weight calculation
        """
        
        # Reconstruction loss
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        
        nll_loss = self.pixel_weight * torch.mean(rec_loss)
        
        # KL loss
        kl_loss = self.kl_loss(posterior)
        kl_loss = self.kl_weight * kl_loss
        
        # GAN part
        if optimizer_idx == 0:
            # Generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            loss = nll_loss + kl_loss + d_weight * disc_factor * g_loss
            
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            
            return loss, log
        
        if optimizer_idx == 1:
            # Discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            
            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            
            return d_loss, log