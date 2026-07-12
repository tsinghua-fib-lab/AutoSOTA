"""
OVLR Example: VAE Training

Demonstrates Variational Autoencoder (VAE) training with OVLR gradient
estimation for the reconstruction loss.

Reference:
    OVLR: Efficient, Scalable, and Robust Training via
    Output-Level Variance-Reduced Likelihood Ratio
    ICML 2026
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    """Variational Autoencoder."""
    def __init__(self, img_size=32, channels=1, latent_dim=32):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Encoder output: mu and logvar
        fc_size = img_size // 8
        self.fc_mu = nn.Linear(128 * fc_size * fc_size, latent_dim)
        self.fc_logvar = nn.Linear(128 * fc_size * fc_size, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * fc_size * fc_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        fc_size = self.img_size // 8
        h = self.decoder_fc(z)
        h = h.view(-1, 128, fc_size, fc_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def recon_loss_mse(recon, x):
    """MSE reconstruction loss for OVLR."""
    return ((recon - x) ** 2).flatten(1).mean(1)


def loss_function_bp(recon_x, x, mu, logvar):
    """Standard VAE loss: MSE recon + KL divergence."""
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return BCE + KLD, BCE, KLD


def train_vae_bp(args, model, dataloader, device):
    """Train VAE with standard backprop."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs('results_vae/bp', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(x)
            loss, bce, kld = loss_function_bp(recon_batch, x, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if i % 100 == 0:
                print(f"[BP] Epoch {epoch:3d} | Batch {i:4d} | "
                      f"Loss: {loss.item():.4f} | BCE: {bce.item():.4f} | KLD: {kld.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[BP] ===> Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")

        # Save samples
        with torch.no_grad():
            z = torch.randn(64, args.latent_dim).to(device)
            sample = model.decode(z)
            save_image(sample, f"results_vae/bp/sample_epoch_{epoch}.png", nrow=8, normalize=True)

    return model


def train_vae_ovlr(args, model, dataloader, device):
    """Train VAE with OVLR for reconstruction loss (KL term still uses BP)."""
    from ovlr import OVLRGradientEstimator, get_noise_fn

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

    os.makedirs('results_vae/ovlr', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.size(0)
            optimizer.zero_grad()

            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)

            # OVLR for reconstruction
            recon_batch = model.decode(z)
            recon_loss = estimator(recon_batch, x, recon_loss_mse, loss_fn_reduction='mean')

            # KL divergence (analytical with BP)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

            loss = recon_loss + kld
            total_loss += loss.item()

            # KL already has gradients, recon gradients are applied by OVLR
            kld.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"[OVLR] Epoch {epoch:3d} | Batch {i:4d} | "
                      f"Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KLD: {kld.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[OVLR] ===> Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")

        # Save samples
        with torch.no_grad():
            z = torch.randn(64, args.latent_dim).to(device)
            sample = model.decode(z)
            save_image(sample, f"results_vae/ovlr/sample_epoch_{epoch}.png", nrow=8, normalize=True)

    return model


def main():
    parser = argparse.ArgumentParser(description='OVLR: VAE Training')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--latent-dim', type=int, default=32, help='latent dimension')
    parser.add_argument('--img-size', type=int, default=32, help='image size')
    parser.add_argument('--n-repeat', type=int, default=20, help='OVLR repeat count')
    parser.add_argument('--noise-scale', type=float, default=0.1, help='OVLR noise scale')
    parser.add_argument('--method', type=str, default='both', choices=['bp', 'ovlr', 'both'],
                        help='training method(s) to run')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'],
                        help='dataset to use')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        channels = 1
    else:
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        channels = 1

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    print("\n" + "=" * 60)
    print(f"VAE Training with OVLR (n_repeat={args.n_repeat}, sigma={args.noise_scale})")
    print("=" * 60 + "\n")

    if args.method in ['bp', 'both']:
        print("Training VAE with standard BP...")
        model_bp = VAE(args.img_size, channels, args.latent_dim).to(device)
        train_vae_bp(args, model_bp, dataloader, device)
        print()

    if args.method in ['ovlr', 'both']:
        print("Training VAE with OVLR (reconstruction)...")
        model_ovlr = VAE(args.img_size, channels, args.latent_dim).to(device)
        train_vae_ovlr(args, model_ovlr, dataloader, device)

    print("\nTraining complete! Generated images saved to results_vae/")


if __name__ == '__main__':
    main()
