"""
OVLR Example: GAN Training with Output-Level Perturbation

OVLR improves GAN training by smoothing the generator output and providing
more informative gradients than standard backpropagation, especially when
the discriminator saturates.

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


class Generator(nn.Module):
    """Standard DCGAN generator."""
    def __init__(self, latent_dim=100, img_size=32, channels=1):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """Standard DCGAN discriminator."""
    def __init__(self, img_size=32, channels=1):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def generator_loss_ce(fake_output, _):
    """Generator loss: BCE with reversed labels (for OVLR)."""
    return -torch.log(fake_output.clamp(min=1e-8)).squeeze()


def train_gan_bp(args, generator, discriminator, dataloader, device):
    """Train GAN with standard backpropagation."""
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    adversarial_loss = nn.BCELoss()

    os.makedirs('results_gan/bp', exist_ok=True)

    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"[BP] Epoch {epoch:3d} | Batch {i:4d} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

        save_image(gen_imgs.data[:25], f"results_gan/bp/epoch_{epoch}.png", nrow=5, normalize=True)

    return generator


def train_gan_ovlr(args, generator, discriminator, dataloader, device):
    """Train GAN generator with OVLR, discriminator with BP."""
    from ovlr import OVLRGradientEstimator, get_noise_fn

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    adversarial_loss = nn.BCELoss()

    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

    os.makedirs('results_gan/ovlr', exist_ok=True)

    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Train Discriminator with BP
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator with OVLR
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)
            d_out = discriminator(gen_imgs)

            # OVLR computes gradient through the generator output
            valid_labels = torch.ones_like(d_out, device=device)
            g_loss = estimator(d_out, valid_labels, generator_loss_ce, loss_fn_reduction='mean')
            optimizer_G.step()

            if i % 100 == 0:
                print(f"[OVLR] Epoch {epoch:3d} | Batch {i:4d} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

        save_image(gen_imgs.data[:25], f"results_gan/ovlr/epoch_{epoch}.png", nrow=5, normalize=True)

    return generator


def main():
    parser = argparse.ArgumentParser(description='OVLR: GAN Training')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--latent-dim', type=int, default=100, help='latent dimension')
    parser.add_argument('--img-size', type=int, default=32, help='image size')
    parser.add_argument('--n-repeat', type=int, default=50, help='OVLR repeat count')
    parser.add_argument('--noise-scale', type=float, default=0.3, help='OVLR noise scale')
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
    print(f"GAN Training with OVLR (n_repeat={args.n_repeat}, sigma={args.noise_scale})")
    print("=" * 60 + "\n")

    if args.method in ['bp', 'both']:
        print("Training GAN with standard BP...")
        generator_bp = Generator(args.latent_dim, args.img_size, channels).to(device)
        discriminator_bp = Discriminator(args.img_size, channels).to(device)
        train_gan_bp(args, generator_bp, discriminator_bp, dataloader, device)
        print()

    if args.method in ['ovlr', 'both']:
        print("Training GAN with OVLR (generator)...")
        generator_ovlr = Generator(args.latent_dim, args.img_size, channels).to(device)
        discriminator_ovlr = Discriminator(args.img_size, channels).to(device)
        train_gan_ovlr(args, generator_ovlr, discriminator_ovlr, dataloader, device)

    print("\nTraining complete! Generated images saved to results_gan/")


if __name__ == '__main__':
    main()
