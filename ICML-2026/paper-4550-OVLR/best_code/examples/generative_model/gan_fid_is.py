"""
OVLR Example: GAN with FID and Inception Score Evaluation.

This is a complete implementation of GAN training with:
- DCGAN-style Generator and Discriminator
- FID (Fréchet Inception Distance) evaluation
- IS (Inception Score) evaluation
- Periodic evaluation during training
- Visualization of generated samples and metric curves

Reference:
    OVLR: Efficient, Scalable, and Robust Training via
    Output-Level Variance-Reduced Likelihood Ratio
    ICML 2026
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy import linalg
import pickle
import math
from torchvision.models import inception_v3


# Config
class Config:
    batch_size = 128
    epochs = 200
    lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    latent_dim = 100
    img_size = 64
    channels = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_interval = 500
    eval_interval = 1000
    n_fid_samples = 5000
    n_is_samples = 5000
    n_is_splits = 10


config = Config()
print(f"Using device: {config.device}")


# Data Loading
def get_dataloader(dataset_name='mnist', shuffle=True, drop_last=None):
    """Load datasets: mnist, fashion_mnist, cifar10."""
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
        config.channels = 3
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root='./data', download=True, transform=transform)
        config.channels = 1
    elif dataset_name == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root='./data', download=True, transform=transform)
        config.channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=2,
        drop_last=shuffle if drop_last is None else drop_last
    )

    return dataloader


# ===================== DCGAN Models =====================

class Generator(nn.Module):
    """DCGAN-style Generator."""
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = config.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(config.latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, config.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """DCGAN-style Discriminator."""
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(config.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = config.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# ===================== FID and IS Evaluation =====================

class InceptionV3ForEvaluation(nn.Module):
    """InceptionV3 for FID and Inception Score calculation."""

    def __init__(self, device='cuda'):
        super(InceptionV3ForEvaluation, self).__init__()

        # Load pre-trained Inception v3
        try:
            from torchvision.models import Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        except Exception:
            self.inception = inception_v3(pretrained=True, transform_input=False)

        # Keep fc for IS, replace with identity for feature extraction
        self.fc = self.inception.fc
        self.inception.fc = nn.Identity()

        self.device = device
        self.to(device)
        self.eval()

    def forward(self, x, return_features=True, return_probs=False):
        x = x.to(self.device)
        x = self.preprocess(x)

        with torch.no_grad():
            features = self.inception(x)

        results = {}
        if return_features:
            results['features'] = features
        if return_probs:
            logits = self.fc(features)
            probs = F.softmax(logits, dim=1)
            results['probs'] = probs

        return results

    def preprocess(self, x):
        """Preprocess: [-1, 1] -> [0, 1] -> resize -> ImageNet normalization."""
        x = (x + 1) / 2.0  # to [0, 1]

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        x = (x - mean) / std

        return x


def calculate_fid(real_features, fake_features):
    """Calculate FID score."""
    if len(real_features) == 0 or len(fake_features) == 0:
        return float('inf')

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        if not np.isfinite(covmean).all():
            return float('inf')

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_inception_score(probs, splits=10):
    """Calculate Inception Score."""
    if len(probs) == 0:
        return 0, 0

    n_samples = len(probs)
    if n_samples < splits:
        splits = max(1, n_samples // 100)

    scores = []
    n_part = n_samples // splits

    for i in range(splits):
        start_idx = i * n_part
        end_idx = start_idx + n_part
        if i == splits - 1:
            end_idx = n_samples

        part = probs[start_idx:end_idx]
        py = part.mean(0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl = kl.sum(1)
        scores.append(np.exp(np.mean(kl)))

    return np.mean(scores), np.std(scores)


def evaluate_gan(generator, dataloader, n_fid_samples=1000, n_is_samples=1000, splits=10, device='cuda'):
    """Evaluate GAN with FID and Inception Score."""
    print(f"\nEvaluating GAN...")
    print(f"FID samples: {n_fid_samples}, IS samples: {n_is_samples}")

    inception_model = InceptionV3ForEvaluation(device=device)
    inception_model.eval()

    try:
        # Collect real image features
        print("\n1. Extracting real image features...")
        real_features = []
        with torch.no_grad():
            for batch, _ in tqdm(dataloader, desc="Real images"):
                batch = batch.to(device)
                results = inception_model(batch, return_features=True, return_probs=False)
                real_features.append(results['features'].cpu().numpy())
                if sum(f.shape[0] for f in real_features) >= n_fid_samples:
                    break

        real_features = np.concatenate(real_features)[:n_fid_samples] if real_features else np.array([])

        # Collect generated image features and probs
        print("\n2. Generating images and extracting features...")
        fake_features = []
        fake_probs = []

        with torch.no_grad():
            n_batches = math.ceil(max(n_fid_samples, n_is_samples) / config.batch_size)
            for i in tqdm(range(n_batches), desc="Generated images"):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake_imgs = generator(z)
                results = inception_model(fake_imgs, return_features=True, return_probs=True)

                fake_features.append(results['features'].cpu().numpy())
                fake_probs.append(results['probs'].cpu().numpy())

        fake_features = np.concatenate(fake_features) if fake_features else np.array([])
        fake_probs = np.concatenate(fake_probs) if fake_probs else np.array([])

        if len(fake_features) > n_fid_samples:
            fake_features = fake_features[:n_fid_samples]
        if len(fake_probs) > n_is_samples:
            fake_probs = fake_probs[:n_is_samples]

        if len(real_features) > len(fake_features):
            real_features = real_features[:len(fake_features)]

        # Calculate FID
        print("\n3. Calculating FID score...")
        fid_score = calculate_fid(real_features, fake_features) if len(real_features) > 10 else float('inf')
        print(f"FID Score: {fid_score:.4f}")

        # Calculate IS
        print("\n4. Calculating Inception Score...")
        is_mean, is_std = calculate_inception_score(fake_probs, splits=splits) if len(fake_probs) > 100 else (0, 0)
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

        # Save results
        os.makedirs('results/gan/evaluation', exist_ok=True)

        results = {
            'fid': fid_score,
            'is_mean': is_mean,
            'is_std': is_std,
            'n_fid_samples': min(len(real_features), len(fake_features)),
            'n_is_samples': len(fake_probs),
        }

        with open('results/gan/evaluation/latest_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        return fid_score, (is_mean, is_std)

    except Exception as e:
        print(f"Evaluation error: {e}")
        return float('inf'), (0, 0)


# ===================== Training =====================

def train_gan_with_evaluation(generator, discriminator, train_loader, eval_loader, epochs):
    """Train GAN with periodic FID/IS evaluation."""
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    adversarial_loss = nn.BCELoss()

    os.makedirs('results/gan', exist_ok=True)
    os.makedirs('results/gan/evaluation', exist_ok=True)

    fid_history = []
    is_history = []
    iteration = 0

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            real_imgs = real_imgs.to(config.device)

            valid = torch.ones((real_imgs.size(0), 1), requires_grad=False).to(config.device)
            fake = torch.zeros((real_imgs.size(0), 1), requires_grad=False).to(config.device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.size(0), config.latent_dim).to(config.device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            iteration += 1

            # Save samples
            if i % config.save_interval == 0 or (i == len(train_loader) - 1 and epoch % 5 == 0):
                save_image(gen_imgs.data[:25], f"results/gan/epoch_{epoch+1}_batch_{i}.pdf",
                          nrow=5, normalize=True)

            # Periodic evaluation
            if (iteration % config.eval_interval == 0 and iteration > 0) or \
               (i == len(train_loader) - 1 and epoch % 5 == 0 and epoch > 0):
                print(f"\nEvaluating at iteration {iteration}...")
                try:
                    fid_score, is_scores = evaluate_gan(
                        generator, eval_loader,
                        n_fid_samples=500, n_is_samples=1000, splits=5, device=config.device
                    )
                    fid_history.append((iteration, fid_score))
                    is_history.append((iteration, is_scores[0], is_scores[1]))

                    if fid_history:
                        np.save('results/gan/evaluation/fid_history.npy', np.array(fid_history))
                    if is_history:
                        np.save('results/gan/evaluation/is_history.npy', np.array(is_history))

                    print(f"FID: {fid_score:.4f}, IS: {is_scores[0]:.4f} ± {is_scores[1]:.4f}")
                except Exception as e:
                    print(f"Evaluation skipped: {e}")
                    continue

    torch.save(generator.state_dict(), 'results/gan/generator_final.pth')
    torch.save(discriminator.state_dict(), 'results/gan/discriminator_final.pth')

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_fid, final_is = evaluate_gan(
        generator, eval_loader,
        n_fid_samples=min(config.n_fid_samples, 2000),
        n_is_samples=min(config.n_is_samples, 2000),
        splits=config.n_is_splits, device=config.device
    )
    print(f"Final FID: {final_fid:.4f}, Final IS: {final_is[0]:.4f} ± {final_is[1]:.4f}")


def plot_evaluation_results():
    """Plot training metrics."""
    try:
        plt.figure(figsize=(12, 5))

        if os.path.exists('results/gan/evaluation/fid_history.npy'):
            fid_history = np.load('results/gan/evaluation/fid_history.npy')
            if len(fid_history) > 0:
                iterations = fid_history[:, 0]
                fid_scores = fid_history[:, 1]

                plt.subplot(1, 2, 1)
                plt.plot(iterations, fid_scores, 'b-', linewidth=2)
                plt.xlabel('Iteration')
                plt.ylabel('FID Score')
                plt.title('FID Score during Training')
                plt.grid(True, alpha=0.3)

        if os.path.exists('results/gan/evaluation/is_history.npy'):
            is_history = np.load('results/gan/evaluation/is_history.npy')
            if len(is_history) > 0:
                is_iterations = is_history[:, 0]
                is_means = is_history[:, 1]
                is_stds = is_history[:, 2]

                plt.subplot(1, 2, 2)
                plt.errorbar(is_iterations, is_means, yerr=is_stds, fmt='o-',
                            capsize=5, color='r', alpha=0.7, markersize=4)
                plt.xlabel('Iteration')
                plt.ylabel('Inception Score')
                plt.title('Inception Score during Training')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/gan/evaluation/training_metrics.png', dpi=150, bbox_inches='tight')
        print("Saved metrics plot to results/gan/evaluation/training_metrics.png")
    except Exception as e:
        print(f"Plotting error: {e}")


if __name__ == "__main__":
    dataset_name = 'mnist'
    mode = 'train'

    if mode == 'train':
        train_loader = get_dataloader(dataset_name, shuffle=True, drop_last=True)
        eval_loader = get_dataloader(dataset_name, shuffle=False, drop_last=False)

        print("Starting GAN training with FID/IS evaluation...")
        generator = Generator().to(config.device)
        discriminator = Discriminator().to(config.device)
        train_gan_with_evaluation(generator, discriminator, train_loader, eval_loader, config.epochs)
        plot_evaluation_results()

    elif mode == 'evaluate':
        eval_loader = get_dataloader(dataset_name, shuffle=False, drop_last=False)
        generator = Generator().to(config.device)
        if os.path.exists('results/gan/generator_final.pth'):
            generator.load_state_dict(torch.load('results/gan/generator_final.pth'))
            generator.eval()
            print("Loaded pre-trained generator.")
            fid_score, is_scores = evaluate_gan(
                generator, eval_loader, n_fid_samples=2000, n_is_samples=2000, splits=10
            )
            print(f"\nFinal Results:")
            print(f"FID: {fid_score:.4f}")
            print(f"IS: {is_scores[0]:.4f} ± {is_scores[1]:.4f}")
        else:
            print("No trained model found.")

    elif mode == 'plot':
        plot_evaluation_results()

    print("\nDone!")
