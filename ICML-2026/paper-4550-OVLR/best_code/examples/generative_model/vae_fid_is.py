"""
OVLR Example: VAE with FID and Inception Score Evaluation.

This is a complete implementation of VAE training with:
- Deep Residual VAE architecture
- FID (Fréchet Inception Distance) evaluation
- IS (Inception Score) evaluation
- KL Annealing
- Periodic evaluation during training
- Reconstruction visualization
- Latent space visualization

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    input_dim = 28 * 28
    base_dim = 256
    latent_dim = 64
    n_blocks = 3
    epochs = 200
    learning_rate = 1e-3
    img_size = 28
    channels = 1
    n_fid_samples = 5000
    n_is_samples = 5000
    n_is_splits = 10
    eval_interval = 5
    save_interval = 5


config = Config()
print(f"Using device: {config.device}")


# ===================== Residual VAE =====================

class ResidualBlock(nn.Module):
    """Residual block for VAE."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out


class VAE(nn.Module):
    """Deep VAE with residual connections."""
    def __init__(self, input_dim=784, base_dim=256, latent_dim=64, n_blocks=3):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_in = nn.Sequential(
            nn.Linear(input_dim, base_dim * 2),
            nn.ReLU(),
            nn.Linear(base_dim * 2, base_dim)
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(base_dim) for _ in range(n_blocks)]
        )

        self.encoder_out = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(base_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(base_dim // 2, latent_dim)

        # Decoder
        self.decoder_in = nn.Sequential(
            nn.Linear(latent_dim, base_dim // 2),
            nn.ReLU(),
            nn.Linear(base_dim // 2, base_dim)
        )

        self.decoder_res_blocks = nn.ModuleList(
            [ResidualBlock(base_dim) for _ in range(n_blocks)]
        )

        self.decoder_out = nn.Sequential(
            nn.Linear(base_dim, base_dim * 2),
            nn.ReLU(),
            nn.Linear(base_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(-1, self.input_dim)
        h = self.encoder_in(x)
        for block in self.res_blocks:
            h = block(h)
        h = self.encoder_out(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_in(z)
        for block in self.decoder_res_blocks:
            h = block(h)
        return self.decoder_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def generate(self, n_samples=1, device='cuda'):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples.view(-1, 1, 28, 28)


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE ELBO loss with KL annealing."""
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


# ===================== FID and IS Evaluation =====================

class InceptionV3ForEvaluation(nn.Module):
    """InceptionV3 for FID and Inception Score calculation."""

    def __init__(self, device='cuda'):
        super(InceptionV3ForEvaluation, self).__init__()

        try:
            from torchvision.models import Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        except Exception:
            self.inception = inception_v3(pretrained=True, transform_input=False)

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
        """Preprocess: [0, 1] -> resize -> ImageNet normalization."""
        x = torch.clamp(x, 0, 1)

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


def evaluate_vae(model, dataloader, n_fid_samples=1000, n_is_samples=1000, splits=10, device='cuda'):
    """Evaluate VAE with FID and Inception Score."""
    print(f"\nEvaluating VAE...")
    print(f"FID samples: {n_fid_samples}, IS samples: {n_is_samples}")

    inception_model = InceptionV3ForEvaluation(device=device)
    inception_model.eval()

    try:
        # Collect real image features
        print("\n1. Extracting real image features...")
        real_features = []
        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc="Real images"):
                data = data.to(device)
                data = torch.clamp(data, 0, 1)
                results = inception_model(data, return_features=True, return_probs=False)
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
                samples = model.generate(config.batch_size, device)
                samples = torch.clamp(samples, 0, 1)
                results = inception_model(samples, return_features=True, return_probs=True)

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
        os.makedirs('results/vae/evaluation', exist_ok=True)

        results = {
            'fid': fid_score,
            'is_mean': is_mean,
            'is_std': is_std,
            'n_fid_samples': min(len(real_features), len(fake_features)),
            'n_is_samples': len(fake_probs),
        }

        with open('results/vae/evaluation/latest_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        return fid_score, (is_mean, is_std)

    except Exception as e:
        print(f"Evaluation error: {e}")
        return float('inf'), (0, 0)


# ===================== Training =====================

def train_vae_with_evaluation(model, train_loader, eval_loader, epochs, device='cuda'):
    """Train VAE with periodic FID/IS evaluation."""
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    os.makedirs('results/vae', exist_ok=True)
    os.makedirs('results/vae/evaluation', exist_ok=True)

    loss_history = []
    fid_history = []
    is_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        beta = min(1.0, epoch / 10)  # KL Annealing

        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)
            loss = vae_loss_function(recon_x, data, mu, logvar, beta=beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        loss_history.append((epoch, avg_loss))
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        # Save model
        if epoch % config.save_interval == 0 or epoch == epochs:
            torch.save(model.state_dict(), f'results/vae/vae_epoch_{epoch}.pth')

        # Evaluate
        if epoch % config.eval_interval == 0 or epoch == epochs:
            print(f"\nEvaluating at epoch {epoch}...")
            try:
                fid_score, is_scores = evaluate_vae(
                    model, eval_loader,
                    n_fid_samples=min(config.n_fid_samples, 1000),
                    n_is_samples=min(config.n_is_samples, 2000),
                    splits=5, device=device
                )
                fid_history.append((epoch, fid_score))
                is_history.append((epoch, is_scores[0], is_scores[1]))

                if loss_history:
                    np.save('results/vae/evaluation/loss_history.npy', np.array(loss_history))
                if fid_history:
                    np.save('results/vae/evaluation/fid_history.npy', np.array(fid_history))
                if is_history:
                    np.save('results/vae/evaluation/is_history.npy', np.array(is_history))

                print(f"FID: {fid_score:.4f}, IS: {is_scores[0]:.4f} ± {is_scores[1]:.4f}")

            except Exception as e:
                print(f"Evaluation skipped: {e}")
                continue

        # Visualize reconstructions
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                data, _ = next(iter(train_loader))
                data = data.to(device)
                recon_x, _, _ = model(data)
                recon_x = recon_x.view(-1, 1, 28, 28).cpu()
                data = data.cpu()

                save_path = f"results/vae/reconstruction_epoch_{epoch}.pdf"
                with PdfPages(save_path) as pdf:
                    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
                    for i in range(5):
                        axes[0, i].imshow(data[i].view(28, 28).numpy(), cmap='gray')
                        axes[0, i].set_title(f"Original {i+1}", fontsize=10)
                        axes[0, i].axis('off')

                        axes[1, i].imshow(recon_x[i].view(28, 28).numpy(), cmap='gray')
                        axes[1, i].set_title(f"Reconstructed {i+1}", fontsize=10)
                        axes[1, i].axis('off')

                    fig.suptitle(f"Epoch {epoch}: Original vs Reconstructed Images", fontsize=12)
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                print(f"Saved reconstructions to {save_path}")

    torch.save(model.state_dict(), 'results/vae/vae_final.pth')

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_fid, final_is = evaluate_vae(
        model, eval_loader,
        n_fid_samples=min(config.n_fid_samples, 2000),
        n_is_samples=min(config.n_is_samples, 2000),
        splits=config.n_is_splits, device=device
    )
    print(f"Final FID: {final_fid:.4f}, Final IS: {final_is[0]:.4f} ± {final_is[1]:.4f}")

    return loss_history, fid_history, is_history


def plot_vae_results():
    """Plot VAE training metrics."""
    try:
        plt.figure(figsize=(15, 5))

        if os.path.exists('results/vae/evaluation/loss_history.npy'):
            loss_history = np.load('results/vae/evaluation/loss_history.npy')
            if len(loss_history) > 0:
                epochs_loss = loss_history[:, 0]
                losses = loss_history[:, 1]

                plt.subplot(1, 3, 1)
                plt.plot(epochs_loss, losses, 'b-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('VAE Training Loss')
                plt.grid(True, alpha=0.3)

        if os.path.exists('results/vae/evaluation/fid_history.npy'):
            fid_history = np.load('results/vae/evaluation/fid_history.npy')
            if len(fid_history) > 0:
                epochs_fid = fid_history[:, 0]
                fids = fid_history[:, 1]

                plt.subplot(1, 3, 2)
                plt.plot(epochs_fid, fids, 'r-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('FID Score')
                plt.title('FID Score during Training')
                plt.grid(True, alpha=0.3)

        if os.path.exists('results/vae/evaluation/is_history.npy'):
            is_history = np.load('results/vae/evaluation/is_history.npy')
            if len(is_history) > 0:
                epochs_is = is_history[:, 0]
                is_means = is_history[:, 1]
                is_stds = is_history[:, 2]

                plt.subplot(1, 3, 3)
                plt.errorbar(epochs_is, is_means, yerr=is_stds, fmt='o-',
                            capsize=5, color='g', alpha=0.7, markersize=4)
                plt.xlabel('Epoch')
                plt.ylabel('Inception Score')
                plt.title('Inception Score during Training')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/vae/evaluation/vae_training_metrics.png', dpi=150, bbox_inches='tight')
        print("Saved metrics plot to results/vae/evaluation/vae_training_metrics.png")
    except Exception as e:
        print(f"Plotting error: {e}")


if __name__ == "__main__":
    mode = 'train'

    # Data loading
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    model = VAE(config.input_dim, config.base_dim, config.latent_dim, config.n_blocks).to(config.device)

    if mode == 'train':
        print("Starting VAE training with FID/IS evaluation...")
        loss_history, fid_history, is_history = train_vae_with_evaluation(
            model, train_loader, eval_loader, config.epochs, config.device
        )
        plot_vae_results()

        # Generate latent space samples
        print("\nGenerating latent space samples...")
        model.eval()
        with torch.no_grad():
            z = torch.randn(16, config.latent_dim).to(config.device)
            samples = model.decode(z).view(-1, 1, 28, 28).cpu()

            save_path = "results/vae/latent_space_samples.pdf"
            with PdfPages(save_path) as pdf:
                fig, axes = plt.subplots(4, 4, figsize=(6, 6))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(samples[i].view(28, 28).numpy(), cmap='gray')
                    ax.axis('off')
                fig.suptitle("Generated Samples from Latent Space", fontsize=14)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            print(f"Saved samples to {save_path}")

    elif mode == 'evaluate':
        print("Evaluating pre-trained VAE...")
        if os.path.exists('results/vae/vae_final.pth'):
            model.load_state_dict(torch.load('results/vae/vae_final.pth'))
            model.eval()
            fid_score, is_scores = evaluate_vae(
                model, eval_loader, n_fid_samples=2000, n_is_samples=2000, splits=10
            )
            print(f"\nFinal Results:")
            print(f"FID: {fid_score:.4f}")
            print(f"IS: {is_scores[0]:.4f} ± {is_scores[1]:.4f}")
        else:
            print("No trained model found.")

    elif mode == 'plot':
        plot_vae_results()

    print("\nDone!")
