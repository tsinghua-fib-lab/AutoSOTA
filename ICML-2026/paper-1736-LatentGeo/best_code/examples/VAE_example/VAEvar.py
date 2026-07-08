import numpy as np
import random
import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
"""
Minimal implementation of a Variational Autoencoder (VAE) with variable decoder variance.
"""



# ----------------------
# Reproducibility
# ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(20)

# ----------------------
# VAE model definition
# ----------------------
class VAE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, latent_dim=2, activation = nn.Tanh()):
        super().__init__()
        #encoding
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), activation, 
                                 nn.Linear(hidden_dim, hidden_dim), activation)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        #decoding
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden_dim), activation,
                                 nn.Linear(hidden_dim, hidden_dim), activation)
        self.dec_mu = nn.Sequential(self.dec,
                                nn.Linear(hidden_dim, input_dim))
        self.dec_logvar = nn.Sequential(self.dec,
                                        nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.fc1(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        #h = self.fc2(z) #fixed decoder variance
        return self.dec_mu(z), self.dec_logvar(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dec_mu, dec_logvar = self.decode(z)
        return dec_mu, dec_logvar, mu, logvar

    # ----------------------
    # Loss function
    # ----------------------
def vae_loss(x, dec_mu, dec_logvar, mu, logvar, beta=1.):

    recon = 0.5 * (dec_logvar.sum(dim=1, keepdim=True) + 
                   ((x - dec_mu).pow(2) / dec_logvar.exp()).sum(dim=1, keepdim=True)).mean()

    kld = -0.5 * ( logvar.sum(dim=1, keepdim=True) - mu.pow(2).sum(dim=1, keepdim=True) - 
                  logvar.exp().sum(dim=1, keepdim=True)).mean()

    return recon + beta * kld


def train(save_path, train_dataset):
    train_dataset = torch.tensor(train_dataset, dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # ----------------------
    # Training
    # ----------------------
    def beta_for_epoch(epoch, warmup=100, beta_max=1.):
        return float(min(1.0, epoch / warmup)) * beta_max

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE(input_dim=3, hidden_dim=32, latent_dim=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir= Path(__file__).parent / "vae_runs/")
    epochs = 100

    beta_max = 0.01

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.view(x.size(0), -1).to(device)
            optimizer.zero_grad()
            dec_mu, dec_logvar, mu, logvar = model(x)

            beta = beta_for_epoch(epoch, warmup= 100, beta_max=beta_max)

            loss = vae_loss(x,dec_mu, dec_logvar, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        if TENSORBOARD_AVAILABLE:
            writer.add_scalar('Loss/train', avg_loss, epoch)
        with torch.no_grad():
            dec_mu, dec_logvar, mu, logvar = model(train_dataset.to(device))
            recon_lossLH =  vae_loss(train_dataset.to(device),dec_mu, dec_logvar, mu, logvar, 0).item() / len(train_loader.dataset)
            recon_loss = ((train_dataset.to(device) - dec_mu)**2).mean().item()
            if TENSORBOARD_AVAILABLE:
                writer.add_scalar('Loss/reconLH',recon_lossLH, epoch)
                writer.add_scalar('Loss/reconmse',recon_loss, epoch)
        print(f'Epoch {epoch+1} / {epochs}, Loss: {avg_loss:.4f}, reconLossLH: {recon_lossLH:.8f}, reconmse: {recon_loss:.8f}, beta: {beta:.4f}')
    
    if TENSORBOARD_AVAILABLE:
        writer.close()


    embedding = model.encode(train_dataset.to(device))[0]
    np.save(save_path / "latent_embedding", embedding.detach().cpu().numpy())
    torch.save(model.state_dict(), save_path / "final_model_state.pth")


if __name__ == "__main__":
    base_path = Path(__file__).parent  # script directory
    save_path = base_path / "saved_models/new/"
    save_path.mkdir(parents=True, exist_ok=True)

    data = np.load(base_path / "data/hole_3d_noise.npy")
    train(save_path, data)