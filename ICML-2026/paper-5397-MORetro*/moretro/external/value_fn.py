"""
Value function trained by authors from Retro* paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate, device):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.device = device

        logging.info("Initializing value model: latent_dim=%d" % self.latent_dim)

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x


def load_value_model(model_path, device):
    model = ValueMLP(
        n_layers=1, fp_dim=2048, latent_dim=128, dropout_rate=0.1, device=device
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from pathlib import Path
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    mol = Chem.MolFromSmiles(smi)
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = fp_generator.GetFingerprintAsNumPy(mol)
    fp = torch.tensor(fp).float().unsqueeze(0)
    device = torch.device("cpu")
    model_path = Path(__file__).parent.parent / "models" / "model_value.pt"
    model = load_value_model(model_path, device)
    print(f"Predicted value for {smi}: {model(fp).item()}")
