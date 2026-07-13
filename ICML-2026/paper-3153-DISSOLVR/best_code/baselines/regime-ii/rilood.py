# rilood.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, Set2Set
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Lipinski
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import pandas as pd
import multiprocessing
import warnings
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")



# Device configuration
NUM_WORKERS = multiprocessing.cpu_count()
torch.set_num_threads(NUM_WORKERS)
torch.set_num_interop_threads(NUM_WORKERS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. CIGIN FEATURE SET (Tables 1 & 2 from CIGIN paper)
# ============================================================================

# Table 1: Atom (Node) Features
# - Atom Type: H, C, N, O, F... (one-hot) - 11 elements
# - Implicit Valence (Binary)
# - Radical Electrons (Binary)
# - Chirality: R, S, None (3)
# - Number of Hydrogens (one-hot, 0-4+unknown = 6)
# - Hybridization: sp, sp2, sp3, sp3d (4)
# - Acidic (Binary)
# - Basic (Binary)
# - Aromatic (Binary)
# - Donor (Binary)
# - Acceptor (Binary)
# Total: 11 + 1 + 1 + 3 + 6 + 4 + 1 + 1 + 1 + 1 + 1 = 31

ATOM_FEATURES = {
    'atom_type': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'unknown'],  # 11
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D'],  # 4
    'chirality': ['R', 'S', 'None'],  # 3
    'num_h': ['0', '1', '2', '3', '4', 'unknown'],  # 6
}

# Table 2: Bond (Edge) Features
# - Bond Type: Single, Double, Triple, Aromatic (4)
# - Bond is in Conjugation (Binary)
# - Bond is in Ring (Binary)
# - Bond Chirality: E, Z, None (3)
# Total: 4 + 1 + 1 + 3 = 9

BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],  # 4
    'stereo': ['E', 'Z', 'None'],  # 3
}

# Feature dimensions
ATOM_FEAT_DIM = 11 + 4 + 3 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # 31
BOND_FEAT_DIM = 4 + 1 + 1 + 3  # 9


def one_hot(value, choices):
    """Create one-hot encoding."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1  # unknown/None category
    return encoding


def get_atom_features(atom) -> List[float]:
    """Extract CIGIN Table 1 compliant atom features."""
    features = []
    
    # 1. Atom Type (11)
    symbol = atom.GetSymbol()
    features += one_hot(symbol, ATOM_FEATURES['atom_type'])
    
    # 2. Implicit Valence (Binary) - use GetValence with explicit=False
    features.append(1.0 if (atom.GetTotalValence() - atom.GetExplicitValence()) > 0 else 0.0)
    
    # 3. Radical Electrons (Binary)
    features.append(1.0 if atom.GetNumRadicalElectrons() > 0 else 0.0)
    
    # 4. Chirality: R, S, None (3)
    try:
        chi = str(atom.GetChiralTag())
        if 'CW' in chi:
            features += one_hot('R', ATOM_FEATURES['chirality'])
        elif 'CCW' in chi:
            features += one_hot('S', ATOM_FEATURES['chirality'])
        else:
            features += one_hot('None', ATOM_FEATURES['chirality'])
    except:
        features += one_hot('None', ATOM_FEATURES['chirality'])
    
    # 5. Number of Hydrogens (6)
    num_h = str(min(atom.GetTotalNumHs(), 4))
    features += one_hot(num_h, ATOM_FEATURES['num_h'])
    
    # 6. Hybridization (4)
    hyb = str(atom.GetHybridization())
    hyb_val = hyb.split('.')[-1] if '.' in hyb else hyb
    if hyb_val in ['S', 'SP']:
        hyb_val = 'SP'
    features += one_hot(hyb_val, ATOM_FEATURES['hybridization'])
    
    # 7. Acidic (Binary) - carboxylic acids, sulfonic acids, phosphoric acids
    # Atoms that can donate H+ (O in -COOH, -SO3H, etc.)
    is_acidic = (atom.GetSymbol() == 'O' and 
                 atom.GetTotalNumHs() > 0 and 
                 any(n.GetSymbol() in ['C', 'S', 'P'] for n in atom.GetNeighbors()))
    features.append(1.0 if is_acidic else 0.0)
    
    # 8. Basic (Binary) - amines, nitrogen with lone pair
    # N atoms that can accept H+
    is_basic = (atom.GetSymbol() == 'N' and 
                not atom.GetIsAromatic() and 
                atom.GetFormalCharge() == 0)
    features.append(1.0 if is_basic else 0.0)
    
    # 9. Aromatic (Binary)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    
    # 10. Donor (Binary) - H-bond donor (N-H, O-H)
    is_donor = atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0
    features.append(1.0 if is_donor else 0.0)
    
    # 11. Acceptor (Binary) - H-bond acceptor (N, O, F with lone pairs)
    is_acceptor = atom.GetSymbol() in ['N', 'O', 'F']
    features.append(1.0 if is_acceptor else 0.0)
    
    return features


def get_bond_features(bond) -> List[float]:
    """Extract CIGIN Table 2 compliant bond features."""
    features = []
    
    # 1. Bond Type (4)
    bond_type = str(bond.GetBondType()).split('.')[-1]
    features += one_hot(bond_type, BOND_FEATURES['bond_type'])
    
    # 2. Bond is in Conjugation (Binary)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    
    # 3. Bond is in Ring (Binary)
    features.append(1.0 if bond.IsInRing() else 0.0)
    
    # 4. Bond Chirality: E, Z, None (3)
    stereo = str(bond.GetStereo())
    if 'Z' in stereo or 'CIS' in stereo:
        features += one_hot('Z', BOND_FEATURES['stereo'])
    elif 'E' in stereo or 'TRANS' in stereo:
        features += one_hot('E', BOND_FEATURES['stereo'])
    else:
        features += one_hot('None', BOND_FEATURES['stereo'])
    
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyG graph with CIGIN features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_feat = get_bond_features(bond)
        edge_attr.extend([bond_feat, bond_feat])
    
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ============================================================================
# 2. GNN ENCODER
# ============================================================================

class MPNNLayer(MessagePassing):
    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__(aggr='add', node_dim=0)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
    
    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
        # Set2Set Readout - input is 2*hidden_dim after interaction concatenation
        self.readout = Set2Set(hidden_dim * 2, processing_steps=3) 
        self.final_proj = nn.Linear(2 * (hidden_dim * 2), hidden_dim)
        
    def forward_nodes(self, data: Data) -> torch.Tensor:
        """Returns node representations [N, hidden_dim]"""
        x = self.node_embed(data.x)
        has_edges = data.edge_index.size(1) > 0
        edge_attr = self.edge_embed(data.edge_attr) if has_edges else None
        
        for layer, norm in zip(self.layers, self.layer_norms):
            if has_edges:
                x_new = layer(x, data.edge_index, edge_attr)
            else:
                x_new = x
            x = norm(x + x_new)
            x = self.dropout(x)
        return x

# ============================================================================
# 3. EXACT ATOMIC INTERACTION MAP (Sec 3.1)
#    I = h1 * h2^T, h_new = I * h2
# ============================================================================

class InteractionMap(nn.Module):
    """
    Bidirectional CIGIN-style interaction map (Sec 3.1, Appendix C.1).
    - h1_new = [h1, I·h2] for solute
    - h2_new = [h2, I^T·h1] for solvent
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h1_nodes, h2_nodes, batch1, batch2):
        """
        Bidirectional per-graph interaction.
        
        Returns:
            h1_enhanced: [Total_N1, 2*D] - solute enriched with solvent
            h2_enhanced: [Total_N2, 2*D] - solvent enriched with solute
        """
        batch_size = batch1.max().item() + 1
        enhanced_h1_list = []
        enhanced_h2_list = []
        
        for i in range(batch_size):
            mask1 = (batch1 == i)
            mask2 = (batch2 == i)
            
            h1_i = h1_nodes[mask1]  # [N_solute, D]
            h2_i = h2_nodes[mask2]  # [N_solvent, D]
            
            if h1_i.size(0) == 0 or h2_i.size(0) == 0:
                enhanced_h1_list.append(torch.cat([h1_i, h1_i], dim=-1))
                enhanced_h2_list.append(torch.cat([h2_i, h2_i], dim=-1))
                continue

            # I = h1 * h2^T / sqrt(d)  [N_solute, N_solvent] - scaled to prevent explosion
            scale = self.hidden_dim ** -0.5
            I = torch.matmul(h1_i, h2_i.t()) * scale
            
            # Clamp I to prevent extreme values
            I = torch.clamp(I, -10, 10)
            
            # h1_new = I * h2  [N_solute, D]
            h1_interaction = torch.matmul(I, h2_i)
            
            # h2_new = I^T * h1  [N_solvent, D]
            h2_interaction = torch.matmul(I.t(), h1_i)
            
            # LayerNorm to stabilize
            h1_interaction = self.layer_norm(h1_interaction)
            h2_interaction = self.layer_norm(h2_interaction)
            
            # Concatenate: [original, interaction]
            enhanced_h1_list.append(torch.cat([h1_i, h1_interaction], dim=-1))
            enhanced_h2_list.append(torch.cat([h2_i, h2_interaction], dim=-1))
            
        return torch.cat(enhanced_h1_list, dim=0), torch.cat(enhanced_h2_list, dim=0)

# ============================================================================
# 4. MCAR: MULTI-GRANULARITY CONTEXT-AWARE REFINEMENT (Eq 7-10)
# ============================================================================

class MCAR(nn.Module):
    """
    MCAR: MLP on E = concat[z, h_solvent] (paper text: E = concat[z, H_2]).
    Stable MLP-based global-local fusion.
    """
    def __init__(self, z_dim: int, h_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_dim = z_dim + h_dim
        self.output_dim = z_dim
        
        # Global interaction (Eq 8): Wc * ReLU(E)
        self.global_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        # Local interaction (Eq 9): PReLU(Wl * E)
        self.local_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.PReLU()
        )
        
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, z: torch.Tensor, h_solvent: torch.Tensor) -> torch.Tensor:
        # E = concat[z, h_solvent]  [batch, z_dim + h_dim]
        E = torch.cat([z, h_solvent], dim=-1)
        
        # Global interaction
        Oc = self.global_proj(E)  # [batch, output_dim]
        
        # Local interaction
        Of = self.local_proj(E)  # [batch, output_dim]
        
        # Hadamard fusion (Eq 10)
        Hc = Oc * Of
        
        # Output with residual and LayerNorm
        out = self.layer_norm(self.output_proj(Hc) + z)
        
        return out

# ============================================================================
# 5. MCVAE: MIXUP-ENHANCED CONDITIONAL VAE (Eq 5-6)
#    With proper Decoder for reconstruction
# ============================================================================

class MCVAE(nn.Module):
    """
    Proper MCVAE with:
    - Encoder: h -> (mu, log_var)
    - Decoder: z -> h_hat (for reconstruction)
    - Sigma for reconstruction uncertainty
    """
    def __init__(self, hidden_dim: int, env_dim: int, latent_dim: int):
        super().__init__()
        # Encoder: p(z|h, e)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mu, log_var
        )
        
        # Decoder: p(h|z, e) - decodes z back to h_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Sigma for reconstruction uncertainty (Eq 6)
        self.log_sigma_rec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, h, e):
        # Encode
        encoded = self.encoder(torch.cat([h, e], dim=-1))
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        
        # Clamp to prevent divergence
        mu = torch.clamp(mu, -10, 10)
        log_var = torch.clamp(log_var, -10, 10)
        
        z = self.reparameterize(mu, log_var)
        
        # Decode z back to h_hat
        h_hat = self.decoder(torch.cat([z, e], dim=-1))
        
        # Reconstruction uncertainty (clamped)
        log_sigma_rec = torch.clamp(self.log_sigma_rec(h), -10, 10)
        
        return z, mu, log_var, h_hat, log_sigma_rec

# ============================================================================
# 6. RILOOD MODEL
# ============================================================================

class RILOOD(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 168, num_layers: int = 3, 
                 num_solvents: int = 200, mixup_alpha: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim  # Paper Appendix: d_z = 168
        self.mixup_alpha = mixup_alpha
        
        self.solute_encoder = GNNEncoder(ATOM_FEAT_DIM, BOND_FEAT_DIM, hidden_dim, num_layers)
        self.solvent_encoder = GNNEncoder(ATOM_FEAT_DIM, BOND_FEAT_DIM, hidden_dim, num_layers)
        
        self.interaction_map = InteractionMap(hidden_dim)
        
        # Paper: latent_dim = 168
        self.mcvae = MCVAE(hidden_dim, num_solvents, latent_dim)
        self.mcar = MCAR(z_dim=latent_dim, h_dim=hidden_dim, num_layers=num_layers)
        
        # Predictor with separate sigma for regression (Eq 4)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1)
        )
        
        # Sigma for regression uncertainty (Eq 4) - separate from MCVAE sigma
        self.log_sigma_reg = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1)
        )
        
        # Projection for MI contrastive loss (h1_graph: hidden_dim -> latent_dim)
        self.h1_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, solute_data, solvent_data, env_idx, 
                targets=None, training=True):
        
        # 1. Encode Nodes
        h1_nodes = self.solute_encoder.forward_nodes(solute_data)
        h2_nodes = self.solvent_encoder.forward_nodes(solvent_data)
        
        # 2. Bidirectional Interaction Map:
        #    h1 ← [h1, I·h2], h2 ← [h2, I^T·h1]
        h1_interactive, h2_interactive = self.interaction_map(
            h1_nodes, h2_nodes, solute_data.batch, solvent_data.batch)
        
        # 3. Set2Set Readout on BOTH interactive features
        h1_graph = self.solute_encoder.readout(h1_interactive, solute_data.batch)
        h1_graph = self.solute_encoder.final_proj(h1_graph)
        
        h2_graph = self.solvent_encoder.readout(h2_interactive, solvent_data.batch)
        h2_graph = self.solvent_encoder.final_proj(h2_graph)
        
        batch_size = h1_graph.size(0)
        device = h1_graph.device
        
        # 4. Environment one-hot
        num_classes = self.mcvae.encoder[0].in_features - self.hidden_dim
        e = F.one_hot(env_idx.clamp(0, num_classes - 1), num_classes=num_classes).float()
        
        # 5. Mixup (Paper Eq 3: H_tilde = lambda*H1 + (1-lambda)*H2)
        # Paper mixes solute with solvent representations
        h1_mix, targets_mix = h1_graph, targets
        if training and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # Paper Eq 3: mix solute with solvent (NOT solute with permuted solute)
            h1_mix = lam * h1_graph + (1 - lam) * h2_graph
        
        # 6. MCVAE (now returns h_hat from decoder)
        z, mu, log_var, h_hat, log_sigma_rec = self.mcvae(h1_mix, e)
        
        # 7. MCAR
        hc = self.mcar(z, h2_graph)
        
        # 8. Prediction + regression uncertainty
        pred = self.predictor(hc).squeeze(-1)
        log_sigma_reg = torch.clamp(self.log_sigma_reg(hc), -10, 10)  # Clamped
        
        # 9. Losses
        losses = {}
        if training and targets is not None:
            # Eq 4: Regression loss with sigma_reg (aleatoric uncertainty of y)
            sigma_sq_reg = torch.exp(log_sigma_reg).squeeze() + 1e-6
            losses['reg'] = ((targets_mix - pred).pow(2) / sigma_sq_reg + log_sigma_reg.squeeze()).mean()
            
            # Eq 6: MCVAE Loss (KLD + reconstruction with sigma_rec)
            # KLD term
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
            
            # Reconstruction: ||Dec(z) - h||^2 / sigma_rec^2 + log(sigma_rec^2)
            sigma_sq_rec = torch.exp(log_sigma_rec).squeeze() + 1e-6
            rec = ((h_hat - h1_mix).pow(2).sum(dim=1) / sigma_sq_rec + log_sigma_rec.squeeze()).mean()
            losses['vae'] = kld + rec
            
            # Eq 11-12: MI contrastive loss
            hc_n = F.normalize(hc, dim=1)  # [batch, latent_dim]
            h1_proj = self.h1_proj(h1_graph)  # [batch, hidden_dim] -> [batch, latent_dim]
            h1_n = F.normalize(h1_proj, dim=1)
            logits = torch.matmul(hc_n, h1_n.T) / 0.5
            losses['mi'] = F.cross_entropy(logits, torch.arange(batch_size, device=device))
            
        return pred, losses

# ============================================================================
# 7. DATASET
# ============================================================================

class SolubilityDataset(torch.utils.data.Dataset):
    def __init__(self, df, solvent_map):
        self.solutes = df['Solute'].tolist()
        self.solvents = df['Solvent'].tolist()
        self.targets = df['LogS'].tolist()
        self.solvent_map = solvent_map
        
        # Pre-cache and validate graphs
        self.solute_graphs = {}
        self.solvent_graphs = {}
        self.valid_indices = []
        
        print("Caching molecular graphs...")
        for i, (sol, solv) in enumerate(tqdm(zip(self.solutes, self.solvents), total=len(self.solutes))):
            if sol not in self.solute_graphs:
                g = smiles_to_graph(sol)
                if g is None:
                    continue
                self.solute_graphs[sol] = g
            elif self.solute_graphs.get(sol) is None:
                continue
                
            if solv not in self.solvent_graphs:
                g = smiles_to_graph(solv)
                if g is None:
                    continue
                self.solvent_graphs[solv] = g
            elif self.solvent_graphs.get(solv) is None:
                continue
            
            self.valid_indices.append(i)
        
        print(f"Valid samples: {len(self.valid_indices)}/{len(self.solutes)}")

    def __len__(self): 
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return {
            'solute': self.solute_graphs[self.solutes[real_idx]], 
            'solvent': self.solvent_graphs[self.solvents[real_idx]],
            'env': self.solvent_map[self.solvents[real_idx]], 
            'target': self.targets[real_idx]
        }


def collate_fn(batch):
    return {
        'solute': Batch.from_data_list([b['solute'] for b in batch]), 
        'solvent': Batch.from_data_list([b['solvent'] for b in batch]),
        'env': torch.tensor([b['env'] for b in batch]), 
        'target': torch.tensor([b['target'] for b in batch]).float()
    }

# ============================================================================
# 8. TRAINING
# ============================================================================

def train_one_seed(seed, train_df, test_df, solvent_map, num_epochs=30, patience=10):
    """Train for one seed, return best test RMSE and R2. Uses 5% val split for early stopping."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 5% validation split from training data
    train_df_shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = int(len(train_df_shuffled) * 0.05)
    val_df = train_df_shuffled[:val_size]
    train_df_split = train_df_shuffled[val_size:]
    
    train_ds = SolubilityDataset(train_df_split, solvent_map)
    val_ds = SolubilityDataset(val_df, solvent_map)
    test_ds = SolubilityDataset(test_df, solvent_map)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn,
                            num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn,
                             num_workers=4, persistent_workers=True)
    
    model = RILOOD(num_solvents=len(solvent_map)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Paper: "learning rate was decreased on plateau by factor of 0.1 from 1e-3 to 1e-5"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5
    )
    
    # Updated loss weights
    alpha, beta = 1e-3, 1e-4
    
    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    best_test_r2 = float('-inf')
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        nan_count = 0
        for b in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            _, L = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                         b['env'].to(DEVICE), targets=b['target'].to(DEVICE))
            loss = L['reg'] + alpha * L['vae'] + beta * L['mi']
            
            # Skip batch if NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter clipping
            optimizer.step()
            total_loss += loss.item()
        
        if nan_count > 0:
            print(f"  Warning: {nan_count} batches had NaN loss")
        
        # Evaluate on validation set
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for b in val_loader:
                p, _ = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                            b['env'].to(DEVICE), training=False)
                val_preds.extend(p.cpu().numpy())
                val_targets.extend(b['target'].numpy())
        
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        # Evaluate on test set
        test_preds, test_targets = [], []
        with torch.no_grad():
            for b in test_loader:
                p, _ = model(b['solute'].to(DEVICE), b['solvent'].to(DEVICE), 
                            b['env'].to(DEVICE), training=False)
                test_preds.extend(p.cpu().numpy())
                test_targets.extend(b['target'].numpy())
        
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        test_r2 = r2_score(test_targets, test_preds)
        
        # Early stopping based on validation RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # ReduceLROnPlateau scheduler step (uses validation metric)
        scheduler.step(val_rmse)
        
        print(f"Seed {seed} | Epoch {epoch+1} | Loss: {total_loss:.4f} | Val: {val_rmse:.4f} | Test: {test_rmse:.4f} (R2: {test_r2:.4f}) | Best: {best_test_rmse:.4f}")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    return best_test_rmse, best_test_r2


def main():
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"

    if not os.path.exists(train_path):
        print("Please ensure ./data/train.csv and ./data/test.csv exist.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    all_solvents = set(train_df['Solvent'].unique()) | set(test_df['Solvent'].unique())
    solvent_map = {s: i for i, s in enumerate(all_solvents)}
    
    print(f"Device: {DEVICE} | CPU threads: {NUM_WORKERS}")
    print(f"Node features: {ATOM_FEAT_DIM} | Edge features: {BOND_FEAT_DIM}")
    print(f"Hyperparams: alpha=1e-3, beta=1e-4")
    print("="*60)
    
    # Run 5 seeds
    seeds = [42, 101, 123, 456, 789]
    rmse_results = []
    r2_results = []
    times = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"STARTING SEED {seed}")
        print(f"{'='*60}")
        start_time = time.time()
        best_rmse, best_r2 = train_one_seed(seed, train_df, test_df, solvent_map, num_epochs=30)
        elapsed = time.time() - start_time
        rmse_results.append(best_rmse)
        r2_results.append(best_r2)
        times.append(elapsed)
        print(f"\n✓ Seed {seed} complete! Best RMSE: {best_rmse:.4f} | R2: {best_r2:.4f} | Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Final report
    mean_rmse = np.mean(rmse_results)
    std_rmse = np.std(rmse_results)
    mean_r2 = np.mean(r2_results)
    std_r2 = np.std(r2_results)
    mean_time = np.mean(times)
    total_time = np.sum(times)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Seeds: {seeds}")
    print(f"RMSEs: {[f'{r:.4f}' for r in rmse_results]}")
    print(f"R2s:   {[f'{r:.4f}' for r in r2_results]}")
    print(f"Times: {[f'{t:.1f}s' for t in times]}")
    print(f"\n>>> RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} <<<")
    print(f">>> R2:   {mean_r2:.4f} ± {std_r2:.4f} <<<")
    print(f">>> Avg time per run: {mean_time:.1f}s ({mean_time/60:.1f} min) <<<")
    print(f">>> Total time: {total_time:.1f}s ({total_time/60:.1f} min) <<<")
    print("="*60)

if __name__ == "__main__":
    main()

