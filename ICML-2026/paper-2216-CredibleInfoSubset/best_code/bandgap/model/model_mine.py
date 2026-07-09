import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RankAwareEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank_dim, state_embed_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False)
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, rank_dim),
            nn.Sigmoid() 
        )
        self.fidelity_proj = nn.Linear(state_embed_dim, rank_dim)
        self.rank_attention = nn.Sequential(
            nn.Linear(rank_dim, 1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x, s):
        x_norm = self.bn(x)
        rank_features = self.feature_net(x_norm)
        fidelity_cond = self.fidelity_proj(s)
        rank_features = rank_features * fidelity_cond
        attention_weights = self.rank_attention(rank_features)
        rank_features = rank_features * attention_weights
        
        return rank_features


class IntervalAwareEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_embed_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.tau_head = nn.Sequential(
            nn.Linear(hidden_dim + state_embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()   # τ >= 0
        )

    def forward(self, x, s):
        feat = self.feature_extractor(x)
        tau = self.tau_head(torch.cat([feat, s], dim=-1))
        return tau  # (B, 1)


class CondVAE(nn.Module):
    def __init__(self, comp_embed_dim=32, total_feat_dim=132, state_embed_dim=8,
                 z_dim=64, hidden_enc=[256, 128], hidden_dec=[128, 256], dropout=0.1):
        super().__init__()
        self.z_dim = z_dim
        enc_in = comp_embed_dim + total_feat_dim + state_embed_dim
        enc_layers = []
        prev = enc_in
        for h in hidden_enc:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(prev, z_dim)
        self.logvar = nn.Linear(prev, z_dim)
        dec_layers = []
        prev = z_dim
        for h in hidden_dec:
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.decoder_net = nn.Sequential(*dec_layers)
        self.out_comp = nn.Linear(prev, 73)
        self.out_total = nn.Linear(prev, total_feat_dim)

    def encode(self, x):
        h = self.encoder_net(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = self.decoder_net(z)
        comp_hat = F.softmax(self.out_comp(h), dim=-1)
        total_hat = self.out_total(h)
        return comp_hat, total_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        comp_hat, total_hat = self.decode(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "comp_hat": comp_hat,
            "total_hat": total_hat
        }


class EvidentialRegressor(nn.Module):
    def __init__(self, in_dim, hidden=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 4)

    def forward(self, x):
        h = self.shared(x)
        out = self.out(h)
        mu = out[:, 0:1]
        logv = out[:, 1:2]
        logalpha = out[:, 2:3]
        logbeta = out[:, 3:4]
        v = F.softplus(logv) + 1e-6
        alpha = F.softplus(logalpha) + 1.0 + 1e-6
        beta = F.softplus(logbeta) + 1e-6
        return mu, v, alpha, beta


def evidential_regression_loss(mu, v, alpha, beta, y, lambda_reg=1e-3):
    nll = 0.5 * (
        torch.log(math.pi / v)
        - torch.log(alpha)
        + (2 * alpha) * torch.log(1 + ((y - mu) ** 2) * v / (2 * beta))
    )
    reg = lambda_reg * torch.mean(torch.abs(y - mu) * (2 * v + 2 * alpha))
    return torch.mean(nll) + reg


class BandModelSE(nn.Module):
    def __init__(self,
                 n_elements=73,
                 hidden_dim=64,
                 comp_embed_dim=32,
                 total_feat_dim=132,
                 z_dim=64,
                 n_states=5,
                 state_embed_dim=32,
                 rank_dim=64,):
        super().__init__()
        self.mlp_emb = nn.Sequential(
            nn.Linear(n_elements, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, comp_embed_dim)
        )
        self.state_embedding = nn.Embedding(n_states, state_embed_dim)
        self.vae = CondVAE(
            comp_embed_dim=comp_embed_dim,
            total_feat_dim=total_feat_dim,
            state_embed_dim=state_embed_dim,
            z_dim=z_dim
        )
        self.rank_encoder = RankAwareEncoder(input_dim=comp_embed_dim+total_feat_dim, hidden_dim=hidden_dim, rank_dim=rank_dim,state_embed_dim=state_embed_dim)
        self.interval_encoder = IntervalAwareEncoder(input_dim=comp_embed_dim+total_feat_dim, hidden_dim=hidden_dim,state_embed_dim=state_embed_dim)
        self.predictor = EvidentialRegressor(in_dim=z_dim+rank_dim)
        self.aux_predictor = EvidentialRegressor(in_dim=rank_dim)  # ALGO-04: skip connection
        self.mlp_sort = nn.Sequential(
            nn.Linear(rank_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x_comp, x_total_feats, x_state):
        x_total_feats = (x_total_feats - x_total_feats.mean(dim=0)) / x_total_feats.std(dim=0).clamp_min(1e-6)
        comp_emb = self.mlp_emb(x_comp)
        state_emb = self.state_embedding(x_state.long())
        x_in = torch.cat([comp_emb, x_total_feats, state_emb], dim=-1)
        vae_out = self.vae(x_in)
        z = vae_out["z"]  
        x_feat = torch.cat([comp_emb, x_total_feats], dim=-1)
        rank_feat = self.rank_encoder(x_feat, s=state_emb)
        tau = self.interval_encoder(x_feat, s=state_emb)
        enhanced = torch.concat((rank_feat, z),dim=-1)
        mu, v, alpha, beta = self.predictor(enhanced)
        # ALGO-04: auxiliary prediction from rank features (skip connection)
        mu_aux, v_aux, alpha_aux, beta_aux = self.aux_predictor(rank_feat)
        v = v / (1.0 + tau)
        beta = beta * (1.0 + tau)
        vae_out["bandgap_pred"] = {
            "mu": mu,
            "v": v,
            "alpha": alpha,
            "beta": beta,
            "tau" : tau,
            "mu_aux": mu_aux,
            "v_aux": v_aux,
            "alpha_aux": alpha_aux,
            "beta_aux": beta_aux
        }
        y_rank=self.mlp_sort(rank_feat)
        return vae_out,y_rank


def vae_reconstruction_loss(x_comp, comp_hat):
    return F.mse_loss(comp_hat, x_comp)


def vae_kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

def rank_loss_comp(y_true, y_pred, k=1.0):
    diff_true = y_true - y_true.T  # (N, N)
    diff_pred = y_pred - y_pred.T  # (N, N)
    mask = torch.triu(torch.ones_like(diff_true), diagonal=1) > 0
    diff_true = diff_true[mask]
    diff_pred = diff_pred[mask]
    concordance = torch.sigmoid(k * diff_true * diff_pred)
    loss = 1.0 - concordance.mean()
    return loss

def combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y_bandgap,
                               alpha_vae=1.0, alpha_abs=1.0, alpha_rank=5e-3, alpha_aux=0.0, kl_weight=1e-3):
    out,y_rank = model(x_comp, x_total, x_state)
    state = x_state.long()
    mu = out["bandgap_pred"]["mu"]
    v = out["bandgap_pred"]["v"]
    alpha = out["bandgap_pred"]["alpha"]
    beta = out["bandgap_pred"]["beta"]
    tau = out["bandgap_pred"]["tau"]
    mu_aux = out["bandgap_pred"]["mu_aux"]
    v_aux = out["bandgap_pred"]["v_aux"]
    alpha_aux_nig = out["bandgap_pred"]["alpha_aux"]
    beta_aux_nig = out["bandgap_pred"]["beta_aux"]
    rank_weight = 1.0 / (1.0 + 0.25*state.float())
    rec_loss = vae_reconstruction_loss(x_comp, out["comp_hat"])
    kld = vae_kl_loss(out["mu"], out["logvar"])
    vae_loss = rec_loss + kl_weight * kld
    abs_weight = 1.0 / (1.0 +2*state.float())
    # ALGO-04: ensemble prediction from main + aux predictors
    mu_ensemble = (mu + mu_aux) / 2.0
    abs_loss =  torch.mean(abs_weight*evidential_regression_loss(mu_ensemble, (v+v_aux)/2.0, (alpha+alpha_aux_nig)/2.0, (beta+beta_aux_nig)/2.0, y_bandgap))
    rank_loss = torch.tensor(0., device=x_comp.device)
    
    unique_states = state.unique()
    for s in unique_states:
        mask = (state == s)
        if mask.sum() > 1:
            y_true = y_bandgap[mask]
            y_pred = y_rank[mask]
            rank_loss += rank_loss_comp(y_true, y_pred)
    rank_loss = rank_loss / len(unique_states)

    total_loss = alpha_vae * vae_loss + alpha_abs * abs_loss  + alpha_rank * rank_loss 
    mae = torch.mean(torch.abs(mu_ensemble - y_bandgap)).item()
    # for s_val in torch.unique(x_state):
    #     mask = (x_state == s_val)
    #     print(f"State {s_val.item()} avg tau:", tau[mask].mean().item())
    return total_loss, {
        "vae_loss": vae_loss.item(),
        "abs_loss": abs_loss.item(),
        "rank_loss": rank_loss.item(),
        "mae": mae,
        "pred": mu_ensemble.detach()
    }



