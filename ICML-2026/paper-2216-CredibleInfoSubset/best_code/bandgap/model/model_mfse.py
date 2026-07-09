import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 n_states=4,
                 state_embed_dim=32):
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
        self.predictor = EvidentialRegressor(in_dim=z_dim)

    def forward(self, x_comp, x_total_feats, x_state):
        x_total_feats = (x_total_feats - x_total_feats.mean(dim=0)) / \
                        x_total_feats.std(dim=0).clamp_min(1e-6)
        comp_emb = self.mlp_emb(x_comp)
        state_emb = self.state_embedding(x_state.long())
        x_in = torch.cat([comp_emb, x_total_feats, state_emb], dim=-1)
        vae_out = self.vae(x_in)
        z = vae_out["z"]
        mu, v, alpha, beta = self.predictor(z)
        vae_out["bandgap_pred"] = {
            "mu": mu,
            "v": v,
            "alpha": alpha,
            "beta": beta
        }
        return vae_out, z


def vae_reconstruction_loss(x_comp, comp_hat):
    return F.mse_loss(comp_hat, x_comp)


def vae_kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y_bandgap,
                                 alpha_v=1.0, beta_v=1.0):
    out, z = model(x_comp, x_total, x_state)
    rec_loss = vae_reconstruction_loss(x_comp, out["comp_hat"])
    kld = vae_kl_loss(out["mu"], out["logvar"])
    vae_loss = rec_loss + 1e-3 * kld
    mu = out["bandgap_pred"]["mu"]
    v = out["bandgap_pred"]["v"]
    alpha = out["bandgap_pred"]["alpha"]
    beta = out["bandgap_pred"]["beta"]
    mae = torch.mean(torch.abs(mu - y_bandgap))
    ev_loss = evidential_regression_loss(mu, v, alpha, beta, y_bandgap)
    total_loss = alpha_v * vae_loss + beta_v * ev_loss
    return total_loss, {
        "vae_loss": vae_loss.item(),
        "pred_loss": ev_loss.item(),
        "mae": mae.item(),
        "pred": mu.detach()
    }

