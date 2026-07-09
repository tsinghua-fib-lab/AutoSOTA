import torch
import torch.nn as nn
import torch.nn.functional as F

from .MOON import Ad_inverse_approx, log_Cd_approx


def _normalize_clip_prototypes(clip_prototypes):
    # Expected shape is [T, D, K], commonly T=1 in this codebase.
    cp = clip_prototypes.cuda().float()
    if cp.dim() == 2:
        cp = cp.unsqueeze(0)
    cp = F.normalize(cp, p=2, dim=1)
    return cp


def _zero_shot_logits(query_features, clip_prototypes):
    logits = 100.0 * torch.matmul(query_features, clip_prototypes)
    # If multiple prompt views exist on dim 0, average them.
    while logits.dim() > 2:
        logits = logits.mean(dim=0)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits


def _entropy_weight(z):
    # gamma_i = 1 - H(z_i)/log(K)
    entropy = -(z * torch.log(z + 1e-12)).sum(dim=1)
    normalizer = torch.log(torch.tensor(float(z.size(1)), dtype=z.dtype, device=z.device))
    return 1.0 - entropy / (normalizer + 1e-12)


class MOON_online_solver(nn.Module):
    """
    Sample-wise online extension of MOON with a memory bank.

    Memory bank stores historical (feature, soft label, weight), and maintains
    sufficient statistics for anchored vMF parameter updates.
    """

    def __init__(
        self,
        K,
        d,
        alpha=1.0,
        soft_beta=False,
        lambda_y_hat=1.0,
        lambda_laplacian=1.0,
        n_neighbors=3,
        bank_capacity=128,
        temperature=100.0,
    ):
        super().__init__()
        self.K = int(K)
        self.d = int(d)
        self.alpha = float(alpha)
        self.soft_beta = bool(soft_beta)
        self.lambda_y_hat = float(lambda_y_hat)
        self.lambda_laplacian = float(lambda_laplacian)
        self.n_neighbors = int(n_neighbors)
        self.bank_capacity = int(bank_capacity)
        self.temperature = float(temperature)

        self.init_prototypes = None  # [K, 1, D]
        self.mu = None               # [K, 1, D]
        self.kappa = None            # [K, 1]

        # Circular bank buffers
        self.bank_features = None    # [B, D]
        self.bank_z = None           # [B, K]
        self.bank_gamma = None       # [B]
        self.bank_hard = None        # [B, K]
        self.bank_size = 0
        self.bank_ptr = 0

        # Sufficient statistics
        self.S = None                # [K, D]
        self.n_soft = None           # [K]
        self.n_hard = None           # [K]

    def reset(self):
        self.mu = None
        self.kappa = None

        self.bank_features = None
        self.bank_z = None
        self.bank_gamma = None
        self.bank_hard = None
        self.bank_size = 0
        self.bank_ptr = 0

        self.S = None
        self.n_soft = None
        self.n_hard = None

    def _ensure_state(self, clip_prototypes):
        cp = _normalize_clip_prototypes(clip_prototypes)
        if self.init_prototypes is None:
            # [T, D, K] -> [K, 1, D]
            self.init_prototypes = cp.permute(2, 0, 1).mean(dim=1, keepdim=True)

        device = self.init_prototypes.device
        dtype = self.init_prototypes.dtype

        if self.mu is None:
            self.mu = self.init_prototypes.clone()
        if self.kappa is None:
            self.kappa = torch.ones((self.K, 1), device=device, dtype=dtype)

        if self.S is None:
            self.S = torch.zeros((self.K, self.d), device=device, dtype=dtype)
            self.n_soft = torch.zeros((self.K,), device=device, dtype=dtype)
            self.n_hard = torch.zeros((self.K,), device=device, dtype=dtype)

        if self.bank_features is None:
            cap = max(1, self.bank_capacity)
            self.bank_features = torch.zeros((cap, self.d), device=device, dtype=dtype)
            self.bank_z = torch.zeros((cap, self.K), device=device, dtype=dtype)
            self.bank_gamma = torch.zeros((cap,), device=device, dtype=dtype)
            self.bank_hard = torch.zeros((cap, self.K), device=device, dtype=dtype)

        return cp

    def _current_bank(self):
        if self.bank_size == 0:
            return None, None
        return self.bank_features[:self.bank_size], self.bank_z[:self.bank_size]

    def _bank_message(self, feat):
        # feat: [1, D]
        if self.bank_size == 0 or self.lambda_laplacian == 0:
            return torch.zeros((1, self.K), device=feat.device, dtype=feat.dtype)

        bank_features, bank_z = self._current_bank()
        sim = torch.matmul(feat, bank_features.T).squeeze(0)  # [B]

        if self.n_neighbors > 0 and sim.numel() > self.n_neighbors:
            vals, idx = torch.topk(sim, k=self.n_neighbors, largest=True)
            z_neighbors = bank_z[idx]
            weights = torch.softmax(vals, dim=0).unsqueeze(0)
        else:
            z_neighbors = bank_z
            weights = torch.softmax(sim, dim=0).unsqueeze(0)

        return self.lambda_laplacian * torch.matmul(weights, z_neighbors)

    def _predict_single(self, feat, y_hat):
        # feat: [1, D], y_hat: [1, K]
        if self.bank_size == 0:
            return y_hat

        core = torch.matmul(feat, self.mu.squeeze(1).T) * self.kappa.T
        norm_const = log_Cd_approx(self.kappa, self.d).T
        log_p = core + norm_const

        lap_msg = self._bank_message(feat)
        prior = torch.clamp(y_hat, min=1e-12) ** self.lambda_y_hat
        logits = torch.exp((log_p + lap_msg) / self.temperature)
        z = prior * logits
        z = z / (z.sum(dim=1, keepdim=True) + 1e-12)
        return z

    def _update_params_from_stats(self):
        eps = 1e-12
        mu_p = self.init_prototypes.squeeze(1)

        v = self.S / (self.n_soft.unsqueeze(1) + eps)
        counts = self.n_soft if self.soft_beta else self.n_hard
        beta = counts / (counts + self.alpha + eps)

        resultant = beta.unsqueeze(1) * v + (1.0 - beta.unsqueeze(1)) * mu_p
        self.mu = F.normalize(resultant, p=2, dim=-1).unsqueeze(1)

        r_bar = torch.linalg.norm(resultant, dim=1, keepdim=True)
        r_bar = torch.clamp(r_bar, min=1e-6, max=0.999)
        self.kappa = torch.clamp(Ad_inverse_approx(r_bar, self.d), min=1e-6, max=500.0)

    def _insert_bank(self, feat, z):
        # feat: [D], z: [K]
        gamma = _entropy_weight(z.unsqueeze(0)).squeeze(0)
        hard = F.one_hot(torch.argmax(z).to(torch.int64), num_classes=self.K).to(z.dtype)

        if self.bank_size < self.bank_capacity:
            idx = self.bank_ptr
            self.bank_size += 1
            self.bank_ptr = (self.bank_ptr + 1) % self.bank_capacity
        else:
            idx = self.bank_ptr
            old_feat = self.bank_features[idx]
            old_z = self.bank_z[idx]
            old_gamma = self.bank_gamma[idx]
            old_hard = self.bank_hard[idx]

            old_soft = old_gamma * old_z
            self.S -= old_soft.unsqueeze(1) * old_feat.unsqueeze(0)
            self.n_soft -= old_soft
            self.n_hard -= old_hard

            self.bank_ptr = (self.bank_ptr + 1) % self.bank_capacity

        self.bank_features[idx] = feat
        self.bank_z[idx] = z
        self.bank_gamma[idx] = gamma
        self.bank_hard[idx] = hard

        weighted_z = gamma * z
        self.S += weighted_z.unsqueeze(1) * feat.unsqueeze(0)
        self.n_soft += weighted_z
        self.n_hard += hard

    @torch.no_grad()
    def forward(self, query_features, query_labels, clip_prototypes):
        # query_labels is unused but kept for consistent solver interface.
        del query_labels

        query_features = F.normalize(query_features.cuda().float(), p=2, dim=-1)
        cp = self._ensure_state(clip_prototypes)

        zs_logits = _zero_shot_logits(query_features, cp)
        y_hat = F.softmax(zs_logits, dim=1)

        preds = torch.empty_like(y_hat)
        for i in range(query_features.size(0)):
            feat_i = query_features[i : i + 1]
            y_i = y_hat[i : i + 1]

            z_i = self._predict_single(feat_i, y_i)
            preds[i : i + 1] = z_i

            self._insert_bank(feat_i.squeeze(0), z_i.squeeze(0))
            self._update_params_from_stats()

        return y_hat.cpu(), preds.cpu()

    @torch.no_grad()
    def predict_without_update(self, query_features, clip_prototypes, batch_size=256):
        """Predict with fixed bank stats, without updating memory/parameters."""
        query_features = F.normalize(query_features.cuda().float(), p=2, dim=-1)
        cp = self._ensure_state(clip_prototypes)

        all_y_hat = []
        all_pred = []
        for start in range(0, query_features.size(0), batch_size):
            end = min(start + batch_size, query_features.size(0))
            feat = query_features[start:end]

            zs_logits = _zero_shot_logits(feat, cp)
            y_hat = F.softmax(zs_logits, dim=1)

            pred = torch.empty_like(y_hat)
            for i in range(feat.size(0)):
                pred[i : i + 1] = self._predict_single(feat[i : i + 1], y_hat[i : i + 1])

            all_y_hat.append(y_hat)
            all_pred.append(pred)

        return torch.cat(all_y_hat, dim=0).cpu(), torch.cat(all_pred, dim=0).cpu()
