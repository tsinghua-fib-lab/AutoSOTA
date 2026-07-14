"""
train.py

Training / validation utilities for the DAG model.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from model import ContentUncertaintyDAG
from obj import compute_elbo
from utils import LOG_VAR_MIN, LOG_VAR_MAX, _log_normal_diag


# -------------------- small utilities --------------------


def _unpack_batch(batch):
    """
    Supports:
      - (x, y)
      - (x, y, assignedGroup)
      - (sid, x, y, assignedGroup)
      - (x, y, z_true, u_true, m_true, c_true)  [synthetic]
      - (sid, x, y, z_true, u_true, m_true, c_true)  [synthetic with ids]

    Returns:
      sid, x, y, z_true, u_true, m_true, c_true
    where sid may be None, and truth tensors may be None for real data.
    """
    n = len(batch)
    sid = None
    z_true = u_true = m_true = c_true = None

    if n == 2:
        x, y = batch
    elif n == 3:
        x, y, c_true = batch
    elif n == 4:
        maybe_sid = batch[0]
        if isinstance(maybe_sid, (str, bytes)) or (
            isinstance(maybe_sid, list) and len(maybe_sid) > 0 and isinstance(maybe_sid[0], str)
        ):
            sid, x, y, c_true = batch
    elif n == 6:
        x, y, z_true, u_true, m_true, c_true = batch
    elif n == 7:
        maybe_sid = batch[0]
        if isinstance(maybe_sid, (str, bytes)) or (
            isinstance(maybe_sid, list) and len(maybe_sid) > 0 and isinstance(maybe_sid[0], str)
        ):
            sid, x, y, z_true, u_true, m_true, c_true = batch

    return sid, x, y, z_true, u_true, m_true, c_true


def cluster_acc_majority(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Majority-vote mapping accuracy from clusters to classes."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")

    mapping: Dict[int, int] = {}
    for k in np.unique(y_pred):
        idx = (y_pred == k)
        if not np.any(idx):
            continue
        labels, counts = np.unique(y_true[idx], return_counts=True)
        mapping[int(k)] = int(labels[int(np.argmax(counts))])

    mapped = np.array([mapping.get(int(k), -1) for k in y_pred], dtype=np.int64)
    return float(np.mean(mapped == y_true))


def cluster_balanced_acc_majority(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy after the same majority-vote cluster -> class mapping."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")

    mapping: Dict[int, int] = {}
    for k in np.unique(y_pred):
        idx = (y_pred == k)
        if not np.any(idx):
            continue
        labels, counts = np.unique(y_true[idx], return_counts=True)
        mapping[int(k)] = int(labels[int(np.argmax(counts))])

    mapped = np.array([mapping.get(int(k), -1) for k in y_pred], dtype=np.int64)

    per_class = []
    for c in np.unique(y_true):
        idx = (y_true == c)
        if np.any(idx):
            per_class.append(float(np.mean(mapped[idx] == c)))
    return float(np.mean(per_class)) if per_class else float("nan")


def _prior_responsibilities(model: ContentUncertaintyDAG, z: torch.Tensor) -> torch.Tensor:
    """
    Responsibilities under the mixture prior p(z) = sum_k pi_k N(z; mu_k, diag(var_k)).

    Args:
      z: (B, d_Z)
    Returns:
      resp: (B, K)
    """
    pi_logits = model.pi_logits  # (K,)
    log_pi = F.log_softmax(pi_logits, dim=0)  # (K,)
    mu = model.mixture_means  # (K, d_Z)
    log_var = model.mixture_log_vars  # (K, d_Z)
    var = torch.exp(log_var)

    diff = z.unsqueeze(1) - mu.unsqueeze(0)  # (B, K, d_Z)
    log_comp = -0.5 * (
        z.shape[1] * math.log(2.0 * math.pi)
        + log_var.sum(dim=-1).unsqueeze(0)
        + (diff.pow(2) / (var.unsqueeze(0) + 1e-8)).sum(dim=-1)
    )  # (B, K)
    log_post_unnorm = log_pi.unsqueeze(0) + log_comp
    resp = F.softmax(log_post_unnorm, dim=1)
    return resp


@torch.no_grad()
def em_update_model_mixture_from_loader(
    model: ContentUncertaintyDAG,
    loader,
    device: torch.device,
    n_steps: int = 1,
    min_pi: float = 0.05,
    var_floor: float = 1e-4,
    use_q_var: bool = True,
    reinit_dead: bool = True,
    random_state: int = 0,
) -> Dict[str, float]:
    """Run (variational) EM-style updates for the model's mixture prior p(z).

    Updates model.pi_logits, model.mu_components, model.log_var_components.
    """
    model.eval()

    K = int(model.num_components)
    d_z = int(model.z_dim)
    eps = 1e-8
    log2pi = math.log(2.0 * math.pi)

    dtype_acc = torch.float64

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(random_state))

    cached_mu_z: List[torch.Tensor] = []
    cached_var_z: List[torch.Tensor] = []

    for _ in range(n_steps):
        # Current params
        pi = torch.softmax(model.pi_logits, dim=0).detach()  # (K,)
        mu_k = model.mixture_means.detach()  # (K, d)
        var_k = torch.exp(model.mixture_log_vars.detach())  # (K, d)

        Nk = torch.zeros(K, device=device, dtype=dtype_acc)
        sum_z = torch.zeros(K, d_z, device=device, dtype=dtype_acc)
        sum_z2 = torch.zeros(K, d_z, device=device, dtype=dtype_acc)

        # Global moments for reseeding / fallback variance
        sum_all = torch.zeros(d_z, device=device, dtype=dtype_acc)
        sum_all2 = torch.zeros(d_z, device=device, dtype=dtype_acc)
        n_all = 0

        for batch in loader:
            _sid, x, y, _z_true, _u_true, _m_true, _c_true = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)

            enc = model.encode(x, y)
            mu_z = enc["mu_z"].detach()  # (B, d)
            if use_q_var:
                var_z = torch.exp(enc["log_var_z"].detach().clamp(LOG_VAR_MIN, LOG_VAR_MAX))  # (B, d)
            else:
                var_z = torch.zeros_like(mu_z)

            cached_mu_z.append(mu_z.detach().cpu())
            cached_var_z.append(var_z.detach().cpu())

            B = mu_z.shape[0]
            n_all += int(B)

            mu_z_acc = mu_z.to(dtype=dtype_acc)
            var_z_acc = var_z.to(dtype=dtype_acc)

            sum_all += mu_z_acc.sum(dim=0)
            sum_all2 += (mu_z_acc.pow(2) + var_z_acc).sum(dim=0)

            # E-step: N(mu_z; mu_k, var_k + var_z)
            diff = mu_z_acc.unsqueeze(1) - mu_k.to(dtype=dtype_acc).unsqueeze(0)  # (B, K, d)
            var_eff = var_k.to(dtype=dtype_acc).unsqueeze(0) + var_z_acc.unsqueeze(1)  # (B, K, d)
            var_eff = var_eff.clamp_min(var_floor)

            log_det = torch.log(var_eff + eps).sum(dim=-1)  # (B, K)
            mahal = (diff.pow(2) / (var_eff + eps)).sum(dim=-1)  # (B, K)
            log_comp = -0.5 * (d_z * log2pi + log_det + mahal)  # (B, K)

            log_pi = torch.log(pi.to(dtype=dtype_acc) + eps).unsqueeze(0)  # (1, K)
            log_post = log_pi + log_comp
            resp = torch.softmax(log_post, dim=1)  # (B, K)

            Nk += resp.sum(dim=0)
            sum_z += (resp.unsqueeze(-1) * mu_z_acc.unsqueeze(1)).sum(dim=0)
            Ez2 = mu_z_acc.pow(2) + var_z_acc  # (B, d)
            sum_z2 += (resp.unsqueeze(-1) * Ez2.unsqueeze(1)).sum(dim=0)

        if n_all <= 0:
            return {}

        Nk_safe = Nk.clamp_min(1e-12)
        pi_new = (Nk / float(n_all)).to(dtype=dtype_acc)

        # Floor and renormalize mixture weights.
        if min_pi is not None and float(min_pi) > 0.0:
            pi_new = pi_new.clamp_min(float(min_pi))
            pi_new = pi_new / pi_new.sum().clamp_min(eps)

        mu_new = sum_z / Nk_safe.unsqueeze(-1)
        var_new = sum_z2 / Nk_safe.unsqueeze(-1) - mu_new.pow(2)
        var_new = var_new.clamp_min(float(var_floor))

        # Re-seed dead components (optional).
        if reinit_dead and min_pi is not None and float(min_pi) > 0.0:
            dead = (Nk < float(min_pi) * float(n_all) * 0.5)
            if bool(dead.any().item()):
                # Global fallback covariance from encoded points
                mean_all = sum_all / float(n_all)
                var_all = (sum_all2 / float(n_all) - mean_all.pow(2)).clamp_min(float(var_floor))

                z_pool = torch.cat(cached_mu_z, dim=0) if cached_mu_z else None
                if z_pool is not None and z_pool.numel() > 0:
                    idx = torch.randint(low=0, high=z_pool.shape[0], size=(int(dead.sum().item()),), generator=gen)
                    reseeds = z_pool[idx].to(device=device, dtype=dtype_acc)
                else:
                    reseeds = mean_all.unsqueeze(0).expand(int(dead.sum().item()), -1)

                dead_idx = torch.nonzero(dead, as_tuple=False).view(-1)
                for j, k in enumerate(dead_idx.tolist()):
                    mu_new[k] = reseeds[j]
                    var_new[k] = var_all
                    pi_new[k] = float(min_pi)
                pi_new = pi_new / pi_new.sum().clamp_min(eps)

        # Write back to model (cast to model dtype)
        with torch.no_grad():
            dtype_model = model.pi_logits.dtype
            device_model = model.pi_logits.device
            model.pi_logits.data.copy_(torch.log(pi_new.to(device=device_model, dtype=dtype_model).clamp_min(eps)))
            model.mu_components.data.copy_(mu_new.to(device=device_model, dtype=dtype_model))
            model.log_var_components.data.copy_(torch.log(var_new.to(device=device_model, dtype=dtype_model).clamp_min(eps)).clamp(LOG_VAR_MIN, LOG_VAR_MAX))

    return {}


# -------------------- core epoch runner --------------------


@torch.no_grad()
def _encode_means(
    model: ContentUncertaintyDAG,
    loader,
    device: torch.device,
) -> np.ndarray:
    """Collect mu_z for all samples in loader."""
    model.eval()
    zs: List[np.ndarray] = []
    for batch in loader:
        _sid, x, y, _z_true, _u_true, _m_true, _c_true = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        enc = model.encode(x, y)
        mu_z = enc["mu_z"].detach().cpu().numpy()
        zs.append(mu_z)
    if not zs:
        return np.zeros((0, model.z_dim), dtype=np.float32)
    return np.concatenate(zs, axis=0)


def _run_epoch(
    epoch: int,
    model: ContentUncertaintyDAG,
    loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_samples_z: int = 4,
    num_samples_u: int = 4,
    sparse_m_lambda: float = 0.0,
    sparse_m_target: Optional[float] = None,
    sparse_m_on: str = "content",
    mask_tv_lambda: float = 0.0,
    mask_tv_samples: int = 1,
    drop_all_x_prob: float = 0.0,
    x_pred_sigma_scale: float = 1.0,
    split: str = "train",
) -> Dict[str, float]:
    """Run one epoch over loader (train if optimizer is provided; else validation)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_samples = 0

    # Modality-drop stats (train only)
    dropped_x_count = 0
    eligible_x_count = 0

    # ELBO decomposition sums over samples
    sum_elbo = 0.0
    sum_LY = 0.0
    sum_LX = 0.0
    sum_KL_zc = 0.0
    sum_KL_u = 0.0
    sum_KL_m = 0.0
    sum_Sparse_M = 0.0
    sum_TV_M = 0.0

    # X metrics accumulators (continuous)
    sum_x_sse = 0.0
    sum_x = 0.0
    sum_x2 = 0.0
    sum_x_ll = 0.0
    n_x_obs = 0.0

    compute_pred_x_metrics = (model.x_distribution == "continuous") and (not is_train)

    # Accumulators for predictive metrics (continuous X only)
    sum_x_pred_sse = 0.0
    sum_x_pred = 0.0
    sum_x_pred2 = 0.0
    sum_x_pred_nll = 0.0
    sum_x_pred_picp90 = 0.0
    sum_x_pred_mpiw90 = 0.0
    n_x_pred_obs = 0.0

    # For selective prediction: store per-subject uncertainty + squared error
    x_pred_sel_unc: List[float] = []
    x_pred_sel_sse: List[float] = []
    x_pred_sel_nobs: List[float] = []

    # Y metrics accumulators
    sum_y_sse = 0.0
    sum_y_sse_weighted = 0.0
    sum_y_ll = 0.0
    n_y_pix = 0.0

    # Cluster stats
    cluster_counts = None  # np.ndarray(K,)
    total_for_cluster = 0
    c_true_all: List[np.ndarray] = []
    c_pred_all: List[np.ndarray] = []

    have_z_true = False
    have_u_true = False
    z_r2_list: List[float] = []
    u_corr_list: List[float] = []

    for batch in loader:
        _sid, x, y, z_true, u_true, m_true, c_true = _unpack_batch(batch)

        x = x.to(device)
        y = y.to(device)
        B = x.shape[0]

        if model.x_distribution == "continuous":
            mask_x_like_bool = torch.isfinite(x)
            mask_x_like = mask_x_like_bool.to(dtype=x.dtype, device=x.device)
        else:
            mask_x_like_bool = (x >= 0)
            mask_x_like = mask_x_like_bool.to(dtype=torch.float32, device=x.device)

        mask_x_enc = mask_x_like
        if is_train and drop_all_x_prob > 0.0:
            has_any = mask_x_like_bool.any(dim=1)
            eligible_x_count += int(has_any.sum().item())
            drop_mask = (torch.rand(B, device=x.device) < float(drop_all_x_prob)) & has_any
            if drop_mask.any():
                mask_x_enc = mask_x_like.clone()
                mask_x_enc[drop_mask] = 0.0
                dropped_x_count += int(drop_mask.sum().item())
        elif is_train:
            has_any = mask_x_like_bool.any(dim=1)
            eligible_x_count += int(has_any.sum().item())

        # ---- ELBO ----
        elbo_vec, terms = compute_elbo(
            model=model,
            x=x,
            y=y,
            num_samples_z=num_samples_z,
            num_samples_u=num_samples_u,
            average_over_batch=False,
            sparse_m_lambda=sparse_m_lambda,
            sparse_m_target=sparse_m_target,
            sparse_m_on=sparse_m_on,
            mask_tv_lambda=mask_tv_lambda,
            mask_tv_samples=mask_tv_samples,
            mask_x=mask_x_like,
            x_encoder=x,
            mask_x_encoder=mask_x_enc,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss = -elbo_vec.mean()
            loss.backward()
            optimizer.step()

        total_samples += B
        sum_elbo += elbo_vec.sum().item()
        sum_LY += terms["L_Y"].sum().item()
        sum_LX += terms["L_X"].sum().item()
        sum_KL_zc += terms["KL_zc"].sum().item()
        sum_KL_u += terms["KL_u"].sum().item()
        sum_KL_m += terms["KL_m"].sum().item()
        sum_Sparse_M += terms["Sparse_M"].sum().item()
        sum_TV_M += terms["TV_M"].sum().item()

        # ---- Encoder outputs ----
        enc = model.encode(x, y)
        mu_z = enc["mu_z"]  # (B, d_Z)
        mu_u = enc["mu_u"]  # (B, d_U)
        y_tokens = enc["y_tokens"]  # (B, N_patches, D)

        # ---- X metrics (continuous only) ----
        mask_x = torch.isfinite(x)
        x_obs = x[mask_x]
        if x_obs.numel() > 0:
            mean_x, log_var_x = model.decode_x_continuous(mu_z, mu_u)  # (B, d_X)
            pred_obs = mean_x[mask_x]

            diff = pred_obs - x_obs
            sum_x_sse += (diff ** 2).sum().item()
            sum_x += x_obs.sum().item()
            sum_x2 += (x_obs ** 2).sum().item()
            n_x_obs += float(x_obs.numel())

            ll = _log_normal_diag(x_obs, pred_obs, log_var_x[mask_x])
            sum_x_ll += ll.sum().item()

            if compute_pred_x_metrics:
                x_nan = torch.full_like(x, float("nan"))
                enc_y = model.encode(x_nan, y)
                mu_z_y = enc_y["mu_z"]
                log_var_z_y = enc_y["log_var_z"]
                mu_u_y = enc_y["mu_u"]
                log_var_u_y = enc_y["log_var_u"]

                A = model.x_decoder.A  # (d_X, d_Z)
                mean_pred = mu_z_y.matmul(A.t()) + mu_u_y  # (B, d_X)

                var_z = torch.exp(log_var_z_y)  # (B, d_Z)
                var_from_z = var_z.matmul(A.pow(2).t())  # (B, d_X)
                var_u = torch.exp(log_var_u_y)  # (B, d_X)
                var_eps = torch.exp(model.x_decoder.log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).unsqueeze(0).expand_as(mean_pred)
                var_pred = (var_from_z + var_u + var_eps).clamp_min(1e-8)  # (B, d_X)

                pred_obs_y = mean_pred[mask_x]
                diff_y = pred_obs_y - x_obs
                sum_x_pred_sse += (diff_y ** 2).sum().item()
                sum_x_pred += x_obs.sum().item()
                sum_x_pred2 += (x_obs ** 2).sum().item()
                n_x_pred_obs += float(x_obs.numel())

                var_obs = var_pred[mask_x]
                nll = 0.5 * (
                    (diff_y ** 2) / (var_obs + 1e-8)
                    + math.log(2.0 * math.pi)
                    + torch.log(var_obs + 1e-8)
                )
                sum_x_pred_nll += nll.sum().item()

                # 90% central prediction interval
                z90 = _z_for_central_coverage(0.90)  # two-sided 90% central interval quantile
                sigma_obs = float(x_pred_sigma_scale) * torch.sqrt(var_obs + 1e-8)
                lower = pred_obs_y - z90 * sigma_obs
                upper = pred_obs_y + z90 * sigma_obs
                covered = ((x_obs >= lower) & (x_obs <= upper)).to(torch.float32)
                sum_x_pred_picp90 += covered.sum().item()
                sum_x_pred_mpiw90 += (2.0 * z90 * sigma_obs).sum().item()

                # Selective prediction: store per-subject uncertainty + squared error
                mask_x_f = mask_x.to(dtype=mean_pred.dtype)
                x_in = torch.where(mask_x, x, torch.zeros_like(x))
                se_subj = ((mean_pred - x_in) ** 2 * mask_x_f).sum(dim=1)
                nobs_subj = mask_x_f.sum(dim=1)

                sigma = float(x_pred_sigma_scale) * torch.sqrt(var_pred + 1e-8)
                unc_subj = (sigma * mask_x_f).sum(dim=1) / (nobs_subj + 1e-12)

                valid = (nobs_subj > 0)
                x_pred_sel_unc.extend(unc_subj[valid].detach().cpu().tolist())
                x_pred_sel_sse.extend(se_subj[valid].detach().cpu().tolist())
                x_pred_sel_nobs.extend(nobs_subj[valid].detach().cpu().tolist())

        # ---- Y metrics ----
        y_flat = y.view(B, -1)
        mu_base, log_var_base = model.image_base_params(batch_size=B)
        mu_cont, log_var_cont = model.image_content_params(mu_z)

        log_var_base = log_var_base.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        log_var_cont = log_var_cont.clamp(LOG_VAR_MIN, LOG_VAR_MAX)

        g = model.mask_probs(y_tokens, mu_z).clamp(0.0, 1.0)  # (B, dY)

        y_hat = g * mu_base + (1.0 - g) * mu_cont
        sum_y_sse += ((y_hat - y_flat) ** 2).sum().item()

        # Weighted expected squared error under selector
        base_err = (mu_base - y_flat) ** 2
        cont_err = (mu_cont - y_flat) ** 2
        sum_y_sse_weighted += (g * base_err + (1.0 - g) * cont_err).sum().item()

        # Expected log-likelihood per pixel under selector (at z=mu_z)
        ell_b = _log_normal_diag(y_flat, mu_base, log_var_base)
        ell_c = _log_normal_diag(y_flat, mu_cont, log_var_cont)
        sum_y_ll += (g * ell_b + (1.0 - g) * ell_c).sum().item()
        n_y_pix += float(y_flat.numel())

        # ---- Cluster stats ----
        resp = _prior_responsibilities(model, mu_z)
        c_pred = resp.argmax(dim=1).detach().cpu().numpy()
        K = resp.shape[1]
        if cluster_counts is None:
            cluster_counts = np.zeros(K, dtype=np.int64)
        cluster_counts[:K] += np.bincount(c_pred, minlength=K)
        total_for_cluster += int(c_pred.shape[0])

        if c_true is not None:
            c_true_all.append(c_true.detach().cpu().numpy().astype(np.int64).reshape(-1))
            c_pred_all.append(np.asarray(c_pred, dtype=np.int64).reshape(-1))

        if z_true is not None:
            have_z_true = True
            z_true = z_true.to(device)
            z_diff = (mu_z - z_true)
            sse = (z_diff ** 2).sum().item()
            z_mean = z_true.mean().item()
            sst = ((z_true - z_mean) ** 2).sum().item()
            z_r2_list.append(float("nan") if sst <= 0 else float(1.0 - sse / (sst + 1e-12)))

        if u_true is not None:
            have_u_true = True
            u_true = u_true.to(device)
            u_pred_np = mu_u.detach().cpu().numpy()
            u_true_np = u_true.detach().cpu().numpy()
            corrs = []
            for d in range(u_true_np.shape[1]):
                a = u_true_np[:, d]
                b = u_pred_np[:, d]
                a = (a - a.mean()) / (a.std() + 1e-12)
                b = (b - b.mean()) / (b.std() + 1e-12)
                corrs.append(float(np.mean(a * b)))
            u_corr_list.append(float(np.mean(corrs)))

    if total_samples == 0:
        return {"epoch": epoch, "split": split}

    # --- aggregate metrics ---
    metrics: Dict[str, float] = {
        "epoch": float(epoch),
        "split": split,
        "elbo": sum_elbo / total_samples,
        "L_Y": sum_LY / total_samples,
        "L_X": sum_LX / total_samples,
        "KL_zc": sum_KL_zc / total_samples,
        "KL_u": sum_KL_u / total_samples,
        "KL_m": sum_KL_m / total_samples,
        "Sparse_M": sum_Sparse_M / total_samples,
        "TV_M": sum_TV_M / total_samples,
    }

    # X metrics
    if n_x_obs > 0:
        sst = sum_x2 - (sum_x * sum_x) / (n_x_obs + 1e-12)
        metrics.update(
            {
                "x_rmse": math.sqrt(sum_x_sse / n_x_obs),
                "x_r2": float("nan") if sst <= 0 else float(1.0 - sum_x_sse / (sst + 1e-12)),
                "x_ll_per_dim": sum_x_ll / n_x_obs,
                "x_obs_frac": float(n_x_obs / (total_samples * model.x_dim)),
            }
        )
    else:
        metrics.update({"x_rmse": float("nan"), "x_r2": float("nan"), "x_ll_per_dim": float("nan"), "x_obs_frac": 0.0})

    if compute_pred_x_metrics and n_x_pred_obs > 0:
        sst_pred = sum_x_pred2 - (sum_x_pred * sum_x_pred) / (n_x_pred_obs + 1e-12)
        x_pred_picp90 = sum_x_pred_picp90 / n_x_pred_obs
        x_pred_mpiw90 = sum_x_pred_mpiw90 / n_x_pred_obs

        x_pred_rmse_sel80 = float("nan")
        if x_pred_sel_unc:
            unc = np.asarray(x_pred_sel_unc, dtype=np.float64)
            sse = np.asarray(x_pred_sel_sse, dtype=np.float64)
            nobs = np.asarray(x_pred_sel_nobs, dtype=np.float64)
            order = np.argsort(unc)
            k = max(1, int(math.ceil(0.80 * len(order))))
            keep = order[:k]
            nobs_keep = float(nobs[keep].sum())
            if nobs_keep > 0:
                x_pred_rmse_sel80 = math.sqrt(float(sse[keep].sum()) / (nobs_keep + 1e-12))

        metrics.update(
            {
                "x_pred_rmse": math.sqrt(sum_x_pred_sse / n_x_pred_obs),
                "x_pred_r2": float("nan") if sst_pred <= 0 else float(1.0 - sum_x_pred_sse / (sst_pred + 1e-12)),
                "x_pred_nll_per_dim": sum_x_pred_nll / n_x_pred_obs,
                "x_pred_picp90": x_pred_picp90,
                "x_pred_mpiw90": x_pred_mpiw90,
                "x_pred_calerr90": abs(x_pred_picp90 - 0.90),
                "x_pred_rmse_sel80": x_pred_rmse_sel80,
                "x_pred_sigma_scale": float(x_pred_sigma_scale),
            }
        )
    else:
        metrics.update(
            {
                "x_pred_rmse": float("nan"),
                "x_pred_r2": float("nan"),
                "x_pred_nll_per_dim": float("nan"),
                "x_pred_picp90": float("nan"),
                "x_pred_mpiw90": float("nan"),
                "x_pred_calerr90": float("nan"),
                "x_pred_rmse_sel80": float("nan"),
                "x_pred_sigma_scale": float(x_pred_sigma_scale),
            }
        )

    # Y metrics
    if n_y_pix > 0:
        metrics.update(
            {
                "y_rmse_all": math.sqrt(sum_y_sse / n_y_pix),
                "y_rmse_weighted": math.sqrt(sum_y_sse_weighted / n_y_pix),
                "y_ll_mix_per_pixel": sum_y_ll / n_y_pix,
            }
        )
    else:
        metrics.update({"y_rmse_all": float("nan"), "y_rmse_weighted": float("nan"), "y_ll_mix_per_pixel": float("nan")})

    # Cluster stats
    if cluster_counts is not None and total_for_cluster > 0:
        p = cluster_counts.astype(np.float64) / float(total_for_cluster)
        p_nonzero = p[p > 0]
        metrics.update(
            {
                "cluster_entropy": float(-(p_nonzero * np.log(p_nonzero + 1e-12)).sum()),
                "cluster_min_frac": float(p.min()),
                "cluster_max_frac": float(p.max()),
            }
        )

        if c_true_all:
            y_true = np.concatenate(c_true_all, axis=0).reshape(-1)
            y_pred = np.concatenate(c_pred_all, axis=0).reshape(-1)
            acc_major = cluster_acc_majority(y_true, y_pred)
            metrics["cluster_acc"] = float(acc_major)
            metrics["cluster_acc_major"] = float(acc_major)
            metrics["cluster_bal_acc_major"] = float(cluster_balanced_acc_majority(y_true, y_pred))
            metrics["cluster_ari"] = float(adjusted_rand_score(y_true, y_pred))
        else:
            metrics["cluster_acc"] = float("nan")
            metrics["cluster_acc_major"] = float("nan")
            metrics["cluster_bal_acc_major"] = float("nan")
            metrics["cluster_ari"] = float("nan")
    else:
        metrics.update({"cluster_entropy": float("nan"), "cluster_min_frac": float("nan"), "cluster_max_frac": float("nan"), "cluster_acc": float("nan")})

    if is_train:
        metrics["drop_all_x_frac"] = float(dropped_x_count) / float(eligible_x_count) if eligible_x_count > 0 else 0.0

    metrics["z_r2"] = float(np.mean(z_r2_list)) if have_z_true and z_r2_list else float("nan")
    metrics["u_corr_mean"] = float(np.mean(u_corr_list)) if have_u_true and u_corr_list else float("nan")

    return metrics


# -------------------- post-hoc predictive interval calibration --------------------


def _z_for_central_coverage(coverage: float) -> float:
    """Return z such that P(|N(0,1)| <= z) = coverage (two-sided central interval)."""
    c = float(coverage)
    c = min(max(c, 1e-6), 1.0 - 1e-6)
    q = (1.0 + c) / 2.0
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(q, dtype=torch.float64))
    return float(z.item())


@torch.no_grad()
def fit_x_pred_sigma_scale(
    model: ContentUncertaintyDAG,
    calib_loader,
    device: torch.device,
    target_coverage: float = 0.90,
    max_points: int = 200_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Fit a scalar sigma scale for predictive intervals p(x|y) via y-only inference.

    Returns a dict with the fitted scale.
    """
    if model.x_distribution != "continuous":
        return {
            "x_pred_sigma_scale": 1.0,
            "x_pred_sigma_scale_target": float(target_coverage),
            "x_pred_sigma_scale_points": 0.0,
        }

    model.eval()
    rng = np.random.RandomState(int(seed))
    z = _z_for_central_coverage(float(target_coverage))

    ratios: List[np.ndarray] = []
    n_kept = 0

    for batch in calib_loader:
        _sid, x, y, _z_true, _u_true, _m_true, _c_true = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)

        mask_x = torch.isfinite(x)
        if not torch.any(mask_x):
            continue

        x_obs = x[mask_x]

        # y-only inference
        x_nan = torch.full_like(x, float("nan"))
        enc_y = model.encode(x_nan, y)
        mu_z_y = enc_y["mu_z"]
        log_var_z_y = enc_y["log_var_z"]
        mu_u_y = enc_y["mu_u"]
        log_var_u_y = enc_y["log_var_u"]

        A = model.x_decoder.A
        mean_pred = mu_z_y.matmul(A.t()) + mu_u_y

        var_z = torch.exp(log_var_z_y)
        var_from_z = var_z.matmul(A.pow(2).t())
        var_u = torch.exp(log_var_u_y)
        var_eps = torch.exp(model.x_decoder.log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).unsqueeze(0).expand_as(mean_pred)
        var_pred = (var_from_z + var_u + var_eps).clamp_min(1e-8)

        pred_obs = mean_pred[mask_x]
        sigma_obs = torch.sqrt(var_pred[mask_x] + 1e-8)
        diff = pred_obs - x_obs

        r = (diff.abs() / (float(z) * sigma_obs + 1e-12)).detach().cpu().numpy()
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue

        if max_points is not None and max_points > 0:
            remaining = int(max_points) - int(n_kept)
            if remaining <= 0:
                p = float(max_points) / float(max_points + r.size)
                keep = rng.rand(r.size) < p
                r = r[keep]
            else:
                if r.size > remaining:
                    r = rng.choice(r, size=remaining, replace=False)
        if r.size == 0:
            continue

        ratios.append(r)
        n_kept += int(r.size)
        if max_points is not None and max_points > 0 and n_kept >= int(max_points):
            break

    if not ratios:
        return {
            "x_pred_sigma_scale": 1.0,
            "x_pred_sigma_scale_target": float(target_coverage),
            "x_pred_sigma_scale_points": 0.0,
        }

    all_r = np.concatenate(ratios, axis=0)
    all_r = all_r[np.isfinite(all_r)]
    if all_r.size == 0:
        return {
            "x_pred_sigma_scale": 1.0,
            "x_pred_sigma_scale_target": float(target_coverage),
            "x_pred_sigma_scale_points": 0.0,
        }

    all_r.sort()
    idx = int(math.ceil(float(target_coverage) * float(all_r.size))) - 1
    idx = max(0, min(idx, int(all_r.size) - 1))
    s = max(float(all_r[idx]), 1e-6)

    return {
        "x_pred_sigma_scale": s,
        "x_pred_sigma_scale_target": float(target_coverage),
        "x_pred_sigma_scale_points": float(all_r.size),
    }


# -------------------- public API --------------------


def train(
    epoch: int,
    model: ContentUncertaintyDAG,
    optimizer: torch.optim.Optimizer,
    train_loader,
    device: torch.device,
    num_samples_z: int = 4,
    num_samples_u: int = 4,
    sparse_m_lambda: float = 0.0,
    sparse_m_target: Optional[float] = None,
    sparse_m_on: str = "content",
    mask_tv_lambda: float = 0.0,
    mask_tv_samples: int = 1,
    drop_all_x_prob: float = 0.0,
) -> Dict[str, float]:
    return _run_epoch(
        epoch=epoch,
        model=model,
        loader=train_loader,
        device=device,
        optimizer=optimizer,
        num_samples_z=num_samples_z,
        num_samples_u=num_samples_u,
        sparse_m_lambda=sparse_m_lambda,
        sparse_m_target=sparse_m_target,
        sparse_m_on=sparse_m_on,
        mask_tv_lambda=mask_tv_lambda,
        mask_tv_samples=mask_tv_samples,
        drop_all_x_prob=drop_all_x_prob,
        split="train",
    )


@torch.no_grad()
def validate(
    epoch: int,
    model: ContentUncertaintyDAG,
    val_loader,
    device: torch.device,
    num_samples_z: int = 4,
    num_samples_u: int = 4,
    sparse_m_lambda: float = 0.0,
    sparse_m_target: Optional[float] = None,
    sparse_m_on: str = "content",
    mask_tv_lambda: float = 0.0,
    mask_tv_samples: int = 1,
    drop_all_x_prob: float = 0.0,
    x_pred_sigma_scale: float = 1.0,
) -> Dict[str, float]:
    return _run_epoch(
        epoch=epoch,
        model=model,
        loader=val_loader,
        device=device,
        optimizer=None,
        num_samples_z=num_samples_z,
        num_samples_u=num_samples_u,
        sparse_m_lambda=sparse_m_lambda,
        sparse_m_target=sparse_m_target,
        sparse_m_on=sparse_m_on,
        mask_tv_lambda=mask_tv_lambda,
        mask_tv_samples=mask_tv_samples,
        drop_all_x_prob=drop_all_x_prob,
        x_pred_sigma_scale=x_pred_sigma_scale,
        split="val",
    )


def update_model_mixture_from_gmm(model: ContentUncertaintyDAG, gmm: GaussianMixture) -> None:
    """
    Copy a fitted diagonal GMM in z-space into the model's mixture prior:

        pi_logits            <- log gmm.weights_
        mu_components        <- gmm.means_
        log_var_components   <- log gmm.covariances_
    """
    with torch.no_grad():
        device = model.pi_logits.device
        dtype = model.pi_logits.dtype

        pi_np = gmm.weights_.astype(np.float32)        # (K,)
        mu_np = gmm.means_.astype(np.float32)          # (K, d_Z)
        cov_np = gmm.covariances_.astype(np.float32)   # (K, d_Z)

        pi = torch.from_numpy(pi_np).to(device=device, dtype=dtype)
        mu = torch.from_numpy(mu_np).to(device=device, dtype=dtype)
        log_var = torch.log(torch.from_numpy(cov_np).to(device=device, dtype=dtype) + 1e-8)

        model.pi_logits.data.copy_(torch.log(pi + 1e-8))
        model.mu_components.data.copy_(mu)
        model.log_var_components.data.copy_(log_var)


def _revive_gmm_components_inplace(
    gmm: GaussianMixture,
    z: np.ndarray,
    min_weight: float = 0.05,
    min_mean_sep: float = 1e-3,
) -> None:
    """Heuristic safeguard against GMM degeneracy.

    Adjusts gmm.weights_, gmm.means_, gmm.covariances_ in place.
    """
    K = int(getattr(gmm, "n_components", len(gmm.weights_)))
    if K <= 1:
        return

    w = np.asarray(gmm.weights_, dtype=np.float64).copy()
    means = np.asarray(gmm.means_, dtype=np.float64).copy()
    cov = np.asarray(gmm.covariances_, dtype=np.float64).copy()

    # Fallback covariance if we need to re-seed a component
    global_cov = np.var(z, axis=0) + 1e-4

    # 1) Revive components with tiny mixture weight
    dead = [k for k in range(K) if w[k] < float(min_weight)]
    if dead:
        dom = int(np.argmax(w))
        d2 = ((z - means[dom]) ** 2).sum(axis=1)
        order = np.argsort(d2)[::-1]  # farthest points first
        ptr = 0

        for k in dead:
            if ptr >= len(order):
                ptr = 0
            means[k] = z[order[ptr]]
            cov[k] = global_cov
            w[k] = float(min_weight)
            ptr += 1

        # renormalize the remaining mass across non-dead components
        remaining = 1.0 - len(dead) * float(min_weight)
        if remaining <= 1e-6:
            w = np.ones(K, dtype=np.float64) / float(K)
        else:
            keep = [k for k in range(K) if k not in dead]
            keep_sum = float(w[keep].sum())
            if keep_sum <= 1e-12:
                w[keep] = remaining / float(len(keep))
            else:
                w[keep] = w[keep] / keep_sum * remaining

    # 2) If component means are (almost) identical, push one mean to a far point
    min_sep = np.inf
    pair = None
    for i in range(K):
        for j in range(i + 1, K):
            dist = float(np.linalg.norm(means[i] - means[j]))
            if dist < min_sep:
                min_sep = dist
                pair = (i, j)

    if pair is not None and float(min_sep) < float(min_mean_sep):
        i, j = pair
        d2 = ((z - means[i]) ** 2).sum(axis=1)
        idx = int(np.argmax(d2))
        means[j] = z[idx]
        cov[j] = global_cov
        if w[j] < float(min_weight):
            w[j] = float(min_weight)
            w = w / float(w.sum())

    # Final sanitize
    w = np.clip(w, 1e-8, None)
    w = w / float(w.sum())
    gmm.weights_ = w
    gmm.means_ = means
    gmm.covariances_ = cov


@torch.no_grad()
def refit_model_mixture_from_loader(
    model: ContentUncertaintyDAG,
    loader,
    device: torch.device,
    n_components: Optional[int] = None,
    random_state: int = 0,
    max_iter: int = 200,
) -> GaussianMixture:
    """
    Fit a diagonal GMM on μ_z over `loader` and copy it into the model prior.

    Returns the fitted GMM.
    """
    z = _encode_means(model, loader, device=device)  # (N, d_z) numpy
    z = np.asarray(z, dtype=np.float64, order="C")

    z_mean = z.mean(axis=0, keepdims=True)
    z_std = z.std(axis=0, keepdims=True)
    z_std = np.maximum(z_std, 1e-6)
    z_w = (z - z_mean) / z_std

    K = int(n_components or model.num_components)
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        init_params="kmeans",
        reg_covar=1e-4,
        random_state=random_state,
        max_iter=max_iter,
    )
    gmm.fit(z_w)

    # Revive dead / collapsed components in whitened space
    _revive_gmm_components_inplace(gmm, z_w, min_weight=0.05, min_mean_sep=1e-3)

    # Map GMM params back to original (unwhitened) z space:
    #   z = z_mean + z_std * z_w
    gmm.means_ = (gmm.means_ * z_std) + z_mean       # (K, d)
    gmm.covariances_ = gmm.covariances_ * (z_std ** 2)  # (K, d)

    update_model_mixture_from_gmm(model, gmm)
    return gmm
