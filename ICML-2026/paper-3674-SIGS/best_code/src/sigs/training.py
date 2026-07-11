

import math, os, random
from typing import Dict, Tuple, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd.functional import hvp

from contextlib import contextmanager, nullcontext

# ---------------- optional topo dep (PH@scale) ----------------
try:
    from torch_tda.nn import RipsLayer
    _HAS_TDA = True
except Exception:
    _HAS_TDA = False
    RipsLayer = None  # type: ignore

from sigs.model import GrammarVAE

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
@contextmanager
def no_param_grad(module: nn.Module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters(): p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(module.parameters(), flags): p.requires_grad_(r)

def _finite(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))

def load_data(data_path: str) -> TensorDataset:
    import h5py
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with h5py.File(data_path, "r") as f:
        raw = torch.from_numpy(f["data"][:]).float()   # [B, 1, L, C] or [B, L, C]
    data = raw.squeeze(1).transpose(1, 2)              # -> [B, C, L]
    targets = data.argmax(1)                           # -> [B, L]
    return TensorDataset(data, targets)

def load_data_with_splits(
    data_path: str, val_split: float = 0.2, test_split: float = 0.1
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    ds = load_data(data_path)
    n = len(ds)
    n_test = int(test_split * n)
    n_val  = int(val_split  * n)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(42)
    train, val, test = random_split(ds, [n_train, n_val, n_test], generator=gen)
    return train, val, test

class LinearWarmupKL:
    def __init__(self, warmup_updates: int, beta0: float = 0.01):
        self.warm = int(warmup_updates); self.beta0 = float(beta0)
    def alpha(self, step: int) -> float:
        if self.warm <= 0: return 1.0
        frac = min(step / float(self.warm), 1.0)
        return self.beta0 + (1.0 - self.beta0) * frac

# --------------------------------------------------------------------------
# GEO regularizers: δ-net reservoir + direction set (hull) + PH@scale helpers
# --------------------------------------------------------------------------
class _ScaleBuf:
    """Maintains a δ-separated δ-net (greedy farthest-point)."""
    def __init__(self, delta: float):
        self.delta = float(delta)
        self.buf: List[torch.Tensor] = []
    def maybe_insert(self, pt: torch.Tensor):
        # CHANGE: keep on CPU in fp32 to be AMP-safe
        pt = pt.to("cpu", dtype=torch.float32)
        if not self.buf:
            self.buf.append(pt); return
        centres = torch.stack(self.buf, 0)  # CPU fp32
        dmin = torch.cdist(pt.unsqueeze(0), centres, p=2.0).amin()
        if dmin > self.delta:
            self.buf.append(pt)
    def tensor(self, device: torch.device) -> torch.Tensor:
        if not self.buf: return torch.empty(0, device=device, dtype=torch.float32)
        return torch.stack(self.buf, 0).to(device=device, dtype=torch.float32)

class MultiScaleReservoir:
    """Streaming δ-net at multiple scales; update with observed latents."""
    def __init__(self, deltas: List[float]):
        self.bufs = [_ScaleBuf(d) for d in sorted(deltas)]
    @torch.no_grad()
    def update(self, Z: torch.Tensor):
        # CHANGE: Always collect on CPU in fp32 (AMP-safe)
        Zcpu = Z.detach().to("cpu", dtype=torch.float32)
        for z in Zcpu:
            for buf in self.bufs:
                buf.maybe_insert(z)
    def all_points(self, device: torch.device) -> torch.Tensor:
        parts = [b.tensor(device) for b in self.bufs if b.buf]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, device=device, dtype=torch.float32)
    def delta_min(self) -> float:
        return min(b.delta for b in self.bufs) if self.bufs else 0.1

class DirectionSet(nn.Module):
    """Fixed U_K ⊂ S^{d-1}; can only grow (no re-sampling)."""
    def __init__(self, d: int, K: int, seed: int = 0, device: Optional[torch.device] = None):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)
        U = torch.randn(K, d, generator=g, device=device if device is not None else "cpu")
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U = U.to(torch.float32)
        # CHANGE: only store U (no U_T buffer) to avoid state_dict mismatch
        self.register_buffer("U", U)  # [K, d]
    @torch.no_grad()
    def grow(self, extra: int, seed: int = 0):
        g = torch.Generator(device=self.U.device).manual_seed(seed)
        V = torch.randn(extra, self.U.size(1), generator=g, device=self.U.device)
        V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
        V = V.to(torch.float32)
        self.U = torch.cat([self.U, V], dim=0)

# --------------------------------------------------------------------------
# LightningModule with GEO loss (Hadamard-ish latent regularization)
# --------------------------------------------------------------------------
class GrammarVAEModel(pl.LightningModule):
    """
    Grammar-VAE training module (paper §3.1 + Appendix A.3).

    Optimizes ELBO = reconstruction loss + β·KL divergence with linear warmup,
    optionally augmented by Geometric regularization (GEO) after val accuracy
    crosses acc_threshold.  GEO terms: convex-hull barrier (ℒ_hull),
    persistent-homology penalty (ℒ_PH), midpoint interior (ℒ_mid),
    Hessian flatness (ℒ_hess), encoder gradient penalty (ℒ_GP).
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config

        # --- core model / schedule ---
        self.model = GrammarVAE(config)
        self.criterion = nn.CrossEntropyLoss()
        tcfg = config["training"]
        self.lr = float(tcfg["learning_rate"])
        self.anneal = LinearWarmupKL(
            warmup_updates=int(tcfg.get("warmup_updates", 7000)),
            beta0=float(tcfg.get("beta0", 0.01)),
        )

        # --- topo config / gates ---
        topo = config.get("topo", {})
        self.enable_topo = False
        self._topo_enabled_epoch: Optional[int] = None
        self.acc_threshold  = float(topo.get("acc_threshold", 50.0))
        self.gamma_base     = float(topo.get("gamma", 1e-2))
        self.ramp_epochs    = int(topo.get("ramp_epochs", 5))
        self.train_every    = int(topo.get("train_every", 100))
        self.val_every      = int(topo.get("val_every", 18))

        # weights inside topo block
        self.w_hull = float(topo.get("w_hull", 1.0))
        self.w_ph   = float(topo.get("w_ph",   1.0))
        self.w_mid  = float(topo.get("w_mid",  0.0))
        self.w_hess = float(topo.get("w_hess", 1e-5))  # CHANGE: training-only
        self.w_gp   = float(topo.get("w_gp",   0.0))   # optional encoder GP
        self.ph_connectivity_weight = float(topo.get("ph_connectivity_weight", 0.1))

        # PH & reservoir
        deltas = tuple(map(float, topo.get("deltas", [0.1, 0.5])))
        self.reservoir = MultiScaleReservoir(list(deltas))
        self.ph_maxdim = int(topo.get("ph_max_dimension", 1))
        self.rips_cpu = RipsLayer(maxdim=self.ph_maxdim).to("cpu") if _HAS_TDA else None

        # hull directions (use latent dim from config)
        z_dim = int(config["model"]["shared"]["z_dim"])
        self.directions = DirectionSet(z_dim, int(topo.get("K_dir", 256)), seed=0)

        # curvature / Hessian settings
        self.hessian_probes = int(topo.get("hessian_probes", 8))
        self.midpoint_chords = int(topo.get("curv_pair_budget", 32))
        self.curv_k_out = int(topo.get("curv_k_out", 4))

        self._curv_idx = None  # for decoder probe

        # cached validators
        self._val_recs: List[torch.Tensor] = []
        self._val_kls: List[torch.Tensor]  = []
        self._val_topos: List[float]       = []
        self._val_elbos: List[torch.Tensor]= []
        self._val_accs: List[torch.Tensor] = []

        # misc
        self.n_dec_samples = int(self.cfg["training"].get("decoder_samples", 3))
        self.ph_max_points = int(topo.get("ph_max_points", 64))

    # ------------------------ AMP guard (for GEO) ------------------------
    def _no_amp_ctx(self):
        # Disable AMP for GEO computations to avoid 'Half' ops
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", enabled=False)
        return nullcontext()

    # ------------------------ data ------------------------
    def setup(self, stage: Optional[str] = None):
        data_path  = self.cfg["data"]["data_path"]
        val_split  = float(self.cfg["data"].get("validation_split", 0.2))
        test_split = float(self.cfg["data"].get("test_split", 0.1))
        self.train_dataset, self.val_dataset, self.test_dataset = load_data_with_splits(
            data_path, val_split, test_split
        )

    # ------------------------ forward ---------------------
    def forward(self, x):
        # Encoder → (μ, log σ²); reparameterize → z; GRU decoder → rule logits
        mu, logvar = self.model.encoder(x)
        z = self.model.sample(mu, logvar, num_samples=self.n_dec_samples)
        logits = self.model.decoder(z)
        return logits, mu, logvar, z

    # ── GEO regularization terms (Appendix A.3) ──────────────────────────────
    def _decode_logits_single(self, z1d: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adapter: decoder expects [B,S,Z]; z1d is [Z]. Return [L,C] or [L,k_out].
        Run in fp32 to keep HVP stable.
        """
        Z = z1d.to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,Z]
        with cudnn.flags(enabled=False):
            out = self.model.decoder(Z)                      # [1,1,L,C]
        out = out[0, 0]                                      # [L,C]
        if idx is not None:
            out = out[..., idx]                              # [L,k_out]
        return out

    def _hull_loss_against(self, z_lat: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        if R.numel() == 0: return z_lat.new_tensor(0.0, dtype=torch.float32)
        U = self.directions.U.to(device=z_lat.device, dtype=torch.float32)  # [K,d]
        h = (R @ U.T).amax(dim=0)                           # [K]
        slack = (z_lat @ U.T) - h                           # [B,K]
        return (slack.clamp_min(0.0) ** 2).mean()

    @staticmethod
    def _subsample_points(pts: torch.Tensor, k: int) -> torch.Tensor:
        n = pts.size(0)
        if n <= k:
            return pts
        idx = torch.randperm(n, device=pts.device)[:k]
        return pts[idx]

    def _ph_loss_against(self, z_lat: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        PH@scale penalty computed on CPU (torch-tda converts to NumPy internally).
        We subsample to <= ph_max_points, run Rips on CPU, then return a scalar on
        the original device (fp32).
        """
        if self.rips_cpu is None:
            return z_lat.new_tensor(0.0, dtype=torch.float32)

        P = torch.cat([z_lat.detach(), R], dim=0) if R.numel() else z_lat.detach()
        if self.ph_max_points > 0:
            P = self._subsample_points(P, self.ph_max_points)
        if P.numel() == 0:
            return z_lat.new_tensor(0.0, dtype=torch.float32)

        P_cpu = P.contiguous().to("cpu", dtype=torch.float32)
        diagrams = self.rips_cpu(P_cpu)   # list of CPU tensors [H0, H1, ...]
        r = math.sqrt(2.0) * self.reservoir.delta_min()

        loss_cpu = torch.tensor(0.0, device="cpu", dtype=torch.float32)

        # H1: penalize short loops (those that die by radius r)
        if len(diagrams) > 1 and diagrams[1] is not None and diagrams[1].numel() > 0:
            dgm1 = diagrams[1]
            b, d = dgm1[:, 0], dgm1[:, 1]
            life = (d - b).clamp_min(0.0)
            mask = (d <= r)
            if mask.any():
                loss_cpu = loss_cpu + (life[mask] ** 2).sum()

        # H0: light connectivity encouragement at the same radius
        if len(diagrams) > 0 and diagrams[0] is not None and diagrams[0].numel() > 0:
            dgm0 = diagrams[0]
            b0, d0 = dgm0[:, 0], dgm0[:, 1]
            life0 = (d0 - b0).clamp_min(0.0)
            mask0 = (d0 <= r)
            if mask0.any():
                loss_cpu = loss_cpu + self.ph_connectivity_weight * (life0[mask0] ** 2).sum()

        return loss_cpu.to(device=z_lat.device, dtype=torch.float32)

    def _midpoint_loss_against(self, z_lat: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        if (self.w_mid <= 0.0) or (z_lat.size(0) < 2) or (R.numel() == 0):
            return z_lat.new_tensor(0.0, dtype=torch.float32)
        B = z_lat.size(0)
        chords = min(self.midpoint_chords, B * (B - 1) // 2)
        if chords <= 0: return z_lat.new_tensor(0.0, dtype=torch.float32)

        idx1 = torch.randint(0, B, (chords,), device=z_lat.device)
        idx2 = torch.randint(0, B, (chords,), device=z_lat.device)
        mask = (idx2 != idx1)
        if mask.sum() == 0: return z_lat.new_tensor(0.0, dtype=torch.float32)
        idx1, idx2 = idx1[mask], idx2[mask]
        tau = torch.rand(idx1.numel(), device=z_lat.device)
        M = (1.0 - tau).unsqueeze(1) * z_lat[idx1] + tau.unsqueeze(1) * z_lat[idx2]  # [C,d]

        # CHANGE: fp32 for cdist (AMP-safe)
        M32 = M.to(torch.float32)
        R32 = R.to(torch.float32)
        d = torch.cdist(M32, R32, p=2.0).amin(dim=1)
        delta = self.reservoir.delta_min()
        return ((d - delta).clamp_min(0.0).pow(2)).mean()

    def _hessian_penalty(self, z_lat: torch.Tensor) -> torch.Tensor:
        # CHANGE: We'll call this ONLY in training (see compute_loss)
        if z_lat.numel() == 0 or self.w_hess <= 0.0:
            return z_lat.new_tensor(0.0, dtype=torch.float32)
        if self._curv_idx is None:
            V = int(self.cfg["model"]["shared"]["output_size"])
            k = min(self.curv_k_out, V)
            self._curv_idx = torch.randperm(V, device=z_lat.device)[:k]

        total = z_lat.new_tensor(0.0, dtype=torch.float32)
        probes = max(1, self.hessian_probes)

        def f(zz):  # use subset
            return self._decode_logits_single(zz.to(torch.float32), idx=self._curv_idx).sum()

        for zi in z_lat:
            zi = zi.detach().to(torch.float32).requires_grad_(True)
            acc = 0.0
            for _ in range(probes):
                v = torch.randn_like(zi, dtype=torch.float32)
                Hv = hvp(f, zi, v)[1]   # [d]
                acc = acc + Hv.pow(2).sum()
            total = total + acc / float(probes)
        return total / float(z_lat.size(0))

    def _enc_grad_penalty(self, x: torch.Tensor) -> torch.Tensor:
        # Can be training-only to save time; keep here but gate in compute_loss
        if self.w_gp <= 0.0: return x.new_tensor(0.0, dtype=torch.float32)
        x = x.detach().requires_grad_(True)
        mu, _ = self.model.encoder(x)
        v = torch.randn_like(mu)
        s = (mu * v).sum()
        (g,) = torch.autograd.grad(s, x, create_graph=False)
        return g.pow(2).sum(dim=1).mean()

    # ── ELBO loss (paper Eq. 5) ───────────────────────────────────────────────
    def compute_loss(self, logits, targets, mu, logvar, step: int, topo_now: bool):
        # reconstruction: cross-entropy over grammar rule predictions
        rec = self.criterion(
            logits.mean(1).view(-1, logits.size(-1)),
            targets.view(-1)
        ) * targets.size(1)

        # KL with linear warmup β-schedule
        kl = self.model.kl_divergence(mu, logvar)
        beta = self.anneal.alpha(step)
        try:
            self.log("beta", float(beta), on_step=False, on_epoch=True)
            self.log("beta_kl", (float(beta) * kl).detach() if isinstance(kl, torch.Tensor) else torch.tensor(float(beta * kl), device=self.device), on_step=False, on_epoch=True)
        except Exception:
            pass

        topo_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        if topo_now and self.enable_topo:
            with self._no_amp_ctx():
                z32 = (mu if mu.requires_grad else mu.detach()).to(torch.float32)
                R_prev = self.reservoir.all_points(self.device).to(torch.float32).detach()

                L_hull = _finite(self._hull_loss_against(z32, R_prev)) if self.w_hull > 0 else z32.new_tensor(0.0)
                L_ph   = _finite(self._ph_loss_against(z32, R_prev))   if (self.w_ph > 0 and _HAS_TDA) else z32.new_tensor(0.0)
                L_mid  = _finite(self._midpoint_loss_against(z32, R_prev)) if self.w_mid > 0 else z32.new_tensor(0.0)

                # HVP and enc-GP only during training (too costly for validation)
                do_hess = self.training and (self.w_hess > 0.0)
                L_hess  = _finite(self._hessian_penalty(z32)) if do_hess else z32.new_tensor(0.0)
                do_gp   = self.training and (self.w_gp > 0.0)
                L_gp    = _finite(self._enc_grad_penalty(self._last_x_for_gp)) if (do_gp and hasattr(self, "_last_x_for_gp")) else z32.new_tensor(0.0)

                # linearly ramp γ over ramp_epochs after GEO activates
                t = 0 if self._topo_enabled_epoch is None else max(0, self.current_epoch - self._topo_enabled_epoch + 1)
                ramp = min(1.0, t / max(1, self.ramp_epochs))
                gamma = float(self.gamma_base * ramp)

                topo_loss = (gamma * (
                    self.w_hull * L_hull + self.w_ph * L_ph + self.w_mid * L_mid
                    + self.w_hess * L_hess + self.w_gp * L_gp
                )).to(torch.float32)

                if self.training:
                    with torch.no_grad():
                        self.reservoir.update(z32.detach())

                self.log_dict({
                    "topo/hull": L_hull, "topo/ph": L_ph, "topo/mid": L_mid,
                    "topo/hess": L_hess, "topo/gp": L_gp, "topo/gamma": torch.tensor(gamma, device=self.device)
                }, on_step=False, on_epoch=True)

        loss = rec + beta * kl + topo_loss
        with torch.no_grad():
        
            main = rec + beta * kl
            self.log("topo/fraction", (topo_loss / (main + 1e-8)).detach(),
                    on_step=False, on_epoch=True)
        return loss, rec, kl, topo_loss

    # ── Training / validation steps ───────────────────────────────────────────
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        self._last_x_for_gp = x
        logits, mu, logvar, z = self(x)
        topo_now = self.enable_topo and (self.global_step % self.train_every == 0)

        loss, rec, kl, topo = self.compute_loss(logits, y, mu, logvar, self.global_step, topo_now)
        token_acc = self.compute_token_accuracy(logits, y)
        seq_acc = self.compute_sequence_accuracy(logits, y)

        self.log("train_loss_full", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_rec", rec, on_step=False, on_epoch=True)
        self.log("train_kl", kl, on_step=False, on_epoch=True)
        self.log("train_topo", topo, on_step=False, on_epoch=True)
        self.log("train_acc", seq_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_token_acc", token_acc, on_step=False, on_epoch=True)
        return loss
    def on_validation_epoch_start(self):
        # initialize per-epoch accumulators
        self._val_recs = []
        self._val_kls = []
        self._val_topos = []
        self._val_elbos = []
        self._val_accs = []
        self._val_token_accs = []

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits, mu, logvar, _ = self(x)
        topo_now = self.enable_topo and ((batch_idx + 1) % self.val_every == 0)
        loss, rec, kl, topo = self.compute_loss(logits, y, mu, logvar, self.global_step, topo_now)

        token_acc = self.compute_token_accuracy(logits, y)
        acc = self.compute_sequence_accuracy(logits, y)

        self._val_recs.append(rec.detach())
        self._val_kls.append(kl.detach())
        self._val_topos.append(float(topo.detach().cpu()))
        self._val_elbos.append(loss.detach())
        self._val_accs.append(torch.tensor(acc, device=self.device))
        self._val_token_accs.append(torch.tensor(token_acc, device=self.device))

        self.log("val_rec", rec, on_step=False, on_epoch=True)
        self.log("val_kl", kl, on_step=False, on_epoch=True)
        self.log("val_topo", topo, on_step=False, on_epoch=True)
        self.log("val_elbo_full", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_token_acc", token_acc, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        recs = torch.stack(self._val_recs)
        kls = torch.stack(self._val_kls)
        topos = torch.tensor(self._val_topos, device=self.device)
        elbos = torch.stack(self._val_elbos)
        accs = torch.stack(self._val_accs)
        token_accs = torch.stack(self._val_token_accs) if len(self._val_token_accs) > 0 else torch.tensor([], device=self.device)

        avg_rec = recs.mean().item()
        avg_kl = kls.mean().item()
        avg_topo = topos.mean().item()
        current_beta = self.anneal.alpha(self.global_step)
        avg_elbo_full = avg_rec + current_beta * avg_kl + avg_topo
        avg_elbo = elbos.mean().item()
        avg_acc = accs.mean().item()
        avg_token_acc = token_accs.mean().item() if token_accs.numel() else float('nan')

        self.log("val_elbo_simple", avg_rec + avg_kl)
        self.log("val_elbo_full_epoch", avg_elbo_full)
        self.log("val_rec_epoch", avg_rec)
        self.log("val_kl_epoch", avg_kl)
        self.log("val_topo_epoch", avg_topo)
        self.log("val_elbo_epoch", avg_elbo)
        self.log("val_acc_epoch", avg_acc)
        self.log("val_token_acc_epoch", avg_token_acc)

        # Gate GEO regularization on reconstruction accuracy
        if not getattr(self.trainer, "sanity_checking", False):
            if not self.enable_topo and avg_acc >= self.acc_threshold:
                self.enable_topo = True
                self._topo_enabled_epoch = 0 if self.acc_threshold <= 0.0 else self.current_epoch

                # RESET SCHEDULER BASELINE when GEO activates
                if hasattr(self.trainer.lr_scheduler_configs[0].scheduler, 'best'):
                    old_best = self.trainer.lr_scheduler_configs[0].scheduler.best
                    self.trainer.lr_scheduler_configs[0].scheduler.best = avg_elbo_full
                    self.print(f"GEO enabled at val acc = {avg_acc:.1f}% (ramp {self.ramp_epochs} epochs)")
                    self.print(f"Scheduler baseline reset from {old_best:.3f} to {avg_elbo_full:.3f}")
                else:
                    self.print(f"GEO enabled at val acc = {avg_acc:.1f}% (ramp {self.ramp_epochs} epochs)")

                self.log("topo/enabled", 1.0, prog_bar=True)

        print(f"Epoch {self.current_epoch:4d} | rec={avg_rec:.4f} kl={avg_kl:.4f} topo={avg_topo:.2e} acc={avg_acc:.2f}% geo={'on' if self.enable_topo else 'off'}")


    # ── Accuracy metrics ──────────────────────────────────────────────────────
    def compute_sequence_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Full-sequence (strict) accuracy: fraction of sequences where every token matches."""
        if logits.ndim == 4: logits = logits.mean(1)
        preds = torch.argmax(logits, dim=-1)
        return (preds == targets).all(dim=1).float().mean().item() * 100.0

    def compute_token_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Per-token accuracy averaged over all positions and batch elements."""
        if logits.ndim == 4: logits = logits.mean(1)
        preds = torch.argmax(logits, dim=-1)
        return (preds == targets).float().mean().item() * 100.0

    # ── Optimiser / dataloaders ───────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_elbo_full_epoch"}} 

    def train_dataloader(self):
        bs = int(self.cfg["training"]["batch_size"])
        return DataLoader(self.train_dataset, batch_size=bs, shuffle=True, num_workers=4)

    def val_dataloader(self):
        bs = int(self.cfg["training"]["batch_size"])
        return DataLoader(self.val_dataset, batch_size=bs, num_workers=4)

    def test_dataloader(self):
        bs = int(self.cfg["training"]["batch_size"])
        return DataLoader(self.test_dataset, batch_size=bs, num_workers=4)

