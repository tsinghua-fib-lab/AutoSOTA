# cleanup_ssps/run.py
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleanup_ssps.cleanup_methods import FlowMatching
from cleanup_ssps.dataset import SSPDataset

# OT pairing helpers (both angular and euclidean)
from utils.ot_pairs import angular_ot_pairs, euclidean_ot_pairs

torch.backends.cudnn.benchmark = True

_GEO_MODES = {"geo_det", "geo_amb_const", "geo_tan_const", "geo_amb_sb", "geo_tan_sb"}

def _renorm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def _proj_tangent(phi, u):
    # project onto tangent space at phi: u - <u,phi> phi
    return u - (u * phi).sum(dim=-1, keepdim=True) * phi

def _random_pair(z0_all, z1_all):
    # independent coupling
    idx = torch.randperm(z1_all.size(0), device=z1_all.device)
    return z0_all, z1_all[idx]


def _dataloader_kwargs(num_workers: int, device: str, prefetch_factor: int = 2) -> dict:
    nw = max(0, int(num_workers))
    kw: dict = {"num_workers": nw, "pin_memory": bool(device == "cuda")}
    if nw > 0:
        kw["prefetch_factor"] = int(prefetch_factor)
    return kw


class FlowTrainer:
    def __init__(
        self,
        encoded_dim,
        architecture,
        data_dir,
        batch_size=250,
        epochs=75,
        lr=1e-4,
        weight_decay=1e-4,
        val_split=0.1,
        signal_strength=0.0,
        noise_type='uniform_hypersphere',
        target_type='coordinate',
        device="cpu",
        # OT controls
        use_ot_train: bool = True,
        ot_method:    str  = "exact",   # "exact", "sinkhorn", or "none"
        ot_reg:       float | None = 0.05,
        ot_cost:      str  = "angular", # "angular" or "euclidean"
        # Flow-matching
        sampling_mode: str  = "geo_det",
        sigma_min:     float= 0.05,
        beta_min:      float= 0.1,
        beta_max:      float= 20.0,
        dataloader_num_workers: int | None = None,
        dataloader_prefetch_factor: int = 2,
    ):
        self.device          = device
        self.encoded_dim     = encoded_dim
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.val_split       = val_split
        self.signal_strength = signal_strength
        self.noise_type      = noise_type
        self.target_type     = target_type
        self.sampling_mode   = sampling_mode

        # --- OT policy for training couplings ---
        self.use_ot_train = bool(use_ot_train)
        self.ot_method = (ot_method or "none").lower()
        self.ot_cost   = (ot_cost or "angular").lower()

        # Only Sinkhorn needs a numeric reg; everything else → None
        if self.use_ot_train and self.ot_method == "sinkhorn":
            self.ot_reg = float(0.05 if ot_reg is None else ot_reg)
        else:
            # exact / none / disabled
            self.ot_reg = None

        self.ot_hard = False

        if dataloader_num_workers is None:
            self.dataloader_num_workers = 0 if sys.platform == "win32" else 4
        else:
            self.dataloader_num_workers = int(dataloader_num_workers)
        self.dataloader_prefetch_factor = int(dataloader_prefetch_factor)

        # Flow wrapper
        self.flow_model = FlowMatching(
            model      = architecture,
            num_steps  = self.epochs,
            sampling   = self.sampling_mode,
            device     = self.device,
            sigma_min  = sigma_min,
            beta_min   = beta_min,
            beta_max   = beta_max,
        )

        self.optimizer = torch.optim.Adam(
            self.flow_model.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Dataset split
        dataset = SSPDataset(
            data_dir        = data_dir,
            ssp_dim         = self.encoded_dim,
            target_type     = self.target_type,
            noise_type      = self.noise_type,
            signal_strength = self.signal_strength,
        )
        self.train_dataset, self.val_dataset = dataset.split_dataset(self.val_split)

        self.criterion = nn.CosineEmbeddingLoss()

    def _pair_batch(self, z0_all, z1_all):
        """Return (z0, z1) using OT or random coupling."""
        if not self.use_ot_train:
            return _random_pair(z0_all, z1_all)

        if self.ot_cost == "euclidean":
            jdx = euclidean_ot_pairs(
                z0_all, z1_all,
                reg=self.ot_reg,
                squared=True,
                hard=self.ot_hard
            )
        else:
            # default to angular if unspecified
            z0n, z1n = _renorm(z0_all), _renorm(z1_all)
            jdx = angular_ot_pairs(
                z0n, z1n,
                reg=self.ot_reg,
                squared=True,
                hard=self.ot_hard
            )
        return z0_all, z1_all[jdx]

    def _maybe_project_pred(self, z_t, u_pred):
        """Project field to tangent for geodesic modes."""
        if self.sampling_mode in _GEO_MODES:
            phi_ref = _renorm(z_t)
            return _proj_tangent(phi_ref, u_pred)
        return u_pred

    def validate(self, dataloader):
        self.flow_model.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in dataloader:
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)

                z0, z1 = self._pair_batch(z0_all, z1_all)

                z_t, t, u_true = self.flow_model.get_train_tuple(z0, z1)
                u_pred = self.flow_model.model(z_t, t)
                u_pred = self._maybe_project_pred(z_t, u_pred)

                loss = self.criterion(u_pred, u_true, torch.ones(u_pred.size(0), device=self.device))
                total += loss.item()
        return total / len(dataloader)

    def train(self):
        dkw = _dataloader_kwargs(
            self.dataloader_num_workers,
            self.device,
            self.dataloader_prefetch_factor,
        )
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, **dkw
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, **dkw
        )

        loss_curve, val_curve = [], []

        for epoch in tqdm(range(self.epochs), desc='Training Progress', file=sys.stdout):
            self.flow_model.model.train()
            total_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)

                # 1) couple (OT with chosen cost or random)
                z0, z1 = self._pair_batch(z0_all, z1_all)

                # 2) FM targets
                z_t, t, u_true = self.flow_model.get_train_tuple(z0, z1)
                u_pred = self.flow_model.model(z_t, t)
                u_pred = self._maybe_project_pred(z_t, u_pred)

                # 3) loss
                loss = self.criterion(u_pred, u_true, torch.ones(u_pred.size(0), device=self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train = total_loss / len(train_loader)
            avg_val   = self.validate(val_loader)
            loss_curve.append(avg_train)
            val_curve.append(avg_val)
            tqdm.write(f"Epoch {epoch+1}/{self.epochs}: train={avg_train:.4e}, val={avg_val:.4e}")

        return (self.flow_model.model,), loss_curve, val_curve


class FeedforwardTrainer:
    def __init__(
        self,
        encoded_dim,
        architecture,
        data_dir,
        batch_size=250,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        val_split=0.1,
        signal_strength=0.0,
        noise_type='uniform_hypersphere',
        target_type='coordinate',
        device="cpu",
        # OT controls
        use_ot_train: bool = False,
        ot_method:    str  = "exact",   # "exact", "sinkhorn", or "none"
        ot_reg:       float | None = 0.05,
        ot_cost:      str  = "angular",    # "angular" or "euclidean"
        dataloader_num_workers: int | None = None,
        dataloader_prefetch_factor: int = 2,
    ):
        self.device          = device
        self.encoded_dim     = encoded_dim
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.val_split       = val_split
        self.signal_strength = signal_strength
        self.noise_type      = noise_type
        self.target_type     = target_type

        if dataloader_num_workers is None:
            self.dataloader_num_workers = 0 if sys.platform == "win32" else 4
        else:
            self.dataloader_num_workers = int(dataloader_num_workers)
        self.dataloader_prefetch_factor = int(dataloader_prefetch_factor)

        # --- OT policy (avoid float(None)) ---
        self.use_ot_train = bool(use_ot_train)
        self.ot_method = (ot_method or "none").lower()
        self.ot_cost   = (ot_cost or "angular").lower()

        if self.use_ot_train and self.ot_method == "sinkhorn":
            self.ot_reg = float(0.05 if ot_reg is None else ot_reg)
        else:
            # exact / none / disabled
            self.ot_reg = None

        ds = SSPDataset(
            data_dir        = data_dir,
            ssp_dim         = self.encoded_dim,
            target_type     = self.target_type,
            noise_type      = self.noise_type,
            signal_strength = self.signal_strength
        )
        self.train_dataset, self.val_dataset = ds.split_dataset(self.val_split)
        self.model = architecture.to(self.device)

        self.criterion_cos = nn.CosineEmbeddingLoss()
    
    def _pair_batch(self, z0_all, z1_all):
        if self.use_ot_train:
            if self.ot_cost == "euclidean":
                # Euclidean OT (cdist^2); do NOT renorm
                jdx = euclidean_ot_pairs(z0_all, z1_all, reg=self.ot_reg, squared=True, hard=False)
            else:
                # Angular OT on the sphere; renorm to be safe
                z0n = _renorm(z0_all)
                z1n = _renorm(z1_all)
                jdx = angular_ot_pairs(z0n, z1n, reg=self.ot_reg, squared=True, hard=False)
            return z0_all, z1_all[jdx]
        # random fallback (previous behavior)
        idx = torch.randperm(z1_all.size(0), device=z1_all.device)
        return z0_all, z1_all[idx]


    def validate(self, dataloader):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in dataloader:
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)
                z0, z1 = self._pair_batch(z0_all, z1_all)
                pred = self.model(z0)
                loss = self.criterion_cos(pred, z1, torch.ones(pred.size(0), device=self.device))
                total += loss.item()
        return total / len(dataloader)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dkw = _dataloader_kwargs(
            self.dataloader_num_workers,
            self.device,
            self.dataloader_prefetch_factor,
        )
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, **dkw
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, **dkw
        )

        loss_curve, val_curve = [], []
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)
                z0, z1 = self._pair_batch(z0_all, z1_all)
                pred = self.model(z0)
                loss = self.criterion_cos(pred, z1, torch.ones(pred.size(0), device=self.device))
                loss.backward()
                optimizer.step()
                total += loss.item()
            avg_train = total / len(train_loader)
            avg_val   = self.validate(val_loader)
            loss_curve.append(avg_train)
            val_curve.append(avg_val)
            print(f"Epoch {epoch+1}/{self.epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")

        return self.model, loss_curve, val_curve
