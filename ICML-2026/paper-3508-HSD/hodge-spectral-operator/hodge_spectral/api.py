"""
High-level API for Hodge Spectral Operator.

Usage:
    # From mesh: 0-form → 1-form (scalar source → vector velocity)
    model = HodgeOperator.from_mesh(points, faces, task="0to1")
    model.fit(X_train, Y_train, epochs=100)
    Y_pred = model.predict(X_test)

    # From point cloud: 0-form → 0-form (scalar evolution)
    model = HodgeOperator.from_pointcloud(points, task="0to0")

    # From graph: 0-form → 1-form
    model = HodgeOperator.from_graph(edge_index, n_nodes, positions, task="0to1")
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from hodge_spectral.adapters import create_adapter
from hodge_spectral.operators import HighOrderSpectralOperators
from hodge_spectral.models.unified import UnifiedHSD, train_unified_hsd
from hodge_spectral.utils import LpLoss


class _SimpleDS(Dataset):
    def __init__(self, c0, c1, c2, xr, yr):
        self.c0, self.c1, self.c2, self.xr, self.yr = c0, c1, c2, xr, yr

    def __len__(self):
        return len(self.c0)

    def __getitem__(self, i):
        return (torch.from_numpy(self.c0[i]), torch.from_numpy(self.c1[i]),
                torch.from_numpy(self.c2[i]), torch.zeros(1),
                torch.from_numpy(self.xr[i]), torch.from_numpy(self.yr[i]))


def _parse_task(task_str):
    """Parse '0to1', '0to0', '1to0', etc."""
    parts = task_str.lower().replace("->", "to").replace("→", "to").split("to")
    return int(parts[0].strip()), int(parts[1].strip())


class HodgeOperator:
    """
    One-line neural operator learning with Hodge decomposition inductive bias.

    Automatically handles:
      - Input adaptation (mesh / point cloud / graph)
      - Spectral decomposition (Hodge Laplacians, eigenbases)
      - De Rham cross-form features
      - Spectral bilinear interaction
      - Adaptive spectral-spatial dual branch
      - Neural gating + commutator
    """

    def __init__(self, points, faces, task="0to1",
                 # Spectral decomposition
                 k=64,
                 normalize_laplacian=True,
                 # Spectral branch architecture
                 hidden_dims=(256, 192),
                 dropout=0.05,
                 # Spatial branch architecture
                 res_hidden=128,
                 res_dropout=0.1,
                 # Training defaults (can also override in .fit())
                 default_lr=3e-3,
                 default_epochs=100,
                 default_patience=25,
                 # Device
                 device=None,
                 # Advanced
                 heat_kernel_laplacian=None):
        """
        Hodge Spectral Operator.

        Args:
            points: (N, D) node coordinates
            faces: (M, 3) triangle indices
            task: differential form task, e.g. "0to0", "0to1", "1to0", "1to1", "0to2"
                  0 = scalar (nodes), 1 = vector/flux (edges), 2 = density (faces)

            --- Spectral decomposition ---
            k: number of eigenmodes per form (higher = more expressive, slower)
            normalize_laplacian: apply symmetric normalization D^{-1/2}LD^{-1/2}

            --- Spectral branch (Hodge-aware MLP) ---
            hidden_dims: tuple of hidden layer sizes. More/larger = more capacity.
                         e.g. (128,) for lightweight, (512, 256) for high capacity.
            dropout: dropout rate in spectral MLP layers

            --- Spatial branch (residual correction) ---
            res_hidden: hidden size of spatial encoder/decoder.
                        Larger = better high-frequency capture.
            res_dropout: dropout in spatial branch

            --- Training defaults ---
            default_lr: learning rate (override in .fit())
            default_epochs: max epochs (override in .fit())
            default_patience: early stopping patience (override in .fit())

            --- Device ---
            device: 'cuda', 'cpu', or None (auto-detect)

            --- Advanced ---
            heat_kernel_laplacian: precomputed (N,N) heat kernel Laplacian for L0.
                                  Used by PointCloudAdapter internally.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_form, self.output_form = _parse_task(task)
        self.k = k
        self._dropout = dropout
        self._res_dropout = res_dropout
        self.points = points.astype(np.float64)
        self.faces = faces.astype(np.int64)
        self.n_nodes = len(points)

        # Build spectral operators
        self.host = HighOrderSpectralOperators(
            self.points, self.faces, k_list=(k, k, k),
            heat_kernel_laplacian=heat_kernel_laplacian
        )

        # VectorFluxMapper for 1-form tasks
        if self.output_form == 1 or self.input_form == 1:
            from hodge_spectral.models.dataset import VectorFluxMapper
            self.mapper = VectorFluxMapper(self.points, self.faces)
        else:
            self.mapper = None

        # Phi for output decoding
        Phi_map = {
            0: self.host.Phi0[:, :k],
            1: self.host.Phi0[:, :k],  # vectors decoded per-component in Phi0
            2: self.host.Phi2[:, :k],
        }
        self.Phi_out = torch.from_numpy(Phi_map[self.output_form].astype(np.float32)).to(self.device)
        Phi1 = self.host.Phi1[:, :k]

        # Output entity count
        n_out = {0: self.n_nodes, 1: self.n_nodes, 2: self.host.n_faces}[self.output_form]

        # Build model
        self.model = UnifiedHSD(
            self.host, self.output_form, k, k, k,
            n_out, k, self.Phi_out,
            hidden_dims=hidden_dims, res_hidden=res_hidden
        ).to(self.device)

        self._Phi0 = self.host.Phi0[:, :k]
        self._Phi1 = Phi1
        self._trained = False
        self.x_scale = None
        self.y_scale = None
        self._default_lr = default_lr
        self._default_epochs = default_epochs
        self._default_patience = default_patience

    @classmethod
    def from_mesh(cls, points, faces, **kwargs):
        """Create from triangulated mesh."""
        return cls(np.asarray(points), np.asarray(faces), **kwargs)

    @classmethod
    def from_pointcloud(cls, points, k_neighbors=15, heat_t=0.5, **kwargs):
        """Create from raw point cloud. Automatically triangulates."""
        adapter = create_adapter("pointcloud", k=k_neighbors, heat_t=heat_t, use_alpha=True)
        pts, fcs = adapter.adapt(points=np.asarray(points))
        heat_L = adapter.get_heat_kernel_laplacian()
        return cls(pts, fcs, heat_kernel_laplacian=heat_L, **kwargs)

    @classmethod
    def from_graph(cls, edge_index, n_nodes, positions=None, **kwargs):
        """Create from graph. Positions required or spectral embedding used."""
        adapter = create_adapter("graph", use_spectral_pos=(positions is None), embed_dim=3)
        pts, fcs = adapter.adapt(
            edge_index=np.asarray(edge_index),
            n_nodes=n_nodes,
            positions=np.asarray(positions) if positions is not None else None
        )
        return cls(pts, fcs, **kwargs)

    def _lift(self, X_norm):
        """Lift input to spectral coefficients."""
        fX, gX, _ = self.host.lift_signal(X_norm)
        c0 = (fX @ self._Phi0).astype(np.float32)
        c1 = (gX @ self._Phi1).astype(np.float32)
        c2 = np.zeros((len(X_norm), self.k), dtype=np.float32)
        return c0, c1, c2

    def fit(self, X_train, Y_train, epochs=None, lr=None, batch_size=64,
            val_ratio=0.15, patience=None, weight_decay=1e-5, verbose=True):
        """
        Train the model.

        Args:
            X_train: input data. Shape depends on input_form.
            Y_train: target data. Shape depends on output_form.
            epochs: max training epochs (default: from constructor)
            lr: learning rate (default: from constructor)
            batch_size: batch size
            val_ratio: fraction held out for validation
            patience: early stopping patience (default: from constructor)
            weight_decay: AdamW weight decay
            verbose: print training progress
        """
        epochs = epochs or self._default_epochs
        lr = lr or self._default_lr
        patience = patience or self._default_patience
        X_train = np.asarray(X_train, dtype=np.float64)
        Y_train = np.asarray(Y_train, dtype=np.float64)

        # Split
        X_tr, X_va, Y_tr, Y_va = train_test_split(X_train, Y_train, test_size=val_ratio, random_state=42)

        # Scale
        self.x_scale = np.max(np.abs(X_tr)) + 1e-9
        self.y_scale = np.max(np.abs(Y_tr)) + 1e-9
        X_tr_n = (X_tr / self.x_scale).astype(np.float32)
        X_va_n = (X_va / self.x_scale).astype(np.float32)
        Y_tr_n = (Y_tr / self.y_scale).astype(np.float32)
        Y_va_n = (Y_va / self.y_scale).astype(np.float32)

        # Lift
        c0_tr, c1_tr, c2_tr = self._lift(X_tr_n)
        c0_va, c1_va, c2_va = self._lift(X_va_n)

        ds_tr = _SimpleDS(c0_tr, c1_tr, c2_tr, X_tr_n, Y_tr_n)
        ds_va = _SimpleDS(c0_va, c1_va, c2_va, X_va_n, Y_va_n)

        logger = type('L', (), {'log': lambda s, m: print(m) if verbose else None})()

        self.model = train_unified_hsd(
            self.model,
            DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
            DataLoader(ds_va, batch_size=batch_size),
            epochs, self.device,
            logger=logger if verbose else None,
            lr=lr, patience=patience
        )
        self._trained = True
        return self

    def predict(self, X_test):
        """
        Predict on new data.

        Args:
            X_test: input data, same format as X_train

        Returns:
            predictions in original scale
        """
        if not self._trained:
            raise RuntimeError("Call .fit() first")

        X_test = np.asarray(X_test, dtype=np.float64)
        X_n = (X_test / self.x_scale).astype(np.float32)
        c0, c1, c2 = self._lift(X_n)

        ds = _SimpleDS(c0, c1, c2, X_n, np.zeros_like(X_n if X_n.ndim == 2 else X_n.reshape(len(X_n), -1)))
        loader = DataLoader(ds, batch_size=64)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for c0_b, c1_b, c2_b, _, xr_b, _ in loader:
                p, _, _ = self.model(
                    c0_b.to(self.device), c1_b.to(self.device),
                    c2_b.to(self.device), xr_b.to(self.device)
                )
                preds.append(p.cpu().numpy())

        return np.concatenate(preds, axis=0) * self.y_scale

    def evaluate(self, X_test, Y_test):
        """Evaluate and return metrics dict."""
        Y_pred = self.predict(X_test)
        Y_test = np.asarray(Y_test, dtype=np.float64)

        # Relative L2
        pred_flat = Y_pred.reshape(len(Y_pred), -1)
        gt_flat = Y_test.reshape(len(Y_test), -1)
        rel_l2 = np.mean(
            np.linalg.norm(pred_flat - gt_flat, axis=1) /
            (np.linalg.norm(gt_flat, axis=1) + 1e-9)
        )

        # Riemannian inner product fidelity
        rip = np.mean(
            np.sum(pred_flat * gt_flat, axis=1) /
            (np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(gt_flat, axis=1) + 1e-9)
        )

        return {
            'relative_l2': float(rel_l2),
            'riemannian_ip_fidelity': float(rip),
            'mse': float(np.mean((Y_pred - Y_test) ** 2)),
        }

    def save(self, path):
        """Save model state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'x_scale': self.x_scale,
            'y_scale': self.y_scale,
            'config': {
                'k': self.k, 'input_form': self.input_form,
                'output_form': self.output_form,
                'n_nodes': self.n_nodes,
            }
        }, path)

    def load(self, path):
        """Load model state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.x_scale = ckpt['x_scale']
        self.y_scale = ckpt['y_scale']
        self._trained = True
        return self
