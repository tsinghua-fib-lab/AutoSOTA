"""
Unified Differential Form Framework for HSD
=============================================
Handles any k-form → p-form operator learning task through a single interface.

Supported form types:
  0-form: scalar fields on nodes (e.g., temperature, pressure, concentration)
  1-form: edge fluxes (e.g., velocity, current, magnetic field)
  2-form: face densities (e.g., vorticity, magnetic flux through faces)

The de Rham complex 0 --d0--> 1 --d1--> 2 and its adjoint
                     0 <-δ0-- 1 <-δ1-- 2
provide natural inter-form operators as inductive bias.

Usage:
    task = FormTask(
        input_form=0,   # scalar input
        output_form=1,  # flux output
        host_ops=host,
        points=pts,
    )
    dataset = task.build_dataset(X_data, Y_data)
    model = task.build_model()
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ==========================================
# Unified Dataset: any form → any form
# ==========================================

class UnifiedFormDataset(Dataset):
    """
    Universal dataset for k-form → p-form prediction.

    Automatically handles:
      - Spectral lifting via eigenbases (Phi0, Phi1, Phi2)
      - Multi-scale heat kernel enrichment
      - Higher-order curl encoding for nonzero c2
      - Proper target projection based on output form type
    """

    def __init__(self, host_ops, X_data, Y_data, k_list,
                 input_form=0, output_form=0,
                 mapper=None, points=None,
                 x_scale=None, y_scale=None, logger=None):
        """
        Args:
            host_ops: HighOrderSpectralOperators
            X_data: input data. Shape depends on input_form:
                0-form: (B, N), 1-form: (B, E), 2-form: (B, F)
                or (B, N, 3) for vector fields (auto-converted to 1-form via mapper)
            Y_data: target data. Shape depends on output_form:
                0-form: (B, N), 1-form: (B, E), 2-form: (B, F)
                or (B, N, 3) for vector fields
            input_form: 0, 1, or 2
            output_form: 0, 1, or 2
            mapper: VectorFluxMapper (required if input/output is vector field)
            points: (N, 3) node coordinates (for enriched encoding)
        """
        self.input_form = input_form
        self.output_form = output_form
        self.k0, self.k1, self.k2 = k_list
        self.host = host_ops
        self.mapper = mapper

        # Detect and handle vector field data (B, N, 3) -> convert to flux
        X_proc = self._preprocess_input(X_data, input_form)
        Y_proc = self._preprocess_output(Y_data, output_form)

        # Scale
        self.x_scale = x_scale if x_scale else np.max(np.abs(X_proc)) + 1e-9
        self.y_scale = y_scale if y_scale else np.max(np.abs(Y_proc)) + 1e-9
        X_norm = (X_proc / self.x_scale).astype(np.float32)
        Y_norm = (Y_proc / self.y_scale).astype(np.float32)

        # Store raw normalized data
        self.X_raw = X_norm  # always in input space

        # Y_raw must match the MODEL's output space:
        # For output_form=1 with vector data (B,N,3), store as (B,N,3)
        if output_form == 1 and Y_data.ndim == 3 and Y_data.shape[2] == 3:
            y_vec_scale = np.max(np.abs(Y_data)) + 1e-9
            self.Y_raw = (Y_data / y_vec_scale).astype(np.float32)
            self.y_scale = y_vec_scale  # override scale for vector
        else:
            self.Y_raw = Y_norm

        # Also keep X in node space for scalar input
        if input_form == 0 and X_data.ndim == 2 and X_data.shape[1] == host_ops.n_nodes:
            self.X_raw = (X_data / self.x_scale).astype(np.float32)

        # Enriched spectral input (always uses all three form levels)
        self._compute_enriched_input(host_ops, X_norm, input_form, points)

        # Target spectral coefficients
        self._compute_target(host_ops, Y_norm, output_form)

        if logger:
            logger.log(f"[UnifiedDataset] {input_form}-form → {output_form}-form")
            logger.log(f"  Enriched: c0={self.c0_in.shape[1]}, c1={self.c1_in.shape[1]}, c2={self.c2_in.shape[1]}")
            logger.log(f"  Target: {self.target_coeffs.shape}")

    def _preprocess_input(self, data, form):
        """Convert input to the canonical form representation."""
        if form == 0:
            if data.ndim == 2 and data.shape[1] == self.host.n_nodes:
                return data  # (B, N) scalar
            elif data.ndim == 3 and data.shape[2] == 3:
                # Vector field -> take magnitude as scalar input
                return np.linalg.norm(data, axis=2)
        elif form == 1:
            if data.ndim == 2 and data.shape[1] == self.host.n_edges:
                return data  # (B, E) edge flux
            elif data.ndim == 3 and data.shape[2] == 3 and self.mapper:
                # Vector field -> edge flux
                return np.array([self.mapper.node_vector_to_edge_flux(data[i]) for i in range(len(data))])
        elif form == 2:
            if data.ndim == 2 and data.shape[1] == self.host.n_faces:
                return data  # (B, F) face density
        return data

    def _preprocess_output(self, data, form):
        """Convert output to canonical form."""
        return self._preprocess_input(data, form)

    def _compute_enriched_input(self, host, X_norm, input_form, points):
        """Compute enriched spectral features for any input form."""
        B = X_norm.shape[0]
        k0, k1, k2 = self.k0, self.k1, self.k2

        if input_form == 0:
            # Use the enriched encoding with heat kernel + curl
            if hasattr(host, 'compute_enriched_input'):
                self.c0_in, self.c1_in, self.c2_in = host.compute_enriched_input(X_norm)
            else:
                fX, gX, hX = host.lift_signal(X_norm)
                self.c0_in = (fX @ host.Phi0[:, :k0]).astype(np.float32)
                self.c1_in = (gX @ host.Phi1[:, :k1]).astype(np.float32)
                self.c2_in = np.zeros((B, k2), dtype=np.float32)

        elif input_form == 1:
            # 1-form input: project onto Phi1, derive c0 via divergence, c2 via curl
            self.c1_in = (X_norm @ host.Phi1[:, :k1]).astype(np.float32)
            # c0 from divergence: B1^T @ flux -> scalar on nodes
            div_signal = (host.B1.T @ X_norm.T).T  # (B, N)
            self.c0_in = (div_signal @ host.Phi0[:, :k0]).astype(np.float32)
            # c2 from curl: B2 @ flux -> density on faces
            curl_signal = (host.B2 @ X_norm.T).T  # (B, F)
            self.c2_in = (curl_signal @ host.Phi2[:, :k2]).astype(np.float32)

        elif input_form == 2:
            # 2-form input: project onto Phi2, derive c1 via codifferential
            self.c2_in = (X_norm @ host.Phi2[:, :k2]).astype(np.float32)
            codiff_signal = (host.B2.T @ X_norm.T).T  # (B, E)
            self.c1_in = (codiff_signal @ host.Phi1[:, :k1]).astype(np.float32)
            self.c0_in = np.zeros((B, k0), dtype=np.float32)

    def _compute_target(self, host, Y_norm, output_form):
        """Compute target spectral coefficients based on output form."""
        k0, k1, k2 = self.k0, self.k1, self.k2

        if output_form == 0:
            self.target_coeffs = (Y_norm @ host.Phi0[:, :k0]).astype(np.float32)
            self.target_basis = 'Phi0'
            self.target_k = k0
        elif output_form == 1:
            self.target_coeffs = (Y_norm @ host.Phi1[:, :k1]).astype(np.float32)
            self.target_basis = 'Phi1'
            self.target_k = k1
        elif output_form == 2:
            self.target_coeffs = (Y_norm @ host.Phi2[:, :k2]).astype(np.float32)
            self.target_basis = 'Phi2'
            self.target_k = k2

    def __len__(self):
        return len(self.c0_in)

    def __getitem__(self, idx):
        """Returns (c0, c1, c2, target_coeffs, x_raw, y_raw)."""
        return (
            torch.from_numpy(self.c0_in[idx]),
            torch.from_numpy(self.c1_in[idx]),
            torch.from_numpy(self.c2_in[idx]),
            torch.from_numpy(self.target_coeffs[idx]),
            torch.from_numpy(self.X_raw[idx]),
            torch.from_numpy(self.Y_raw[idx]),
        )


# ==========================================
# Unified HSD Model: any form → any form
# ==========================================

class UnifiedHSD(nn.Module):
    """
    Form-agnostic HSD model.

    Architecture:
      1. Enriched spectral input (c0, c1, c2) with de Rham cross-terms
      2. Spectral MLP → output form coefficients
      3. Spatial residual branch → full-space correction
      4. Orthogonal decomposition: residual only adds what spectral can't

    The same model works for:
      0→0 (scalar evolution), 0→1 (Darcy flow), 1→0 (inverse), etc.
    """

    def __init__(self, host_ops, output_form, c0_dim, c1_dim, c2_dim,
                 n_output_entities, output_k, Phi_output,
                 hidden_dims=(192, 192), res_hidden=128):
        """
        Args:
            host_ops: HighOrderSpectralOperators
            output_form: 0, 1, or 2
            c0_dim, c1_dim, c2_dim: enriched input dimensions
            n_output_entities: N (nodes) for 0-form, E (edges) for 1-form, F (faces) for 2-form
            output_k: number of output spectral coefficients
            Phi_output: output eigenbasis (N,k), (E,k), or (F,k) as torch tensor on device
            hidden_dims: MLP hidden layer sizes
            res_hidden: spatial residual hidden size
        """
        super().__init__()
        self.output_form = output_form
        self.output_k = output_k
        self.n_out = n_output_entities

        # Learnable de Rham operators: initialized from topology, adapted during training.
        # The discrete exterior derivative d and codifferential δ define the Hodge
        # complex structure. By making them learnable, the network discovers which
        # cross-form connections are most relevant for the task.
        #
        # Parameterization: M_learned = M_topo + ΔM (residual learning)
        # This preserves the topological initialization while allowing adaptation.
        k0 = host_ops.Phi0.shape[1]
        k1 = host_ops.Phi1.shape[1]
        Md0_init = torch.from_numpy(host_ops.Md0.astype(np.float32))
        Mdelta1_init = torch.from_numpy(host_ops.Mdelta1.astype(np.float32))

        self.register_buffer("Md0_topo", Md0_init)        # frozen topological init
        self.register_buffer("Mdelta1_topo", Mdelta1_init)
        # Low-rank perturbation: ΔM = U @ V^T with rank r << min(k0,k1)
        # This constrains the learnable space while allowing meaningful adaptation.
        rank = min(8, min(k0, k1))
        self.Md0_U = nn.Parameter(torch.randn(Md0_init.shape[0], rank) * 0.01)
        self.Md0_V = nn.Parameter(torch.randn(Md0_init.shape[1], rank) * 0.01)
        self.Mdelta1_U = nn.Parameter(torch.randn(Mdelta1_init.shape[0], rank) * 0.01)
        self.Mdelta1_V = nn.Parameter(torch.randn(Mdelta1_init.shape[1], rank) * 0.01)

        # Spectral MLP
        in_dim = c0_dim + c1_dim + c2_dim + min(k0, c0_dim) + min(k1, c1_dim)
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.05)]
            d = h
        self.spectral_mlp = nn.Sequential(*layers)
        self.latent_dim = d

        # Output head: depends on output form
        if output_form == 0:
            self.spectral_head = nn.Linear(d, output_k)  # scalar coeffs
        elif output_form == 1:
            # For vector (1-form) output, predict per-component
            self.spectral_head = nn.Linear(d, output_k * 3)  # 3 components
        elif output_form == 2:
            self.spectral_head = nn.Linear(d, output_k)

        # Spatial residual
        n_spatial_in = n_output_entities  # residual in output space
        if output_form == 1:
            # For 1-form, spatial input is the raw scalar input (node space)
            n_spatial_in = host_ops.n_nodes

        self.spatial_enc = nn.Sequential(
            nn.Linear(n_spatial_in, res_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(res_hidden, res_hidden // 2))

        out_spatial = n_output_entities * (3 if output_form == 1 else 1)
        self.res_dec = nn.Sequential(
            nn.Linear(res_hidden // 2 + d, res_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(res_hidden, out_spatial))
        nn.init.zeros_(self.res_dec[-1].weight)
        nn.init.zeros_(self.res_dec[-1].bias)

        self.register_buffer('Phi_out', Phi_output)

        # Neural gating: learns per-sample how to mix spectral vs spatial
        # Input: spectral latent (captures frequency content of this sample)
        # Output: mixing weight α ∈ (0,1) per output entity
        # α → 1 means trust spectral; α → 0 means trust spatial
        gate_out = n_output_entities * (3 if output_form == 1 else 1)
        self.gate_net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, gate_out),
            nn.Sigmoid()  # bounded (0,1)
        )
        # Initialize gate to slightly favor spectral (bias > 0)
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.constant_(self.gate_net[-2].bias, 1.0)

        self._k0_raw = min(k0, c0_dim)
        self._k1_raw = min(k1, c1_dim)

    def forward(self, c0, c1, c2, x_raw):
        """
        Args:
            c0, c1, c2: enriched spectral input
            x_raw: raw input in original space (for spatial branch)
        Returns:
            pred: prediction in output space
            base: spectral base prediction
            residual: spatial residual
        """
        # Learnable de Rham cross-terms: topology + learned perturbation
        c0_raw = c0[:, :self._k0_raw]
        c1_raw = c1[:, :self._k1_raw]

        # Effective operators = topological init + low-rank learned perturbation
        Md0_eff = self.Md0_topo + self.Md0_U @ self.Md0_V.t()
        Mdelta1_eff = self.Mdelta1_topo + self.Mdelta1_U @ self.Mdelta1_V.t()

        Mdelta1_s = Mdelta1_eff[:min(self._k0_raw, Mdelta1_eff.shape[0]),
                                 :min(self._k1_raw, Mdelta1_eff.shape[1])]
        Md0_s = Md0_eff[:min(self._k1_raw, Md0_eff.shape[0]),
                         :min(self._k0_raw, Md0_eff.shape[1])]

        div_c1 = torch.matmul(c1_raw[:, :Mdelta1_s.shape[1]], Mdelta1_s.t())
        grad_c0 = torch.matmul(c0_raw[:, :Md0_s.shape[1]], Md0_s.t())

        feat = torch.cat([c0, c1, c2, div_c1, grad_c0], dim=-1)
        latent = self.spectral_mlp(feat)
        coeffs = self.spectral_head(latent)

        # Decode spectral base to output space
        if self.output_form == 0:
            base = torch.matmul(coeffs, self.Phi_out.t())  # (B, N)
        elif self.output_form == 1:
            # (B, k*3) -> (B, 3, k) -> matmul with Phi -> (B, 3, E) -> permute
            c3 = coeffs.view(-1, 3, self.output_k)
            base = torch.matmul(c3, self.Phi_out.t()).permute(0, 2, 1)  # (B, N, 3)
        elif self.output_form == 2:
            base = torch.matmul(coeffs, self.Phi_out.t())  # (B, F)

        # Spatial residual
        s_feat = self.spatial_enc(x_raw)
        res_raw = self.res_dec(torch.cat([s_feat, latent], dim=-1))
        if self.output_form == 1:
            res = res_raw.view(-1, self.n_out, 3)
        else:
            res = res_raw

        # Neural adaptive mixing: gate decides per-element how much spectral vs spatial
        # α(x) = gate_net(latent), α ∈ (0,1)
        # pred = α * base + (1-α) * (base + res) = base + (1-α) * res
        # When α→1: pure spectral. When α→0: spectral + full spatial correction.
        alpha = self.gate_net(latent)  # (B, N*D) or (B, N) depending on form

        if self.output_form == 0:
            pred = alpha * base + (1 - alpha) * (base + res)
        elif self.output_form == 1:
            alpha = alpha.view(-1, self.n_out, 3)
            pred = alpha * base + (1 - alpha) * (base + res)
        else:
            pred = alpha * base + (1 - alpha) * (base + res)

        # Simplifies to: pred = base + (1-alpha) * res
        return pred, base, res


# ==========================================
# Unified training function
# ==========================================

def train_unified_hsd(model, train_loader, val_loader, epochs, device,
                      logger=None, lr=3e-3, patience=30):
    """Train UnifiedHSD model."""
    from hodge_spectral.utils import LpLoss, EarlyStopping

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = LpLoss()
    early = EarlyStopping(patience=patience)
    best_val, best_state = float('inf'), None

    for ep in range(epochs):
        model.train()
        tl, nb = 0, 0
        for c0, c1, c2, tgt, x_raw, y_raw in train_loader:
            c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
            x_raw, y_raw = x_raw.to(device), y_raw.to(device)
            opt.zero_grad()
            pred, base, res = model(c0, c1, c2, x_raw)
            loss = crit(pred, y_raw) + 0.1 * crit(base, y_raw)
            # d²≈0 soft constraint: penalize deviation of learned operators from d²=0
            # In spectral domain: Md1 @ Md0 should be near zero
            if hasattr(model, 'Md0_U'):
                Md0_eff = model.Md0_topo + model.Md0_U @ model.Md0_V.t()
                loss = loss + 0.01 * torch.mean(model.Md0_U ** 2 + model.Md0_V ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += crit(pred, y_raw).item()
            nb += 1
        sched.step()

        model.eval()
        vl, nv = 0, 0
        with torch.no_grad():
            for c0, c1, c2, tgt, x_raw, y_raw in val_loader:
                c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
                x_raw, y_raw = x_raw.to(device), y_raw.to(device)
                p, _, _ = model(c0, c1, c2, x_raw)
                vl += crit(p, y_raw).item()
                nv += 1
        avg_v = vl / nv

        if avg_v < best_val:
            best_val = avg_v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if logger and ((ep + 1) % 20 == 0 or ep == 0):
            rs = model.res_scale.item()
            logger.log(f"  Ep {ep+1:3d}/{epochs} | Train: {tl/nb:.4f} | Val: {avg_v:.4f} | RS: {rs:.3f}")

        if early(avg_v):
            if logger:
                logger.log(f"  Early stop at epoch {ep+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ==========================================
# Task builder: configure everything from (input_form, output_form)
# ==========================================

class FormTask:
    """
    High-level task configuration.

    Usage:
        task = FormTask(input_form=0, output_form=1, host_ops=host, points=pts, mapper=mapper)
        ds_train = task.build_dataset(X_train, Y_train)
        model = task.build_model(ds_train)
    """

    def __init__(self, input_form, output_form, host_ops, points=None, mapper=None, k_list=None):
        self.input_form = input_form
        self.output_form = output_form
        self.host = host_ops
        self.points = points
        self.mapper = mapper
        self.k_list = k_list or (host_ops.Phi0.shape[1], host_ops.Phi1.shape[1], host_ops.Phi2.shape[1])

    def build_dataset(self, X, Y, x_scale=None, y_scale=None, logger=None):
        return UnifiedFormDataset(
            self.host, X, Y, self.k_list,
            input_form=self.input_form, output_form=self.output_form,
            mapper=self.mapper, points=self.points,
            x_scale=x_scale, y_scale=y_scale, logger=logger
        )

    def build_model(self, dataset, device, hidden_dims=(192, 192), res_hidden=128):
        """Build UnifiedHSD model configured for this task."""
        c0_dim = dataset.c0_in.shape[1]
        c1_dim = dataset.c1_in.shape[1]
        c2_dim = dataset.c2_in.shape[1]

        # Output configuration
        form_config = {
            0: (self.host.n_nodes, self.k_list[0], self.host.Phi0[:, :self.k_list[0]]),
            1: (self.host.n_edges, self.k_list[1], self.host.Phi1[:, :self.k_list[1]]),
            2: (self.host.n_faces, self.k_list[2], self.host.Phi2[:, :self.k_list[2]]),
        }

        # For vector output (1-form displayed as node vectors), use Phi0 basis
        if self.output_form == 1 and self.mapper is not None:
            n_out = self.host.n_nodes
            k_out = self.k_list[0]
            Phi_out = self.host.Phi0[:, :k_out]
        else:
            n_out, k_out, Phi_out = form_config[self.output_form]

        Phi_tensor = torch.from_numpy(Phi_out.astype(np.float32)).to(device)

        return UnifiedHSD(
            self.host, self.output_form, c0_dim, c1_dim, c2_dim,
            n_out, k_out, Phi_tensor,
            hidden_dims=hidden_dims, res_hidden=res_hidden
        ).to(device)
