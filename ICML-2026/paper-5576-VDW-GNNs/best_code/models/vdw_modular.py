"""
Modular SO(d)-equivariant VDW variant with simplified, clean structure.

Class: VDWModular

Key ideas:
- Separate scalar and vector tracks (both ablatable)
- 0th-, 1st-, 2nd-order scattering as in VDW
- Optional BatchNorm for scalars (not for vectors)
- Within-track wavelet mixing via configurable MLPs
  - Scalar: standard MLP across wavelet axis
  - Vector: bias-free linears and scalar gating nonlinearity (equivariant)
- Track mixing: concatenate scalar flattened features with vector invariants
- Node/graph-level heads for scalar and vector targets

This module reuses utilities (e.g., batch_scatter) and BaseModule.

--- LaTeX summary ---
\section*{{\modelname} architecture}
Given a graph \(G=(V,E)\) with \(N=|V|\) nodes, optional scalar features \(X_s\in\mathbb{R}^{N\times C}\),
optional vector features \(X_v\in\mathbb{R}^{N\times d}\), and diffusion operators \(P\in \mathbb{R}^{N \times N}\) (scalar) and \(Q \in \mathbb{R}^{Nd \times Nd}\) (vector).

\subsection*{Scattering per track}
\begin{itemize}
\item Scalars: \(W_s^{(0)}=X_s\), \(W_s^{(1)}=\mathcal{W}(X_s;P)\), and second-order interactions
\(W_s^{(2)}=\{\, W_s^{(1)}[\cdots,i] \odot W_s^{(1)}[\cdots,j] \,\}_{i<j}\).
Concatenate along the wavelet axis: \(\tilde W_s=[W_s^{(0)},W_s^{(1)},W_s^{(2)}]\in\mathbb{R}^{N\times C\times W}\). Since scattering coefficients across orders can be of very different magnitudes, optionally apply batch normalization to the scalar scattering coefficients (independently to each channel-wavelet combination).
\item Vectors: similarly apply \(\mathcal{W}(\cdot;Q)\) to \(X_v\) reshaped to \((N \cdot d, 1)\), and reshape result to
\(\tilde W_v\in\mathbb{R}^{N\times 1\times d\times W}\).
\end{itemize}

\subsection*{Within-track wavelet mixing (along the wavelet axis)}
\begin{itemize}
\item Scalars: \(\hat W_s = \mathrm{MLP}_s(\tilde W_s)\), yielding \(\hat W_s\in\mathbb{R}^{N\times C\times W'}\).
\item Vectors (SO(d)-equivariant): bias-free linear layers with scalar gates \(\sigma(\alpha_\ell)\):
\[ y^{(\ell+1)} = \sigma(\alpha_\ell)\, y^{(\ell)} A_\ell, \quad A_\ell \in \mathbb{R}^{W_\ell\times W_{\ell+1}}, \]
applied uniformly across coordinates; output \(\hat W_v\in\mathbb{R}^{N\times 1\times d\times W'}\).
\end{itemize}

\subsection*{Cross-track invariant features}
Form \(t\) by concatenating: (i) flattened scalars \(s=\mathrm{vec}(\hat W_s)\in\mathbb{R}^{N\times (C W')}\)
when present; (ii) vector invariants per wavelet \(n_w=\lVert \hat v_w\rVert_2\) and neighbor cosine statistics
\(\mathrm{cos}_w(u,v)=\tfrac{\langle\hat v_w(u),\hat v_w(v)\rangle}{\lVert\hat v_w(u)\rVert_2\,\lVert\hat v_w(v)\rVert_2}\),
pooled per node via mean/max.

\subsection*{Node-level heads}
\begin{itemize}
\item Scalar targets (or if the vector track is ablated): \(y_s = h_s(t)\in\mathbb{R}^{N\times d_{\text{tar}}}\).
\item Vector targets: gate a weighted sum of vector wavelets. Let \(v_w\in\mathbb{R}^{N\times d}\) be the
\(w\)-th vector wavelet (averaged across the singleton channel). Gates are
\(g=\mathrm{softmax}(h_v(t_{\text{gate}}))\in\Delta^{W'}\) or learned static logits; the prediction is
\[ y_v = \sum_{w=1}^{W'} g_w\, v_w \in \mathbb{R}^{N\times d}. \]
The gate input \(t_{\text{gate}}\) includes vector invariants and (optionally) scalar features.
\end{itemize}

\subsection*{Graph vs. node tasks and final routing}
If the task is graph-level, aggregate node predictions with a permutation-invariant reduce
\(\oplus\in\{\mathrm{sum},\,\mathrm{mean},\,\mathrm{max}\}\) per graph in the batch:
\[ Y^{\text{graph}}_s[b] = \bigoplus_{n\in\mathcal{B}_b} y_s[n], \qquad
   Y^{\text{graph}}_v[b] = \bigoplus_{n\in\mathcal{B}_b} y_v[n]. \]
The module returns node-level outputs for node tasks and aggregated graph-level outputs for graph tasks;
for vector tasks the vector head is used, otherwise the scalar head.

\subsection*{Options} Ablatable: each scalar/vector track, second-order scattering terms can be ablated; the scalar scattering batch normalization layer.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from models.base_module import BaseModule
from models import nn_utilities as nnu
from torch_scatter import scatter
from geo_scat import VectorBatchNorm, batch_scatter, subset_second_order_wavelets
from pyg_utilities import infer_device_from_batch


class ScalarScatPathMixer(nn.Module):
    """
    Applies an MLP along the last dimension (wavelet axis).

    Input shape: (..., W)
    Output shape: (..., W_out)

    This layer operates independently at each node and channel, viewing the
    wavelet-filtered coefficients as a length-\(W\) vector and learning a
    mapping on that axis only. With no hidden layers, it reduces to a single
    Linear that forms new wavelet coefficients as learned linear combinations
    of the original ones (e.g., \(w'_k = \sum_j a_{kj} w_j\)). With hidden
    layers and nonlinearities, it performs a nonlinear mixing that still acts
    exclusively across the wavelet axis (no mixing across nodes or channels).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        nonlin: type[nn.Module] = nn.SiLU,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims: List[int] = [in_dim] + list(hidden_dims or []) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            layers.append(nonlin())
            if dropout_p and dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_dim = x.shape[-1]
        x_flat = x.reshape(-1, last_dim)
        y = self.net(x_flat)
        return y.reshape(*x.shape[:-1], -1)


class VectorScatPathMixer(nn.Module):
    """
    Consolidated linear/gated mixer.

    If gating is disabled, this is a single bias-free Linear along the wavelet
    axis. If gating is enabled, it applies a SimpleAffineGate on norms of the
    mixed vector wavelets. Preserves SO(d) equivariance in both cases.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_gate: bool = False,
        gate_sigma: str = 'softplus',
        use_norm_only: bool = False,
        hidden_dims: Optional[Sequence[int]] = None,
        gate_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.use_gate = bool(use_gate)
        # gate_mode supports: None (legacy), 'norm_only', 'simple_affine', 'param_norm', 'param_only'
        self.gate = (
            SimpleAffineGate(
                num_channels=out_dim,
                sigma=gate_sigma,
                use_norm_only=use_norm_only,
                gate_mode=gate_mode,
            )
            if self.use_gate else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Accepts either (..., W) or (N, d, 1, W)
        if x.dim() == 4:
            N, d, one, W = x.shape
            y = x.reshape(-1, W)  # (N*d, W)
            y = self.linear(y).reshape(N, d, one, -1)  # (N, d, 1, W)
            if self.use_gate and self.gate is not None:
                return self.gate(y)
            return y
        else:
            raise ValueError(f"VectorScatPathMixer expects input with 4 dims, got {x.dim()}")


class SimpleAffineGate(nn.Module):
    """
    Input-dependent scalar gate per wavelet channel based on vector norms.

    g_j = sigma( softplus(a_j) * ||v_j|| + b_j ), with learnable a_j (init 1.0) and b_j (init 0.0).
    Preserves SO(d) equivariance by scaling uniformly across coordinates for each wavelet channel.
    We take softplus of a_j to ensure positivity and stabilize gradients.
    """
    def __init__(
        self,
        num_channels: int,
        sigma: str = 'sigmoid',
        *,
        use_norm_only: bool = False,
        gate_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.sigma_name = str(sigma).lower()
        self.use_norm_only = bool(use_norm_only)
        # gate_mode supersedes use_norm_only if provided
        self.gate_mode = (str(gate_mode).lower() if gate_mode is not None else None)
        if self.gate_mode in ('param_norm', 'param_only'):
            # Learn a per-channel multiplicative weight on the norm
            self.w = nn.Parameter(torch.ones(self.num_channels, dtype=torch.float32))
            self.a = None  # type: ignore[assignment]
            self.b = None  # type: ignore[assignment]
        elif not self.use_norm_only:
            self.a = nn.Parameter(torch.ones(self.num_channels, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(self.num_channels, dtype=torch.float32))
            self.w = None  # type: ignore[assignment]
        else:
            self.a = None  # type: ignore[assignment]
            self.b = None  # type: ignore[assignment]
            self.w = None  # type: ignore[assignment]

    def _sigma(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma_name == 'sigmoid':
            return torch.sigmoid(x)
        if self.sigma_name == 'softplus':
            return F.softplus(x)
        if self.sigma_name == 'tanh':
            return torch.tanh(x)
        return x

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # Expect (N, d, 1, W). Coerce common layouts when necessary.
        if v.dim() == 4:
            if v.shape[2] == 1 and v.shape[1] <= 16:
                v_nd1w = v
            else:
                # assume (N, W, 1, d)
                v_nd1w = v.permute(0, 3, 2, 1).contiguous()
        elif v.dim() == 3:
            # (N, W, d) -> (N, d, 1, W)
            v_nd1w = v.permute(0, 2, 1).unsqueeze(2).contiguous()
        else:
            raise ValueError("SimpleAffineGate expects input with 3 or 4 dims")

        N, d, one, W = v_nd1w.shape
        if W != self.num_channels:
            # Lazily resize parameters to match channel count
            with torch.no_grad():
                if self.gate_mode in ('param_norm', 'param_only'):
                    if self.w is not None:
                        new_w = torch.ones(W, dtype=self.w.dtype, device=self.w.device)
                        k = min(W, self.num_channels)
                        new_w[:k] = self.w[:k]
                        self.w = nn.Parameter(new_w)
                elif not self.use_norm_only:
                    if (self.a is not None) and (self.b is not None):
                        new_a = torch.ones(W, dtype=self.a.dtype, device=self.a.device)
                        new_b = torch.zeros(W, dtype=self.b.dtype, device=self.b.device)
                        k = min(W, self.num_channels)
                        new_a[:k] = self.a[:k]
                        new_b[:k] = self.b[:k]
                        self.a = nn.Parameter(new_a)
                        self.b = nn.Parameter(new_b)
            self.num_channels = W

        # Norm across coordinates -> (N, 1, W)
        norms = torch.linalg.norm(v_nd1w, dim=1, keepdim=False)
        if (self.gate_mode is not None) and (self.gate_mode in ('param_norm', 'param_only')):
            # param_norm: sigma( softplus(w_j) * ||v_j|| )
            # param_only: sigma( w_j )
            if self.gate_mode == 'param_norm':
                w_pos = F.softplus(self.w).view(1, 1, W) if self.w is not None else 1.0
                gates = self._sigma(w_pos * norms)  # (N, 1, W)
            else:  # param_only
                w_raw = self.w.view(1, 1, W) if self.w is not None else 0.0
                gates = self._sigma(w_raw)  # (N, 1, W)
        elif self.use_norm_only:
            gates = self._sigma(norms)
        else:
            # simple_affine: sigma( softplus(a_j) * ||v_j|| + b_j )
            a_pos = F.softplus(self.a).view(1, 1, W)  # type: ignore[arg-type]
            b = self.b.view(1, 1, W)                  # type: ignore[arg-type]
            gates = self._sigma(a_pos * norms + b)  # (N, 1, W)
        gates = gates.unsqueeze(1)              # (N, 1, 1, W)
        return v_nd1w * gates


class MultiLayerVectorScatPathMixer(nn.Module):
    """
    Stack of VectorScatPathMixer layers operating along the wavelet axis.

    Each layer applies a bias-free Linear on the last dim and an optional gate.
    """
    def __init__(
        self,
        in_dim: int,
        layer_out_dims: Sequence[int],
        *,
        use_gate: bool,
        gate_sigma: str,
        gate_mode: Optional[str],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for out_dim in layer_out_dims:
            layers.append(
                VectorScatPathMixer(
                    in_dim=prev,
                    out_dim=out_dim,
                    use_gate=use_gate,
                    gate_sigma=gate_sigma,
                    use_norm_only=(gate_mode == 'norm_only'),
                    gate_mode=gate_mode,
                )
            )
            prev = out_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

 

class VDWModular(BaseModule):

    MIXING_DEFAULTS = {
        'scalar_hidden_dims': [32, 32],
        'scalar_dropout_p': 0.0,
        'scalar_nonlin': nn.SiLU,
        'W_out_scalar': None,  # if None, keep input W
        'W_out_vector': None,
        'use_scalar_batch_norm': True,
        'use_vector_wavelet_batch_norm': False,
        'vector_bn_momentum': 0.1,
        'vector_bn_eps': 1e-6,
        'vector_bn_track_running_stats': True,
        # Optional override for vector mixer output dim when W_out_vector is None
        'vector_wavelet_mixer_linear_dim': None,
    }

    HEAD_DEFAULTS = {
        'node_scalar_head_hidden': [64, 64],
        'node_scalar_head_nonlin': nn.SiLU,
        'node_scalar_head_dropout': 0.0,
        'vector_gate_hidden': [128, 128],
        'vector_gate_mlp_nonlin': nn.SiLU,
        # Gating mode: if True -> Sigmoid + L1 normalize; if False -> Softmax with learned temperature
        'vector_gate_use_sigmoid': True,
        'vector_gate_init_temperature': 1.0,
        # If True, normalize final vector gates; if False, use raw gates
        'normalize_final_vector_gates': False,
        # Optional final rotation layer for vector node targets
        'vec_target_use_final_rotation_layer': False,
        # Vector gating feature controls
        # - If True, include flattened scalar features in vector gate input
        # (if True, this might leak invariant information from the scalar track and
        # break equivariant learning)
        'use_scalar_in_vector_gate': True,
        # - If True, include neighbor cosine stats in vector gate input
        'use_neighbor_cosines': True,
        # - If True, ignore inputs and use learned static per-wavelet weights
        'use_learned_static_vector_weights': False,
    }

    READOUT_DEFAULTS = { # graph-level tasks
        'type': 'mlp',  # 'mlp' (INVARIANT) or 'agg' (EQUIVARIANT)
        'mlp_hidden_dims': [128, 64, 32, 16],
        'mlp_nonlin': 'silu',  # 'silu' or 'relu'
        'node_pool_stats': ['mean', 'max'],  # supports 'mean', 'max', 'sum'
    }

    NEIGHBOR_DEFAULTS = {
        # 'equal_degree': False,  # deprecated
        'k_neighbors': 5,
        # 'use_padding': True,  # deprecated; for when not equal_degree
        'pool_stats': ['max', 'mean', 'var'],  # supports 'percentiles'|'quantiles', 'min', 'max', 'mean', 'var'
        # NOTE: quantile support is slow
        'quantiles_stride': 0.2,
    }
    
    def __init__(
        self,
        *,
        base_module_kwargs: Dict[str, Any],
        ablate_scalar_track: bool,
        ablate_vector_track: bool,
        scalar_track_kwargs: Dict[str, Any],
        vector_track_kwargs: Optional[Dict[str, Any]],
        mixing_kwargs: Optional[Dict[str, Any]] = None,
        neighbor_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
        readout_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(**base_module_kwargs)

        # Ablations
        self.ablate_scalar_track = bool(ablate_scalar_track)
        self.ablate_vector_track = bool(ablate_vector_track)

        # Signal to the trainer that this module initializes some submodules lazily
        # (e.g., LazyBatchNorm1d, within-track mixers, heads). The trainer will run
        # a dummy forward pass before DDP wrapping to materialize parameters.
        self.has_lazy_parameter_initialization = True

        # Track kwargs
        self.scalar_track_kwargs = scalar_track_kwargs
        self.vector_track_kwargs = vector_track_kwargs or {}
        self.vector_dim = self.vector_track_kwargs.get('vector_dim', 3)
        self.replace_nonfinite_second_order = bool(
            self.vector_track_kwargs.get('replace_nonfinite_second_order', True)
        )

        # Mixing MLP configuration
        self.mixing_kwargs = {
            **self.MIXING_DEFAULTS, 
            **(mixing_kwargs or {})
        }

        # Neighbor/cosine handling
        self.neighbor_kwargs = {
            **self.NEIGHBOR_DEFAULTS, 
            **(neighbor_kwargs or {})
        }

        # Lazily-created static vector wavelet logits (when configured)
        self.vector_wavelet_logits: Optional[nn.Parameter] = None

        # Optional pre-second-order nonlinearities/alignment
        # Scalar: apply nonlinearity to first-order scattering before second order
        # Apply when an explicit nonlinearity is requested (not None)
        self.apply_scalar_first_order_nonlin: bool = (
            self.scalar_track_kwargs.get('scalar_scatter_first_order_nonlin', None) is not None
        )
        self.scalar_first_order_nonlin_type: str = str(
            self.scalar_track_kwargs.get(
                'scalar_scatter_first_order_nonlin',
                self.scalar_track_kwargs.get('first_order_nonlin', 'abs'),
            )
        ).lower()

        # Vector: per-wavelet, per-graph alignment + gated decomposition before second order
        self.apply_vector_first_order_align_gating: bool = (
            (self.vector_track_kwargs is not None)
            and (self.vector_track_kwargs.get('first_order_align_gating_mode', 'ref_align') in ('ref_align', 'simple_affine', 'norm_only'))
            and (self.vector_track_kwargs.get('apply_vector_first_order_align_gating', None) is not None)
        )
        # Lazy params for vector alignment gates (depend on number of first-order wavelets)
        self._vector_align_params: Optional[nn.ParameterDict] = None

        # Scalar BN after scattering (optional)
        self.scalar_bn = None
        if (not self.ablate_scalar_track) and self.mixing_kwargs.get('use_scalar_batch_norm', True):
            # Lazy BN over flattened (C*W_total)
            self.scalar_bn = nn.LazyBatchNorm1d(eps=1e-5, momentum=0.1, affine=True)

        # Vector BN after mixing (optional)
        self.use_vector_wavelet_batch_norm = bool(
            self.mixing_kwargs.get('use_vector_wavelet_batch_norm', False)
        )
        self.vector_bn_momentum = float(
            self.mixing_kwargs.get('vector_bn_momentum', 0.1)
        )
        self.vector_bn_eps = float(
            self.mixing_kwargs.get('vector_bn_eps', 1e-6)
        )
        self.vector_bn_track_running_stats = bool(
            self.mixing_kwargs.get('vector_bn_track_running_stats', True)
        )
        self.vector_bn: Optional[VectorBatchNorm] = None

        # Placeholders for per-track mixing MLPs; lazy init when W_total known
        self.scalar_mixer: Optional[ScalarScatPathMixer] = None
        self.vector_mixer: Optional[nn.Module] = None

        # Final heads
        # Node-level scalar head: configurable MLP (bias allowed)
        self.node_scalar_head: Optional[nn.Sequential] = None
        # Node-level vector gating head: produces W' weights for gating sum
        self.node_vector_gate: Optional[nn.Sequential] = None
        # Optional final rotation MLP producing (alpha, beta)
        self.vector_final_rotation_mlp: Optional[nn.Sequential] = None
        # Optional learned temperature for softmax gating (log-parameterized)
        self.vector_gate_log_temperature: Optional[nn.Parameter] = None

        # Graph-level aggregation type
        self.graph_agg: Literal['sum', 'mean', 'max'] = 'sum'

        # Head configuration (can be passed via head_kwargs)
        self.head_kwargs = {
            **self.HEAD_DEFAULTS, 
            **(head_kwargs or {})
        }

        # Readout configuration (graph-level)
        self.readout_kwargs = {
            **self.READOUT_DEFAULTS,
            **(readout_kwargs or {})
        }

        # Graph-level readout MLPs (lazy)
        self.graph_scalar_readout_mlp: Optional[nn.Sequential] = None
        # Note: For vector graph readouts, we keep aggregation-only to preserve equivariance

        # Cache for upper-triangular pair indices by wavelet width (nW)
        # Stored on CPU and moved to the active device on demand
        self._triu_indices_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------- Helpers -------------------------
    def _scatter(
        self,
        *,
        track: Literal['scalar', 'vector'],
        x0: torch.Tensor,
        P_or_Q: torch.Tensor,
        kwargs: Dict[str, Any],
        batch_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenate 0th, 1st, and 2nd order scattering outputs along wavelet axis.
        - Scalar input x0 shape: (N, C) -> returns (N, C, W)
        - Vector input x0 shape: (N, d) -> returns (N, 1, d, W)
        """
        # print(f'_scatter: track={track}')
        num_layers = kwargs['num_layers']
        nnu.raise_if_nonfinite_tensor(x0, name=f"_scatter: {track} x0")
        if P_or_Q is not None and isinstance(P_or_Q, torch.Tensor):
            if P_or_Q.is_sparse:
                nnu.raise_if_nonfinite_tensor(P_or_Q._values(), name=f"_scatter: {track} P_or_Q values")
            else:
                nnu.raise_if_nonfinite_tensor(P_or_Q, name=f"_scatter: {track} P_or_Q")

        if track == 'scalar':

            # 1st-order scattering
            W1 = batch_scatter(
                x=x0,
                P_sparse=P_or_Q,
                # vector_dim=None, # deprecated
                **kwargs['diffusion_kwargs'],
            )  # (N, C, W1)
            nnu.raise_if_nonfinite_tensor(W1, name="_scatter: scalar W1")

            # Preserve original 1st-order for output; optionally transform for 2nd-order only
            W1_orig = W1
            W1_for_second = W1_orig
            if self.apply_scalar_first_order_nonlin \
            and num_layers > 1:
                W1_for_second = self._apply_scalar_first_order_nonlin(
                    W1_orig,
                    nonlin=self.scalar_first_order_nonlin_type,
                )

            # 2nd-order: re-apply wavelets to first-order coefficients, then subset (i<j)
            W2 = None
            if num_layers > 1:
                N, C, nW = int(W1_for_second.shape[0]), int(W1_for_second.shape[1]), int(W1_for_second.shape[-1])
                if nW >= 2:
                    x_second = W1_for_second.reshape(N, C * nW)  # (N, C*W1)
                    W2raw = batch_scatter(
                        x=x_second,
                        P_sparse=P_or_Q,
                        **kwargs['diffusion_kwargs'],
                    )  # (N, C*W1, W1)
                    if not torch.isfinite(W2raw).all():
                        if self.replace_nonfinite_second_order:
                            print(
                                "[WARN] Replacing non-finite values in scalar "
                                "second-order scattering with zeros."
                            )
                            W2raw = torch.nan_to_num(
                                W2raw,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                        else:
                            nnu.raise_if_nonfinite_tensor(W2raw, name="_scatter: scalar W2raw")
                    # (N, C, W1_prev, W1_new)
                    W2raw = W2raw.view(N, C, nW, -1)
                    W2 = subset_second_order_wavelets(
                        W2raw,
                        feature_type='scalar',
                    )  # (N, C, W2)
                    nnu.raise_if_nonfinite_tensor(W2, name="_scatter: scalar W2")

                # Alternative: use interactions of 1st-order coefficients, via lower-on-higher indexed pairs
                # nW = W1_for_second.shape[-1]
                # i_idx, j_idx = self._get_triu_pair_indices(nW, device=W1_orig.device)
                # W2 = W1_for_second[..., i_idx] * W1_for_second[..., j_idx]  # (N, C, nW*(nW-1)/2)

            # Zeroth order
            # print(f'x0.shape: {x0.shape}')
            # print(f'W1_orig.shape: {W1_orig.shape}')
            W0 = x0.unsqueeze(-1)  # (N, 1) or (N, C, 1)
            # print(f'W0.shape: {W0.shape}')
            # print(f'W2.shape: {W2.shape}')
            if W0.ndim == 2:
                W0 = W0.unsqueeze(-1)  # (N, C, 1) for sure

            # Concat 0th, 1st, 2nd-order scattering coefficients
            W_to_cat = [W0, W1_orig] + [W2] if W2 is not None else [W0, W1_orig]
            out = torch.cat(W_to_cat, dim=-1)  # (N, C, W)
            nnu.raise_if_nonfinite_tensor(out, name="_scatter: scalar out")
            return out

        # Vector track
        elif track == 'vector':
            N, d = x0.shape
            flat = x0.reshape(N * d, 1)
            nnu.raise_if_nonfinite_tensor(flat, name="_scatter: vector flat")

            # 1st-order scattering
            W1v = batch_scatter(
                x=flat,
                P_sparse=P_or_Q,
                # vector_dim=d, # deprecated
                **kwargs['diffusion_kwargs'],
            )  # (N*d, 1, W1)
            nnu.raise_if_nonfinite_tensor(W1v, name="_scatter: vector W1v")

            # Between 1st and 2nd-order scattering
            # Preserve original 1st-order for output; optionally transform for 2nd-order only
            W1v_orig = W1v
            W1v_for_second = W1v_orig
            # Optional vector alignment/gating before 2nd-order products (equivariant)
            if self.apply_vector_first_order_align_gating \
            and (batch_index is not None) \
            and num_layers > 1:
                gating_mode = str((self.vector_track_kwargs or {}).get('first_order_align_gating_mode', 'ref_align')).lower()
                if gating_mode in ('simple_affine', 'norm_only'):
                    # Apply simple affine gate per wavelet channel
                    W1 = W1v_orig.shape[-1]
                    # Build (N, d, 1, W)
                    v_nd1w = W1v_orig.squeeze(1).view(N, d, W1).unsqueeze(2)

                    # Lazy-init or resize gating parameters to match W1
                    if not hasattr(self, '_vector_first_order_simple_gate') \
                    or (self._vector_first_order_simple_gate is None) \
                    or (self._vector_first_order_simple_gate.num_channels != W1):
                        self._vector_first_order_simple_gate = SimpleAffineGate(
                            num_channels=W1,
                            sigma=str(self.vector_track_kwargs['vector_first_order_align_gating_nonlinearity']).lower(),
                            use_norm_only=(gating_mode == 'norm_only'),
                        ).to(W1v_orig.device)

                    # Apply simple affine gate per wavelet channel
                    v_gated = self._vector_first_order_simple_gate(v_nd1w)  # (N, d, 1, W)
                    W1v_for_second = v_gated.squeeze(2).view(N * d, 1, W1)

                elif gating_mode == 'ref_align':
                    W1v_for_second = self._apply_vector_first_order_align_gating(
                        W1v_orig,
                        N=N,
                        d=d,
                        batch_index=batch_index,
                    )  # (N*d, 1, W1)
                else:
                    raise ValueError(f"Invalid gating mode: {gating_mode}")            

            # 2nd-order: re-apply wavelets to first-order coefficients, then subset (i<j)
            W2v = None
            nW = int(W1v_for_second.shape[-1])
            if num_layers > 1 and nW >= 2:
                x_second = W1v_for_second.squeeze(1)  # (N*d, W1)
                W2raw = batch_scatter(
                    x=x_second,
                    P_sparse=P_or_Q,
                    **kwargs['diffusion_kwargs'],
                )  # (N*d, W1, W1)
                if not torch.isfinite(W2raw).all():
                    if self.replace_nonfinite_second_order:
                        print(
                            "[WARN] Replacing non-finite values in vector "
                            "second-order scattering with zeros."
                        )
                        W2raw = torch.nan_to_num(
                            W2raw,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    else:
                        nnu.raise_if_nonfinite_tensor(W2raw, name="_scatter: vector W2raw")
                Nd = int(x_second.shape[0])
                W2raw = W2raw.view(Nd, nW, -1)
                W2v = subset_second_order_wavelets(
                    W2raw,
                    feature_type='vector',
                )  # (N*d, 1, W2)
                nnu.raise_if_nonfinite_tensor(W2v, name="_scatter: vector W2v")

            # 0th-order scattering
            W0v = flat.unsqueeze(-1)  # (N*d, 1, 1)

            # Concat 0th, 1st, [optionally] 2nd-order scattering coefficients and reshape
            list_to_cat = [W0v, W1v_orig] + [W2v] if W2v is not None else [W0v, W1v_orig]
            W_tot = torch.cat(list_to_cat, dim=-1)  # (N*d, 1, W)
            # out = W_tot.view(N, d, 1, -1)
            # out = out.permute(0, 2, 1, 3)  # (N, 1, d, W)
            out = W_tot.view(N, 1, d, -1)  # (N, 1, d, W)
            nnu.raise_if_nonfinite_tensor(out, name="_scatter: vector out")

            return out

    
    # --------- 2nd-order wavelets helper ---------
    def _get_triu_pair_indices(
        self, 
        nW: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cached (i,j) indices for the strictly upper-triangular pairs of an nW x nW matrix.
        Indices are cached on CPU and moved to the requested device.
        """
        if nW not in self._triu_indices_cache:
            ij = torch.triu_indices(nW, nW, offset=1, device='cpu')  # (2, K)
            self._triu_indices_cache[nW] = (ij[0].contiguous(), ij[1].contiguous())
        i_cpu, j_cpu = self._triu_indices_cache[nW]
        return i_cpu.to(device), j_cpu.to(device)


    # --------- 1st-order wavelets nonlinearity helper ---------
    def _apply_scalar_first_order_nonlin(
        self,
        W1: torch.Tensor,
        *,
        nonlin: str = 'abs',
    ) -> torch.Tensor:
        """
        Apply a pointwise nonlinearity to first-order scalar scattering coefficients.

        Args:
            W1: torch.Tensor of shape (N, C, W1)
            nonlin: Name of nonlinearity; currently supports 'abs', 'relu', 'tanh'.
        Returns:
            torch.Tensor with the same shape as W1.
        """
        nl = str(nonlin).lower()
        if nl == 'abs' or nl == 'mod' or nl == 'modulus':
            return torch.abs(W1)
        if nl == 'relu':
            return F.relu(W1)
        if nl == 'tanh':
            return torch.tanh(W1)
        # Fallback: identity
        return W1


    # --------- 1st-order vector align gating helper ---------
    def _apply_vector_first_order_align_gating(
        self,
        W1v: torch.Tensor,
        *,
        N: int,
        d: int,
        batch_index: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Equivariant alignment and gated decomposition for first-order vector scattering.

        Steps per wavelet channel j within each graph b:
        - Compute reference unit vector u_hat[b, j, :] = mean_n v[n, j, :] / ||mean||.
        - Decompose v into parallel and perpendicular components w.r.t. u_hat.
        - Form per-node invariant scalars: |proj| and ||v_perp||.
        - Compute per-wavelet gates via affine-sigmoid using learned parameters.
        - Combine: v' = alpha * v_parallel + beta * v_perp.

        Args:
            W1v: torch.Tensor of shape (N*d, 1, W1)
            N: Number of nodes
            d: Vector dimensionality
            batch_index: torch.Tensor of shape (N,) with graph indices per node
        Returns:
            torch.Tensor of shape (N*d, 1, W1)
        """
        device = W1v.device
        # Reshape to (N, d, W1) then to (N, W1, d) for convenient broadcasting
        W1v_ndw = W1v.squeeze(1).view(N, d, -1)  # (N, d, W1)
        W1 = W1v_ndw.shape[-1]
        v = W1v_ndw.permute(0, 2, 1).contiguous()  # (N, W1, d)

        # Lazy-init or resize gating parameters to match W1
        # intialize a's to 1 and b's to 0
        def _init_or_update_align_params(nW: int) -> None:
            if (self._vector_align_params is None) \
            or (self._vector_align_params['a_par'].numel() != nW):
                self._vector_align_params = nn.ParameterDict({
                    'a_par': nn.Parameter(torch.ones(nW, dtype=torch.float32, device=device)),
                    'b_par': nn.Parameter(torch.zeros(nW, dtype=torch.float32, device=device)),
                    'a_perp': nn.Parameter(torch.ones(nW, dtype=torch.float32, device=device)),
                    'b_perp': nn.Parameter(torch.zeros(nW, dtype=torch.float32, device=device)),
                })
            else:
                # Ensure params on correct device
                for k, p in self._vector_align_params.items():
                    if p.device != device:
                        self._vector_align_params[k] = nn.Parameter(p.detach().to(device))

        _init_or_update_align_params(W1)

        # Compute per-graph means per wavelet channel: (B, W1, d)
        batch_index = batch_index.to(device=device, dtype=torch.long)
        B = int(batch_index.max().item()) + 1 \
            if batch_index.numel() > 0 else 1
        # Sum across nodes by graph
        v_flat = v.reshape(N, W1 * d)  # (N, W1*d)
        sum_by_graph = scatter(
            v_flat, batch_index, dim=0, dim_size=B, reduce='sum'
        )  # (B, W1*d)
        counts = scatter(
            torch.ones(N, 1, device=device, dtype=v.dtype), batch_index, dim=0, dim_size=B, reduce='sum'
        )  # (B, 1)
        counts = torch.clamp(counts, min=1.0)
        mean_by_graph = (sum_by_graph / counts).reshape(B, W1, d)  # (B, W1, d)
        # Normalize to unit vectors
        u_norm = torch.linalg.norm(mean_by_graph, dim=-1, keepdim=True)  # (B, W1, 1)
        u_hat = mean_by_graph / (u_norm + eps)
        # Broadcast u_hat to nodes
        u_hat_nodes = u_hat[batch_index]  # (N, W1, d)

        # Decompose v per node
        proj = (v * u_hat_nodes).sum(dim=-1, keepdim=True)  # (N, W1, 1)
        v_parallel = proj * u_hat_nodes  # (N, W1, d) [projection of v onto u_hat]
        v_perp = v - v_parallel  # (N, W1, d) [perpendicular component of v projected onto u_hat]
        perp_norm = torch.linalg.norm(v_perp, dim=-1, keepdim=True)  # (N, W1, 1)

        # Per-wavelet affine-sigmoid gates
        a_par = self._vector_align_params['a_par'].view(1, W1, 1)  # (1, W1, 1)
        b_par = self._vector_align_params['b_par'].view(1, W1, 1)
        a_perp = self._vector_align_params['a_perp'].view(1, W1, 1)
        b_perp = self._vector_align_params['b_perp'].view(1, W1, 1)
        alpha = torch.sigmoid(a_par * torch.abs(proj) + b_par)  # (N, W1, 1)
        beta = torch.sigmoid(a_perp * perp_norm + b_perp)       # (N, W1, 1)

        # Normalize alpha and beta so that alpha**2 + beta**2 = 1 (no in-place ops)
        # (this ensures that these parameters control only rotation/alignment of v_prime,
        # and don't re-scale)
        alpha_beta_norm = torch.sqrt(alpha.pow(2) + beta.pow(2)).clamp_min(1e-8)
        alpha_n = alpha / alpha_beta_norm
        beta_n = beta / alpha_beta_norm

        # Combine parallel and perpendicular components
        v_prime = (alpha_n * v_parallel) + (beta_n * v_perp)  # (N, W1, d)

        # Re-normalize v_prime to have the same norm as v
        v_prime_norm = torch.linalg.norm(v_prime, dim=-1, keepdim=True)
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        v_prime = v_prime * (v_norm / v_prime_norm)

        # Reshape back to (N*d, 1, W1)
        v_prime_ndw = v_prime.permute(0, 2, 1).contiguous()  # (N, d, W1)
        out = v_prime_ndw.view(N * d, 1, W1)  # (N*d, 1, W1)
        return out


    def _lazy_init_within_track_mixers(
        self,
        W_scalar: Optional[torch.Tensor],
        W_vector: Optional[torch.Tensor],
    ) -> None:
        # Determine in/out wavelet dims
        if (self.scalar_mixer is None) and (W_scalar is not None):
            W_in = W_scalar.shape[-1]
            W_out = self.mixing_kwargs.get('W_out_scalar') or W_in
            self.scalar_mixer = ScalarScatPathMixer(
                in_dim=W_in,
                out_dim=W_out,
                hidden_dims=self.mixing_kwargs.get('scalar_hidden_dims'),
                nonlin=self.mixing_kwargs.get('scalar_nonlin', nn.SiLU),
                dropout_p=self.mixing_kwargs.get('scalar_dropout_p', 0.0),
                bias=True,
            ).to(W_scalar.device)

        if (self.vector_mixer is None) and (W_vector is not None):
            W_in = W_vector.shape[-1]
            W_out_cfg = self.mixing_kwargs.get('W_out_vector')
            mixer_dims_cfg = self.mixing_kwargs.get('vector_wavelet_mixer_linear_dim')
            v_mix_mode = str(self.mixing_kwargs.get('vector_wavelet_mixing_gate_mode', 'norm_only')).lower()
            gate_sigma = str(self.mixing_kwargs.get('vector_wavelet_mixer_gate_nonlinearity', 'sigmoid')).lower()
            use_gate = v_mix_mode in ('simple_affine', 'norm_only', 'param_norm')

            # Normalize dims to a list[str/int] -> List[int]
            def _to_list(v):
                if v is None:
                    return None
                if isinstance(v, (list, tuple)):
                    return list(map(int, v))
                return [int(v)]

            out_list = _to_list(W_out_cfg)
            mix_list = _to_list(mixer_dims_cfg)

            # Prefer explicit W_out list; otherwise use mixer dims; default to identity (single layer with same dim)
            layer_out_dims: List[int]
            if out_list is not None:
                layer_out_dims = out_list
            elif mix_list is not None:
                layer_out_dims = mix_list
            else:
                layer_out_dims = [W_in]

            # If both provided, ensure same length
            if (out_list is not None) \
            and (mix_list is not None) \
            and (len(out_list) != len(mix_list)):
                raise ValueError("W_out_vector and vector_wavelet_mixer_linear_dim must have the same length when both are provided.")

            if len(layer_out_dims) == 1:
                self.vector_mixer = VectorScatPathMixer(
                    in_dim=W_in,
                    out_dim=layer_out_dims[0],
                    use_gate=use_gate,
                    gate_sigma=gate_sigma,
                    use_norm_only=(v_mix_mode == 'norm_only'),
                    gate_mode=v_mix_mode,
                ).to(W_vector.device)
            else:
                self.vector_mixer = MultiLayerVectorScatPathMixer(
                    in_dim=W_in,
                    layer_out_dims=layer_out_dims,
                    use_gate=use_gate,
                    gate_sigma=gate_sigma,
                    gate_mode=v_mix_mode,
                ).to(W_vector.device)


    def _optional_scalar_bn(
        self,
        W_scalar: torch.Tensor,
    ) -> torch.Tensor:
        if self.scalar_bn is None:
            return W_scalar
        N, C, W = W_scalar.shape
        flat = W_scalar.reshape(N, C * W)
        flat = self.scalar_bn(flat)
        return flat.reshape(N, C, W)

    def _optional_vector_bn(
        self,
        W_vector: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_vector_wavelet_batch_norm:
            return W_vector

        if W_vector.dim() != 4:
            raise ValueError(f"Vector BN expects 4D input, got {W_vector.shape}")

        # Accept both (N, 1, d, W) and (N, W, 1, d) layouts
        if W_vector.shape[1] == 1:
            num_wavelets = int(W_vector.shape[3])
            Wv = W_vector
            perm_back = False
        else:
            num_wavelets = int(W_vector.shape[1])
            Wv = W_vector.permute(0, 2, 3, 1)  # (N, 1, d, W)
            perm_back = True

        if (self.vector_bn is None) or (self.vector_bn.num_wavelets != num_wavelets):
            self.vector_bn = VectorBatchNorm(
                num_wavelets=num_wavelets,
                momentum=self.vector_bn_momentum,
                eps=self.vector_bn_eps,
                track_running_stats=self.vector_bn_track_running_stats,
            ).to(W_vector.device)

        Wv = self.vector_bn(Wv)
        if perm_back:
            return Wv.permute(0, 3, 1, 2)  # (N, W, 1, d)
        return Wv


    def _get_vector_invariants(
        self,
        x: torch.Tensor,
        batch: Batch,
        *,
        invariant_mode: str = 'neighbor_cosines',
    ) -> torch.Tensor:  # type: ignore[name-defined]
        """
        Unified vector invariants:
        - If x has shape (N, W', 1, d): return (N, W' + W'*m) where first W' are norms per wavelet and rest are pooled neighbor cosine stats.
        - If x has shape (N, d): return (N, 1 + m) where first col is norm and rest are pooled neighbor cosine stats (if enabled).

        The parameter 'invariant_mode' controls the type of vector invariants to compute.
        - 'neighbor_cosines': compute cosine similarities between neighbors
        - 'intra_wavelet_dot': compute dot products between normalized wavelet-filtered vectors within each node
        """
        include_cosines = bool(self.head_kwargs.get('use_neighbor_cosines', True))
        edge_index = batch.edge_index  # (2, E)
        src, dst = edge_index[0], edge_index[1]

        # Case A: per-wavelet invariants (N, W', 1, d)
        if x.dim() == 4:
            N, Wp = x.shape[0], x.shape[1]
            v = x.squeeze(2)  # (N, W', d)
            norms = torch.linalg.norm(v, dim=-1)  # (N, W')

            # If cosines disabled and not using intra-wavelet dots, return only norms per wavelet
            if (not include_cosines) \
            and (invariant_mode == 'neighbor_cosines'):
                return norms

            if invariant_mode == 'neighbor_cosines':
                # Cosines per wavelet between neighbors
                v_norm = torch.clamp(
                    torch.linalg.norm(v, dim=-1, keepdim=True), 
                    min=1e-8
                )  # (N, W', 1)
                v_dst = v[dst]  # (E, W', d)
                v_src = v[src]  # (E, W', d)
                denom = (v_norm[dst].squeeze(-1) * v_norm[src].squeeze(-1))  # (E, W')
                cos_all = (v_dst * v_src).sum(dim=-1) / denom  # (E, W')

                stats: Sequence[str] = self.neighbor_kwargs.get(
                    'pool_stats', ['percentiles', 'min', 'max', 'mean']
                )
                pooled_cols: List[torch.Tensor] = []
                pooled_mean: Optional[torch.Tensor] = None
                if ('mean' in stats) or ('var' in stats):
                    pooled_mean = scatter(cos_all, dst, dim=0, dim_size=N, reduce='mean')  # (N, W')
                    if 'mean' in stats:
                        pooled_cols.append(pooled_mean)
                if 'max' in stats:
                    pooled_cols.append(scatter(cos_all, dst, dim=0, dim_size=N, reduce='max'))
                if 'min' in stats:
                    pooled_cols.append(scatter(cos_all, dst, dim=0, dim_size=N, reduce='min'))
                if 'var' in stats:
                    mean_sq = scatter(cos_all * cos_all, dst, dim=0, dim_size=N, reduce='mean')
                    var_vals = torch.clamp(mean_sq - pooled_mean.pow(2), min=0.0)  # type: ignore[arg-type]
                    pooled_cols.append(var_vals)

                # Quantiles
                use_quantiles = ('quantiles' in stats) or ('percentiles' in stats)
                if use_quantiles:
                    stride = float(self.neighbor_kwargs.get('quantiles_stride', 0.2))
                    if stride > 0.0 and stride < 1.0:
                        levels = torch.arange(stride, 1.0, stride, device=x.device)
                        pooled_cols.extend(self._compute_per_node_quantiles(cos_all, dst, N, levels))

                pooled = torch.cat(pooled_cols, dim=1) if pooled_cols else torch.zeros(N, 0, device=x.device)
                return torch.cat([norms, pooled], dim=1)  # (N, W' + W'*m)

            elif invariant_mode == 'intra_wavelet_dot':
                # Pairwise dot products between normalized wavelet-filtered vectors within each node
                v_norm = torch.clamp(
                    torch.linalg.norm(v, dim=-1, keepdim=True), 
                    min=1e-8
                )  # (N, W', 1)
                v_unit = v / v_norm  # (N, W', d)
                # Compute Gram matrix per node: (N, W', W')
                G = torch.einsum('nwd,nvd->nwv', v_unit, v_unit)
                # Extract upper triangle without diagonal for each node and flatten
                idx_i, idx_j = torch.triu_indices(
                    Wp, Wp, offset=1, device=x.device
                )
                G_pairs = G[:, idx_i, idx_j]  # (N, W'*(W'-1)/2)
                return torch.cat([norms, G_pairs], dim=1)
            else:
                raise ValueError(
                    f"Unknown invariant_mode: {invariant_mode}"
                )

        # Case B: single vector per node (N, d)
        elif x.dim() == 2:
            N = x.shape[0]
            norms = torch.linalg.norm(x, dim=-1, keepdim=True)  # (N, 1)
            if (not include_cosines) and (invariant_mode == 'neighbor_cosines'):
                return norms

            if invariant_mode == 'neighbor_cosines':
                v = x  # (N, d)
                v_norm = torch.clamp(
                    torch.linalg.norm(v, dim=-1, keepdim=True), 
                    min=1e-8
                )  # (N, 1)
                v_dst = v[dst]  # (E, d)
                v_src = v[src]  # (E, d)
                denom = (v_norm[dst] * v_norm[src]).squeeze(-1)  # (E,)
                cos_all = ((v_dst * v_src).sum(dim=-1) / denom).unsqueeze(-1)  # (E, 1)

                stats: Sequence[str] = self.neighbor_kwargs.get(
                    'pool_stats', ['percentiles', 'min', 'max', 'mean']
                )
                pooled_cols: List[torch.Tensor] = []
                pooled_mean: Optional[torch.Tensor] = None
                if ('mean' in stats) or ('var' in stats):
                    pooled_mean = scatter(cos_all, dst, dim=0, dim_size=N, reduce='mean')  # (N, 1)
                    if 'mean' in stats:
                        pooled_cols.append(pooled_mean)
                if 'max' in stats:
                    pooled_cols.append(scatter(cos_all, dst, dim=0, dim_size=N, reduce='max'))
                if 'min' in stats:
                    pooled_cols.append(scatter(cos_all, dst, dim=0, dim_size=N, reduce='min'))
                if 'var' in stats:
                    mean_sq = scatter(cos_all * cos_all, dst, dim=0, dim_size=N, reduce='mean')
                    var_vals = torch.clamp(mean_sq - pooled_mean.pow(2), min=0.0)  # type: ignore[arg-type]
                    pooled_cols.append(var_vals)

                # Quantiles
                use_quantiles = ('quantiles' in stats) or ('percentiles' in stats)
                if use_quantiles:
                    stride = float(self.neighbor_kwargs.get('quantiles_stride', 0.2))
                    if stride > 0.0 and stride < 1.0:
                        levels = torch.arange(stride, 1.0, stride, device=x.device)
                        pooled_cols.extend(self._compute_per_node_quantiles(cos_all, dst, N, levels))

                pooled = torch.cat(pooled_cols, dim=1) if pooled_cols else torch.zeros(N, 0, device=x.device)
                return torch.cat([norms, pooled], dim=1)  # (N, 1 + m)
            elif invariant_mode == 'intra_wavelet_dot':
                # No wavelet axis in this case; return norms only
                return norms
            else:
                raise ValueError(f"Unknown invariant_mode: {invariant_mode}")

        else:
            raise ValueError("_get_vector_invariants expects input of shape (N, W', 1, d) or (N, d)")


    def _compute_per_node_quantiles(
        self,
        cos_all: torch.Tensor,    # (E, C)
        dst: torch.Tensor,        # (E,)
        N: int,
        levels: torch.Tensor,     # (Q,)
    ) -> List[torch.Tensor]:
        """
        Compute per-node quantiles for each column in cos_all at the requested levels.
        Returns list of tensors, one per level, each of shape (N, C).
        """
        if levels.numel() == 0:
            return []
        # Group edges by destination to form contiguous segments per node
        perm = torch.argsort(dst, stable=True)
        dst_sorted = dst[perm]
        cos_sorted = cos_all[perm]  # (E, C)
        counts = torch.bincount(dst_sorted, minlength=N)
        starts = torch.cumsum(counts, dim=0) - counts  # (N,)
        result: List[torch.Tensor] = []
        C = cos_sorted.shape[1]
        device = cos_all.device
        for p in levels.tolist():
            out_p = torch.zeros((N, C), device=device, dtype=cos_all.dtype)
            for n in range(N):
                cnt = int(counts[n].item())
                if cnt <= 0:
                    continue
                seg = cos_sorted[starts[n]: starts[n] + cnt, :]  # (cnt, C)
                k = int((cnt - 1) * p)
                out_p[n] = torch.kthvalue(seg, k=k + 1, dim=0).values
            result.append(out_p)
        return result

    # --------- Node-level vector output helper ---------
    def _learn_gated_node_vector(
        self,
        *,
        W_vector: torch.Tensor,               # (N, W', 1, d)
        W_scalar: Optional[torch.Tensor],     # (N, W', C) or None
        vector_invariants: torch.Tensor,
        t_gate: torch.Tensor,
        batch: Batch,                   # for neighbor info if needed
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute node-level vector predictions by per-wavelet gating and summation.

        Strategy:
        - Build per-node gate features t_gate by concatenating, when enabled:
          (a) flattened scalar track (if present), 
          (b) vector norms per wavelet, and
          (c) optional neighbor cosine statistics.
        - Two gating modes are supported:
          1) Static per-wavelet weights: a single learnable logit per wavelet shared across nodes.
          2) Data-driven gates: an MLP maps t_gate to per-node logits.
        - Normalization behavior is controlled by head_kwargs['normalize_final_vector_gates']:
          - If True and using sigmoid gates: apply sigmoid then L1 normalization across wavelets.
          - If True and using softmax mode: apply temperature-softmax across wavelets.
          - If False: use raw sigmoid outputs (sigmoid mode) or raw logits (softmax mode) without normalization.
        - The final node vector is the weighted sum across wavelets of W_vector, preserving SO(d) equivariance.

        Returns:
        - torch.Tensor of shape (N, 1, d) containing node-level vector predictions.
        """
        use_scalar_in_gate = bool(self.head_kwargs.get('use_scalar_in_vector_gate', True))
        # Number of mixed wavelets
        Wp = W_vector.shape[1]

        # Assemble gate features by optionally removing scalar-flat from provided t_gate
        s_flat_dim = 0
        if W_scalar is not None:
            Ns, Wps, C = W_scalar.shape
            s_flat_dim = Wps * C
        if use_scalar_in_gate or (s_flat_dim == 0):
            t_gate_used = t_gate
        else:
            # Exclude scalar-flat portion; keep invariants only
            t_gate_used = t_gate[:, s_flat_dim:]

        normalize_final: bool = bool(self.head_kwargs.get('normalize_final_vector_gates', True))
        use_sigmoid_gate: bool = bool(self.head_kwargs.get('vector_gate_use_sigmoid'))

        # Case 1: static per-wavelet weights shared across nodes
        if self.head_kwargs.get('use_learned_static_vector_weights'):
            if (self.vector_wavelet_logits is None) or (self.vector_wavelet_logits.numel() != Wp):
                self.vector_wavelet_logits = nn.Parameter(torch.randn(Wp, device=device))
            if normalize_final:
                gates = torch.softmax(self.vector_wavelet_logits, dim=0).unsqueeze(0).expand(W_vector.size(0), -1)
            else:
                gates = self.vector_wavelet_logits.unsqueeze(0).expand(W_vector.size(0), -1)

        # Case 2: data-driven gates from t_gate via MLP
        elif t_gate_used is not None:
            need_reinit_gate = False
            expected_in_dim = t_gate_used.shape[1]
            if self.node_vector_gate is None:
                need_reinit_gate = True
            else:
                try:
                    first_linear = next(m for m in self.node_vector_gate if isinstance(m, nn.Linear))
                    last_linear = next(m for m in self.node_vector_gate[::-1] if isinstance(m, nn.Linear))
                    if last_linear.out_features != Wp or first_linear.in_features != expected_in_dim:
                        need_reinit_gate = True
                except StopIteration:
                    need_reinit_gate = True
            if need_reinit_gate:
                self._lazy_init_node_vector_gate(
                    in_dim=expected_in_dim,
                    out_dim=Wp, 
                    device=device,
                )

            # Learn learnable gating weights from invariant inputs
            logits = self.node_vector_gate(t_gate_used)  # (N, W')

            # Apply gating and (optional) normalization
            if use_sigmoid_gate:
                gates = torch.sigmoid(logits)
                if normalize_final:
                    # L1 normalization
                    gates_denom = gates.sum(dim=1, keepdim=True) + 1e-8
                    gates = gates / gates_denom
            else:
                if normalize_final:
                    if self.vector_gate_log_temperature is None:
                        init_temp = float(self.head_kwargs.get('vector_gate_init_temperature', 1.0))
                        init_temp = max(init_temp, 1e-3)
                        self.vector_gate_log_temperature = nn.Parameter(
                            torch.log(torch.tensor(init_temp, dtype=torch.float32, device=logits.device))
                        )
                    temperature = torch.exp(self.vector_gate_log_temperature)
                    gates = torch.softmax(
                        logits / temperature.clamp_min(1e-3), 
                        dim=1,
                    )
                else:
                    gates = logits

        # Case 3: uniform fallback
        else:
            gates = torch.full(
                (W_vector.size(0), Wp), 
                1.0 / max(Wp, 1), 
                device=device,
            )

        # Weighted sum across wavelets -> (N, 1, d)
        vec = (W_vector.squeeze(2) * gates.unsqueeze(-1)).sum(dim=1)
        return vec.unsqueeze(1)


    def _lazy_init_node_vector_final_rotation(
        self,
        *,
        in_dim: int,
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.head_kwargs.get('vector_gate_hidden'))
        nonlin_cls: type[nn.Module] = self.head_kwargs.get('vector_gate_mlp_nonlin', nn.SiLU)
        dims = [in_dim] + hidden + [2]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.vector_final_rotation_mlp = nn.Sequential(*layers).to(device)


    def _learn_node_vector_rotation(
        self,
        *,
        W_scalar: Optional[torch.Tensor],       # (N, W', C) or None
        vec_pred: torch.Tensor,                  # (N, 1, d)
        batch: Batch,
        device: torch.device,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Learn a final rotation of the predicted node vectors using an MLP that
        outputs (alpha, beta), normalized such that alpha**2 + beta**2 = 1.

        The MLP input concatenates the same scalar hidden features used in t
        (flattened W_scalar if enabled) with invariants recomputed from the
        current node vector predictions (norms and optionally neighbor cosine
        pooled statistics). The rotation aligns with a per-graph reference
        unit vector computed as the mean vector normalized to unit length.
        """
        v = vec_pred.squeeze(1)  # (N, d)
        N, d = v.shape

        # Build rotation MLP input
        feats: List[torch.Tensor] = []
        if self.head_kwargs['use_scalar_in_vector_gate'] \
        and (W_scalar is not None):
            # Include flattened scalar features
            N_s, W_s, C = W_scalar.shape
            feats.append(W_scalar.reshape(N_s, W_s * C))  # (N, Ws*C)
        # Get vector invariants for node vectors
        feats.append(self._get_vector_invariants(v, batch))  # (N, 1 + m)
        t_rot = torch.cat(feats, dim=1)  # (N, Ws*C + 1 + m)

        # Lazy init / resize rotation head
        need_reinit = False
        expected_in_dim = t_rot.shape[1]
        if self.vector_final_rotation_mlp is None:
            need_reinit = True
        else:
            try:
                first_linear = next(m for m in self.vector_final_rotation_mlp if isinstance(m, nn.Linear))
                last_linear = next(m for m in self.vector_final_rotation_mlp[::-1] if isinstance(m, nn.Linear))
                if last_linear.out_features != 2 \
                or first_linear.in_features != expected_in_dim:
                    need_reinit = True
            except StopIteration:
                need_reinit = True
        if need_reinit:
            self._lazy_init_node_vector_final_rotation(
                in_dim=expected_in_dim,
                device=device,
            )

        # Get predicted alpha and beta
        alpha_beta = self.vector_final_rotation_mlp(t_rot)  # (N, 2)
        alpha_raw, beta_raw = alpha_beta[:, 0:1], alpha_beta[:, 1:2]  # (N,1) each

        # Normalize alpha and beta so that alpha**2 + beta**2 = 1
        norm_ab = torch.sqrt(alpha_raw.pow(2) + beta_raw.pow(2) + eps)
        alpha = alpha_raw / norm_ab
        beta = beta_raw / norm_ab

        # Reference vector per graph (mean vec, normalized)
        batch_index = batch.batch if hasattr(batch, 'batch') else None
        if batch_index is None:
            mean_v = v.mean(dim=0, keepdim=True)  # (1, d)
            u_hat_nodes = mean_v / (torch.linalg.norm(mean_v, dim=-1, keepdim=True) + eps)
            u_hat_nodes = u_hat_nodes.expand(N, -1)
        else:
            batch_index = batch_index.to(device=v.device, dtype=torch.long)
            B = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
            sum_by_graph = scatter(v, batch_index, dim=0, dim_size=B, reduce='sum')  # (B, d)
            counts = scatter(torch.ones(N, 1, device=v.device, dtype=v.dtype), batch_index, dim=0, dim_size=B, reduce='sum')  # (B,1)
            counts = torch.clamp(counts, min=1.0)
            mean_by_graph = sum_by_graph / counts  # (B, d)
            u_hat = mean_by_graph / (torch.linalg.norm(mean_by_graph, dim=-1, keepdim=True) + eps)  # (B, d)
            u_hat_nodes = u_hat[batch_index]  # (N, d)

        # Decompose and rotate
        proj = (v * u_hat_nodes).sum(dim=-1, keepdim=True)  # (N,1)
        v_parallel = proj * u_hat_nodes  # (N, d)
        v_perp = v - v_parallel          # (N, d)
        v_prime = (alpha * v_parallel) + (beta * v_perp)  # (N, d)

        # Preserve original norm
        v_prime_norm = torch.linalg.norm(v_prime, dim=-1, keepdim=True)
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        v_prime = v_prime * (v_norm / (v_prime_norm + eps))

        return v_prime.unsqueeze(1)  # (N, 1, d)


    # --------------------------- Forward ---------------------------
    def forward(
        self, batch
    ) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        
        # Initialize outputs dict
        outputs: Dict[str, torch.Tensor] = {}

        # Include additional attributes with predictions/targets in the model output dict
        # in case the loss function needs them
        if self.attributes_to_include_with_preds is not None:
            for attr in self.attributes_to_include_with_preds:
                if attr in batch:
                    outputs[attr] = getattr(batch, attr)

        # Extract base tensors
        feature_keys: List[str] = []
        operator_keys: List[str] = []
        if not self.ablate_scalar_track:
            feature_keys.append(self.scalar_track_kwargs.get('feature_key'))
            operator_keys.append(self.scalar_track_kwargs.get('diffusion_op_key'))
        if not self.ablate_vector_track:
            feature_keys.append(self.vector_track_kwargs.get('feature_key'))
            operator_keys.append(self.vector_track_kwargs.get('diffusion_op_key'))
        device = infer_device_from_batch(
            batch,
            feature_keys=feature_keys,
            operator_keys=operator_keys,
        )
        x_s = None if self.ablate_scalar_track \
            else getattr(batch, self.scalar_track_kwargs['feature_key'])
        x_v = None if self.ablate_vector_track \
            else getattr(batch, self.vector_track_kwargs['feature_key'])
        P = getattr(batch, self.scalar_track_kwargs['diffusion_op_key']) \
            if not self.ablate_scalar_track else None
        Q = getattr(batch, self.vector_track_kwargs['diffusion_op_key']) \
            if not self.ablate_vector_track else None

        if x_s is not None:
            x_s = x_s.to(device)
            P = P.to(device)
            nnu.raise_if_nonfinite_tensor(x_s, name="forward: x_s")
            if P is not None and isinstance(P, torch.Tensor):
                if P.is_sparse:
                    nnu.raise_if_nonfinite_tensor(P._values(), name="forward: P values")
                else:
                    nnu.raise_if_nonfinite_tensor(P, name="forward: P")
        if x_v is not None:
            x_v = x_v.to(device)
            Q = Q.to(device)
            nnu.raise_if_nonfinite_tensor(x_v, name="forward: x_v")
            if Q is not None and isinstance(Q, torch.Tensor):
                if Q.is_sparse:
                    nnu.raise_if_nonfinite_tensor(Q._values(), name="forward: Q values")
                else:
                    nnu.raise_if_nonfinite_tensor(Q, name="forward: Q")

        # Task flag used to decide which tracks are needed
        task = self.task

        # --- 1. Scattering ---
        # 0th, 1st, 2nd order scattering coeffs, concatenated
        W_scalar = None
        
        # Always compute scalar track when not ablated so it can participate in vector tasks
        if (not self.ablate_scalar_track) \
        and (x_s is not None) \
        and (P is not None):
            W_scalar = self._scatter(
                track='scalar', 
                x0=x_s, 
                P_or_Q=P, 
                kwargs=self.scalar_track_kwargs,
                batch_index=(batch.batch if hasattr(batch, 'batch') else None),
            )  # (N, C, W)
            # Optional BN over flattened features
            W_scalar = self._optional_scalar_bn(W_scalar)
            nnu.raise_if_nonfinite_tensor(W_scalar, name="forward: W_scalar")

        W_vector = None
        if not self.ablate_vector_track:
            W_vector = self._scatter(
                track='vector', 
                x0=x_v, 
                P_or_Q=Q, 
                kwargs=self.vector_track_kwargs,
                batch_index=(batch.batch if hasattr(batch, 'batch') else None),
            )  # (N, 1, d, W)
            nnu.raise_if_nonfinite_tensor(W_vector, name="forward: W_vector")
            W_vector = self._optional_vector_bn(W_vector)
            # nnu.raise_if_nonfinite_tensor(W_vector, name="forward: W_vector bn")

        # --- 2. Within-track mixing ---
        # Initialize mixers lazily now that W dims are known
        vec_for_init = W_vector.permute(0, 2, 1, 3) \
            if (W_vector is not None) else None  # (..., W)
        self._lazy_init_within_track_mixers(W_scalar, vec_for_init)

        # Apply within-track wavelet mixing MLPs
        if W_scalar is not None and self.scalar_mixer is not None:
            # (N, C, W) -> (N, C, W') then permute to (N, W', C)
            W_scalar = self.scalar_mixer(W_scalar)
            W_scalar = W_scalar.permute(0, 2, 1)
            nnu.raise_if_nonfinite_tensor(W_scalar, name="forward: W_scalar mixed")

        if W_vector is not None and self.vector_mixer is not None:
            # Rearrange to (N, d, 1, W) -> apply mixer on last dim -> (N, d, 1, W')
            Wv = W_vector.permute(0, 2, 1, 3)
            # Apply vector track mixer
            # (vector track mixer linear layers have no bias)
            Wv = self.vector_mixer(Wv)
            # Return to (N, W', 1, d)
            W_vector = Wv.permute(0, 3, 2, 1)
            nnu.raise_if_nonfinite_tensor(W_vector, name="forward: W_vector mixed")

        # --- 3. Cross-track mixing ---
        # Gather invariant features from both tracks (use norms and cosines from vector track)
        t_list: List[torch.Tensor] = []
        vector_invariants: Optional[torch.Tensor] = None
        if W_scalar is not None:
            Ns, Wp, C = W_scalar.shape
            t_list.append(W_scalar.reshape(Ns, Wp * C))
        if W_vector is not None:
            vector_invariants = self._get_vector_invariants(W_vector, batch)
            t_list.append(vector_invariants)
        t = None
        if t_list:
            t = torch.cat(t_list, dim=1)  # (N, W' + W'*m)
            nnu.raise_if_nonfinite_tensor(t, name="forward: t (cross-track features)")

        # Short-circuit, if graph-level scalar target task: MLP readout directly from pooled t (assumes batch.batch is present)
        if ('graph' in task) and (('vector' not in task) or self.ablate_vector_track):
            readout_type = str(self.readout_kwargs.get('type', 'mlp')).lower()
            if readout_type == 'mlp' \
            and (t is not None) \
            and hasattr(batch, 'batch'):
                outputs['graph_scalar'] = self._graph_scalar_readout_from_pooled_t(
                    t=t, 
                    batch_index=batch.batch,
                )
                outputs['preds'] = outputs['graph_scalar']
                return outputs

        # --- 4. Node-level predictions ---
        # (always computed, and pooled / read out after if needed)
        # a. Scalar targets or ablated vector track
        if ('vector' not in task) or self.ablate_vector_track:
            # Scalar target (node-level): MLP head from t -> (N, d_target)
            if self.node_scalar_head is None:
                in_dim = (t.shape[1] if t is not None else 0)
                self._lazy_init_node_scalar_head(
                    in_dim=in_dim, 
                    out_dim=self.target_dim, 
                    device=device,
                )
            pred = self.node_scalar_head(t)
            nnu.raise_if_nonfinite_tensor(pred, name="forward: node_scalar pred")
            # For node-level scalar targets with no vector track -> prediction is a MLP head from t -> (N, d_target)
            outputs['node_scalar'] = pred

        # b. Vector targets (and not-ablated vector track)
        # We use a vector gating MLP to gate the sum of the wavelets
        # (when not ablating the vector track)
        else:
            # Vector target (node-level): delegate to helper
            outputs['node_vector'] = self._learn_gated_node_vector(
                W_vector=W_vector,
                W_scalar=W_scalar,
                vector_invariants=vector_invariants,
                t_gate=t,
                batch=batch,
                device=device,
            )
            nnu.raise_if_nonfinite_tensor(outputs['node_vector'], name="forward: node_vector pred")

            # Optional final rotation layer to modulate output vector angle
            if self.head_kwargs['vec_target_use_final_rotation_layer']:
                outputs['node_vector'] = self._learn_node_vector_rotation(
                    W_scalar=W_scalar,
                    vec_pred=outputs['node_vector'],
                    batch=batch,
                    device=device,
                )

        # --- 5. [Optional] Graph-level readout heads ---
        # (Simple aggregations of node-level features)
        if 'graph' in task:
            agg = getattr(self, 'graph_agg', 'sum')
            batch_index = batch.batch if hasattr(batch, 'batch') else None
            # Graph-level vector target: aggregate node vector predictions
            if 'vector' in task and 'node_vector' in outputs:
                v = outputs['node_vector'].squeeze(1)  # (N, d)
                out_vec = self._aggregate_nodes(v, batch_index=batch_index, agg=agg)
                outputs['graph_vector'] = out_vec # (B, d)
                
            if 'node_scalar' in outputs:
                s = outputs['node_scalar']  # (N, d_target)
                out_s = self._aggregate_nodes(s, batch_index=batch_index, agg=agg)  # (B, d_target)
                readout_type = str(self.readout_kwargs.get('type', 'mlp')).lower()
                if readout_type == 'mlp':
                    # Lazily build MLP and apply
                    if self.graph_scalar_readout_mlp is None:
                        self._lazy_init_graph_scalar_readout_mlp(
                            in_dim=out_s.shape[1],
                            out_dim=self.target_dim,
                            device=out_s.device,
                        )
                    outputs['graph_scalar'] = self.graph_scalar_readout_mlp(out_s)
                else:
                    outputs['graph_scalar'] = out_s # (B, d_target)

        # --- 6. Select final predictions ---
        # Decide which tensor is the final prediction depending on task/target
        if 'graph' in task:
            if 'vector' in task and ('graph_vector' in outputs):
                outputs['preds'] = outputs['graph_vector']
            elif ('graph_scalar' in outputs):
                outputs['preds'] = outputs['graph_scalar']
        elif 'node' in task:
            if 'vector' in task and ('node_vector' in outputs):
                outputs['preds'] = outputs['node_vector']
            elif ('node_scalar' in outputs):
                outputs['preds'] = outputs['node_scalar']

        return outputs

    # -------------------- Epoch-zero initializer --------------------
    def run_epoch_zero_methods(self, batch: Batch) -> None:  # type: ignore[name-defined]
        """
        Materialize lazily-created submodules (mixers/heads) without requiring
        a full training step. This is invoked by the trainer at epoch 0 and is
        also safe to call during a pre-DDP dummy pass.
        """
        feature_keys: List[str] = []
        operator_keys: List[str] = []
        if not self.ablate_scalar_track:
            feature_keys.append(self.scalar_track_kwargs.get('feature_key'))
            operator_keys.append(self.scalar_track_kwargs.get('diffusion_op_key'))
        if not self.ablate_vector_track:
            feature_keys.append(self.vector_track_kwargs.get('feature_key'))
            operator_keys.append(self.vector_track_kwargs.get('diffusion_op_key'))
        device = infer_device_from_batch(
            batch,
            feature_keys=feature_keys,
            operator_keys=operator_keys,
        )
        x_s = None if self.ablate_scalar_track \
            else getattr(batch, self.scalar_track_kwargs['feature_key'])
        x_v = None if self.ablate_vector_track \
            else getattr(batch, self.vector_track_kwargs['feature_key'])
        P = getattr(batch, self.scalar_track_kwargs['diffusion_op_key']) \
            if not self.ablate_scalar_track else None
        Q = getattr(batch, self.vector_track_kwargs['diffusion_op_key']) \
            if not self.ablate_vector_track else None

        if x_s is not None:
            x_s = x_s.to(device)
            P = P.to(device)
        if x_v is not None:
            x_v = x_v.to(device)
            Q = Q.to(device)

        # Compute minimal scattering tensors to determine mixer/head shapes
        W_scalar = None
        if not self.ablate_scalar_track and x_s is not None and P is not None:
            W_scalar = self._scatter(
                track='scalar', 
                x0=x_s, 
                P_or_Q=P, 
                kwargs=self.scalar_track_kwargs,
                batch_index=(batch.batch if hasattr(batch, 'batch') else None)
            )
            W_scalar = self._optional_scalar_bn(W_scalar)

        W_vector = None
        if not self.ablate_vector_track and x_v is not None and Q is not None:
            W_vector = self._scatter(
                track='vector', 
                x0=x_v, 
                P_or_Q=Q, 
                kwargs=self.vector_track_kwargs,
                batch_index=(batch.batch if hasattr(batch, 'batch') else None)
            )

        # Initialize mixers based on discovered W-dimensions
        vec_for_init = W_vector.permute(0, 2, 1, 3) if (W_vector is not None) else None
        self._lazy_init_within_track_mixers(W_scalar, vec_for_init)

        # Apply within-track mixers identically to forward()
        if W_scalar is not None and self.scalar_mixer is not None:
            W_scalar = self.scalar_mixer(W_scalar)
            W_scalar = W_scalar.permute(0, 2, 1)  # (N, W', C)
        if W_vector is not None and self.vector_mixer is not None:
            Wv = W_vector.permute(0, 2, 1, 3)      # (N, d, 1, W)
            Wv = self.vector_mixer(Wv)
            W_vector = Wv.permute(0, 3, 2, 1)      # (N, W', 1, d)

        # Build cross-track feature t identically to forward()
        t_list: List[torch.Tensor] = []
        vector_invariants: Optional[torch.Tensor] = None
        if W_scalar is not None:
            Ns, Wp, C = W_scalar.shape
            t_list.append(W_scalar.reshape(Ns, Wp * C))
        if W_vector is not None:
            vector_invariants = self._get_vector_invariants(W_vector, batch)
            t_list.append(vector_invariants)
        t = torch.cat(t_list, dim=1) if len(t_list) > 0 else None

        # Initialize heads according to task routing
        task = self.task
        if ('vector' not in task) or self.ablate_vector_track:
            # For graph-scalar tasks with MLP readout, skip node head init to avoid unused params
            use_graph_scalar_mlp = (
                ('graph' in task)
                and (str(self.readout_kwargs.get('type', 'mlp')).lower() == 'mlp')
            )
            if not use_graph_scalar_mlp:
                # Node scalar head for scalar tasks
                in_dim = (t.shape[1] if t is not None else 0)
                if self.node_scalar_head is None:
                    self._lazy_init_node_scalar_head(in_dim=in_dim, out_dim=self.target_dim, device=device)
        else:
            # Vector gate head for vector tasks
            if W_vector is not None:
                Wp = W_vector.shape[1]
                # Build gate input exactly as in forward()
                gate_feats: List[torch.Tensor] = []
                # Always include flattened scalar features when scalar track is enabled
                if W_scalar is not None:
                    Ns, Wps, C = W_scalar.shape
                    gate_feats.append(W_scalar.reshape(Ns, Wps * C))
                # Reuse precomputed vector invariants; compute if missing
                if vector_invariants is None:
                    vector_invariants = self._get_vector_invariants(W_vector, batch)
                # Norms per wavelet (first Wp cols)
                gate_feats.append(vector_invariants[:, :Wp])  # (N, W')
                # Optional neighbor cosine pooled stats (remaining cols)
                if self.head_kwargs.get('use_neighbor_cosines'):
                    gate_feats.append(vector_invariants[:, Wp:])
                t_gate = torch.cat(gate_feats, dim=1) if gate_feats else None

                if self.node_vector_gate is None:
                    in_dim = (t_gate.shape[1] if t_gate is not None else 0)
                    self._lazy_init_node_vector_gate(in_dim=in_dim, out_dim=Wp, device=device)

        # Initialize graph-level scalar readout MLP if requested and task is graph-scalar
        if ('graph' in task) and (('vector' not in task) or self.ablate_vector_track):
            if str(self.readout_kwargs.get('type', 'mlp')).lower() == 'mlp':
                # Build pooled t to determine correct input dim (assumes batch.batch present)
                if hasattr(batch, 'batch'):
                    t_list: List[torch.Tensor] = []
                    if W_scalar is not None:
                        Ns, Wp, C = W_scalar.shape
                        t_list.append(W_scalar.reshape(Ns, Wp * C))
                    if W_vector is not None:
                        # Reuse precomputed invariants; compute if missing
                        if vector_invariants is None:
                            vector_invariants = self._get_vector_invariants(W_vector, batch)
                        t_list.append(vector_invariants)
                    t_epoch0 = torch.cat(t_list, dim=1) if len(t_list) > 0 else None
                    if t_epoch0 is not None:
                        _ = self._graph_scalar_readout_from_pooled_t(t=t_epoch0, batch_index=batch.batch)

        # If vector task and using final rotation layer, materialize its params by running once
        if ('vector' in task) and (not self.ablate_vector_track):
            if bool(self.head_kwargs.get('vec_target_use_final_rotation_layer', False)) and (W_vector is not None):
                # Build t as in forward
                t_list: List[torch.Tensor] = []
                vector_invariants: Optional[torch.Tensor] = None
                if W_scalar is not None:
                    Ns, Wp, C = W_scalar.shape
                    t_list.append(W_scalar.reshape(Ns, Wp * C))
                vector_invariants = self._get_vector_invariants(W_vector, batch)
                t_list.append(vector_invariants)
                t = torch.cat(t_list, dim=1) if len(t_list) > 0 else None
                # Get a provisional vec prediction to derive invariants for rotation head
                vec_pred = self._learn_gated_node_vector(
                    W_vector=W_vector,
                    W_scalar=W_scalar,
                    vector_invariants=vector_invariants,
                    t_gate=t,
                    batch=batch,
                    device=device,
                )
                # This call will lazily init the rotation head
                _ = self._learn_node_vector_rotation(
                    W_scalar=W_scalar,
                    vec_pred=vec_pred,
                    batch=batch,
                    device=device,
                )

    # --------------------------- Lazy init helpers ---------------------------
    def _lazy_init_node_scalar_head(
        self, 
        *, 
        in_dim: int, 
        out_dim: int, 
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.head_kwargs.get('node_scalar_head_hidden'))
        nonlin_cls: type[nn.Module] = self.head_kwargs.get('node_scalar_head_nonlin', nn.SiLU)
        dropout_p: float = float(self.head_kwargs.get('node_scalar_head_dropout', 0.0))
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.node_scalar_head = nn.Sequential(*layers).to(device)

    def _lazy_init_node_vector_gate(
        self,
        *, 
        in_dim: int, 
        out_dim: int, 
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.head_kwargs.get('vector_gate_hidden'))
        nonlin_cls: type[nn.Module] = self.head_kwargs.get('vector_gate_mlp_nonlin', nn.SiLU)
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.node_vector_gate = nn.Sequential(*layers).to(device)

        # Initialize learned temperature parameter for softmax gating when requested
        use_sigmoid: bool = bool(self.head_kwargs.get('vector_gate_use_sigmoid', True))
        if not use_sigmoid:
            init_temp = float(self.head_kwargs.get('vector_gate_init_temperature', 1.0))
            init_temp = max(init_temp, 1e-3)
            if (self.vector_gate_log_temperature is None) or (not isinstance(self.vector_gate_log_temperature, nn.Parameter)):
                self.vector_gate_log_temperature = nn.Parameter(
                    torch.log(torch.tensor(init_temp, dtype=torch.float32, device=device))
                )
            else:
                # Ensure it's on the correct device
                self.vector_gate_log_temperature = nn.Parameter(self.vector_gate_log_temperature.detach().to(device))

    def _lazy_init_graph_scalar_readout_mlp(
        self,
        *,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.readout_kwargs.get('mlp_hidden_dims', [128, 64, 32, 16]))
        nonlin_name: str = str(self.readout_kwargs.get('mlp_nonlin', 'silu')).lower()
        nonlin_cls: type[nn.Module] = nn.SiLU if nonlin_name == 'silu' else nn.ReLU
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.graph_scalar_readout_mlp = nn.Sequential(*layers).to(device)

    def _graph_scalar_readout_from_pooled_t(
        self,
        *,
        t: torch.Tensor,                  # (N, F)
        batch_index: torch.Tensor,        # (N,)
    ) -> torch.Tensor:                    # returns (B, target_dim)
        batch_index = batch_index.to(device=t.device, dtype=torch.long)
        B = int(batch_index.max().item()) + 1
        stats: Sequence[str] = self.readout_kwargs.get('node_pool_stats', ['mean', 'max'])
        pooled_stats: List[torch.Tensor] = []
        if 'mean' in stats:
            pooled_stats.append(scatter(t, batch_index, dim=0, dim_size=B, reduce='mean'))
        if 'max' in stats:
            pooled_stats.append(scatter(t, batch_index, dim=0, dim_size=B, reduce='max'))
        if 'sum' in stats:
            pooled_stats.append(scatter(t, batch_index, dim=0, dim_size=B, reduce='sum'))
        t_agg = torch.cat(pooled_stats, dim=1) if pooled_stats else scatter(t, batch_index, dim=0, dim_size=B, reduce='mean')
        if self.graph_scalar_readout_mlp is None:
            self._lazy_init_graph_scalar_readout_mlp(
                in_dim=t_agg.shape[1],
                out_dim=self.target_dim,
                device=t_agg.device,
            )
        return self.graph_scalar_readout_mlp(t_agg)

    # --------------------------- Aggregation helper ---------------------------
    def _aggregate_nodes(
        self,
        x: torch.Tensor,
        *,
        batch_index: Optional[torch.Tensor],
        agg: Literal['sum', 'mean', 'max'] = 'sum',
    ) -> torch.Tensor:
        if batch_index is None:
            return x.unsqueeze(0)
        batch_index = batch_index.to(device=x.device, dtype=torch.long)
        B = int(batch_index.max().item()) + 1
        out = scatter(
            src=x, 
            index=batch_index,
            dim=0, 
            dim_size=B, 
            reduce=agg,
        )
        return out


