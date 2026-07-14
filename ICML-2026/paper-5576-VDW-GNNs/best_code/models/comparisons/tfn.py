"""
Tensor Field Network model implementatation adapted from Geometric GNN Dojo (MIT license).
https://github.com/chaitjo/geometric-gnn-dojo/blob/main/models/tfn.py

Note the original TFN implementation uses TensorFlow.
https://github.com/tensorfieldnetworks/tensorfieldnetworks

This file adapts GNN Dojo's TFN implementation for datasets (e.g., wind) where we want to:
- **Define neighbors in one geometry** (Earth coordinates), but
- **Perform equivariant geometric message passing in another vector space**
  (wind velocity vectors).

Wind-style setup:
- `edge_index` is constructed from geographic positions stored on `Data.pos`.
- At runtime, the comparison wrapper (`models/comparisons/comparison_module.py`)
  can temporarily swap `batch.pos` to a vector feature (via `pos_input_key`),
  so TFN builds spherical harmonics from wind-difference vectors `(v_i - v_j)`
  while still aggregating messages only over geographic neighbors.
- For wind graphs, `edge_weight` is a normalized adjacency weight (not a distance).
  In the wind-vector-space setup we recompute edge lengths from the swapped `pos`
  (wind differences) for radial features, and optionally concatenate the *geographic*
  `edge_weight` as a separate scalar edge-strength feature.
- The TFN scatter aggregation uses `dim_size=num_nodes` so directed graphs with
  isolated / only-incoming nodes keep consistent tensor shapes.
"""

from typing import Optional, List, Literal
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter
import e3nn
from e3nn import o3
from e3nn.nn import Gate, BatchNorm, Activation  # type: ignore
# Radial kernel reuse (for optional kernel-based radial embedding)
from data_processing.process_pyg_data import get_local_pca_kernel_weights
# from e3nn.util.jit import compile_mode
# from models.mace_modules.irreps_tools import irreps2gate
# from models.mace_modules.blocks import RadialEmbeddingBlock
# from models.layers.tfn_layer import TensorProductConvLayer


# ------------------------------------------------------------------
# Radial function classes
# ------------------------------------------------------------------
class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. 
    Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        return envelope * (x < self.r_max)


class RadialEmbeddingBlock(torch.nn.Module):
    """
    Radial embedding with two modes:
    - 'bessel_cutoff': Bessel basis with polynomial cutoff (legacy path).
    - 'mlp_gates': Original TFN-style learnable radial MLP producing per-l scalar gates
      that modulate spherical harmonics Y_l (mirrors the original TFN paper behavior).

    In 'mlp_gates' mode, this module outputs [n_edges, max_ell + 1] gates. The caller
    is responsible for expanding these gates to match the spherical harmonics basis
    dimensions (repeat each gate for 2l+1 components) and applying them to Y_l.

    Note that also in mlp_gates mode, the edge feature MLP is disabled (weights are per-edge 
    learned bias); distances affect messages only via the learned Y_l gates, as in the 
    original TFN. In 'bessel_cutoff' mode, the edge feature MLP is enabled, and gets fed
    the precomputed Bessel-cutoff basis function values. (The Geometric GNN Dojo version
    uses only the 'bessel_cutoff' strategy.)
    """
    def __init__(
        self, 
        r_max: float, 
        num_bessel: int, 
        num_polynomial_cutoff: int,
        *,
        mode: Literal[
            'bessel_cutoff', 'mlp_gates', 'gaussian', 'cosine_cutoff', 'epanechnikov'
        ] = "mlp_gates",
        max_ell: int = 2,
        radial_mlp_hidden: Optional[List[int]] = None,
        radial_mlp_activation: str = "relu",
        # Kernel gaussian width (only used for gaussian variant)
        radial_kernel_gaussian_eps: Optional[float] = None,
    ):
        super().__init__()
        self.mode = str(mode).lower()

        # Normalize kernel variants into a single 'kernel' implementation
        _kernel_variants = {"gaussian", "cosine_cutoff", "epanechnikov"}

        # Legacy path
        if self.mode == "bessel_cutoff":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
            self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        elif self.mode == "mlp_gates":
            # Original TFN path parameters (Tetris task)
            self.max_ell = int(max_ell)
            if radial_mlp_hidden is None:
                radial_mlp_hidden = [64, 64]

            act_name = str(radial_mlp_activation).lower()
            if act_name in {"swish", "silu"}:
                act_cls = torch.nn.SiLU
            elif act_name == "gelu":
                act_cls = torch.nn.GELU
            else:
                act_cls = torch.nn.ReLU

            layers: List[torch.nn.Module] = []
            in_dim = 1
            for h in radial_mlp_hidden:
                layers.append(torch.nn.Linear(in_dim, int(h)))
                layers.append(act_cls())
                in_dim = int(h)
            # Output one gate per l (0..max_ell)
            layers.append(torch.nn.Linear(in_dim, self.max_ell + 1))
            self.radial_mlp = torch.nn.Sequential(*layers)
        elif self.mode in _kernel_variants:
            # Kernel-based scalar radial feature using get_local_pca_kernel_weights
            # Store parameters; computation happens in forward
            self.r_max = float(r_max)
            # Determine specific kernel type from mode
            if self.mode == 'kernel':
                self.radial_kernel_type = 'gaussian'
            else:
                self.radial_kernel_type = self.mode
            self.radial_kernel_gaussian_eps = radial_kernel_gaussian_eps
        else:
            raise ValueError(f"Invalid radial embedding mode: {self.mode}")

        # Public interface: edge feature dimension seen by conv layers.
        # In 'mlp_gates', we gate Y_l directly, so no explicit edge features are used.
        if self.mode == "bessel_cutoff":
            self.out_dim = num_bessel
        elif self.mode == "mlp_gates":
            self.out_dim = 0
        else:  # kernel modes produce a single scalar feature per edge
            self.out_dim = 1

    def forward(
        self, edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        if self.mode == "bessel_cutoff":
            bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
            cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
            return bessel * cutoff  # [n_edges, n_basis]
        elif self.mode == "mlp_gates":
            # Original TFN path: return per-l scalar gates g_l(r_ij)
            # Caller will expand each gate across its (2l+1) components of Y_l.
            gates = self.radial_mlp(edge_lengths)  # [n_edges, max_ell + 1]
            return gates
        else:
            # Kernel-based scalar feature per edge
            # edge_lengths: [n_edges, 1] -> r: [n_edges]
            r = edge_lengths.squeeze(-1)
            feats = get_local_pca_kernel_weights(
                r=r,
                kernel=self.radial_kernel_type,  # type: ignore[arg-type]
                r_cut=self.r_max,
                gaussian_eps=self.radial_kernel_gaussian_eps,
            )
            return feats.view(-1, 1)


# ------------------------------------------------------------------
# TFN Layer
# ------------------------------------------------------------------
def irreps2gate(irreps):
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = '0e'
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


class TensorProductConvLayer(torch.nn.Module):
    """Tensor Field Network GNN Layer in e3nn

    Implements a Tensor Field Network equivariant GNN layer for higher-order tensors, using e3nn.
    Implementation adapted from: https://github.com/gcorso/DiffDock/

    Paper: Tensor Field Networks, Thomas, Smidt et al.
    """
    def __init__(
        self,
        in_irreps,
        out_irreps,
        sh_irreps,
        edge_feats_dim,
        mlp_dim,
        aggr="add",
        batch_norm=False,
        gate=False,
        self_interaction: bool = True,
    ):
        """
        Args:
            in_irreps: (e3nn.o3.Irreps) Input irreps dimensions
            out_irreps: (e3nn.o3.Irreps) Output irreps dimensions
            sh_irreps: (e3nn.o3.Irreps) Spherical harmonic irreps dimensions
            edge_feats_dim: (int) Edge feature dimensions
            mlp_dim: (int) Hidden dimension of MLP for computing tensor product weights
            aggr: (str) Message passing aggregator
            batch_norm: (bool) Whether to apply equivariant batch norm
            gate: (bool) Whether to apply gated non-linearity
        """
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.edge_feats_dim = edge_feats_dim
        self.aggr = aggr
        self.self_interaction = self_interaction

        if gate:
            # Optionally apply gated non-linearity
            irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
                e3nn.o3.Irreps(out_irreps)
            )
            act_scalars = [torch.nn.functional.silu for _, ir in irreps_scalars]
            act_gates = [torch.sigmoid for _, ir in irreps_gates]
            if irreps_gated.num_irreps == 0:
                self.gate = Activation(out_irreps, acts=[torch.nn.functional.silu])
            else:
                self.gate = Gate(
                    irreps_scalars,
                    act_scalars,  # scalar
                    irreps_gates,
                    act_gates,  # gates (scalars)
                    irreps_gated,  # gated tensors
                )
                # Output irreps for the tensor product must be updated
                self.out_irreps = out_irreps = self.gate.irreps_in
        else:
            self.gate = None

        # Tensor product over edges to construct messages
        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        # Optional self-interaction (learned linear on node features), mirroring
        # original TFN module design (applied every module before nonlinearity).
        self.self_si = o3.Linear(in_irreps, out_irreps) if self_interaction else None

        # MLP used to compute weights of tensor product (optional when edge features are absent)
        self.edge_feats_dim = int(edge_feats_dim)
        if self.edge_feats_dim > 0:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.edge_feats_dim, mlp_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_dim, self.tp.weight_numel),
            )
        else:
            # Fallback: a learnable per-edge weight vector (expanded to all edges)
            self.fc = None
            self.register_parameter(
                name="weight_bias",
                param=torch.nn.Parameter(torch.zeros(self.tp.weight_numel))
            )

        # Optional equivariant batch norm
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_sh, edge_feat):
        src, dst = edge_index
        # Compute messages
        if self.fc is not None and edge_feat is not None:
            weights = self.fc(edge_feat)
        else:
            # Expand learnable bias to per-edge weights
            num_edges = edge_sh.shape[0]
            weights = self.weight_bias.unsqueeze(0).expand(num_edges, -1)
        # Standard edge_index convention: edge_index[0]=source, edge_index[1]=target
        # Messages are computed from source node features and aggregated to target nodes.
        tp = self.tp(node_attr[src], edge_sh, weights)
        # Aggregate messages (ensure output has one row per node, even for isolated nodes)
        out = scatter(tp, dst, dim=0, reduce=self.aggr, dim_size=node_attr.shape[0])
        # Add self-interaction path
        if self.self_si is not None:
            out = out + self.self_si(node_attr)
        # Optionally apply gated non-linearity and/or batch norm
        if self.gate:
            out = self.gate(out)
        if self.batch_norm:
            out = self.batch_norm(out)
        return out

# ------------------------------------------------------------------
# TFNModel
# ------------------------------------------------------------------
class TFNModel(torch.nn.Module):
    """
    Tensor Field Network model from "Tensor Field Networks".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        num_layers: int = 5,
        emb_dim: int = 64,
        hidden_irreps: Optional[e3nn.o3.Irreps] = None,
        mlp_dim: int = 256,
        in_dim: Optional[int] = 1,
        out_dim: int = 1,
        aggr: str = "sum",
        pool_types: Optional[List[str]] = None,
        gate: bool = True,
        batch_norm: bool = False,
        residual: bool = True,
        equivariant_pred: bool = False,
        predict_per_node: bool = False,
        vector_target: bool = False,
        use_bias_if_no_atoms: bool = True,
        # Edge feature wiring for decoupled-geometry datasets (e.g., wind)
        use_edge_weight_as_length: bool = True,
        use_edge_strength_feature: bool = False,
        edge_strength_key: str = "edge_weight",
        # Radial embedding (original TFN) configuration
        radial_mode: str = "mlp_gates",
        radial_mlp_hidden: Optional[List[int]] = None,
        radial_mlp_activation: str = "relu",
        # Equivariant vector head choice
        unbiased_vector_pred_head: bool = True,
        # Kernel radial options
        radial_kernel_type: Optional[Literal['gaussian', 'cosine_cutoff', 'epanechnikov']] = None,
        radial_kernel_gaussian_eps: Optional[float] = None,
    ):
        """
        Parameters:
        - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
        - num_bessel (int): Number of Bessel basis functions (default: 8)
        - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
        - max_ell (int): Maximum degree of spherical harmonics basis functions (default: 2)
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Scalar feature embedding dimension (default: 64)
        - hidden_irreps (Optional[e3nn.o3.Irreps]): Hidden irreps (default: None)
        - mlp_dim (int): Dimension of MLP for computing tensor product weights (default: 256)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - aggr (str): Aggregation method to be used (default: "sum")
        - pool (str): Global pooling method to be used (default: "sum")
        - gate (bool): Whether to use gated equivariant non-linearity (default: True)
        - batch_norm (bool): Whether to use batch normalization (default: False)
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        - predict_per_node (bool): Whether to predict per node (default: False)
        - vector_target (bool): Whether to predict a vector (default: False)
        - use_bias_if_no_atoms (bool): Whether to use a bias if no atoms are provided (default: True)
        - radial_mode (str): Radial embedding mode (default: "mlp_gates")
        - radial_mlp_hidden (Optional[List[int]]): Hidden dimensions of the radial MLP (default: None)
        - radial_mlp_activation (str): Activation function of the radial MLP (default: "relu")
        - unbiased_vector_pred_head (bool): Whether to use an unbiased vector prediction head (default: True);
          if False, the vector prediction head is biased by a ``scale*h_l0 * pos`` to encourage stability
          (at the cost of potentially biasing predictions towards the trivial 'scaled position'.)

        Note:
        - If `hidden_irreps` is None, the irreps for the intermediate features are computed 
          using `emb_dim` and `max_ell`.
        - The `equivariant_pred` parameter determines whether it is an equivariant prediction task.
          If set to True, equivariant prediction will be performed.
        - At present, only one of `gate` and `batch_norm` can be True.
        """
        super().__init__()
        
        self.r_max = r_max
        self.max_ell = max_ell
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.residual = residual
        self.batch_norm = batch_norm
        self.gate = gate
        self.hidden_irreps = hidden_irreps
        self.equivariant_pred = equivariant_pred
        self.predict_per_node = predict_per_node
        self.vector_target = vector_target
        self.use_bias_if_no_atoms = use_bias_if_no_atoms
        self.use_edge_weight_as_length = bool(use_edge_weight_as_length)
        self.use_edge_strength_feature = bool(use_edge_strength_feature)
        self.edge_strength_key = str(edge_strength_key)
        self.radial_mode = str(radial_mode).lower()
        self.unbiased_vector_pred_head = bool(unbiased_vector_pred_head)
        
        # Edge embedding
        # For fidelity with the original paper, use a per-module radial embedding when in
        # MLP-gated mode so each module has its own learnable radial function.
        if self.radial_mode == 'mlp_gates':
            self.radial_embeddings = torch.nn.ModuleList([
                RadialEmbeddingBlock(
                    r_max=r_max,
                    num_bessel=num_bessel,
                    num_polynomial_cutoff=num_polynomial_cutoff,
                    mode=self.radial_mode,
                    max_ell=max_ell,
                    radial_mlp_hidden=radial_mlp_hidden,
                    radial_mlp_activation=radial_mlp_activation,
                ) for _ in range(num_layers)
            ])
            # No explicit radial edge features in mlp_gates mode, but we may still
            # include a separate scalar edge-strength feature.
            _edge_feat_dim = 1 if self.use_edge_strength_feature else 0
        elif self.radial_mode == 'kernel':
            self.radial_embeddings = torch.nn.ModuleList([
                RadialEmbeddingBlock(
                    r_max=r_max,
                    num_bessel=num_bessel,
                    num_polynomial_cutoff=num_polynomial_cutoff,
                    mode=self.radial_mode,
                    max_ell=max_ell,
                    radial_kernel_type=radial_kernel_type,
                    radial_kernel_gaussian_eps=radial_kernel_gaussian_eps,
                ) for _ in range(num_layers)
            ])
            _edge_feat_dim = 2 if self.use_edge_strength_feature else 1
        else:
            self.radial_embeddings = torch.nn.ModuleList([
                RadialEmbeddingBlock(
                    r_max=r_max,
                    num_bessel=num_bessel,
                    num_polynomial_cutoff=num_polynomial_cutoff,
                    mode=self.radial_mode,
                    max_ell=max_ell,
                    radial_mlp_hidden=radial_mlp_hidden,
                    radial_mlp_activation=radial_mlp_activation,
                )
            ])
            _edge_feat_dim = self.radial_embeddings[0].out_dim + (1 if self.use_edge_strength_feature else 0)
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        # Precompute component slices per l for SH basis to expand gates
        self._sh_component_slices = []  # list of (start, end) for l=0..max_ell
        _offset = 0
        for l in range(max_ell + 1):
            dim_l = 2 * l + 1
            self._sh_component_slices.append((_offset, _offset + dim_l))
            _offset += dim_l

        # Embedding lookup for initial node features (optional)
        self.emb_in = None
        if in_dim is not None and int(in_dim) > 0:
            self.emb_in = torch.nn.Embedding(int(in_dim), emb_dim)
        # Learnable bias used when no atoms provided
        if not self.emb_in:
            self.bias_h = torch.nn.Parameter(torch.zeros(emb_dim))

        # Set hidden irreps if none are provided
        if hidden_irreps is None:
            hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
            # Note: This defaults to O(3) equivariant layers
            # It is possible to use SO(3) equivariance by passing the appropriate irreps

        # Precompute output l-block slices for the hidden representation to extract l=0
        # For hidden_irreps = (sh_irreps * emb_dim), each l block is contiguous with size emb_dim*(2l+1)
        self._hidden_l_slices = []  # list of (start, end) for l=0..max_ell
        _hoff = 0
        for l in range(max_ell + 1):
            dim_block = emb_dim * (2 * l + 1)
            self._hidden_l_slices.append((_hoff, _hoff + dim_block))
            _hoff += dim_block

        self.convs = torch.nn.ModuleList()
        # First conv layer: scalar only -> tensor
        self.convs.append(
            TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f'{emb_dim}x0e'),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=_edge_feat_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
                self_interaction=True,
            )
        )
        # Intermediate conv layers: tensor -> tensor
        for _ in range(num_layers - 1):
            conv = TensorProductConvLayer(
                in_irreps=hidden_irreps,
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=_edge_feat_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
                self_interaction=True,
            )
            self.convs.append(conv)

        # Pooling configuration (invariant paths can use multiple poolings)
        if pool_types is None:
            pool_types = ["sum"]
        if isinstance(pool_types, str):
            pool_types = [pool_types]
        self.pool_types = [str(p).lower() for p in pool_types]
        self._pool_map = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }
        # Primary pool for vector outputs (single pooling)
        primary = self.pool_types[0]
        self.pool = self._pool_map.get(primary, global_add_pool)

        # Heads mirroring EGNN behavior
        if self.equivariant_pred:
            # Construct heads only when needed to avoid unused parameters.
            self.scale_head = None
            # Need scale-based head only when using the robust path
            _need_scale = (not self.unbiased_vector_pred_head)
            if _need_scale:
                # Scalar scale s(h_l0) for constructing equivariant vectors from positions
                self.scale_head = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, 1),
                )
            # Equivariant vector head (recommended): map hidden irreps -> one vector irrep (1o)
            # Needed for vector targets or when using unbiased head for scalar tasks
            self.vector_head = o3.Linear(hidden_irreps, o3.Irreps('1o')) \
            if (self.vector_target or self.unbiased_vector_pred_head) else None
            if not self.vector_target:
                if self.predict_per_node:
                    self.inv_pred_node = torch.nn.Sequential(
                        torch.nn.Linear(emb_dim + 1, emb_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(emb_dim, out_dim),
                    )
                else:
                    num_pools_local = len(self.pool_types)
                    graph_in_dim = emb_dim * num_pools_local + 2 * num_pools_local
                    self.inv_pred_graph = torch.nn.Sequential(
                        torch.nn.Linear(graph_in_dim, emb_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(emb_dim, out_dim),
                    )
            self.pred = None
        else:
            if self.predict_per_node:
                self.pred_scalar_node = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, out_dim)
                )
            else:
                num_pools_local = len(self.pool_types)
                self.pred_scalar_graph = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim * num_pools_local, emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, out_dim)
                )
    
    def _select_scalar_features(self, h: torch.Tensor) -> torch.Tensor:
        # Robustly select the l=0 block (size = emb_dim), mirroring original TFN invariant path
        start, end = self._hidden_l_slices[0]
        return h[:, start:end]

    def _expand_l_gates_to_components(self, gates_l: torch.Tensor) -> torch.Tensor:
        """
        Expand per-l gates [n_edges, L+1] to per-component gates [n_edges, sum_{l}(2l+1)].
        Each gate for order l is repeated across its (2l+1) spherical harmonic components.
        """
        n_edges = gates_l.size(0)
        expanded = []
        for l, (s, e) in enumerate(self._sh_component_slices):
            num_comp = e - s
            g = gates_l[:, l:l+1].expand(n_edges, num_comp)
            expanded.append(g)
        return torch.cat(expanded, dim=-1)

    def _concat_pooled_features(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        pooled_list = []
        for p in self.pool_types:
            if p in self._pool_map:
                pooled = self._pool_map[p](x, batch_index)
            else:
                raise ValueError(f"Unsupported pooling type '{p}'")
            pooled_list.append(pooled)
        return torch.cat(pooled_list, dim=-1)

    def forward(self, batch):
        # Node feature initialization
        if hasattr(batch, "atoms") and (batch.atoms is not None) and (self.emb_in is not None):
            h = self.emb_in(batch.atoms)
        else:
            if not self.use_bias_if_no_atoms:
                raise ValueError(
                    "TFNModel expected `batch.atoms` but it was missing/None, and `use_bias_if_no_atoms` is False."
                )
            num_nodes = batch.pos.size(0)
            h = self.bias_h.unsqueeze(0).expand(num_nodes, -1)

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        # Edge lengths: optionally use `edge_weight` as a precomputed *distance* when it truly
        # represents a distance. For wind graphs, we disable this (use_edge_weight_as_length=False)
        # because edge_weight is a normalized adjacency weight.
        lengths = None
        if self.use_edge_weight_as_length:
            if hasattr(batch, 'edge_weight') and getattr(batch, 'edge_weight') is not None:
                lengths = getattr(batch, 'edge_weight').view(-1, 1)
        if lengths is None:
            lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        # Optional additional scalar edge-strength feature (e.g., geographic weights).
        edge_strength = None
        if self.use_edge_strength_feature:
            if not hasattr(batch, self.edge_strength_key):
                raise ValueError(
                    f"TFNModel expected edge strength key '{self.edge_strength_key}' but it was missing on the batch."
                )
            ew = getattr(batch, self.edge_strength_key)
            if not torch.is_tensor(ew):
                raise ValueError(f"TFNModel expected '{self.edge_strength_key}' to be a torch.Tensor.")
            if ew.dim() == 1:
                ew = ew.unsqueeze(1)
            elif ew.dim() != 2 or ew.shape[1] != 1:
                raise ValueError("TFNModel edge strength must have shape (e,) or (e,1).")
            if ew.shape[0] != lengths.shape[0]:
                raise ValueError("TFNModel edge strength must align with number of edges.")
            edge_strength = ew.to(device=lengths.device, dtype=lengths.dtype)

        edge_sh_base = self.spherical_harmonics(vectors)

        for layer_idx, conv in enumerate(self.convs):
            if self.radial_mode == 'mlp_gates':
                # Layer-specific gates
                if lengths is not None:
                    gates_l = self.radial_embeddings[layer_idx](lengths)  # [n_edges, L+1]
                    gates_comp = self._expand_l_gates_to_components(gates_l)
                    edge_sh = edge_sh_base * gates_comp
                else:
                    edge_sh = edge_sh_base
                edge_feats = edge_strength if self.use_edge_strength_feature else None
            else:
                # Single shared bessel-cutoff features
                edge_sh = edge_sh_base
                emb_block = self.radial_embeddings[0]
                edge_feats_base = emb_block(lengths) if lengths is not None else None
                if self.use_edge_strength_feature:
                    if edge_feats_base is None:
                        edge_feats = edge_strength
                    else:
                        edge_feats = torch.cat([edge_feats_base, edge_strength], dim=-1)
                else:
                    edge_feats = edge_feats_base

            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_sh, edge_feats)

            # Update node features
            h = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1])) if self.residual else h_update

        # Decide prediction granularity
        if self.predict_per_node:
            if not self.equivariant_pred:
                h_scalar = self._select_scalar_features(h)
                return self.pred_scalar_node(h_scalar)
            else:
                if self.vector_target:
                    # Equivariant node-level vector prediction
                    if self.unbiased_vector_pred_head:
                        vec = self.vector_head(h) if self.vector_head is not None else batch.pos * 0.0
                    else:
                        h_scalar = self._select_scalar_features(h)
                        scale = self.scale_head(h_scalar) if self.scale_head is not None else 0.0
                        base_vec = self.vector_head(h) if self.vector_head is not None else 0.0
                        vec = base_vec + scale * batch.pos
                    return vec
                else:
                    # Invariant per-node: build an equivariant vector then use its norm
                    if self.unbiased_vector_pred_head:
                        vec = self.vector_head(h) if self.vector_head is not None else batch.pos * 0.0
                        h_scalar = self._select_scalar_features(h)
                    else:
                        h_scalar = self._select_scalar_features(h)
                        scale = self.scale_head(h_scalar) if self.scale_head is not None else 0.0  # (n, 1)
                        vec = scale * batch.pos  # (n, pos_dim)
                    vec_norm = torch.norm(vec, dim=-1, keepdim=True)
                    inv_feat = torch.cat([h_scalar, vec_norm], dim=-1)
                    return self.inv_pred_node(inv_feat)
        else:
            if not self.equivariant_pred:
                h_scalar = self._select_scalar_features(h)
                pooled_h_multi = self._concat_pooled_features(h_scalar, batch.batch)
                return self.pred_scalar_graph(pooled_h_multi)
            else:
                h_scalar = self._select_scalar_features(h)
                if self.vector_target:
                    # Graph-level equivariant vector prediction
                    if self.unbiased_vector_pred_head:
                        vec = self.vector_head(h) if self.vector_head is not None else batch.pos * 0.0
                    else:
                        scale = self.scale_head(h_scalar) if self.scale_head is not None else 0.0
                        base_vec = self.vector_head(h) if self.vector_head is not None else 0.0
                        vec = base_vec + scale * batch.pos
                    pooled_vec = self.pool(vec, batch.batch)
                    return pooled_vec
                # Graph-level invariant: build an equivariant vector then use invariant stats
                if self.unbiased_vector_pred_head:
                    vec = self.vector_head(h) if self.vector_head is not None else batch.pos * 0.0
                else:
                    scale = self.scale_head(h_scalar) if self.scale_head is not None else 0.0
                    vec = scale * batch.pos
                pooled_h_multi = self._concat_pooled_features(h_scalar, batch.batch)
                vec_norm = torch.norm(vec, dim=-1, keepdim=True)
                vec_sqnorm = (vec * vec).sum(dim=-1, keepdim=True)
                pooled_vec_norm_multi = self._concat_pooled_features(vec_norm, batch.batch)
                pooled_vec_sqnorm_multi = self._concat_pooled_features(vec_sqnorm, batch.batch)
                inv_feat = torch.cat([pooled_h_multi, pooled_vec_norm_multi, pooled_vec_sqnorm_multi], dim=-1)
                return self.inv_pred_graph(inv_feat)