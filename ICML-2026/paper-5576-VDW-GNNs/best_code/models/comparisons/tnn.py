"""
Tangent Bundle NN wrapper for the training pipeline.

Example runs:
- Ellipsoids (diameter or normals), config-driven:
    python3 scripts/python/main_training.py \\
        --config config/yaml_files/ellipsoids/tnn_diameter.yaml
    # or node-vector task:
    python3 scripts/python/main_training.py \\
        --config config/yaml_files/ellipsoids/tnn_normals.yaml

- Macaque inductive (k-fold CV per day via helper launcher):
    python3 scripts/python/run_macaque_multiday_cv.py \\
        --model vdw \\
        --days 0-2 \\
        --root_dir /path/to/project/root \\
        --model_config config/yaml_files/macaque/tnn.yaml
  (Wraps main_training; TNN uses ambient embeddings + kNN imputation + SVM probe.)

Note: scripts/python/run_wind_tnn.py is a standalone runner for the wind dataset.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from typing import List, Tuple

from models.base_module import BaseModule
from models.nn_utilities import ProjectionMLP
from models.comparisons.tangent_bundle_nn import (
    TNN,
    set_hparams,
    FEATURES,
    SIGMA,
    READOUT_SIGMA,
    KAPPA,
    LOSS_FUNCTION,
    WEIGHT_DECAY,
    LEARN_RATE,
    get_laplacians,
    EPSILON,
    EPSILON_PCA,
    GAMMA,
    READOUT_MLP_HIDDEN_DIMS,
    USE_READOUT_MLP,
)


def _reshape_sheaf_signal(signal: torch.Tensor, d_hat: int) -> torch.Tensor:
    """
    Reshape a (n*d_hat, c) stacked sheaf signal to (n, d_hat, c).
    """
    n_d_hat = signal.shape[0]
    if n_d_hat % d_hat != 0:
        raise RuntimeError(
            f"Cannot reshape sheaf signal of length {n_d_hat} with d_hat={d_hat}."
        )
    node_count = n_d_hat // d_hat
    return signal.view(node_count, d_hat, -1)


class _ScalarHead(nn.Module):
    """
    Pooled MLP head for graph-level regression.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        pool_types: Tuple[str, ...],
        activation: str = "silu",
        dropout_p: float | None = None,
    ) -> None:
        super().__init__()
        self.pool_types_list = list(pool_types)
        act_cls = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }.get(activation.lower(), nn.SiLU)
        layers: List[nn.Module] = []
        dims = [in_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
                if dropout_p is not None:
                    layers.append(nn.Dropout(dropout_p))
        self.mlp_head = nn.Sequential(*layers)
        self._pool_map = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }

    def forward(
        self,
        h_node: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_parts: List[torch.Tensor] = []
        for pt in self.pool_types_list:
            if pt == "moments":
                # moments not used in ellipsoids configs; skip to reduce complexity
                pooled_m = h_node.mean(dim=0, keepdim=True)
                pooled_parts.append(pooled_m)
            else:
                if batch_index is None:
                    pooled = self._pool_map[pt](h_node, batch=torch.zeros(h_node.size(0), dtype=torch.long, device=h_node.device))
                else:
                    pooled = self._pool_map[pt](h_node, batch=batch_index)
                pooled_parts.append(pooled)
        pooled_concat = pooled_parts[0] if len(pooled_parts) == 1 else torch.cat(pooled_parts, dim=-1)
        return self.mlp_head(pooled_concat)


class TNNComparisonModel(BaseModule):
    """
    Wrap the tangent bundle network for use in the training pipeline.

    Supports:
      - graph-level scalar regression via pooled MLP head
      - node-level vector regression (returns ambient vectors)
    """

    def __init__(
        self,
        *,
        base_module_kwargs: dict,
        tnn_operator: torch.Tensor,
        d_hat: int,
        vector_feat_key: str,
        pool_types: Tuple[str, ...],
        mlp_hidden_dim: List[int],
        mlp_nonlin: str,
        mlp_dropout_p: float | None,
        target_task: str,
    ) -> None:
        super().__init__(**base_module_kwargs)
        self.vector_feat_key = vector_feat_key
        self.d_hat = int(d_hat)
        self.target_task = target_task.lower()
        self._device_hint = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Defaults for macaque inductive probes
        self.eval_mode = "weighted_average"
        self.holdout_k_probe = 10

        features_out = FEATURES.copy()
        features_out[-1] = 1  # single channel per sheaf component

        if tnn_operator.shape[0] % self.d_hat != 0:
            raise RuntimeError(
                f"TNN operator rows {tnn_operator.shape[0]} not divisible by d_hat={self.d_hat}"
            )
        node_count = int(tnn_operator.shape[0] // self.d_hat)

        hparams = set_hparams(
            n=node_count,
            L=[tnn_operator],
            in_features=self.d_hat,
            features=features_out,
            lr=float(LEARN_RATE),
            weight_decay=float(WEIGHT_DECAY),
            sigma=SIGMA,
            readout_sigma=READOUT_SIGMA,
            kappa=KAPPA,
            loss_function=LOSS_FUNCTION,
            device=self._device_hint,
        )
        self.tnn = TNN(**hparams).to(self._device_hint)

        if "graph" in self.target_task:
            # Pool over nodes then predict scalar
            pooled_in_dim = self.d_hat * len(pool_types)
            self.scalar_head = _ScalarHead(
                in_dim=pooled_in_dim,
                hidden_dims=mlp_hidden_dim,
                pool_types=pool_types,
                activation=mlp_nonlin,
                dropout_p=mlp_dropout_p,
            )
        else:
            self.scalar_head = None

        self.readout_mlp = None
        if self.scalar_head is None and USE_READOUT_MLP:
            self.readout_mlp = ProjectionMLP(
                in_dim=self.d_hat,
                hidden_dim=READOUT_MLP_HIDDEN_DIMS,
                embedding_dim=self.target_dim,
                activation=nn.SiLU,
                use_batch_norm=False,
                dropout_p=None,
                residual_style=False,
            ).to(self._device_hint)

    def forward(self, batch: Data) -> dict:
        x = getattr(batch, self.vector_feat_key)
        if not hasattr(batch, "tnn_operator") or not hasattr(batch, "tnn_O"):
            self._compute_and_cache_tnn(batch)
        operator = batch.tnn_operator.to(x.device)
        d_hat = int(getattr(batch, "tnn_d_hat"))
        if d_hat != self.d_hat:
            raise RuntimeError(f"d_hat mismatch: expected {self.d_hat}, got {d_hat}")
        if operator.shape[0] % d_hat != 0:
            raise RuntimeError(
                f"Operator rows {operator.shape[0]} not divisible by d_hat={d_hat}"
            )

        # Update per-graph operator (batch_size=1); keep dtype/device in sync
        for layer in self.tnn.tnn:
            layer.L = operator

        sheaf_signal = self._project_to_sheaf(x, batch.tnn_O.to(x.device))
        preds_sheaf = self.tnn(sheaf_signal)
        preds_sheaf_nodes = _reshape_sheaf_signal(preds_sheaf, self.d_hat).squeeze(-1)

        if self.scalar_head is not None:
            graph_out = self.scalar_head(preds_sheaf_nodes, getattr(batch, "batch", None))
            return {"preds": graph_out}

        if self.readout_mlp is not None:
            node_out = self.readout_mlp(preds_sheaf_nodes)
            return {"preds": node_out}

        # Node-level vector: back-project to ambient
        preds_ambient = self._project_to_ambient(
            preds_sheaf_nodes,
            batch.tnn_O.to(x.device),
        )
        return {"preds": preds_ambient}

    @torch.no_grad()
    def compute_embeddings(self, data: Data) -> torch.Tensor:
        """
        Compute per-node ambient embeddings for inductive probes (e.g., SVM).
        """
        if not hasattr(data, "tnn_operator") or not hasattr(data, "tnn_O"):
            self._compute_and_cache_tnn(data)
        x = getattr(data, self.vector_feat_key)
        operator = data.tnn_operator.to(x.device)
        d_hat = int(getattr(data, "tnn_d_hat"))
        if d_hat != self.d_hat:
            raise RuntimeError(f"d_hat mismatch: expected {self.d_hat}, got {d_hat}")
        if operator.shape[0] % d_hat != 0:
            raise RuntimeError(
                f"Operator rows {operator.shape[0]} not divisible by d_hat={d_hat}"
            )
        for layer in self.tnn.tnn:
            layer.L = operator
        sheaf_signal = self._project_to_sheaf(x, data.tnn_O.to(x.device))
        preds_sheaf = self.tnn(sheaf_signal)
        preds_sheaf_nodes = _reshape_sheaf_signal(preds_sheaf, self.d_hat).squeeze(-1)
        return self._project_to_ambient(
            preds_sheaf_nodes,
            data.tnn_O.to(x.device),
        )

    def _compute_and_cache_tnn(self, data: Data) -> None:
        """
        Best-effort sheaf Laplacian/O-frame computation when cache is missing.
        """
        if hasattr(data, "pos"):
            coords = data.pos
        elif hasattr(data, self.vector_feat_key):
            coords = getattr(data, self.vector_feat_key)
        else:
            raise RuntimeError(
                f"TNN requires either 'pos' or '{self.vector_feat_key}' on Data."
            )
        coords_np = coords.detach().cpu().numpy()
        Delta_n_numpy, _, _, O_i_collection, d_hat, _ = get_laplacians(
            coords_np,
            epsilon=EPSILON,
            epsilon_pca=EPSILON_PCA,
            gamma_svd=GAMMA,
            tnn_or_gnn="tnn",
        )
        operator = torch.from_numpy(Delta_n_numpy).to(torch.float32)
        O_tensor = torch.stack([torch.from_numpy(o).to(torch.float32) for o in O_i_collection], dim=0)
        data.tnn_operator = operator
        data.tnn_O = O_tensor
        data.tnn_d_hat = int(d_hat)
        data.tnn_node_count = int(coords_np.shape[0])

    def _project_to_sheaf(
        self,
        feat: torch.Tensor,
        O: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project node features into stacked sheaf basis (n*d_hat, 1).
        If features are already in tangent coordinates (dim == d_hat), bypass O.
        """
        if feat.dim() != 2:
            raise RuntimeError(f"Expected 2D node features, got {feat.shape}")
        ambient_dim = int(O.shape[1])
        d_hat = int(O.shape[2])
        if feat.shape[1] == d_hat:
            proj = feat
        elif feat.shape[1] == ambient_dim:
            proj = torch.einsum("nij,nj->ni", O.transpose(1, 2), feat)
        else:
            raise RuntimeError(
                f"Feature dim {feat.shape[1]} must match d_hat={d_hat} "
                f"or ambient_dim={ambient_dim}."
            )
        return proj.reshape(-1, 1)

    def _project_to_ambient(
        self,
        sheaf_nodes: torch.Tensor,
        O: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map sheaf coordinates back to ambient vectors.
        """
        return torch.einsum("nij,ni->nj", O, sheaf_nodes)

