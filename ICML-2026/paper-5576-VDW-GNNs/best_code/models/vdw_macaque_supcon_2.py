"""
Simplified inductive VDW variant for macaque reaching with SupCon loss.

- Operates directly on the train-day spatial graph (train nodes only)
- Applies diffusion-wavelet scattering to neural velocity vectors
- Optional VectorBatchNorm over scattering coefficients
- Projection MLP maps flattened coefficients -> node embeddings
- SupConLoss optimizes node embeddings using condition labels
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Callable, Tuple, Sequence, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor

from models.base_module import BaseModule
from models.custom_loss_fns import SupConLoss
from models.nn_utilities import ProjectionMLP
from geo_scat import (
    LearnableMahalanobisTopK,
    LearnableP,
    MultiLearnableP,
    MultiViewScatter,
    VectorBatchNorm,
    build_learnable_diffusion_ops,
    vector_multiorder_scatter,
)


class VDW_macaque_supcon_2(BaseModule):
    """
    Node-level SupCon model that learns embeddings on the training spatial graph.
    """

    def __init__(
        self,
        *,
        vector_feat_key: str,
        scalar_feature_key: Optional[str],
        target_key: str,
        diffusion_kwargs: Dict[str, Any],
        num_scattering_layers: int = 1,
        use_vector_batch_norm: bool = True,
        projection_hidden_dim: int = 256,
        projection_embedding_dim: int = 3,
        projection_activation: type[nn.Module] = nn.ReLU,
        projection_residual_style: bool = False,
        projection_dropout_p: float | None = None,
        temperature: float = 0.1,
        supcon_learnable_temperature: bool = True,
        num_classes: int = 7,
        use_distill_inference: bool = False,
        distill_hidden_dim: Sequence[int] | int = (256, 256, 256),
        distill_activation: type[nn.Module] = nn.ReLU,
        verbosity: int = 0,
        device: Optional[torch.device] = None,
        supcon_sampling_max_nodes: Optional[int] = None,
        supcon_neighbor_k: int = 1,
        pos_pairs_per_anchor: Optional[int] = None,
        neg_topk_per_positive: int = 7,
        random_negatives_per_anchor: int = 8,
        learnable_P: bool = False,
        learnable_p_kwargs: Optional[Dict[str, Any]] = None,
        learnable_p_num_views: int = 1,
        learnable_p_view_aggregation: Literal['concat', 'mean', 'sum', 'first'] = 'concat',
        learnable_p_softmax_temps: Optional[Sequence[float]] = None,
        learnable_p_laziness_inits: Optional[Sequence[float]] = None,
        learnable_p_fix_alpha: bool = False,
        learnable_p_use_softmax: bool = False,
        learnable_p_use_attention: bool = False,
        learnable_p_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_learnable_topk: bool = False,
        learnable_topk_k: Optional[int] = None,
        learnable_topk_proj_dim: Optional[int] = None,
        learnable_topk_temperature: float = 1.0,
        learnable_topk_eps: float = 1e-8,
    ) -> None:
        super().__init__(
            task="node_multi_classification",
            loss_fn=None,
            target_name=target_key,
            metrics_kwargs={
                "num_outputs": projection_embedding_dim,
                "num_classes": num_classes,
            },
            device=device,
            has_lazy_parameter_initialization=True,
            verbosity=verbosity,
        )
        self.vector_feat_key = vector_feat_key
        self.scalar_feat_key = scalar_feature_key
        self.target_key = target_key
        self.diffusion_kwargs = diffusion_kwargs
        self.num_scattering_layers = num_scattering_layers
        self.use_vector_batch_norm = use_vector_batch_norm
        self.temperature = temperature
        self.supcon_learnable_temperature = bool(supcon_learnable_temperature)
        self.supcon_sampling_max_nodes = supcon_sampling_max_nodes
        self.supcon_neighbor_k = supcon_neighbor_k
        self.pos_pairs_per_anchor = pos_pairs_per_anchor
        self.neg_topk_per_positive = neg_topk_per_positive
        self.random_negatives_per_anchor = random_negatives_per_anchor
        self.num_classes = int(num_classes)
        self.learnable_P = bool(learnable_P)
        self.learnable_p_kwargs = learnable_p_kwargs or {}
        self.learnable_p_num_views = max(1, int(learnable_p_num_views))
        self.learnable_p_view_aggregation = learnable_p_view_aggregation
        self.learnable_p_fix_alpha = bool(learnable_p_fix_alpha)
        self.learnable_p_use_softmax = bool(learnable_p_use_softmax)
        self.learnable_p_use_attention = bool(learnable_p_use_attention)
        self.learnable_p_attention_kwargs = dict(learnable_p_attention_kwargs or {})
        self.multi_view_scatter = MultiViewScatter(
            combine_mode=self.learnable_p_view_aggregation
        )
        self._last_learnable_diffusion_ops: list | None = None
        if self.learnable_P and self.scalar_feat_key is None:
            raise ValueError(
                "Learnable P requires 'scalar_feature_key' to be set in the configuration."
            )

        self.use_learnable_topk = bool(use_learnable_topk)
        self.learnable_topk_k = (
            int(learnable_topk_k)
            if learnable_topk_k is not None
            else int(self.diffusion_kwargs.get("scattering_k", 0))
        )
        if self.use_learnable_topk and self.learnable_topk_k <= 0:
            raise ValueError(
                "Learnable Mahalanobis top-k requires a positive top-k value "
                "(provide learnable_topk_k or ensure diffusion_kwargs includes 'scattering_k')."
            )
        self.learnable_topk_proj_dim = learnable_topk_proj_dim
        self.learnable_topk_temperature = float(learnable_topk_temperature)
        self.learnable_topk_eps = float(learnable_topk_eps)
        self.learnable_topk_layer: Optional[LearnableMahalanobisTopK] = None

        self.vector_bn: Optional[VectorBatchNorm] = None
        self.projection_mlp: Optional[ProjectionMLP] = None
        self.projection_hidden_dim = projection_hidden_dim
        self.projection_embedding_dim = projection_embedding_dim
        self.projection_activation = projection_activation
        self.projection_residual_style = bool(projection_residual_style)
        self.projection_dropout_p = projection_dropout_p
        self.learnable_p_layer: Optional[nn.Module]
        self.use_distill_inference = bool(use_distill_inference)
        if isinstance(distill_hidden_dim, (list, tuple)):
            self.distill_hidden_dim = [int(h) for h in distill_hidden_dim]
        else:
            self.distill_hidden_dim = [int(distill_hidden_dim)]
        self.distill_activation = distill_activation
        self.distill_output_dim = int(num_classes)
        self.distill_mlp: Optional[ProjectionMLP] = None
        self.distill_loss_fn = nn.MSELoss()
        self._distill_warned_shape_mismatch = False
        
        if self.learnable_P:
            laziness_defaults = learnable_p_laziness_inits or (0.5,)
            temps_defaults = learnable_p_softmax_temps or (1.0,)
            if self.learnable_p_num_views > 1:
                self.learnable_p_layer = MultiLearnableP(
                    num_views=self.learnable_p_num_views,
                    learn_laziness=not self.learnable_p_fix_alpha,
                    edge_mlp_kwargs=self.learnable_p_kwargs,
                    laziness_inits=laziness_defaults,
                    softmax_temps=temps_defaults,
                    use_softmax=self.learnable_p_use_softmax,
                    use_attention=self.learnable_p_use_attention,
                    attention_kwargs=self.learnable_p_attention_kwargs,
                )
            else:
                self.learnable_p_layer = LearnableP(
                    learn_laziness=not self.learnable_p_fix_alpha,
                    edge_mlp_kwargs=self.learnable_p_kwargs,
                    laziness_init=float(laziness_defaults[0]),
                    use_softmax=self.learnable_p_use_softmax,
                    softmax_temp=float(temps_defaults[0]),
                    use_attention=self.learnable_p_use_attention,
                    attention_kwargs=self.learnable_p_attention_kwargs,
                )
        else:
            self.learnable_p_layer = None

        self.supcon_loss = SupConLoss(
            temperature=temperature,
            learnable_temperature=self.supcon_learnable_temperature,
        )
        self.supcon2_context: Dict[str, Any] | None = None
        self.supcon2_accelerator = None
        self.supcon2_eval_fn: Optional[
            Callable[..., tuple]
        ] = None

    def set_supcon2_context(
        self,
        context: Optional[Dict[str, Any]],
        accelerator: Optional[Any] = None,
        eval_fn: Optional[Callable[..., tuple]] = None,
    ) -> None:
        """
        Attach or clear fold-specific context for SVM evaluation during training.
        """
        self.supcon2_context = context
        self.supcon2_accelerator = accelerator
        self.supcon2_eval_fn = eval_fn

    # ---------------------- helper utilities ----------------------
    def _compose_learnable_p_edge_features(
        self,
        data: Data,
        device: torch.device,
        vector_features: torch.Tensor,
        *,
        edge_index_override: Optional[torch.Tensor] = None,
        edge_weight_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Validate input shapes
        if self.scalar_feat_key is None:
            raise ValueError("scalar_feat_key must be set to build learnable P features.")
        edge_index = edge_index_override
        if edge_index is None:
            edge_index = getattr(data, "diffusion_edge_index", None)
        if edge_index is None:
            edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("Diffusion edge_index is required for learnable P.")

        # Get edge weights
        edge_weight = edge_weight_override
        if edge_weight is None:
            edge_weight = getattr(
                data,
                "diffusion_edge_weight",
                getattr(data, "edge_weight", None),
            )
        if edge_weight is None:
            raise ValueError("Diffusion edge weights are required for learnable P.")

        # Get scalar features
        scalar_feats = getattr(data, self.scalar_feat_key, None)
        if scalar_feats is None:
            raise ValueError(
                f"Scalar features '{self.scalar_feat_key}' missing on Data for learnable P."
            )

        # Move inputs to device
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.view(-1, 1).to(device)
        scalar_feats = scalar_feats.to(device)  # (N, 1)
        if scalar_feats.dim() == 1:
            scalar_feats = scalar_feats.unsqueeze(-1)  # (N, 1)

        # Compute invariant vector features (norm, dot)
        vec_src = vector_features[edge_index[0]]
        vec_dst = vector_features[edge_index[1]]
        vel_l1 = torch.norm(vec_src - vec_dst, p=1, dim=-1, keepdim=True)
        vel_dot = (vec_src * vec_dst).sum(dim=-1, keepdim=True)

        # Get scalar features of source and destination nodes
        scalar_src = scalar_feats[edge_index[0]]
        scalar_dst = scalar_feats[edge_index[1]]

        # Concatenate features
        edge_features = torch.cat(
            (edge_weight, vel_l1, vel_dot, scalar_src, scalar_dst),
            dim=-1,
        )
        return edge_index, edge_features

    def _build_learnable_diffusion(
        self,
        data: Data,
        vector_features: torch.Tensor,
        *,
        edge_index_override: Optional[torch.Tensor] = None,
        edge_weight_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.learnable_p_layer is None:
            raise RuntimeError("LearnableP layer has not been initialized.")
        device = vector_features.device
        edge_index, edge_features = self._compose_learnable_p_edge_features(
            data,
            device,
            vector_features,
            edge_index_override=edge_index_override,
            edge_weight_override=edge_weight_override,
        )
        diffusion_ops = build_learnable_diffusion_ops(
            learnable_module=self.learnable_p_layer,
            data=data,
            edge_index=edge_index,
            edge_features=edge_features,
            vector_dim=vector_features.shape[1],
            device=device,
        )
        if not isinstance(diffusion_ops, list):
            diffusion_ops = [diffusion_ops]

        if (
            self.learnable_P
            and self.learnable_p_num_views > 1
        ):
            self._last_learnable_diffusion_ops = diffusion_ops
        return [
            op.to(device=device, dtype=vector_features.dtype) \
            for op in diffusion_ops
        ]

    def _lazy_init_layers(
        self,
        *,
        num_wavelets: int,
        vector_dim: int,
        device: torch.device,
    ) -> None:
        if self.use_vector_batch_norm and self.vector_bn is None:
            self.vector_bn = VectorBatchNorm(num_wavelets=num_wavelets).to(device)
        flattened_dim = vector_dim * num_wavelets
        if self.projection_mlp is None:
            self.projection_mlp = ProjectionMLP(
                in_dim=flattened_dim,
                hidden_dim=self.projection_hidden_dim,
                embedding_dim=self.projection_embedding_dim,
                activation=self.projection_activation,
                residual_style=self.projection_residual_style,
                dropout_p=self.projection_dropout_p,
            ).to(device)

    def _lazy_init_distill_mlp(
        self,
        *,
        input_dim: int,
        device: torch.device,
    ) -> None:
        if not self.use_distill_inference:
            return
        if self.distill_mlp is None:
            self.distill_mlp = ProjectionMLP(
                in_dim=input_dim,
                hidden_dim=self.distill_hidden_dim,
                embedding_dim=self.distill_output_dim,
                activation=self.distill_activation,
                use_batch_norm=True,
                dropout_p=None,
                residual_style=False,
            ).to(device)

    def _compose_distill_features(
        self,
        *,
        data: Data,
        vector_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Composes features for knowledge distillation from vector and scalar features.
        """
        device = vector_features.device
        scalar_feats = None
        if self.scalar_feat_key is not None \
        and hasattr(data, self.scalar_feat_key):
            scalar_feats = getattr(data, self.scalar_feat_key)
            if scalar_feats is not None:
                scalar_feats = scalar_feats.to(device)
                if scalar_feats.dim() == 1:
                    scalar_feats = scalar_feats.unsqueeze(-1)
                scalar_dim = int(scalar_feats.shape[-1])
            distill_inputs = torch.cat((vector_features, scalar_feats), dim=-1)
        else:
            distill_inputs = vector_features
        self._lazy_init_distill_mlp(
            input_dim=int(distill_inputs.shape[-1]),
            device=device,
        )
        return distill_inputs

    # def _match_distill_target_dim(
    #     self,
    #     *,
    #     target: torch.Tensor,
    #     desired_dim: int,
    # ) -> torch.Tensor:
    #     current_dim = int(target.shape[-1])
    #     if current_dim == desired_dim:
    #         return target
    #     if current_dim > desired_dim:
    #         return target[..., :desired_dim]
    #     pad = desired_dim - current_dim
    #     pad_tensor = target.new_zeros(target.shape[:-1] + (pad,))
    #     return torch.cat((target, pad_tensor), dim=-1)

    def _prepare_batch(self, batch: Batch | Data) -> Data:
        if isinstance(batch, Batch):
            data = batch
        else:
            data = batch
        if self.learnable_P:
            for attr in ("Q_unwt", "Q_block_pairs", "Q_block_edge_ids"):
                if not hasattr(data, attr):
                    raise ValueError(
                        f"Learnable-P mode requires '{attr}' on the Batch/Data object."
                    )
        elif not hasattr(data, "Q"):
            raise ValueError(
                "VDW-GNN SupCon v2 model requires 'Q' diffusion operator attached to the Batch."
            )
        return data

    def _lazy_init_learnable_topk_layer(
        self,
        *,
        scalar_dim: int,
        device: torch.device,
    ) -> None:
        if self.learnable_topk_layer is None:
            self.learnable_topk_layer = LearnableMahalanobisTopK(
                in_dim=scalar_dim,
                proj_dim=self.learnable_topk_proj_dim,
                temperature=self.learnable_topk_temperature,
                eps=self.learnable_topk_eps,
            ).to(device)

    # ---------------------- BaseModule overrides ----------------------
    def forward(
        self,
        batch: Batch | Data,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward architecture summary:
        1. Learnable P builds Q, if enabled
        2. Use Q for vector scattering
        3. Vector batch norm 
        4. Projection MLP -> embeddings
        5. SupCon loss
        """
        data = self._prepare_batch(batch)
        x_v = getattr(data, self.vector_feat_key)  # (N, d)

        # Build or grab the diffusion operator
        if self.learnable_P:
            edge_index_override = None
            edge_weight_override = None
            if self.use_learnable_topk:
                scalar_feats = getattr(data, self.scalar_feat_key, None)
                if scalar_feats is None:
                    raise ValueError(
                        f"Scalar features '{self.scalar_feat_key}' missing on Data for learnable top-k."
                    )
                if scalar_feats.dim() == 1:
                    scalar_feats = scalar_feats.unsqueeze(-1)
                self._lazy_init_learnable_topk_layer(
                    scalar_dim=int(scalar_feats.shape[-1]),
                    device=x_v.device,
                )
                base_edge_index = getattr(
                    data,
                    "diffusion_edge_index",
                    getattr(data, "edge_index", None),
                )
                if base_edge_index is None:
                    raise ValueError("Diffusion edge_index is required for learnable top-k.")
                edge_index_override, edge_weight_override = self.learnable_topk_layer(
                    scalar_feats.to(x_v.device),
                    base_edge_index.to(x_v.device),
                    topk=int(self.learnable_topk_k),
                    temperature=None,
                )
            diffusion_op = self._build_learnable_diffusion(
                data,
                x_v,
                edge_index_override=edge_index_override,
                edge_weight_override=edge_weight_override,
            )
        else:
            diffusion_op = getattr(data, "Q")

        # Scatter the vector features across the graph using the diffusion operator(s)
        coeffs = self.multi_view_scatter(
            x_v,
            diffusion_op,
            diffusion_kwargs=self.diffusion_kwargs,
            num_scattering_layers=self.num_scattering_layers,
        )
        N, d, W = coeffs.shape

        # Normalize the scattered coefficients using VectorBatchNorm
        if self.vector_bn is None or self.projection_mlp is None:
            self._lazy_init_layers(
                num_wavelets=W,
                vector_dim=d,
                device=x_v.device,
            )
        if self.use_vector_batch_norm and self.vector_bn is not None:
            coeffs_bn = coeffs.view(N, 1, d, W)
            coeffs_bn = self.vector_bn(coeffs_bn)
            coeffs = coeffs_bn.view(N, d, W)

        # Flatten the scattered coefficients and project to embedding space
        features = coeffs.reshape(N, d * W)
        embeddings = self.projection_mlp(features) \
            if self.projection_mlp is not None else features
        teacher_embeddings = embeddings
        distill_preds = None
        distill_target = None

        # Knowledge distillation path
        if self.use_distill_inference:
            distill_inputs = self._compose_distill_features(
                data=data,
                vector_features=x_v.to(features.device),
            )
            if self.distill_mlp is not None:
                distill_preds = self.distill_mlp(distill_inputs)
                distill_target = teacher_embeddings.detach()

                # In case of shape mismatch, pad teacher embeddings for MSE loss
                # if distill_preds.shape[-1] != distill_target.shape[-1]:
                #     if not self._distill_warned_shape_mismatch:
                #         print(
                #             "[Distill] Target and distill dimensions differ; "
                #             "padding/truncating teacher embeddings for MSE alignment.",
                #             flush=True,
                #         )
                #         self._distill_warned_shape_mismatch = True
                #     distill_target = self._match_distill_target_dim(
                #         target=distill_target,
                #         desired_dim=int(distill_preds.shape[-1]),
                #     )

        # Build SupCon training subset
        target_subset = None
        if self.training:
            supcon_subset = self._build_supcon_training_subset(
                data=data,
                embeddings=embeddings,
            )
            if supcon_subset is not None:
                embeddings, target_subset = supcon_subset

        # Return output dictionary
        out = {
            "preds": embeddings,
            "embeddings": embeddings,
            "target_subset": target_subset,
        }
        if self.use_distill_inference:
            out["distill_preds"] = distill_preds
            out["distill_target"] = distill_target

        return out

    def get_learnable_topk_transform(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return (proj_weight, metric_diag) for the learned top-k layer, if enabled.
        """
        if not self.use_learnable_topk:
            return None
        if self.learnable_topk_layer is None:
            return None
        proj_weight = self.learnable_topk_layer.proj.weight.detach()
        metric_diag = F.softplus(self.learnable_topk_layer.log_diag) + self.learnable_topk_layer.eps
        return proj_weight, metric_diag

    def loss(
        self,
        input_dict: Dict[str, torch.Tensor],
        output_dict: Dict[str, torch.Tensor],
        phase: str,
    ) -> Dict[str, torch.Tensor]:
        embeddings = output_dict["embeddings"]
        labels = output_dict.get("target_subset")
        if labels is None:
            labels = input_dict["target"].view(-1)
        else:
            labels = labels.view(-1)
        loss = self.supcon_loss(embeddings, labels)
        loss_dict: Dict[str, torch.Tensor] = {
            "loss": loss, 
            "size": embeddings.shape[0]
        }
        if self.use_distill_inference:
            distill_preds = output_dict.get("distill_preds")
            distill_target = output_dict.get("distill_target")
            if distill_preds is not None and distill_target is not None:
                distill_loss = self.distill_loss_fn(distill_preds, distill_target)
                loss = loss + distill_loss
                loss_dict["loss"] = loss
                loss_dict["distill_loss"] = distill_loss
                loss_dict["distill_size"] = torch.tensor(distill_preds.shape[0])
        return loss_dict

    def update_metrics(
        self,
        phase: str,
        loss_dict: Dict[str, torch.Tensor],
        input_dict: Dict[str, torch.Tensor],
        output_dict: Dict[str, torch.Tensor],
    ) -> None:
        def _to_float(value: torch.Tensor | float | int) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        if self.loss_keys is None:
            self.loss_keys = [
                key for key in loss_dict.keys() if "loss" in key.lower()
            ] or ["loss"]
            for phase_name, stats in self.epoch_loss_dict.items():
                for loss_key in self.loss_keys:
                    stats.setdefault(loss_key, 0.0)
        phase_stats = self.epoch_loss_dict.setdefault(phase, {"size": 0.0})
        for loss_key in self.loss_keys:
            if loss_key not in phase_stats:
                phase_stats[loss_key] = 0.0
        loss_val = _to_float(loss_dict["loss"])
        size = int(loss_dict.get("size", output_dict["preds"].shape[0]))
        phase_stats["size"] += size
        for loss_key in self.loss_keys:
            if loss_key in loss_dict:
                phase_stats[loss_key] += _to_float(loss_dict[loss_key]) * size
            else:
                phase_stats[loss_key] += loss_val * size

    def calc_metrics(
        self, 
        epoch: int, 
        is_validation_epoch: bool
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {"epoch": epoch}
        loss_keys = self.loss_keys or ["loss"]
        phases = ("train", "valid") if is_validation_epoch else ("train",)
        for phase in phases:
            phase_stats = self.epoch_loss_dict.setdefault(phase, {"size": 0.0})
            sample_count = float(phase_stats.get("size", 0.0))
            for loss_key in loss_keys:
                loss_sum = float(phase_stats.get(loss_key, 0.0))
                avg_loss = loss_sum / sample_count if sample_count > 0 else 0.0
                metrics[f"{loss_key}_{phase}"] = avg_loss
                phase_stats[loss_key] = 0.0
            phase_stats["size"] = 0.0
        if (
            is_validation_epoch
            and self.supcon2_context is not None
            and self.supcon2_eval_fn is not None
            and self.supcon2_accelerator is not None
        ):
            acc = self.supcon2_accelerator
            train_acc = float("nan")
            val_acc = float("nan")
            run_eval = getattr(acc, "is_main_process", True)
            if run_eval:
                try:
                    _, svm_stats, _ = self.supcon2_eval_fn(
                        model=self,
                        train_data=self.supcon2_context['train_data'],
                        fold_data=self.supcon2_context['fold_data'],
                        expected_nodes=self.supcon2_context.get('expected_nodes'),
                        svm_kernel=self.supcon2_context['svm_kernel'],
                        svm_C=self.supcon2_context['svm_c'],
                        svm_gamma=self.supcon2_context['svm_gamma'],
                        include_test=False,
                        return_split_features=False,
                    )
                    train_acc = float(svm_stats.get("train_accuracy", float("nan")))
                    val_acc = float(svm_stats.get("val_accuracy", float("nan")))
                    if hasattr(acc, "print"):
                        if not math.isnan(train_acc):
                            acc.print(
                                f"[SupCon2] Train SVM accuracy: {train_acc:.4f}"
                            )
                        if not math.isnan(val_acc):
                            acc.print(
                                f"[SupCon2] Validation SVM accuracy: {val_acc:.4f}"
                            )
                except Exception as exc:
                    if hasattr(acc, "print"):
                        acc.print(
                            f"[SupCon2] Validation SVM evaluation failed: {exc}"
                        )
            broadcastable = [train_acc, val_acc]
            try:
                acc.broadcast_object_list(broadcastable)
                train_acc = float(broadcastable[0])
                val_acc = float(broadcastable[1])
            except Exception:
                pass
            metrics["svm_train_accuracy"] = train_acc
            metrics["svm_val_accuracy"] = val_acc
        if (
            self.learnable_P
            and self.learnable_p_num_views > 1
            and hasattr(self.learnable_p_layer, "compute_view_similarities")
        ):
            sims_vals: list[float] = []
            try:
                if self._last_learnable_diffusion_ops:
                    sims = self.learnable_p_layer.compute_view_similarities(
                        self._last_learnable_diffusion_ops
                    )
                    sims_vals = [float(s[2]) for s in sims]
                if sims_vals:
                    metrics["learnableP_frob_mean"] = float(torch.tensor(sims_vals).mean().item())
                    metrics["learnableP_frob_max"] = float(max(sims_vals))
                    print(
                        "[LearnableP] view Frobenius norms (post-epoch): "
                        + ", ".join(f"{v:.4e}" for v in sims_vals),
                        flush=True,
                    )
            except Exception as exc:
                print(f"[LearnableP] view similarity (post-epoch) failed: {exc}", flush=True)
            finally:
                self._last_learnable_diffusion_ops = None
        return metrics

    def print_epoch_metrics(self, epoch_metrics: Dict[str, Any]) -> list[str]:
        lines: list[str] = []
        for key, label in (
            ("svm_train_accuracy", "SupCon train SVM acc"),
            ("svm_val_accuracy", "SupCon valid SVM acc"),
        ):
            if key not in epoch_metrics:
                continue
            try:
                val = float(epoch_metrics[key])
            except (TypeError, ValueError):
                continue
            if math.isnan(val):
                continue
            lines.append(f"{label}: {val:.6f}")
        if lines:
            return ["SupCon metrics:"] + [f"\t{ln}" for ln in lines]
        return []

    # ---------------------- public helper ----------------------
    def run_epoch_zero_methods(self, batch: Batch | Data) -> None:
        """
        Materialize lazily initialized layers so parameter counts are non-zero.
        """
        try:
            data = self._prepare_batch(batch)
        except ValueError:
            return

        param = next(self.parameters(), None)
        if param is not None:
            device = param.device
        else:
            vec = getattr(data, self.vector_feat_key, None)
            if isinstance(vec, torch.Tensor):
                device = vec.device
            else:
                device = torch.device("cpu")

        if hasattr(data, "to"):
            data = data.to(device)

        x_v = getattr(data, self.vector_feat_key, None)
        diffusion_op = getattr(data, "Q", None)
        if x_v is None or diffusion_op is None:
            return

        with torch.no_grad():
            coeffs = vector_multiorder_scatter(
                x_v.to(device), 
                diffusion_op.to(device), 
                self.diffusion_kwargs,
                self.num_scattering_layers,
            )
            _, d, W = coeffs.shape
            self._lazy_init_layers(
                num_wavelets=W,
                vector_dim=d,
                device=device,
            )
            if self.use_distill_inference:
                scalar_feats = None
                if self.scalar_feat_key is not None and hasattr(data, self.scalar_feat_key):
                    scalar_feats = getattr(data, self.scalar_feat_key)
                    if scalar_feats is not None:
                        scalar_feats = scalar_feats.to(device)
                        if scalar_feats.dim() == 1:
                            scalar_feats = scalar_feats.unsqueeze(-1)
                scalar_dim = int(scalar_feats.shape[-1]) if scalar_feats is not None else 0
                self._lazy_init_distill_mlp(
                    input_dim=int(d + scalar_dim),
                    device=device,
                )

    @torch.no_grad()
    def compute_embeddings(self, data: Data) -> torch.Tensor:
        device = self.get_device()
        self.eval()
        batch = data.to(device)
        with torch.no_grad():
            outputs = self(batch)
        use_distill = (
            self.use_distill_inference
            and getattr(self, "eval_mode", None) == "distill"
        )
        if use_distill:
            distill_preds = outputs.get("distill_preds")
            if distill_preds is not None:
                return distill_preds.detach().cpu()
        else:
            return outputs["embeddings"].detach().cpu()

    def _build_supcon_training_subset(
        self,
        *,
        data: Data,
        embeddings: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Slice embeddings down to anchor-centric subsets with controlled pos/neg sampling.

        Steps:
        1. Pick anchor nodes stratified by label (at least one per class).
        2. For each anchor, sample up to ``pos_pairs_per_anchor`` positives (same label).
        3. For every anchor-positive pair, gather nearest negatives plus global random negatives.
        4. Record positive/negative pair statistics for quick monitoring.
        """
        if self.supcon_sampling_max_nodes is None:
            return None
        labels = getattr(data, self.target_key, None)
        if labels is None:
            return None
        labels = labels.view(-1)
        total_nodes = int(labels.numel())
        if total_nodes == 0:
            return None
        device = labels.device
        max_nodes = int(self.supcon_sampling_max_nodes)
        if max_nodes <= 0:
            return None

        trial_ids = getattr(data, "node_trial_id", None)
        if trial_ids is not None:
            trial_ids = trial_ids.view(-1).to(device)
        edge_index = getattr(data, "edge_index", None)

        unique_labels = torch.unique(labels)
        sample_size = min(max_nodes, total_nodes)
        sample_size = max(sample_size, int(unique_labels.numel()))
        anchor_indices = self._stratified_anchor_indices(
            labels=labels,
            sample_size=sample_size,
        )
        if anchor_indices.numel() == 0:
            return None
        anchors_device = anchor_indices.device

        selections: list[torch.Tensor] = []
        used_anchor_count = 0
        for anchor_idx in anchor_indices.tolist():
            anchor = int(anchor_idx)
            anchor_label = labels[anchor]

            allowed_mask = torch.ones(labels.shape, dtype=torch.bool, device=device)
            if trial_ids is not None:
                allowed_mask &= trial_ids != trial_ids[anchor]
            allowed_mask[anchor] = True

            region_mask = torch.ones(labels.shape, dtype=torch.bool, device=device)
            if (
                self.supcon_neighbor_k > 0
                and edge_index is not None
                and edge_index.numel() > 0
            ):
                anchor_for_subgraph = torch.tensor(
                    [anchor],
                    dtype=torch.long,
                    device=edge_index.device,
                )
                region_nodes, _, _, _ = k_hop_subgraph(
                    anchor_for_subgraph,
                    int(self.supcon_neighbor_k),
                    edge_index,
                    relabel_nodes=False,
                )
                region_mask = torch.zeros(labels.shape, dtype=torch.bool, device=device)
                region_mask[region_nodes.to(device)] = True

            positive_mask = (labels == anchor_label) & allowed_mask & region_mask
            positive_mask[anchor] = False
            positive_candidates = torch.nonzero(
                positive_mask,
                as_tuple=False,
            ).view(-1)
            if positive_candidates.numel() == 0:
                continue

            num_pos = positive_candidates.numel()
            pos_limit = (
                int(self.pos_pairs_per_anchor)
                if self.pos_pairs_per_anchor is not None
                else num_pos
            )
            pos_limit = max(1, pos_limit)
            pos_limit = min(pos_limit, num_pos)
            perm = torch.randperm(num_pos, device=anchors_device)
            positives = positive_candidates[perm[:pos_limit]]

            negative_mask = (labels != anchor_label) & allowed_mask
            negative_candidates = torch.nonzero(
                negative_mask,
                as_tuple=False,
            ).view(-1)
            negatives = self._select_negatives_for_anchor(
                positives=positives,
                negative_candidates=negative_candidates,
                embeddings=embeddings,
            )

            anchor_tensor = torch.tensor(
                [anchor],
                device=anchors_device,
                dtype=torch.long,
            )
            anchor_group = [anchor_tensor, positives]
            if negatives.numel() > 0:
                anchor_group.append(negatives)
            selections.append(torch.cat(anchor_group))
            used_anchor_count += 1

        if not selections:
            return None

        subset_nodes = torch.unique(torch.cat(selections), sorted=False)
        if subset_nodes.numel() < 2:
            return None
        subset_labels = labels[subset_nodes]
        pos_pairs, neg_pairs = self._compute_pair_stats(subset_labels)
        self._print_supcon_sampling_stats(
            num_total=int(subset_nodes.numel()),
            num_anchors=used_anchor_count,
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
        )
        return embeddings[subset_nodes], subset_labels

    def _stratified_anchor_indices(
        self,
        *,
        labels: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """
        Draw anchor indices by filling per-class bins before assigning remainders.

        The procedure:
        1. Compute a uniform per-class quota (sample_size // num_classes).
        2. Iterate over a random permutation of node indices, filling class bins
           until each reaches the quota (discarding overflow).
        3. After the bins are full, add the leftover remainder slots with any class.
        """
        device = labels.device
        total_nodes = int(labels.numel())
        if total_nodes == 0 or sample_size <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        sample_size = min(sample_size, total_nodes)

        classes, inverse = torch.unique(labels, return_inverse=True)
        num_classes = int(classes.numel())
        if num_classes == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        base_quota = sample_size // num_classes
        permuted = torch.randperm(total_nodes, device=device)
        per_class_counts = torch.zeros(num_classes, dtype=torch.int32, device=device)
        selected_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
        selected: list[int] = []

        if base_quota > 0:
            remaining_base = base_quota * num_classes
            for idx in permuted.tolist():
                if len(selected) >= sample_size or remaining_base == 0:
                    break
                class_idx = int(inverse[idx].item())
                if per_class_counts[class_idx] >= base_quota:
                    continue
                per_class_counts[class_idx] += 1
                remaining_base -= 1
                selected.append(idx)
                selected_mask[idx] = True
            # If we exhausted all nodes before filling the quota, allow remainder anyway.

        if len(selected) < sample_size:
            for idx in permuted.tolist():
                if len(selected) >= sample_size:
                    break
                if selected_mask[idx]:
                    continue
                selected.append(idx)
                selected_mask[idx] = True

        if not selected:
            return torch.empty(0, dtype=torch.long, device=device)
        anchors = torch.tensor(selected, dtype=torch.long, device=device)
        return torch.unique(anchors, sorted=False)

    @staticmethod
    def _compute_pair_stats(labels: torch.Tensor) -> Tuple[int, int]:
        """
        Count positive and negative label pairs for quick SupCon diagnostics.
        """
        if labels.numel() < 2:
            return 0, 0
        compare = labels.unsqueeze(0) == labels.unsqueeze(1)
        upper = torch.triu(compare, diagonal=1)
        pos_pairs = int(upper.sum().item())
        total_pairs = int(labels.numel() * (labels.numel() - 1) // 2)
        neg_pairs = total_pairs - pos_pairs
        return pos_pairs, neg_pairs

    def _select_negatives_for_anchor(
        self,
        *,
        positives: torch.Tensor,
        negative_candidates: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine nearest-neighbor negatives with globally sampled random negatives.
        """
        device = embeddings.device
        if positives.numel() == 0 or negative_candidates.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        selections: list[torch.Tensor] = []
        topk = self._topk_negatives_for_positives(
            positives=positives,
            negative_candidates=negative_candidates,
            embeddings=embeddings,
        )
        if topk.numel() > 0:
            selections.append(topk)

        rand_neg = self._sample_random_negatives(
            negatives=negative_candidates,
            exclude=torch.cat(selections) if selections else None,
            count=self.random_negatives_per_anchor,
        )
        if rand_neg.numel() > 0:
            selections.append(rand_neg)

        if not selections:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.unique(torch.cat(selections), sorted=False)

    def _topk_negatives_for_positives(
        self,
        *,
        positives: torch.Tensor,
        negative_candidates: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return unique negative nodes that are nearest to each positive embedding.
        """
        device = embeddings.device
        if (
            positives.numel() == 0
            or negative_candidates.numel() == 0
            or self.neg_topk_per_positive <= 0
        ):
            return torch.empty(0, dtype=torch.long, device=device)

        pos_emb = embeddings[positives]
        neg_emb = embeddings[negative_candidates]
        k = min(self.neg_topk_per_positive, neg_emb.shape[0])
        pos_norm = F.normalize(pos_emb, dim=1)
        neg_norm = F.normalize(neg_emb, dim=1)
        cosine_sim = torch.matmul(pos_norm, neg_norm.T)
        cosine_dist = 1 - cosine_sim
        distances = torch.clamp(cosine_dist, min=0.0)
        topk_idx = torch.topk(distances, k=k, dim=1, largest=False).indices
        neg_indices = negative_candidates[topk_idx.reshape(-1)]
        return torch.unique(neg_indices, sorted=False)

    def _sample_random_negatives(
        self,
        *,
        negatives: torch.Tensor,
        exclude: Optional[torch.Tensor],
        count: int,
    ) -> torch.Tensor:
        """
        Sample a random subset of negatives, excluding previously selected ones.
        """
        device = negatives.device
        if negatives.numel() == 0 or count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        available = negatives
        if exclude is not None and exclude.numel() > 0:
            keep_mask = ~torch.isin(negatives, exclude)
            available = negatives[keep_mask]
        if available.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        sample_size = min(int(count), int(available.numel()))
        perm = torch.randperm(available.numel(), device=device)
        return available[perm[:sample_size]]

    def _print_supcon_sampling_stats(
        self,
        *,
        num_total: int,
        num_anchors: int,
        pos_pairs: int,
        neg_pairs: int,
    ) -> None:
        """
        Print concise SupCon sampling statistics during training.
        """
        if num_total == 0:
            return
        ratio = float(pos_pairs) / float(neg_pairs) if neg_pairs > 0 else float("inf")
        print(
            "[SupCon2] sampled_nodes="
            f"{num_total} (anchors={num_anchors}), "
            f"pos_pairs={pos_pairs}, neg_pairs={neg_pairs}, "
            f"pos_to_neg_ratio={ratio:.3f}"
        )

