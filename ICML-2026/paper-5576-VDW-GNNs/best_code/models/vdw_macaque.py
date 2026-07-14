"""
Simplified VDW variant for macaque reaching: vector-only scattering from spike_data
with node-level or graph-level predictions for kinematics targets (regression) or
condition classification (multi-class classification).

Model class: VDW_macaque

Tasks supported:
1) Regression (node-level or graph-level kinematics prediction)
2) Multi-class classification (graph-level condition classification, 7 classes)

Forward pipeline for regression (two modes after scattering and mixing scattering paths):
1) Invariant path (use_vec_invariants=True):
   - Vector scattering on (tangent-space-projected) 'neural velocity' vectors using Q.
   - Within-track equivariant vector wavelet mixing along wavelet axis.
   - Invariants from mixed vector wavelets (norms and cosine similarities).
   - Concatenate raw scalar time feature `x` to invariants per node.
   - [If task_level='graph'] Flatten all node features per graph, preserving temporal structure.
   - Kinematics decoder: MLP with separate pos/vel heads (multi-task) or single head.
   - Nonlinearity: SiLU (default for regression)

2) Vector-preserving path (use_vec_invariants=False):
   - Vector scattering and mixing as above.
   - Flatten vector wavelets and concatenate raw scalar time feature.
   - [If task_level='graph'] Sum node vectors to graph-level, preserving directional information.
   - Linear projection to target(s) without intermediate MLP.
   - Preserves directional information, relies on wavelet mixing for expressivity.

Forward pipeline for classification:
- Always uses invariant path (forced)
- Always graph-level (forced)
- Vector scattering and mixing as above
- Invariants from mixed vector wavelets (norms and cosine similarities)
- Concatenate raw scalar time feature `x` to invariants per node
- Flatten all node features per graph, preserving temporal structure
- Classification head: MLP to num_classes logits
- Nonlinearity: LeakyReLU (forced for classification)
- Output: logits (no softmax applied)

Graph-level predictions:
- Set task_level='graph' to aggregate node features to graph-level before prediction.
- Invariant path aggregation modes (set graph_aggregation_mode):
  * 'flatten': Flattens node invariants per graph (N*F dims), preserving temporal ordering
  * 'pool_stats': Computes mean and variance across nodes (2*F dims), dimension-reducing
- Vector path: Sums node vectors per graph, preserving directional information.
- Target preprocessing: Graph-level targets (e.g., final position) should be set upstream in data loading (macaque_prepare_kth_fold) to match prediction shape.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter
from models.vdw_modular import VDWModular
from models.base_module import euclidean_distance_loss
from pyg_utilities import infer_device_from_batch
# from models.base_module import MultiTaskLoss # imported if needed in __init__


USE_VEC_TRACK_MIXING: bool = True
USE_VEC_INVARIANTS: bool = True
GRAPH_AGGREGATION_MODE: Literal['flatten', 'pool_stats'] = 'pool_stats'


class VDW_macaque(VDWModular):
    def __init__(
        self,
        *,
        base_module_kwargs: Dict[str, Any],
        vector_track_kwargs: Dict[str, Any],
        mixing_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
        target_names: Optional[str | tuple[str, ...]] = None,
        use_vec_track_mixing: bool = USE_VEC_TRACK_MIXING,
        use_vec_invariants: bool = USE_VEC_INVARIANTS,
        graph_aggregation_mode: Literal['flatten', 'pool_stats'] = GRAPH_AGGREGATION_MODE,
        task_names: str | tuple[str] = ('pos', 'vel'),
        task: Optional[str] = None,
        num_classes: Optional[int] = None,
        scatter_path_graphs: bool = False,
    ) -> None:
        """
        Initialize a macaque-specific VDWModular subclass model.

        Args:
            base_module_kwargs: Passed to BaseModule (task, loss, metrics, device, etc.).
            vector_track_kwargs: Keys used by VDWModular._scatter for the vector track:
                - 'feature_key': str (e.g., 'spike_data')
                - 'diffusion_op_key': str (e.g., 'Q')
                - 'vector_dim': int (K channels in spike_data)
                - 'diffusion_kwargs': dict containing wavelet scales settings
                - 'num_layers': int (1 or 2 for 0th/1st/(2nd))
            mixing_kwargs: Uses VDWModular.MIXING_DEFAULTS; only vector mixer is used here.
            head_kwargs: Node MLP configuration. Reuses VDWModular.HEAD_DEFAULTS keys:
                - 'node_scalar_head_hidden': List[int]
                - 'node_scalar_head_nonlin': nn.Module class
                - 'node_scalar_head_dropout': float
            target_names: Name(s) of target attribute(s). Single string for single-task,
                tuple of strings for multi-task (e.g., ('pos_xy', 'vel_xy')).
                If a tuple with len > 1, model is set to multi-task mode.
            use_vec_invariants: If True, convert vector wavelets to invariants before prediction.
                If False, preserve vector features and use linear projection to targets.
            graph_aggregation_mode: How to aggregate node-level invariants to graph-level.
                - 'flatten': Flatten all node features (N*F dims, assumes fixed graph size)
                - 'pool_stats': Compute mean and variance across nodes (2*F dims, efficient)
            task: Task string (e.g., 'node_regression', 'graph_multi_classification').
                Must contain level ('node' or 'graph') and type ('regression' or 'class'/'classification').
                Used to determine classification vs regression and node vs graph level.
            num_classes: Number of classes for classification tasks. Required if task contains
                'class' or 'classification'.
            scatter_path_graphs: If True, use per-trajectory path graphs (legacy mode).
                If False, use single spatial graph with one Q for all nodes (new pipeline).
        """
        # Store pipeline mode
        self.scatter_path_graphs = scatter_path_graphs
        
        # Extract task type and level from task string
        task_str = (task or base_module_kwargs.get('task', '')).lower()
        self.is_classification = ('multi' in task_str) and ('class' in task_str)
        self.num_classes = num_classes
        
        # Extract task level from task string
        task_level = 'graph' if 'graph' in task_str else 'node'
        
        # Override settings for classification
        if self.is_classification:
            task_level = 'graph'  # Force graph-level for classification
            use_vec_invariants = True  # Force invariant path for classification
            # Override head nonlinearity to LeakyReLU for classification
            if head_kwargs is None:
                head_kwargs = {}
            head_kwargs['node_scalar_head_nonlin'] = nn.LeakyReLU
        else:
            # Regression: use Euclidean distance loss for 2D vector targets (pos, vel)
            if 'loss_fn' not in base_module_kwargs:
                base_module_kwargs['loss_fn'] = euclidean_distance_loss
            # Regression: ensure SiLU is used (default behavior)
            if head_kwargs is None:
                head_kwargs = {}
            if 'node_scalar_head_nonlin' not in head_kwargs:
                head_kwargs['node_scalar_head_nonlin'] = nn.SiLU
        
        super().__init__(
            base_module_kwargs=base_module_kwargs,
            ablate_scalar_track=True,  # use raw time scalar, no scalar scattering
            ablate_vector_track=False,  # vector track enabled
            scalar_track_kwargs={},  # no scalar scattering in this model
            vector_track_kwargs=vector_track_kwargs,
            mixing_kwargs=mixing_kwargs,
            neighbor_kwargs=None,  # invariants use batch.edge_index
            head_kwargs=head_kwargs,
            readout_kwargs=None,  # graph readout not used
        )
        self.use_vec_track_mixing = use_vec_track_mixing
        self.use_vec_invariants = use_vec_invariants
        self.graph_aggregation_mode = graph_aggregation_mode
        self.task_level = task_level
        
        # Debug flags (one-time print)
        self._debug_invariants_once = False
        self._debug_pooling_once = False
        self._debug_vector_norm_once = False
        
        # For regression: infer multitask from target_names
        if not self.is_classification:
            if target_names is not None and isinstance(target_names, (tuple, list)):
                is_multitask = len(target_names) > 1
                self.target_names = tuple(target_names)
            else:
                is_multitask = False
                self.target_names = target_names if target_names is not None else task_names[0]
            self.is_multitask = is_multitask
        else:
            # Classification: single-task only
            self.is_multitask = False
            self.target_names = None
        
        # Set up prediction heads based on task type
        if self.is_classification:
            # Classification mode: single MLP head to num_classes (lazy init)
            self.task_names = None
            self.kin_decoder = None
            self.node_pred_mlp = None
            self.classification_head: Optional[nn.Module] = None  # Will be lazy-initialized in forward()
            self.linear_proj = None
            self.pre_mlp_norm: Optional[nn.LayerNorm] = None  # Will be lazy-initialized in forward()
            # Loss function already set by BaseModule (cross_entropy for multiclass)
        elif self.is_multitask:
            # Multi-task regression mode: dual-head decoder (lazy init)
            self.task_names = task_names
            self.kin_decoder: Optional[MLPKinematicsDecoder] = None
            self.node_pred_mlp = None  # Not used in multi-task mode
            self.classification_head = None
            self.pre_mlp_norm: Optional[nn.LayerNorm] = None  # Will be lazy-initialized in forward()
            # Linear projection heads for alternative forward path (lazy init)
            self.linear_proj_pos: Optional[nn.Linear] = None
            self.linear_proj_vel: Optional[nn.Linear] = None
            # Switch BaseModule to uncertainty-weighted multitask loss
            from models.base_module import MultiTaskLoss  # local import to avoid cycles
            self.loss_fn = MultiTaskLoss(self.task_names)
        else:
            # Single-task regression mode
            self.task_names = None
            self.kin_decoder = None  # Not used in single-task mode
            self.node_pred_mlp = None  # Will be lazy-initialized in forward()
            self.classification_head = None
            self.pre_mlp_norm: Optional[nn.LayerNorm] = None  # Will be lazy-initialized in forward()
            # Linear projection head for alternative forward path (lazy init)
            self.linear_proj: Optional[nn.Linear] = None
            # Keep default MSE loss from BaseModule
            # self.loss_fn is already F.mse_loss from BaseModule.__init__

        # Enable R^2 metric for regression by default on macaque
        if not self.is_classification:
            self.metrics_kwargs['use_r2'] = True
            # Reinitialize metrics stack to apply new flag
            self._set_up_metrics()


    def forward(
        self,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Forward pass with support for both pipelines:
        - scatter_path_graphs=True: Per-trajectory path graphs (legacy)
        - scatter_path_graphs=False: Single spatial graph with masks (new)
        """
        # Branch based on pipeline mode
        if self.scatter_path_graphs:
            return self._forward_path_graphs(batch)
        else:
            return self._forward_spatial_graph(batch)
    
    
    def _forward_path_graphs(
        self,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Original forward pass for per-trajectory path graphs.
        """
        outputs: Dict[str, torch.Tensor] = {}

        # [Optional] Include requested attributes alongside predictions for custom losses
        if self.attributes_to_include_with_preds is not None:
            for attr in self.attributes_to_include_with_preds:
                if attr in batch:
                    outputs[attr] = getattr(batch, attr)

        device = infer_device_from_batch(
            batch,
            feature_keys=[self.vector_track_kwargs.get('feature_key')],
            operator_keys=[self.vector_track_kwargs.get('diffusion_op_key')],
        )

        # Vector inputs (N, d) and operator Q (Nd x Nd)
        vec_key = self.vector_track_kwargs.get('feature_key')
        op_key = self.vector_track_kwargs.get('diffusion_op_key')
        if (vec_key is None) or (op_key is None):
            raise ValueError("vector_track_kwargs must define 'feature_key' and 'diffusion_op_key' (attributes of the PyG Batch objects).")

        x_v = getattr(batch, vec_key).to(device)  # (N, d)
        Q = getattr(batch, op_key).to(device)

        # Optional time scalar (N, 1): concatenate later to vector features or invariants
        x_time = getattr(batch, 'x', None)
        if x_time is not None:
            x_time = x_time.to(device)
            if x_time.dim() == 1:
                x_time = x_time.unsqueeze(-1)

        # 1) Vector scattering: (N, d) → (N, 1, d, W_total)
        W_vector_unmixed = self._scatter(
            track='vector',
            x0=x_v,
            P_or_Q=Q,
            kwargs=self.vector_track_kwargs,
            batch_index=(batch.batch if hasattr(batch, 'batch') else None),
        )
        
        # Normalize scattered vectors (rotation-equivariant): (N, 1, d, W) -> (N, W, 1, d) for normalization
        W_vector_unmixed_for_norm = W_vector_unmixed.permute(0, 3, 1, 2)  # (N, W, 1, d)
        W_vector_unmixed_normalized = self._normalize_vector_wavelets(W_vector_unmixed_for_norm)
        # Permute back to (N, 1, d, W) for mixing
        W_vector_unmixed = W_vector_unmixed_normalized.permute(0, 2, 3, 1).contiguous()

        # 2) Within-track (rotation-equivariant) vector mixing (along wavelet axis)
        if self.use_vec_track_mixing:
            W_vector_to_mix = W_vector_unmixed.permute(0, 2, 1, 3)  # (N, d, 1, W)
            self._lazy_init_within_track_mixers(
                W_scalar=None, 
                W_vector=W_vector_to_mix,
            )
        if self.vector_mixer is not None:
            mixed = self.vector_mixer(W_vector_to_mix)  # (N, d, 1, W')
            W_vector = mixed.permute(0, 3, 2, 1).contiguous()  # (N, W', 1, d)
            # Normalize mixed vectors (rotation-equivariant)
            W_vector = self._normalize_vector_wavelets(W_vector)
        else:
            # No mixing: W_vector_unmixed is already normalized, just permute
            # (N, 1, d, W) -> (N, W, 1, d) to match expected downstream shape
            W_vector = W_vector_unmixed.permute(0, 3, 1, 2).contiguous()  # (N, W, 1, d)

        # 3) Branch: classification vs regression, invariant path vs. vector-preserving path
        if self.is_classification:
            # Classification: always use invariant path
            preds = self._forward_classification(
                W_vector_unmixed, W_vector, x_time, batch, device
            )
        elif self.use_vec_invariants:
            # Regression: invariant path
            preds = self._forward_vec_invariants_mlp(
                W_vector_unmixed, W_vector, x_time, batch, device
            )
        else:
            # Regression: vector-preserving path
            preds = self._forward_vec_linear_proj(
                W_vector, x_time, batch, device
            )
        
        outputs.update(preds)
        return outputs
    
    
    def _forward_spatial_graph(
        self,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single spatial graph with masks (no path graphs).
        
        Steps:
        1. Extract vector features and Q from spatial graph
        2. Scatter vectors across entire graph using Q
        3. Mix wavelets (equivariant mixing)
        4. Compute invariants per node (if use_vec_invariants)
        5. Pool nodes by trajectory if task_level is trajectory-level
        6. Pass to MLP head for predictions
        7. Return predictions with masks for loss computation
        """
        outputs: Dict[str, torch.Tensor] = {}
        device = infer_device_from_batch(
            batch,
            feature_keys=[self.vector_track_kwargs.get('feature_key')],
            operator_keys=[self.vector_track_kwargs.get('diffusion_op_key')],
        )
        
        # Extract vector features and Q
        vec_key = self.vector_track_kwargs.get('feature_key')
        op_key = self.vector_track_kwargs.get('diffusion_op_key')
        if (vec_key is None) or (op_key is None):
            raise ValueError(
                "vector_track_kwargs must define 'feature_key' and 'diffusion_op_key'"
            )
        
        x_v = getattr(batch, vec_key).to(device)  # (N, d)
        Q = getattr(batch, op_key).to(device)  # (Nd x Nd)
        
        # 1) Vector scattering: (N, d) → (N, 1, d, W_total)
        W_vector_unmixed = self._scatter(
            track='vector',
            x0=x_v,
            P_or_Q=Q,
            kwargs=self.vector_track_kwargs,
            batch_index=None,  # Single graph, no batch index needed
        )
        
        # Normalize scattered vectors (rotation-equivariant): (N, 1, d, W) -> (N, W, 1, d) for normalization
        W_vector_unmixed_for_norm = W_vector_unmixed.permute(0, 3, 1, 2)  # (N, W, 1, d)
        W_vector_unmixed_normalized = self._normalize_vector_wavelets(W_vector_unmixed_for_norm)
        # Permute back to (N, 1, d, W) for mixing
        W_vector_unmixed = W_vector_unmixed_normalized.permute(0, 2, 3, 1).contiguous()
        
        # 2) Within-track (rotation-equivariant) vector mixing (along wavelet axis)
        if self.use_vec_track_mixing:
            W_vector_to_mix = W_vector_unmixed.permute(0, 2, 1, 3)  # (N, d, 1, W)
            self._lazy_init_within_track_mixers(
                W_scalar=None,
                W_vector=W_vector_to_mix,
            )
        
        if self.vector_mixer is not None:
            mixed = self.vector_mixer(W_vector_to_mix)  # (N, d, 1, W')
            W_vector = mixed.permute(0, 3, 2, 1).contiguous()  # (N, W', 1, d)
            # Normalize mixed vectors (rotation-equivariant)
            W_vector = self._normalize_vector_wavelets(W_vector)
        else:
            # No mixing: W_vector_unmixed is already normalized, just permute
            # (N, 1, d, W) -> (N, W, 1, d)
            W_vector = W_vector_unmixed.permute(0, 3, 1, 2).contiguous()
        
        # 3) Compute invariants per node (always use invariant path for now)
        if not self.use_vec_invariants:
            raise NotImplementedError(
                "Vector-preserving path (use_vec_invariants=False) not yet "
                "implemented for spatial graph pipeline (scatter_path_graphs=False)"
            )
        
        # Compute vector invariants for all nodes
        vec_invars_unmixed = self._get_vector_invariants(
            W_vector_unmixed.permute(0, 3, 1, 2), batch, invariant_mode='intra_wavelet_dot'
        )  # (N, F_u)
        vec_invars_mixed = self._get_vector_invariants(
            W_vector, batch, invariant_mode='intra_wavelet_dot'
        )  # (N, F_m)
        vec_invars = torch.cat([vec_invars_unmixed, vec_invars_mixed], dim=1)  # (N, F)
        
        # 4) Detect task level and pool if needed
        # Trajectory-level task: pool nodes by trial (detected by 'final_' in target_key)
        # Node-level task: keep node-level features
        is_trajectory_level = (self.task_level == 'graph') or \
                               ('final' in str(getattr(batch, 'target_key', '')).lower())
        
        # Debug: Print task level detection once
        if not hasattr(self, '_debug_task_level_printed'):
            print(f"[DEBUG] task_level={self.task_level}, is_trajectory_level={is_trajectory_level}")
            print(f"[DEBUG] batch has target_key: {hasattr(batch, 'target_key')}")
            if hasattr(batch, 'target_key'):
                print(f"[DEBUG] batch.target_key={batch.target_key}")
            self._debug_task_level_printed = True
        
        if is_trajectory_level:
            # Pool by trajectory
            node_features = vec_invars  # (N, F)
            trajectory_features = self._pool_by_trajectory(node_features, batch)  # (num_trajectories, F)
            
            # Apply layer normalization
            if self.pre_mlp_norm is None:
                self.pre_mlp_norm = nn.LayerNorm(trajectory_features.shape[1]).to(device)
            t = self.pre_mlp_norm(trajectory_features)
            
            # 5) Pass to MLP head
            if self.is_multitask:
                if self.kin_decoder is None:
                    self._lazy_init_kinematics_decoder(in_dim=t.shape[1], device=device)
                preds_tasks = self.kin_decoder(t)
                main_key = 'vel' if 'vel' in self.task_names[0].lower() else 'pos'
                outputs['preds'] = preds_tasks[main_key]
                outputs['preds_tasks'] = preds_tasks
            else:
                if self.node_pred_mlp is None:
                    self._lazy_init_node_pred_mlp(in_dim=t.shape[1], out_dim=2, device=device)
                outputs['preds'] = self.node_pred_mlp(t)  # (num_trajectories, 2)
            
            # 6) Pool targets by trajectory (mean over nodes per trial)
            # Targets are stored as node-level; pool to trajectory-level
            pos_xy = batch.pos_xy.to(device)  # (N, 2)
            vel_xy = batch.vel_xy.to(device) if hasattr(batch, 'vel_xy') else None  # (N, 2)
            
            pos_xy_pooled = self._pool_targets_by_trajectory(pos_xy, batch)  # (num_trajectories, 2)
            if vel_xy is not None:
                vel_xy_pooled = self._pool_targets_by_trajectory(vel_xy, batch)
            
            # Attach pooled targets and trajectory-level masks
            outputs['targets'] = pos_xy_pooled
            if vel_xy is not None and self.is_multitask:
                outputs['targets_tasks'] = {
                    'pos': pos_xy_pooled,
                    'vel': vel_xy_pooled,
                }
            
            # Attach trajectory-level masks (derived from node masks)
            train_traj_mask = self._get_trajectory_masks(batch, 'train')
            valid_traj_mask = self._get_trajectory_masks(batch, 'valid')
            test_traj_mask = self._get_trajectory_masks(batch, 'test')
            
            # Debug: Print mask statistics once
            if self.training and not hasattr(self, '_debug_masks_printed'):
                print(f"[DEBUG] Trajectory masks: train={train_traj_mask.sum().item()}/{len(train_traj_mask)}, "
                      f"valid={valid_traj_mask.sum().item()}/{len(valid_traj_mask)}, "
                      f"test={test_traj_mask.sum().item()}/{len(test_traj_mask)}")
                print(f"[DEBUG] Node masks: train={batch.train_mask.sum().item()}/{len(batch.train_mask)}, "
                      f"valid={batch.valid_mask.sum().item()}/{len(batch.valid_mask)}, "
                      f"test={batch.test_mask.sum().item()}/{len(batch.test_mask)}")
                self._debug_masks_printed = True
            
            outputs['train_mask'] = train_traj_mask
            outputs['valid_mask'] = valid_traj_mask
            outputs['test_mask'] = test_traj_mask
            
            # CRITICAL: Replace batch-level node masks with trajectory-level masks
            # so the training loop uses the correct masks for trajectory-level predictions
            batch.train_mask = train_traj_mask
            batch.valid_mask = valid_traj_mask
            batch.test_mask = test_traj_mask
            
            # Also update batch.y to point to trajectory-level targets
            # so training loop can access them correctly
            batch.y = pos_xy_pooled
            
            # Debug: Print output shapes once
            if not hasattr(self, '_debug_output_shapes_printed'):
                print(f"[DEBUG] Output shapes (trajectory-level):")
                print(f"  preds: {outputs['preds'].shape}")
                print(f"  targets: {outputs['targets'].shape}")
                print(f"  train_mask: {train_traj_mask.shape}, sum={train_traj_mask.sum()}")
                print(f"  valid_mask: {valid_traj_mask.shape}, sum={valid_traj_mask.sum()}")
                print(f"  test_mask: {test_traj_mask.shape}, sum={test_traj_mask.sum()}")
                self._debug_output_shapes_printed = True
        else:
            # Node-level task: keep node features
            if self.pre_mlp_norm is None:
                self.pre_mlp_norm = nn.LayerNorm(vec_invars.shape[1]).to(device)
            t = self.pre_mlp_norm(vec_invars)
            
            # Pass to MLP head
            if self.is_multitask:
                if self.kin_decoder is None:
                    self._lazy_init_kinematics_decoder(in_dim=t.shape[1], device=device)
                preds_tasks = self.kin_decoder(t)
                main_key = 'vel' if 'vel' in self.task_names[0].lower() else 'pos'
                outputs['preds'] = preds_tasks[main_key]
                outputs['preds_tasks'] = preds_tasks
            else:
                if self.node_pred_mlp is None:
                    self._lazy_init_node_pred_mlp(in_dim=t.shape[1], out_dim=2, device=device)
                outputs['preds'] = self.node_pred_mlp(t)  # (N, 2)
            
            # Attach node-level targets and masks
            outputs['targets'] = batch.pos_xy.to(device)
            if hasattr(batch, 'vel_xy') and self.is_multitask:
                outputs['targets_tasks'] = {
                    'pos': batch.pos_xy.to(device),
                    'vel': batch.vel_xy.to(device),
                }
            
            # Attach node-level masks
            outputs['train_mask'] = batch.train_mask
            outputs['valid_mask'] = batch.valid_mask
            outputs['test_mask'] = batch.test_mask
            
            # Ensure batch.y points to correct targets for training loop
            if not hasattr(batch, 'y'):
                batch.y = batch.pos_xy.to(device)
        
        return outputs


    def _compute_invariant_features(
        self,
        W_vector_unmixed: torch.Tensor,
        W_vector: torch.Tensor,
        x_time: Optional[torch.Tensor],
        batch: Batch,
        flatten_for_graph_level: bool = False,
    ) -> torch.Tensor:
        """
        Shared helper: compute invariants from vector wavelets, concatenate time scalar,
        and optionally aggregate to graph-level.
        
        Returns:
            Tensor of shape (N, F) for node-level or (num_graphs, F') for graph-level
        """
        # Compute invariants from both unmixed and mixed vector wavelets
        vec_invars_unmixed = self._get_vector_invariants(
            W_vector_unmixed.permute(0, 3, 1, 2), batch, invariant_mode='intra_wavelet_dot'
        )  # (N, F_u)
        vec_invars_mixed = self._get_vector_invariants(
            W_vector, batch, invariant_mode='intra_wavelet_dot'
        )  # (N, F_m)
        vec_invars = torch.cat(
            [vec_invars_unmixed, vec_invars_mixed], 
            dim=1
        )  # (N, F_u + F_m)
        
        # Debug: Check if vector normalization controlled invariant magnitudes
        if self.training and hasattr(self, '_debug_invariants_once') and not self._debug_invariants_once:
            print(f"[DEBUG] vec_invars_unmixed stats: mean={vec_invars_unmixed.mean().item():.4f}, "
                  f"std={vec_invars_unmixed.std().item():.4f}, "
                  f"shape={vec_invars_unmixed.shape}")
            print(f"[DEBUG] vec_invars_mixed stats: mean={vec_invars_mixed.mean().item():.4f}, "
                  f"std={vec_invars_mixed.std().item():.4f}, "
                  f"shape={vec_invars_mixed.shape}")
            print(f"[DEBUG] W_vector stats (normalized): mean={W_vector.mean().item():.4f}, "
                  f"std={W_vector.std().item():.4f}, "
                  f"shape={W_vector.shape}")
            # Check vector norms per channel
            W_norms = torch.norm(W_vector, p=2, dim=-1)  # (N, W, 1)
            print(f"[DEBUG] W_vector norms: mean={W_norms.mean().item():.4f}, "
                  f"std={W_norms.std().item():.4f}")
            self._debug_invariants_once = True

        # Concatenate raw scalar feature (time index) to invariants
        if x_time is not None:
            x_time = x_time.to(dtype=vec_invars.dtype)
            t = torch.cat([vec_invars, x_time], dim=1)
        else:
            t = vec_invars
        
        # Graph-level aggregation if requested
        if flatten_for_graph_level:
            if self.graph_aggregation_mode == 'flatten':
                t = self._flatten_node_features_per_graph(t, batch)
            elif self.graph_aggregation_mode == 'pool_stats':
                t_before = t
                t = self._pool_scalar_node_features_per_graph(t, batch)
                # Debug: Check pooling output
                if self.training and hasattr(self, '_debug_pooling_once') and not self._debug_pooling_once:
                    print(f"[DEBUG] Before pooling: shape={t_before.shape}, "
                          f"mean={t_before.mean().item():.4f}, std={t_before.std().item():.4f}, "
                          f"min={t_before.min().item():.4f}, max={t_before.max().item():.4f}")
                    print(f"[DEBUG] After pooling: shape={t.shape}, "
                          f"mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
                          f"min={t.min().item():.4f}, max={t.max().item():.4f}")
                    print(f"[DEBUG] graph_aggregation_mode={self.graph_aggregation_mode}")
                    self._debug_pooling_once = True
            else:
                raise ValueError(
                    f"Unknown graph_aggregation_mode: {self.graph_aggregation_mode}. "
                    f"Supported values are 'flatten' and 'pool_stats'."
                )
        
        return t


    def _forward_vec_invariants_mlp(
        self,
        W_vector_unmixed: torch.Tensor,
        W_vector: torch.Tensor,
        x_time: Optional[torch.Tensor],
        batch: Batch,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Invariant path: convert vectors to invariants, then use MLP (KinematicsDecoder) for prediction.
        """
        outputs: Dict[str, torch.Tensor] = {}
        
        # Compute invariant features
        t = self._compute_invariant_features(
            W_vector_unmixed, 
            W_vector, 
            x_time, 
            batch,
            flatten_for_graph_level=(self.task_level == 'graph')
        )
        
        # Apply layer normalization for numerical stability (lazy init)
        if self.pre_mlp_norm is None:
            self.pre_mlp_norm = nn.LayerNorm(t.shape[1]).to(device)
        t = self.pre_mlp_norm(t)

        # Prediction head: single-task or multi-task MLP
        if self.is_multitask:
            if self.kin_decoder is None:
                self._lazy_init_kinematics_decoder(in_dim=t.shape[1], device=device)
            preds_tasks = self.kin_decoder(t)
            # Use first target name to determine main output
            main_key = 'vel' if 'vel' in self.task_names[0].lower() else 'pos'
            outputs['preds'] = preds_tasks[main_key]
            outputs['preds_tasks'] = preds_tasks

            # Provide targets for multi-task loss
            if len(self.target_names) >= 2:
                targets_tasks: Dict[str, torch.Tensor] = {
                    'pos': getattr(batch, self.target_names[0]).to(device),
                    'vel': getattr(batch, self.target_names[1]).to(device),
                }
                outputs['targets_tasks'] = targets_tasks
        else:
            if self.node_pred_mlp is None:
                self._lazy_init_node_pred_mlp(
                    in_dim=t.shape[1], out_dim=2, device=device
                )
            outputs['preds'] = self.node_pred_mlp(t)
        
        return outputs


    def _forward_vec_linear_proj(
        self,
        W_vector: torch.Tensor,
        x_time: Optional[torch.Tensor],
        batch: Batch,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Vector-preserving path: flatten vectors, concatenate time, linear projection for kinematics decoding.
        """
        outputs: Dict[str, torch.Tensor] = {}
        
        # Flatten mixed vector wavelets: (N, W', 1, d) -> (N, W'*d)
        N, W_prime, _, d = W_vector.shape
        vec_flat = W_vector.view(N, W_prime * d)
        
        # Concatenate time scalar to flattened vector features
        if x_time is not None:
            x_time = x_time.to(dtype=vec_flat.dtype)
            t = torch.cat([vec_flat, x_time], dim=1)
        else:
            t = vec_flat
        
        # Graph-level aggregation: sum vectors to preserve directional information
        if self.task_level == 'graph':
            t = self._aggregate_node_vectors(t, batch)
        
        # Linear projection to target(s)
        if self.is_multitask:
            if self.linear_proj_pos is None:
                self.linear_proj_pos = nn.Linear(t.shape[1], 2, bias=True).to(device)
            if self.linear_proj_vel is None:
                self.linear_proj_vel = nn.Linear(t.shape[1], 2, bias=True).to(device)
            
            preds_tasks = {
                'pos': self.linear_proj_pos(t),
                'vel': self.linear_proj_vel(t),
            }
            # Use first target name to determine main output
            main_key = 'vel' if 'vel' in self.target_names[0].lower() else 'pos'
            outputs['preds'] = preds_tasks[main_key]
            outputs['preds_tasks'] = preds_tasks

            # Provide targets for multi-task loss, if applicable
            if len(self.target_names) >= 2:
                targets_tasks: Dict[str, torch.Tensor] = {
                    'pos': getattr(batch, self.target_names[0]).to(device),
                    'vel': getattr(batch, self.target_names[1]).to(device),
                }
                outputs['targets_tasks'] = targets_tasks
        else:  # single target projection
            if self.linear_proj is None:
                self.linear_proj = nn.Linear(t.shape[1], 2, bias=True).to(device)
            outputs['preds'] = self.linear_proj(t)
        
        return outputs


    def _forward_classification(
        self,
        W_vector_unmixed: torch.Tensor,
        W_vector: torch.Tensor,
        x_time: Optional[torch.Tensor],
        batch: Batch,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Classification path: compute invariants, flatten per graph, predict class logits.
        Returns logits of shape (num_graphs, num_classes).
        """
        outputs: Dict[str, torch.Tensor] = {}
        
        # Compute invariant features (always graph-level for classification)
        t = self._compute_invariant_features(
            W_vector_unmixed, 
            W_vector, 
            x_time, 
            batch,
            flatten_for_graph_level=True
        )
        
        # Apply layer normalization for numerical stability (lazy init)
        if self.pre_mlp_norm is None:
            self.pre_mlp_norm = nn.LayerNorm(t.shape[1]).to(device)
        t = self.pre_mlp_norm(t)
        
        # Classification head: MLP to num_classes logits
        if self.classification_head is None:
            self._lazy_init_classification_head(in_dim=t.shape[1], device=device)
        
        logits = self.classification_head(t)  # (num_graphs, num_classes)
        outputs['preds'] = logits
        
        return outputs


    def _flatten_node_features_per_graph(
        self,
        node_features: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Flatten node features per graph, preserving all node information.
        Assumes all graphs have the same number of nodes.
        Used for invariant features where temporal ordering matters.
        
        Args:
            node_features: (N, F) tensor of node features where N = num_graphs * num_nodes_per_graph
            batch: PyG Batch object with batch assignment vector
            
        Returns:
            (num_graphs, num_nodes_per_graph * F) tensor of flattened features per graph
        """
        batch_index = batch.batch
        num_graphs = int(batch_index.max().item()) + 1
        total_nodes = node_features.shape[0]
        feat_dim = node_features.shape[1]
        
        # Infer number of nodes per graph (assumes all graphs have same number of nodes)
        num_nodes_per_graph = total_nodes // num_graphs
        
        # Reshape: (N, F) -> (num_graphs, num_nodes_per_graph, F) -> (num_graphs, num_nodes_per_graph * F)
        graph_features = node_features.view(num_graphs, num_nodes_per_graph, feat_dim)
        flattened = graph_features.view(num_graphs, num_nodes_per_graph * feat_dim)
        
        return flattened


    def _pool_scalar_node_features_per_graph(
        self,
        node_features: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Pool node features per graph by computing mean and variance across nodes.
        Uses torch_scatter for efficiency. This is dimension-reducing: (N, F) -> (num_graphs, 2*F).
        
        Note: No pre-pooling standardization needed since vectors are normalized before
        computing invariants, keeping magnitudes controlled.
        
        Args:
            node_features: (N, F) tensor of node features
            batch: PyG Batch object with batch assignment vector
            
        Returns:
            (num_graphs, 2*F) tensor of pooled features per graph (mean and variance concatenated)
        """
        batch_index = batch.batch
        num_graphs = int(batch_index.max().item()) + 1
        
        # Compute mean per graph
        mean = scatter(
            node_features,
            batch_index,
            dim=0,
            dim_size=num_graphs,
            reduce='mean',
        )  # (num_graphs, F)
        
        # Compute variance per graph: E[X^2] - E[X]^2
        mean_sq = scatter(
            node_features.pow(2),
            batch_index,
            dim=0,
            dim_size=num_graphs,
            reduce='mean',
        )  # (num_graphs, F)
        var = torch.clamp(mean_sq - mean.pow(2), min=0.0)  # (num_graphs, F)
        
        # Concatenate mean and variance
        pooled = torch.cat([mean, var], dim=1)  # (num_graphs, 2*F)
        
        return pooled


    def _aggregate_node_vectors(
        self,
        node_features: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Aggregate node-level vector features to graph-level using summation.
        Used for vector-preserving path to maintain directional information.
        
        Args:
            node_features: (N, F) tensor of node features
            batch: PyG Batch object with batch assignment vector
            
        Returns:
            (num_graphs, F) tensor of summed features per graph
        """
        batch_index = batch.batch
        num_graphs = int(batch_index.max().item()) + 1
        
        return scatter(
            node_features,
            batch_index,
            dim=0,
            dim_size=num_graphs,
            reduce='sum',
        )


    def _normalize_vector_wavelets(
        self,
        W_vector: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Normalize vector wavelets in a rotation-equivariant way by rescaling each
        wavelet channel so that the mean norm of vectors in that channel is 1.
        
        This preserves direction (rotation equivariance) while controlling magnitude
        to prevent invariants from exploding.
        
        Args:
            W_vector: (N, W, 1, d) tensor of vector wavelets
            eps: Small constant for numerical stability
            
        Returns:
            (N, W, 1, d) tensor of normalized vector wavelets
        """
        # Compute norms per vector: (N, W, 1, d) -> (N, W, 1)
        norms = torch.norm(W_vector, p=2, dim=-1, keepdim=True)  # (N, W, 1, 1)
        
        # Compute mean norm per wavelet channel across nodes: (N, W, 1, 1) -> (1, W, 1, 1)
        mean_norms = norms.mean(dim=0, keepdim=True)  # (1, W, 1, 1)
        
        # Normalize: divide vectors by mean norm per channel
        # This makes mean norm = 1 per channel while preserving directions
        W_normalized = W_vector / (mean_norms + eps)  # (N, W, 1, d)
        
        return W_normalized


    def _pool_by_trajectory(
        self,
        node_features: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Pool node features by trajectory using trial_ptr from spatial graph.
        Uses mean pooling (or pool_stats if graph_aggregation_mode is set).
        
        Args:
            node_features: (N, F) tensor of node features
            batch: Spatial graph Data object with trial_ptr
            
        Returns:
            (num_trajectories, F) or (num_trajectories, 2*F) tensor of pooled features
        """
        trial_ptr = batch.trial_ptr  # (num_trajectories + 1,)
        num_trajectories = len(trial_ptr) - 1
        
        # Create trajectory assignment index per node
        trial_assignment = torch.zeros(
            node_features.shape[0], 
            dtype=torch.long, 
            device=node_features.device
        )
        for trial_idx in range(num_trajectories):
            start = int(trial_ptr[trial_idx].item())
            end = int(trial_ptr[trial_idx + 1].item())
            trial_assignment[start:end] = trial_idx
        
        # Pool based on aggregation mode
        if self.graph_aggregation_mode == 'pool_stats':
            # Compute mean and variance per trajectory
            mean = scatter(
                node_features,
                trial_assignment,
                dim=0,
                dim_size=num_trajectories,
                reduce='mean',
            )
            mean_sq = scatter(
                node_features.pow(2),
                trial_assignment,
                dim=0,
                dim_size=num_trajectories,
                reduce='mean',
            )
            var = torch.clamp(mean_sq - mean.pow(2), min=0.0)
            pooled = torch.cat([mean, var], dim=1)  # (num_trajectories, 2*F)
        else:
            # Simple mean pooling
            pooled = scatter(
                node_features,
                trial_assignment,
                dim=0,
                dim_size=num_trajectories,
                reduce='mean',
            )
        
        return pooled


    def _pool_targets_by_trajectory(
        self,
        node_targets: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Pool node-level targets by trajectory (mean).
        
        Args:
            node_targets: (N, target_dim) tensor of node-level targets
            batch: Spatial graph Data object with trial_ptr
            
        Returns:
            (num_trajectories, target_dim) tensor of pooled targets
        """
        trial_ptr = batch.trial_ptr
        num_trajectories = len(trial_ptr) - 1
        
        # Create trajectory assignment index per node
        trial_assignment = torch.zeros(
            node_targets.shape[0],
            dtype=torch.long,
            device=node_targets.device,
        )
        for trial_idx in range(num_trajectories):
            start = int(trial_ptr[trial_idx].item())
            end = int(trial_ptr[trial_idx + 1].item())
            trial_assignment[start:end] = trial_idx
        
        # Mean pooling for targets
        pooled = scatter(
            node_targets,
            trial_assignment,
            dim=0,
            dim_size=num_trajectories,
            reduce='mean',
        )
        
        return pooled


    def _get_trajectory_masks(
        self,
        batch: Batch,
        split: Literal['train', 'valid', 'test'],
    ) -> torch.Tensor:
        """
        Get trajectory-level boolean mask from node-level masks.
        A trajectory is in a split if ALL its nodes are in that split.
        
        Args:
            batch: Spatial graph Data object with node masks and trial_ptr
            split: Split name ('train', 'valid', or 'test')
            
        Returns:
            (num_trajectories,) boolean tensor
        """
        # Get node-level mask
        mask_name = f'{split}_mask'
        node_mask = getattr(batch, mask_name)  # (N,)
        
        trial_ptr = batch.trial_ptr
        num_trajectories = len(trial_ptr) - 1
        
        # Check if all nodes in each trajectory are in the split
        trajectory_mask = torch.zeros(
            num_trajectories,
            dtype=torch.bool,
            device=node_mask.device,
        )
        for trial_idx in range(num_trajectories):
            start = int(trial_ptr[trial_idx].item())
            end = int(trial_ptr[trial_idx + 1].item())
            trajectory_mask[trial_idx] = node_mask[start:end].all()
        
        return trajectory_mask


    # Override epoch-zero initializer to avoid creating unused scalar heads
    def run_epoch_zero_methods(self, batch: Batch) -> None:  # type: ignore[name-defined]
        if self.use_vec_track_mixing:
            device = infer_device_from_batch(
                batch,
                feature_keys=[self.vector_track_kwargs.get('feature_key')],
                operator_keys=[self.vector_track_kwargs.get('diffusion_op_key')],
            )
            vec_key = self.vector_track_kwargs.get('feature_key')
            op_key = self.vector_track_kwargs.get('diffusion_op_key')
            x_v = getattr(batch, vec_key)
            Q = getattr(batch, op_key)
            x_v = x_v.to(device)
            Q = Q.to(device)
            W_vector_unmixed = self._scatter(
                track='vector',
                x0=x_v,
                P_or_Q=Q,
                kwargs=self.vector_track_kwargs,
                batch_index=(batch.batch if hasattr(batch, 'batch') else None),
            )
            vec_for_init = W_vector_unmixed.permute(0, 2, 1, 3)
            # Initialize mixers only; prediction heads will be lazy-initialized in forward()
            self._lazy_init_within_track_mixers(W_scalar=None, W_vector=vec_for_init)
        else:
            pass


    def _lazy_init_kinematics_decoder(
        self,
        *,
        in_dim: int,
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.head_kwargs.get('node_scalar_head_hidden'))
        nonlin_cls: type[nn.Module] = self.head_kwargs.get('node_scalar_head_nonlin', nn.SiLU)
        dropout_p: float = float(self.head_kwargs.get('node_scalar_head_dropout', 0.0))
        self.kin_decoder = MLPKinematicsDecoder(
            in_dim=in_dim,
            hidden=hidden,
            nonlin_cls=nonlin_cls,
            dropout_p=dropout_p,
            device=device,
        )


    def _build_mlp(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden: Optional[List[int]] = None,
        nonlin_cls: Optional[type[nn.Module]] = None,
        dropout_p: float = 0.0,
        device: torch.device,
    ) -> nn.Sequential:
        """
        Shared helper to build MLP with configurable architecture and activation.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden: List of hidden layer dimensions (from head_kwargs if None)
            nonlin_cls: Activation class (from head_kwargs if None)
            dropout_p: Dropout probability (from head_kwargs if 0.0)
            device: Device to place the MLP on
            
        Returns:
            nn.Sequential MLP module
        """
        if hidden is None:
            hidden = list(self.head_kwargs.get('node_scalar_head_hidden', []))
        if nonlin_cls is None:
            nonlin_cls = self.head_kwargs.get('node_scalar_head_nonlin', nn.SiLU)
        if dropout_p == 0.0:
            dropout_p = float(self.head_kwargs.get('node_scalar_head_dropout', 0.0))
        
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        
        return nn.Sequential(*layers).to(device)


    def _lazy_init_node_pred_mlp(
        self,
        *,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> None:
        """Initialize regression MLP for single-task node/graph predictions."""
        self.node_pred_mlp = self._build_mlp(
            in_dim=in_dim,
            out_dim=out_dim,
            device=device
        )


    def _lazy_init_classification_head(
        self,
        *,
        in_dim: int,
        device: torch.device,
    ) -> None:
        """
        Initialize classification head MLP mapping invariant features to class logits.
        Uses same architecture as regression heads but outputs num_classes logits.
        """
        self.classification_head = self._build_mlp(
            in_dim=in_dim,
            out_dim=self.num_classes,
            device=device
        )


class MLPKinematicsDecoder(nn.Module):
    """
    MLP-based Kinematics decoder for macaque reaching task. Uses a shared-trunk, dual-head decoder for node-level lever position and velocity predictions (here, node-level equals per-timepoint, for a given trial trajectory graph).
    """
    def __init__(
        self,
        *,
        in_dim: int,
        hidden: List[int],
        nonlin_cls: type[nn.Module] = nn.SiLU,
        dropout_p: float = 0.0,
        head_output_dim: int = 2,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Trunk uses all but the last hidden width
        trunk_hidden_dims = list(hidden[:-1]) if len(hidden) > 0 else []
        dims = [in_dim] + trunk_hidden_dims
        trunk_layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            trunk_layers.append(nonlin_cls())
            if dropout_p > 0.0:
                trunk_layers.append(nn.Dropout(p=dropout_p))
        self.trunk = nn.Sequential(*trunk_layers)

        # Heads: map trunk_out_dim -> last_hidden -> 2
        trunk_out_dim = dims[-1] if len(dims) > 0 else in_dim
        head_hidden = hidden[-1] if len(hidden) > 0 else trunk_out_dim

        def make_head() -> nn.Sequential:
            layers: List[nn.Module] = []
            if trunk_out_dim != head_hidden:
                layers.append(nn.Linear(trunk_out_dim, head_hidden, bias=True))
                layers.append(nonlin_cls())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(head_hidden, head_output_dim, bias=True))
            return nn.Sequential(*layers)

        self.pos_head = make_head()
        self.vel_head = make_head()
        self.to(device)

    def forward(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(t) if len(self.trunk) > 0 else t
        pos = self.pos_head(h)
        vel = self.vel_head(h)
        return {'pos': pos, 'vel': vel}
