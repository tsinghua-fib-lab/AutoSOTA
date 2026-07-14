"""
This file contains useful classes
and functions for extending the
torch.nn.Module class for model-building,
and works with the accelerate library: 

(1) Class definition for 'BaseModule',
an extension of torch.nn.Module with
built-in loss and metrics methods for
regressor or binary classifier models.

(2) Function 'test_nn', which computes
basic metrics for regression and binary
classification models built from 
BaseModule.
"""

import models.nn_utilities as nnu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    # Dataset, 
    DataLoader
)
from torchmetrics.regression import (
    MeanSquaredError,
    R2Score,
    MeanAbsoluteError
)
from torchmetrics.classification import (
    Accuracy,
    BinaryAccuracy,
    BinaryRecall,
    BinaryF1Score,
    BinaryPrecision,
    BinarySpecificity,
    BinaryAUROC,
    BinaryConfusionMatrix
)
from torchmetrics.clustering import DunnIndex
from torchmetrics import Metric, MetricCollection
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional, Callable, Any, List
import torch.distributed as dist

# Import custom loss functions
from models.custom_loss_fns import MultiTaskLoss

# Import custom metrics
# NOTE: these have hardcoded defaults in their calls here,
# flexible params not implemented yet!
from models.custom_metrics import (
    MultiTargetMSE,
    SilhouetteScore,
    LogisticLinearAccuracy,
    SVMAccuracy,
    KMeansAccuracy,
    SpectralClusteringAccuracy,
)

# Dunn Index parameter for torchmetrics DunnIndex
DUNN_INDEX_P = 1


def euclidean_distance_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute Euclidean distance loss between input and target vectors.
    
    For 2D targets (e.g., position or velocity), this computes the L2 norm
    of the difference vector for each sample, then reduces across samples.
    This treats (x, y) as a single vector quantity rather than independent
    coordinates, which can help avoid 'tug of war' dynamics in training.
    
    Args:
        input: Predictions of shape (N, D) where D is the vector dimension
        target: Targets of shape (N, D)
        reduction: 'mean', 'sum', or 'none'
            - 'mean': return mean Euclidean distance across samples
            - 'sum': return sum of Euclidean distances
            - 'none': return per-sample Euclidean distances (N,)
    
    Returns:
        Scalar loss if reduction is 'mean' or 'sum', otherwise (N,) tensor
        
    Example:
        >>> preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> targets = torch.tensor([[1.5, 2.5], [3.2, 4.1]])
        >>> loss = euclidean_distance_loss(preds, targets)
        # Computes mean(sqrt((1.0-1.5)^2 + (2.0-2.5)^2), sqrt((3.0-3.2)^2 + (4.0-4.1)^2))
    """
    if input.shape != target.shape:
        raise ValueError(
            f"Input and target must have the same shape. "
            f"Got input: {input.shape}, target: {target.shape}"
        )
    
    # Compute difference vector
    diff = input - target
    
    # Compute Euclidean distance for each sample: sqrt(sum_over_dims(diff^2))
    # For numerical stability, use torch.norm which handles the sqrt properly
    distances = torch.norm(diff, p=2, dim=-1)  # (N,)
    
    if reduction == 'mean':
        return distances.mean()
    elif reduction == 'sum':
        return distances.sum()
    elif reduction == 'none':
        return distances
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'.")


class BaseModule(nn.Module):
    """
    Subclass of torch.nn.Module, designed to work
    flexibly with the 'train_model' function (in
    train_fn.py). Has built-in loss functions and 
    metrics calculation methods specific to regression 
    and binary classification models (if not overridden).
    
    __init__ args:
        task: string key description of the model task, e.g.,
            'regression' or 'binary classification'.
        loss_fn: (optional) functional loss to use; if
            'None', will attempt to assign a default loss
            function based on the 'task' argument in 
            __init__().
        loss_fn_kwargs: for a torch.nn.functional loss.
        target_name: string key for the prediction target.
        metrics_kwargs: kwargs for setting up metric calcu-
            lator objects, e.g., num_classes for multiclass
            accuracy.
        key_prefix: string prefix for each metric column
             name in the training records. Should end in '_'.
        on_best_model_kwargs: kwargs for the 'on_best_model'
            method (overridden in subclasses, if implemented).
        target_preproc_stats: Optional dictionary containing 
            target preprocessing statistics
        device: manual device onto which to move the model
            weights.
        has_lazy_parameter_initialization: Whether the model 
            instantiates parameters lazily (after __init__)
        has_normalized_train_targets: Whether the model has 
            train set target normalization
        attributes_to_include_with_preds: Optional list of 
            attributes to include with predictions/targets
            in the model output dict, in case the loss function 
            needs them
    """
    def __init__(
        self,
        task: str,
        loss_fn: Optional[Callable] = None,
        loss_fn_kwargs: Dict[str, Any] = {},
        target_name: str = None,
        metrics_kwargs: Dict[str, Any] = {},
        key_prefix: str = '',
        on_best_model_kwargs: Dict[str, Any] = {},
        target_preproc_stats: Optional[Dict[str, Any]] = None,
        device = None,
        has_lazy_parameter_initialization: bool = False,
        has_normalized_train_targets: bool = False,
        attributes_to_include_with_preds: Optional[List[str]] = None,
        verbosity: int = 0
    ):
        super(BaseModule, self).__init__()
        # ------------------------------------------------------------------
        # Whether the model instantiates parameters lazily (after __init__)
        # If True, training utilities should run a dummy forward pass before
        # wrapping the model with DistributedDataParallel so that all
        # parameters are registered. Sub-classes can override this flag.
        # ------------------------------------------------------------------
        self.has_lazy_parameter_initialization = has_lazy_parameter_initialization

        # Whether the model has train set target normalization
        # (Subclasses can override this flag)
        self.has_normalized_train_targets = has_normalized_train_targets

        self.task = task.lower()
        self.device = device
        self.target_name = target_name
        self.metrics_kwargs = metrics_kwargs
        self.target_dim = metrics_kwargs.get('num_outputs', 1)
        self.key_prefix = key_prefix
        self.on_best_model_kwargs = on_best_model_kwargs
        self.verbosity = verbosity
        self.attributes_to_include_with_preds = attributes_to_include_with_preds

        # If needed, store train target preprocessing statistics 
        # for de-normalization during metrics calculation
        self.target_preproc_stats = None
        if has_normalized_train_targets \
        and target_preproc_stats is not None \
        and ('center' in target_preproc_stats) \
        and ('scale' in target_preproc_stats):
            
            # Flag to indicate the model has normalized train set targets
            self.has_normalized_train_targets = True
            center = target_preproc_stats['center']
            scale = target_preproc_stats['scale']
            self.register_buffer('_target_center', center)
            self.register_buffer('_target_scale', scale)
            self.target_preproc_stats = {'center': center, 'scale': scale}

        self._set_up_metrics()
        self.set_device()
        
        if loss_fn is None:
            if 'reg' in self.task:
                self.loss_fn = F.mse_loss
            elif 'class' in self.task and 'bin' in self.task:
                # F.binary_cross_entropy_with_logits removes need
                # for sigmoid activation after last layer, but targets
                # need to be floats between 0 and 1
                # https://pytorch.org/docs/stable/generated/
                # torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
                self.loss_fn = F.binary_cross_entropy_with_logits
            elif 'class' in self.task and 'multi' in self.task:
                self.loss_fn = F.cross_entropy
            else:
                raise NotImplementedError(
                    f"No loss function implemented for task = '{self.task}'!"
                )
        else:
            self.loss_fn = loss_fn

        self.loss_fn_kwargs = {'reduction': 'mean'} \
            if loss_fn_kwargs is None else loss_fn_kwargs
        self.loss_keys = None # set in first call to update_metrics()

        self.epoch_loss_dict = {
            'train': {'size': 0.}, # init as floats since used in divison
            'valid': {'size': 0.}
        }

        self._wandb_watched = False  # Track if wandb.watch has been called
        

    def set_device(self):
        # optional: manually enforce device
        if self.device is not None:
            # self.device = torch.device(device)
            self.to(self.device)

    
    def get_device(self):
        # find device that the model weights are on
        return next(self.parameters()).device


    def print_epoch_metrics(self, epoch_metrics: Dict[str, Any]) -> List[str]:
        """
        Hook for subclasses to append custom epoch-level metric lines to logs.
        Return a list of strings; BaseModule provides a no-op default.
        """
        return []


    def _loss(
        self, 
        input_dict, 
        output_dict,
        phase
    ):
        """
        This fn wraps loss_fn so it takes dicts storing 
        preds, targets as inputs, and outputs a loss_dict.
        Separated out in case this module is just one head
        of a multi-head model, and its loss is just one
        term of a composite loss function.
        """
        preds = output_dict['preds']
        # may need to de-normalize preds for valid set
        # (if the model is trained to predict normalized targets)
        if phase == 'valid' and self.has_normalized_train_targets:
            preds = self.get_denormalized(preds)
            if self.verbosity > 0:
                print(f"[DEBUG] BaseModule._loss: de-normalizing valid set preds")
                preds_print = [f'{v:.2f}' for v in preds[:5].squeeze().detach().cpu().tolist()]
                print(f"\tpreds: {preds_print}")
                
        # print('preds.shape =', preds.shape)
        targets = input_dict['target']
        # print('targets.shape =', targets.shape)

        # [Optional] Print targets for debugging
        # if phase == 'valid' and self.verbosity > 0:
        #     targets_print = [f'{v:.2f}' for v in targets[:5].squeeze().detach().cpu().tolist()]
        #     print(f"\ttargets = {targets_print}")
        
        # 'targets' may itself be a dict holding
        # multiple targets
        if (self.target_name is not None) \
        and (isinstance(targets, dict)):
            targets = targets[self.target_name]
        # print('targets.shape =', targets.shape)
        # print('targets =', targets)
        
        # Generalized compound-loss module path: if loss_fn is a module and returns a
        # dict containing a 'loss' key, propagate it so component losses (if any)
        # can be logged without baking dataset-specific logic here.
        # if isinstance(self.loss_fn, nn.Module):
        #     out = None
        #     try:
        #         # Preferred signature: module accepts predictions, targets, and input_dict
        #         out = self.loss_fn(
        #             preds, 
        #             targets, 
        #             input_dict=input_dict
        #         )
        #     except Exception as e:
        #         raise e
        #     if isinstance(out, dict) and ('loss' in out):
        #         out['size'] = targets.shape[0]
        #         return out

        # If the loss function needs additional attributes, add them to the loss function 
        # kwargs so they get passed to the loss function
        if self.attributes_to_include_with_preds is not None:
            for attr in self.attributes_to_include_with_preds:
                if attr in input_dict:
                    self.loss_fn_kwargs[attr] = input_dict[attr]

        # Special handling: uncertainty-weighted multi-task loss
        # If loss_fn is an nn.Module that expects dicts (like MultiTaskLoss), support it
        if isinstance(self.loss_fn, nn.Module):
            # Build preds/targets dicts if available
            preds_dict = None
            targets_dict = None
            if 'preds_tasks' in output_dict:
                preds_dict = output_dict['preds_tasks']
            elif isinstance(preds, dict):
                preds_dict = preds

            if preds_dict is not None:
                # Prefer targets provided by model output to avoid changing dataloaders
                if 'targets_tasks' in output_dict:
                    targets_dict = output_dict['targets_tasks']
                else:
                    # Fallback: split a concatenated target into two 2D tasks [pos(2), vel(2)]
                    targets_full = input_dict['target']
                    if targets_full.dim() == 2 and targets_full.shape[1] >= 4:
                        targets_dict = {
                            'pos': targets_full[:, 0:2],
                            'vel': targets_full[:, 2:4],
                        }
                if targets_dict is None:
                    raise ValueError("Multi-task loss requires 'targets_tasks' in model outputs or a 4-D concatenated target in input_dict['target'].")

                total_loss, per_task = self.loss_fn(preds_dict, targets_dict)  # type: ignore[arg-type]
                # Ensure tensor type
                if not torch.is_tensor(total_loss):
                    total_loss = torch.as_tensor(total_loss, device=self.get_device())
                # Compose loss dict compatible with epoch accumulation
                loss_dict = {
                    'loss': total_loss,
                    'size': targets_dict[next(iter(targets_dict))].shape[0]
                }
                # Include component losses for logging if desired
                for k, v in per_task.items():
                    try:
                        loss_dict[f'loss_{k}'] = torch.as_tensor(v, device=total_loss.device, dtype=total_loss.dtype)
                    except Exception:
                        pass
                return loss_dict

        # [BYPASSED] Align loss with metric definition for vector node regression:
        # when the task is vector-node regression and using MSE, compute the
        # mean of squared vector norms per node (sum over coordinates, mean over nodes),
        # which matches MultiTargetMSE(mode='vector').
        # if (
        #     ('reg' in self.task) and ('vector' in self.task) and ('node' in self.task)
        #     and (self.loss_fn is F.mse_loss)
        # ):
        #     diff = (preds.squeeze() - targets.squeeze())
        #     # sum across last dim (coordinates), mean across nodes/samples
        #     loss = (diff * diff).sum(dim=-1).mean()
        # else:
        loss = self.loss_fn(
            input=preds.squeeze(),
            target=targets.squeeze(),
            **self.loss_fn_kwargs 
        )
        # A custom loss function may return a dict instead of a scalar
        if isinstance(loss, dict):
            loss_dict = loss
        else:
            loss_dict = {
                'loss': loss,
                'size': targets.shape[0]
            }
        return loss_dict

    
    def loss(self, input_dict, output_dict, phase):
        """
        Simply grabs preds and targets from dictionary
        containers and calls '_loss', unless overridden by subclass.
        """
        loss_dict = self._loss(input_dict, output_dict, phase)
        return loss_dict

    
    def _set_up_metrics(self):
        """
        Convenience method to set output layer activation and 
        metrics based on model task type.
        """
        if 'cluster' in self.task:
            self._set_up_metrics_clustering()
        elif 'reg' in self.task:
            self._set_up_metrics_regression()
        elif 'class' in self.task:
            self._set_up_metrics_classification()
        else:
            raise NotImplementedError(
                f"Metrics for task='{self.task}' not yet implemented"
                f" in BaseModule!"
            )


    def _set_up_metrics_regression(self) -> None:
        """
        Initialize regression metrics (MSE/MAE and optional R^2), including
        node-level support via `MultiTargetMSE`. Also prepares container for
        optional per-task metric objects for multi-task models.
        """
        if 'node' in self.task:
            # Node-level: use custom MultiTargetMSE so we can average per-graph
            # (no MAE equivalent yet)

            # Legacy: use MultiTargetMSE for node-level regression on vector targets
            # self.mse = MultiTargetMSE(
            #     num_targets=self.target_dim,
            #     mode=('vector' if 'vector' in self.task else 'per_target')
            # )
            self.mse = MeanSquaredError(sync_on_compute=True)
        else:
            # Graph-level: standard torchmetrics
            if self.target_dim == 1:
                self.mse = MeanSquaredError(sync_on_compute=True)
                self.mae = MeanAbsoluteError(sync_on_compute=True)
            else:
                self.mse = MeanSquaredError(sync_on_compute=True)
                self.mae = MeanAbsoluteError(
                    num_outputs=self.target_dim,
                    sync_on_compute=True
                )

        # Optional R^2 metric (graph-level). For node-level, compute standard R^2 across nodes.
        use_r2 = bool(self.metrics_kwargs.get('use_r2', False))
        if use_r2:
            try:
                r2_kwargs = self._get_r2_kwargs(None if self.target_dim == 1 else self.target_dim)
                self.r2 = R2Score(**r2_kwargs)
            except Exception:
                # Final fallback: disable R^2 if unavailable in this environment
                self.r2 = None

        # Prepare container for optional per-task metrics (lazy init occurs on demand)
        self._task_metric_objs = {}


    def _set_up_metrics_clustering(self) -> None:
        """
        Initialize clustering metrics for supervised contrastive learning.
        Supports multiple clustering metrics such as Dunn Index and Silhouette
        score to measure clustering quality (higher is better for both).
        
        The DunnIndex metric is DDP-aware and handles gathering across ranks
        automatically when sync_on_compute=True.
        """
        # Determine which clustering metric to use. Default is Dunn Index for
        # backward compatibility. When metrics_kwargs['cluster_metric'] is set
        # to 'silhouette_score' or 'logistic_linear_accuracy', use the
        # corresponding custom metric instead.
        self.cluster_metric_name = self.metrics_kwargs.get('cluster_metric', 'svm_accuracy')
        cluster_metric_params = self.metrics_kwargs.get('cluster_metric_params', {})
        inferred_clusters = cluster_metric_params.get(
            'n_clusters',
            self.metrics_kwargs.get('num_classes', self.metrics_kwargs.get('num_outputs', 1)),
        )

        if self.cluster_metric_name == 'silhouette_score':
            # Optional: allow overriding the distance metric; default to 'cosine'.
            silhouette_metric = self.metrics_kwargs.get('silhouette_metric', 'cosine')
            self.cluster_metric = SilhouetteScore(metric=silhouette_metric)
        elif self.cluster_metric_name in ('svm_accuracy', 'svm'):
            self.cluster_metric = SVMAccuracy()
        elif self.cluster_metric_name in ('kmeans_accuracy', 'kmeans'):
            self.cluster_metric = KMeansAccuracy(
                n_clusters=inferred_clusters,
                n_init=cluster_metric_params.get('n_init', 10),
                max_iter=cluster_metric_params.get('max_iter', 300),
                random_state=cluster_metric_params.get('random_state'),
            )
        elif self.cluster_metric_name in ('spectral_clustering_accuracy', 'spectral'):
            self.cluster_metric = SpectralClusteringAccuracy(
                n_clusters=inferred_clusters,
                n_neighbors=cluster_metric_params.get('n_neighbors', 10),
                assign_labels=cluster_metric_params.get('assign_labels', 'cluster_qr'),
                n_jobs=cluster_metric_params.get('n_jobs', 1),
            )
        elif self.cluster_metric_name == 'logistic_linear_accuracy':
            self.cluster_metric = LogisticLinearAccuracy()
        else:
            # Initialize Dunn Index metric (DDP-aware).
            # p=2 is for Euclidean norm when metric is 'euclidean'
            # For 'cosine', p is not used
            # sync_on_compute=True ensures DDP gathering happens automatically
            self.cluster_metric = DunnIndex(
                p=DUNN_INDEX_P,
                # metric=distance_metric,  # not supported in torchmetrics DunnIndex
                sync_on_compute=True
            )
            # Preserve legacy attribute name for backward compatibility
            self.dunn_index = self.cluster_metric


    def _set_up_metrics_classification(self) -> None:
        """
        Initialize classification metrics for binary and multiclass tasks.
        """
        # Binary classification
        if 'bin' in self.task:
            self.accuracy = BinaryAccuracy(sync_on_compute=True)
            self.balanced_accuracy = Accuracy(
                task='multiclass', 
                num_classes=2, 
                average='macro',
                sync_on_compute=True
            )
            self.specificity = BinarySpecificity(sync_on_compute=True)
            self.f1 = BinaryF1Score(sync_on_compute=True)
            self.f1_neg = BinaryF1Score(sync_on_compute=True)
            self.auroc = BinaryAUROC(sync_on_compute=True)
            self.class_1_pred_ct = 0

        # Multi-class classification
        elif 'multi' in self.task:
            self.accuracy = Accuracy(
                task='multiclass',
                num_classes=self.metrics_kwargs['num_classes'],
                sync_on_compute=True
            )
            self.balanced_accuracy = Accuracy(
                task='multiclass', 
                num_classes=self.metrics_kwargs['num_classes'], 
                average='macro',
                sync_on_compute=True
            )


    def _get_r2_kwargs(
        self,
        num_outputs: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build kwargs for R2Score robustly across torchmetrics versions.

        If num_outputs is 1 or None, use single-output defaults; otherwise set
        num_outputs when supported and use uniform multioutput averaging when available.
        """
        import inspect
        r2_kwargs: Dict[str, Any] = {}
        sig = inspect.signature(R2Score.__init__)
        if 'sync_on_compute' in sig.parameters:
            r2_kwargs['sync_on_compute'] = True
        is_single = (num_outputs is None) or (num_outputs == 1)
        if is_single:
            if 'num_outputs' in sig.parameters:
                r2_kwargs['num_outputs'] = None
            if 'multioutput' in sig.parameters:
                r2_kwargs['multioutput'] = 'uniform_average'
        else:
            if 'num_outputs' in sig.parameters:
                r2_kwargs['num_outputs'] = num_outputs
            if 'multioutput' in sig.parameters:
                r2_kwargs['multioutput'] = 'uniform_average'
        return r2_kwargs


    def _init_multitask_reg_metrics(
        self,
        preds_tasks: dict,
        targets_tasks: dict | None = None,
    ) -> None:
        """
        Lazily initialize per-task regression metric objects for multi-task models.
        Initializes for any task present in `preds_tasks` (optionally intersected
        with `targets_tasks`).
        """
        if not hasattr(self, '_task_metric_objs'):
            self._task_metric_objs = {}
        for tname, p in preds_tasks.items():
            if (targets_tasks is not None) and (tname not in targets_tasks):
                continue
            if tname in self._task_metric_objs:
                continue
            # device = p.device if hasattr(p, 'device') else torch.device('cpu')
            _mse = MeanSquaredError(sync_on_compute=True).to(self.device)
            _mae = MeanAbsoluteError(sync_on_compute=True).to(self.device)
            # infer outputs
            _nout = p.shape[1] if (p.dim() == 2) else 1
            # Version-robust R2Score initialization (same as _set_up_metrics) via helper
            _r2 = None
            try:
                r2_kwargs = self._get_r2_kwargs(None if _nout == 1 else _nout)
                _r2 = R2Score(**r2_kwargs).to(self.device)
            except Exception as e:
                if self.verbosity > 0:
                    print(f"[DEBUG] Could not create R2Score for task '{tname}': {e}")
                _r2 = None
            self._task_metric_objs[tname] = {'mse': _mse, 'mae': _mae, 'r2': _r2}
            if self.verbosity > 1:
                print(f"[DEBUG] Created per-task metrics for '{tname}' with {_nout} outputs on {self.device} (R2: {_r2 is not None})")

    
    def update_metrics(
        self, 
        phase,
        loss_dict,
        input_dict = None, 
        output_dict = None
    ) -> None:
        if self.verbosity > 0 and phase == 'valid':
            print(f"[DEBUG] BaseModule.update_metrics: phase = {phase}")
        
        device = self.get_device()
        
        # on first call only: initialize loss counters
        if self.loss_keys is None:
            self.loss_keys = [
                k for k, v in loss_dict.items() \
                if 'loss' in k.lower()
            ]
            for k in self.epoch_loss_dict.keys():
                for loss_key in self.loss_keys:
                    self.epoch_loss_dict[k][loss_key] = 0.0
                    
        # Accumulate *sum* of loss, not mean, so final average is correct
        safe_loss_keys = self.loss_keys or ()
        for loss_key in safe_loss_keys:
            # loss_dict[loss_key] is a per-batch MEAN; multiply by batch size to get the sum
            self.epoch_loss_dict[phase][loss_key] += loss_dict[loss_key] * loss_dict['size']
        # Keep running total of number of samples seen
        self.epoch_loss_dict[phase]['size'] += loss_dict['size'] 

        # validation metrics
        if phase == 'valid':
            preds = output_dict['preds']
            # print('preds.shape:', preds.shape)
            target = input_dict['target']
            
            # 'target' may itself be a dict containing
            # multiple targets
            if (self.target_name is not None) \
            and (isinstance(target, dict)):
                target = target[self.target_name]
                # print('target:', target)
            
            if 'cluster' in self.task:
                # For clustering tasks (e.g., supervised contrastive learning),
                # update the configured clustering metric with batch embeddings
                # and cluster labels.
                embeddings = output_dict.get('embeddings', None)
                
                # Labels for clustering come from target (e.g., condition indices)
                # target is already extracted above from input_dict['target']
                labels = target
                
                metric_obj = getattr(self, 'cluster_metric', None)
                if (embeddings is not None) and (labels is not None) and (metric_obj is not None):
                    # Update metric – DunnIndex (DDP-aware) or SilhouetteScore wrapper.
                    metric_obj.update(embeddings, labels)
                elif self.verbosity > 0:
                    print(f"[DEBUG] update_metrics: clustering mode but missing embeddings or labels")
                    print(f"  embeddings present: {embeddings is not None}")
                    print(f"  labels (target) present: {labels is not None}")
                    print(f"  output_dict keys: {list(output_dict.keys())}")
                    print(f"  input_dict keys: {list(input_dict.keys())}")
                
            elif 'reg' in self.task:
                # If we normalized train set targets, de-normalize 
                # before computing metrics
                if self.has_normalized_train_targets:
                    # Model is trained to predict normalized targets...
                    if phase == 'train':
                        # For train set metrics (if calculated), both preds and 
                        # targets need de-normalization, since targets were 
                        # normalized during data loading
                        preds = self.get_denormalized(preds)
                        target = self.get_denormalized(target)
                    elif phase == 'valid':
                        # De-normalize predictions to match never-normalized targets
                        if self.verbosity > 0:
                            print(f"\tde-normalizing valid set preds")
                        preds = self.get_denormalized(preds)
                        # (targets are not normalized in valid set, so no 
                        # de-normalization needed)
                    else:
                        raise ValueError(f"Invalid phase: {phase}")
                
                # Squeeze after de-normalization (same order as loss calculation)
                if 'multi' not in self.task:
                    preds = preds.squeeze()
                    target = target.squeeze()
                    
                # if phase == 'valid' and self.verbosity > 0:
                #     preds_print = [f'{v:.2f}' for v in preds[:5].detach().cpu().tolist()]
                #     target_print = [f'{v:.2f}' for v in target[:5].detach().cpu().tolist()]
                #     print(f"\tpreds: {preds_print}")
                #     print(f"\ttarget: {target_print}")

                # print("update_metrics:")
                # print(f"\tpreds.shape: {preds.shape}")
                # print(f"\ttarget.shape: {target.shape}")
                # Optional node-level normalization by per-graph node counts
                node_counts = None
                batch_index = None
                if isinstance(input_dict, dict):
                    node_counts = input_dict.get('node_counts', None)
                    batch_index = input_dict.get('batch_index', None)
                if isinstance(self.mse, MultiTargetMSE):
                    self.mse.update(preds, target, batch_index=batch_index, node_counts=node_counts)
                else:
                    self.mse.update(preds, target)
                if hasattr(self, 'mae'):
                    self.mae.update(preds, target)
                if hasattr(self, 'r2'):
                    try:
                        self.r2.update(preds, target)
                    except Exception:
                        pass

                # Multi-task per-head metrics (if provided by model outputs)
                try:
                    preds_tasks = output_dict.get('preds_tasks', None) if isinstance(output_dict, dict) else None
                    targets_tasks = output_dict.get('targets_tasks', None) if isinstance(output_dict, dict) else None

                    if self.verbosity > 1 and phase == 'valid':
                        print(f"[DEBUG] update_metrics phase={phase}: preds_tasks type={type(preds_tasks)}, targets_tasks type={type(targets_tasks)}")
                        if preds_tasks is not None:
                            print(f"[DEBUG] preds_tasks keys: {list(preds_tasks.keys()) if isinstance(preds_tasks, dict) else 'not a dict'}")
                        if targets_tasks is not None:
                            print(f"[DEBUG] targets_tasks keys: {list(targets_tasks.keys()) if isinstance(targets_tasks, dict) else 'not a dict'}")

                    if (preds_tasks is not None) \
                    and (targets_tasks is not None):
                        # Lazy init of per-task metrics moved to helper
                        self._init_multitask_reg_metrics(preds_tasks, targets_tasks)
                        for tname, p in preds_tasks.items():
                            if tname not in targets_tasks:
                                continue
                            y = targets_tasks[tname]
                            objs = self._task_metric_objs.get(tname, {})
                            if 'mse' in objs:
                                objs['mse'].update(p, y)
                            if 'mae' in objs:
                                objs['mae'].update(p, y)
                            if ('r2' in objs) and (objs['r2'] is not None):
                                try:
                                    objs['r2'].update(p, y)
                                    if self.verbosity > 1 and phase == 'valid':
                                        print(f"[DEBUG] Updated R2 for task '{tname}': p.shape={p.shape}, y.shape={y.shape}")
                                except Exception as e:
                                    if self.verbosity > 1:
                                        print(f"[DEBUG] R2 update failed for task '{tname}': {e}")
                    elif self.verbosity > 1 and phase == 'valid':
                        print(f"[DEBUG] Per-task metrics skipped: preds_tasks={preds_tasks is not None}, targets_tasks={targets_tasks is not None}")
                except Exception as e:
                    if self.verbosity > 0:
                        print(f"[DEBUG] Error in per-task metric update: {e}")
                        import traceback
                        traceback.print_exc()
                
                # R^2
                # R2_score = self.R2_score.compute().detach().cpu().numpy()
                # if self.num_targets == 1:
                #     R2_score = R2_score.item()
                # metrics_dict = metrics_dict \
                #     | {(self.key_prefix + 'R2_valid'): R2_score}
                
            elif 'class' in self.task and 'bin' in self.task:
                # accuracy and f1
                # when using BCE with logits, need to convert
                # logit preds to 0 or 1 -> 
                # logit = log(p/(1-p)) -> logit>0 -> p>0.5 -> predicted '1'
                class_preds = torch.tensor(
                    [(logit > 0.0) for logit in preds],
                    dtype=torch.long,
                    device=device
                )
                class_targets = torch.tensor(
                    [int(t) for t in target],
                    dtype=torch.long,
                    device=device
                )
                self.accuracy.update(class_preds, class_targets)
                self.balanced_accuracy.update(class_preds, class_targets)
                self.f1.update(class_preds, class_targets)
                self.f1_neg.update(
                    torch.logical_not(class_preds).to(torch.long), 
                    torch.logical_not(class_targets).to(torch.long)
                )
                self.specificity.update(class_preds, class_targets)
                # auroc detects logits if preds are outside of [0, 1]
                self.auroc.update(preds, class_targets)
                self.class_1_pred_ct += torch.sum(preds > 0.).item()

            elif 'class' in self.task and 'multi' in self.task:
                class_preds = torch.argmax(preds, dim=1)
                print(f"class_preds.shape: {class_preds.shape}")
                print(f"target.shape: {target.shape}")
                class_targets = torch.tensor(
                    [int(t) for t in target],
                    dtype=torch.long,
                    device=device
                )
                self.accuracy.update(class_preds, class_targets)
                self.balanced_accuracy.update(class_preds, class_targets)
                

    def _calc_regression_metrics(
        self,
        metrics_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute and reset regression validation metrics, including optional
        per-task metrics for multi-task models.
        """
        # For node-level regression we use a custom metric that maintains
        # running numerator/denominator so we can safely aggregate across DDP ranks
        if isinstance(self.mse, MultiTargetMSE):
            # Capture local accumulators before reset
            local_numer = self.mse.sum_mse_across_graphs.detach().clone()
            local_denom = self.mse.graph_count.detach().clone().to(torch.float32)

            # Compute local mean only if we actually updated on this rank
            if local_denom.item() > 0:
                local_mean = (local_numer / local_denom).detach().cpu().numpy()
                if self.target_dim == 1:
                    local_mean = local_mean.item()
            else:
                # No updates on this rank; avoid calling compute() to prevent warnings
                # Report NaN locally; a proper global mean will be computed upstream using numer/denom
                local_mean = float('nan') if self.target_dim == 1 else (local_numer.detach().cpu().numpy() * float('nan'))

            # Stash accumulators for weighted cross-rank reduction in the trainer
            metrics_dict[self.key_prefix + 'mse_valid_numer'] = local_numer.detach().cpu().numpy().item()
            metrics_dict[self.key_prefix + 'mse_valid_denom'] = float(local_denom.item())

            # Coerce size-1 arrays to scalar for logging/printing compatibility
            try:
                if hasattr(local_mean, 'size') and getattr(local_mean, 'size', 0) == 1:
                    local_mean = local_mean.item()
            except Exception:
                pass
            metrics_dict = metrics_dict | {(self.key_prefix + 'mse_valid'): local_mean}
            self.mse.reset()
        else:
            # Graph-level regression: standard torchmetrics compute (already DDP-aware)
            mse_score = self.mse.compute().detach().cpu().numpy()
            if self.target_dim == 1:
                mse_score = mse_score.item()
            metrics_dict = metrics_dict | {(self.key_prefix + 'mse_valid'): mse_score}
            self.mse.reset()

        # MAE
        if hasattr(self, 'mae'):
            mae_score = self.mae.compute().detach().cpu().numpy()
            if self.target_dim == 1:
                mae_score = mae_score.item()
            metrics_dict = metrics_dict \
                | {(self.key_prefix + 'mae_valid'): mae_score}
            self.mae.reset()

        # R^2: if it doesn't work, skip
        if hasattr(self, 'r2'):
            try:
                r2_score_val = self.r2.compute().detach().cpu().numpy()
                if self.target_dim == 1:
                    r2_score_val = r2_score_val.item()
                metrics_dict = metrics_dict | {(self.key_prefix + 'r2_valid'): r2_score_val}
                self.r2.reset()
            except Exception:
                pass

        # Compute per-task metrics if present (fail fast on errors)
        if hasattr(self, '_task_metric_objs') \
        and isinstance(self._task_metric_objs, dict):
            if self.verbosity > 1:
                print(f"[DEBUG] calc_metrics: Found {len(self._task_metric_objs)} task metric objects: {list(self._task_metric_objs.keys())}")
            for tname, objs in self._task_metric_objs.items():
                if not isinstance(objs, dict):
                    continue
                if 'mse' in objs:
                    v = objs['mse'].compute().detach().cpu().numpy()
                    if hasattr(v, 'item'):
                        v = v.item()
                    metrics_dict[self.key_prefix + f'mse_valid_{tname}'] = v
                    objs['mse'].reset()
                if 'mae' in objs:
                    v = objs['mae'].compute().detach().cpu().numpy()
                    if hasattr(v, 'item'):
                        v = v.item()
                    metrics_dict[self.key_prefix + f'mae_valid_{tname}'] = v
                    objs['mae'].reset()
                if ('r2' in objs) and (objs['r2'] is not None):
                    v = objs['r2'].compute().detach().cpu().numpy()
                    if hasattr(v, 'item'):
                        v = v.item()
                    metrics_dict[self.key_prefix + f'r2_valid_{tname}'] = v
                    objs['r2'].reset()

        return metrics_dict


    def _calc_classification_metrics(
        self,
        metrics_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute and reset classification validation metrics for binary and
        multiclass tasks.
        """
        if 'bin' in self.task:
            accuracy_score = self.accuracy.compute().detach().cpu().numpy().item()
            bal_accuracy_score = self.balanced_accuracy.compute().detach().cpu().numpy().item()
            f1_score = self.f1.compute().detach().cpu().numpy().item()
            f1_neg_score = self.f1_neg.compute().detach().cpu().numpy().item()
            specificity_score = self.specificity.compute().detach().cpu().numpy().item()
            auroc_score = self.auroc.compute().detach().cpu().numpy().item()
            metrics_dict = metrics_dict \
                | {(self.key_prefix + 'accuracy_valid'): accuracy_score} \
                | {(self.key_prefix + 'bal_accuracy_valid'): bal_accuracy_score} \
                | {(self.key_prefix + 'f1_valid'): f1_score} \
                | {(self.key_prefix + 'f1_neg_valid'): f1_neg_score} \
                | {(self.key_prefix + 'specificity_valid'): specificity_score} \
                | {(self.key_prefix + 'auroc_valid'): auroc_score} \
                | {(self.key_prefix + 'class_1_pred_ct_valid'): self.class_1_pred_ct}
            self.accuracy.reset()
            self.balanced_accuracy.reset()
            self.f1.reset()
            self.f1_neg.reset()
            self.specificity.reset()
            self.auroc.reset()
            self.class_1_pred_ct = 0
            return metrics_dict

        if 'multi' in self.task:
            accuracy_score = self.accuracy.compute().detach().cpu().numpy().item()
            bal_accuracy_score = self.balanced_accuracy.compute().detach().cpu().numpy().item()
            metrics_dict = metrics_dict \
                | {(self.key_prefix + 'accuracy_valid'): accuracy_score} \
                | {(self.key_prefix + 'bal_accuracy_valid'): bal_accuracy_score}
            self.accuracy.reset()
            self.balanced_accuracy.reset()
            return metrics_dict

        return metrics_dict


    def _calc_clustering_metrics(
        self,
        metrics_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute and reset clustering validation metrics.
        
        Computes the configured clustering metric (Dunn Index by default, or
        Silhouette score when requested), which measures clustering quality
        (higher is better). The DunnIndex metric handles DDP gathering
        internally when sync_on_compute=True.
        """
        metric_obj = getattr(self, 'cluster_metric', None)
        if metric_obj is None:
            return metrics_dict

        # Compute metric (DunnIndex handles DDP gathering automatically; SilhouetteScore
        # operates on accumulated CPU tensors).
        metric_score = metric_obj.compute().detach().cpu().numpy()

        # Convert to scalar if needed
        if hasattr(metric_score, 'item'):
            metric_score = metric_score.item()

        # Determine the appropriate validation metric key based on the selected
        # clustering metric. Default to Dunn Index for backward compatibility.
        cluster_name = getattr(self, 'cluster_metric_name', 'dunn_index')
        # Map to canonical validation metric name (e.g., 'silhouette_score_valid')
        try:
            validation_key = MetricDefinitions.get_validation_metric_name(cluster_name)
        except Exception:
            # Fallback: append '_valid' if mapping is not defined
            validation_key = f"{cluster_name}_valid"

        metrics_dict[self.key_prefix + validation_key] = metric_score

        # Reset metric for next epoch
        metric_obj.reset()
        
        return metrics_dict


    def calc_metrics(
        self,
        epoch: int,
        is_validation_epoch: bool = True,
        # input_dict: Optional[Dict[str, Any]] = None, 
        # output_dict: Optional[Dict[str, Any]] = None, 
        # loss_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float | int]:

        phases = ('train', 'valid') if is_validation_epoch else ('train', )
        metrics_dict = {'epoch': epoch}
        
        # include train (and maybe validation) mean losses in metrics_dict
        loss_keys_iterable = self.loss_keys or ()
        for phase in phases:
            for loss_key in loss_keys_iterable:
                sample_count = self.epoch_loss_dict[phase]['size']
                # Avoid division-by-zero on ranks that saw zero batches
                # (e.g., small validation shard under DDP)
                if sample_count > 0:
                    avg_loss = self.epoch_loss_dict[phase][loss_key] / sample_count
                else:
                    # Use a zero tensor on the correct device to preserve dtype/device
                    avg_loss = torch.tensor(0.0, device=self.get_device())
                
                metrics_dict = metrics_dict \
                    | {(loss_key + '_' + phase): avg_loss.item()}

                # reset epoch loss to 0.
                self.epoch_loss_dict[phase][loss_key] = 0.
                
            self.epoch_loss_dict[phase]['size'] = 0.

        # in validation epochs, calc validation set metrics        
        if is_validation_epoch:
            if 'cluster' in self.task:
                metrics_dict = self._calc_clustering_metrics(metrics_dict)
            elif 'reg' in self.task:
                metrics_dict = self._calc_regression_metrics(metrics_dict)
            elif 'class' in self.task:
                metrics_dict = self._calc_classification_metrics(metrics_dict)
                
        return metrics_dict


    def on_fully_initialized_for_wandb(self, config=None):
        """
        Call this method after the model (and its MLP head, if any) is fully initialized and after wandb.init() (or acc.init_trackers(...)) has been called. This will call wandb.watch(self) only once, if wandb is available, logging is enabled, and wandb.run is not None.
        """
        if getattr(self, '_wandb_watched', False):
            print("[BaseModule] wandb.watch already called, skipping")
            return
        try:
            import wandb
            if config is not None \
            and getattr(config, 'use_wandb_logging', False) \
            and (getattr(wandb, 'run', None) is not None):
                wandb_log_freq = getattr(config, 'wandb_log_freq', 2048)
                print(f"[BaseModule] Calling wandb.watch with log_freq={wandb_log_freq}")
                wandb.watch(self, log='all', log_freq=wandb_log_freq)
                self._wandb_watched = True
                print("[BaseModule] wandb.watch called successfully")
            else:
                print(f"[BaseModule] wandb.watch conditions not met: config={config is not None}, use_wandb_logging={getattr(config, 'use_wandb_logging', False) if config else False}, wandb.run={getattr(wandb, 'run', None) is not None}")
        except Exception as e:
            print(f"[BaseModule] error using wandb; will not log to wandb: {e}")


    def on_best_model(self) -> None:
        """
        Overridable method to perform special
        methods whenever a new best model is
        achieved during training.
        """
        return None

    
    def visualize_embeddings(
        self,
        dataloader: Any,
        save_dir: str,
        labels_attr: Optional[str] = None,
    ) -> None:
        """
        Optional hook for visualizing learned embeddings after training.

        Subclasses can override this method to generate plots or other
        visualizations of embeddings computed on a given dataloader.

        Args:
            dataloader: DataLoader or iterable yielding batches for visualization.
            save_dir: Directory path where any visualization artifacts should be saved.
            labels_attr: Optional name of the attribute on each batch providing
                labels for coloring (for example, the dataset target key).
        """
        # Default implementation is a no-op; subclasses may override.
        return None


    def get_printable_metrics(self) -> List[str]:
        """
        Get the list of metrics that should be printed for this model's task.
        Filters out R² if not enabled on this model instance.
        
        Returns:
            List of metric names that should be printed during training/evaluation
        """
        metrics = MetricDefinitions.get_printable_metrics_for_task(self.task)
        
        # Filter out R² if not available on this model
        if 'reg' in self.task.lower():
            if not (hasattr(self, 'r2') and self.r2 is not None):
                metrics = [m for m in metrics if m != 'r2']
        
        return metrics


    def on_fully_initialized(self, config=None):
        """
        Call this method after the model (and its MLP head, if any) is fully initialized and after wandb.init() (or acc.init_trackers(...)) has been called. This will call any post-initialization hooks, such as wandb.watch.
        """
        self.on_fully_initialized_for_wandb(config)


    def run_epoch_zero_methods(self, Any):
        """
        Run any methods that need to be run at the start of the first epoch.
        """
        return None


    def get_denormalized(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization of targets using the stored statistics.

        The center / scale buffers might have shape (num_targets,) or
        (1, num_targets) (older checkpoints). To ensure broadcasting works
        regardless of dimensionality we flatten away any singleton
        dimensions.
        """
        if not self.has_normalized_train_targets:
            return tensor

        center = self._target_center.to(tensor.device)
        scale = self._target_scale.to(tensor.device)

        # Remove possible (leading) singleton dimensions, e.g. (1, D) -> (D,)
        if center.dim() > 1:
            center = center.squeeze()
        if scale.dim() > 1:
            scale = scale.squeeze()

        return tensor * scale + center


def test_nn(
    trained_model: BaseModule,
    data_container: Dict[str, DataLoader] | Data,
    task: str,
    device: str = 'cpu',
    target_name: str = 'target',
    set_key: str = 'test',
    metrics_kwargs: Dict[str, Any] = {},
    using_pytorch_geo: bool = False,
    accelerator: Optional[Any] = None,  # Add accelerator parameter
    verbosity: int = 0
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Computes standard regression or binary
    classification metrics for the 'test' set 
    in a data_container, given a trained 
    BaseModule (or extension).

    Args:
        trained_model: trained BaseModule model.
        task: string key coding model task, e.g.,
            'regression' or 'binary classification.'
        device: device key on which the tensors live, 
            e.g. 'cpu' or 'cuda.'
        target_name: string key for the model target.
        data_container: dictionary of Data-
            Loaders, with a keyed 'test' set, or
            a pytorch geometric Data object (e.g.
            of one graph, with train/valid/test masks).
        set_key: which set ('train'/'valid'/'test') to 
            compute metrics for (default: 'test').
        metrics_kwargs: kwargs for setting up metric calcu-
            lator objects, e.g., num_classes for multiclass
            accuracy.
        using_pytorch_geo: whether the DataLoaders
            hold PyTorch Geometric datasets (i.e.
            where data are loaded as sparse block-
            diagonal matrices, requiring batch indices).
        accelerator: Optional accelerator object for DDP synchronization.
        verbosity: verbosity level for debug print statements
    Returns:
        2-tuple: (1) dictionary of metric scores,
        and (2) dictionary of other task-specific
        metric objects (e.g. a confusion matrix
        calculator object for classification).
    """
    if verbosity > 2:
        if accelerator is not None:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Starting test_nn evaluation')
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Calculating metrics for evaluation ({set_key}) set:')
        else:
            print(f'[DEBUG] Calculating metrics for evaluation ({set_key}) set:')

    task = task.lower()
    num_targets = metrics_kwargs.get('num_outputs', 1)
    
    # record raw model preds and targets, for option
    # to calculate other metrics no calculated here
    auxiliary_metrics_dict = {
        'preds': [],
        'targets': []
    }
    
    # set up metrics collections based on task
    metric_collection = None
    r2_calculator = None
    if 'reg' in task:
        if ('node' in task):
            # Node-level: custom MSE (graph-aware)
            # No MAE equivalent yet
            mse_calculator = MultiTargetMSE(
                num_targets=num_targets,
                mode='vector' if 'vector' in task else 'per_target'
            )
            # no MAE equivalent yet
            # metric_collection = MetricCollection({'mae': mae_calculator}).to(device)
        else:
            # Graph-level: standard torchmetrics for both MSE and MAE
            if num_targets > 1:
                mae_calculator = MeanAbsoluteError(
                    num_outputs=num_targets,
                    sync_on_compute=True
                )
            else:
                mae_calculator = MeanAbsoluteError(sync_on_compute=True)
            mse_standard = MeanSquaredError(sync_on_compute=True)
            
            # Add R² calculator for regression tasks
            try:
                import inspect
                r2_kwargs = {}
                sig = inspect.signature(R2Score.__init__)
                if 'sync_on_compute' in sig.parameters:
                    r2_kwargs['sync_on_compute'] = True
                if num_targets == 1:
                    if 'num_outputs' in sig.parameters:
                        r2_kwargs['num_outputs'] = None
                    if 'multioutput' in sig.parameters:
                        r2_kwargs['multioutput'] = 'uniform_average'
                else:
                    if 'num_outputs' in sig.parameters:
                        r2_kwargs['num_outputs'] = num_targets
                    if 'multioutput' in sig.parameters:
                        r2_kwargs['multioutput'] = 'uniform_average'
                r2_calculator = R2Score(**r2_kwargs)
            except Exception:
                r2_calculator = None
            
            metric_dict = {
                'mse': mse_standard,
                'mae': mae_calculator,
            }
            if r2_calculator is not None:
                metric_dict['r2'] = r2_calculator
            
            metric_collection = MetricCollection(metric_dict).to(device)
        
    elif 'class' in task and 'bin' in task:
        metric_collection = MetricCollection({
            'acc': BinaryAccuracy(),
            'f1': BinaryF1Score(),
            'ppv': BinaryPrecision(),
            'sensitivity': BinaryRecall(),
            'specificity': BinarySpecificity(),
            'auroc': BinaryAUROC()
        }).to(device)

        # auxiliary metrics (not part of the MetricCollection)
        bal_acc_calculator = Accuracy(
            task='multiclass', 
            num_classes=2, 
            average='macro'
        ).to(device)
        f1_neg_calculator = BinaryF1Score().to(device)
        confusion_mat_calculator = BinaryConfusionMatrix().to(device)
        auxiliary_metrics_dict |= {
            'bal_acc': bal_acc_calculator,
            'f1_neg': f1_neg_calculator,
            'confusion_matrix': confusion_mat_calculator,
        }

    elif 'class' in task and 'multi' in task:
        metric_collection = MetricCollection({
            'acc': Accuracy(
                task='multiclass', 
                num_classes=metrics_kwargs['num_classes']
            ),
            'bal_acc': Accuracy(
                task='multiclass', 
                num_classes=metrics_kwargs['num_classes'], 
                average='macro'
            )
        }).to(device)

    def update(preds, targets, *, batch_obj=None) -> None:
        """
        Inner function to update metrics (in containers);
        called more than once if calculating in batches.
        """
        # --------------------------------------------------
        # Ensure targets match prediction dimensionality
        # --------------------------------------------------
        # Subset target tensor if extra dimensions are present
        if ('reg' in task) and (preds.dim() <= targets.dim()):
            tii = metrics_kwargs.get('target_include_indices', None)
            if (tii is not None) and (targets.dim() == 2):
                # Slice columns before any further processing
                targets = targets[:, tii]

        # Record predictions and (possibly sliced) targets for auxiliary output
        auxiliary_metrics_dict['preds'].append(preds)
        auxiliary_metrics_dict['targets'].append(targets)
        
        if 'reg' in task:
            if ('node' in task):
                # Optional node-level grouping for per-graph normalization
                batch_index = None
                node_counts = None
                if using_pytorch_geo and (batch_obj is not None) and hasattr(batch_obj, 'batch'):
                    batch_index = batch_obj.batch
                    node_counts = torch.bincount(batch_index)
                elif (batch_obj is not None) and hasattr(batch_obj, 'valid_mask'):
                    # Single-graph masked case (valid/test mask applied upstream when present)
                    mask_attr = 'test_mask' if 'test' in set_key else ('val_mask' if 'val' in set_key else 'train_mask')
                    if hasattr(batch_obj, mask_attr):
                        mask = getattr(batch_obj, mask_attr)
                        node_counts = mask.sum()
                # Update custom MSE with grouping info when available
                mse_calculator.update(preds, targets, batch_index=batch_index, node_counts=node_counts)
            
        elif 'class' in task and 'bin' in task:
            targets = targets.long()
            preds_binary = (preds > 0.).long()

            bal_acc_calculator.update(preds_binary, targets)
            f1_neg_calculator.update(
                torch.logical_not(preds_binary).to(torch.long),
                torch.logical_not(targets).to(torch.long)
            )
            confusion_mat_calculator.update(preds_binary, targets)

        elif 'class' in task and 'multi' in task:
            targets = targets.long()
            preds = torch.argmax(preds, dim=1)

        if metric_collection is not None:
            metric_collection.update(preds, targets.squeeze())

    # Ensure all processes are synchronized before evaluation
    if accelerator is not None:
        if verbosity > 2:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Waiting for all processes before evaluation')
        accelerator.wait_for_everyone()
        if verbosity > 2:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: All processes synchronized, starting evaluation')
    
    # get model predictions on test set
    trained_model.eval()
    with torch.set_grad_enabled(False):
        if isinstance(data_container, dict):
            if accelerator is not None and verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Starting batch processing')
            
            # For single-graph training, the dataloader dict only has 'train' key
            # (containing the single graph with train/valid/test masks)
            actual_key = set_key
            if set_key not in data_container and 'train' in data_container:
                # Single-graph mode: use 'train' key which contains the full graph
                actual_key = 'train'
            
            for batch_i, batch in enumerate(data_container[actual_key]):
                if accelerator is not None and batch_i % 10 == 0 and verbosity > 2:
                    accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Processing batch {batch_i}')
                
                if using_pytorch_geo:
                    data = batch.to(device)
                    batch_output_dict = trained_model(data)
                    preds = batch_output_dict['preds'].squeeze()
                    
                    # For single-graph with masks, use appropriate targets and masks
                    if (batch.num_graphs == 1):
                        mask_attr = f'{set_key}_mask'
                        # Handle 'valid' vs 'val' naming
                        if set_key == 'val' or set_key == 'valid':
                            mask_attr = 'valid_mask'
                        
                        if hasattr(batch, mask_attr):
                            mask = getattr(batch, mask_attr)
                            # Use batch.y which is updated to trajectory-level targets in forward pass
                            # for spatial graph pipeline, or node-level targets for node tasks
                            if hasattr(batch, 'y') and batch.y is not None:
                                targets = batch.y
                            else:
                                targets = batch[target_name]
                            preds = preds[mask]
                            targets = targets[mask]
                        else:
                            targets = batch[target_name]
                    else:
                        targets = batch[target_name]
                else:
                    if isinstance(batch, (tuple, list)):
                        features, targets = batch
                        features = features.to(device)
                        targets = targets.to(device).squeeze()
                        batch_output_dict = trained_model(features)
                        preds = batch_output_dict['preds'].squeeze() 
                    else:
                        batch_output_dict = trained_model(batch)
                        targets = batch[target_name]
                        if isinstance(targets, dict):
                            targets = targets[target_name]
                        targets = targets.squeeze()
                        preds = batch_output_dict['preds'].squeeze()
                
                # Unwrap model if it's a DDP model (wrapped by accelerate)
                if (accelerator is not None) and hasattr(trained_model, 'module'):
                    trained_model = trained_model.module

                if trained_model.has_normalized_train_targets:
                    preds = trained_model.get_denormalized(preds)
                    # note: targets are not normalized in test set
                update(preds, targets, batch_obj=batch if using_pytorch_geo else None)
                
                # Print progress only on main process
                if batch_i % 10 == 0 and verbosity > 2:
                    if (accelerator is not None) and accelerator.is_main_process:
                        accelerator.print(f"Processed {batch_i} batches...")
                    elif accelerator is None:
                        print(f"Processed {batch_i} batches...")

        elif using_pytorch_geo:
            if accelerator is not None and verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Processing single PyG graph')
            data = data_container.to(device)
            output_dict = trained_model(data)
            
            # Check if masks and targets are returned in output_dict (spatial graph pipeline)
            if 'train_mask' in output_dict \
            and 'valid_mask' in output_dict \
            and 'test_mask' in output_dict:
                # Spatial graph pipeline: predictions, targets, and masks are already at trajectory level
                preds = output_dict['preds'].squeeze()
                if 'targets' in output_dict:
                    targets = output_dict['targets']
                else:
                    targets = data[target_name]
                
                # Select appropriate mask
                if 'val' in set_key:
                    mask = output_dict['valid_mask']
                elif 'test' in set_key:
                    mask = output_dict['test_mask']
                elif 'train' in set_key:
                    mask = output_dict['train_mask']
                else:
                    mask = output_dict['test_mask']  # default
            else:
                # Standard path graph pipeline: use batch-level masks
                preds = output_dict['preds'].squeeze()
                targets = data[target_name]
                if 'val' in set_key:
                    mask = data.val_mask
                elif 'test' in set_key:
                    mask = data.test_mask
                elif 'train' in set_key:
                    mask = data.train_mask
                else:
                    mask = data.test_mask  # default
            
            if trained_model.has_normalized_train_targets:
                preds = trained_model.get_denormalized(preds)
                # targets are not normalized in test set
            update(preds[mask], targets[mask], batch_obj=data)
            
        # Ensure all processes have finished processing their batches
        if accelerator is not None:
            if verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Waiting for all processes to finish batch processing')
            accelerator.wait_for_everyone()
            if verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: All processes finished batch processing')
            
        # Compute metrics
        if accelerator is not None and verbosity > 2:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Computing metrics')
        if metric_collection is not None:
            metric_scores_dict = metric_collection.compute()
        else:
            metric_scores_dict = {}

        # Add custom MSE result for node-level regression; for graph-level it is already in metric_collection
        if 'reg' in task and ('node' in task):
            metric_scores_dict['mse'] = mse_calculator.compute()
        
        # Process auxiliary metrics
        if auxiliary_metrics_dict is not None:
            if accelerator is not None and verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Processing auxiliary metrics')
            for k, v in auxiliary_metrics_dict.items():
                if k in ('preds', 'targets'):
                    # Patch: handle empty list case for DDP
                    if len(v) == 0:
                        # Try to infer shape from metric_collection or metrics_kwargs
                        if 'reg' in task:
                            # Regression: shape (0, num_targets)
                            shape = (0, num_targets)
                            dtype = torch.float32
                        else:
                            # Classification: shape (0,)
                            shape = (0,)
                            dtype = torch.float32
                        device_ = device if isinstance(device, torch.device) else torch.device(device)
                        metric_scores_dict[k] = torch.empty(shape, dtype=dtype, device=device_)
                    else:
                        metric_scores_dict[k] = torch.cat(v)
                else:
                    metric_scores_dict[k] = v.compute()

        # For auxiliary metrics that aren't DDP-aware, we need to gather them
        if accelerator is not None and accelerator.num_processes > 1:
            if accelerator.is_main_process and verbosity > 2:
                accelerator.print(f'[DEBUG] Main process: Starting auxiliary metrics gathering')
            # Patch: ensure all processes have a tensor to gather, even if empty
            for k in ['preds', 'targets']:
                if k in metric_scores_dict:
                    tensor = metric_scores_dict[k]
                    if tensor is None or tensor.numel() == 0:
                        # Try to infer shape from num_targets
                        if 'reg' in task:
                            shape = (0, num_targets)
                            dtype = torch.float32
                        else:
                            shape = (0,)
                            dtype = torch.float32
                        device_ = device if isinstance(device, torch.device) else torch.device(device)
                        metric_scores_dict[k] = torch.empty(shape, dtype=dtype, device=device_)
            # Gather predictions and targets for auxiliary metrics
            gathered_preds = accelerator.gather(metric_scores_dict['preds'])
            gathered_targets = accelerator.gather(metric_scores_dict['targets'])
            
            # Only main process computes auxiliary metrics
            if accelerator.is_main_process and verbosity > 2:
                accelerator.print(f'[DEBUG] Main process: Computing auxiliary metrics on gathered data')
                
                # Update auxiliary metrics with full dataset
                if 'class' in task and 'bin' in task:
                    bal_acc_calculator.reset()
                    f1_neg_calculator.reset()
                    confusion_mat_calculator.reset()
                    
                    preds_binary = (gathered_preds > 0.).long()
                    bal_acc_calculator.update(preds_binary, gathered_targets)
                    f1_neg_calculator.update(
                        torch.logical_not(preds_binary).to(torch.long),
                        torch.logical_not(gathered_targets).to(torch.long)
                    )
                    confusion_mat_calculator.update(preds_binary, gathered_targets)
                    
                    # Update auxiliary metrics in results
                    metric_scores_dict['bal_acc'] = bal_acc_calculator.compute()
                    metric_scores_dict['f1_neg'] = f1_neg_calculator.compute()
                    metric_scores_dict['confusion_matrix'] = confusion_mat_calculator.compute()
                    metric_scores_dict['preds'] = gathered_preds
                    metric_scores_dict['targets'] = gathered_targets
            
            # Broadcast auxiliary metrics to all processes
            if accelerator.is_main_process and verbosity > 2:
                accelerator.print(f'[DEBUG] Main process: Broadcasting auxiliary metrics')
            for k in ['bal_acc', 'f1_neg', 'confusion_matrix', 'preds', 'targets']:
                if k in metric_scores_dict and isinstance(metric_scores_dict[k], torch.Tensor):
                    if dist.is_initialized():
                        dist.broadcast(metric_scores_dict[k], src=0)

        if accelerator is not None:
            if verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Evaluation complete')
            accelerator.print('\nDone calculating metrics.')
        else:
            print('\nDone calculating metrics.')
        
        return metric_scores_dict



def get_metric_direction(metric_name: str) -> str:
    """
    Get the optimization direction for a metric ('higher' or 'lower').
    
    Uses the METRIC_DIRECTION_MAP from class_maps.py to determine whether
    higher or lower values are better for a given metric.
    
    Args:
        metric_name: Name of the metric (e.g., 'mse', 'dunn_index', 'accuracy')
        
    Returns:
        'higher' or 'lower' indicating the optimization direction
        
    Raises:
        ValueError: If metric_name is not in the map
    """
    from models.class_maps import METRIC_DIRECTION_MAP
    
    # Try exact match first
    if metric_name in METRIC_DIRECTION_MAP:
        return METRIC_DIRECTION_MAP[metric_name]
    
    # Try with _valid suffix if not present
    if not metric_name.endswith('_valid'):
        candidate = metric_name + '_valid'
        if candidate in METRIC_DIRECTION_MAP:
            return METRIC_DIRECTION_MAP[candidate]
    
    # Try without _valid suffix if present
    if metric_name.endswith('_valid'):
        candidate = metric_name[:-6]
        if candidate in METRIC_DIRECTION_MAP:
            return METRIC_DIRECTION_MAP[candidate]
    
    raise ValueError(
        f"Metric '{metric_name}' not found in METRIC_DIRECTION_MAP. "
        f"Please add it to models/class_maps.py or specify main_metric_is_better explicitly."
    )


class MetricDefinitions:
    """
    Centralized definitions for metrics used across different tasks.
    Ensures consistency between training validation metrics and test metrics.
    
    Note: R² is included in the regression metrics list but will be filtered out
    by BaseModule.get_printable_metrics() if not enabled for a specific model instance
    (controlled by metrics_kwargs['use_r2'] during model initialization).
    """
    
    # Core metrics that should be printed for each task type
    # Note: 'r2' is conditional - only printed if enabled on the model
    PRINTABLE_METRICS = {
        'regression': [
            'mse', 'mae', 'r2'
        ],
        'binary_classification': [
            'acc', 'f1', 'sensitivity', 'specificity', 'auroc'
        ],
        'multiclass_classification': [
            'acc', 'bal_acc'
        ],
        'clustering': [
            'dunn_index',
            'silhouette_score',
            'svm_accuracy',
            'kmeans_accuracy',
            'knn_accuracy',
            'spectral_clustering_accuracy',
            'logistic_linear_accuracy',
        ]
    }
    
    # Mapping between different naming conventions used in calc_metrics vs test_nn
    METRIC_NAME_MAPPING = {
        # Standard name -> validation name (with _valid suffix)
        'acc': 'accuracy_valid',
        'bal_acc': 'bal_accuracy_valid', 
        'f1': 'f1_valid',
        'f1_neg': 'f1_neg_valid',
        'sensitivity': 'sensitivity_valid',  # Note: sensitivity = recall
        'specificity': 'specificity_valid',
        'auroc': 'auroc_valid',
        'mse': 'mse_valid',
        'mae': 'mae_valid',
        'r2': 'r2_valid',
        'dunn_index': 'dunn_index_valid',
        'silhouette_score': 'silhouette_score_valid',
        'logistic_linear_accuracy': 'logistic_linear_accuracy_valid',
        'svm_accuracy': 'svm_accuracy_valid',
        'kmeans_accuracy': 'kmeans_accuracy_valid',
        'knn_accuracy': 'knn_accuracy_valid',
        'spectral_clustering_accuracy': 'spectral_clustering_accuracy_valid',
    }
    
    @classmethod
    def get_printable_metrics_for_task(cls, task: str) -> List[str]:
        """
        Get the list of metrics that should be printed for a given task.
        
        Args:
            task: Task string (e.g., 'regression', 'binary_classification', 'clustering')
            
        Returns:
            List of metric names to print
        """
        task_lower = task.lower()
        
        if 'cluster' in task_lower:
            return cls.PRINTABLE_METRICS['clustering']
        elif 'reg' in task_lower:
            return cls.PRINTABLE_METRICS['regression']
        elif 'class' in task_lower and 'bin' in task_lower:
            return cls.PRINTABLE_METRICS['binary_classification'] 
        elif 'class' in task_lower and 'multi' in task_lower:
            return cls.PRINTABLE_METRICS['multiclass_classification']
        else:
            # Return empty list for unknown tasks
            return []
    
    @classmethod
    def get_validation_metric_name(cls, metric_name: str) -> str:
        """
        Get the validation metric name (with _valid suffix) for a given metric.
        
        Args:
            metric_name: Standard metric name (e.g., 'acc', 'mse')
            
        Returns:
            Validation metric name (e.g., 'accuracy_valid', 'mse_valid')
        """
        return cls.METRIC_NAME_MAPPING.get(metric_name, metric_name + '_valid')
