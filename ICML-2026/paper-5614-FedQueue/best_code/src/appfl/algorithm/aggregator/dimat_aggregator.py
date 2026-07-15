"""
DIMATaggregator: Server-side aggregator implementing the DIMAT algorithm
(Decentralized Iterative Merging-And-Training) for federated learning.

DIMAT aligns neural network feature spaces via activation matching before
merging models, solving the permutation symmetry problem. This aggregator
performs the activation-matching merge on the server using all client models,
producing a single global model broadcast to all clients.

Reference:
    DIMAT: Decentralized Iterative Merging-And-Training for Deep Learning Models
    (CVPR 2024)
"""

import copy
import importlib
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

from appfl.algorithm.aggregator import BaseAggregator


class DIMATaggregator(BaseAggregator):
    """
    DIMAT aggregator that performs activation-matching model merging.

    Required aggregator_kwargs in YAML config:
        graph_func (str): Name of graph function (e.g., "resnet18_appfl", "resnet20",
            "resnet50", "vgg16").
        proxy_dataset_path (str): Path to Python file containing proxy dataset loader.
        proxy_dataset_name (str): Function name in that file that returns a Dataset.
        proxy_batch_size (int): Batch size for proxy dataloader. Default: 64.
        device (str): Device for merge computation. Default: "cpu".

    Optional aggregator_kwargs:
        match_func (str): Matching function name. Default: "match_tensors_permute".
            Options: "match_tensors_permute", "match_tensors_zipit",
                     "match_tensors_optimal", "match_tensors_identity".
        transform_kwargs (dict): Extra kwargs for the matching function.
            Default: {} (use matching function defaults).
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs

        # Required configs
        self.graph_func_name = aggregator_configs.get("graph_func", "resnet18_appfl")
        self.match_func_name = aggregator_configs.get(
            "match_func", "match_tensors_permute"
        )
        self.device = aggregator_configs.get("device", "cpu")
        self.proxy_batch_size = aggregator_configs.get("proxy_batch_size", 64)
        self.transform_kwargs = dict(aggregator_configs.get("transform_kwargs", {}))

        # Global state dict after aggregation
        self.global_state = None

        # Lazy-loaded proxy dataloader and function references
        self._proxy_dataloader = None
        self._graph_func = None
        self._match_func = None

    def _get_graph_func(self):
        """Lazy-load the graph function."""
        if self._graph_func is None:
            from appfl.misc.dimat_utils.resnet_graph import (
                resnet20,
                resnet50,
                resnet18,
                resnet18_appfl,
            )
            from appfl.misc.dimat_utils.vgg_graph import vgg11, vgg16
            from appfl.misc.dimat_utils.cnn_graph import cnn

            graph_registry = {
                "resnet20": resnet20,
                "resnet50": resnet50,
                "resnet18": resnet18,
                "resnet18_appfl": resnet18_appfl,
                "vgg11": vgg11,
                "vgg16": vgg16,
                "cnn": cnn,
            }
            if self.graph_func_name not in graph_registry:
                raise ValueError(
                    f"Unknown graph function '{self.graph_func_name}'. "
                    f"Available: {list(graph_registry.keys())}"
                )
            self._graph_func = graph_registry[self.graph_func_name]
        return self._graph_func

    def _get_match_func(self):
        """Lazy-load the matching function."""
        if self._match_func is None:
            from appfl.misc.dimat_utils.matching_functions import (
                match_tensors_permute,
                match_tensors_zipit,
                match_tensors_optimal,
                match_tensors_identity,
            )

            match_registry = {
                "match_tensors_permute": match_tensors_permute,
                "match_tensors_zipit": match_tensors_zipit,
                "match_tensors_optimal": match_tensors_optimal,
                "match_tensors_identity": match_tensors_identity,
            }
            if self.match_func_name not in match_registry:
                raise ValueError(
                    f"Unknown match function '{self.match_func_name}'. "
                    f"Available: {list(match_registry.keys())}"
                )
            self._match_func = match_registry[self.match_func_name]
        return self._match_func

    def _get_proxy_dataloader(self):
        """Lazy-load the proxy dataset and create a DataLoader."""
        if self._proxy_dataloader is None:
            proxy_dataset_path = self.aggregator_configs.get("proxy_dataset_path", None)
            proxy_dataset_name = self.aggregator_configs.get("proxy_dataset_name", None)
            proxy_dataset_kwargs = dict(
                self.aggregator_configs.get("proxy_dataset_kwargs", {})
            )

            if proxy_dataset_path is None or proxy_dataset_name is None:
                raise ValueError(
                    "DIMATaggregator requires 'proxy_dataset_path' and "
                    "'proxy_dataset_name' in aggregator_kwargs."
                )

            spec = importlib.util.spec_from_file_location(
                "proxy_dataset_module", proxy_dataset_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            dataset_fn = getattr(module, proxy_dataset_name)
            dataset = dataset_fn(**proxy_dataset_kwargs)

            self._proxy_dataloader = DataLoader(
                dataset,
                batch_size=self.proxy_batch_size,
                shuffle=False,
                drop_last=False,
            )
        return self._proxy_dataloader

    def get_parameters(self, **kwargs) -> Dict:
        """Return global model state dict."""
        if self.global_state is not None:
            return self.global_state
        elif self.model is not None:
            return self.model.state_dict()
        else:
            raise ValueError("DIMATaggregator has no model or global state to return.")

    def aggregate(self, local_models, **kwargs) -> Dict:
        """
        Aggregate local models using DIMAT activation-matching merge.

        All client models are merged into a single global model using
        activation matching to align feature spaces before averaging.

        Args:
            local_models: dict mapping client_id -> state_dict

        Returns:
            Single merged state dict (broadcast to all clients).
        """
        from appfl.misc.dimat_utils.model_merger import ModelMerge
        from appfl.misc.dimat_utils.am_utils import reset_bn_stats

        if self.model is None:
            raise ValueError("DIMATaggregator requires a model to be provided.")

        graph_func = self._get_graph_func()
        match_func = self._get_match_func()
        proxy_dataloader = self._get_proxy_dataloader()

        # 1. Reconstruct full models from state dicts
        models = []
        for client_id, state_dict in local_models.items():
            model_copy = copy.deepcopy(self.model)
            model_copy.load_state_dict(state_dict, strict=False)
            models.append(model_copy)

        if self.logger:
            self.logger.info(
                f"[DIMAT] Merging {len(models)} client models using "
                f"graph={self.graph_func_name}, match={self.match_func_name}"
            )

        # 2. Reset BN stats on all models using proxy dataloader
        for model in models:
            model.to(self.device)
            reset_bn_stats(model, proxy_dataloader)

        # 3. Create graph representations for each model
        graphs = [graph_func(model).graphify() for model in models]

        # 4. Create merged model template
        merged_model = copy.deepcopy(self.model).to(self.device)

        # 5. Create ModelMerge and run the full merge pipeline
        num_models = len(models)
        interp_w = [1.0 / num_models] * num_models

        merger = ModelMerge(*graphs, device=self.device)
        merger.transform(
            merged_model,
            proxy_dataloader,
            transform_fn=match_func,
            interp_w=interp_w,
            **self.transform_kwargs,
        )

        # 6. Reset BN stats on merged model
        reset_bn_stats(merger.merged_model, proxy_dataloader)

        # 7. Extract merged state dict
        self.global_state = {
            k: v.cpu() for k, v in merger.merged_model.state_dict().items()
        }

        # Clean up
        merger.clear_hooks()
        del merger, graphs, models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.global_state
