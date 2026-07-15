"""
Model merging orchestration for DIMAT algorithm.
Ported from DIMAT/utils/model_merger.py.
"""

import torch
from torch import nn
from time import time
from tqdm.auto import tqdm

from .base_graph import NodeType
from .metric_calculators import CovarianceMetric, MeanMetric
from .matching_functions import match_tensors_zipit


class MergeHandler:
    """
    Handles all (un)merge transformations on top of a graph architecture.
    merge/unmerge is a dict whose keys are graph nodes and values are
    merges/unmerges to be applied at the graph node.
    """

    def __init__(self, graph, merge, unmerge):
        self.graph = graph
        self.module_handlers = {
            "BatchNorm2d": self.handle_batchnorm2d,
            "Conv2d": self.handle_conv2d,
            "Linear": self.handle_linear,
            "LayerNorm": self.handle_layernorm,
            "GELU": self.handle_fn,
            "AdaptiveAvgPool2d": self.handle_fn,
            "LeakyReLU": self.handle_fn,
            "ReLU": self.handle_fn,
            "Tanh": self.handle_fn,
            "MaxPool2d": self.handle_fn,
            "AvgPool2d": self.handle_fn,
            "SpaceInterceptor": self.handle_linear,
            "Identity": self.handle_fn,
        }

        self.merge = merge
        self.unmerge = unmerge

    def handle_batchnorm2d(self, forward, node, module):
        """Apply (un)merge operation to batchnorm parameters."""
        if forward:
            for parameter_name in ["weight", "bias", "running_mean", "running_var"]:
                parameter = getattr(module, parameter_name)
                merge = self.merge
                parameter.data = merge @ parameter

            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            assert len(self.graph.preds(node)) == 1, "BN expects one predecessor"
            self.prop_back(self.graph.preds(node)[0])

    def handle_layernorm(self, forward, node, module):
        """Apply (un)merge operation to layernorm parameters."""
        if forward:
            parameter_names = ["weight", "bias"]
            for parameter_name in parameter_names:
                parameter = getattr(module, parameter_name)
                parameter.data = self.merge @ parameter

            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            assert len(self.graph.preds(node)) == 1, "LN expects one predecessor"
            self.prop_back(self.graph.preds(node)[0])

    def handle_fn(self, forward, node, module):
        """Apply (un)merge operation to parameterless layers."""
        if forward:
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
        else:
            assert len(self.graph.preds(node)) == 1, (
                "Function node expects one predecessor"
            )
            self.prop_back(self.graph.preds(node)[0])

    def handle_conv2d(self, forward, node, module):
        """Apply (un)merge operation to conv2d layer parameters."""
        if forward:  # unmerge
            module.weight.data = torch.einsum(
                "OIHW,IU->OUHW", module.weight, self.unmerge
            )
        else:  # merge
            module.weight.data = torch.einsum(
                "UO,OIHW->UIHW", self.merge, module.weight
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data = self.merge @ module.bias

    def handle_linear(self, forward, node, module):
        """Apply (un)merge operation to linear layer parameters."""
        if forward:  # unmerge
            module.weight.data = module.weight @ self.unmerge
        else:
            info = self.graph.get_node_info(node)

            lower = 0
            upper = module.weight.shape[0]

            if info["chunk"] is not None:
                idx, num_chunks = info["chunk"]
                chunk_size = upper // num_chunks

                lower = idx * chunk_size
                upper = (idx + 1) * chunk_size

            module.weight.data[lower:upper] = self.merge @ module.weight[lower:upper]
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data[lower:upper] = self.merge @ module.bias[lower:upper]

    def prop_back(self, node):
        """Propagate (un)merge metrics backwards through a node graph."""
        if node in self.graph.merged:
            return

        info = self.graph.get_node_info(node)
        self.graph.merged.add(node)

        for succ in self.graph.succs(node):
            self.prop_forward(succ)

        if info["type"] in (NodeType.OUTPUT, NodeType.INPUT):
            raise RuntimeError(
                f"Unexpectedly reached node type {info['type']} when merging."
            )
        elif info["type"] == NodeType.CONCAT:
            merge = self.merge.chunk(len(self.graph.preds(node)), dim=1)
            for pred, m in zip(self.graph.preds(node), merge):
                MergeHandler(self.graph, m, self.unmerge).prop_back(pred)
        elif info["type"] == NodeType.MODULE:
            module = self.graph.get_module(info["layer"])
            self.module_handlers[module.__class__.__name__](False, node, module)
        elif info["type"] == NodeType.EMBEDDING:
            param = self.graph.get_parameter(info["param"])
            self.handle_embedding(False, node, param)
        else:
            # Default case (also for SUM)
            for pred in self.graph.preds(node):
                self.prop_back(pred)

    def prop_forward(self, node):
        """Propagate (un)merge transformations up a network graph."""
        if node in self.graph.unmerged:
            return

        info = self.graph.get_node_info(node)
        self.graph.unmerged.add(node)

        if info["type"] in (NodeType.OUTPUT, NodeType.INPUT):
            raise RuntimeError(
                f"Unexpectedly reached node type {info['type']} when unmerging."
            )
        elif info["type"] == NodeType.MODULE:
            module = self.graph.get_module(info["layer"])
            self.module_handlers[module.__class__.__name__](True, node, module)
        elif info["type"] == NodeType.SUM:
            for succ in self.graph.succs(node):
                self.prop_forward(succ)
            for pred in self.graph.preds(node):
                self.prop_back(pred)
        elif info["type"] == NodeType.CONCAT:
            num_to_concat = len(self.graph.preds(node))

            if node not in self.graph.working_info:
                self.graph.working_info[node] = []
            self.graph.working_info[node].append(self.unmerge)

            if len(self.graph.working_info[node]) < num_to_concat:
                self.graph.unmerged.remove(node)
            else:
                unmerge = torch.block_diag(*self.graph.working_info[node])
                del self.graph.working_info[node]

                new_handler = MergeHandler(self.graph, self.merge, unmerge)
                for succ in self.graph.succs(node):
                    new_handler.prop_forward(succ)


class ModelMerge(nn.Module):
    """
    Handles all merge operations for zipping arbitrary numbers of models.
    Expects a list of architecture graphs (one per model).
    """

    def __init__(self, *graphs, device=0):
        super().__init__()

        self.stop_at = None
        self.start_at = None

        self.hooks = []

        self.init(graphs, device)

    def init(self, graphs, device):
        """Initialize merge attributes with new set of graphs."""
        for g in graphs:
            g.model.to(device).eval()

        self.graphs = graphs
        self.device = device

        self.merged_model = None
        self.head_models = nn.ModuleList([g.model for g in self.graphs])
        for graph in self.graphs:
            graph.add_hooks(device=device)

    def compute_metrics(self, dataloader, metric_classes, covsave_path):
        """
        Compute pairwise alignment metrics between all graph models.
        Returns: dictionary of graph nodes to metrics computed at those nodes.
        """
        self.metrics = None
        if not isinstance(dataloader, list):
            dataloader_list = [dataloader]
        else:
            dataloader_list = dataloader

        numel = 0
        for dataloader in dataloader_list:
            for x, _ in tqdm(dataloader, desc="Forward Pass to Compute Merge Metrics"):
                x = x.to(self.device)

                numel += x.shape[0]
                intermediates = [g.compute_intermediates(x) for g in self.graphs]
                nodes = list(intermediates[0].keys())
                if self.metrics is None:
                    self.metrics = {
                        n: {k: v() for k, v in metric_classes.items()} for n in nodes
                    }

                for node, node_metrics in self.metrics.items():
                    for metric in node_metrics.values():
                        intermeds_float = [i[node].float() for i in intermediates]
                        metric.update(x.shape[0], *intermeds_float)

        for node, node_metrics in self.metrics.items():
            for metric_name, metric in node_metrics.items():
                self.metrics[node][metric_name] = metric.finalize(
                    numel, covsave_path, node
                )

        return self.metrics

    def compute_transformations(
        self, transform_fn, corrsave_path, reduce_ratio=0.5, **kwargs
    ):
        """
        Compute merge/unmerge transformations at each PREFIX and POSTFIX node.
        """
        start_time = time()
        self.merges = {}
        self.unmerges = {}

        nodes = list(self.metrics.keys())
        nodes.sort()

        for node in tqdm(nodes, desc="Computing transformations"):
            if self.start_at is None or node >= self.start_at:
                metric = self.metrics[node]
                info = self.graphs[0].get_node_info(node)
                if info["special_merge"] is not None:
                    merge, unmerge = _get_merging_fn(info["special_merge"])(
                        metric, reduce_ratio, **kwargs
                    )
                else:
                    merge, unmerge = transform_fn(
                        metric, corrsave_path, node, reduce_ratio, **kwargs
                    )

                # Hack to deal with things not merged
                merge = merge * len(self.graphs)

                self.merges[node] = merge.chunk(len(self.graphs), dim=1)
                self.unmerges[node] = unmerge.chunk(len(self.graphs), dim=0)

                if self.stop_at is not None and node == self.stop_at:
                    break

        self.compute_transform_time = time() - start_time
        return self.merges, self.unmerges

    def apply_transformations(self):
        """Apply transformations found by compute_transformations."""
        for node in self.merges:
            merges = self.merges[node]
            unmerges = self.unmerges[node]
            for merge, unmerge, graph in zip(merges, unmerges, self.graphs):
                merger = MergeHandler(graph, merge, unmerge)
                merger.prop_back(node)

    def get_merged_state_dict(self, interp_w=None):
        """
        Post transformations, obtain state dictionary for merged model by
        linearly interpolating between transformed models in each graph.
        """
        state_dict = {}
        merged_state_dict = self.merged_model.state_dict()
        keys = list(self.graphs[0].model.state_dict().keys())
        try:
            for key in keys:
                if key in merged_state_dict:
                    param = self.graphs[0].model.state_dict()[key]
                    if (
                        interp_w is not None
                        and param.shape == merged_state_dict[key].shape
                    ):
                        new_value = sum(
                            graph.model.state_dict()[key] * w
                            for graph, w in zip(self.graphs, interp_w)
                        )
                    else:
                        new_value = sum(
                            graph.model.state_dict()[key] for graph in self.graphs
                        )
                    state_dict[key] = new_value
        except RuntimeError as e:
            if "size" not in str(e):
                raise e
        return state_dict

    def clear_hooks(self):
        """Clears all hooks from graphs."""
        for g in self.graphs:
            g.clear_hooks()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def transform(
        self,
        model,
        dataloader,
        covsave_path="",
        corrsave_path="",
        metric_classes=(CovarianceMetric, MeanMetric),
        transform_fn=match_tensors_zipit,
        stop_at=None,
        start_at=None,
        interp_w=None,
        **transform_kwargs,
    ):
        """
        Full merge pipeline. Note: this consumes the models given to the graphs.
        """
        self.stop_at = stop_at
        self.start_at = start_at
        self.merged_model = model.to(self.device)

        if not isinstance(metric_classes, dict):
            metric_classes = {x.name: x for x in metric_classes}

        self.metric_classes = metric_classes
        self.transform_fn = transform_fn

        self.compute_metrics(
            dataloader, metric_classes=metric_classes, covsave_path=covsave_path
        )
        self.compute_transformations(
            transform_fn,
            corrsave_path=corrsave_path,
            reduce_ratio=1 - 1.0 / len(self.graphs),
            **transform_kwargs,
        )
        self.apply_transformations()

        self.merged_model.load_state_dict(
            self.get_merged_state_dict(interp_w), strict=False
        )

        # Remove hooks after transformation
        self.clear_hooks()

    def forward(self, x):
        """Evaluate the merged model."""
        return self.merged_model(x)


def _get_merging_fn(name):
    """Get alignment function from name."""
    from . import matching_functions

    from inspect import getmembers, isfunction

    matching_fns = {
        k: v
        for (k, v) in getmembers(matching_functions, isfunction)
        if "match_tensors" in k
    }
    return matching_fns[name]
