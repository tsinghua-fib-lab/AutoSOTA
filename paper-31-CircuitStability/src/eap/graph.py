from typing import List, Dict, Union, Tuple, Literal, Optional, Set
from collections import defaultdict
from pathlib import Path
import json
import heapq

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np
import pygraphviz as pgv

from .visualization import EDGE_TYPE_COLORS, generate_random_color


class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs,
    while the out_hook gets its outputs.
    """

    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set["Node"]
    parent_edges: Set["Edge"]
    children: Set["Node"]
    child_edges: Set["Edge"]
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(
        self,
        name: str,
        layer: int,
        in_hook: List[str],
        out_hook: str,
        index: Tuple,
        qkv_inputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Node({self.name}, in_graph: {self.in_graph})"

    def __hash__(self):
        return hash(self.name)


class LogitNode(Node):
    def __init__(self, n_layers: int):
        name = "logits"
        index = slice(None)
        super().__init__(
            name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", "", index
        )


class MLPNode(Node):
    def __init__(self, layer: int):
        name = f"m{layer}"
        index = slice(None)
        super().__init__(
            name,
            layer,
            f"blocks.{layer}.hook_mlp_in",
            f"blocks.{layer}.hook_mlp_out",
            index,
        )


class AttentionNode(Node):
    head: int
    kv_head: int

    def __init__(self, layer: int, head: int, cfg: Dict):
        name = f"a{layer}.h{head}"
        self.head = head
        self.kv_head = head // (cfg["n_heads"] // cfg["n_kv_heads"])
        index = (slice(None), slice(None), head)
        super().__init__(
            name,
            layer,
            f"blocks.{layer}.hook_attn_in",
            f"blocks.{layer}.attn.hook_result",
            index,
            [f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"],
        )


class InputNode(Node):
    def __init__(self):
        name = "input"
        index = slice(None)
        super().__init__(
            name, 0, "", "hook_embed", index
        )  # "blocks.0.hook_resid_pre", index)


class Edge:
    """An Edge in a graph.
    Attributes:
        name: (str): the edge's name, given as [PARENT]->[CHILD]<[OPTIONAL QKV>]; the latter applies only if [CHILD] is an AttentionNode
        parent: (Node): the parent node of the edge
        child: (Node): the child node of the edge
        hook: (str): the hook into the child node
        index: (Tuple): the index of the child node (only really relevant for AttentionNodes)
        score: (Optional[float]): the score of the edge (given by an attribution method)
        in_graph: (bool): whether the edge is in the graph or not"""

    name: str
    parent: Node
    child: Node
    hook: str
    index: Tuple
    score: Optional[float]
    in_graph: bool

    def __init__(
        self,
        parent: Node,
        child: Node,
        qkv: Union[None, Literal["q"], Literal["k"], Literal["v"]] = None,
    ):
        self.name = (
            f"{parent.name}->{child.name}"
            if qkv is None
            else f"{parent.name}->{child.name}<{qkv}>"
        )
        self.parent = parent
        self.child = child
        self.qkv = qkv
        self.score = None
        self.in_graph = True
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(
                    f"Edge({self.name}): Edges to attention heads must have a non-none value for qkv."
                )
            self.hook = f"blocks.{child.layer}.hook_{qkv}_input"
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook

    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF0000"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})"

    def __hash__(self):
        return hash(self.name)


class Graph:
    """
    Represents a graph that consists of nodes and edges.

    Attributes:
        nodes (Dict[str, Node]): A dictionary of nodes in the graph, where the key is the node name and the value is the node object.
        edges (Dict[str, Edge]): A dictionary of edges in the graph, where the key is the edge name and the value is the edge object.
        n_forward (int): The number of forward nodes in the graph, i.e. the # of nodes whose output activations we care about
        n_backward (int): The number of backward nodes/indices in the graph, i.e. the # of nodes whose input gradients we care about. Note that attention heads have 3 inputs that need to be dealt with during a backward pass
        cfg (HookedTransformerConfig): The configuration object for the graph.
    """

    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    n_forward: int
    n_backward: int
    cfg: HookedTransformerConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(
        self,
        parent: Node,
        child: Node,
        qkv: Union[None, Literal["q"], Literal["k"], Literal["v"]] = None,
    ):
        edge = Edge(parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def prev_index(self, node: Node) -> Union[int, slice]:
        """Return the forward index before which all nodes contribute to the input of the given node
        Args:
            node (Node): The node to get the prev forward index of

        Returns:
            Union[int, slice]: an index representing the prev forward index of the node
        """
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            if self.cfg["parallel_attn_mlp"]:
                return 1 + node.layer * (self.cfg["n_heads"] + 1)
            else:
                return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        elif isinstance(node, AttentionNode):
            i = 1 + node.layer * (self.cfg["n_heads"] + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def forward_index(self, node: Node, attn_slice=True):
        """Return the forward index of the given node. Every node but the logits has a forward index, representing where in a tensor of n_forward x d_model activations the node's activations are stored.
        Args:
            node (Node): The node to get the forward index of
            attn_slice (bool): whether to return a slice for the attention heads or an integer for the node
        Returns:
            Union[int, slice]: an index representing the forward index of the node"""
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            raise ValueError("No forward index for logits node")
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        elif isinstance(node, AttentionNode):
            i = 1 + node.layer * (self.cfg["n_heads"] + 1)
            return slice(i, i + self.cfg["n_heads"]) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def backward_index(self, node: Node, qkv=None, attn_slice=True):
        """Return the backward index of the given node. Every node but the input node has a backward index, representing where in a tensor of n_backward x d_model activations the node's activations are stored. Note that because attention heads have 3 inputs, they correspondingly 3 backward indices, depending on which of q, k, or v you want.
        Args:
            node (Node): The node to get the backward index of
            qkv (str): which of q, k, or v to get the backward index of
            attn_slice (bool): whether to return a slice for the attention heads or an integer for the node
        Returns:
            Union[int, slice]: an index representing the backward index of the node"""
        if isinstance(node, InputNode):
            raise ValueError(f"No backward for input node")
        elif isinstance(node, LogitNode):
            return self.n_backward - 1
        elif isinstance(node, MLPNode):
            total_heads_per_layer = self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"]
            mlp_index = (
                (node.layer) * total_heads_per_layer
                + self.cfg["n_heads"]
                + 2 * self.cfg["n_kv_heads"]
            )
            assert (
                mlp_index < self.n_backward
            ), f"MLP index {mlp_index} >= n_backward {self.n_backward}"
            return (
                (node.layer) * total_heads_per_layer
                + self.cfg["n_heads"]
                + 2 * self.cfg["n_kv_heads"]
            )
        elif isinstance(node, AttentionNode):
            assert qkv in "qkv", f"Must give qkv for AttentionNode, but got {qkv}"
            total_heads_per_layer = self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"]
            layer_offset = node.layer * total_heads_per_layer

            if qkv == "q":
                # Query heads remain the same
                i = layer_offset
                slice_size = self.cfg["n_heads"]
            else:
                # Key and value heads are grouped
                kv_offset = (
                    self.cfg["n_heads"]
                    if qkv == "k"
                    else (self.cfg["n_heads"] + self.cfg["n_kv_heads"])
                )
                i = layer_offset + kv_offset
                slice_size = self.cfg["n_kv_heads"]
            final_index = i + (
                slice_size
                if attn_slice
                else (node.head if qkv == "q" else node.kv_head)
            )
            assert (
                final_index < self.n_backward
            ), f"Attention index {final_index} >= n_backward {self.n_backward}"
            return (
                slice(i, i + slice_size)
                if attn_slice
                else i + (node.head if qkv == "q" else node.kv_head)
            )
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def get_scores(self, nonzero=False, in_graph=False, sort=True):
        """Return the scores of the edges in the graph
        Args:
            nonzero (bool): whether to return only the nonzero scores
            in_graph (bool): whether to return only the scores of the edges that are in the graph
            sort (bool): whether to sort the scores before returning them
        Returns:
            torch.Tensor: a tensor of the scores of the edges in the graph"""
        s = (
            torch.tensor(
                [
                    edge.score
                    for edge in self.edges.values()
                    if edge.score != 0 and (edge.in_graph or not in_graph)
                ]
            )
            if nonzero
            else torch.tensor([edge.score for edge in self.edges.values()])
        )
        return torch.sort(s).values if sort else s

    def get_dst_nodes(self):
        heads = []
        for layer in range(self.cfg["n_layers"]):
            for letter in "qkv":
                for attention_head in range(self.cfg["n_heads"]):
                    heads.append(f"a{layer}.h{attention_head}<{letter}>")
            heads.append(f"m{layer}")
        heads.append("logits")
        return heads

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())

    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())

    def apply_threshold(self, threshold: float, absolute: bool):
        """Apply a threshold to the graph, setting the in_graph attribute of edges to True if the score is above the threshold
        Args:
            threshold (float): the threshold to apply
            absolute (bool): whether to take the absolute value of the scores before applying the threshold"""
        threshold = float(threshold)
        for node in self.nodes.values():
            node.in_graph = True

        for edge in self.edges.values():
            edge.in_graph = (
                abs(edge.score) >= threshold if absolute else edge.score >= threshold
            )

    def apply_topn(self, n: int, absolute: bool):
        """Apply a top n filter to the graph, setting the in_graph attribute of the n edges with the highest scores to True
        Args:
            n (int): the number of edges to include
            absolute (bool): whether to take the absolute value of the scores before applying the threshold"""
        a = abs if absolute else lambda x: x
        for node in self.nodes.values():
            node.in_graph = False

        sorted_edges = sorted(
            list(self.edges.values()), key=lambda edge: a(edge.score), reverse=True
        )
        for edge in sorted_edges[:n]:
            edge.in_graph = True
            edge.parent.in_graph = True
            edge.child.in_graph = True

        for edge in sorted_edges[n:]:
            edge.in_graph = False

    def apply_greedy(self, n_edges, reset=True, absolute: bool = True):
        """Perform a greedy search on the graph, starting from the logits node and selecting the reachable edge with the highest score at each step (of n_edges). An edge is reachable if its child is in the graph; if an edge is selected but its parent is not in the graph, the parent is added.
        Args:
            n_edges (int): the number of edges to include
            reset (bool): whether to reset the in_graph attribute of all nodes and edges before applying the greedy search (defaults to True, you probably want to keep it that way)
            absolute (bool): whether to take the absolute value of the scores before applying the threshold"""
        if reset:
            for node in self.nodes.values():
                node.in_graph = False
            for edge in self.edges.values():
                edge.in_graph = False
            self.nodes["logits"].in_graph = True

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted(
            [edge for edge in self.edges.values() if edge.child.in_graph],
            key=lambda edge: abs_id(edge.score),
            reverse=True,
        )

        edges = heapq.merge(
            candidate_edges, key=lambda edge: abs_id(edge.score), reverse=True
        )
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted(
                    [parent_edge for parent_edge in parent.parent_edges],
                    key=lambda edge: abs_id(edge.score),
                    reverse=True,
                )
                edges = heapq.merge(
                    edges,
                    parent_parent_edges,
                    key=lambda edge: abs_id(edge.score),
                    reverse=True,
                )

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        self.nodes["logits"].in_graph = any(
            parent_edge.in_graph for parent_edge in self.nodes["logits"].parent_edges
        )

        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue

            if any(child_edge.in_graph for child_edge in node.child_edges):
                node.in_graph = True
            else:
                if prune_childless:
                    node.in_graph = False
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else:
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if (
                    not isinstance(node, InputNode)
                    and node.in_graph
                    and not any(
                        parent_edge.in_graph for parent_edge in node.parent_edges
                    )
                ):
                    node.in_graph = False
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False

    @classmethod
    def from_model(
        cls, model_or_config: Union[HookedTransformer, HookedTransformerConfig, Dict]
    ):
        graph = Graph()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg

            nkv_heads = getattr(cfg, "n_key_value_heads", cfg.n_heads)
            if nkv_heads is None:
                nkv_heads = cfg.n_heads

            graph.cfg = {
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "parallel_attn_mlp": cfg.parallel_attn_mlp,
                "n_kv_heads": nkv_heads,  # Add GQA support
            }
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config

            nkv_heads = getattr(cfg, "n_key_value_heads", cfg.n_heads)
            if nkv_heads is None:
                nkv_heads = cfg.n_heads

            graph.cfg = {
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "parallel_attn_mlp": cfg.parallel_attn_mlp,
                "n_kv_heads": nkv_heads,  # Add GQA support
            }
        else:
            graph.cfg = model_or_config

        input_node = InputNode()
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg["n_layers"]):
            attn_nodes = [
                AttentionNode(layer, head, graph.cfg)
                for head in range(graph.cfg["n_heads"])
            ]
            mlp_node = MLPNode(layer)

            for attn_node in attn_nodes:
                graph.nodes[attn_node.name] = attn_node
            graph.nodes[mlp_node.name] = mlp_node

            if graph.cfg["parallel_attn_mlp"]:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in "qkv":
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)

                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in "qkv":
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg["n_layers"])
        for node in residual_stream:
            graph.add_edge(node, logit_node)

        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg["n_layers"] * (graph.cfg["n_heads"] + 1)
        heads_per_layer = graph.cfg["n_heads"] + 2 * graph.cfg["n_kv_heads"] + 1
        graph.n_backward = (graph.cfg["n_layers"] * heads_per_layer) + 1
        # Verify the size is sufficient
        max_index_needed = max(
            graph.cfg["n_heads"] - 1,  # Max Q index
            graph.cfg["n_kv_heads"] - 1,  # Max K/V index
        )
        assert (
            graph.cfg["n_layers"] - 1
        ) * heads_per_layer + max_index_needed < graph.n_backward, (
            "n_backward calculation insufficient"
        )

        return graph

    def edge_matrices(self):
        edge_scores = torch.zeros((self.n_forward, self.n_backward))
        edges_in_graph = torch.zeros((self.n_forward, self.n_backward)).bool()
        for edge in self.edges.values():
            edge_scores[
                self.forward_index(edge.parent, attn_slice=False),
                self.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ] = float(edge.score)
            edges_in_graph[
                self.forward_index(edge.parent, attn_slice=False),
                self.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
            ] = edge.in_graph

        return edge_scores, edges_in_graph

    def to_json(self, filename: str):
        """Save the graph to a json file
        Args:
            filename (str): the filename to save the graph to"""
        # non serializable info
        d = {
            "cfg": self.cfg,
            "nodes": {
                str(name): bool(node.in_graph) for name, node in self.nodes.items()
            },
            "edges": {
                str(name): {
                    "score": None if edge.score is None else float(edge.score),
                    "in_graph": bool(edge.in_graph),
                }
                for name, edge in self.edges.items()
            },
        }
        with open(filename, "w") as f:
            json.dump(d, f)

    def to_pt(self, filename: str):
        """Save the graph to a torch file
        Args:
            filename (str): the filename to save the graph to"""
        src_nodes = {
            node.name: node.in_graph
            for node in self.nodes.values()
            if not isinstance(node, LogitNode)
        }
        dst_nodes = self.get_dst_nodes()
        edge_scores, edges_in_graph = self.edge_matrices()
        d = {
            "cfg": self.cfg,
            "src_nodes": src_nodes,
            "dst_nodes": dst_nodes,
            "edges": edge_scores,
            "edges_in_graph": edges_in_graph,
        }
        torch.save(d, filename)

    @classmethod
    def from_json(cls, filename):
        """
        Load a Graph object from a JSON file.
        The JSON should have the following keys:
        1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
        2. 'nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
        3. 'edges': Dict[str, Dict] which maps an edge name ('node->node') to a dictionary contains values
        """
        with open(filename, "r") as f:
            d = json.load(f)
        g = Graph.from_model(d["cfg"])
        for name, in_graph in d["nodes"].items():
            g.nodes[name].in_graph = in_graph

        for name, info in d["edges"].items():
            g.edges[name].score = info["score"]
            g.edges[name].in_graph = info["in_graph"]

        return g

    @classmethod
    def from_pt(cls, filename):
        """Load a graph object from a pytorch-serialized file.
        The file should contain a dict with the following items -
        1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
        2. 'src_nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
        3. 'dst_nodes': List[str] containing the names of the possible destination nodes, in the same order as the edges tensor.
        4. 'edges': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents the edge score between the src node and dst node.
        5. 'edges_in_graph': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents if the edge is in the graph or not.
        """
        d = torch.load(filename)
        assert all(
            [
                k in d.keys()
                for k in ["cfg", "src_nodes", "dst_nodes", "edges", "edges_in_graph"]
            ]
        ), f"Bad torch circuit file format. Found keys - {d.keys()}, missing keys - {set(['cfg', 'src_nodes', 'dst_nodes', 'edges', 'edges_in_graph']) - set(d.keys())}"
        assert (
            d["edges"].shape
            == d["edges_in_graph"].shape
            == (len(d["src_nodes"]), len(d["dst_nodes"]))
        ), "Bad edges array shape"

        g = Graph.from_model(d["cfg"])

        for name, in_graph in d["src_nodes"].items():
            g.nodes[name].in_graph = in_graph

        # Enumerate over the tensor and fill the edge values in the graph
        for src_idx, src_name in enumerate(d["src_nodes"]):
            for dst_idx, dst_name in enumerate(d["dst_nodes"]):
                edge_name = f"{src_name}->{dst_name}"
                if edge_name in g.edges.keys():
                    g.edges[edge_name].score = d["edges"][src_idx, dst_idx]
                    g.edges[edge_name].in_graph = d["edges_in_graph"][src_idx, dst_idx]

        return g

    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (
            set(self.edges.keys()) == set(other.edges.keys())
        )
        if not keys_equal:
            return False

        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False

        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(
                edge.score, other.edges[name].score
            ):
                return False
        return True

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str = "dot",
        seed: Optional[int] = None,
    ) -> pgv.AGraph:
        """
        Convert the graph to a pygraphviz graph object for visualization.
        Colorscheme: a cmap colorscheme
        """
        g = pgv.AGraph(
            directed=True,
            bgcolor="white",
            overlap="false",
            splines="true",
            layout=layout,
        )

        if seed is not None:
            np.random.seed(seed)

        colors = {
            node.name: generate_random_color(colorscheme)
            for node in self.nodes.values()
        }

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(
                    node.name,
                    fillcolor=colors[node.name],
                    color="black",
                    style="filled, rounded",
                    shape="box",
                    fontname="Helvetica",
                )

        scores = self.get_scores().abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if edge.score is None else edge.score
                normalized_score = (
                    (abs(score) - min_score) / (max_score - min_score)
                    if max_score != min_score
                    else abs(score)
                )
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(
                    edge.parent.name,
                    edge.child.name,
                    penwidth=str(penwidth),
                    color=edge.get_color(),
                )
        return g
