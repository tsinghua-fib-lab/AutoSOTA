from abc import ABC
from itertools import chain
from typing import Callable, Dict, List, Set, Union

import networkx as nx
import numpy as np

from RCAEval.graph_heads import finalize_directed_adj
from RCAEval.utility import dump_json, load_json


def topological_sort(
    nodes: Set,
    predecessors: Callable,
    successors: Callable,
) -> List[Set]:
    """
    Sort nodes with predecessors first
    """
    graph = {node: set(successors(node)) for node in nodes}
    components = list(nx.strongly_connected_components(nx.DiGraph(graph)))
    node2component = {
        node: index for index, component in enumerate(components) for node in component
    }
    super_graph = {
        index: {node2component[child] for node in component for child in graph[node]} - {index}
        for index, component in enumerate(components)
    }
    return [
        set(chain(*[components[index] for index in layer]))
        for layer in nx.topological_generations(nx.DiGraph(super_graph))
    ]


class Node:
    """
    The element of a graph
    """

    def __init__(self, entity: str, metric: str):
        self._entity = entity
        self._metric = metric

    @property
    def entity(self) -> str:
        """
        Entity getter
        """
        return self._entity

    @property
    def metric(self) -> str:
        """
        Metric getter
        """
        return self._metric

    def asdict(self) -> Dict[str, str]:
        """
        Serialized as a dict
        """
        return {"entity": self._entity, "metric": self._metric}

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Node):
            return self.entity == obj.entity and self.metric == obj.metric
        return False

    def __hash__(self) -> int:
        return hash((self.entity, self.metric))

    def __repr__(self) -> str:
        return f"Node{(self.entity, self.metric)}"


class LoadingInvalidGraphException(Exception):
    """
    This exception indicates that Graph tries to load from a broken file
    """


class Graph(ABC):
    """
    The abstract interface to access relations
    """

    def __init__(self):
        self._nodes: Set[Node] = set()
        self._sorted_nodes: List[Set[Node]] = None

    def dump(self, filename: str) -> bool:
        """
        Dump a graph into the given file

        Return whether the operation succeeds
        """
        return False

    @classmethod
    def load(cls, filename: str) -> Union["Graph", None]:
        """
        Load a graph from the given file

        Returns:
        - A graph, if available
        - None, if dump/load is not supported
        - Raise LoadingInvalidGraphException if the file cannot be parsed
        """
        return None

    @property
    def nodes(self) -> Set[Node]:
        """
        Get the set of nodes in the graph
        """
        return self._nodes

    @property
    def edges(self) -> Set[tuple]:
        return [(i, j) for i, j in self._graph.edges]

    @property
    def str_edges(self) -> Set[tuple]:
        try:
            return [
                (f"{i.entity}_{i.metric}", f"{j.entity}_{j.metric}") for i, j in self._graph.edges
            ]
        except Exception:
            return self._graph.edges

    @property
    def topological_sort(self) -> List[Set[Node]]:
        """
        Sort nodes with parents first

        The graph specifies the parents of each node.
        """
        if self._sorted_nodes is None:
            self._sorted_nodes = topological_sort(
                nodes=self.nodes, predecessors=self.parents, successors=self.children
            )
        return self._sorted_nodes

    def children(self, node: Node, **kwargs) -> Set[Node]:
        """
        Get the children of the given node in the graph
        """
        raise NotImplementedError

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        """
        Get the parents of the given node in the graph
        """
        raise NotImplementedError


class MemoryGraph(Graph):
    """
    Implement Graph with data in memory
    """

    def __init__(self, graph: nx.DiGraph):
        """
        graph: The whole graph
        """
        super().__init__()
        self._graph = graph
        self._nodes.update(self._graph.nodes)

    def dump(self, filename: str) -> bool:
        nodes: List[Node] = list(self._graph.nodes)
        node_indexes = {node: index for index, node in enumerate(nodes)}
        edges = [(node_indexes[cause], node_indexes[effect]) for cause, effect in self._graph.edges]
        try:
            data = dict(nodes=[node.asdict() for node in nodes], edges=edges)
        except Exception:
            data = dict(nodes=[node for node in nodes], edges=edges)
        dump_json(filename=filename, data=data)

    @classmethod
    def load(cls, filename: str) -> Union["MemoryGraph", None]:
        data: dict = load_json(filename=filename)
        if "nodes" not in data or "edges" not in data:
            raise LoadingInvalidGraphException(filename)
        try:
            nodes: List[Node] = [Node(**node) for node in data["nodes"]]
        except Exception:
            nodes = [node for node in data["nodes"]]

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((nodes[cause], nodes[effect]) for cause, effect in data["edges"])
        return MemoryGraph(graph)

    @classmethod
    def from_adj(cls, adj: np.ndarray, nodes: List[Node]) -> Union["MemoryGraph", None]:
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        if isinstance(adj, list) and len(adj) == 0:
            return MemoryGraph(graph)

        adj = finalize_directed_adj(adj)

        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i, j] == 1:
                    graph.add_edge(nodes[j], nodes[i])

        return MemoryGraph(graph)

    def children(self, node: Node, **kwargs) -> Set[Node]:
        if not self._graph.has_node(node):
            return set()
        return set(self._graph.successors(node))

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        if not self._graph.has_node(node):
            return set()
        return set(self._graph.predecessors(node))