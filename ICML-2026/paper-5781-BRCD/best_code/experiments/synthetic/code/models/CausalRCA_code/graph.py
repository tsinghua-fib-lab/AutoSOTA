"""
Define the interface for algorithms to access relations
"""
from abc import ABC
from typing import Dict
from typing import List
from typing import Set
from typing import Union

import networkx as nx
import json

ENCODING = "UTF-8"
def dump_json(filename: str, data):
    """
    Dump data into a json file
    """
    with open(filename, "w", encoding=ENCODING) as obj:
        json.dump(data, obj, ensure_ascii=False, indent=2)


def load_json(filename: str):
    """
    Load data from a json file
    """
    with open(filename, encoding=ENCODING) as obj:
        return json.load(obj)



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
        # pylint: disable=no-self-use, unused-argument
        """
        Dump a graph into the given file

        Return whether the operation succeeds
        """
        return False

    @classmethod
    def load(cls, filename: str) -> Union["Graph", None]:
        # pylint: disable=unused-argument
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
    def topological_sort(self) -> List[Set[Node]]:
        """
        Sort nodes with parents first

        The graph specifies the parents of each node.
        """
        if self._sorted_nodes:
            return self._sorted_nodes

        degrees = {node: len(self.parents(node)) for node in self.nodes}

        nodes: List[Set[Node]] = []
        while degrees:
            minimum = min(degrees.values())
            node_set = {node for node, degree in degrees.items() if degree == minimum}
            nodes.append(node_set)
            for node in node_set:
                degrees.pop(node)
                for child in self.children(node):
                    if child in degrees:
                        degrees[child] -= 1

        self._sorted_nodes = nodes
        return nodes

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
        edges = [
            (node_indexes[cause], node_indexes[effect])
            for cause, effect in self._graph.edges
        ]
        data = dict(nodes=[node.asdict() for node in nodes], edges=edges)
        dump_json(filename=filename, data=data)

    @classmethod
    def load(cls, filename: str) -> Union["MemoryGraph", None]:
        data: dict = load_json(filename=filename)
        if "nodes" not in data or "edges" not in data:
            raise LoadingInvalidGraphException(filename)
        nodes: List[Node] = [Node(**node) for node in data["nodes"]]
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(
            (nodes[cause], nodes[effect]) for cause, effect in data["edges"]
        )
        return MemoryGraph(graph)

    def children(self, node: Node, **kwargs) -> Set[Node]:
        if not self._graph.has_node(node):
            return set()
        return set(self._graph.successors(node))

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        if not self._graph.has_node(node):
            return set()
        return set(self._graph.predecessors(node))
