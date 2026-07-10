from collections import defaultdict
from Graph.GraphClass_PAG import Node
from typing import Optional
import warnings
class Separation_Set:

    def __init__(self, node_set: set[Node]):
        """
        Initialize the Separation_Set with a set of nodes.
        """
        if not node_set:
            warnings.warn("node_set must be a non-empty set of nodes.", UserWarning)
        
        self.node_set = node_set
        self.sepset = defaultdict(set)  # Each key maps directly to a separation set

    def _ensure_order(self, node1: Node, node2: Node) -> tuple[Node, Node]:
        """
        Ensure consistent ordering of node pairs.
        """
        return (node1, node2) if node1.index < node2.index else (node2, node1)

    def _add(self, node1: Node, node2: Node, sepset: set[Node]):
        """
        Add a separation set for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        self.sepset[(node1, node2)] = sepset

    def has_sepset(self, node1: Node, node2: Node) -> bool:
        """
        Check if a separation set exists for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        return (node1, node2) in self.sepset
    
    def get_sepset(self, node1: Node, node2: Node) -> set[Node]:
        node1, node2 = self._ensure_order(node1, node2)
        if (node1, node2) in self.sepset:
            return self.sepset[(node1, node2)]
        else:
            return None

    def is_in_sepset(self, target: Node, node1: Node, node2: Node) -> Optional[bool]:
        """
        Check if a target node is in the separation set of the given pair of nodes.

        Returns None if no separation set exists for the pair.
        
        """
        node1, node2 = self._ensure_order(node1, node2)
        if not self.has_sepset(node1, node2):
            return None
        else:
            return target in self.sepset[(node1, node2)]

    def __repr__(self):
        """
        Return a string representation of the separation set dictionary.
        """
        result = []
        for (node1, node2), sepset in self.sepset.items():
            result.append(f"{node1} - {node2}: {sepset}")
        return "\n".join(result)


if __name__ == "__main__":
    # Example usage

    node_set = {Node("A", 0), Node("B", 1), Node("C", 2), Node("D", 3)}
    sepset = Separation_Set(node_set)
    sepset._add(Node("A", 0), Node("B", 1), {Node("C", 2)})
    sepset._add(Node("A", 0), Node("C", 2), {Node("D", 3)})
    print(sepset)  # Outputs: A - B: {C}, A - C: {D}

    print(sepset.is_in_sepset(Node("A", 0),  Node("C", 2),  Node("B", 1)))  # Outputs: False