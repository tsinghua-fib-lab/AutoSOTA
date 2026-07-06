"""
This file is inspired from the implementation of DESP to get the ancestor/descandant nodes in the graph
"""

import networkx as nx


class AndOrGraph(nx.DiGraph):
    """
    A directed graph subclassed from networkx
    """

    def __init__(self):
        super().__init__()

    def get_ancestors(self, node):
        """
        Get all ancestors of a node in the graph.

        Parameters
        ----------
        node : Node
            The node for which to find ancestors.
        """
        ancestors = nx.ancestors(self, node)
        ancestors = [ancestor.smiles for ancestor in ancestors] + [node.smiles]
        return ancestors

    def get_descendants(self, node):
        """
        Get all descendants of a node in the graph.

        Parameters
        ----------
        node : Node
            The node for which to find descendants.
        """
        descendants = nx.descendants(self, node)
        descendants = [descendant.smiles for descendant in descendants] + [node.smiles]
        return descendants
