"""
Implements a recursive tree data structure for internal use.

"""

# Copyright (C) 2020
# Benjamin Paaßen
# The University of Sydney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class Tree:
    """ Models a tree as a recursive data structure with a label and
    children.

    Attributes
    ----------
    _label: str
        The label of this tree.
    _children: list
        The children of this tree. Should be a list of trees.

    """
    def __init__(self, label, children = None):
        self._label = label
        if(children is None):
            self._children = []
        else:
            self._children = children

    def __str__(self):
        out = str(self._label)
        if(not self._children):
            return out
        else:
            child_strs = [str(child) for child in self._children]
            return out + '(%s)' % ', '.join(child_strs)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Tree):
            return self._label == other._label and self._children == other._children
        return False

    def to_list_format(self):
        """ Convers this tree to node list/adjacency list format via depth
        first search.

        Returns
        -------
        nodes: list
            the node list.
        adj: list
            the adjacency list.

        """
        # initialize node and adjacency list
        nodes = []
        adj   = []
        # perform the depth first search via a stack which stores
        # the parent index and the current tree
        stk = [(-1, self)]
        while(stk):
            p, tree = stk.pop()
            i = len(nodes)
            # append the label to the node list
            nodes.append(tree._label)
            # append a new empty child list
            adj.append([])
            # append the current node to its parent
            if(p >= 0):
                adj[p].append(i)
            # push the children onto the stack
            for c in range(len(tree._children)-1, -1, -1):
                if(tree._children[c] is None):
                    continue
                if(isinstance(tree._children[c], list)):
                    for c2 in range(len(tree._children[c]) -1, -1, -1):
                        stk.append((i, tree._children[c][c2]))
                    continue
                stk.append((i, tree._children[c]))
        return nodes, adj

# This code is copied from
# https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances/-/blob/master/edist/tree_utils.py

def root(adj):
    """ Returns the root of a tree and raises an error if the input adjacency
    matrix does not correspond to a tree.

    Parameters
    ----------
    adj: list
        a directed graph in adjacency list format.

    Returns
    -------
    root: int
        the index of the root of the tree.

    Raises
    ------
    ValueError
        if the given adjacency list does not form a tree.

    """
    if(not adj):
        raise ValueError("The input tree is empty!")

    par = np.full(len(adj), -1, dtype=int)
    for i in range(len(adj)):
        for j in adj[i]:
            if(par[j] < 0):
                par[j] = i
            else:
                raise ValueError("Input is not a tree because node %d has multiple parents" % j)
    root = -1
    for i in range(len(adj)):
        if(par[i] < 0):
            if(root < 0):
                root = i
            else:
                raise ValueError("Input is not a tree because there is more than one root")
    return root


def tree_to_string(nodes, adj, indent = False, with_indices = False):
    """ Prints a tree in node list/adjacency list format as string.

    Parameters
    ----------
    nodes: list
        The node list of the tree.
    adj: list
        The adjacency list of the tree.
    indent: bool (default = False)
        A boolean flag; if True, each node is printed on a new line.
    with_indices: bool (default = False)
        A boolean flag; if True, each node is printed with its index.

    Raises
    ------
    ValueError
        if the adjacency list does not correspond to a tree.

    """
    r = root(adj)
    if(indent):
        indent = 1
    else:
        indent = None
    # translate recursively
    return _tree_to_string(nodes, adj, r, indent, with_indices = with_indices)

def _tree_to_string(nodes, adj, i, indent = None, with_indices = False):
    # the initial string is just the node label
    if(with_indices):
        tree_string = '%d: %s' % (i, str(nodes[i]))
    else:
        tree_string = str(nodes[i])
    # consider the case where node i has chidlren
    if(adj[i]):
        # first, translate the children to their tree strings
        children_strings = []
        for j in adj[i]:
            if(indent is None):
                children_strings.append(_tree_to_string(nodes, adj, j, with_indices = with_indices))
            else:
                children_strings.append(_tree_to_string(nodes, adj, j, indent + 1, with_indices = with_indices))
        # and then join all these strings and append them in brackets
        if(indent is None):
            tree_string += '(' + ', '.join(children_strings) + ')'
        else:
            tree_string += '(\n' + ('\t' * indent) + (',\n' + '\t' * indent).join(children_strings) + '\n' + ('\t' * (indent - 1)) + ')'
    return tree_string

