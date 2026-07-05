import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from typing import Type, Optional, Set, Tuple, Iterable, FrozenSet
from typing import Hashable
from itertools import combinations, product

from functools import lru_cache



class Graph:
    def __init__(self):
        # super().__init__()
        self._directed_g = nx.DiGraph()
        self._confounded_g = nx.Graph()
        self._undirected_g = nx.Graph()
        self._uncertain_g = nx.DiGraph()

        self._g = nx.MultiDiGraph()

        # self._directed_g.nodes = self._g.nodes
        # self._confounded_g.nodes = self._g.nodes
        # self._undirected_g.nodes = self._g.nodes
        # self._uncertain_g.nodes = self._g.nodes

        self._list_certain_edge_types = ['<->', '->', '-']
        self._list_uncertain_edge_types = ['o-o', 'o->', 'o-', '--', '-->', '-||']
        self.list_pointed_edge_types = ['<->', '->', 'o->', '-->']

    def copy(self):
        new_graph = Graph()
        new_graph._g = deepcopy(self._g)
        new_graph._directed_g = deepcopy(self._directed_g)
        new_graph._confounded_g = deepcopy(self._confounded_g)
        new_graph._undirected_g = deepcopy(self._undirected_g)
        new_graph._uncertain_g = deepcopy(self._uncertain_g)
        new_graph._list_certain_edge_types = deepcopy(self._list_certain_edge_types)
        new_graph._list_uncertain_edge_types = deepcopy(self._list_uncertain_edge_types)
        new_graph.list_pointed_edge_types = deepcopy(self.list_pointed_edge_types)
        return new_graph

    def add_vertex(self, vertex: Hashable) -> None:
        if vertex not in self._g.nodes:
            self._g.add_node(vertex)
            self._directed_g.add_node(vertex)
            self._confounded_g.add_node(vertex)
            self._undirected_g.add_node(vertex)
            self._uncertain_g.add_node(vertex)

    def add_vertices(self, vertices_list: list) -> None:
        for vertex in vertices_list:
            self.add_vertex(vertex)

    def get_vertices(self):
        return set(self._g.nodes)

    def add_directed_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        # raise NotImplementedError("Please Implement this method")
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        self._directed_g.add_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_i, vertex_j, key='->')
        # if (vertex_i, vertex_j) not in self.directed_edges:
        #     self.add_vertices([vertex_i, vertex_j])
        #     self._g.add_edge(vertex_i, vertex_j, type='->')
        #     self.directed_edges.append((vertex_i, vertex_j))
        #     # self.directed_edges = [(vertex_i, vertex_j) for vertex_i, vertex_j, attrs in self.edges(data=True) if attrs.get("type") == '->']
        # else:
        #     print("Warning: Edge already exists")

    def add_directed_edges_from(self, edge_list: list) -> None:
        for (vertex_i, vertex_j) in edge_list:
            self.add_directed_edge(vertex_i, vertex_j)

    def remove_directed_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        if self._directed_g.has_edge(vertex_i, vertex_j):
            self._directed_g.remove_edge(vertex_i, vertex_j)
        if self._g.has_edge(vertex_i, vertex_j, key='->'):
            self._g.remove_edge(vertex_i, vertex_j, key='->')

    def add_confounded_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        self._confounded_g.add_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_i, vertex_j, key='<->')
        self._g.add_edge(vertex_j, vertex_i, key='<->')

    def add_confounded_edges_from(self, edge_list: list) -> None:
        for (vertex_i, vertex_j) in edge_list:
            self.add_confounded_edge(vertex_i, vertex_j)

    def remove_confounded_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        if self._confounded_g.has_edge(vertex_i, vertex_j):
            self._confounded_g.remove_edge(vertex_i, vertex_j)
        
        if self._g.has_edge(vertex_i, vertex_j, key='<->'):
            self._g.remove_edge(vertex_i, vertex_j, key='<->')
        if self._g.has_edge(vertex_j, vertex_i, key='<->'):
            self._g.remove_edge(vertex_j, vertex_i, key='<->')

    def add_undirected_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        self._undirected_g.add_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_i, vertex_j, key='-')
        self._g.add_edge(vertex_j, vertex_i, key='-')

    def add_undirected_edges_from(self, edge_list: list) -> None:
        for (vertex_i, vertex_j) in edge_list:
            self.add_undirected_edge(vertex_i, vertex_j)

    def remove_undirected_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        if self._undirected_g.has_edge(vertex_i, vertex_j):
            self._undirected_g.remove_edge(vertex_i, vertex_j)
        if self._g.has_edge(vertex_i, vertex_j, key='-'):
            self._g.remove_edge(vertex_i, vertex_j, key='-')
        if self._g.has_edge(vertex_j, vertex_i, key='-'):
            self._g.remove_edge(vertex_j, vertex_i, key='-')

    def add_uncertain_edge(self, vertex_i: Hashable, vertex_j: Hashable, edge_type='o-o') -> None:
        """

        :param vertex_i:
        :param vertex_j:
        :param edge_type: '*-o' or '*->' or '*-' or '--' or '-->' or '-||'
        :return:
        """
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        assert edge_type in self._list_uncertain_edge_types
        self._uncertain_g.add_edge(vertex_i, vertex_j, type=edge_type)
        self._g.add_edge(vertex_i, vertex_j, key=edge_type)

    def remove_uncertain_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        if self._uncertain_g.has_edge(vertex_i, vertex_j):
            self._uncertain_g.remove_edge(vertex_i, vertex_j)
        if self._uncertain_g.has_edge(vertex_j, vertex_i):
            self._uncertain_g.remove_edge(vertex_j, vertex_i)

        for edge_type in self._list_uncertain_edge_types:
            if self._g.has_edge(vertex_i, vertex_j, key=edge_type):
                self._g.remove_edge(vertex_i, vertex_j, key=edge_type)
            if self._g.has_edge(vertex_j, vertex_i, key=edge_type):
                self._g.remove_edge(vertex_j, vertex_i, key=edge_type)

    # def update_uncertain_edge(self, vertex_i, vertex_j, edge_type='*-o') -> None:
    #     assert edge_type in self._list_uncertain_edge_types
    #     self._uncertain_g.add_edge(vertex_i, vertex_j, type=edge_type)
    #     for edge_type in self._list_uncertain_edge_types:
    #         self._g.remove_edge(vertex_i, vertex_j, key=edge_type)
    #     self._g.add_edge(vertex_j, vertex_i, key=edge_type)

    def uncertain_to_certain_edge(self, vertex_i: Hashable, vertex_j: Hashable, edge_type="->") -> None:
        assert edge_type in self._list_certain_edge_types
        self.remove_uncertain_edge(vertex_i, vertex_j)
        if edge_type == "->":
            self.add_directed_edge(vertex_i, vertex_j)
        elif edge_type == "<->":
            self.add_confounded_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_j, vertex_i, key=edge_type)

    def remove_incoming_edges(self, vertices) -> None:
        for vertex_j in vertices:
            for vertex_i in self.get_parents(vertex_j):
                self.remove_directed_edge(vertex_i, vertex_j)
            for vertex_i in self.get_confounded_adjacencies(vertex_j):
                self.remove_confounded_edge(vertex_i, vertex_j)
                self.remove_confounded_edge(vertex_j, vertex_i)

    def remove_outgoing_edges(self, vertices) -> None:
        for vertex_i in vertices:
            for vertex_j in self.get_children(vertex_i):
                self.remove_directed_edge(vertex_i, vertex_j)

    def is_pointed_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> bool:
        try:
            edge_types = self.get_edge_types(vertex_i, vertex_j)
        except Exception:
            # if no edge data available, assume no pointing
            edge_types = set()

        # If any of the edge types corresponds to an arrow into dst, return True
        for et in edge_types:
            if et in self.list_pointed_edge_types:
                return True
        return False

    def is_adjacent(self, vertex_i: Hashable, vertex_j: Hashable) -> bool:
        """True if there is any adjacency between u and v (directed, confounded, undirected, uncertain)."""
        return vertex_i in self.get_adjacencies(vertex_j)

    def is_active_path(self,path: list, adjustment_set: set = None) -> bool:
        """
        # TODO
        """
        if adjustment_set is None:
            adjustment_set = set()

        colliders = set()
        has_seen_right_arrow = False
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            # check connectivity and direction
            a_to_b = self.is_pointed_edge(a, b)
            b_to_a = self.is_pointed_edge(b, a)
            if a_to_b and b_to_a:
                # bidirectional (treat like <->)
                pass
            elif a_to_b and not b_to_a:
                # a -> b
                if (i > 0) and (path[i] in adjustment_set):
                    return False
                has_seen_right_arrow = True
            elif (not a_to_b) and b_to_a:
                # a <- b
                if has_seen_right_arrow:
                    colliders.add(path[i])
                    has_seen_right_arrow = False
            else:
                # no directed arrow either way; check if there is an adjacency (undirected/uncertain without arrows)
                if self.is_adjacent(a, b):
                    # treat as non-collider segment: if the middle node is in adjustment_set, path is blocked
                    if (i > 0) and (path[i] in adjustment_set):
                        return False
                    # non-collider resets seen-right-arrow
                    has_seen_right_arrow = False
                    continue
                # not connected at all
                raise ValueError("Path is not a path in graph: {} and {} are not connected.".format(a, b))
        # For each collider, at least one descendant is in adjustment_set (or collider itself)
        for c in colliders:
            if c in adjustment_set:
                continue
            for d in self.get_descendants(c):
                if d in adjustment_set:
                    break
            else:
                return False
        return True

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self._directed_g)

    # def _has_directed_edge(self, vertex_i: str, vertex_i: str) -> bool:
    #     """True if there is a definite or uncertain arrow u->v in the graph's representation."""
    #     # check directed edges
    #     if (vertex_i, vertex_i) in self.get_directed_edges():
    #         return True
    #     # check multigraph edge types
    #     try:
    #         types = graph.get_edge_types(u, v)
    #     except Exception:
    #         types = set()
    #     for t in types:
    #         if isinstance(t, str) and (t in {"->", "*->", "-->"} or t.endswith('>')):
    #             return True
    #         if t == '<->':
    #             return True
    #     return False

    def get_edges(self):
        return set(self._g.edges(keys=True))

    def get_directed_edges(self):
        return set(self._directed_g.edges)

    def get_confounded_edges(self):
        return set(self._confounded_g.edges)

    # def get_undirected_edges(self):
    #     return set(self._undirected_g.edges)
    
    def get_undirected_edges(self): # Corrected to have the symmetry
        return set(e for edge in self._undirected_g.edges() for e in [edge, edge[::-1]])
    
    def get_uncertain_edges(self):
        return set(self._uncertain_g.edges)

    def get_edge_types(self, vertex_i: Hashable, vertex_j: Hashable):
        return set(list(self._g.get_edge_data(vertex_i, vertex_j).keys()))

    def get_parents(self, vertex: Hashable) -> set:
        """
        :param vertex:
        :return:
        """
        return set(self._directed_g.predecessors(vertex))

    def get_children(self, vertex: Hashable) -> set:
        return set(self._directed_g.successors(vertex))

    # def get_adjacencies(self, vertex: str) -> set:
    #     parents = self.get_parents(vertex)
    #     children = self.get_children(vertex)
    #     adjacencies = parents.union(children)
    #     return adjacencies
    
    def get_adjacencies(self, vertex: Hashable) -> set:
        adjacents = set()
        if vertex in self._directed_g:
            adjacents.update(self._directed_g.predecessors(vertex))
            adjacents.update(self._directed_g.successors(vertex))
        if vertex in self._confounded_g:
            adjacents.update(self._confounded_g.neighbors(vertex))
        if vertex in self._undirected_g:
            adjacents.update(self._undirected_g.neighbors(vertex))
        if vertex in self._uncertain_g:
            adjacents.update(self._uncertain_g.predecessors(vertex))
            adjacents.update(self._uncertain_g.successors(vertex))
        return adjacents

    # def get_unshielded_triples(self):
    #     """
    #     Returns all unshielded triples in the graph.
    #     An unshielded triple is a triple (X, Z, Y) where:
    #         - X and Z are adjacent
    #         - Y and Z are adjacent
    #         - X and Y are NOT adjacent
    #     """
    #     triples = []
    #     adj = {v: self.get_adjacencies(v) for v in self.get_vertices()}

    #     for z in self.get_vertices():
    #         neighbors = list(adj[z])
    #         if len(neighbors) < 2:
    #             continue
    #         # on parcourt toutes les paires de voisins
    #         for i in range(len(neighbors)):
    #             x = neighbors[i]
    #             for j in range(i + 1, len(neighbors)):
    #                 y = neighbors[j]
    #                 if y not in adj[x]: 
    #                     triples.append((x, z, y))
    #     return triples


    def get_ancestors(self, vertex: Hashable) -> set:
        """
        sds
        :param vertex:
        :return:
        """
        def ancestor_recursive(vertex_i: Hashable, sublist: list):
            sublist.append(vertex_i)
            if self.get_parents(vertex_i):
                for parent in self.get_parents(vertex_i):
                    if parent not in sublist:
                        return sublist + ancestor_recursive(parent, sublist)
                    else:
                        return sublist
            else:
                return sublist
        return set(ancestor_recursive(vertex, []))

    def get_descendants(self, vertex: Hashable) -> set:
        """
        sds
        :param vertex:
        :return:
        """
        def descendant_recursive(vertex_i: Hashable, sublist: list):
            sublist.append(vertex_i)
            if self.get_children(vertex_i):
                for child in self.get_children(vertex_i):
                    if child not in sublist:
                        return sublist + descendant_recursive(child, sublist)
                    else:
                        return sublist
            else:
                return sublist
        return set(descendant_recursive(vertex, []))

    def get_non_descendants(self, vertex: Hashable) -> set:
        return self.get_vertices().difference(self.get_descendants(vertex))

    def get_confounded_adjacencies(self, vertex: Hashable) -> set:
        '''
        Returns the set of vertices adjacent to the vertex in parameter through confounded edges.
        
        :param self: Graph of interest
        :param vertex: Vertex of interest
        :type vertex: Hashable
        :return: Set of adjacent vertices through confounded edges.
        :rtype: set
        '''
        return set(self._confounded_g.adj[vertex])
        # confounded_adjacencies = []
        # for potential_adj in self._g.pred[vertex]:
        #     for idx in self._g.pred[vertex][potential_adj]:
        #         if self._g.pred[vertex][potential_adj][idx]["type"] == '<->':
        #             if potential_adj not in confounded_adjacencies:
        #                 confounded_adjacencies.append(potential_adj)
        # The symmetry of the confounded adjacencies is ensured by adding edges from both vertex_i to vertex_j and
        # from vertex_j to vertex_i in the function add_bidirectional_edge.
        # for potential_adj in self._g.succ[vertex]:
        #     for idx in self._g.succ[vertex][potential_adj]:
        #         if self._g.succ[vertex][potential_adj][idx]["type"] == '<->':
        #             if potential_adj not in confounded_adjacencies:
        #                 confounded_adjacencies.append(potential_adj)
        # return confounded_adjacencies

    # def all_paths(self, vertex_i: str, vertex_j: str):
    #     # TODO
    #     1

    def get_simple_paths(self, vertex_i: Hashable, vertex_j: Hashable, allowed_nodes: Iterable[Hashable] = None,
                                cutoff: int = 10):
        """
        Générateur de chemins simples entre source et target en se basant uniquement sur `get_adjacencies`.
        Evite la création d'un nouveau graphe en itérant récursivement sur les voisins.
        """
        allowed = None if allowed_nodes is None else set(allowed_nodes)

        if allowed is not None and (vertex_i not in allowed or vertex_j not in allowed):
            return

        visited = [vertex_i]

        def dfs(current):
            if len(visited) > cutoff:
                return
            if current == vertex_j:
                yield list(visited)
                return
            # neighbors via adjacencies
            neighs = self.get_adjacencies(current)
            if allowed is not None:
                neighs = neighs & allowed
            for n in neighs:
                if n in visited:
                    continue
                visited.append(n)
                for p in dfs(n):
                    yield p
                visited.pop()

        for p in dfs(vertex_i):
            yield p

    def get_active_paths(self, vertex_i: Hashable, vertex_j: Hashable, allowed_nodes: Iterable[Hashable] = None,
                            max_path_length: int = 10):
        """
        Recherche s'il existe un chemin actif entre u et v dans `graph` sans construire un `sg` séparé.
        """
        p = []
        try:
            for path in self.get_simple_paths(vertex_i, vertex_j, allowed_nodes=allowed_nodes, cutoff=max_path_length):
                if self.is_active_path(path, adjustment_set=set()):
                    p.append(path)
        except ValueError:
            return p
        return p

    def get_confounded_paths(self, vertex_i: Hashable, vertex_j: Hashable):
        # TODO
        1

    def get_all_counfounded_components(self):
        # TODO
        1

    def get_all_colliders(self) -> Set[Tuple[Hashable, Hashable, Hashable]]:
        colliders = set()
        vertices = list(self.get_vertices())

        for z in vertices:
            adj = self.get_adjacencies(z)
            if len(adj) < 2:
                continue
            adj_list = list(adj)
            n = len(adj_list)
            for i in range(n):
                x = adj_list[i]
                for j in range(i + 1, n):
                    y = adj_list[j]
                    # Now check if edges point towards z from x and y
                    x_points = self.is_pointed_edge(x, z)
                    y_points = self.is_pointed_edge(y, z)
                    if x_points and y_points:
                        a, b = (x, y) if x <= y else (y, x)
                        colliders.add((a, z, b))
        return colliders

    def get_all_unshielded_colliders(self) -> Set[Tuple[Hashable, Hashable, Hashable]]:
        all_colliders = self.get_all_colliders()
        unshielded = set()
        for (x, z, y) in all_colliders:
            # x and y are not adjacent
            if y not in self.get_adjacencies(x):
                unshielded.add((x, z, y))
        return unshielded

    def _all_nonempty_subsets(self, iterable: Iterable[Hashable]):
        s = list(iterable)
        for r in range(1, len(s) + 1):
            for comb in combinations(s, r):
                yield frozenset(comb)

    def get_all_cluster_super_unshielded_colliders(self, max_vertices_for_search: int = 12,
                                               max_path_length: int = 8) -> Set[
        Tuple[FrozenSet[Hashable], FrozenSet[Hashable], FrozenSet[Hashable]]]:
        """
        Retourne l'ensemble des clustered super-unshielded colliders (CSUC) du graphe `graph`.

        Implémentation exhaustive utilisant uniquement les méthodes/attributs du `Graph` fourni.
        """
        vertices = list(self.get_vertices())
        n = len(vertices)
        if n == 0:
            return set()
        if n > max_vertices_for_search:
            raise ValueError(
                f"Graph too large for exhaustive CSUC search (|V|={n}). Increase max_vertices_for_search with caution.")

        # Precompute active_any between all pairs (unrestricted)
        active_any = {}
        for u, v in product(vertices, repeat=2):
            if u == v:
                active_any[(u, v)] = False
            else:
                test = len(self.get_active_paths(u, v, allowed_nodes=None, max_path_length=max_path_length))>0
                active_any[(u, v)] = test

        candidates = set()
        Vset = set(vertices)

        # enumerate Y (non-empty)
        for Y in self._all_nonempty_subsets(vertices):
            outside = Vset - Y
            if not outside:
                continue
            # enumerate X as non-empty subset of outside
            for X in self._all_nonempty_subsets(outside):
                rest = outside - X
                if not rest:
                    continue
                # enumerate Z as non-empty subset of rest
                for Z in self._all_nonempty_subsets(rest):
                    # ensure disjointness by construction
                    if X & Y or Y & Z or X & Z:
                        continue
                    # Check separation across sides: no active path between any x in X and z in Z
                    separated = True
                    for x in X:
                        for z in Z:
                            if active_any.get((x, z), False) or active_any.get((z, x), False):
                                separated = False
                                break
                        if not separated:
                            break
                    if not separated:
                        continue
                    # Connection to the middle: for each x in X and y in Y exists active path within X∪Y
                    XY_allowed = X | Y
                    XY_ok = True
                    for x in X:
                        for y in Y:
                            test = len(self.get_active_paths(x, y, allowed_nodes=XY_allowed, max_path_length=max_path_length)) > 0
                            if not test:
                                XY_ok = False
                                break
                        if not XY_ok:
                            break
                    if not XY_ok:
                        continue
                    # Similarly for ZY
                    ZY_allowed = Z | Y
                    ZY_ok = True
                    for z in Z:
                        for y in Y:
                            test = len(self.get_active_paths(z, y, allowed_nodes=ZY_allowed, max_path_length=max_path_length)) > 0
                            if not test:
                                ZY_ok = False
                                break
                        if not ZY_ok:
                            break
                    if not ZY_ok:
                        continue
                    # At this point (X,Y,Z) satisfy (1)-(2)
                    candidates.add((frozenset(X), frozenset(Y), frozenset(Z)))

        # Filter maximal triplets (no strict superset in candidates)
        maximal = set()
        for trip in candidates:
            X, Y, Z = trip
            is_max = True
            for other in candidates:
                if other == trip:
                    continue
                X2, Y2, Z2 = other
                if X <= X2 and Y <= Y2 and Z <= Z2 and (X < X2 or Y < Y2 or Z < Z2):
                    is_max = False
                    break
            if is_max:
                maximal.add(trip)

        return maximal

    def get_all_simple_cycles(self):
        return list(nx.simple_cycles(self._directed_g))

    def get_simple_cycles(self, vertex: Hashable):
        # cyc_list = []
        # for cyc in self.get_all_simple_cycles():
        #     if vertex in cyc:
        #         cyc_list.append(cyc)
        cycles = self.get_all_simple_cycles()
        nodes_in_cycles = set()
        for cyc in cycles:
            nodes_in_cycles.update(cyc)
        # Also return the subset related to y: if y in cycle include the cycle nodes
        related = set()
        for cyc in cycles:
            if vertex in cyc:
                for i in range(len(cyc)):
                    a = cyc[i]
                    b = cyc[(i + 1) % len(cyc)]
                    related.add((a, b))
        if related:
            return related
        return nodes_in_cycles

    def get_all_strongly_connected_components(self):
        return list(nx.strongly_connected_components(self._directed_g))

    def get_strongly_connected_components(self, vertex: Hashable):
        for comp in self.get_all_strongly_connected_components():
            if vertex in comp:
                return set(comp)
        return set()

    def old_draw_graph(self, treatment: set = None, outcome: set = None):
        vertex_color = "lightblue"
        font_color = "black"
        directed_edge_color = "gray"
        confounded_edge_color = "black"
        undirected_edge_color = "black"
        uncertain_edge_color = "#F7B617"
        treatment_color = "#c82804"
        outcome_color = "#4851a1"

        all_nodes = list(self._g.nodes)
        outcome = set(outcome or [])
        treatment = set(treatment or [])
        non_outcome_nodes = [n for n in all_nodes if n not in outcome]

        # Layout
        circular_pos = nx.circular_layout(self._g.subgraph(non_outcome_nodes))
        center = np.array([0.0, 0.0])
        pos = {n: center for n in outcome}
        pos.update(circular_pos)

        fig, ax = plt.subplots()

        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(treatment), node_color=treatment_color)
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(outcome), node_color=outcome_color)

        set_vertices = set(self._g.nodes) - treatment - outcome
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(set_vertices), node_color=vertex_color)
        nx.draw_networkx_labels(self._g, pos, ax=ax, font_color=font_color)

        acyclic_edges = [edge for edge in self.get_directed_edges() if (edge[1], edge[0]) not in self.get_directed_edges()]
        cyclic_edges = [edge for edge in self.get_directed_edges() if (edge[1], edge[0]) in self.get_directed_edges()]
        confounded_edges = self.get_confounded_edges()
        undirected_edges = self.get_undirected_edges()

        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=acyclic_edges, edge_color=directed_edge_color, arrowstyle='->')
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=cyclic_edges, edge_color=directed_edge_color, arrowstyle='->')

        # Draw confounded edges with FancyArrowPatch to ensure curved arcs with arrowheads on both sides.
        from matplotlib.patches import FancyArrowPatch
        try:
            confounded_list = list(confounded_edges)
        except Exception:
            confounded_list = []
        for idx, (u, v) in enumerate(confounded_list):
            if u not in pos or v not in pos:
                continue
            # alternate curvature to reduce overlap
            rad = 0.18 if (idx % 2 == 0) else -0.18
            xyA = tuple(pos[u])
            xyB = tuple(pos[v])
            # shrink so arrows don't overlap node markers (in points)
            shrink_pts = 8
            # Draw a single curved, dashed, bidirected arrow (arrowheads both ends)
            arrow = FancyArrowPatch(xyA, xyB,
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle='<->',
                                    mutation_scale=18,
                                    color=confounded_edge_color,
                                    linewidth=1.5,
                                    shrinkA=shrink_pts, shrinkB=shrink_pts,
                                    linestyle='dashed')
            # ensure arrowheads are drawn above nodes and not clipped
            arrow.set_zorder(3)
            arrow.set_clip_on(False)
            ax.add_patch(arrow)

        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=undirected_edges, arrowstyle='-', edge_color=undirected_edge_color)

        dashed_arrow = [(u, v) for (u, v, t) in self.get_edges() if t == '-->']
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=dashed_arrow, edge_color=uncertain_edge_color,
                            style='dashed', arrowstyle='->')

        arrow_double_bar = [(u, v) for (u, v, t) in self.get_edges() if t == '-||']
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=arrow_double_bar, edge_color="red",
                            style='solid', arrowstyle='-[')

        plt.axis('off')
        plt.show()

    def draw_graph(self, treatment: set = None, outcome: set = None):
        import matplotlib.pyplot as plt
        import numpy as np
        import networkx as nx
        from matplotlib.patches import FancyArrowPatch

        vertex_color = "lightblue"
        font_color = "black"
        directed_edge_color = "gray"
        confounded_edge_color = "black"
        undirected_edge_color = "black"
        uncertain_edge_color = "#F7B617"
        treatment_color = "#c82804"
        outcome_color = "#4851a1"

        all_nodes = list(self._g.nodes)
        outcome = set(outcome or [])
        treatment = set(treatment or [])
        non_outcome_nodes = [n for n in all_nodes if n not in outcome]

        # Layout
        circular_pos = nx.circular_layout(self._g.subgraph(non_outcome_nodes))
        center = np.array([0.0, 0.0])
        pos = {n: center for n in outcome}
        pos.update(circular_pos)

        fig, ax = plt.subplots()

        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(treatment), node_color=treatment_color)
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(outcome), node_color=outcome_color)

        set_vertices = set(self._g.nodes) - treatment - outcome
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(set_vertices), node_color=vertex_color)
        nx.draw_networkx_labels(self._g, pos, ax=ax, font_color=font_color)

        directed_edges = list(self.get_directed_edges())
        confounded_edges = self.get_confounded_edges()
        undirected_edges = self.get_undirected_edges()

        # Split directed edges into self-loops and non-self-loops
        directed_self_loops = [(u, v) for (u, v) in directed_edges if u == v]
        directed_non_self = [(u, v) for (u, v) in directed_edges if u != v]

        acyclic_edges = [(u, v) for (u, v) in directed_non_self if (v, u) not in directed_edges]
        cyclic_edges = [(u, v) for (u, v) in directed_non_self if (v, u) in directed_edges]

        nx.draw_networkx_edges(
            self._g, pos, ax=ax,
            edgelist=acyclic_edges,
            edge_color=directed_edge_color,
            arrowstyle='->'
        )

        nx.draw_networkx_edges(
            self._g, pos, ax=ax,
            edgelist=cyclic_edges,
            edge_color=directed_edge_color,
            arrowstyle='->',
            connectionstyle="arc3,rad=0.35"
        )

        # ---- draw directed self-loops manually ----
        for u, _ in directed_self_loops:
            x, y = pos[u]

            # small offset so the loop is visible outside the node
            dx = 0.05
            dy = 0.05

            loop = FancyArrowPatch(
                (x, y + dy),  # start slightly above node
                (x + dx, y),  # end slightly to the right
                connectionstyle="arc3,rad=-2.5",
                arrowstyle='->',
                mutation_scale=18,
                color=directed_edge_color,
                linewidth=1.8,
                zorder=4
            )
            loop.set_clip_on(False)
            ax.add_patch(loop)

        # Draw confounded edges
        try:
            confounded_list = list(confounded_edges)
        except Exception:
            confounded_list = []

        for idx, (u, v) in enumerate(confounded_list):
            if u not in pos or v not in pos:
                continue

            # self-loop confounding, if needed
            if u == v:
                x, y = pos[u]
                dx = 0.14
                dy = -0.14
                arrow = FancyArrowPatch(
                    (x, y + dy),
                    (x - dx, y),
                    connectionstyle="arc3,rad=2.5",
                    arrowstyle='<->',
                    mutation_scale=18,
                    color=confounded_edge_color,
                    linewidth=1.5,
                    linestyle='dashed',
                    zorder=4
                )
                arrow.set_clip_on(False)
                ax.add_patch(arrow)
                continue

            rad = 0.18 if (idx % 2 == 0) else -0.18
            xyA = tuple(pos[u])
            xyB = tuple(pos[v])
            shrink_pts = 8

            arrow = FancyArrowPatch(
                xyA, xyB,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='<->',
                mutation_scale=18,
                color=confounded_edge_color,
                linewidth=1.5,
                shrinkA=shrink_pts,
                shrinkB=shrink_pts,
                linestyle='dashed'
            )
            arrow.set_zorder(3)
            arrow.set_clip_on(False)
            ax.add_patch(arrow)

        nx.draw_networkx_edges(
            self._g, pos, ax=ax,
            edgelist=undirected_edges,
            arrowstyle='-',
            edge_color=undirected_edge_color
        )

        dashed_arrow = [(u, v) for (u, v, t) in self.get_edges() if t == '-->']
        dashed_self_loops = [(u, v) for (u, v) in dashed_arrow if u == v]
        dashed_arrow = [(u, v) for (u, v) in dashed_arrow if u != v]

        nx.draw_networkx_edges(
            self._g, pos, ax=ax,
            edgelist=dashed_arrow,
            edge_color=uncertain_edge_color,
            style='dashed',
            arrowstyle='->'
        )

        for u, _ in dashed_self_loops:
            x, y = pos[u]
            dx = -0.12
            dy = 0.12
            loop = FancyArrowPatch(
                (x, y + dy),
                (x + dx, y),
                connectionstyle="arc3,rad=2.5",
                arrowstyle='->',
                mutation_scale=18,
                color=uncertain_edge_color,
                linewidth=1.5,
                linestyle='dashed',
                zorder=4
            )
            loop.set_clip_on(False)
            ax.add_patch(loop)

        arrow_double_bar = [(u, v) for (u, v, t) in self.get_edges() if t == '-||']
        nx.draw_networkx_edges(
            self._g, pos, ax=ax,
            edgelist=arrow_double_bar,
            edge_color="red",
            style='solid',
            arrowstyle='-['
        )

        # expand limits slightly so loops are not cut off
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        if xs and ys:
            ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4)
            ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)

        plt.axis('off')
        plt.show()

# class FullySpecifiedGraph(Graph):
#     def __init__(self):
#         super(Graph, self).__init__()
#         self.add_directed_edge = self.add_directed_edge if self._remain_acyclic() else 1


class DirectedMixedGraph(Graph):
    def __init__(self):
        super().__init__()

    def add_undirected_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)

    def remove_undirected_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)

    def add_uncertain_edge(self, vertex_i: Hashable, vertex_j: Hashable, edge_type='o-o') -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)

    def remove_uncertain_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)


class AcyclicDirectedMixedGraph(DirectedMixedGraph):
    def __init__(self):
        super().__init__()
        # ensure current state is acyclic
        if not self.is_acyclic():
            raise ValueError("Initial directed edges contain a cycle; AcyclicDirectedMixedGraph must be acyclic")


class DirectedAcyclicGraph(AcyclicDirectedMixedGraph):
    def __init__(self):
        super(AcyclicDirectedMixedGraph, self).__init__()

    def add_directed_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        """Add a directed edge only if it does not create a cycle. If adding the edge would create a
        directed cycle, the addition is rolled back and a ValueError is raised.
        """
        # Use the Graph implementation to add the edge, then validate acyclicity.
        super().add_directed_edge(vertex_i, vertex_j)
        if not self.is_acyclic():
            # revert the change
            super().remove_directed_edge(vertex_i, vertex_j)
            raise ValueError(f"Adding directed edge {vertex_i}->{vertex_j} would create a cycle in AcyclicDirectedMixedGraph")

    def add_confounded_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)

    def remove_confounded_edge(self, vertex_i: Hashable, vertex_j: Hashable) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)


def create_random_admg(num_v: int,
                             p_edge: float = 0.2,
                             seed: Optional[int] = None):
    """
    Generate a random ADMG.

    Parameters
    ----------
    num_v: int
        Number of vertices in the graph.
    p_edge: float (default=0.2)
        Probability of adding an edge between any pair of vertices.
    seed: int (optional)
        Random seed for reproducibility.
    Returns admg: AcyclicDirectedMixedGraph
    """
    rng = np.random.default_rng(seed)

    v = [f"X{i}" for i in range(num_v)]

    # produce a random topological order
    order = list(v)
    rng.shuffle(order)

    admg = AcyclicDirectedMixedGraph()
    for var in v:
        admg.add_vertex(var)

    # add edges from earlier -> later in the random order with probability p_edge
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            src = order[i]
            tgt = order[j]
            if rng.random() < p_edge:
                admg.add_directed_edge(src, tgt)
            if rng.random() < p_edge:
                admg.add_confounded_edge(src, tgt)
    return admg


def create_random_dag(num_v: int,
                             p_edge: float = 0.2,
                             seed: Optional[int] = None):
    """
    Generate a random DAG.

    Parameters
    ----------
    num_v: int
        Number of vertices in the graph.
    p_edge: float (default=0.2)
        Probability of adding an edge between any pair of vertices.
    seed: int (optional)
        Random seed for reproducibility.
    Returns dag: DirectedAcyclicGraph
    """
    rng = np.random.default_rng(seed)

    v = [f"X{i}" for i in range(num_v)]

    # produce a random topological order
    order = list(v)
    rng.shuffle(order)

    dag = AcyclicDirectedMixedGraph()
    for var in v:
        dag.add_vertex(var)

    # add edges from earlier -> later in the random order with probability p_edge
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            src = order[i]
            tgt = order[j]
            if rng.random() < p_edge:
                dag.add_directed_edge(src, tgt)
    return dag
