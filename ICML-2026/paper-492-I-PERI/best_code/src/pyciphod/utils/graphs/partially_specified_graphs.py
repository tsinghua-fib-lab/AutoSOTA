from pyciphod.utils.graphs.graphs import Graph, DirectedAcyclicGraph, DirectedMixedGraph, AcyclicDirectedMixedGraph
from pyciphod.utils.graphs.orientation_rules import meek_rule_1, meek_rule_2, meek_rule_3
from typing import Hashable
from pyciphod.utils.time_series.data_format import DTimeVar

class PartiallySpecifiedGraph(Graph):
    def __init__(self):
        super().__init__()


class PartiallySpecifiedDirectedGraph(PartiallySpecifiedGraph):
    def __init__(self):
        super().__init__()


class ClusterDirectedMixedGraph(PartiallySpecifiedGraph, DirectedMixedGraph):
    def __init__(self):
        super().__init__()


class ClusterAcyclicDirectedMixedGraph(PartiallySpecifiedDirectedGraph, AcyclicDirectedMixedGraph):
    def __init__(self):
        super().__init__()


class ClusterDirectedAcyclicGraph(PartiallySpecifiedDirectedGraph, DirectedAcyclicGraph):
    def __init__(self):
        super().__init__()


class SummaryCausalGraph(ClusterDirectedMixedGraph):
    def __init__(self):
        super().__init__()
        self._lag_max = None
    
    def add_lag_max(self, lag_max: int) -> None:
        self._lag_max = lag_max

    def get_lag_max(self) -> int:
        return self._lag_max

    def get_possible_parents(self, vertex : Hashable, gamma : int) -> set:
        possible_parents = set()
        parents = self.get_parents(vertex)
        lag_max = self.get_lag_max()
        for p in parents:
            for t in range(0, lag_max + 1):
                possible_parents.add(DTimeVar(p,gamma + t))
        possible_parents.discard(DTimeVar(vertex,gamma))
        return possible_parents

        

class ExtendedSummaryCausalGraph(ClusterAcyclicDirectedMixedGraph):
    def __init__(self):
        super().__init__()


class LocalIndependenceGraph(ClusterDirectedMixedGraph):
    def __init__(self):
        super().__init__()


class DifferenceGraph(PartiallySpecifiedGraph, DirectedMixedGraph):
    def __init__(self):
        super().__init__()


class PartiallyDirectedGraphs(Graph):
    def __init__(self):
        super().__init__()


class CompletedPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()
        
        
class TemporalPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()


class FtCompletedPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()

class LocalEssentialGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()


class PartialAncestralGraphs(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()


class CompletedPartiallyDirectedAcyclicDifferenceGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()

    def construct_from_dag(self, dag: AcyclicDirectedMixedGraph):
        """Construct CPDAG (pattern) from a fully directed DAG-like graph `dag`.

        Uses internal graph API and the package's Meek rules (meek_rule_1/2/3).
        Updates self in-place to become the CPDAG.
        """
        # add vertices
        for v in dag.get_vertices():
            self.add_vertex(v)

        verts = sorted(dag.get_vertices())
        # skeleton: add undirected edge for any adjacency in dag
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if dag.is_adjacent(u, v):
                    # add undirected edge once (graphs.get_undirected_edges returns symmetric pairs)
                    if (u, v) not in self.get_undirected_edges() and (v, u) not in self.get_undirected_edges():
                        self.add_undirected_edge(u, v)

        # Identify v-structures (unshielded colliders) using dag's method when available
        try:
            # Graph provides get_all_unshielded_colliders()
            unshielded_colliders = dag.get_all_unshielded_colliders()
        except Exception:
            # fallback: compute from parent sets
            unshielded_colliders = []
            for k in verts:
                parents = sorted(list(dag.get_parents(k)))
                for a in range(len(parents)):
                    for b in range(a + 1, len(parents)):
                        i = parents[a]
                        j = parents[b]
                        if not dag.is_adjacent(i, j):
                            unshielded_colliders.append((i, k, j))

        # Orient v-structures in the skeleton
        for (i, k, j) in unshielded_colliders:
            # remove undirected edges i-k and j-k if present
            if (i, k) in self.get_undirected_edges() or (k, i) in self.get_undirected_edges():
                try:
                    self.remove_undirected_edge(i, k)
                except Exception:
                    pass
            if (j, k) in self.get_undirected_edges() or (k, j) in self.get_undirected_edges():
                try:
                    self.remove_undirected_edge(j, k)
                except Exception:
                    pass
            # add directed edges i->k, j->k (avoid duplicates)
            if (i, k) not in self.get_directed_edges():
                self.add_directed_edge(i, k)
            if (j, k) not in self.get_directed_edges():
                self.add_directed_edge(j, k)

        # Iteratively apply Meek rules until convergence
        # Repeat until no rule produces an orientation
        while True:
            changed = False
            nodes = list(self.get_vertices())
            adj = {x: set(self.get_adjacencies(x)) for x in nodes}

            # Rule 1 and Rule 2 as in ConstraintBased._apply_meek_rules
            for x in nodes:
                for y in sorted(adj[x]):
                    # Rule 1: for z in adj[y] \ adj[x]
                    for z in sorted(set(adj[y]) - set(adj[x])):
                        try:
                            if meek_rule_1(self, x, y, z):
                                # orient y - z as y -> z
                                try:
                                    self.remove_undirected_edge(z, y)
                                except Exception:
                                    pass
                                try:
                                    if (y, z) not in self.get_directed_edges():
                                        self.add_directed_edge(y, z)
                                except Exception:
                                    pass
                                changed = True
                        except Exception:
                            pass

                    # Rule 2: for z in adj[y] & adj[x]
                    for z in sorted(set(adj[y]) & set(adj[x])):
                        try:
                            if meek_rule_2(self, x, y, z):
                                # orient x - z as x -> z
                                try:
                                    self.remove_undirected_edge(x, z)
                                except Exception:
                                    pass
                                try:
                                    if (x, z) not in self.get_directed_edges():
                                        self.add_directed_edge(x, z)
                                except Exception:
                                    pass
                                changed = True
                        except Exception:
                            pass

            # Rule 3
            for x in nodes:
                for y in list(adj[x]):
                    if (x, y) not in self.get_directed_edges():
                        continue
                    for z in sorted(set(adj[y]) - set(adj[x])):
                        if (z, y) not in self.get_directed_edges():
                            continue
                        for w in sorted(set(adj[x]) & set(adj[y]) & set(adj[z])):
                            try:
                                if meek_rule_3(self, x, y, z, w):
                                    try:
                                        self.remove_undirected_edge(w, y)
                                    except Exception:
                                        pass
                                    try:
                                        if (w, y) not in self.get_directed_edges():
                                            self.add_directed_edge(w, y)
                                    except Exception:
                                        pass
                                    changed = True
                                    break
                            except Exception:
                                pass
                        if changed:
                            break
                if changed:
                    break
            # if no change in this pass, we are done
            if not changed:
                break

