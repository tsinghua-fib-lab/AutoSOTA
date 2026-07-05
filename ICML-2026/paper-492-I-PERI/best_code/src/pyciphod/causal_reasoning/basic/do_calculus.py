from __future__ import annotations

from typing import Iterable, Set
from typing import Type, Optional, Set, Tuple, Iterable, FrozenSet
import networkx as nx

from pyciphod.utils.graphs.graphs import DirectedMixedGraph, DirectedAcyclicGraph
from pyciphod.utils.graphs.separation import d_separated, m_separated

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from collections import deque
from typing import Callable, FrozenSet, Iterable, Optional, Sequence




def _to_set(nodes: Iterable) -> Set[str]:
    if nodes is None:
        return set()
    if isinstance(nodes, (str,)):
        return {nodes}
    return set(nodes)


def _build_augmented_dag(dmg: DirectedMixedGraph) -> DirectedAcyclicGraph:
    """Build a DAG suitable for d-separation checks by converting each confounded (bidirected) edge u<->v
    into a latent node L_uv with edges L_uv -> u and L_uv -> v. Directed edges are kept as-is.
    Returns a DirectedAcyclicGraph (package class).
    """
    aug = DirectedAcyclicGraph()
    # add observed nodes
    for v in dmg.get_vertices():
        aug.add_vertex(v)
    # add directed edges
    for (u, v) in dmg.get_directed_edges():
        aug.add_directed_edge(u, v)
    # add latent nodes for confounded edges
    for (u, v) in dmg.get_confounded_edges():
        # canonical order
        a, b = sorted((u, v))
        latent = f"__L__{a}__{b}"
        # avoid collision with existing nodes
        if latent not in aug.get_vertices():
            aug.add_vertex(latent)
        aug.add_directed_edge(latent, u)
        aug.add_directed_edge(latent, v)
    return aug


def rule1_applies(dmg: DirectedMixedGraph, Y: Iterable, X: Iterable, Z: Iterable, W: Iterable = None) -> bool:
    """Rule 1 (insertion/deletion of observations)
    If Y is independent of Z given X and W in G_{bar{X}} (graph where incoming edges to X are removed),
    then P(Y | do(X), Z, W) = P(Y | do(X), W).

    Returns True if the condition holds on the graph (i.e., rule can be applied).
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)
    Ws = _to_set(W)

    g_manipulated = dmg.copy()
    # remove incoming edges to X
    g_manipulated.remove_incoming_edges(Xs)

    # test d-separation: Y _||_ Z | X U W in G_barX
    cond = Xs | Ws
    print(cond, "|", Ys, "|", Zs)
    return m_separated(g_manipulated, Ys, Zs, cond)


def rule2_applies(dmg: DirectedMixedGraph, Y: Iterable, X: Iterable, Z: Iterable, W: Iterable = None) -> bool:
    """Rule 2 (action/observation exchange)
    If Y is independent of Z given X and W in G_{bar{X},underline{Z}} (incoming edges to X removed, outgoing from Z removed),
    then P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W).

    Returns True if the condition holds on the graph.
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)
    Ws = _to_set(W)

    g_manipulated = dmg.copy()
    g_manipulated.remove_incoming_edges(Xs)
    g_manipulated.remove_outgoing_edges(Zs)

    cond = Xs | Ws
    return m_separated(g_manipulated, Ys, Zs, cond)


def rule3_applies(dmg: DirectedMixedGraph, Y: Iterable, X: Iterable, Z: Iterable, W: Iterable = None) -> bool:
    """Rule 3 (insertion/deletion of actions)
    Let Z* be the subset of Z that are not ancestors of any W in G_{bar{X}}. If Y is independent of Z given X and W
    in G_{bar{X},overline{Z*}} (incoming edges to X removed and incoming edges to Z* removed), then
    P(Y | do(X), do(Z), W) = P(Y | do(X), W).

    Returns True if the condition holds on the graph.
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)
    Ws = _to_set(W)

    # G_barX
    g_manipulated = dmg.copy()
    g_manipulated.remove_incoming_edges(Xs)

    # compute ancestors of W in g_barx (in directed sense using augmented DAG)
    aug_barx = _build_augmented_dag(g_manipulated)
    ancestors_of_W = set()
    for w in Ws:
        if w in aug_barx.get_vertices():
            ancestors_of_W.add(w)
            ancestors_of_W.update(aug_barx.get_ancestors(w))

    # Z* are elements of Z that are NOT ancestors of any W
    Z_star = {z for z in Zs if z not in ancestors_of_W}

    # G_barX_barZstar: remove incoming edges to nodes in Z*
    g_manipulated.remove_incoming_edges(Z_star)

    cond = Xs | Ws
    return m_separated(g_manipulated, Ys, Zs, cond)




# VarSet = FrozenSet[str]
#
#
# def _fset(xs: Iterable[str]) -> VarSet:
#     return frozenset(xs)
#
#
# def _powerset_nonempty(xs: Iterable[str]) -> list[VarSet]:
#     xs = list(xs)
#     out: list[VarSet] = []
#     for r in range(1, len(xs) + 1):
#         for c in combinations(xs, r):
#             out.append(frozenset(c))
#     return out
#
#
# def _fmt_set(xs: VarSet) -> str:
#     if not xs:
#         return "∅"
#     return ", ".join(sorted(xs))
#
#
# def _fmt_prob(y: VarSet, do_x: VarSet, cond: VarSet) -> str:
#     left = _fmt_set(y)
#     cond_parts = []
#     if do_x:
#         cond_parts.append(f"do({_fmt_set(do_x)})")
#     if cond:
#         cond_parts.append(_fmt_set(cond))
#     if cond_parts:
#         return f"P({left} | {'; '.join(cond_parts)})"
#     return f"P({left})"
#
#
# @dataclass(frozen=True)
# class Query:
#     y: VarSet
#     x: VarSet
#     c: VarSet
#
#     def formula(self) -> str:
#         return _fmt_prob(self.y, self.x, self.c)
#
#
# class IdentifiabilitySolver:
#     def __init__(
#         self,
#         graph,
#         all_vars: Iterable[str],
#         rule1_applies: Callable[[object, VarSet, VarSet, VarSet, VarSet], bool],
#         rule2_applies: Callable[[object, VarSet, VarSet, VarSet, VarSet], bool],
#         rule3_applies: Callable[[object, VarSet, VarSet, VarSet, VarSet], bool],
#         max_search_nodes: int = 5000,
#         max_marginal_extra: int = 50,
#     ) -> None:
#         self.graph = graph
#         self.all_vars: VarSet = _fset(all_vars)
#         self.rule1_applies = rule1_applies
#         self.rule2_applies = rule2_applies
#         self.rule3_applies = rule3_applies
#         self.max_search_nodes = max_search_nodes
#         self.max_marginal_extra = max_marginal_extra
#
#         # protects against recursive loops
#         self._in_progress: set[Query] = set()
#
#     def identify(
#         self,
#         Y: Iterable[str],
#         X: Iterable[str],
#         Z: Iterable[str] = (),
#         W: Iterable[str] = (),
#     ) -> tuple[bool, Optional[str]]:
#         Yf = _fset(Y)
#         Xf = _fset(X)
#         Zf = _fset(Z)
#         Wf = _fset(W)
#
#         if (Yf | Xf | Zf | Wf) - self.all_vars:
#             unknown = (Yf | Xf | Zf | Wf) - self.all_vars
#             raise ValueError(f"Unknown variables: {sorted(unknown)}")
#
#         if Yf & Xf:
#             raise ValueError("Y and X must be disjoint.")
#
#         if (Zf & Xf) or (Wf & Xf):
#             raise ValueError("Conditioning variables must be disjoint from interventions X.")
#
#         q = Query(y=Yf, x=Xf, c=Zf | Wf)
#         formula = self._identify_query(q)
#         return formula is not None, formula
#
#     @lru_cache(maxsize=None)
#     def _identify_query(self, q: Query) -> Optional[str]:
#         # break recursion cycles
#         if q in self._in_progress:
#             return None
#
#         self._in_progress.add(q)
#         try:
#             return self._identify_query_impl(q)
#         finally:
#             self._in_progress.remove(q)
#
#     def _identify_query_impl(self, q: Query) -> Optional[str]:
#         # Base case: observational query
#         if not q.x:
#             return _fmt_prob(q.y, frozenset(), q.c)
#
#         # First try do-calculus-only reduction
#         direct = self._reduce_to_observational_by_search(q)
#         if direct is not None:
#             return direct
#
#         # 1. Conditional-as-ratio
#         for s in _powerset_nonempty(q.c):
#             c_rest = q.c - s
#             num_q = Query(y=q.y | s, x=q.x, c=c_rest)
#             den_q = Query(y=s, x=q.x, c=c_rest)
#
#             num = self._identify_query(num_q)
#             if num is None:
#                 continue
#
#             den = self._identify_query(den_q)
#             if den is None:
#                 continue
#
#             return f"({num}) / ({den})"
#
#         # 2. Chain rule
#         # Try this before marginalization, because it is usually safer
#         if len(q.y) >= 2:
#             y_list = sorted(q.y)
#             for r in range(1, len(y_list)):
#                 for left in combinations(y_list, r):
#                     y1 = frozenset(left)
#                     y2 = q.y - y1
#
#                     q1 = Query(y=y1, x=q.x, c=q.c | y2)
#                     q2 = Query(y=y2, x=q.x, c=q.c)
#
#                     f1 = self._identify_query(q1)
#                     if f1 is None:
#                         continue
#
#                     f2 = self._identify_query(q2)
#                     if f2 is None:
#                         continue
#
#                     return f"({f1}) * ({f2})"
#
#         # 3. Marginalization
#         # Keep this conservative because it easily causes recursion blowups.
#         candidates = list(self.all_vars - q.y - q.x - q.c)
#         for r in range(1, min(self.max_marginal_extra, len(candidates)) + 1):
#             for extra in combinations(candidates, r):
#                 T = frozenset(extra)
#                 sup_q = Query(y=q.y | T, x=q.x, c=q.c)
#                 sup_formula = self._identify_query(sup_q)
#                 if sup_formula is not None:
#                     return f"Σ_{{{_fmt_set(T)}}} {sup_formula}"
#
#         return None
#
#     def _reduce_to_observational_by_search(self, start: Query) -> Optional[str]:
#         queue = deque([start])
#         visited = {start}
#         explored = 0
#
#         while queue and explored < self.max_search_nodes:
#             explored += 1
#             q = queue.popleft()
#
#             if not q.x:
#                 return _fmt_prob(q.y, frozenset(), q.c)
#
#             # Rule 1: drop observed variables
#             for z in _powerset_nonempty(q.c):
#                 w = q.c - z
#                 if self.rule1_applies(self.graph, q.y, q.x, z, w):
#                     new_q = Query(y=q.y, x=q.x, c=w)
#                     if new_q not in visited:
#                         visited.add(new_q)
#                         queue.append(new_q)
#
#             # Rule 2: ONLY use forward direction
#             # This is much safer than allowing backward conversion.
#             for z in _powerset_nonempty(q.x):
#                 x_rest = q.x - z
#                 w = q.c
#                 if self.rule2_applies(self.graph, q.y, x_rest, z, w):
#                     new_q = Query(y=q.y, x=x_rest, c=q.c | z)
#                     if new_q not in visited:
#                         visited.add(new_q)
#                         queue.append(new_q)
#
#             # Rule 3: drop interventions
#             for z in _powerset_nonempty(q.x):
#                 x_rest = q.x - z
#                 w = q.c
#                 if self.rule3_applies(self.graph, q.y, x_rest, z, w):
#                     new_q = Query(y=q.y, x=x_rest, c=q.c)
#                     if new_q not in visited:
#                         visited.add(new_q)
#                         queue.append(new_q)
#
#         return None
#
# # def identifiable_by_do_calculus(dmg: DirectedMixedGraph, Y: Iterable, X: Iterable) -> bool:
#
# #
# # __all__ = [
# #     "rule1_applies",
# #     "rule2_applies",
# #     "rule3_applies",
# # ]
#
#
# if __name__ == "__main__":
#
#     g = DirectedMixedGraph()
#     g.add_vertices(["X", "Y", "M", "Z"])
#     g.add_directed_edge("X", "M")
#     g.add_directed_edge("M", "Y")
#     g.add_confounded_edge("X", "Y")
#
#     g.add_directed_edge("X", "Y")
#
#     # g.add_directed_edge("Z", "M")
#     # g.add_directed_edge("Z", "X")
#
#
#     solver = IdentifiabilitySolver(
#         graph=g,
#         all_vars=["X", "Y", "M", "Z"],
#         rule1_applies=rule1_applies,
#         rule2_applies=rule2_applies,
#         rule3_applies=rule3_applies,
#     )
#
#     identifiable, formula = solver.identify(Y=["Y"], X=["X"])
#     print("Identifiable:", identifiable)
#     print("Formula:", formula)
