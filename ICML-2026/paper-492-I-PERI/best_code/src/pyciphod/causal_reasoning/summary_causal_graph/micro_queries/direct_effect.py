from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from typing import Hashable
from pyciphod.utils.time_series.data_format import DTimeVar

def CDE_is_identifiable(Gs: SummaryCausalGraph, X : Hashable, Y : Hashable, gamma : int) -> bool:
    '''
    Implements Theorem 1 of "Average Controlled and Average Natural Micro Direct Effects in Summary
    Causal Graphs" by Ferreira and Assaad UAI 2026.
    
    :param Gs: Summary Causal Graph of interest
    :type summary: SummaryCausalGraph
    :param X: Treatment of interest
    :type X: Hashable
    :param Y: Outcome of interest
    :type Y: Hashable
    :param gamma: Lag of interest
    :type gamma: int
    :return: Boolean describing whether the Controlled Direct Effect is identifiable or not
    :rtype: bool
    '''
    
    if DTimeVar(X,gamma) not in Gs.get_possible_parents(Y,0):
        return True
    scc_Y = Gs.get_strongly_connected_components(Y)
    if len(scc_Y) > 1:
        return False
    confounded_adjacencies_Y = Gs.get_confounded_adjacencies(Y)
    ancestors_Y = Gs.get_ancestors(Y)
    if bool(set.intersection(confounded_adjacencies_Y, ancestors_Y)):
        return False
    return True

def NDE_is_identifiable(Gs: SummaryCausalGraph, X : Hashable, Y : Hashable, gamma : int) -> bool:
    '''
    Implements Theorem 2 of "Average Controlled and Average Natural Micro Direct Effects in Summary
    Causal Graphs" by Ferreira and Assaad UAI 2026.
    
    :param Gs: Summary Causal Graph of interest
    :type summary: SummaryCausalGraph
    :param X: Treatment of interest
    :type X: Hashable
    :param Y: Outcome of interest
    :type Y: Hashable
    :param gamma: Lag of interest
    :type gamma: int
    :return: Boolean describing whether the Natural Direct Effect is identifiable or not
    :rtype: bool
    '''

    if DTimeVar(X,gamma) not in Gs.get_possible_parents(Y,0):
        return True
    scc_Y = Gs.get_strongly_connected_components(Y)
    if len(scc_Y) > 1:
        return False
    scc_X = Gs.get_strongly_connected_components(X)
    if len(scc_X) > 1:
        return False
    pp_X = Gs.get_possible_parents(X, gamma)
    pp_Y = Gs.get_possible_parents(Y, 0)
    if bool(set.intersection(pp_X, pp_Y)):
        return False
    confounded_adjacencies_Y = Gs.get_confounded_adjacencies(Y)
    ancestors_Y = Gs.get_ancestors(Y)
    if bool(set.intersection(confounded_adjacencies_Y, ancestors_Y)):
        return False
    confounded_adjacencies_X = Gs.get_confounded_adjacencies(X)
    if bool(set.intersection(confounded_adjacencies_X, ancestors_Y)):
        return False
    return True


# def is_active(sg, path, adjustment_set = set()):
#     """
#     Determines whether a path is active in a (cyclic) summary causal graph sg when adjusting on adjustment_set.
#     :param sg: networkx directed graph
#     :param path: List nodes
#     :param adjustment_set: Set nodes
#     :return: bool
#     """
#     assert is_subset(path, sg.nodes)
#     assert is_subset(adjustment_set, sg.nodes)
#
#     colliders = set()
#     has_seen_right_arrow = False
#     for i in range(len(path) - 1):
#         if (sg.has_edge(path[i], path[i + 1])) and (sg.has_edge(path[i + 1], path[i])):     #V^i <-> V^{i+1}
#             pass
#         elif (sg.has_edge(path[i], path[i + 1])) and (not sg.has_edge(path[i + 1], path[i])): #V^i  -> V^{i+1}
#             if (i>0) and (path[i] in adjustment_set):
#                 return False
#             has_seen_right_arrow = True
#         elif (not sg.has_edge(path[i], path[i + 1])) and (sg.has_edge(path[i + 1], path[i])): #V^i <-  V^{i+1}
#             if has_seen_right_arrow:
#                 colliders.add(path[i])
#                 has_seen_right_arrow = False
#         else:                                                                           #V^i     V^{i+1}
#             raise ValueError("Path is not a path in sg: ", path[i], " and ", path[i + 1], " are not connected.")
#     for c in colliders:
#         if c in adjustment_set:
#             continue
#         for d in nx.descendants(sg, c):
#             if d in adjustment_set:
#                 break
#         else:
#             return False
#     return True
#
#
# def is_subset(small_iterable, big_iterable):
#     """
#     Determines whether small_iterable is a subset of big_iterable.
#     :param small_iterable: iterable nodes
#     :param big_iterable: iterable nodes
#     :return: bool
#     """
#
#     for v in small_iterable:
#         if v not in big_iterable:
#             return False
#     return True
#
#
# def is_non_direct(sg, path):
#     """
#     Determines whether path is non-direct. (path != X->Y ?)
#     :param sg: networkx directed graph
#     :param path: List nodes
#     :return: bool
#     """
#     assert is_subset(path, sg.nodes)
#     assert len(path) >= 2
#     assert path[0] != path[-1]
#
#     if len(path) > 2:
#         return True
#
#     x = path[0]
#     y = path[-1]
#     if sg.has_edge(y, x):
#         return True
#     if not sg.has_edge(x, y):
#         return False
#
#     return True
#
#
# # -----------------Code of paper {identifiability of direct from summary causal graphs}------------------------
#
# def is_identifiable_with_cycles(sg, x, y, gamma_xy, gamma_max = 1):
#     """
#     Determines whether the direct effect from x to y with lag gamma_xy is identifiable
#     for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
#     This corresponds to Theorem 1 in {identifiability of direct from summary causal graphs}.
#     :param sg: networkx directed graph
#     :param gamma_max: int
#     :param x: node
#     :param y: node
#     :param gamma_xy: int
#     :return: bool
#     """
#     assert x != y
#     assert (x in sg.nodes) and (y in sg.nodes)
#     #todo Charles: if  gamma_xy>gamma_max then identifiable and =0
#     assert (0 <= gamma_xy) and (gamma_xy <= gamma_max)
#
# # This code has exactly the same structure as Theorem 1 in {identifiability of direct effect from summary causal graphs}
#     if not sg.has_edge(x, y):
#         return True
#
#     undirected_sg = sg.to_undirected()
#     # Removed the edge X->Y from all_simple_paths except if Y->X is in SCG
#     all_simple_paths = list(nx.all_simple_paths(undirected_sg, x, y))
#     if (y, x) not in sg.edges:
#         all_simple_paths.remove([x, y])
#     for path in all_simple_paths:
#         if is_active(sg, path) and is_subset(path[1:-1], nx.descendants(sg, y)):
#             if gamma_xy == 0 and is_non_direct(sg, path):
#                 return False
#             if gamma_xy > 0:
#                 n = len(path)
#                 if n > 2:
#                     path_has_left_arrow = False
#                     for i in range(n - 1):
#                         if (not sg.has_edge(path[i], path[i + 1])) and sg.has_edge(path[i + 1], path[i]):
#                             path_has_left_arrow = True
#                             break
#                     if not path_has_left_arrow:
#                         return False
#                 if n == 2:
#                     if x in nx.descendants(sg,y):
#                         print("Error: cycle_basis is not not implemented for directed graphs")
#                         #todo Charles: networkx.exception.NetworkXNotImplemented: not implemented for directed type
#                         cycles_x = nx.cycle_basis(sg, x)
#                         for c in cycles_x:
#                             if not (y in c):
#                                 return False
#     return True
#
#
# def huge_adjustment_set(sg, x, y, gamma_xy, gamma_max = 1):
#     """
#     Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if it is identifiable
#     for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
#     Otherwise fails.
#     This corresponds to Corollary 1 in {identifiability of direct from summary causal graphs}.
#     The returned value of this function (x,S) can be interpreted as follows:
#     if x != None:
#         x is the direct effect.
#     else x == None:
#         The direct effect is can be estimated using the adjustment set S.
#         S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.
#
#     :param sg: networkx directed graph
#     :param gamma_max: int
#     :param x: node
#     :param y: node
#     :param gamma_xy: int
#     :return: (float, Set (nodes, int))
#     """
#
#     assert x != y
#     assert (x in sg.nodes) and (y in sg.nodes)
#     # There is no gamma_min_xy
#     # assert (0 <= gamma_min_xy) and (gamma_min_xy <= gamma_max)
#
#     if not sg.has_edge(x, y):
#         return (0,set())
#
#     # The function "is_identifiable_with_cycles" was called with wrong parameters
#     # if not is_identifiable_with_cycles(sg, gamma_max, x, y, 0):
#     if not is_identifiable_with_cycles(sg, x, y, 0, gamma_max):
#         raise ValueError("Direct effect of " + str(x) + "_t-" + gamma_xy + " on " + str(y) +"_t is not identifiable")
#
#     D = nx.descendants(sg, y)
#     A = sg.nodes - D
#     adjustment_set = set()
#     for v in D:
#         for gamma in range(1, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     for v in A:
#         for gamma in range(0, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     adjustment_set.remove((x,gamma_xy))
#
#     return (None,adjustment_set)
#
#
# def smaller_adjustment_set(sg, x, y, gamma_xy, gamma_max = 1):
#     """
#     Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if it is identifiable
#     for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
#     Otherwise fails.
#     This corresponds to Proposition 1 in {identifiability of direct from summary causal graphs}.
#     The returned value of this function (x,S) can be interpreted as follows:
#     if x != None:
#         x is the direct effect.
#     else x == None:
#         The direct effect is can be estimated using the adjustment set S.
#         S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.
#
#     :param sg: networkx directed graph
#     :param gamma_max: int
#     :param x: node
#     :param y: node
#     :param gamma_xy: int
#     :return: Set (nodes, int) : (v,gamma) corresponds to the vertex v_{t-gamma}
#     """
#     assert x != y
#     assert (x in sg.nodes) and (y in sg.nodes)
#     # There is no gamma_min_xy
#     # assert (0 <= gamma_min_xy) and (gamma_min_xy <= gamma_max)
#
#     if not sg.has_edge(x, y):
#         print("x not in Parents(sg,y) so direct effect is zero")
#         return (0,set())
#
#     if not is_identifiable_with_cycles(sg, gamma_max, x, y, 0):
#         raise ValueError("Direct effect of " + str(x) + "_t-" + gamma_xy + " on " + str(y) +"_t is not identifiable")
#
#     D = (nx.ancestors(sg, y)).intersection(nx.descendants(sg, y))
#     A = nx.ancestors(sg, y) - D
#     adjustment_set = set()
#     for v in D:
#         for gamma in range(1, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     for v in A:
#         for gamma in range(0, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     adjustment_set.remove((x,gamma_xy))
#
#     return (None,adjustment_set)
#
#
# if __name__ == "__main__":
#     gamma_xy = 1
#     gamma_max = 2
#
#     ascg_1 = nx.DiGraph()
#     ascg_1.add_edge("X", "Y")
#     ascg_1.add_edge("X", "U")
#     ascg_1.add_edge("U", "Z")
#     ascg_1.add_edge("Z", "W")
#     ascg_1.add_edge("W", "Y")
#
#     ascg_2 = nx.DiGraph()
#     ascg_2.add_edge("X", "Y")
#     ascg_2.add_edge("U", "X")
#     ascg_2.add_edge("Z", "U")
#     ascg_2.add_edge("Z", "W")
#     ascg_2.add_edge("W", "Y")
#
#     ascg_3 = nx.DiGraph()
#     ascg_3.add_edge("X", "Y")
#     ascg_3.add_edge("X", "U")
#     ascg_3.add_edge("U", "Z")
#     ascg_3.add_edge("W", "Z")
#     ascg_3.add_edge("Y", "W")
#
#     list_ascgs = [ascg_1, ascg_2, ascg_3]
#
#     def add_fixed_edges(scg):
#         scg.add_edges_from([("X", "X"), ("Y", "Y"), ("U", "U"), ("W", "W"), ("Z", "Z")])
#         scg.add_edge("X", "Y")
#         scg.add_edge("X", "U")
#         scg.add_edge("U", "X")
#         scg.add_edge("Y", "W")
#         scg.add_edge("W", "Y")
#
#     scg1_1 = nx.DiGraph()
#     add_fixed_edges(scg1_1)
#     scg1_1.add_edge("U", "Z")
#     scg1_1.add_edge("Z", "W")
#
#     scg1_2 = nx.DiGraph()
#     add_fixed_edges(scg1_2)
#     scg1_2.add_edge("U", "Z")
#     scg1_2.add_edge("Z", "U")
#     scg1_2.add_edge("Z", "W")
#
#     scg1_3 = nx.DiGraph()
#     add_fixed_edges(scg1_3)
#     scg1_3.add_edge("U", "Z")
#     scg1_3.add_edge("Z", "W")
#     scg1_3.add_edge("W", "Z")
#
#     scg1_4 = nx.DiGraph()
#     add_fixed_edges(scg1_4)
#     scg1_4.add_edge("U", "Z")
#     scg1_4.add_edge("W", "Z")
#
#     scg1_5 = nx.DiGraph()
#     add_fixed_edges(scg1_5)
#     scg1_5.add_edge("Z", "U")
#     scg1_5.add_edge("Z", "W")
#
#     list_scgs_identifiable = [scg1_1, scg1_2, scg1_3, scg1_4, scg1_5]
#
#     scg2_1 = nx.DiGraph()
#     add_fixed_edges(scg2_1)
#     scg2_1.add_edge("Z", "U")
#     scg2_1.add_edge("W", "Z")
#
#     scg2_2 = nx.DiGraph()
#     add_fixed_edges(scg2_2)
#     scg2_2.add_edge("Z", "U")
#     scg2_2.add_edge("W", "Z")
#     scg2_2.add_edge("Z", "W")
#
#     scg2_3 = nx.DiGraph()
#     add_fixed_edges(scg2_3)
#     scg2_3.add_edge("Z", "U")
#     scg2_3.add_edge("U", "Z")
#     scg2_3.add_edge("W", "Z")
#
#     list_scgs_identifiable_with_cond = [scg2_1, scg2_2, scg2_3]
#
#     scg3_1 = nx.DiGraph()
#     add_fixed_edges(scg3_1)
#     scg3_1.add_edge("Z", "U")
#     scg3_1.add_edge("U", "Z")
#     scg3_1.add_edge("W", "Z")
#     scg3_1.add_edge("Z", "W")
#
#     list_scgs_not_identifiable = [scg3_1]
#
#     print("ASCGs")
#     for ascg in list_ascgs:
#         res = is_identifiable_with_cycles(ascg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#
#     print("Identifiable SCGs")
#     for scg in list_scgs_identifiable:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#         # if res:
#         #     print(huge_adjustment_set(scg, "X", "Y", gamma_xy, gamma_max=gamma_max))
#
#     print("Identifiable SCGs if gamma_xy >0")
#     for scg in list_scgs_identifiable_with_cond:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#
#     print("Not identifiable SCGs")
#     for scg in list_scgs_not_identifiable:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#
#     special_scg = nx.DiGraph()
#     add_fixed_edges(special_scg)
#     special_scg.add_edge("Y", "X")
#     list_special_scgs = [special_scg]
#
#     print("Special SCGs")
#     for scg in list_special_scgs:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
