import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest


from itertools import combinations

def order_to_dense_graph(order):
    """
    Convert a partial order like [['A','B'], ['C'], ['D','E']]
    into pairwise relations.

    Returns a dict rel[(x,y)] with values:
        -1 : x before y
         0 : x same level as y
         1 : x after y
    """
    level = {}
    for i, block in enumerate(order):
        for x in block:
            level[x] = i

    nodes = list(level.keys())

    g = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    g.add_vertices(nodes)

    for x, y in combinations(nodes, 2):
        if level[x] < level[y]:
            g.add_directed_edge(x, y)
        elif level[x] > level[y]:
            g.add_directed_edge(y, x)
        else:
            g.add_undirected_edge(y, x)

    return g


def f1_score_o(g, g_hat):
    def arrowhead_set(graph):
        out = set()
        vertices = list(graph.get_vertices())
        for u in vertices:
            for v in vertices:
                if u == v:
                    continue
                if graph.is_adjacent(u, v) and graph.is_pointed_edge(u, v):
                    out.add((u, v))
        return out

    true_set = arrowhead_set(g)
    pred_set = arrowhead_set(g_hat)
    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def f1_score_a(g, g_hat):
    def adjacency_set(graph):
        out = set()
        vertices = list(graph.get_vertices())
        for u in vertices:
            for v in graph.get_adjacencies(u):
                if u == v:
                    continue
                out.add(frozenset((u, v)))
        return out

    # dir= g.get_directed_edges()
    # undir = g.get_undirected_edges()
    # true_set = dir.union(undir)
    # dir= g_hat.get_directed_edges()
    # undir = g_hat.get_undirected_edges()
    # pred_set = dir.union(undir)
    #
    # TP = set()
    # for p in list(true_set):
    #     if p in list(pred_set):
    #         if p not in TP:
    #             TP.add(p)
    #
    # FP = set()
    # for p in list(pred_set):
    #     if p not in list(true_set):
    #         if p not in FP:
    #             FP.add(p)
    #
    # FN = set()
    # for p in list(true_set):
    #     if p not in list(pred_set):
    #         if p not in FP:
    #             FN.add(p)
    # TP = len(TP)
    # FP = len(FP)
    # FN = len(FN)

    true_set = adjacency_set(g)
    pred_set = adjacency_set(g_hat)

    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def partial_causal_levels(g):
    """
    Return a partial causal order as levels.

    Output format:
        [
            ['A', 'B'],      # same level / same spot
            ['C'],
            ['D', 'E', 'F']
        ]

    Rules:
    - directed edge u -> v means u must be before v
    - undirected edge u - v is treated as bidirectional
    - directed cycles and undirected connections are collapsed into the same block
    - all blocks available at the same topological step are merged into one level

    Parameters
    ----------
    nodes : iterable
        Node labels.
    has_directed_edge : callable
        has_directed_edge(u, v) -> bool
    has_undirected_edge : callable
        has_undirected_edge(u, v) -> bool
    """
    nodes = sorted(list(g.get_vertices()), key=str)

    def has_mixed_edge(u, v, g):
        # directed edge u -> v
        if has_directed_edge(u, v, g):
            return True
        # undirected edge behaves both ways
        if has_undirected_edge(u, v, g):
            return True
        return False

    # ------------------------------------------------------------
    # 1) Strongly connected components in the mixed graph
    #    (directed edges + undirected edges as bidirectional)
    # ------------------------------------------------------------
    visited = set()
    finish_order = []

    def dfs1(u):
        visited.add(u)
        for v in nodes:
            if v != u and v not in visited and has_mixed_edge(u, v, g):
                dfs1(v)
        finish_order.append(u)

    for u in nodes:
        if u not in visited:
            dfs1(u)

    visited.clear()
    sccs = []

    def dfs2(u, comp):
        visited.add(u)
        comp.append(u)
        for v in nodes:
            if v != u and v not in visited and has_mixed_edge(v, u, g):
                dfs2(v, comp)

    for u in reversed(finish_order):
        if u not in visited:
            comp = []
            dfs2(u, comp)
            sccs.append(sorted(comp, key=str))

    # deterministic ordering of SCCs
    sccs.sort(key=lambda comp: [str(x) for x in comp])

    # map node -> block id
    block_id = {}
    for i, comp in enumerate(sccs):
        for u in comp:
            block_id[u] = i

    # ------------------------------------------------------------
    # 2) Build DAG between SCC blocks
    # ------------------------------------------------------------
    m = len(sccs)
    children = [set() for _ in range(m)]
    indeg = [0] * m

    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if has_mixed_edge(u, v, g):
                bu = block_id[u]
                bv = block_id[v]
                if bu != bv and bv not in children[bu]:
                    children[bu].add(bv)
                    indeg[bv] += 1

    # ------------------------------------------------------------
    # 3) Layered topological sort
    #    All currently parentless blocks go into the same level
    # ------------------------------------------------------------
    remaining = set(range(m))
    levels = []

    while remaining:
        current = sorted(
            [b for b in remaining if indeg[b] == 0],
            key=lambda b: [str(x) for x in sccs[b]]
        )

        if not current:
            raise ValueError("Unexpected cycle after SCC compression.")

        # Merge all variables available at this step into one same-level list
        level_nodes = []
        for b in current:
            level_nodes.extend(sccs[b])

        level_nodes = sorted(level_nodes, key=str)
        levels.append(level_nodes)

        for b in current:
            remaining.remove(b)
            for c in children[b]:
                indeg[c] -= 1

    return levels


def has_directed_edge(u, v, g):
    return (u, v) in g.get_directed_edges() and (v, u) not in g.get_directed_edges()

def has_undirected_edge(u, v, g):
    return u != v and (u, v) in g.get_undirected_edges() and (v, u) in g.get_undirected_edges()


if __name__ == '__main__':
    sig_level = 0.05

    ################ TPC 2PT ##########
    tpc_res_matrix = pd.read_csv("./res_birds/tpc_2pt/consensus_adjacency_50_percent.csv", index_col=0)

    tpc_res_matrix.index = tpc_res_matrix.columns
    g_tpc = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    nodes_t = []
    for node in list(tpc_res_matrix.columns):
        if node[-1] == "1":
            if node[:-2] not in nodes_t:
                nodes_t.append(node[:-2])
        else:
            if node not in nodes_t:
                nodes_t.append(node)
    g_tpc.add_vertices(nodes_t)

    for i in range(len(tpc_res_matrix.columns)):
        col_i = tpc_res_matrix.columns[i]
        for j in range(i + 1, len(tpc_res_matrix.columns)):
            col_j = tpc_res_matrix.columns[j]
            if col_i[-1] == "1":
                col_i_name = col_i[:-2]
            else:
                col_i_name = col_i
            if col_j[-1] == "1":
                col_j_name = col_j[:-2]
            else:
                col_j_name = col_j

            # if temp_i or temp_j:
            if col_i_name != col_j_name:
                if tpc_res_matrix[col_i].loc[col_j] == 1 and tpc_res_matrix[col_j].loc[col_i] == 1:
                    g_tpc.add_undirected_edge(col_j_name, col_i_name)
                elif tpc_res_matrix[col_i].loc[col_j] == 1 and tpc_res_matrix[col_j].loc[col_i] == 0:
                    g_tpc.add_directed_edge(col_j_name, col_i_name)
                elif tpc_res_matrix[col_i].loc[col_j] == 0 and tpc_res_matrix[col_j].loc[col_i] == 1:
                    g_tpc.add_directed_edge(col_i_name, col_j_name)

    co_tpc = partial_causal_levels(g_tpc)
    print("CO TPC", co_tpc)

    # csuc = g_tpc.get_all_cluster_super_unshielded_colliders(max_vertices_for_search=50, max_path_length=8)
    # print("CSUC TPC", csuc)

    ##############  PC 1PT ##########
    f1_a_pc_list = []
    f1_o_pc_list = []
    f1_order_pc_list = []
    non_contradiction_pc_list = []
    g_pc_list = []
    folder = Path("./res_birds/pc_adj/")
    for file in folder.glob("*.csv"):
        pc_res_matrix = pd.read_csv(file)
        pc_res_matrix.index = pc_res_matrix.columns
        g_pc = CompletedPartiallyDirectedAcyclicDifferenceGraph()
        g_pc.add_vertices(list(pc_res_matrix))
        for i in range(len(pc_res_matrix.columns)):
            col_i = pc_res_matrix.columns[i]
            for j in range(i+1, len(pc_res_matrix.columns)):
                col_j = pc_res_matrix.columns[j]
                if pc_res_matrix[col_i].loc[col_j] == 1 and pc_res_matrix[col_j].loc[col_i] == 1:
                    g_pc.add_undirected_edge(col_i, col_j)
                elif pc_res_matrix[col_i].loc[col_j] == 1 and pc_res_matrix[col_j].loc[col_i] == 0:
                    g_pc.add_directed_edge(col_j, col_i)
                elif pc_res_matrix[col_i].loc[col_j] == 0 and pc_res_matrix[col_j].loc[col_i] == 1:
                    g_pc.add_directed_edge(col_i, col_j)

        g_pc_list.append(g_pc)
        co_pc = partial_causal_levels(g_pc)
        print("CO PC", co_pc)

        f1_a_pc = f1_score_a(g_tpc, g_pc)
        f1_o_pc = f1_score_o(g_tpc, g_pc)
        f1_a_pc_list.append(f1_a_pc)
        f1_o_pc_list.append(f1_o_pc)
        print("F1 adj PC", f1_a_pc)
        print("F1 orient PC", f1_o_pc)

        g_co_tpc = order_to_dense_graph(co_tpc)
        g_co_pc = order_to_dense_graph(co_pc)
        partial_order_scores_result = f1_score_o(g_co_tpc, g_co_pc)
        f1_order_pc_list.append(partial_order_scores_result)
        print("Partial order scores (PC vs TPC):", partial_order_scores_result)

        contraditions = 0
        contraditions_list = []
        non_contradiction = 0
        for edge in g_co_tpc.get_directed_edges():
            node_i = edge[0]
            node_j = edge[1]
            if node_j in g_co_pc.get_parents(node_i):
                contraditions = contraditions + 1
                contraditions_list.append((node_i, node_j))
            else:
                non_contradiction = non_contradiction + 1

        print("Contradictions PC", contraditions)
        print("Non Contradiction PC", non_contradiction)
        try:
            print(non_contradiction/(non_contradiction+contraditions))
            non_contradiction_pc_list.append(non_contradiction/(non_contradiction+contraditions))
        except:
            print("No directed edge in PC result, skipping non-contradiction score.")
            non_contradiction_pc_list.append(1.0)  # Assuming perfect non-contradiction if no directed edges are present
        # print(non_contradiction/(non_contradiction+contraditions))


    ################  RestPC 1PT ##########
    f1_a_rest_pc_list = []
    f1_o_rest_pc_list = []
    non_contradiction_rest_pc_list = []
    f1_order_rest_pc_list = []
    g_rest_pc_list = []
    folder = Path("./res_birds/p_value_mat/")
    for file in folder.glob("*.csv"):
        m = pd.read_csv(file)
        m.index = m.columns

        g = CompletedPartiallyDirectedAcyclicDifferenceGraph()
        g.add_vertices(list(m.columns))
        count_sep = 0
        for i in range(len(m.columns)):
            col_i = m.columns[i]
            for j in range(i+1, len(m.columns)):
                col_j = m.columns[j]
                if m[col_i].loc[col_j] < sig_level:
                    g.add_undirected_edge(col_i, col_j)
                else:
                    count_sep = count_sep + 1

        print(count_sep)

        restpc = RestPC(sparsity=sig_level)
        restpc.g_hat = g
        restpc._uc_rule()
        # suc = restpc.g_hat.get_all_unshielded_colliders()
        # print(len(suc))

        max_parents = 0
        vertex_with_max_parents = None
        for v in restpc.g_hat.get_vertices():
            nb_parents = len(restpc.g_hat.get_parents(v))
            if nb_parents > max_parents:
                max_parents = nb_parents
                vertex_with_max_parents = v

        list_max_parents = [vertex_with_max_parents]
        for v in restpc.g_hat.get_vertices():
            nb_parents = len(restpc.g_hat.get_parents(v))
            if nb_parents == max_parents:
                list_max_parents.append(v)

        print(vertex_with_max_parents, max_parents)
        # print(list_max_parents)

        g_rest_pc_list.append(restpc.g_hat)
        co = partial_causal_levels(restpc.g_hat)
        print("RestPC CO", co)

        f1_a_rest_pc = f1_score_a(g_tpc, restpc.g_hat)
        f1_o_rest_pc = f1_score_o(g_tpc, restpc.g_hat)
        f1_a_rest_pc_list.append(f1_a_rest_pc)
        f1_o_rest_pc_list.append(f1_o_rest_pc)
        print("F1 adj RestPC", f1_a_rest_pc)
        print("F1 orient RestPC", f1_o_rest_pc)

        g_co_restpc = order_to_dense_graph(co)
        partial_order_scores_result = f1_score_o(g_co_tpc, g_co_restpc)
        print("Partial order scores (RestPC vs TPC):", partial_order_scores_result)
        f1_order_rest_pc_list.append(partial_order_scores_result)

        contraditions = 0
        non_contradiction = 0
        contraditions_list = []
        for edge in g_co_tpc.get_directed_edges():
            node_i = edge[0]
            node_j = edge[1]
            # if node_j in g_co_tpc.get_vertices() and node_i in g_co_tpc.get_vertices():
            # if node_j in g_co_tpc.get_parents(node_i):
            if node_j in g_co_restpc.get_parents(node_i):
                contraditions = contraditions + 1
                contraditions_list.append((node_i, node_j))
            else:
                non_contradiction = non_contradiction + 1


        print("Contradictions RestPC", contraditions)
        print("Non Contradiction RestPC", non_contradiction)
        print(non_contradiction/(non_contradiction+contraditions))
        non_contradiction_rest_pc_list.append(non_contradiction/(non_contradiction+contraditions))
        # print("Contradictions list RestPC", contraditions_list)

        # csuc = restpc.g_hat.get_all_cluster_super_unshielded_colliders(max_vertices_for_search=100, max_path_length=10)
        # print("CSUC RestPC", csuc)

    print("Average F1 adj PC", sum(f1_a_pc_list)/len(f1_a_pc_list), np.var(f1_a_pc_list))
    print("Average F1 orient PC", sum(f1_o_pc_list)/len(f1_o_pc_list),  np.var(f1_o_pc_list))
    print("Average F1 order scores PC", sum(f1_order_pc_list)/len(f1_order_pc_list), np.var(f1_order_pc_list))
    print("Average non contrad PC", sum(non_contradiction_pc_list)/len(non_contradiction_pc_list), np.var(non_contradiction_pc_list))
    print("Average F1 adj RestPC", sum(f1_a_rest_pc_list)/len(f1_a_rest_pc_list),  np.var(f1_a_rest_pc_list))
    print("Average F1 orient RestPC", sum(f1_o_rest_pc_list)/len(f1_o_rest_pc_list), np.var(f1_o_rest_pc_list))
    print("Average F1 order scores RestPC", sum(f1_order_rest_pc_list)/len(f1_order_rest_pc_list), np.var(f1_order_rest_pc_list))
    print("Average non contrad RestPC", sum(non_contradiction_rest_pc_list)/len(non_contradiction_rest_pc_list), np.var(non_contradiction_rest_pc_list))

    ###### final print of all results for each file

    # Create final graph
    g_pc_final = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    g_pc_final.add_vertices(list(g_pc_list[0].get_vertices()))

    n_graphs = len(g_pc_list)
    threshold = 0.5 * n_graphs

    directed_counts = Counter()
    undirected_counts = Counter()

    for g_pc in g_pc_list:
        # Count directed edges as ordered pairs
        for edge in g_pc.get_directed_edges():
            directed_counts[edge] += 1

        # Count undirected edges as unordered pairs
        # (avoid double-counting if both (u,v) and (v,u) are returned)
        seen_undir = set()
        for u, v in g_pc.get_undirected_edges():
            e = tuple(sorted((u, v)))
            if e not in seen_undir:
                undirected_counts[e] += 1
                seen_undir.add(e)

    # Add edges appearing in more than 50% of graphs
    for (u, v), count in directed_counts.items():
        if count > threshold:
            g_pc_final.add_directed_edge(u, v)

    for (u, v), count in undirected_counts.items():
        if count > threshold:
            # avoid conflict if already added as directed
            if (u, v) not in g_pc_final.get_directed_edges() and (v, u) not in g_pc_final.get_directed_edges():
                g_pc_final.add_undirected_edge(u, v)

    co_pc_final = partial_causal_levels(g_pc_final)
    print("CO PC final", co_pc_final)

    g_rest_pc_final = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    g_rest_pc_final.add_vertices(list(g_rest_pc_list[0].get_vertices()))

    directed_counts = Counter()
    undirected_counts = Counter()

    for g_rest_pc in g_rest_pc_list:
        # Count directed edges as ordered pairs
        for edge in g_rest_pc.get_directed_edges():
            directed_counts[edge] += 1

        # Count undirected edges as unordered pairs
        # (avoid double-counting if both (u,v) and (v,u) are returned)
        seen_undir = set()
        for u, v in g_rest_pc.get_undirected_edges():
            e = tuple(sorted((u, v)))
            if e not in seen_undir:
                undirected_counts[e] += 1
                seen_undir.add(e)

    # Add edges appearing in more than 50% of graphs
    for (u, v), count in directed_counts.items():
        if count > threshold:
            g_rest_pc_final.add_directed_edge(u, v)

    for (u, v), count in undirected_counts.items():
        if count > threshold:
            # avoid conflict if already added as directed
            if (u, v) not in g_rest_pc_final.get_directed_edges() and (v, u) not in g_rest_pc_final.get_directed_edges():
                g_rest_pc_final.add_undirected_edge(u, v)

    co_rest_pc_final = partial_causal_levels(g_rest_pc_final)
    print("CO RestPC final", co_rest_pc_final)
