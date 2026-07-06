import pandas as pd
from pathlib import Path
from collections import Counter


from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest



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

    ##############  PC 1PT ##########
    f1_a_pc_list = []
    f1_o_pc_list = []
    f1_order_pc_list = []
    non_contradiction_pc_list = []
    g_pc_list = []
    folder = Path("./res_lucas/pc_adj/")
    for file in folder.glob("*.csv"):
        pc_res_matrix = pd.read_csv(file)
        pc_res_matrix.index = pc_res_matrix.columns
        g_pc = CompletedPartiallyDirectedAcyclicDifferenceGraph()
        g_pc.add_vertices(list(pc_res_matrix))
        for i in range(len(pc_res_matrix.columns)):
            col_i = pc_res_matrix.columns[i]
            for j in range(i + 1, len(pc_res_matrix.columns)):
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




    ################  RestPC 1PT ##########
    f1_a_rest_pc_list = []
    f1_o_rest_pc_list = []
    non_contradiction_rest_pc_list = []
    f1_order_rest_pc_list = []
    g_rest_pc_list = []
    folder = Path("./res_lucas/p_value_mat/")
    for file in folder.glob("*.csv"):
        m = pd.read_csv(file)
        m.index = m.columns

        # m = m.iloc[10:20, 10:20]

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

        restpc = RestPC(sparsity=sig_level)
        restpc.g_hat = g
        restpc._uc_rule()
        suc = restpc.g_hat.get_all_unshielded_colliders()

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
        print(list_max_parents)

        g_rest_pc_list.append(restpc.g_hat)

        co = partial_causal_levels(restpc.g_hat)
        print("CO RestPC", co)
        # csuc = restpc.g_hat.get_all_cluster_super_unshielded_colliders(max_vertices_for_search=100, max_path_length=10)
        # print(csuc)

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


    ###############################
    ###############################
    ###############################
    ###############################
    ###############################

    ######## mean graph of PC and RestPC ########
    pc_res_matrix = pd.read_csv("./res_lucas/mean_matrices/pc_mean_adj_matrix.csv")
    pc_res_matrix.index = pc_res_matrix.columns
    g_pc = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    g_pc.add_vertices(list(pc_res_matrix))
    for i in range(len(pc_res_matrix.columns)):
        col_i = pc_res_matrix.columns[i]
        for j in range(i + 1, len(pc_res_matrix.columns)):
            col_j = pc_res_matrix.columns[j]
            if pc_res_matrix[col_i].loc[col_j] == 1 and pc_res_matrix[col_j].loc[col_i] == 1:
                g_pc.add_undirected_edge(col_i, col_j)
            elif pc_res_matrix[col_i].loc[col_j] == 1 and pc_res_matrix[col_j].loc[col_i] == 0:
                g_pc.add_directed_edge(col_j, col_i)
            elif pc_res_matrix[col_i].loc[col_j] == 0 and pc_res_matrix[col_j].loc[col_i] == 1:
                g_pc.add_directed_edge(col_i, col_j)

    co_pc = partial_causal_levels(g_pc)
    print("CO PC", co_pc)

    m = pd.read_csv("./res_lucas/mean_matrices/restpc_mean_p_value_matrix.csv")
    m.index = m.columns

    # m = m.iloc[10:20, 10:20]

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

    restpc = RestPC(sparsity=sig_level)
    restpc.g_hat = g
    restpc._uc_rule()
    suc = restpc.g_hat.get_all_unshielded_colliders()

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

    g_rest_pc_list.append(restpc.g_hat)

    co = partial_causal_levels(restpc.g_hat)
    print("CO RestPC", co)
