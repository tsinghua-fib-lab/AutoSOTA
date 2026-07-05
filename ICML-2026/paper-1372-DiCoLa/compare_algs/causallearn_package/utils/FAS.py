from __future__ import annotations

from itertools import combinations
from typing import List, Dict, Tuple, Set

from numpy import ndarray
from tqdm.auto import tqdm

from compare_algs.causallearn_package.graph.GeneralGraph import GeneralGraph
from compare_algs.causallearn_package.graph.GraphClass import CausalGraph
from compare_algs.causallearn_package.graph.Node import Node
from compare_algs.causallearn_package.utils.PCUtils.Helper import append_value
from compare_algs.causallearn_package.utils.cit import *
from compare_algs.causallearn_package.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def fas(data: ndarray, nodes: List[Node], independence_test_method: CIT_Base, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1,
        verbose: bool = False, stable: bool = False, show_progress: bool = True) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int, Set[int]], float]]:
    """
    Implements the "fast adjacency search" used in several causal algorithm in this file. In the fast adjacency
    search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
    of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
    procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
    the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
    search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
    S({x, y}) is returned for edges x *-* y that have been removed.

    Parameters
    ----------
    data: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    nodes: The search nodes.
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    knowledge: background background_knowledge
    depth: the depth for the fast adjacency search, or -1 if unlimited
    verbose: True is verbose output should be printed or logged
    stable: run stabilized skeleton discovery if True (default = True)
    show_progress: whether to use tqdm to show progress bar
    Returns
    -------
    graph: Causal graph skeleton, where graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j.
    sep_sets: Separated sets of graph
    test_results: Results of conditional independence tests
    """
    ## ------- check parameters ------------
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if not all(isinstance(node, Node) for node in nodes):
        raise TypeError("'nodes' must be 'List[Node]' type!")
    if not isinstance(independence_test_method, CIT_Base):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    if knowledge is not None and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'knowledge' must be 'BackgroundKnowledge' type!")
    if type(depth) != int or depth < -1:
        raise TypeError("'depth' must be 'int' type >= -1!")
    ## ------- end check parameters ------------

    if depth == -1:
        depth = float('inf')

    no_of_var = data.shape[1]
    node_names = [node.get_name() for node in nodes]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(independence_test_method)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    test_results: Dict[Tuple[int, int, Set[int]], float] = {}

    def remove_if_exists(x: int, y: int) -> None:
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)

    # IDEA-03: Pairwise marginal independence pre-screening using correlation matrix.
    # Removes obviously independent edges with O(n^2) cheap correlation lookups,
    # avoiding CI test overhead and reducing adjacency sets for depth >= 1.
    # Uses a conservative threshold (higher p-value) to avoid false edge removals.
    if hasattr(independence_test_method, 'correlation_matrix'):
        corr_mat = independence_test_method.correlation_matrix
        n_samples = independence_test_method.sample_size
        from scipy.stats import norm as norm_scipy
        # Pre-screening threshold: p > 0.5 (very conservative, only remove clearly independent pairs)
        # z = 0.5 * ln((1+r)/(1-r)) * sqrt(n - 3)
        # For p=0.5: z_crit = norm.ppf(0.75) ≈ 0.674
        # r_threshold = tanh(z_crit / sqrt(n - 3))
        z_crit_prescreen = norm_scipy.ppf(0.75)  # p=0.5 for two-sided test
        r_threshold = np.tanh(z_crit_prescreen / np.sqrt(max(n_samples - 3, 1)))
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                if abs(corr_mat[i, j]) < r_threshold:
                    remove_if_exists(i, j)
                    remove_if_exists(j, i)
                    append_value(cg.sepset, i, j, ())
                    append_value(cg.sepset, j, i, ())
                    sep_sets[(i, j)] = set()
                    sep_sets[(j, i)] = set()

    # Get correlation matrix for sorting conditioning sets (IDEA-04)
    if hasattr(independence_test_method, 'correlation_matrix'):
        corr_mat = independence_test_method.correlation_matrix
    else:
        corr_mat = None

    var_range = tqdm(range(no_of_var), leave=True) if show_progress \
        else range(no_of_var)
    current_depth: int = -1
    # IDEA-02 (ICD-style): Track active nodes that had edge removals in previous depth.
    # Only nodes with removals need testing at the next depth.
    active_nodes = set(range(no_of_var))
    while cg.max_degree() - 1 > current_depth and current_depth < depth:
        current_depth += 1
        edge_removal = set()
        next_active = set()
        for x in var_range:
            if show_progress:
                var_range.set_description(f'Depth={current_depth}, working on node {x}')
                var_range.update()
            # ICD-style: skip nodes that had no removals at previous depth (depth >= 1)
            if current_depth >= 1 and x not in active_nodes:
                continue
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < current_depth - 1:
                continue
            for y in Neigh_x:
                sepsets = set()
                if (knowledge is not None and
                    knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    if not stable:
                        remove_if_exists(x, y)
                        remove_if_exists(y, x)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        sep_sets[(x, y)] = set()
                        sep_sets[(y, x)] = set()
                        next_active.add(x)
                        next_active.add(y)
                        break
                    else:
                        edge_removal.add((x, y))  # after all conditioning sets at
                        edge_removal.add((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                # IDEA-04: Sort conditioning set candidates by correlation strength (descending).
                # Variables more correlated with x are more likely to be in the separating set,
                # so testing them first finds independence faster and breaks out of the loop earlier.
                if corr_mat is not None and len(Neigh_x_noy) > 0:
                    sorted_neighbors = sorted(Neigh_x_noy,
                        key=lambda n: abs(corr_mat[x, n]) if x < corr_mat.shape[0] and n < corr_mat.shape[1] else 0,
                        reverse=True)
                else:
                    sorted_neighbors = Neigh_x_noy
                for S in combinations(sorted_neighbors, current_depth):
                    p = cg.ci_test(x, y, S)
                    test_results[(x, y, S)] = p
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            remove_if_exists(x, y)
                            remove_if_exists(y, x)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            sep_sets[(x, y)] = set(S)
                            sep_sets[(y, x)] = set(S)
                            next_active.add(x)
                            next_active.add(y)
                            break
                        else:
                            edge_removal.add((x, y))  # after all conditioning sets at
                            edge_removal.add((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        # Update active nodes for next depth (ICD-style)
        active_nodes = next_active

        for (x, y) in edge_removal:
            remove_if_exists(x, y)
            if cg.sepset[x, y] is not None:
                origin_set = set(l_in for l_out in cg.sepset[x, y]
                                 for l_in in l_out)
                sep_sets[(x, y)] = origin_set
                sep_sets[(y, x)] = origin_set

    return cg.G, sep_sets, test_results
