import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from .LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.search.PermutationBased.gst import GST;
from .LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag

##### New helper function for background knowledge integration #####
from collections import defaultdict, deque

def _normalize_required_edges(required_edges, node_names, p):
    """
    Convert required edges to index form [(a_idx, b_idx), ...].
    Accepts either integer indices or node names.
    """
    if not required_edges:
        return []

    name_to_idx = {name: i for i, name in enumerate(node_names)}
    out = []
    for a, b in required_edges:
        if isinstance(a, str):
            if a not in name_to_idx or b not in name_to_idx:
                raise ValueError(f"Unknown node name in required edge: {(a, b)}")
            a, b = name_to_idx[a], name_to_idx[b]
        out.append((int(a), int(b)))

    # remove duplicates
    out = sorted(set(out))
    return out


def _build_required_maps(required_edges, p):
    """
    required_parents[child] = set(required parent indices)
    required_children[parent] = set(required child indices)
    """
    required_parents = {i: set() for i in range(p)}
    required_children = {i: set() for i in range(p)}
    for a, b in required_edges:
        if a == b:
            raise ValueError(f"Self-edge is not allowed: {(a, b)}")
        required_parents[b].add(a)
        required_children[a].add(b)
    return required_parents, required_children


def _topological_init_order(p, required_edges):
    """
    Build an initial order that already respects all required edges.
    Falls back to natural order if no constraints.
    """
    if not required_edges:
        return list(range(p))

    indeg = [0] * p
    children = [[] for _ in range(p)]
    for a, b in required_edges:
        children[a].append(b)
        indeg[b] += 1

    q = deque([i for i in range(p) if indeg[i] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for w in children[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)

    if len(order) != p:
        raise ValueError("required_edges contain a cycle, so they cannot all be true in a DAG order.")

    return order


def _position_map(order):
    return {v: i for i, v in enumerate(order)}


def _order_respects_required(order, required_edges):
    pos = _position_map(order)
    return all(pos[a] < pos[b] for a, b in required_edges)

###########################################################

def _total_score(order_, gsts_):
    total = 0.0  # start as Python float
    prefix = []
    for w in order_:
        val = gsts_[w].trace(prefix)
        if not np.isfinite(val):
            return float('-inf')
        total += float(val)        # force Python float accumulation
        prefix.append(w)
    return float(total)            # <- ensure native float

def boss(
    X: np.ndarray,
    score_func: str = "local_score_BIC_from_cov",
    parameters: Optional[Dict[str, Any]] = None,
    verbose: Optional[bool] = True,
    node_names: Optional[List[str]] = None,
    required_edges: Optional[List[tuple]] = None,
) -> GeneralGraph:
    """
    Perform a best order score search (BOSS) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BIC_from_cov', 'local_score_BDeu')).
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.
    verbose : whether to print the time cost and verbose output of the algorithm.

    Returns
    -------
    G : learned causal graph, where G.graph[j,i] = 1 and G.graph[i,j] = -1 indicates i --> j, G.graph[i,j] = G.graph[j,i] = -1 indicates i --- j.
    """

    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    if score_func == "local_score_CV_general":
        # % k-fold negative cross validated likelihood based on regression in RKHS
        if parameters is None:
            parameters = {
                "kfold": 10,  # 10 fold cross validation
                "lambda": 0.01,
            }  # regularization parameter
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters
        )
    elif score_func == "local_score_marginal_general":
        # negative marginal likelihood based on regression in RKHS
        parameters = {}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters
        )
    elif score_func == "local_score_CV_multi":
        # k-fold negative cross validated likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {
                "kfold": 10,
                "lambda": 0.01,
                "dlabel": {},
            }  # regularization parameter
            for i in range(X.shape[1]):
                parameters["dlabel"]["{}".format(i)] = i
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters
        )
    elif score_func == "local_score_marginal_multi":
        # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(X.shape[1]):
                parameters["dlabel"]["{}".format(i)] = i
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters
        )
    elif score_func == "local_score_BIC":
        # SEM BIC score
        warnings.warn("Using 'local_score_BIC_from_cov' instead for efficiency")
        if parameters is None:
            parameters = {"lambda_value": 2}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )
    elif score_func == "local_score_BIC_from_cov":
        # SEM BIC score
        if parameters is None:
            parameters = {"lambda_value": 2}
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
        )
    elif score_func == "local_score_BDeu":
        # BDeu score
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None
        )
    else:
        raise Exception("Unknown function!")
    
    score = localScoreClass
    gsts = [GST(i, score) for i in range(p)]

    node_names = [("X%d" % (i + 1)) for i in range(p)] if node_names is None else node_names
    required_edges = _normalize_required_edges(required_edges, node_names, p)
    required_parents, required_children = _build_required_maps(required_edges, p)
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    G = GeneralGraph(nodes)

    runtime = time.perf_counter()
        
    #order = [v for v in range(p)]
    order = _topological_init_order(p, required_edges)

    gsts = [GST(v, score) for v in order]
    parents = {v: [] for v in order}
    
    variables = [v for v in order]
    # while True:
    #     improved = False
    #     random.shuffle(variables)
    #     if verbose:
    #         for i, v in enumerate(order):
    #             parents[v].clear()
    #             gsts[v].trace(order[:i], parents[v])
    #         sys.stdout.write("\rBOSS edge count: %i    " % np.sum([len(parents[v]) for v in range(p)]))
    #         sys.stdout.flush()

    #     for v in variables:
    #         improved |= better_mutation(v, order, gsts)
    #     if not improved: break
        # --- safeguards & helpers ---
    tol = 1e-8                  # strict improvement tolerance
    max_rounds = 2000           # hard stop (can increase)
    detect_cycles = True        # set False if you don't want the check

    def _total_score(order_, gsts_):
        total = 0.0
        prefix = []
        for w in order_:
            val = gsts_[w].trace(prefix)
            if not np.isfinite(val):
                return -np.inf
            total += val
            prefix.append(w)
        return total

    rounds = 0
    visited = set()

    # (re)build the working list each sweep so it reflects the current order
    while True:
        rounds += 1
        if rounds > max_rounds:
            if verbose:
                print("\n[BOSS] stopping: reached max_rounds =", max_rounds)
            break

        variables = list(order)

        # optional: cycle detection on (order, score)
        current_total = _total_score(order, gsts)
        if detect_cycles:
            qscore = float('-inf') if not np.isfinite(current_total) else float(np.round(current_total, 10))
            sig = (tuple(map(int, order)), qscore)
            if sig in visited:
                if verbose:
                    print("\n[BOSS] stopping: detected cycle at score", float(current_total))
                break
            visited.add(sig)

        # live edge-count preview (as you had)
        if verbose:
            for i, v in enumerate(order):
                parents[v].clear()
                gsts[v].trace(order[:i], parents[v])
            sys.stdout.write("\rBOSS edge count: %i    " % sum(len(parents[v]) for v in range(p)))
            sys.stdout.flush()

        # one full sweep
        improved = False
        random.shuffle(variables)
        for v in variables:
            #if better_mutation(v, order, gsts, tol):   # NOTE: changed signature
            if better_mutation(
                v,
                order,
                gsts,
                required_parents=required_parents,
                required_children=required_children,
                tol=tol,
            ):
                improved = True

        # require a net improvement across the sweep
        new_total = _total_score(order, gsts)
        if (not improved) or (not np.isfinite(new_total)) or (new_total <= current_total + tol):
            break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])
        # hard-enforce required parents that are available in the prefix
        forced = required_parents.get(v, set())
        missing = forced.difference(parents[v])
        for pa in sorted(missing):
            if pa not in order[:i]:
                raise RuntimeError(
                    f"Required parent {pa} for node {v} is not before the child in the final order."
                )
            parents[v].append(pa)

        parents[v] = sorted(set(parents[v]))

    runtime = time.perf_counter() - runtime
    
    if verbose:
        sys.stdout.write("\nBOSS completed in: %.2fs \n" % runtime)
        sys.stdout.flush()

    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)

    return G


def reversed_enumerate(iter, j):
    for w in reversed(iter):
        yield j, w
        j -= 1

def better_mutation(v, order, gsts, required_parents=None, required_children=None, tol=1e-8):
    if required_parents is None:
        required_parents = {}
    if required_children is None:
        required_children = {}
    i = order.index(v)
    p = len(order)

    # v must be AFTER all its required parents
    min_pos = 0
    for pa in required_parents.get(v, set()):
        min_pos = max(min_pos, order.index(pa) + 1)

    # v must be BEFORE all its required children
    max_pos = p
    for ch in required_children.get(v, set()):
        max_pos = min(max_pos, order.index(ch))

    # legal insertion positions are j in [min_pos, max_pos]
    legal = np.zeros(p + 1, dtype=bool)
    legal[min_pos:max_pos + 1] = True

    # initialize to -inf so we can safely take argmax even if some positions are invalid
    scores = np.full(p + 1, -np.inf, dtype=float)

    # forward pass: score placing v before each position j
    prefix = []
    accum = 0.0
    for j, w in enumerate(order):
        sv = gsts[v].trace(prefix)
        if legal[j] and np.isfinite(sv) and np.isfinite(accum):
            scores[j] = sv + accum
        if v != w:
            sw = gsts[w].trace(prefix)
            accum = accum + sw if np.isfinite(sw) and np.isfinite(accum) else -np.inf
            prefix.append(w)

    sv_end = gsts[v].trace(prefix)
    if legal[p] and np.isfinite(sv_end) and np.isfinite(accum):
        scores[p] = sv_end + accum
    else:
        scores[j] = -np.inf

    # choose best insertion index
    if not np.any(np.isfinite(scores)):
        return False
    best = int(np.nanargmax(scores))  # nan-safe argmax; we set invalid to -inf anyway

    # accept only if STRICTLY better by > tol
    if not (scores[best] > scores[i] + tol):
        return False

    # perform the move
    order.remove(v)
    order.insert(best - int(best > i), v)
    return True
    