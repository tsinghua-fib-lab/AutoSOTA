import numpy as np
from util import *
from eq_search import eq_search_v2
from joblib import Parallel, delayed
from math import factorial
from copy import deepcopy

def term2list(term, n):
    orders = [0] * n
    for index, order in term:
        if 0 < index <= n:
            orders[index - 1] = order
    return orders

def interactionsBySkeleton(skeleton_terms, n):
    """
    Generate interaction items from skeleton items (find adjacent u-v-w), only for triples with at least 2 skeleton edges.
    """
    edges = set()
    for st in skeleton_terms:
        term = list(st)
        edges.add(frozenset({term[0][0] - 1, term[1][0] - 1}))

    rec = set()
    for a, b, c in combinations(range(n), 3):
        edge_count = (
            (frozenset({a, b}) in edges)
            + (frozenset({a, c}) in edges)
            + (frozenset({b, c}) in edges)
        )
        if edge_count >= 2:
            rec.add((a, b, c))
            rec.add((a, a, b, c))
            rec.add((a, b, b, c))
            rec.add((a, b, c, c))
            rec.add((a, a, b, b, c))
            rec.add((a, a, b, c, c))
            rec.add((a, b, b, c, c))
            rec.add((a, b, c, c, c))
            rec.add((a, c, b, b, b))
            rec.add((b, c, a, a, a))
            rec.add((a, b, b, c, c, c))
            rec.add((b, a, a, c, c, c))
            rec.add((c, a, a, b, b, b))
            rec.add((a, c, c, b, b, b))
            rec.add((b, c, c, a, a, a))
    return list(rec)

def process_term(term, data, n, bootstrap_round, p_value, threshold, seed=42):
    """Process a single term"""
    order_list = term2list(term, n)
    factorial_product = lambda lst: np.prod([factorial(order) for order in lst])
    # Generate a deterministic seed for each term
    local_seed = seed + abs(hash(term)) % (2**32)
    bootstrap_rec = bootstrap_test(data, order_list, round=bootstrap_round, seed=local_seed, one=False)/factorial_product(order_list)
    # NOTE: You can obtain a better performance by adjusting the threshold according to the order of the term
    # flag = hypothesis_test(
    #     bootstrap_rec,
    #     p=p_value,
    #     interval=[-threshold, threshold]
    # )
    flag = hypothesis_test_significance_one_side(
        bootstrap_rec,
        p=p_value
    )
    return term, bootstrap_rec, flag


def learn_skeleton(data:np.ndarray, threshold=1e-4, p_value=0.05, bootstrap_round=20, n_jobs=-1, seed=42):
    """ Learn the skeleton of the causal graph """
    n = data.shape[1]
    interactions = [(u, v) for u in range(0, n) for v in range(u+1, n)]
    terms = interactions2terms(interactions)
    # parallel version
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_term)(term, data, n, bootstrap_round, p_value, threshold, seed)
        for term in list(terms)
    )
    # debug version
    # results = []
    # for term in list(terms):
    #     result = process_term(term, data, n, bootstrap_round, p_value, threshold, seed)
    #     results.append(result)
    skeleton_terms = []
    for result in results:
        term, _, flag = result
        if flag == False:
            skeleton_terms.append(term)
    return skeleton_terms

def contradictory_check(data:np.ndarray, u:int, v:int):
    """
    Resolving R2 conflicts
    """
    u = u-1
    v = v-1
    n = data.shape[1]
    z_list = np.ones(n) * 0.3
    z_list[[u, v]] = 0.9

    # u->v
    order_list = np.zeros(n, dtype=int)
    order_list[[u, v]] = 1, 2
    d3_PGF = diff_get_epgf_gradient_np(data.T, z_list, order_list=order_list)
    order_list[[u, v]] = 1, 0
    d1_PGF = diff_get_epgf_gradient_np(data.T, z_list, order_list=order_list)
    res1 = d3_PGF * d1_PGF

    # v->u
    order_list = np.zeros(n, dtype=int)
    order_list[[u, v]] = 2, 1
    d3_PGF = diff_get_epgf_gradient_np(data.T, z_list, order_list=order_list)
    order_list[[u, v]] = 0, 1
    d1_PGF = diff_get_epgf_gradient_np(data.T, z_list, order_list=order_list)
    res2 = d3_PGF * d1_PGF
    return res1 > res2


def learn_r2(data:np.ndarray, skeleton_terms, threshold=1e-4, p_value=0.05, bootstrap_round=20, n_jobs=-1, seed=42):
    n = data.shape[1]
    # parallel version: Collect all r2 terms and call process_term in parallel at once.
    all_terms = []
    for st in skeleton_terms:
        term = list(deepcopy(st))
        all_terms.append(frozenset({(term[0][0], 1), (term[1][0], 2)}))
        all_terms.append(frozenset({(term[0][0], 2), (term[1][0], 1)}))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_term)(term, data, n, bootstrap_round, p_value, threshold, seed)
        for term in all_terms
    )
    # debug version
    # results = []
    # for st in skeleton_terms:
    #     term = list(deepcopy(st))
    #     results.append(process_term(term=frozenset({(term[0][0], 1), (term[1][0], 2)}), data=data, n=n, bootstrap_round=bootstrap_round, p_value=p_value, threshold=threshold, seed=seed))
    #     results.append(process_term(term=frozenset({(term[0][0], 2), (term[1][0], 1)}), data=data, n=n, bootstrap_round=bootstrap_round, p_value=p_value, threshold=threshold, seed=seed))
    r2_terms = []
    for i in range(0, len(results), 2):
        _, rec1, flag1 = results[i]
        _, rec2, flag2 = results[i+1]
        term = list(deepcopy(skeleton_terms[i // 2]))
        if flag1|flag2 == False:
            # Both directions significant: use relative magnitude to decide
            mean1 = np.mean(rec1)
            mean2 = np.mean(rec2)
            if mean1 >= mean2:
                r2_terms.append(frozenset({(term[0][0], 1), (term[1][0], 2)}))
            else:
                r2_terms.append(frozenset({(term[0][0], 2), (term[1][0], 1)}))
        elif flag1^flag2 == True:
            # only one direction
            if flag1 == False:
                r2_terms.append(frozenset({(term[0][0], 1), (term[1][0], 2)}))
            else:
                r2_terms.append(frozenset({(term[0][0], 2), (term[1][0], 1)}))
    return r2_terms


def pgf_confounder_partial(data:np.ndarray, skeleton_threshold=1e-4, r2_threshold=1e-4, threshold=1e-4, p_value=0.05, bootstrap_round=20, interactions=None, n_jobs=-1, seed=42, verbose=False):
    n = data.shape[1]

    # learn skeleton
    skeleton_terms = learn_skeleton(data, threshold=skeleton_threshold, p_value=p_value, bootstrap_round=bootstrap_round, n_jobs=n_jobs, seed=seed)
    if verbose:
        print(f"skeleton_terms:\n {skeleton_terms}")

    # learn confounded edges
    r2_terms = learn_r2(data, skeleton_terms, threshold=r2_threshold, p_value=p_value, bootstrap_round=bootstrap_round, n_jobs=n_jobs, seed=seed)
    if verbose:
        print(f"confounded_edge_terms:\n {r2_terms}")

    # learn leftover interactions
    left_interactions = interactionsBySkeleton(skeleton_terms, n)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_term)(term, data, n, bootstrap_round, p_value, threshold, seed)
        for term in interactions2terms(left_interactions)
    )
    if verbose:
        print(f"left_interactions_terms:\n {np.array([result[0] for result in results if result[2] == False])}")

    rec = skeleton_terms + r2_terms
    for result in results:
        term, _, flag = result
        if flag == False:
            rec.append(term)

    mag = eq_search_v2(n, list(rec))
    return rec, mag



if __name__ == "__main__":
    from util import pbscm, graph2pgf
    from PGF import PGF
    case_graph =(
        3,
        [.02, .03, .04, .05],
        [
            [0, 0, 0.3, 0],
            [0, 0, 0.4, 0],
            [0, 0, 0, 0],
            [0.5, 0.6, 0, 0],
        ]
    )
    n = case_graph[0]
    mu = case_graph[1]
    graph = case_graph[2]
    print(np.array(graph))

    pgf = PGF(graph)
    expr = pgf.compute(mu, s=[pgf.s[k] if k < n else 1 for k in range(len(mu))])
    print("True PGF expression:")
    print(expr.as_expr())

    data = pbscm(graph=graph, mu=mu, sample=10000)
    data = data[:, :n]

    terms, mag = pgf_confounder_partial(data, bootstrap_round=200, p_value=.05, verbose=True)
    print("Learned PPADMG:")
    print(mag2graph(mag))