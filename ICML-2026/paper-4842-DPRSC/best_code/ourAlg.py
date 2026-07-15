import math
import time
import numpy as np

import range_tree
import preprocessing
import find_patterns

def relative_error(estimated, true, n):
    return abs(estimated - true) / max(true, 0.001 * n)

def calc_mu_std(sum_err, sum_err2, count):
    if count <= 0:
        return (0.0, 0.0)
    mu = sum_err / count
    if count <= 1:
        return (mu, 0.0)
    var = max((sum_err2 - sum_err * mu) / (count - 1), 0.0)
    std = math.sqrt(var)
    return (mu, std)

def query_true(n, m, d, Q, test_input_nodes, logger):
    d2 = d << 1
    mag = 1
    root = None
    for attr in test_input_nodes:
        root = range_tree.insertSplit(root, attr, 0, m - 1, 0, m, d2 - 1, mag)
    logger.info('finish building range tree dynamically for query true')
    range_tree.make_consistent_split(root, mag)
    true = []
    for q in Q:
        (root, weight, noise) = range_tree.querySplit(root, 0, m - 1, q, 0, m, d2 - 1, mag)
        true.append(weight)
    return true
    
def pure_DP(n, m, d, Q, eps, test_input_nodes, pattern, logger):
    d2 = d << 1
    mag = preprocessing.pure_DP_mag(n, (math.ceil(math.log2(m)) + 1)**(d2), eps, pattern)
    root = None
    for attr in test_input_nodes:
        root = range_tree.insertSplit(root, attr, 0, m - 1, 0, m, d2 - 1, mag)
    logger.info('finish building range tree dynamically for pure DP')

    sum_err = 0.0
    sum_err2 = 0.0
    for q in Q:
        (root, weight, noise) = range_tree.querySplit(root, 0, m - 1, q, 0, m, d2 - 1, mag)
        """
        # verify the correctness
        true_weight = 0
        for attr in test_input_nodes:
            in_range = True
            for i in range(d2):
                if not (q[i][0] <= attr[i] <= q[i][1]):
                    in_range = False
                    break
            if in_range: 
                true_weight += attr[-1]
        print(true_weight, weight)
        #assert(true_weight == weight)              
        """  
        err = relative_error(weight + noise, weight, n) 
        sum_err += err
        sum_err2 += err * err
    logger.info('finish answering queries for pure DP')
    return sum_err, sum_err2

def pure_DP_qtime(n, m, d, Q, eps, test_input_nodes, pattern):
    d2 = d << 1
    mag = preprocessing.pure_DP_mag(n, (math.ceil(math.log2(m)) + 1)**(d2), eps, pattern)
    root = None
    for attr in test_input_nodes:
        root = range_tree.insertSplit(root, attr, 0, m - 1, 0, m, d2 - 1, mag)
    range_tree.make_consistent_split(root, mag)
    sum_qt = 0.0
    sum_qt2 = 0.0
    for q in Q:
        start = time.perf_counter()
        (root, weight, noise) = range_tree.querySplit(root, 0, m - 1, q, 0, m, d2 - 1, mag)
        qt = time.perf_counter() - start
        sum_qt += qt
        sum_qt2 += qt * qt
    mu, std = calc_mu_std(sum_qt, sum_qt2, len(Q))
    return mu * 1e6, std * 1e6 / math.sqrt(len(Q) - 1)

def pure_DP_prtime(n, m, d, eps, edges, h, repeat_times, pattern):
    sum_qt = 0.0
    sum_qt2 = 0.0
    for _ in range(repeat_times):
        d2 = d << 1
        dict = {}
        start = time.perf_counter()
        if (pattern == 'triangle'):
            deg = np.zeros(n, dtype=int)
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            dict = find_patterns.enumerate_triangles_project(edges, deg, n, h, d)
        elif (pattern == '2star'):
            dict = find_patterns.enumerate_2stars_project(edges, n, h, d)
        else:
            dict = find_patterns.enumerate_edges_project(edges, h, d)
        mag = preprocessing.pure_DP_mag(n, (math.ceil(math.log2(m)) + 1)**(d2), eps, pattern)
        root = None
        for attr, weight in dict.items():
            root = range_tree.insertSplit(root, (*attr, weight), 0, m - 1, 0, m, d2 - 1, mag)
        qt = time.perf_counter() - start
        sum_qt += qt    
        sum_qt2 += qt * qt
    mu, std = calc_mu_std(sum_qt, sum_qt2, repeat_times)
    se = std / 60.0 / math.sqrt(max(repeat_times - 1, 1))
    return mu / 60.0, se

def approx_DP(n, m, d, Q, d_max, eps, delta, test_input_nodes, pattern, logger):
    d2 = d << 1
    mag = preprocessing.approx_DP_mag(n, (math.ceil(math.log2(m)) + 1)**(d2), d_max, eps, delta, pattern)
    root = None
    for attr in test_input_nodes:
        root = range_tree.insertSplit(root, attr, 0, m - 1, 0, m, d2 - 1, mag)
    logger.info('finish building range tree dynamically for approximate DP')
    range_tree.make_consistent_split(root, mag)

    sum_err = 0.0
    sum_err2 = 0.0
    for q in Q:
        (root, weight, noise) = range_tree.querySplit(root, 0, m - 1, q, 0, m, d2 - 1, mag)
        err = relative_error(weight + noise, weight, n)
        sum_err += err
        sum_err2 += err * err
    logger.info('finish answering queries for approximate DP')
    return sum_err, sum_err2

def approx_DP_qtime(n, m, d, Q, d_max, eps, delta, test_input_nodes, pattern):
    d2 = d << 1
    mag = preprocessing.approx_DP_mag(n, (math.ceil(math.log2(m)) + 1)**(d2), d_max, eps, delta, pattern)
    root = None
    for attr in test_input_nodes:
        root = range_tree.insertSplit(root, attr, 0, m - 1, 0, m, d2 - 1, mag)
    range_tree.make_consistent_split(root, mag)
    sum_qt = 0.0
    sum_qt2 = 0.0
    for q in Q:
        start = time.perf_counter()
        (root, weight, noise) = range_tree.querySplit(root, 0, m - 1, q, 0, m, d2 - 1, mag)
        qt = time.perf_counter() - start
        sum_qt += qt
        sum_qt2 += qt * qt
    mu, std = calc_mu_std(sum_qt, sum_qt2, len(Q))
    return mu * 1e6, std * 1e6 / math.sqrt(len(Q) - 1)

def approx_DP_prtime(n, m, d, eps, delta, edges, h, repeat_times, pattern):
    # Pre-compute f1 once outside the loop (deterministic, same for all repeats)
    d2 = d << 1
    f1_cached = 0.0
    deg_cached = np.zeros(n, dtype=int)
    for u, v in edges:
        deg_cached[u] += 1
        deg_cached[v] += 1
    if (pattern == 'triangle'):
        f1_cached = preprocessing.calc_f1_triangle(edges, deg_cached, n)
    elif (pattern == '2star'):
        f1_cached = preprocessing.calc_f1_2star(edges, deg_cached, n)
    
    sum_qt = 0.0
    sum_qt2 = 0.0
    for _ in range(repeat_times):
        dict = {}
        start = time.perf_counter()
        if (pattern == 'triangle'):
            dict = find_patterns.enumerate_triangles_project(edges, deg_cached, n, h, d)
        elif (pattern == '2star'):
            dict = find_patterns.enumerate_2stars_project(edges, n, h, d)
        else:
            dict = find_patterns.enumerate_edges_project(edges, h, d)
        mag = preprocessing.approx_DP_mag(n, (math.ceil(math.log2(m)) + 1)**(d2), f1_cached, eps, delta, pattern)
        root = None
        for attr, weight in dict.items():
            root = range_tree.insertSplit(root, (*attr, weight), 0, m - 1, 0, m, d2 - 1, mag)
        range_tree.make_consistent_split(root, mag)
        qt = time.perf_counter() - start
        sum_qt += qt
        sum_qt2 += qt * qt
    mu, std = calc_mu_std(sum_qt, sum_qt2, repeat_times)
    se = std / 60.0 / math.sqrt(max(repeat_times - 1, 1))
    return mu / 60.0, se