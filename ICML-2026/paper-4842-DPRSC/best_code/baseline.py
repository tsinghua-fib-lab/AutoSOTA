import numpy as np
import time
import concurrent.futures
import math

import find_patterns
import preprocessing

def relative_error(estimated, true, n):
    return abs(estimated - true) / max(true, 0.001 * n)

def calc_mu_std(sum_err, sum_err2, count):
    mu = sum_err / count
    std = math.sqrt((sum_err2 - sum_err * mu) / (count - 1))
    return (mu, std)

def base_comp_single(n, Q_num, mag, true):
    sum_err = 0.0
    sum_err2 = 0.0
    for i in range(Q_num):
        err = relative_error(true[i] + np.random.laplace(0, mag), true[i], n)
        sum_err += err
        sum_err2 += err * err
    return (sum_err, sum_err2)
            
def base_comp(n, Q_num, repeat_times, eps, true, pattern, logger):
    sum_err = 0.0
    sum_err2 = 0.0
    mag = preprocessing.pure_DP_mag(n, Q_num, eps, pattern)
    
    max_workers = 120
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future = [executor.submit(base_comp_single, n, Q_num, mag, true) for _ in range(repeat_times)]
        for i in range(repeat_times):
            err, err2 = future[i].result()
            sum_err += err
            sum_err2 += err2
    logger.info('finish answering queries in base_comp') 
    return calc_mu_std(sum_err, sum_err2, Q_num * repeat_times)

def base_comp_qtime(n, d, Q, eps, edges, h, pattern):
    mag = preprocessing.pure_DP_mag(n, len(Q), eps, pattern)
    sum_qt = 0.0
    sum_qt2 = 0.0
    for q in Q:
        start = time.perf_counter()
        deg = np.zeros(n, dtype=int)  
        new_edges = []
        ans = 0.0
        for x, y in edges:
            flag = True
            for i in range(d):
                l = i << 1
                r = l + 1
                flag = False
                if (q[l][0] <= h[x, i] <= q[r][1] and q[l][0] <= h[y, i] <= q[r][1]):
                    flag = True
                if (flag == False):
                    break 
            if (flag == True):
                new_edges.append((x, y))
                deg[x] += 1
                deg[y] += 1
        if (pattern == 'triangle'):
            ans = find_patterns.enumerate_triangles(new_edges, deg, n)
        elif (pattern == '2star'):
            ans = find_patterns.enumerate_2stars(deg, n)
        ans += np.random.laplace(0, mag)
        qt = time.perf_counter() - start
        sum_qt += qt
        sum_qt2 += qt * qt
    mu, std = calc_mu_std(sum_qt, sum_qt2, len(Q))
    return mu * 1e6, std * 1e6 / math.sqrt(len(Q) - 1)

def base_comp_ADP_single(n, Q_num, mag, true):
    sum_err = 0.0
    sum_err2 = 0.0
    for i in range(Q_num):
        err = relative_error(true[i] + np.random.laplace(0, mag), true[i], n)
        sum_err += err
        sum_err2 += err * err
    return (sum_err, sum_err2)

def base_comp_ADP(n, Q_num, repeat_times, d_max, eps, delta, true, pattern, logger):
    sum_err = 0.0
    sum_err2 = 0.0
    mag = preprocessing.approx_DP_mag(n, Q_num, d_max, eps, delta, pattern)
    
    max_workers = 120
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future = [executor.submit(base_comp_ADP_single, n, Q_num, mag, true) for _ in range(repeat_times)]
        for i in range(repeat_times):
            err, err2 = future[i].result()
            sum_err += err
            sum_err2 += err2
    logger.info('finish answering queries in base_comp_ADP') 
    return calc_mu_std(sum_err, sum_err2, Q_num * repeat_times)

def base_comp_ADP_qtime(n, d, Q, d_max, eps, delta, edges, h, pattern):
    mag = preprocessing.approx_DP_mag(n, len(Q), d_max, eps, delta, pattern)
    sum_qt = 0.0
    sum_qt2 = 0.0
    for q in Q:
        start = time.perf_counter()
        deg = np.zeros(n, dtype=int)  
        new_edges = []
        ans = 0.0
        for x, y in edges:
            flag = True
            for i in range(d):
                l = i << 1
                r = l + 1
                flag = False
                if (q[l][0] <= h[x, i] <= q[r][1] and q[l][0] <= h[y, i] <= q[r][1]):
                    flag = True
                if (flag == False):
                    break 
            if (flag == True):
                new_edges.append((x, y))
                deg[x] += 1
                deg[y] += 1
        if (pattern == 'triangle'):
            ans = find_patterns.enumerate_triangles(new_edges, deg, n)
        elif (pattern == '2star'):
            ans = find_patterns.enumerate_2stars(deg, n)
        ans += np.random.laplace(0, mag)
        qt = time.perf_counter() - start
        sum_qt += qt
        sum_qt2 += qt * qt
    mu, std = calc_mu_std(sum_qt, sum_qt2, len(Q))
    return mu * 1e6, std * 1e6 / math.sqrt(len(Q) - 1)

def base_comp_ADP_prtime(n, d, edges, h, repeat_times, pattern):
    if (pattern == 'edge'):
        return 0, 0    
    sum_qt = 0.0
    sum_qt2 = 0.0
    for _ in range(repeat_times):
        start = time.perf_counter()
        d2 = d << 1
        f1 = 0.0
        deg = np.zeros(n, dtype=int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if (pattern == 'triangle'):
            f1 = preprocessing.calc_f1_triangle(edges, deg, n)
        elif (pattern == '2star'):
            f1 = preprocessing.calc_f1_2star(edges, deg, n)
        qt = time.perf_counter() - start
        sum_qt += qt
        sum_qt2 += qt * qt    
    mu, std = calc_mu_std(sum_qt, sum_qt2, repeat_times)
    return mu / 60.0, std / 60.0 / math.sqrt(repeat_times - 1)
