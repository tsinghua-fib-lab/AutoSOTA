import numpy as np
import find_patterns
import random
import math
import array

def pure_DP_mag(n, comp_num, eps, pattern):
    GS = 0.0
    if (pattern == 'triangle' or pattern == '2star'):
        GS = n - 2
    else: # pattern == 'edge'
        GS = 1
    return GS * comp_num / eps

def approx_DP_mag(n, comp_num, d_max, eps, delta, pattern):
    if (pattern == 'edge'):
        return (math.sqrt(comp_num) * 2.0 * math.sqrt(2.0 * math.log(1.0 / delta))) / eps
    if (pattern == 'triangle'):
        m_H = 3
    else: # pattern == '2star'
        m_H = 2
    eps_prime = eps / (m_H + 1.0)
    delta_prime = delta / max(2.0 * math.exp(m_H * eps_prime) + m_H * math.exp(eps_prime) + 1, m_H * math.exp(2 * eps_prime) + math.exp(eps_prime) + 2)
    delta_double_prime = min(math.exp(-eps_prime/ 8.0), delta_prime)
    HS = 0.0
    if (pattern == 'triangle'):
        HS_2 = 1.0 + math.log(1.0 / delta_prime) / eps_prime + np.random.laplace(0, 1.0 / eps_prime)
        HS = d_max + HS_2 * math.log(1.0 / delta_prime) / eps_prime + np.random.laplace(0, HS_2 / eps_prime)
    else: # pattern == '2star'
        HS = d_max + math.log(1.0 / delta_prime) / eps_prime + np.random.laplace(0, 1.0 / eps_prime)
    mag = HS * (math.sqrt(comp_num) * 2.0 * math.sqrt(2.0 * math.log(1.0 / delta_double_prime))) / eps_prime
    return mag

def generate_skewed_queries(Q_num, m, d): #small-range queries
    queries = []
    max_len = min(m, int(20 * math.sqrt(Q_num))) 
    for _ in range(Q_num):
        length = random.randint(1, max_len)
        l = random.randint(0, m - length)
        r = l + length - 1
        queries.append([(l, m - 1), (0, r)])
    return queries
    
def generate_random_queries(Q_num, m, d): # d = 2
    queries = []
    for _ in range(Q_num):
        interval = []
        for i in range(d):
            len = random.randint(m // 2, m - 1)
            l = random.randint(0, m - len)
            r = l + len - 1
            interval.extend([(l, m - 1), (0, r)])
        queries.append(interval)
    return queries

def generate_queries(Q_num, m, d): # d = 1 (Q_num^0.5 << m to ensure |V_q| = \Theta(n))
    L = math.ceil(Q_num**0.5)
    A = list(range(L))
    B = list(range(L))
    random.shuffle(A)
    random.shuffle(B)
    Lpairs = []
    Rpairs = []
    for a in A:         
        for b in B:
            Lpairs.append((a, m - 1))
            Rpairs.append((0, m - 1 - b))
    queries = [[] for _ in range(Q_num)]
    random.shuffle(Lpairs)
    random.shuffle(Rpairs)
    for i in range(Q_num):
        queries[i].append(Lpairs[i])
        queries[i].append(Rpairs[i])
    return queries

def cmp(x, y, deg):
    return deg[x] < deg[y] or (deg[x] == deg[y] and x < y)

def calc_f1_2star(edges, deg, n):
    matrix = np.zeros((n, n),dtype=np.bool_)
    for edge in edges:
        x, y = edge
        matrix[x][y] = matrix[y][x] = True
    max = 0
    for i in range(n):
        for j in range(i + 1, n):
            cnt = deg[i] + deg[j]
            if (matrix[i][j]):
                cnt -= 1
            if (cnt > max):
                max = cnt     
    return max

def calc_f1_triangle(edges, deg, n):
    adj_di_list = [array.array('i') for _ in range(n)]
    adj_list = [array.array('i') for _ in range(n)]
    matrix = np.zeros((n, n),dtype=np.bool_)
    for edge in edges:
        x, y = edge
        if not cmp(x, y, deg):
            x, y = y, x
        adj_di_list[x].append(y)
        adj_list[x].append(y)
        adj_list[y].append(x)
        matrix[x][y] = True 
        matrix[y][x] = True

    count = {}
    for x in range(n):
        len_adj = len(adj_list[x])
        for i in range(len_adj):
            y = adj_list[x][i]
            for j in range(i + 1, len_adj):
                z = adj_list[x][j]
                p, q = y, z
                if not cmp(p, q, deg):
                    p, q = q, p
                count[(p, q)] = count.get((p, q), 0) + 1
    return max(count.values())

def graph_data_load(datasetName, pattern, n, logger, d):
    graph_file_name = f'{datasetName}/{datasetName}_graph.txt'
    pattern_file_name = f'{datasetName}/{datasetName}_{pattern}_{n}.txt'
    attr_file_name = f'{datasetName}/{datasetName}_attribute_d={d}.txt'
    # attr_file_name = f'{datasetName}/{datasetName}_attribute_original.txt'
    
    fo = open(graph_file_name)
    line = fo.readline()
    edges = set()
    deg = np.zeros(n, dtype=int) 
             
    while line:
        u, v = map(int, line.strip().split())
        if (u > v):
            u, v = v, u
        if (u != v):
            edges.add((u, v))
        line = fo.readline()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1 
    print(len(edges))
            
    fo = open(attr_file_name)
    line = fo.readline()
    a = [0] * n
    cnt = 0
    while line:
        a[cnt] = list(map(float, line.strip().split()))
        line = fo.readline()
        cnt += 1
        if (cnt == n):
            break 
    fo.close()
    logger.info(f'attribute for {datasetName} finished!')

    A = np.array(a) 
    m = n
    h = np.zeros((n, d), dtype=int)
    for j in range(d):
        col = A[:, j]
        unique_vals = sorted(set(col))  
        val_to_rank = {v: i for i, v in enumerate(unique_vals)}
        h[:, j] = [val_to_rank[x] for x in col]
    
    try:
        # the first line in patternFileName is d_max
        # the following lines are occurrences
        with open(pattern_file_name, 'r') as f:
            d_max = int(f.readline().strip())
            test_input_nodes = []
            for line in f:
                nums = list(map(int, line.strip().split()))
                w = nums[-1]
                coords = nums[:-1] 
                interval = tuple(coords)
                test_input_nodes.append((*interval, w))
    except FileNotFoundError:
        d_max = 0
        if (pattern == 'triangle'):
            d_max = calc_f1_triangle(edges, deg, n)
        elif (pattern == '2star'):
            d_max = calc_f1_2star(edges, deg, n)
        fo.close()
        logger.info('finish calc f1')
        
        w = {}
        if (pattern == 'triangle'):
            w = find_patterns.enumerate_triangles_project(edges, deg, n, h, d)
        elif (pattern == '2star'):
            w = find_patterns.enumerate_2stars_project(edges, n, h, d)
        else: # pattern == 'edge'
            w = find_patterns.enumerate_edges_project(edges, h, d)
        
        test_input_nodes = []
        for i in w:
            test_input_nodes.append((*i, w[i]))
        with open(pattern_file_name, 'w') as f:
            f.write(f"{d_max}\n")
            for attr in test_input_nodes:
                line = " ".join(map(str, attr)) + "\n"
                f.write(line)
    logger.info(f'{pattern} counting for {datasetName} finished!')   
    return (d_max, test_input_nodes, edges, h, m)
