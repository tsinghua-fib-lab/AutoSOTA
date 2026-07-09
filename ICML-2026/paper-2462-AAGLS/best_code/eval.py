#!/usr/bin/env python3
import sys, time, os
sys.path.insert(0, '/repo')
import helper_functions, sparsifier, treeDP

GRAPH = '/datasets/ca-GrQc.txt'
K = 10
BISECT = 'spectralbalanced_10'

G = helper_functions.read_graph(GRAPH)
t0 = time.perf_counter()
s = sparsifier.Sparsifier(G, alg=BISECT)
t1 = time.perf_counter()
L = treeDP.solveGivenK(s.tree, K, root=s.root)
Lmapped = [res for key, res in s.mapping.items() if key in L]
t2 = time.perf_counter()
tau, cut = helper_functions.cut_set(G, Lmapped)
t3 = time.perf_counter()

print(f'METRIC real_time {t3-t0:.1f}')
print(f'METRIC sparsifier_time {t1-t0:.1f}')
print(f'METRIC dp_time {t2-t1:.1f}')
print(f'METRIC cut_verify_time {t3-t2:.1f}')
print(f'METRIC quality_tau {tau:.6f}')
print(f'METRIC selected_vertices {len(Lmapped)}')
