import os
import sys
sys.path.insert(0, os.path.abspath('src'))

from pyciphod.utils.graphs.graphs import Graph


def run_tests():
    print('Test 1: simple unshielded A->C<-B')
    g = Graph()
    g.add_directed_edge('A', 'C')
    g.add_directed_edge('B', 'C')
    # res = get_cluster_super_unshielded_colliders(g, max_vertices_for_search=10)
    # print('CSUCs:', res)
    res = g.get_cluster_super_unshielded_colliders( max_vertices_for_search=10)
    print('CSUCs:', res)

    print('\nTest 2: shielded (add undirected edge A-B)')
    g2 = Graph()
    g2.add_directed_edge('A', 'C')
    g2.add_directed_edge('B', 'C')
    g2.add_undirected_edge('A', 'B')
    # res2 = get_cluster_super_unshielded_colliders(g2, max_vertices_for_search=10)
    # print('CSUCs:', res2)
    res2 = g2.get_cluster_super_unshielded_colliders(max_vertices_for_search=10)
    print('CSUCs:', res2)

    print('\nTest 3: cluster example where X={A1,A2}, Y={C1, C2}, Z={B1,B2}')
    g3 = Graph()
    # construct cluster connections: A1->C, A2->A1 (to connect within X), B1->C, B2->B1
    g3.add_directed_edge('A1', 'C1')
    g3.add_directed_edge('A2', 'A1')
    g3.add_directed_edge('B1', 'C1')
    g3.add_directed_edge('B2', 'B1')
    g3.add_directed_edge('C1', 'C2')
    #g3.add_directed_edge('D', 'C2')
    g3.add_directed_edge('D', 'C1')
    # res3 = get_cluster_super_unshielded_colliders(g3, max_vertices_for_search=12)
    # print('CSUCs:', res3)
    res3 = g3.get_cluster_super_unshielded_colliders(max_vertices_for_search=10)
    print('CSUCs:', res3)

    # for cluster in res3:
    #     print('Cluster:')
    #     for x in cluster[0]:
    #         for y in cluster[1]:
    #             for z in cluster[2]:
    #                 print(f'  {x} -> {y} <- {z}')

if __name__ == '__main__':
    run_tests()
