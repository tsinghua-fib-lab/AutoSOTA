import helper_functions
import networkx as nx

def cesa_bianchi_et_al(G, k):
    #adding support for disconnected graphs    
    H = G.copy()
    G_comps = list(nx.connected_components(H))
    rooted = [False]*len(G_comps)
    roots = [-1]*len(G_comps)
    trees = {}
    T = nx.Graph()
    for comp in G_comps:
        T = nx.union(T, helper_functions.random_spanning_tree(H.subgraph(comp)))
    L = set()
    while len(L) < k:
        components = list(nx.connected_components(T))
        largest_component = {}
        largest_size = -1
        for comp in components:
            if len(comp) > largest_size:
                largest_size = len(comp)
                largest_component = comp
        subtree = T.subgraph(largest_component)
        u, size = helper_functions.find_balanced_vertex(subtree)
        u_comp = -1
        for i in range(len(G_comps)):
            if u in G_comps[i]:
                u_comp = i
                break
        if not rooted[u_comp]:
            T_r = helper_functions.create_rooted_tree(T.subgraph(G_comps[u_comp]), u)
            for _, _, data in T_r.edges(data=True):
                data['L'] = False
            trees[u_comp] = T_r
            roots[u_comp] = u
            rooted[u_comp] = True
            T.remove_node(u)
            L.add(u)
        else:
            T.remove_node(u)
            L.add(u)
            T_r = trees[u_comp]
            root = roots[u_comp]
            while u != root and len(L) < k: 
                parent = list(T_r.in_edges(u))[0][0]
                if parent == root:
                    break
                T_r[parent][u]['L'] = True
                add_vertex = False
                for v1, v2 in T_r.out_edges(parent):
                    if v2 != u and T_r[v1][v2]['L']:
                        add_vertex = True
                        break
                if add_vertex:
                    L.add(parent)
                    if parent in T.nodes():
                        T.remove_node(parent)
                u = parent
    l_, _ = helper_functions.cut_set(G, L)
    return l_, L


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="Path to graph file")
    parser.add_argument("k", type=int, help="k")
    args = parser.parse_args()

    G = helper_functions.read_graph(args.graph)
    k = args.k

    if k >= G.number_of_nodes():
        raise Exception("k must be smaller than n!")

    print(cesa_bianchi_et_al(G,k)[0])
