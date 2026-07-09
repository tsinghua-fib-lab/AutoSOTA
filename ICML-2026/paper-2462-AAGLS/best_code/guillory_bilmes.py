import helper_functions
import networkx as nx
import random
import concurrent.futures


def guillory_bilmes(G, k):
    #Compute sum of capacities
    W = 0
    n = G.number_of_nodes()
    #set capacity to 1 in case graph is unweighted
    for u, v, data in G.edges(data=True):
        if not 'capacity' in data:
            data['capacity'] = 1.
        W += data['capacity']
            
    #find the initial vertex
    best_tau = W
    best_set = {}
    for v in G.nodes():
        tau_val, cut = helper_functions.cut_set(G, {v})
        if tau_val < best_tau:
            best_tau = tau_val
            best_set = cut            
    #add a random vertex from the worst cut
    L = set()
    L.add(random.choice(list(best_set)))
    
            
    while len(L) < k:
        _, T = helper_functions.cut_set(G, L)
        L.add(random.choice(list(T)))
    l_, _ = helper_functions.cut_set(G, L)
    return l_, L

def parallel_guillory_bilmes(G, k, n_processors=4):
    #Compute sum of capacities
    W = 0
    n = G.number_of_nodes()
    #set capacity to 1 in case graph is unweighted
    for u, v, data in G.edges(data=True):
        if not 'capacity' in data:
            data['capacity'] = 1.
        W += data['capacity']
            
    #find the initial vertex
    best_tau = W
    best_set = {}
    nodes = list(G.nodes())
    vertices = []
    start = 0
    avg = n//n_processors
    for i in range(n_processors):
        end = min(start + avg + 1, len(nodes))
        vertices.append(nodes[start:end])
        start = end
    graphs = [G]*n_processors
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
        results = list(executor.map(helper_functions.cut_set, graphs, vertices))
    for res in results:
        if best_tau > res[0]:
            best_tau = res[0]
            best_set = res[1]

    #add a random vertex from the worst cut
    L = set()
    L.add(random.choice(list(best_set)))
    
    while len(L) < k:
        _, T = helper_functions.cut_set(G, L)
        L.add(random.choice(list(T)))
    l_, _ = helper_functions.cut_set(G, L)
    return l_, L


if __name__ == "__main__":
    import argparse

    n_processors = 4
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="Path to graph file")
    parser.add_argument("k", type=int, help="k")
    parser.add_argument("--n_processors", type=int, default=n_processors, help="Number of processors")
    args = parser.parse_args()

    G = helper_functions.read_graph(args.graph)
    k = args.k
    n_processors = args.n_processors

    if k >= G.number_of_nodes():
        raise Exception("k must be smaller than n!")

    print(parallel_guillory_bilmes(G,k,n_processors=n_processors)[0])
