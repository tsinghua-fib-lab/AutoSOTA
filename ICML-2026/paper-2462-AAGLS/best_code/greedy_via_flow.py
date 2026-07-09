import helper_functions
import networkx as nx
import concurrent.futures

def t_gls(G, tau, k = None, eps = 10e-5):
    W = 0
    n = G.number_of_nodes()
    if k is None:
        k = n
    
    for u, v, data in G.edges(data=True):
        if 'capacity' in data:
            W += data['capacity']
        else:
            data['capacity'] = 1.
            W += 1.
    
    if W < tau:
        return False, set()
    
    G_directed = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        G_directed.add_edge(u, v, capacity = data['capacity'])
        G_directed.add_edge(v, u, capacity = data['capacity'])
    
    s = 's' 
    t = 't'
    G_directed.add_node(s)
    G_directed.add_node(t)
    for v in G.nodes():
        G_directed.add_edge(s, v, capacity=tau)
    
    flow_value, _ = nx.maximum_flow(G_directed, s, t)
    L = set()
        
    #save some maxflows by remembering the gap from before. This is an upper bound on the gap in the future
    upper_bounds = {}
        
    while len(L) < k and flow_value < n*tau - eps:
        best = -1
        gap = 0
        for v in G.nodes():
            if v not in L: 
                if v in upper_bounds and upper_bounds[v] < (n*tau - eps - flow_value)/(k - len(L)):
                    continue
                G_directed.add_edge(v, t, capacity = tau + W + 1)
                new_flow, _ = nx.maximum_flow(G_directed, s, t)
                upper_bounds[v] = new_flow - flow_value
                if gap < new_flow - flow_value:
                    gap = new_flow - flow_value
                    best = v
                G_directed.remove_edge(v, t)
        #terminate early if nothing was found
        if gap < (n*tau - eps - flow_value)/(k - len(L)) or best == -1:
            return False, set()
        
        G_directed.add_edge(best, t, capacity = tau + W + 1)
                
        flow_value = flow_value + gap
        L.add(best)
    threshold_achieved = flow_value >= n*tau - eps
    return threshold_achieved, L


def flow_computes(G_directed, vertices, cap):
    #flow function for parallel_t_gls
    flow = -1
    best = -1
    upper_bounds = []
    t = 't'
    s = 's'
    for v in vertices:
        G_directed.add_edge(v, t, capacity = cap)
        new_flow, _ = nx.maximum_flow(G_directed, s, t)
        upper_bounds.append(new_flow)
        if new_flow > flow:
            flow = new_flow
            best = v
        G_directed.remove_edge(v, t)
    return flow, best, upper_bounds

def parallel_t_gls(G, tau, k = None, eps = 10e-4, n_processors = 4):
    #parallel implementation of the threshold gls function
    #G: graph
    #tau: threshold
    #k: budget
    #eps: accuracy
    #n_processors: number of processors
    W = 0
    n = G.number_of_nodes()
    if k is None:
        k = n
    
    for u, v, data in G.edges(data=True):
        if 'capacity' in data:
            W += data['capacity']
        else:
            data['capacity'] = 1.
            W += 1.
    
    if W < tau:
        return False, set()
    
    G_directed = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        G_directed.add_edge(u, v, capacity = data['capacity'])
        G_directed.add_edge(v, u, capacity = data['capacity'])
    
    s = 's' 
    t = 't'
    G_directed.add_node(s)
    G_directed.add_node(t)
    for v in G.nodes():
        G_directed.add_edge(s, v, capacity=tau)
    
    flow_value, _ = nx.maximum_flow(G_directed, s, t)
    L = set()
        
    #save some maxflows by remembering the gap from before. This is an upper bound on the gap in the future
    upper_bounds = {}
        
    while len(L) < k and flow_value < n*tau - eps:
        graphs = []
        nodes = set(G.nodes()) - L
        infeasible_nodes = set()
        for v in nodes:
            if v in upper_bounds and upper_bounds[v] < (n*tau - eps - flow_value)/(k - len(L)):
                infeasible_nodes.add(v)
        nodes = list(nodes - infeasible_nodes)
        for j in range(n_processors):
            graphs.append(G_directed.copy())
        avg = len(nodes) // n_processors
        start = 0
        vertices = []
        cap = [W + tau + 1]*n_processors
        for i in range(n_processors):
            end = min(start + avg + 1, len(nodes))
            vertices.append(nodes[start:end])
            start = end
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
            results = list(executor.map(flow_computes, graphs, vertices, cap))
        best = -1
        gap = 0
        for i in range(len(results)):
            if gap < results[i][0] - flow_value:
                gap = results[i][0] - flow_value
                best = results[i][1]
        if gap < (n*tau - eps - flow_value)/(k - len(L)) or best == -1:
            return False, set()
        #update upper bounds
        for j in range(n_processors):
            for i in range(len(vertices[j])):
                upper_bounds[vertices[j][i]] = results[j][2][i] - flow_value
        flow_value = flow_value + gap
        L.add(best)
        G_directed.add_edge(best, t, capacity = tau + W + 1)
    threshold_achieved = flow_value >= n*tau - eps
    return threshold_achieved, L

def gls(G, k, eps = 1e-3, parallel=False, tau_lower = None, n_processors=4):
    #gls algorithm via binary search
    #G: graph
    #k: budget
    #eps: accuracy
    #parallel: use parallelism
    #tau_lower = lower bound for threshold, this is useful if we ran an experiment for the same graph with lower budget. 
    W = 0
    n = G.number_of_nodes()
    for u, v, data in G.edges(data=True):
        if 'capacity' in data:
            W += data['capacity']
        else:
            data['capacity'] = 1.
            W += 1.
    edge_capacities = [data['capacity'] for u, v, data in G.edges(data=True)]
    min_capacity = min(edge_capacities)
    if not tau_lower is None:
        tau_1 = 0.95*tau_lower
    else:
        tau_1 = min_capacity/n
    weighted_degrees = []
    for node in G.nodes:
        weighted_degree = sum(data.get('capacity', 1) for _, _, data in G.edges(node, data=True))
        weighted_degrees.append(weighted_degree)
    weighted_degrees.sort()
    tau_2 = weighted_degrees[k]
    L = set()
    L.add(list(G.nodes)[0])
    while tau_2 - tau_1 > eps:
        tau_mid = (tau_2 - tau_1)/2  + tau_1
        threshold_achieved = False
        L_ = set()
        if parallel:
            threshold_achieved, L_ = parallel_t_gls(G, tau_mid, k, n_processors=n_processors)
        else:
            threshold_achieved, L_ = t_gls(G, tau_mid, k)
        if threshold_achieved:
            tau_1 = tau_mid
            L = L_
        else:
            tau_2 = tau_mid
    tau = helper_functions.cut_set(G, L)[0]
    return tau, L


if __name__ == "__main__":
    import argparse

    n_processors = 32
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

    print(gls(G,k,parallel=True, n_processors=n_processors)[0])
