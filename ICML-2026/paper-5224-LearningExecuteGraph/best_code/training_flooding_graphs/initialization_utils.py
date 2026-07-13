"""
Enumerate all undirected graphs on n nodes with self-loops and max degree <= D,
run flooding of all l-bit messages from every possible source node,
compute true 2-hop coloring (colors in {1,...,D^2+1}),
and produce per-case outputs:
    - adjacency matrix A (numpy array)
    - X_test_list (list of node samples as requested)
    - final_messages (list of l-bit messages at each node after flooding)
"""

import itertools
from collections import defaultdict
import numpy as np
import networkx as nx

def nested_dict(depth):
    """Return a nested defaultdict of specified depth (leaf is a normal dict)."""
    if depth <= 1:
        return {}
    return defaultdict(lambda: nested_dict(depth - 1))

def get_sample(l, d):
    sample = {
        'message': [0]*l,
        'sent': [0]*l,
        'sent_message': [[0]*l for _ in range(d)],
        'my_slot': [[0]*l for _ in range(d)],
        'received': [[0]*l for _ in range(d)],
        'message_pipe': [[0]*l for _ in range(d)],
        'message_slot': [[0]*l for _ in range(d)]
    }
    return nested_dict(2), sample

def generate_graphs(n, D):
    """
    Generate all undirected adjacency matrices with self-loops (A[i,i]=1)
    and maximum degree (row-sum) <= D.
    Returns a list of numpy integer arrays shape (n,n).
    Warning: exponential in n.
    """
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_graphs = []
    # iterate over all choices of edges in the upper triangle (excluding diagonal)
    for bits in itertools.product([0, 1], repeat=len(edges)):
        A = np.eye(n, dtype=int)  # set self-loops
        for bit, (i, j) in zip(bits, edges):
            if bit:
                A[i, j] = 1
                A[j, i] = 1
        # degree check (sum of row includes self-loop)
        degs = A.sum(axis=1)
        if int(degs.max()) <= D+1:
            all_graphs.append(A)
    return all_graphs

def two_hop_coloring(A, D):
    """
    True 2-hop coloring: colors nodes so that any two nodes at distance <= 2
    have different colors. Colors are integers in 1..(D^2+1).
    We compute the square graph G^2 and run a greedy coloring.
    """
    n = A.shape[0]
    # build graph G; include edges where A[u,v]==1 and u!=v
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 1:
                G.add_edge(u, v)
    # power-2 graph (edges between nodes with distance <= 2, excluding self loops)
    G2 = nx.power(G, 2)
    # ensure no self-loop edges in G2 (nx.power shouldn't add them, but be safe)
    if G2.has_edge(0, 0):  # quick check (only if present)
        for node in list(G2.nodes()):
            if G2.has_edge(node, node):
                G2.remove_edge(node, node)

    # greedy coloring (returns colors in 0..k-1)
    color_map = nx.coloring.greedy_color(G2, strategy='largest_first')
    # map to 1..(D^2+1)
    colors = [color_map[i] + 1 for i in range(n)]
    max_color_allowed = D**2 + 1
    used_max = max(colors)
    if used_max > max_color_allowed:
        raise RuntimeError(
            f"Greedy 2-hop coloring used {used_max} colors, exceeds allowed {max_color_allowed}."
        )
    return colors  # list of length n, integers in 1..D^2+1

def flooding_final_messages(A, source, message):
    """
    Flooding implemented as BFS from source: every reachable node receives the full message.
    A is adjacency matrix with self-loops. We ignore sending to self (not necessary).
    Returns list 'final_messages' of length n where each entry is either the l-bit list
    (if the node received it) or None.
    """
    n = A.shape[0]
    visited = [False]*n
    q = [source]
    visited[source] = True
    final = [[0]*len(message) for _ in range(n)]
    final[source] = list(message)
    while q:
        u = q.pop(0)
        # neighbors v where A[u,v]==1 and v != u
        for v in range(n):
            if v == u:
                continue
            if A[u, v] == 1 and not visited[v]:
                visited[v] = True
                final[v] = list(message)
                q.append(v)
    return final

def create_node_features(n, s, l, d, message, local_ids):
    X_test_list = []
    for node in range(n):
        _, sample = get_sample(l, d)
        # if node is source, set sample['message'] to the binary message
        if node == s:
            sample['message'] = message.copy()
        # set node's my_slot at index local_id-1 to all ones
        jid = local_ids[node] - 1  # zero-based
        sample['my_slot'][jid] = [1]*l
        X_test_list.append(sample)
    return X_test_list

def graph_diameter(adj_matrix):
    """
    Calculate diameter using NetworkX library.
    
    Args:
        adj_matrix: 2D numpy array or list of lists representing adjacency matrix
    
    Returns:
        Diameter of the graph, or float('inf') if graph is disconnected
    """
    G = nx.from_numpy_array(np.array(adj_matrix))
    
    # Check if graph is connected
    if not nx.is_connected(G):
        return float('inf')
    
    return nx.diameter(G)

def enumerate_all_cases(n, D, l, return_results=True):
    """
    Enumerate all graphs G with n nodes and max degree <= D (self-loops included),
    all binary messages of length l, and all source nodes.
    For each case returns (A, X_test_list, final_messages).
    If return_results is False, only returns the total number of cases.
    """
    graphs = generate_graphs(n, D)
    num_graphs = len(graphs)
    num_messages = 2**l
    total_cases = num_graphs * num_messages * n

    if not return_results:
        return total_cases

    # iterate and produce detailed results (warning: can be huge!)
    results = []  # list of tuples (A, X_test_list, final_messages)
    for A in graphs:
        # compute true 2-hop local ids
        local_ids = two_hop_coloring(A, D)  # values in 1..D^2+1
        d = D**2 + 1
        # iterate all binary messages
        for bits in itertools.product([0, 1], repeat=l):
            message = list(bits)
            # all possible sources
            for s in range(n):
                final_messages = flooding_final_messages(A, s, message)
                X_test_list = create_node_features(n, s, l, d, message, local_ids)
                # store (make a shallow copy of A to avoid accidental mutation)
                results.append((A.copy(), X_test_list, final_messages))
    return total_cases, results

# -------------------------
# Example usage (small n)
# -------------------------
if __name__ == "__main__":
    n = 6   # small for tractability
    D = 5
    l = 1

    print("Enumerating graphs (careful—exponential growth!).")
    total_cases, results = enumerate_all_cases(n, D, l, return_results=True)
    print(f"Total cases: {total_cases}")
    