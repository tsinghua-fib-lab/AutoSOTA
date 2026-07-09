import networkx as nx
import random

#Returns the value of the cut (S, V | S)
def cut_value(G, S):
    cut_value = 0
    for u, v, data in G.edges(data=True):
        if (u in S and v not in S) or (v in S and u not in S):
            cut_value += data['capacity']
    return cut_value

#Returns the achieved ratio and a cut that achieves it
def cut_set(G, L, eps = 1e-4):
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
    tau_1 = min_capacity/n
    tau_2 = W
    
    G_directed = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        G_directed.add_edge(u, v, capacity = data['capacity'])
        G_directed.add_edge(v, u, capacity = data['capacity'])
    
    s = 's' 
    t = 't'
    G_directed.add_node(s)
    G_directed.add_node(t)
    for v in L:
        G_directed.add_edge(v, t, capacity = 2*W + 1)
        
    for v in G.nodes():
        G_directed.add_edge(s, v, capacity = tau_1)
    
    while tau_2 - tau_1 > eps:
        tau_mid = (tau_2 - tau_1)/2  + tau_1
        for v in G.nodes():
            G_directed[s][v]['capacity'] = tau_mid
        cut_val, _ = nx.maximum_flow(G_directed, s, t)
        if cut_val >= n*tau_mid - 1e-6:
            tau_1 = tau_mid
        else:
            tau_2 = tau_mid
            
    for v in G.nodes():
        G_directed[s][v]['capacity'] = tau_2

    _, partition = nx.minimum_cut(G_directed, s, t)

    return tau_1, partition[0].intersection(set(G.nodes()))

def find_balanced_vertex(tree):
    """
    Find the vertex whose removal minimizes the size of the largest connected component.

    Parameters:
        tree (nx.Graph): A NetworkX undirected tree.

    Returns:
        balanced_vertex (int): The vertex that minimizes the largest connected component size.
        min_largest_component (int): The size of the largest connected component after removal.
    """
    # Ensure the input is a tree
    if not nx.is_tree(tree):
        raise ValueError("Input graph must be a tree.")
    
    n = len(tree)
    subtree_sizes = {}  # To store sizes of subtrees
    largest_components = {}  # To store largest component size for each node
    for node in tree.nodes():
        largest_components[node] = 0

    # Perform a DFS to compute subtree sizes
    def dfs(node, parent):
        subtree_size = 1  # Start with the current node
        for neighbor in tree.neighbors(node):
            if neighbor != parent:
                size = dfs(neighbor, node)
                subtree_size += size
                largest_components[node] = max(largest_components[node], size)
        subtree_sizes[node] = subtree_size
        return subtree_size

    dfs(list(tree.nodes)[0], -1)

    # Calculate the largest connected component size for each node
    for node in tree.nodes():
        largest_from_parent = n - subtree_sizes[node]  # Component formed by removing this subtree
        largest_components[node] = max(largest_components[node], largest_from_parent)

    # Find the vertex with the minimum largest component size
    balanced_vertex = min(set(tree.nodes()), key=lambda x: largest_components[x])
    min_largest_component = largest_components[balanced_vertex]

    return balanced_vertex, min_largest_component

def random_spanning_tree(graph):
    """
    Generate a random spanning tree of an undirected graph using Wilson's Algorithm.

    Parameters:
        graph (nx.Graph): An undirected NetworkX graph.

    Returns:
        tree (nx.Graph): A spanning tree represented as an undirected NetworkX graph.
    """
    if not isinstance(graph, nx.Graph):
        raise ValueError("Input must be an undirected graph.")
    
    if not nx.is_connected(graph):
        raise ValueError("Input graph must be connected to generate a spanning tree.")
    
    # Initialize the tree
    tree = nx.Graph()
    nodes = list(graph.nodes)
    visited = set()
    
    # Start with a random node
    start = random.choice(nodes)
    visited.add(start)
    tree.add_node(start)
    
    # Loop until all nodes are in the tree
    while len(visited) < len(nodes):
        # Start a random walk from a node not in the tree
        current = random.choice([node for node in nodes if node not in visited])
        walk = [current]
        
        while current not in visited:
            # Choose a random neighbor
            current = random.choice(list(graph.neighbors(current)))
            # Avoid cycles by erasing loops
            if current in walk:
                walk = walk[:walk.index(current) + 1]
            else:
                walk.append(current)
        
        # Add the path to the tree
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i + 1]
            tree.add_edge(u, v)
            visited.add(u)
            visited.add(v)

    return tree

def create_rooted_tree(tree, root):
    """
    Create a rooted tree from a given undirected tree and root vertex.

    Parameters:
        tree (nx.Graph): The input tree represented as an undirected graph.
        root (int): The vertex to be used as the root.

    Returns:
        rooted_tree (nx.DiGraph): A directed tree rooted at the specified vertex.
    """
    if not nx.is_tree(tree):
        raise ValueError("Input graph must be a tree.")

    if root not in tree:
        raise ValueError("The specified root must be a vertex in the tree.")

    # Create a directed graph to represent the rooted tree
    rooted_tree = nx.DiGraph()
    
    # Perform BFS or DFS to build the rooted tree
    def dfs(node, parent):
        for neighbor in tree.neighbors(node):
            if neighbor != parent:  # Avoid traversing back to the parent
                rooted_tree.add_edge(node, neighbor)
                dfs(neighbor, node)

    dfs(root, None)
    return rooted_tree


def read_graph(graph):
    # Remove self-loops, return largest component
    G = nx.Graph()
    G = nx.read_edgelist(open(graph, "r"), nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))

    size = 0
    largest_comp = set()
    for comp in nx.connected_components(G):
        if len(comp) > size:
            size = len(comp)
            largest_comp = comp
    G = G.subgraph(largest_comp)

    return G