import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from templates import bellman_ford
import json
import random
import argparse
from collections import deque



class GraphInitializer:
    def __init__(self, adjacency_matrix, weight_matrix, color_list, D, l):
        self.A = adjacency_matrix
        self.W = weight_matrix
        self.colors = color_list
        self.n_nodes = len(adjacency_matrix)
        self.D = D  # Maximum degree
        self.l = l # Bits needed for node IDs
        
    def get_binary_value(self, value):
        """Convert node index (1-based) to binary representation with l bits in Little-Endian"""
        # Node indices are 1-based, so we use node_index directly (not node_index-1)
        binary_str = bin(value)[2:]  # Remove '0b' prefix, use 1-based indexing
        # Pad with zeros to make it l bits long and reverse for Little-Endian
        padded_binary = [int(bit) for bit in binary_str.zfill(self.l)]
        return list(reversed(padded_binary))  # Reverse for Little-Endian
    
    def binary_to_decimal_le(self, binary_list):
        """Convert Little-Endian binary list to decimal"""
        return sum(bit * (2 ** i) for i, bit in enumerate(binary_list))
    
    def initialize_node(self, node_index):
        """Initialize a single node with given index (1-based)"""
        # Create a sample node structure
        node = bellman_ford.get_sample(self.l,self.D)[1]

        node['num_nodes'] =  self.get_binary_value(self.n_nodes)
        node['n4round'] =  self.get_binary_value(self.n_nodes)
        node['round'] = [1] + [0] * (self.l - 1)
        node['backup_round'] = [1] + [0] * (self.l - 1)
        node['received_pointer'] = [1] + [0] * (self.l - 1)
        node['compare_round'] = [0] * (self.l - 1) + [1]

        node['v_inf_backup'] = [[1] * self.l for _ in range(self.D)]
        node['v_dist'] = [[1] * self.l for _ in range(self.D)]
        
        # Set node ID (Little-Endian)
        node['u_id'] = self.get_binary_value(node_index)
        
        # Set u_priority for source node (node 1)
        if node_index != 1:
            node['u_inf'] = [1] * (self.l)
            node['u_dist'] = [1] * (self.l)
        
        # Set determine_slot based on color (1-based indexing)
        color = self.colors[node_index - 1]  # Colors list is 0-indexed
        if 1 <= color <= len(node['determine_slot']):
            node['determine_slot'][color - 1] = 1
        
        # Find neighbors and populate v_id, v_white, v_gray
        neighbors = []
        for i in range(self.n_nodes):
            if self.A[node_index - 1, i] == 1 and i + 1 != node_index:  # Skip self (1-based comparison)
                neighbors.append(i + 1)  # Convert to 1-based index
        
        # Sort neighbors by their ID for consistent ordering
        neighbors.sort()
        
        # Populate v_id with neighbor IDs (Little-Endian)
        for i, neighbor_id in enumerate(neighbors):
            if i < self.D:  # Only fill up to D neighbors
                node['v_id'][i] = self.get_binary_value(neighbor_id)
                node['weight'][i] = self.get_binary_value(self.W[node_index - 1, neighbor_id - 1])
                
                # Set v_white and v_gray based on neighbor properties
                if neighbor_id == 1:  # Source node
                    node['v_inf_backup'][i] = [0] * (self.l)
                    node['v_dist'][i] = [0] * (self.l)
        
        return node
    
    def initialize_all_nodes(self):
        """Initialize all nodes in the graph"""
        nodes = []
        for i in range(1, self.n_nodes + 1):  # 1-based indexing
            nodes.append(self.initialize_node(i))
        return nodes


def two_hop_coloring(A, D):
    """
    Performs a two-hop graph coloring on a graph represented by an adjacency matrix.

    In a two-hop coloring, any two nodes that are at a distance of 1 or 2 from
    each other must have different colors. This function uses a greedy approach
    to assign the smallest possible color to each node.

    Args:
        A (np.ndarray): The adjacency matrix of the graph (n x n).
                        It's assumed that A[i, i] = 1 for all i (self-loops).
        D (int): The maximum degree of any node in the graph (excluding the self-loop).
                 The number of colors used will be at most D^2 + 1.

    Returns:
        list[int]: A list of length n, where the i-th element is the color
                   assigned to node i. Colors are integers starting from 1.
    """
    # Get the number of nodes from the shape of the adjacency matrix
    n = A.shape[0]

    # Initialize a list to store the color of each node. 0 means uncolored.
    colors = [0] * n

    # Pre-calculate the square of the adjacency matrix.
    # A_squared[i, j] > 0 indicates a path of length 2 between nodes i and j.
    A_squared = np.dot(A, A)

    # Iterate through each node to assign a color
    for i in range(n):
        # This set will store the colors of all nodes within a 2-hop distance
        forbidden_colors = set()

        # Identify all 1-hop and 2-hop neighbors and collect their colors
        for j in range(n):
            # A node is in the 2-hop neighborhood if it's a 1-hop neighbor
            # (A[i, j] == 1) or a 2-hop neighbor (A_squared[i, j] > 0).
            # We exclude the node itself (i != j).
            if i != j and (A[i, j] == 1 or A_squared[i, j] > 0):
                # If the neighbor has been colored, add its color to the forbidden set
                if colors[j] != 0:
                    forbidden_colors.add(colors[j])

        # Find the smallest integer color that is not in the forbidden set
        color = 1
        while color in forbidden_colors:
            color += 1

        # Assign the found color to the current node
        colors[i] = color

    return colors

def is_connected(adj_matrix, n):
    """
    Check if the graph represented by the adjacency matrix is connected.
    
    Args:
        adj_matrix: 2D list or numpy array representing the adjacency matrix
        n (int): Number of nodes in the graph
    
    Returns:
        bool: True if the graph is connected, False otherwise
    """
    if n == 0:
        return True
    
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    
    while queue:
        node = queue.popleft()
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return all(visited)

def generate_random_graph(n, D, max_weight=3):
    """
    Generates a single random, connected graph with n nodes, a maximum degree of D,
    and random edge weights.

    Args:
        n (int): The number of nodes in the graph. Must be > D.
        D (int): The maximum degree for any node (excluding self-loops).
        max_weight (int): The maximum weight for any edge. Weights are random integers in [0, max_weight].
                         Default is 1 (unweighted graph).

    Returns:
        tuple: A tuple containing:
            - adj_matrix (2D list): The adjacency matrix of the generated graph
            - weight_matrix (2D list): The weight matrix with random edge weights
    """
    if n <= D:
        raise ValueError("Number of nodes (n) must be greater than the max degree (D).")
    if max_weight < 0:
        raise ValueError("max_weight must be non-negative.")

    max_attempts = 100  # Safety break to prevent potential infinite loops.
    for _ in range(max_attempts):
        # 1. Initialize an n x n adjacency matrix with zeros.
        adj_matrix = np.zeros((n, n), dtype=int)
        weight_matrix = np.zeros((n, n), dtype=int)

        # 2. Add self-loops for all nodes by setting the diagonal to 1.
        np.fill_diagonal(adj_matrix, 1)
        # Self-loops typically have weight 0, but you can change this if needed
        np.fill_diagonal(weight_matrix, 0)

        # 3. Add more random edges, ensuring no node's degree exceeds D.
        degrees = np.sum(adj_matrix, axis=1) - 1  # Subtract 1 to exclude self-loops

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 0 and degrees[i] < D and degrees[j] < D:
                    if random.random() < 0.5:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
                        
                        # Assign random weight to the edge
                        weight = random.randint(1, max_weight)
                        weight_matrix[i, j] = weight
                        weight_matrix[j, i] = weight
                        
                        degrees[i] += 1
                        degrees[j] += 1
        
        # 4. Check for connectivity. If connected, the graph is valid.
        if is_connected(adj_matrix, n):
            return adj_matrix, weight_matrix

    # This is reached only if the loop finishes without finding a connected graph.
    raise RuntimeError(f"Failed to generate a connected graph for n={n}, D={D} after {max_attempts} attempts.")

# Alternative optimized version using Floyd-Warshall algorithm
def graph_diameter_floyd_warshall(adj_matrix):
    """
    Calculate the diameter using Floyd-Warshall algorithm.
    More efficient for dense graphs.
    """
    adj_matrix = adj_matrix.tolist()

    n = len(adj_matrix)
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Set initial distances
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif adj_matrix[i][j] == 1:
                dist[i][j] = 1
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Find maximum shortest path
    max_distance = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] != float('inf') and dist[i][j] > max_distance:
                max_distance = dist[i][j]
    
    return max_distance

def find_max_longest_path(adj_matrix, weight_matrix):
    """
    Finds the length of the longest simple path in a weighted, undirected graph.

    This function uses a backtracking approach to explore all possible simple
    paths starting from every node and keeps track of the maximum length found.

    Args:
        adj_matrix (np.ndarray): The binary adjacency matrix of the graph.
        weight_matrix (np.ndarray): The matrix of edge weights.

    Returns:
        int or float: The maximum weight of the longest simple path between any
                      two nodes in the graph.
    """
    num_nodes = len(adj_matrix)
    # Using a list for max_len to make it mutable inside the nested function
    max_len = [0]

    def dfs_backtrack(u, visited, current_length):
        """Recursively explores paths from node u."""
        # Mark the current node as visited for the current path
        visited[u] = True

        # Check all neighbors of the current node
        for v in range(num_nodes):
            # If there's an edge and the neighbor hasn't been visited in this path
            if adj_matrix[u][v] == 1 and not visited[v]:
                # Explore the path further
                new_length = current_length + weight_matrix[u][v]
                dfs_backtrack(v, visited, new_length)

        # After exploring all paths from u, update the max length if this path is longer
        # This captures the length of the path ending at node u
        if current_length > max_len[0]:
            max_len[0] = current_length

        # Backtrack: Unmark the node as visited to allow it in other paths
        visited[u] = False

    # Run the DFS starting from every node to find all possible paths
    for i in range(num_nodes):
        # Initialize a new visited array for each starting node
        visited = [False] * num_nodes
        dfs_backtrack(i, visited, 0)

    return int(max_len[0])

def generate_random_connected_graph(max_D, max_nodes, max_weight=3):
    """
    Generates a random connected graph with parameters chosen randomly within given bounds.
    
    Args:
        max_D (int): Maximum allowed degree (excluding self-loops)
        max_nodes (int): Maximum number of nodes
    
    Returns:
        tuple: (n, D, adjacency_matrix) where:
            - n: randomly generated number of nodes (2 to max_nodes)
            - D: randomly generated maximum degree (min_degree to max_D)
            - adjacency_matrix: 2D list representing the graph adjacency
    """
    # Generate random n between 2 and max_nodes
    n = random.randint(2, max_nodes)
    
    # Calculate minimum degree for a connected graph with n nodes
    # For a connected graph, the minimum degree is 1 (tree structure)
    min_degree = 1 if n==2 else 2
    
    # Ensure max_D is at least min_degree and at most n-1
    max_D = min(max_D, n - 1)
    max_D = max(max_D, min_degree)
    
    # Generate random D between min_degree and max_D
    D = random.randint(min_degree, max_D)
    
    # Generate the graph using the provided function
    adjacency_matrix, weight_matrix = generate_random_graph(n, D, max_weight)

    d_G = graph_diameter_floyd_warshall(adjacency_matrix)
    p_L = find_max_longest_path(adjacency_matrix, weight_matrix)
    
    return n, D, d_G, p_L, adjacency_matrix, weight_matrix


def binary_to_int(binary_list):
    """Convert little-endian binary list to integer"""
    return int(''.join(map(str, binary_list[::-1])), 2)

def standard_bellman_ford(adj_matrix, weight_matrix):
    """
    Run Bellman-Ford algorithm on a graph given its binary adjacency matrix and weight matrix.
    
    Args:
        adj_matrix: Binary adjacency matrix (numpy array)
        weight_matrix: Weight matrix (numpy array)
    
    Returns:
        list: Distance from source (node 0) to each node
    """
    n = len(adj_matrix)  # Number of nodes
    source = 0  # Source is always the first node
    
    # Initialize distances: 0 for source, infinity for others
    dist = [float('inf')] * n
    dist[source] = 0
    
    # Relax edges n-1 times
    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                # Check if edge exists and we can relax it
                if adj_matrix[u][v] == 1 and dist[u] != float('inf') and dist[u] + weight_matrix[u][v] < dist[v]:
                    dist[v] = dist[u] + weight_matrix[u][v]
    
    # Check for negative weight cycles
    for u in range(n):
        for v in range(n):
            if adj_matrix[u][v] == 1 and dist[u] != float('inf') and dist[u] + weight_matrix[u][v] < dist[v]:
                print("Warning: Graph contains negative weight cycle")
                return dist
    
    return dist

def verify_bellman_ford_equivalence(adj_matrix, weight_matrix, X_test_list, source=1):
    """
    Verify that the binary logic Bellman-Ford produces the same distances as standard Bellman-Ford
    """
    # Run standard Bellman-Ford
    standard_distances = standard_bellman_ford(adj_matrix, weight_matrix)
        
    # Extract distances from X_test_list (u_dist attribute)
    test_distances = [binary_to_int(node['u_dist']) for node in X_test_list]
    
    print("Standard Bellman-Ford distances:", standard_distances)
    print("X_test_list u_dist values:", test_distances)
    
    # Check if all distances match
    standard_vs_test = all(standard_distances[i] == test_distances[i] for i in range(len(standard_distances)))
    
    return standard_vs_test


if __name__ == "__main__":

    n, D, d_G, p_L, A, W = generate_random_connected_graph(max_D=3, max_nodes=7)
    colors = two_hop_coloring(A, D)
    l = max(p_L.bit_length(), n.bit_length())+1

    # Initialize the graph
    initializer = GraphInitializer(A, W, colors, D, l)
    X_test_list = initializer.initialize_all_nodes()

    # Print results with Little-Endian interpretation
    print("Graph Initialization Results (Little-Endian Convention)")
    print("=" * 60)
    print(f"Adjacancy matrix: {initializer.A}")
    print(f"Weight matrix: {initializer.W}")
    print(f"Number of nodes: {initializer.n_nodes}")
    print(f"Maximum degree (D): {initializer.D}")
    print(f"Bits needed for node IDs (l): {initializer.l}")
    print(f"Colors: {colors}")
    print("=" * 60)

    for i, node in enumerate(X_test_list, 1):
        node_id_decimal = initializer.binary_to_decimal_le(node['u_id'])
        print(f"Node {i}:")
        print(f"  u_id: {node['u_id']} -> {node_id_decimal}")
        print(f"  u_dist: {node.get('u_dist', 'N/A')}")
        print(f"  u_inf: {node['u_inf']}")
        print(f"  v_inf_backup: {node['v_inf_backup']}")
        print(f"  v_dist: {node['v_dist']}")
        print(f"  weight: {node['weight']}")
        
        # Convert neighbor IDs from Little-Endian to decimal for readability
        neighbor_ids = []
        for v_id in node['v_id']:
            if any(v_id):  # If not all zeros
                neighbor_decimal = initializer.binary_to_decimal_le(v_id)
                neighbor_ids.append(neighbor_decimal)
            else:
                neighbor_ids.append(0)
        
        print(f"  v_id: {node['v_id']} -> Neighbors: {neighbor_ids}")
        
        # Find the color position
        color_pos = 0
        for j, slot in enumerate(node['determine_slot']):
            if slot == 1:
                color_pos = j + 1
                break
        
        print(f"  determine_slot: {node['determine_slot']} (color position: {color_pos})")
        
        print("-" * 50)

    # Verify node IDs are correct
    print("\nNode ID Verification:")
    for i, node in enumerate(X_test_list, 1):
        node_id_decimal = initializer.binary_to_decimal_le(node['u_id'])
        print(f"Node {i}: binary {node['u_id']} -> decimal {node_id_decimal} {'✓' if node_id_decimal == i else '✗'}")