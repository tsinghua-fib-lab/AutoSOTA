import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from templates import bfs
import json
import random
import argparse
from collections import deque



class GraphInitializer:
    def __init__(self, adjacency_matrix, color_list, D, l):
        self.A = adjacency_matrix
        self.colors = color_list
        self.n_nodes = len(adjacency_matrix)
        self.D = D  # Maximum degree
        self.l = l # Bits needed for node IDs
        
    def get_node_id_binary(self, node_index):
        """Convert node index (1-based) to binary representation with l bits in Little-Endian"""
        # Node indices are 1-based, so we use node_index directly (not node_index-1)
        binary_str = bin(node_index)[2:]  # Remove '0b' prefix, use 1-based indexing
        # Pad with zeros to make it l bits long and reverse for Little-Endian
        padded_binary = [int(bit) for bit in binary_str.zfill(self.l)]
        return list(reversed(padded_binary))  # Reverse for Little-Endian
    
    def binary_to_decimal_le(self, binary_list):
        """Convert Little-Endian binary list to decimal"""
        return sum(bit * (2 ** i) for i, bit in enumerate(binary_list))
    
    def initialize_node(self, node_index):
        """Initialize a single node with given index (1-based)"""
        # Create a sample node structure
        node = bfs.get_sample(self.l,self.D)[1]

        node['q_pointer']= [1] + [0] * (self.l - 1)
        node['priority_comparison_counter']= [1] + [0] * self.l
        node['compare_u_priority']= [1] * self.l
        node['q_pointer']= [1] + [0] * (self.l - 1)
        
        # Set node ID (Little-Endian)
        node['u_id'] = self.get_node_id_binary(node_index)
        
        # Set u_priority for source node (node 1)
        if node_index == 1:
            node['u_priority'] = [1] + [0] * (self.l - 1)  # Little-Endian: LSB first
            node['u_gray'] = 1
            
            # Set v_last_in_q_augend and v_last_in_q for source node
            node['v_last_in_q_augend'] = [[0] * self.l for _ in range(self.D)]
            node['v_last_in_q'] = [[0] * self.l for _ in range(self.D)]
            
            # Set first array to 2 (binary: [0, 1, 0, 0] in Little-Endian for l=4)
            # In Little-Endian, 2 = [0, 1, 0, 0] (binary 0100 = 2 in decimal)
            if self.l >= 2:
                node['v_last_in_q_augend'][0][1] = 1  # Little-Endian: second position = 2^1
                node['v_last_in_q'][0][1] = 1
        else:
            node['u_white'] = 1
        
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
                node['v_id'][i] = self.get_node_id_binary(neighbor_id)
                
                # Set v_white and v_gray based on neighbor properties
                if neighbor_id == 1:  # Source node
                    node['v_gray'][i] = 1
                else:
                    node['v_white'][i] = 1
        
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

def generate_random_graph(n, D):
    """
    Generates a single random, connected graph with n nodes and a maximum degree of D.

    Args:
        n (int): The number of nodes in the graph. Must be > D.
        D (int): The maximum degree for any node (excluding self-loops).

    Returns:
        list: A 2D list representing the adjacency matrix of the generated graph.
    """
    if n <= D:
        raise ValueError("Number of nodes (n) must be greater than the max degree (D).")

    max_attempts = 100  # Safety break to prevent potential infinite loops.
    for _ in range(max_attempts):
        # 1. Initialize an n x n adjacency matrix with zeros.
        adj_matrix = np.zeros((n, n), dtype=int)

        # 2. Add self-loops for all nodes by setting the diagonal to 1.
        np.fill_diagonal(adj_matrix, 1)

        # 3. Add more random edges, ensuring no node's degree exceeds D.
        degrees = np.sum(adj_matrix, axis=1) - 1  # Subtract 1 to exclude self-loops

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 0 and degrees[i] < D and degrees[j] < D:
                    if random.random() < 0.5:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
                        degrees[i] += 1
                        degrees[j] += 1
        
        # 4. Check for connectivity. If connected, the graph is valid.
        if is_connected(adj_matrix, n):
            return adj_matrix

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

def generate_random_connected_graph(max_D, max_nodes):
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
    adjacency_matrix = generate_random_graph(n, D)

    d_G = graph_diameter_floyd_warshall(adjacency_matrix)
    
    return n, D, d_G, adjacency_matrix


def binary_to_int(binary_list):
    """Convert little-endian binary list to integer"""
    return int(''.join(map(str, binary_list[::-1])), 2)

def standard_bfs(adj_matrix, source=1):
    """Standard BFS implementation with tie-breaking by smaller node ID"""
    n = len(adj_matrix)
    dist = [-1] * n
    visited = [False] * n
    
    # Convert to 0-indexed
    source_idx = source - 1
    dist[source_idx] = 0
    visited[source_idx] = True
    
    queue = deque([source_idx])
    
    while queue:
        current = queue.popleft()
        
        # Get neighbors and sort by ID for tie-breaking
        neighbors = []
        for i in range(n):
            if adj_matrix[current][i] == 1 and not visited[i]:
                neighbors.append(i)
        
        # Sort neighbors by their ID (smaller first)
        neighbors.sort()
        
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    
    return dist

def verify_bfs_equivalence(adj_matrix, X_test_list, source=1):
    """
    Verify that the binary logic BFS produces the same distances as standard BFS
    """
    # Run standard BFS
    standard_distances = standard_bfs(adj_matrix, source)
        
    # Extract distances from X_test_list (u_dist attribute)
    test_distances = [binary_to_int(node['u_dist']) for node in X_test_list]
    
    print("Standard BFS distances:", standard_distances)
    print("X_test_list u_dist values:", test_distances)
    
    # Check if all distances match
    standard_vs_test = all(standard_distances[i] == test_distances[i] for i in range(len(standard_distances)))
    
    return standard_vs_test


if __name__ == "__main__":

    n, D, d_G, A = generate_random_connected_graph(max_D=3, max_nodes=7)
    colors = two_hop_coloring(A, D)
    l=n.bit_length()+1

    # Initialize the graph
    initializer = GraphInitializer(A, colors, D, l)
    X_test_list = initializer.initialize_all_nodes()

    # Print results with Little-Endian interpretation
    print("Graph Initialization Results (Little-Endian Convention)")
    print("=" * 60)
    print(f"Adjacancy matrix: {initializer.A}")
    print(f"Number of nodes: {initializer.n_nodes}")
    print(f"Maximum degree (D): {initializer.D}")
    print(f"Bits needed for node IDs (l): {initializer.l}")
    print(f"Colors: {colors}")
    print("=" * 60)

    for i, node in enumerate(X_test_list, 1):
        node_id_decimal = initializer.binary_to_decimal_le(node['u_id'])
        print(f"Node {i}:")
        print(f"  u_id: {node['u_id']} -> {node_id_decimal}")
        print(f"  u_priority: {node.get('u_priority', 'N/A')}")
        print(f"  v_white: {node['v_white']}")
        print(f"  v_gray: {node['v_gray']}")
        
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
        print(f"  u_gray: {node.get('u_gray', 0)}")
        print(f"  u_white: {node.get('u_white', 0)}")
        
        if 'v_last_in_q_augend' in node:
            augend_values = []
            for augend in node['v_last_in_q_augend']:
                if any(augend):
                    augend_decimal = initializer.binary_to_decimal_le(augend)
                    augend_values.append(augend_decimal)
                else:
                    augend_values.append(0)
            print(f"  v_last_in_q_augend: {node['v_last_in_q_augend']} -> {augend_values}")
        
        if 'v_last_in_q' in node:
            last_in_q_values = []
            for last_in_q in node['v_last_in_q']:
                if any(last_in_q):
                    last_decimal = initializer.binary_to_decimal_le(last_in_q)
                    last_in_q_values.append(last_decimal)
                else:
                    last_in_q_values.append(0)
            print(f"  v_last_in_q: {node['v_last_in_q']} -> {last_in_q_values}")
        
        print("-" * 50)

    # Verify node IDs are correct
    print("\nNode ID Verification:")
    for i, node in enumerate(X_test_list, 1):
        node_id_decimal = initializer.binary_to_decimal_le(node['u_id'])
        print(f"Node {i}: binary {node['u_id']} -> decimal {node_id_decimal} {'✓' if node_id_decimal == i else '✗'}")