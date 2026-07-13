import json
import random
import numpy as np
import argparse
import os

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
    Checks if a graph is connected using Breadth-First Search (BFS).

    Args:
        adj_matrix (np.ndarray): The graph's adjacency matrix as a NumPy array.
        n (int): The number of nodes in the graph.

    Returns:
        bool: True if the graph is connected, False otherwise.
    """
    if n <= 1:
        return True
    
    visited = set()
    queue = [0]  # Start BFS from the first node
    visited.add(0)
    
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        
        # Find neighbors of the current node
        for neighbor in range(n):
            # If an edge exists and the neighbor hasn't been visited, add it to the queue.
            if adj_matrix[node, neighbor] == 1 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # If the number of visited nodes equals the total number of nodes, the graph is connected.
    return len(visited) == n

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

    max_attempts = 100 # Safety break to prevent potential infinite loops.
    for _ in range(max_attempts):
        # 1. Initialize an n x n adjacency matrix with zeros.
        adj_matrix = np.zeros((n, n), dtype=int)

        # 2. Add self-loops for all nodes by setting the diagonal to 1.
        np.fill_diagonal(adj_matrix, 1)

        # 3. Ensure at least one node has the maximum degree D.
        max_degree_node_idx = random.randint(0, n - 1)
        potential_neighbors = list(range(n))
        potential_neighbors.remove(max_degree_node_idx)
        neighbors_to_connect = random.sample(potential_neighbors, D)
        
        for neighbor_idx in neighbors_to_connect:
            adj_matrix[max_degree_node_idx, neighbor_idx] = 1
            adj_matrix[neighbor_idx, max_degree_node_idx] = 1

        # 4. Add more random edges, ensuring no node's degree exceeds D.
        degrees = np.sum(adj_matrix, axis=1) - 1

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == 0 and degrees[i] < D and degrees[j] < D:
                    if random.random() < 0.5:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
                        degrees[i] += 1
                        degrees[j] += 1
        
        # 5. Check for connectivity. If connected, the graph is valid.
        if is_connected(adj_matrix, n):
            return adj_matrix.tolist()

    # This is reached only if the loop finishes without finding a connected graph.
    raise RuntimeError(f"Failed to generate a connected graph for n={n}, D={D} after {max_attempts} attempts.")


def generate_and_save_graphs(num_graphs, D, filename="graphs.json"):
    """
    Generates a specified number of graphs and saves them to a JSON file.

    Args:
        num_graphs (int): The total number of graphs to generate.
        D (int): The maximum degree for any node in the graphs.
        filename (str): The name of the output JSON file.
    """
    all_graphs_data = []
    print(f"Generating {num_graphs} graphs with max degree D={D}...")

    for i in range(num_graphs):
        # The number of nodes 'n' must be larger than the degree 'D'
        # to be able to connect to D *other* nodes.
        min_n = max(5, D + 1)
        if min_n > 20:
             print(f"Warning: With D={D}, n must be at least {D+1}. The range [5-20] is not possible.")
             print(f"Adjusting n to be in range [{min_n}-30] for this D.")
             n = random.randint(min_n, 30)
        else:
            n = random.randint(min_n, 20)
        
        # Generate the adjacency matrix for one graph.
        graph_adj_matrix = generate_random_graph(n, D)
        
        all_graphs_data.append({
            "graph_id": i,
            "n": n,
            "D": D,
            "adjacency_matrix": graph_adj_matrix
        })

    # Save the complete list of graphs to a JSON file.
    try:
        with open(filename, 'w') as f:
            json.dump(all_graphs_data, f, indent=4)
        print(f"Successfully generated and saved {len(all_graphs_data)} graphs to '{filename}'")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # You can change these values before running the script.
    NUM_GRAPHS_TO_GENERATE = 1000
    
    # Set up an argument parser to get D from the command line.
    parser = argparse.ArgumentParser(
        description="Generate random graphs with a specified maximum degree."
    )
    parser.add_argument(
        '--degree', 
        type=int, 
        default=3, 
        help='Maximum degree (D) for the nodes in the graphs. Default is 3.'
    )
    args = parser.parse_args()

    MAX_DEGREE_D = args.degree
    
    # Create the directory if it doesn't exist
    output_dir = "saved_adjacencies"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the file path
    filepath = os.path.join(output_dir, f"graphs_D{MAX_DEGREE_D}.json")
    
    # --- Execution ---
    generate_and_save_graphs(
        num_graphs=NUM_GRAPHS_TO_GENERATE, 
        D=MAX_DEGREE_D,
        filename=filepath  # Pass the full file path
    )


