import ast
import argparse
from collections import defaultdict, deque

# ==========================================
#      USER CONFIGURATION SECTION
# ==========================================

def get_tree_configuration():
    """
    Define the Tree T, Set B (leaves), and Subset V here.
    """
    # 1. Define edges of the Ternary Tree T.
    #    Format: list of tuples/lists, e.g., [('A', 's'), ('s', 'B')]
    #    Since edges are undirected, order in pair doesn't matter.
    T_edges = [
        ('A', 's'),
        ('B', 's'),
        ('s', 'r'),
        ('r', 'D'),
        ('r', 'P'),
        ('P', 'Q'),
        ('P', 'R'),
    ]

    # 2. Define the set of Leaves B.
    #    Format: set of strings
    B = {'A', 'B', 'D', 'Q', 'R'}

    # 3. Define the subset V (must be subset of B).
    #    Format: set of strings
    V = {'A', 'B', 'D'}

    return T_edges, B, V

# ==========================================
#          VALIDATION LOGIC
# ==========================================

class Graph:
    def __init__(self, edges=None, nodes=None):
        self.adj = defaultdict(set)
        self.nodes = set()
        if nodes:
            self.nodes.update(nodes)
        if edges:
            for u, v in edges:
                self.add_edge(u, v)

    def add_edge(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.nodes.add(u)
        self.nodes.add(v)

    def remove_edges(self, edges_to_remove):
        for u, v in edges_to_remove:
            # Try removing u->v
            if v in self.adj[u]: self.adj[u].remove(v)
            # Try removing v->u
            if u in self.adj[v]: self.adj[v].remove(u)

    def get_components(self):
        """Returns a list of sets, where each set is a connected component."""
        visited = set()
        components = []
        for node in self.nodes:
            if node not in visited:
                component = set()
                queue = deque([node])
                visited.add(node)
                while queue:
                    curr = queue.popleft()
                    component.add(curr)
                    for neighbor in self.adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
        return components

    def is_tree(self):
        """Checks if the graph is a valid tree (Connected and Acyclic)."""
        if not self.nodes:
            return True # Empty graph is a forest, technically a tree if size 0 or 1? 
                        # Usually tree requires |V| >= 1. Assuming checks elsewhere handles empty.
        
        # 1. Check Cycle using DFS
        visited = set()
        parent = {}
        
        # We also need to check connectivity. 
        # If we start from one node and can't visit all, it's not a tree.
        start_node = next(iter(self.nodes))
        stack = [start_node]
        visited.add(start_node)
        parent[start_node] = None
        
        nodes_count = 0

        while stack:
            u = stack.pop()
            nodes_count += 1
            for v in self.adj[u]:
                if v == parent[u]:
                    continue
                if v in visited:
                    return False # Cycle detected
                visited.add(v)
                parent[v] = u
                stack.append(v)

        # 2. Check Connectivity
        if nodes_count != len(self.nodes):
            return False # Disconnected

        return True

    def is_forest(self):
        """Checks if graph is a forest (Acyclic). Connectivity not required."""
        visited = set()
        for node in self.nodes:
            if node not in visited:
                stack = [node]
                visited.add(node)
                parent = {node: None}
                while stack:
                    u = stack.pop()
                    for v in self.adj[u]:
                        if v == parent[u]:
                            continue
                        if v in visited:
                            return False # Cycle detected
                        visited.add(v)
                        parent[v] = u
                        stack.append(v)
        return True

def parse_line(line: str):
    """Parses the specific string format into a dictionary."""
    # Example: V_star:['A']; S_minus:[('B', 's')]; S_plus:('r', 'B'); T_star:[('A', 'B')]
    data = {}
    line = line.replace('X', 'R').replace('Y', 'Q')
    parts = line.strip().split(';')
    for part in parts:
        if ':' not in part: continue
        key, val_str = part.split(':', 1)
        key = key.strip()
        val = ast.literal_eval(val_str.strip())
        data[key] = val
    return data

def validate_split(split_idx, split_data, T_edges, B, V):
    errors = []

    # Extract split data
    try:
        V_star = set(split_data['V_star'])
        S_minus = split_data['S_minus'] # list of tuples
        S_plus = set(split_data['S_plus']) # input is tuple/list, convert to set
        T_star_edges = split_data['T_star']
    except KeyError as e:
        return False, [f"Missing key in input: {e}"]

    # --- Condition 1: V* should be a subset of V ---
    if not V_star.issubset(V):
        errors.append(f"V_star {V_star} is not a subset of V {V}")

    # --- Condition 2: S- should be a subset of E ---
    # Create set of normalized edges for T
    T_edges_set = set()
    for u, v in T_edges:
        T_edges_set.add(tuple(sorted((u, v))))

    # Check S-
    for edge in S_minus:
        norm_edge = tuple(sorted(edge))
        if norm_edge not in T_edges_set:
            errors.append(f"Edge {edge} in S_minus is not in Tree edges")

    # --- Condition 3: After deleting S-, all points in V* are isolated ---
    # Build Graph T
    all_nodes = set()
    for u, v in T_edges:
        all_nodes.add(u); all_nodes.add(v)

    G_cut = Graph(edges=T_edges, nodes=all_nodes)
    G_cut.remove_edges(S_minus)

    for node in V_star:
        if node not in G_cut.nodes:
            # Node might exist in V* but not in Tree? Should have been caught in setup,
            # but technically degree 0 if not in graph.
            continue
        degree = len(G_cut.adj[node])
        if degree > 0:
            errors.append(f"Node '{node}' in V_star is not isolated (degree {degree})")

    # --- Condition 4: S+ checks: Connectivity of B \ V* ---

    # [CONSTRAINT 1]: S+ should not contain nodes in V*
    intersection_v_star = S_plus.intersection(V_star)
    if intersection_v_star:
        errors.append(f"S_plus contains nodes from V_star: {intersection_v_star}")

    # [CONSTRAINT 2]: B \ V* must be connected via S+
    # We define a "Representative ID" for every node in B \ V*.
    # - If a node belongs to a component touching S+, its ID is -1 (The Super-Component).
    # - If a node belongs to a component NOT touching S+, its ID is that component's unique index.
    # Finally, the set of Representative IDs must have size <= 1.

    B_minus_V_star = B - V_star
    components = G_cut.get_components() # List of sets (components)

    # 1. Identify which components are part of the Super-Component (touch S+)
    #    and ensure S+ nodes are valid.
    s_plus_components_indices = set()
    s_plus_nodes_found = 0

    for idx, comp in enumerate(components):
        intersection = comp.intersection(S_plus)
        if intersection:
            s_plus_components_indices.add(idx)
            s_plus_nodes_found += len(intersection)

    # Sanity Check: Did we find all S+ nodes in the graph?
    if s_plus_nodes_found != len(S_plus):
        errors.append(f"S_plus contains {len(S_plus)} nodes, but graph components only contain {s_plus_nodes_found}. (Check for invalid/deleted nodes in S+)")

    # 2. Check the "location" of every node in B \ V*
    locations = set()

    for idx, comp in enumerate(components):
        # Find B-nodes in this component
        b_nodes_in_comp = comp.intersection(B_minus_V_star)

        if not b_nodes_in_comp:
            continue

        # Determine the "Location ID" for these B-nodes
        if idx in s_plus_components_indices:
            locations.add(-1) # Represents the Super-Component connected by S+
        else:
            locations.add(idx) # Represents an isolated component

    # 3. Validation: B \ V* must form a single connected entity
    if len(locations) > 1:
        errors.append(f"B \\ V* is disconnected. Parts found in isolated components {locations} (where -1 is the S+ merged set).")
    elif len(locations) == 0 and len(B_minus_V_star) > 0:
        # This theoretically shouldn't happen if B_minus_V_star is subset of graph nodes
        errors.append("Nodes in B \\ V* could not be located in any component.")

    # --- Condition 5: t* validation ---
    # t* is a forest on B
    # Check if t* uses only nodes in B
    t_star_nodes = set()
    for u, v in T_star_edges:
        t_star_nodes.add(u); t_star_nodes.add(v)

    if not t_star_nodes.issubset(B):
        errors.append(f"T_star uses nodes not in B: {t_star_nodes - B}")

    G_t_star = Graph(edges=T_star_edges, nodes=B)
    if not G_t_star.is_forest():
        errors.append("T_star is not a forest (contains cycles on B)")

    # Contract B \ V* into supernode
    # We build a new graph for the contraction
    # Nodes: V* elements (preserved) and "SUPERNODE" (representing B \ V*)
    contracted_nodes = set(V_star)
    supernode_label = "SUPERNODE"
    contracted_nodes.add(supernode_label)

    G_contracted = Graph(nodes=contracted_nodes)

    # Add edges based on T_star
    for u, v in T_star_edges:
        u_mapped = u if u in V_star else supernode_label
        v_mapped = v if v in V_star else supernode_label

        if u_mapped == v_mapped:
            # Self-loop created. Trees cannot have self-loops.
            # If t* has edges strictly within B \ V*, this forms a loop on Supernode.
            errors.append(f"Contraction creates self-loop from edge ({u}, {v}). Resulting graph is not a tree.")
        else:
            # Check if edge already exists (Multi-edge check)
            # A simple tree cannot have multiple edges between same pair.
            if v_mapped in G_contracted.adj[u_mapped]:
                errors.append(f"Contraction creates multi-edge between {u_mapped} and {v_mapped}. Resulting graph is not a simple tree.")
            G_contracted.add_edge(u_mapped, v_mapped)

    # Check if contracted graph is a tree
    if not G_contracted.is_tree():
        errors.append("Contracted T_star is not a valid tree (disconnected or contains cycles).")

    if not errors:
        return True, ["Valid"]
    else:
        return False, errors

# ==========================================
#              MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='splits.txt', help="The file containing splits.")
    args = parser.parse_args()

    # 1. Load Configuration
    T_edges, B, V = get_tree_configuration()

    print(f"Loaded Tree with {len(T_edges)} edges.")
    print(f"B (Leaves): {B}")
    print(f"V (Subset): {V}")
    print("-" * 40)

    # 2. Read file
    # Replace 'splits.txt' with your actual file path
    filename = args.file

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # Fallback for demonstration if file doesn't exist
        print(f"File '{filename}' not found.")
        exit(0)

    # 3. Validate
    all_passed = True
    for i, line in enumerate(lines):
        if not line.strip(): continue

        print(f"Checking Split #{i+1}...")
        try:
            split_data = parse_line(line)
            is_valid, messages = validate_split(i, split_data, T_edges, B, V)

            if is_valid:
                print("  [PASS] Valid Split.")
            else:
                print("  [FAIL] Invalid Split:")
                for msg in messages:
                    print(f"    - {msg}")
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] Could not parse or process line: {e}")
            all_passed = False
        print("-" * 20)

    if all_passed:
        print("\nAll splits are valid.")
    else:
        print("\nSome splits failed validation.")

if __name__ == "__main__":
    main()
