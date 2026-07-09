import ast
import argparse
from collections import deque

# ==========================================
#      USER CONFIGURATION SECTION
# ==========================================

def get_tree_and_vars():
    """
    1. Define the Tree structure (Adjacency List).
    2. Define the mapping from Variable Name to Tree Edge.
    """
    
    # ---------------------------------------------------------
    # PART A: Define the Tree T
    # Format: Adjacency list dictionary. 
    # Since edges are undirected, ensure A->B implies B->A.
    # ---------------------------------------------------------
    T_adj = {
        'A': ['s'],
        'B': ['s'],
        's': ['A', 'B', 'r'],
        'r': ['s', 'D', 'P'],
        'P': ['r', 'Q', 'R'],
        'D': ['r'],
        'Q': ['P'],
        'R': ['P'],
    }

    # ---------------------------------------------------------
    # PART B: Variable to Edge Mapping
    # Format: {'var_name': ('node1', 'node2')}
    # Note: Order of nodes in tuple doesn't matter, code normalizes it.
    # ---------------------------------------------------------
    var_map = {
        'b': ('B', 's'),
        's': ('s', 'r'),
        'c': ('r', 'P'),
        'd': ('r', 'D'),
        'e': ('P', 'R'),
        'f': ('P', 'Q'),
    }

    # ---------------------------------------------------------
    # PART C: List of all valid variables to check against
    # ---------------------------------------------------------
    all_vars = ['b', 'c', 'd', 's', 'e', 'f'] 

    return T_adj, var_map, all_vars

# ==========================================
#          CORE LOGIC
# ==========================================

def normalize_edge(u, v):
    """Returns a sorted tuple to represent an undirected edge."""
    return tuple(sorted((u, v)))

def bfs_path_edges(T_adj, start, end):
    """
    Returns a set of normalized edges lying on the path between start and end.
    Returns None if no path exists.
    """
    if start == end:
        return set()
    
    queue = deque([(start, [])]) # (current_node, path_of_edges)
    visited = {start}
    
    while queue:
        curr, path = queue.popleft()
        if curr == end:
            return set(path)
        
        if curr in T_adj:
            for neighbor in T_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_edge = normalize_edge(curr, neighbor)
                    queue.append((neighbor, path + [new_edge]))
    return set()

def parse_line(line):
    """Parses the input string format."""
    data = {}
    parts = line.strip().split(';')
    for part in parts:
        if ':' not in part: continue
        key, val_str = part.split(':', 1)
        key = key.strip()
        val = ast.literal_eval(val_str.strip())
        data[key] = val
    return data

def check_mono(split_data, var, T_adj, var_map):
    """
    Implements the Monotonicity Check Procedure.
    Returns True if monotonicity is satisfied, False otherwise.
    """
    
    T_star = split_data['T_star']
    S_minus = [normalize_edge(u, v) for u, v in split_data['S_minus']]
    S_plus_seq = split_data['S_plus'] # Assuming tuple/list sequence

    # --- 0. Corner Case: |S+| > 3 ---
    # When |S+| > 3, all variables except 'f' are NOT mono (return False).
    # 'f' proceeds to standard logic.
    if len(S_plus_seq) > 3:
        if var != 'f':
            return False

    # --- 1. Handle Corner Cases (X and Y nodes in T*) ---
    # These edges are not in T, so we check explicit dependencies.
    
    for u, v in T_star:
        edge_set = {u, v}
        
        # Case 1: ('A', 'X') depends on {c, s}
        if edge_set == {'A', 'X'}:
            if var in ['c', 's']:
                return False
        
        # Case 2: ('D', 'Y') depends on {d, s, f}
        elif edge_set == {'D', 'Y'}:
            if var in ['d', 'c', 'f']:
                return False
        
        # Case 3: ('D', 'X') depends on {c, d, s, e}
        elif edge_set == {'D', 'X'}:
            if var in ['c', 'd', 's', 'e']:
                return False
        
        # Other cases: Error
        elif 'X' in edge_set or 'Y' in edge_set:
            raise RuntimeError(f"unknown edge {edge_set}.")
    
    # If variable is not mapped to a physical edge in T, and wasn't caught above,
    # it is not involved in the standard topology logic.
    if var not in var_map:
        # If a variable is not in the tree map, and passed the above checks,
        # it contributes 0 to all terms. 0+0+0=0 => True.
        return True

    target_edge = normalize_edge(*var_map[var])

    # --- 2. Calculate Terms ---
    
    # Delta S- (Contribution from removed structure)
    # Check if target_edge is in S_minus
    delta_s_minus = 1 if target_edge in S_minus else 0

    # N t* (Contribution from connector)
    n_t_star = 0
    for u, v in T_star:
        # Skip special edges X/Y for path counting
        if 'X' in (u, v) or 'Y' in (u, v):
            continue
        
        path_edges = bfs_path_edges(T_adj, u, v)
        if target_edge in path_edges:
            n_t_star += 1

    # Delta S+ (Contribution from component sequence)
    delta_s_plus = 0
    # Iterate adjacent pairs in the S+ sequence
    if len(S_plus_seq) > 1:
        for i in range(len(S_plus_seq) - 1):
            u, v = S_plus_seq[i], S_plus_seq[i+1]
            path_edges = bfs_path_edges(T_adj, u, v)
            if target_edge in path_edges:
                delta_s_plus = 1
                break # It's an indicator function (0 or 1 for the whole set)

    # --- 3. Verification Logic ---
    total = delta_s_minus + n_t_star + delta_s_plus
    
    if total == 0:
        return True
    
    if (n_t_star + delta_s_plus <= 1) and (delta_s_minus == 1):
        return True
    
    return False

# ==========================================
#              MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='splits.txt', help="The file containing splits.")
    args = parser.parse_args()

    # 1. Load Configuration
    T_adj, var_map, all_vars = get_tree_and_vars()
    
    print("Configuration Loaded.")
    print(f"Variables to check: {all_vars}")
    print("-" * 50)

    # 2. Read File
    filename = args.file
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        exit(0)

    # 3. Process each split
    all_splits_correct = True
    
    for i, line in enumerate(lines):
        if not line.strip(): continue
        
        try:
            data = parse_line(line)
            
            # Get the claimed valid variables from the file
            claimed_mono_vars = set(data.get('mono_vars', []))
            
            # Calculate the actual valid variables using our function
            calculated_mono_vars = set()
            for var in all_vars:
                if check_mono(data, var, T_adj, var_map):
                    calculated_mono_vars.add(var)
            
            # Compare
            if claimed_mono_vars == calculated_mono_vars:
                print(f"Split #{i+1}: [PASS] mono_vars match.")
            else:
                print(f"Split #{i+1}: [FAIL] Mismatch detected.")
                print(f"  Claimed in file: {sorted(list(claimed_mono_vars))}")
                print(f"  Calculated     : {sorted(list(calculated_mono_vars))}")
                
                missing = calculated_mono_vars - claimed_mono_vars
                extra = claimed_mono_vars - calculated_mono_vars
                if missing: print(f"  Missing (should be True): {missing}")
                if extra:   print(f"  Extra (should be False) : {extra}")
                
                all_splits_correct = False
        
        except Exception as e:
            print(f"Split #{i+1}: [ERROR] {e}")
            all_splits_correct = False

    if all_splits_correct:
        print("\nAll mono_vars attributes are certified correct.")
    else:
        print("\nSome splits have incorrect mono_vars attributes.")

if __name__ == "__main__":
    main()
