import heapq
import itertools
from collections import Counter
from typing import List, Optional, Generator
import numpy as np

# ---------------------------
# (Prufer -> edges, tree adjacency generator)
# ---------------------------

def prufer_to_edges(prufer: List[int], n: int):
    """Convert a Prüfer sequence into edges of the labeled tree on nodes 0..n-1."""
    degree = [1] * n
    for v in prufer:
        degree[v] += 1

    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    edges = []
    for v in prufer:
        u = heapq.heappop(leaves)
        edges.append((u, v))
        degree[u] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, v)

    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    edges.append((u, v))
    return edges

def generate_tree_adjacency_matrices(n: int,
                                     D: int,
                                     include_self_loop: bool = True,
                                     as_generator: bool = False
                                     ) -> Optional[Generator[np.ndarray, None, None]] or List[np.ndarray]:
    """
    Generate adjacency matrices A (binary numpy arrays, shape (n,n)) of all labeled trees
    on n nodes whose maximum (ordinary) degree equals D. If include_self_loop is True,
    ones are placed on the diagonal.
    """
    from collections import Counter

    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        if D == 0:
            A = np.array([[1 if include_self_loop else 0]], dtype=int)
            return ( (a for a in [A]) if as_generator else [A] )
        else:
            return ( (x for x in []) if as_generator else [] )

    if not (0 <= D <= n-1):
        raise ValueError(f"D must be between 0 and n-1 (here n-1={n-1}).")

    if D == 0:
        return ( (x for x in []) if as_generator else [] )

    seq_length = n - 2

    def generator():
        rng = range(n)
        for prufer in itertools.product(rng, repeat=seq_length):
            cnt = Counter(prufer)
            degrees = [1 + cnt[i] for i in range(n)]
            if max(degrees) != D:
                continue
            edges = prufer_to_edges(list(prufer), n)
            A = np.zeros((n, n), dtype=int)
            for u, v in edges:
                A[u, v] = 1
                A[v, u] = 1
            if include_self_loop:
                np.fill_diagonal(A, 1)
            yield A

    if as_generator:
        return generator()
    else:
        return list(generator())

# ---------------------------
# Square coloring implementation
# ---------------------------

def build_square_neighbors_from_tree_adj(A: np.ndarray) -> List[set]:
    """
    Given adjacency matrix A of a tree (may include diagonal ones),
    return list S where S[u] is set of vertices at distance 1 or 2 from u
    (excluding u itself). This is the conflict neighborhood for square-coloring.
    """
    n = A.shape[0]
    # Build simple adjacency (exclude diagonal/self loops)
    G = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j]:
                G[i].add(j)

    S = [set() for _ in range(n)]
    for u in range(n):
        # distance-1
        for v in G[u]:
            S[u].add(v)
        # distance-2
        for v in list(G[u]):
            for w in G[v]:
                if w != u:
                    S[u].add(w)
    return S

def color_square_of_tree(A: np.ndarray, D: int, verbose: bool = False) -> List[int]:
    """
    Color the square of a tree given by adjacency matrix A (n x n),
    using at most D+1 colors. Returns a list `colors` of length n where
    colors[i] is an integer in 1..D+1.

    Raises ValueError if it cannot find a coloring with <= D+1 colors.
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=int)
    n = A.shape[0]
    if n == 0:
        return []

    max_colors = D + 1
    if max_colors < 1:
        raise ValueError("D must be >= 0.")

    # Build square-neighborhood sets (distance 1 or 2, excluding self)
    S = build_square_neighbors_from_tree_adj(A)

    # Quick sanity checks:
    # Ordinary degree (excluding self-loop) maximum must be <= D
    degrees = [len({v for v in range(n) if v!=i and A[i,v]}) for i in range(n)]
    if max(degrees) > D:
        raise ValueError(f"Input tree has ordinary max degree {max(degrees)} > D={D}.")

    # Heuristic initialization: greedy color using smallest available color
    colors = [0] * n

    # Order nodes by descending ordinary degree (break ties by index)
    order = sorted(range(n), key=lambda u: (-len(S[u]), u))
    for u in order:
        forbidden = {colors[v] for v in S[u] if colors[v] != 0}
        # pick smallest color in 1..max_colors not in forbidden
        for c in range(1, max_colors + 1):
            if c not in forbidden:
                colors[u] = c
                break
        # If greedy fails, leave 0; we'll solve with DSATUR/backtracking below
    if all(c != 0 for c in colors):
        used = len(set(colors))
        if verbose:
            print(f"Greedy succeeded with {used} colors (<= {max_colors})")
        if used <= max_colors:
            return colors

    # DSATUR + backtracking to guarantee a solution using <= max_colors
    # Precompute adjacency for degrees and for faster tie-breaking
    degree_sq = [len(S[u]) for u in range(n)]

    # Helper functions for DSATUR selection
    def select_uncolored_vertex(curr_colors):
        # returns index of uncolored vertex with highest saturation; tie-break by degree_sq
        best = None
        best_sat = -1
        best_deg = -1
        for u in range(n):
            if curr_colors[u] == 0:
                neigh_colors = {curr_colors[v] for v in S[u] if curr_colors[v] != 0}
                sat = len(neigh_colors)
                deg = degree_sq[u]
                if sat > best_sat or (sat == best_sat and deg > best_deg) or (sat == best_sat and deg == best_deg and (best is None or u < best)):
                    best = u
                    best_sat = sat
                    best_deg = deg
        return best

    # fill initial colors from greedy where available (helps pruning)
    initial_colors = colors[:]  # may contain zeros

    # backtracking recursion
    assigned_count = sum(1 for c in initial_colors if c != 0)
    curr_colors = initial_colors[:]

    # Order of colors to try: prefer smaller color numbers
    color_range = list(range(1, max_colors + 1))

    # To speed, maintain for every v the set of used neighbor colors (updated incrementally)
    neighbor_color_sets = [set() for _ in range(n)]
    for u in range(n):
        for v in S[u]:
            if curr_colors[v] != 0:
                neighbor_color_sets[u].add(curr_colors[v])

    # Pre-calc adjacency lists for efficient iteration
    Sq = S

    # limit recursion depth: n
    import sys
    sys.setrecursionlimit(max(10000, 2 * n + 100))

    def backtrack(assigned):
        if assigned == n:
            return True
        v = select_uncolored_vertex(curr_colors)
        if v is None:
            return True
        forbidden = neighbor_color_sets[v]
        # Try colors, prefer smallest first
        for c in color_range:
            if c not in forbidden:
                # assign c
                curr_colors[v] = c
                # update neighbor_color_sets
                changed = []
                for w in Sq[v]:
                    if c not in neighbor_color_sets[w]:
                        neighbor_color_sets[w].add(c)
                        changed.append(w)
                # recursion
                if backtrack(assigned + 1):
                    return True
                # undo
                curr_colors[v] = 0
                for w in changed:
                    neighbor_color_sets[w].remove(c)
        return False

    success = backtrack(assigned_count)
    if not success:
        raise ValueError(f"Could not color the square of the tree with <= {max_colors} colors.")
    used_colors = set(curr_colors)
    if 0 in used_colors:
        used_colors.remove(0)
    if len(used_colors) > max_colors:
        raise ValueError(f"Coloring used {len(used_colors)} colors which is > {max_colors}.")
    return curr_colors

# ---------------------------
# main: only calls the final function
# ---------------------------

if __name__ == "__main__":
    # Example: build one tree adjacency matrix (n=7, D=3) and color its square
    n = 4
    D = 3
    # Generate trees with ordinary max degree D (include self loops on diagonal)
    mats = generate_tree_adjacency_matrices(n, D, include_self_loop=True, as_generator=False)
    if len(mats) == 0:
        print(f"No labeled trees on n={n} with ordinary max degree D={D}.")
    else:
        max_colors=[]
        # take the first generated adjacency matrix
        for A in mats:
            print("ffff",A.type)
            colors = color_square_of_tree(A, D)
            max_colors.append(max(colors))
        print(len(max_colors))
        all_equal_to_D = len(set(max_colors)) == 1 and max_colors[0] == D+1
        print(all_equal_to_D)
