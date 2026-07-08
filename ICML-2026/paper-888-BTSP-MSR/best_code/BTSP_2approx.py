import numpy as np
import networkx as nx
from find_hamiltonian_cycle import find_hamiltonian_cycle as find_hamiltonian_cycle_import

def square_graph(G: nx.Graph) -> nx.Graph:
    """
    Given a graph G, return its square graph H, 
    where two vertices are adjacent in H if they are at distance at most 2 in G.
        - G: input graph
    Returns:
        - H: the square graph of G
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u in G.nodes():
        for v in G.neighbors(u):
            H.add_edge(u, v)
        for w in G.neighbors(u):
            for v in G.neighbors(w):
                if u != v:
                    H.add_edge(u, v)
    return H

def _rot_left(lst, k):
    k %= len(lst)
    return lst[k:] + lst[:k]

def build_threshold_graph(D, theta):
    """
    Build the threshold graph G(theta) from the distance matrix D and threshold theta.
        - D: n x n distance matrix
        - theta: distance threshold
    Returns:
        - G: the threshold graph G(theta)
    """
    n = len(D)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= theta + 1e-12:
                G.add_edge(i, j)
    return G

def open_ears_via_chain(G):
    """
    Decompose the graph G into open ears using chain decomposition.
        - G: input graph
    Returns:
        - ears: a list of open ears, where each ear is a list of vertices in order
    """
    ears = []
    for chain in nx.chain_decomposition(G):
        order = [chain[0][0]]
        cur = order[0]
        for u, v in chain:
            if u == cur:
                nxt = v
            elif v == cur:
                nxt = u
            else:
                order = [order[0]] + order[:0:-1]
                cur = order[-1]
                nxt = v if u == cur else u
            order.append(nxt)
            cur = nxt
        if order[0] == order[-1]:
            order.pop()
        ears.append(order)
    return ears

def _splice_spider(cycle, ear):
    u, *mid, v = ear
    if u not in cycle or v not in cycle:
        raise RuntimeError("ear endpoints missing in cycle")

    cycle = _rot_left(cycle, cycle.index(u))

    while cycle[1] != v:
        cycle = _rot_left(cycle, 1)

    return [cycle[0]] + mid + cycle[1:]


def hamilton_cycle_from_ears(ears):
    """
    Given a list of open ears, construct a Hamiltonian cycle by splicing the ears together.
        - ears: a list of open ears, where each ear is a list of vertices in order
    Returns:
        - cycle: a list of vertices in order representing the Hamiltonian cycle
    """
    cycle = ears[0][:]
    for ear in ears[1:]:
        cycle = _splice_spider(cycle, ear)
    return cycle

def find_hamiltonian_cycle_cp(H: nx.Graph):
    """
    Find a Hamiltonian cycle in the graph H using a constraint programming approach.
        - H: input graph (should be the square graph of a 2-connected graph)
    Returns:
        - cycle: a list of vertices in order representing the Hamiltonian cycle, or None if no cycle exists
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise ImportError(
            "OR-Tools is required for the CP fallback in BTSP 2-approximation. "
            "Install it with `pip install ortools`."
        ) from e
    n = H.number_of_nodes()
    model = cp_model.CpModel()

    x = {}
    for i in H.nodes():
        for j in H.neighbors(i):
            if i != j:
                x[i, j] = model.NewBoolVar(f"x[{i},{j}]")

    for i in H.nodes():
        model.Add(sum(x[i, j] for j in H.neighbors(i) if i != j) == 1)
        model.Add(sum(x[j, i] for j in H.neighbors(i) if i != j) == 1)

    u = {}
    for i in range(1, n):
        u[i] = model.NewIntVar(1, n - 1, f"u[{i}]")

    for i in range(1, n):
        for j in range(1, n):
            if i != j and (i, j) in x:
                model.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        path = []
        visited = set()
        current = 0
        path.append(current)
        while len(path) < n:
            for j in H.neighbors(current):
                if solver.BooleanValue(x[current, j]):
                    current = j
                    if current in visited:
                        return None
                    path.append(current)
                    visited.add(current)
                    break
        if (path[0], path[-1]) not in x or not solver.BooleanValue(x[path[-1], path[0]]):
            return None
        path.append(path[0])
        return path
    else:
        return None

def bottleneck_tsp_2approx(D):
    """
    Given a distance matrix D, find a Hamiltonian cycle (tour) that approximates the optimal bottleneck TSP solution within a factor of 2.
        - D: n x n distance matrix
    Returns:
        - tour: a list of vertices in order representing the Hamiltonian cycle
        - bottleneck: the maximum edge weight in the tour
        - theta_star: the threshold distance used to construct the tour
    """
    n = len(D)
    dists = np.unique(D[np.triu_indices(n, 1)])

    lo, hi = 0, len(dists) - 1
    theta_star = dists[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        G = build_threshold_graph(D, dists[mid])
        if nx.is_connected(G) and nx.node_connectivity(G) >= 2:
            theta_star = dists[mid]
            hi = mid - 1
        else:
            lo = mid + 1

    G_star = build_threshold_graph(D, theta_star)
    for i,j in G_star.edges():
        G_star[i][j]["length"] = max(D[i, j], D[j, i])
    try:
        tour = find_hamiltonian_cycle_import(G_star)
    except:
        H = square_graph(G_star)
        tour = find_hamiltonian_cycle_cp(H)
    bottleneck = max(D[tour[i], tour[i + 1]] for i in range(n))
    assert bottleneck <= 2 * theta_star + 1e-9, f"bottleneck exceeds 2*theta_star; bottleneck = {bottleneck}, theta_star = {theta_star}"
    return tour, bottleneck, theta_star