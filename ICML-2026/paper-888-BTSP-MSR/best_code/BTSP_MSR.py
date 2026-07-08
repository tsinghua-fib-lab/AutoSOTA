import numpy as np
import math
import argparse
from BTSP_2approx import bottleneck_tsp_2approx
import networkx as nx
from time import time
from datetime import datetime

def compute_D(filename, compute_shortest_path=True):
    """
    For .gml files, compute D based on the "distance" attribute 
    and then apply shortest path algorithm to get the final D.
        - filename: path to the .gml file
        - compute_shortest_path: if True, compute shortest path distances 
                                 and take the minimum with direct distances
    Returns:
        - D: n x n distance matrix
    """
    G = nx.read_gml(filename)
    nodes = sorted(G.nodes())
    n = len(nodes)

    node_index = {node: i for i, node in enumerate(nodes)}

    d_matrix = np.full((n, n), np.inf)

    for i in range(n):
        d_matrix[i, i] = 0.0
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        l_ij = G.adj[u][v]["distance"]
        d_value = (l_ij * 0.0085) + 4
        G.adj[u][v]["delay"] = d_value
        d_matrix[i, j] = d_value
        if d_matrix[j, i] == np.inf:
            d_matrix[j, i] = d_value
        else:
            if d_matrix[j, i] != d_value:
                raise ValueError(f"Edge ({u}, {v}) has inconsistent distances: {d_matrix[j, i]} vs {d_value}")

    if not compute_shortest_path:
        return d_matrix

    d_matrix_sp = np.zeros_like(d_matrix)
    for i,j in np.ndindex(n, n):
        d_matrix_sp[i, j] = nx.shortest_path_length(G, source=nodes[i], target=nodes[j], weight='delay', method='dijkstra')
    d_matrix = np.minimum(d_matrix, d_matrix_sp)
    return d_matrix



def solve_BCD_with_scip(D: np.ndarray, S_list: list):
    """
    Solve the BCD problem using a MIP formulation with PySCIPOpt.
        - D: n x n distance matrix
        - S_list: list of skip sets for each round
    Returns:
        - pi_opt: optimal assignment of virtual nodes to physical nodes
        - objval: optimal BCD cost
    """
    try:
        from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
    except ImportError as e:
        raise ImportError(
            "PySCIPOpt is required only for --milp_opt. "
            "Install it with SCIP support, or run without --milp_opt."
        ) from e
    n = D.shape[0]
    tau = len(S_list)

    model = Model("BCD_MIP")

    model.setPresolve(SCIP_PARAMSETTING.OFF)
    # model.setHeuristics(SCIP_PARAMSETTING.OFF)

    physical_nodes = list(range(n))
    virtual_nodes = list(range(n))

    pi_initial = np.random.permutation(n).tolist()

    # x[i,u] = 1 if virtual node i is assigned to physical node u
    x = {(i, u): model.addVar(vtype="B", name=f"x_{i}_{u}")
         for i in virtual_nodes for u in physical_nodes}

    # M_l = maximum distance for round l
    M = {}
    for l in range(tau):
        M[l] = model.addVar(vtype="C", lb=0.0, name=f"M_{l}")

    # y[l,i,s,u,v] = x[i,u] * x[i+s,v]
    y = {
        (l, i, s % n): {
            (u, v): model.addVar(vtype="B", name=f"y_{l}_{i}_{s % n}_{u}_{v}")
            for u in physical_nodes for v in physical_nodes if u != v
        }
        for l, S_l in enumerate(S_list)
        for s in S_l
        for i in virtual_nodes
    }

    for i in range(n):
        model.addCons(quicksum(x[i, u] for u in physical_nodes) == 1, f"assign_{i}")

    for u in range(n):
        model.addCons(quicksum(x[i, u] for i in virtual_nodes) == 1, f"position_{u}")

    for l, S_l in enumerate(S_list):
        for s in S_l:
            s_mod = s % n
            if s_mod == 0:
                raise ValueError(f"Invalid skip width s={s}: s % n == 0.")
            for i in virtual_nodes:
                for u in physical_nodes:
                    for v in physical_nodes:
                        if u == v:
                            continue
                        j = (i + s_mod) % n
                        model.addCons(
                            y[l, i, s_mod][u, v] <= x[i, u],
                            f"y_leq_x1_{l}_{i}_{s_mod}_{u}_{v}"
                        )
                        model.addCons(
                            y[l, i, s_mod][u, v] <= x[j, v],
                            f"y_leq_x2_{l}_{i}_{s_mod}_{u}_{v}"
                        )
                        model.addCons(
                            y[l, i, s_mod][u, v] >= x[i, u] + x[j, v] - 1,
                            f"y_geq_x_{l}_{i}_{s_mod}_{u}_{v}"
                        )
                        model.addCons(
                            M[l] >= D[u, v] * y[l, i, s_mod][u, v],
                            f"M_leq_D_{l}_{i}_{s_mod}_{u}_{v}"
                        )

    model.setObjective(quicksum(M[l] for l in range(tau)), "minimize")

    model.setParam("limits/time", 1800)
    model.optimize()

    pi_opt = [-1] * n
    try:
        for i in virtual_nodes:
            for u in physical_nodes:
                if model.getVal(x[i, u]) > 0.5:
                    pi_opt[i] = u
        objval = model.getObjVal()
    except:
        return None, None

    return pi_opt, objval

def compute_BCD_cost(pi: list, D: np.ndarray, S_list: list) -> float:
    """
    Compute the BCD cost for a given assignment pi and distance matrix D.
        - pi: list of physical node indices assigned to virtual nodes in order
        - D: n x n distance matrix
        - S_list: list of skip sets for each round
    Returns:
        - total_cost: the computed BCD cost
    """
    n = len(pi)
    total_cost = 0.0
    for S in S_list:
        max_dist_l = 0.0
        S_p = [min(d, n - d) for d in S]
        for i in range(n):
            u = pi[i]
            for d_p in S_p:
                j = (i + d_p) % n
                v = pi[j]
                dist_uv = D[u, v]
                if dist_uv > max_dist_l:
                    max_dist_l = dist_uv
        total_cost += max_dist_l
    return total_cost

def solve_BCD_with_greedy(D: np.ndarray, S_list:list) -> list:
    """
    Solve the BCD problem using a greedy heuristic.
        - D: n x n distance matrix
        - S_list: list of skip sets for each round
    Returns:
        - pi: assignment of virtual nodes to physical nodes
    """
    n = D.shape[0]
    remaining = set(range(n))
    
    pi = [np.random.choice(list(remaining))]
    remaining.remove(pi[0])

    while len(pi) < n:
        best_candidate = None
        best_cost = float('inf')

        for v in remaining:
            pi_candidate = pi + [v]
            cost = compute_partial_BCD_cost_mod(pi_candidate, D, S_list)
            if cost < best_cost:
                best_cost = cost
                best_candidate = v

        pi.append(best_candidate)
        remaining.remove(best_candidate)

    return pi

def compute_partial_BCD_cost_mod(pi_partial: list, D: np.ndarray, S_list: list) -> float:
    """
    Compute the partial BCD cost for a given partial assignment pi_partial and distance matrix D.
        - pi_partial: list of physical node indices assigned to virtual nodes in order (partial)
        - D: n x n distance matrix
        - S_list: list of skip sets for each round
    Returns:
        - total_partial_cost: the computed partial BCD cost
    """
    k = len(pi_partial)
    n = D.shape[0]
    total_partial_cost = 0.0

    for S in S_list:
        max_dist_l = 0.0
        S_p = [min(d, n - d) for d in S]
        for i in range(k):
            u = pi_partial[i]
            for d_p in S_p:
                j = (i + d_p) % n
                if j < k:
                    v = pi_partial[j]
                    dist_uv = D[u, v]
                    if dist_uv > max_dist_l:
                        max_dist_l = dist_uv
        total_partial_cost += max_dist_l

    return total_partial_cost

def compute_bottleneck_value(pi: list, D: np.ndarray) -> float:
    """
    Compute the bottleneck value (maximum distance) 
    for a given assignment pi and distance matrix D.
        - pi: list of physical node indices assigned to virtual nodes in order
        - D: n x n distance matrix
    Returns:
        - bottleneck_value: the maximum distance between consecutive nodes in pi
    """
    n = len(pi)
    max_len = 0.0
    for idx in range(n):
        u = pi[idx]
        v = pi[(idx + 1) % n]
        if D[u, v] > max_len:
            max_len = D[u, v]
    return max_len

def is_coprime(a: int, b: int) -> bool:
    """
    Check if a and b are coprime (i.e., gcd(a, b) == 1).
        - a: first integer
        - b: second integer
    Returns:
        - True if a and b are coprime, False otherwise
    """
    return math.gcd(a, b) == 1

def MSR(n: int, S_list: list, pi: list, D: np.ndarray) -> tuple:
    """
    Perform the MSR optimization to find the best p and corresponding pi_star.
        - n: number of nodes
        - S_list: list of skip sets for each round
        - pi: initial assignment of virtual nodes to physical nodes
        - D: n x n distance matrix
    Returns:
        - best_p: the best value of p
        - pi_star: the corresponding optimal assignment
    """
    best_p = None
    best_cost = float('inf')

    for p in range(1, n):
        if not is_coprime(p, n):
            continue

        S_list_p = [[(d * p) % n for d in S] for S in S_list]
        cost_p = compute_BCD_cost(pi, D, S_list_p)

        if cost_p < best_cost:
            best_cost = cost_p
            best_p = p

    pi_star = [pi[(best_p * i) % n] for i in range(n)]
    return best_p, pi_star

def generate_S_list(n: int, nw_topology: str) -> list:
    """
    Generate the list of skip sets S_list based on the network topology.
        - n: number of nodes
        - nw_topology: type of network topology ("ring", "exponential", "one_peer_exp", "sparse_exp")
    Returns:
        - S_list: list of skip sets for each round
    """
    S_list = []
    if nw_topology == "ring":
        S_list = [[1]]
    elif nw_topology == "exponential":
        k_max = int(math.floor(math.log(n - 1, 2)))
        static_S = [2**k for k in range(k_max + 1) if 2**k < n]
        S_list = [static_S]
    elif nw_topology == "one_peer_exp":
        k_max = int(math.floor(math.log(n - 1, 2)))
        static_elems = [2**k for k in range(k_max + 1) if 2**k < n]
        S_list = [[d] for d in static_elems]
    elif nw_topology == "sparse_exp":
        k_max = int(math.floor(math.log(n - 1, 2)))
        static_S = [2**k for k in range(k_max + 1) if 2**k < n]
        S_list = [[1, static_S[-2]]]
    else:
        raise ValueError(f"Unknown nw_topology: {nw_topology}")
    return S_list

def compute_delays_on_ring(pi, D):
    """
    Compute the average delay and maximum delay for a given assignment pi on a ring topology.
        - pi: list of physical node indices assigned to virtual nodes in order
        - D: n x n distance matrix
    Returns:
        - average_delay: the average delay between consecutive nodes in pi
        - max_delay: the maximum delay between consecutive nodes in pi
    """
    average_delay = 0
    max_delay = 0
    n = len(pi)

    for i in range(n):
        u = pi[i]
        v = pi[(i+1)%n]
        delay = D[u,v]
        average_delay += delay / n
        max_delay = max(delay, max_delay)
    return average_delay, max_delay

def solve_BCD_with_SA(D: np.ndarray, S_list: list, max_iter: int = 100, seed: int = 0) -> list:
    """
    Solve the BCD problem using a simulated annealing heuristic.
        - D: n x n distance matrix
        - S_list: list of skip sets for each round
        - max_iter: maximum number of iterations for the simulated annealing
        - seed: random seed for reproducibility
    Returns:
        - pi: assignment of virtual nodes to physical nodes
    """
    n = D.shape[0]
    rng = np.random.default_rng(seed)

    # initialize with a random assignment
    pi = list(range(n))
    rng.shuffle(pi)

    best_pi = pi.copy()
    best_cost = compute_partial_BCD_cost_mod(pi, D, S_list)

    T = 1.0
    alpha = 0.9

    for _ in range(max_iter):
        # swap two random positions
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue

        pi_new = pi.copy()
        pi_new[i], pi_new[j] = pi_new[j], pi_new[i]

        cost = compute_partial_BCD_cost_mod(pi, D, S_list)
        cost_new = compute_partial_BCD_cost_mod(pi_new, D, S_list)
        delta = cost_new - cost

        # decrease temperature and decide whether to accept the new solution
        if delta < 0 or rng.random() < np.exp(-delta / T):
            pi = pi_new
            cost = cost_new

        # update best solution
        if cost < best_cost:
            best_cost = cost
            best_pi = pi.copy()

        # update temperature
        T *= alpha

    return best_pi

def main(nw_topology="ring", filename=None, verbose=True, seed=42):
    """
    Main function to run the BCD optimization experiment.
    """

    parser = argparse.ArgumentParser(description="minimizing BCD")
    parser.add_argument("--nw_topology", choices=["ring", "exponential", "one_peer_exp", "sparse_exp"],
                        default=nw_topology, help="NW topology")
    parser.add_argument("--seed", type=int, default=seed,
                        help="random seed")
    parser.add_argument("--greedy_opt", action="store_true",
                        help="set True to use greedy optimization")
    parser.add_argument("--no_opt", action="store_true",
                        help="set True to use random assignment")
    parser.add_argument("--physical_nw_file", type=str, default=filename,
                        help="distance matrix file path")
    parser.add_argument("--milp_opt", action="store_true",
                        help="set True to use SCIP")
    parser.add_argument("--sa_opt", action="store_true",
                        help="set True to use simulated annealing")
    parser.add_argument("--verbose", action="store_true",
                        help="set True to verbose log")
    parser.add_argument("--output_file", type=str, default="output.json",
                        help="output JSON file path to save the result")

    args = parser.parse_args()
    if args.physical_nw_file is None:
        raise ValueError(
            "physical_nw_file must be specified. "
            "Use --physical_nw_file path/to/network.npy or path/to/network.gml."
        )
    result = run_one_experiment(args.nw_topology, args.physical_nw_file, args.seed, args.greedy_opt, args.no_opt, args.sa_opt, args.milp_opt, args.verbose, args.output_file)
    return result

def run_one_experiment(nw_topology, physical_nw_file, seed=0, greedy_opt=False, no_opt=False, sa_opt=False, milp_opt=False, verbose=False, output_file=None):
    """
    Run one experiment of BCD optimization with the specified parameters.
        - nw_topology: type of network topology ("ring", "exponential", "one_peer_exp", "sparse_exp")
        - physical_nw_file: path to the distance matrix file (either .npy or .gml)
        - seed: random seed for reproducibility
        - greedy_opt: if True, use greedy optimization
        - no_opt: if True, use random assignment without optimization
        - sa_opt: if True, use simulated annealing optimization
        - milp_opt: if True, use MILP optimization with SCIP
        - verbose: if True, print detailed logs
    Returns:
        - result: a dictionary containing the results of the experiment
    """
    best_p = 0
    np.random.seed(seed)
    t = time()
    if physical_nw_file.endswith('.gml'):
        D = compute_D(physical_nw_file)
    else:
        D = np.load(physical_nw_file)
    n = D.shape[0]

    S_list = generate_S_list(n, nw_topology)

    pi_initial = list(range(n))
    np.random.shuffle(pi_initial)

    if greedy_opt:
        pi = solve_BCD_with_greedy(D, S_list)
    elif no_opt:
        pi = list(range(n))
    elif milp_opt:
        pi, objval = solve_BCD_with_scip(D, S_list)
        if pi is None:
            pi = pi_initial
    elif sa_opt: 
        pi = solve_BCD_with_SA(D, S_list, seed=seed)
    else: # BTSP-MSR
        pi, _, _ = bottleneck_tsp_2approx(D)
        pi = pi[:-1]

    if not milp_opt and not greedy_opt and not no_opt and not sa_opt:
        best_p, pi_star = MSR(n, S_list, pi, D)
    else:
        best_p = 1
        pi_star = pi.copy()
    t = time() - t
    if verbose:
        print(f"n                       = {n}")
        print(f"physical_nw_file        = {physical_nw_file}")
        print(f"nw_topology             = {nw_topology}")
        print(f"S_list                  = {S_list}")
        print(f"optimal c        = {best_p}")
        print(f"pi_star         = {pi_star}")
    
    cost_pi = compute_BCD_cost(pi, D, S_list) / len(S_list)
    cost_final = compute_BCD_cost(pi_star, D, S_list) / len(S_list)
    cost_initial = []
    for _ in range(5): # compute average BCD cost with random permutations
        pi_random = pi_initial.copy()
        np.random.shuffle(pi_random)
        cost_random = compute_BCD_cost(pi_random, D, S_list) / len(S_list)
        cost_initial.append(cost_random)
    cost_initial = np.mean(cost_initial)
    method = "milp" if milp_opt else "greedy" if greedy_opt else "no_opt" if no_opt else "sa_opt" if sa_opt else "btsp_msr"
    if verbose:
        print(f"BCD cost (random average) = {cost_initial:.4f}")
        print(f"BCD cost ({method if method != 'btsp_msr' else 'BTSP'})         = {cost_pi:.4f}")
        if method == "btsp_msr":
            print(f"BCD cost (BTSP-MSR)   = {cost_final:.4f} (c = {best_p})")

    # average_delay, max_delay = compute_delays_on_ring(pi, D)

    result = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n": int(n),
            "seed": int(seed),
            "topology": nw_topology,
            "physical_nw_file": physical_nw_file,
            "method": (
                "milp" if milp_opt else
                "greedy" if greedy_opt else
                "no_opt" if no_opt else
                "sa_opt" if sa_opt else
                "btsp_msr"
            ),
            "S_list": S_list,
        },
        "metrics": {
            "time_sec": float(t),
            "BCD": float(cost_final),
            "BCD_pi": float(cost_pi),
            "BCD_initial_avg": float(cost_initial),
            "c": int(best_p),
        },
        "solution": {
            "pi": [int(i) for i in pi],
            "pi_star":[int(i) for i in pi_star]
        }
    }

    # To save the result to a JSON file, result should include the file path and timestamp in the meta information
    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)

    # return cost_initial, cost_pi, cost_final
    return result

if __name__ == "__main__":
    result = main()
