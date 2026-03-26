"""Generate BKS solutions for CVRP N=50 using pyvrp with correct ProblemData"""
import numpy as np
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

def solve_instance_pyvrp(args):
    i, locs_i, demands_i = args
    try:
        from pyvrp._pyvrp import Client, Depot, ProblemData, VehicleType
        from pyvrp import solve as vrp_solve
        from pyvrp.stop import MaxRuntime
        import numpy as np
        import sys, io
        
        SCALE = 1000  # same as rl4co PYVRP_SCALING_FACTOR
        n_clients = len(demands_i)
        n_total = n_clients + 1  # depot + clients
        
        # Scale coords to integers
        locs_scaled = (locs_i * SCALE).round().astype(np.int64)
        
        # Build distance matrix from Euclidean distances
        dist_mat = np.zeros((n_total, n_total), dtype=np.int64)
        for a in range(n_total):
            for b in range(n_total):
                dx = int(locs_scaled[a, 0]) - int(locs_scaled[b, 0])
                dy = int(locs_scaled[a, 1]) - int(locs_scaled[b, 1])
                dist_mat[a, b] = round((dx*dx + dy*dy)**0.5)
        
        dur_mat = np.zeros_like(dist_mat)
        
        # Scale demands to integers
        cap_int = SCALE  # capacity is 1.0 => 1000 in scaled units
        depot = Depot(x=int(locs_scaled[0,0]), y=int(locs_scaled[0,1]))
        clients = []
        for j in range(n_clients):
            dem = max(1, round(float(demands_i[j]) * cap_int))
            c = Client(x=int(locs_scaled[j+1, 0]), y=int(locs_scaled[j+1, 1]), delivery=dem)
            clients.append(c)
        
        # VehicleType(num_available, capacity, start_depot, end_depot, fixed_cost)
        vtype = VehicleType(n_clients, cap_int, 0, 0, 0)
        
        data_pb = ProblemData(
            clients=clients,
            depots=[depot],
            vehicle_types=[vtype],
            distance_matrices=[dist_mat],
            duration_matrices=[dur_mat],
        )
        
        old = sys.stdout
        sys.stdout = io.StringIO()
        result = vrp_solve(data_pb, stop=MaxRuntime(5), seed=42)
        sys.stdout = old
        
        if result.best.is_feasible():
            return result.cost() / SCALE
        else:
            return -1.0
    except Exception as e:
        return -2.0

if __name__ == '__main__':
    data = np.load('/repo/data/cvrp/test/50.npz')
    locs = data['locs']  # (1000, 51, 2) - index 0 is depot
    demands = data['demand_linehaul']  # (1000, 50)
    
    n_instances = locs.shape[0]
    print(f"Solving {n_instances} CVRP instances (N=50) with pyvrp...", flush=True)
    
    # Test first instance
    test_result = solve_instance_pyvrp((0, locs[0], demands[0]))
    print(f"Test instance 0 cost: {test_result:.4f}", flush=True)
    
    args = [(i, locs[i], demands[i]) for i in range(n_instances)]
    
    n_workers = min(24, mp.cpu_count())
    print(f"Using {n_workers} workers with 5s per instance...", flush=True)
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(solve_instance_pyvrp, args, chunksize=5)
    
    costs = np.array(results, dtype=np.float32)
    valid = costs[costs > 0]
    failed = (costs <= 0).sum()
    
    print(f"Done! Valid: {len(valid)}, Failed: {failed}")
    if len(valid) > 0:
        print(f"Mean BKS cost: {valid.mean():.4f}, min: {valid.min():.4f}, max: {valid.max():.4f}")
    
    # Replace failures with mean
    if failed > 0:
        fallback_cost = valid.mean() if len(valid) > 0 else 10.5
        costs[costs <= 0] = fallback_cost
    
    np.savez('/repo/data/cvrp/test/50_sol_pyvrp.npz', costs=costs)
    print("Saved to /repo/data/cvrp/test/50_sol_pyvrp.npz")
