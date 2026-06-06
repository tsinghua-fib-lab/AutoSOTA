"""Generate BKS solutions for CVRP using pyvrp - fast parallel version"""
import numpy as np
import pyvrp
from pyvrp import Model
from pyvrp.stop import MaxRuntime
import multiprocessing as mp
import sys
import os

# Disable warning spam
import warnings
warnings.filterwarnings('ignore')

def solve_instance(args):
    i, locs_i, demands_i, cap = args
    try:
        SCALE = 10000
        depot = locs_i[0]
        customers = locs_i[1:]
        
        m = Model()
        m.add_depot(x=int(depot[0] * SCALE), y=int(depot[1] * SCALE))
        
        for j in range(len(customers)):
            demand_int = max(1, round(float(demands_i[j])))
            m.add_client(
                x=int(customers[j, 0] * SCALE),
                y=int(customers[j, 1] * SCALE),
                delivery=demand_int
            )
        
        cap_int = round(float(cap))
        m.add_vehicle_type(capacity=cap_int, num_available=100)
        
        result = m.solve(stop=MaxRuntime(0.3), seed=42, display=False)
        return result.cost() / SCALE
    except Exception as e:
        return -1.0

if __name__ == '__main__':
    data = np.load('/repo/data/cvrp/test/50.npz')
    locs = data['locs']
    demands_raw = data['demand_linehaul']
    vehicle_capacity = data['vehicle_capacity']
    
    n = locs.shape[0]
    print(f"Solving {n} CVRP instances in parallel...", flush=True)
    
    # Scale demands back to integers
    caps = vehicle_capacity[:, 0].tolist()
    demands_int = []
    for i in range(n):
        cap = caps[i]
        d = demands_raw[i] * cap
        demands_int.append(d)
    
    args = [(i, locs[i], demands_int[i], caps[i]) for i in range(n)]
    
    n_workers = min(48, n)
    with mp.Pool(n_workers) as pool:
        results = pool.map(solve_instance, args)
    
    costs = np.array(results, dtype=np.float32)
    print(f"Done! Mean cost: {costs.mean():.3f}, min: {costs.min():.3f}, max: {costs.max():.3f}")
    
    np.savez('/repo/data/cvrp/test/50_sol_pyvrp.npz', costs=costs)
    print("Saved to /repo/data/cvrp/test/50_sol_pyvrp.npz")
