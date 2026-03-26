"""Generate BKS solutions for CVRP using pyvrp"""
import numpy as np
import pyvrp
from pyvrp import Model
from pyvrp.stop import MaxRuntime
import time

# Load CVRP test data
data = np.load('/repo/data/cvrp/test/50.npz')
locs = data['locs']  # (1000, 51, 2) - first is depot
demands_raw = data['demand_linehaul']  # (1000, 50), fractional
vehicle_capacity = data['vehicle_capacity']  # (1000, 1)

n_instances = locs.shape[0]
print(f"Solving {n_instances} CVRP instances (N=50) with pyvrp...")

costs = []
SCALE = 10000  # Integer scaling for pyvrp

for i in range(n_instances):
    if i % 100 == 0:
        print(f"  Solved {i}/{n_instances}...")
    
    depot = locs[i, 0]  # (2,)
    customers = locs[i, 1:]  # (50, 2)
    cap = float(vehicle_capacity[i, 0])
    demands = demands_raw[i] * cap  # un-normalize to raw demand
    
    m = Model()
    
    # Add depot
    m.add_depot(x=int(depot[0] * SCALE), y=int(depot[1] * SCALE))
    
    # Add clients
    for j in range(len(customers)):
        demand_int = max(1, round(float(demands[j])))
        m.add_client(
            x=int(customers[j, 0] * SCALE),
            y=int(customers[j, 1] * SCALE),
            delivery=demand_int
        )
    
    # Add vehicle type
    cap_int = round(float(cap))
    m.add_vehicle_type(
        capacity=cap_int,
        num_available=n_instances  # unlimited vehicles
    )
    
    # Solve with short time limit
    result = m.solve(stop=MaxRuntime(0.5), seed=42)
    # Cost in original scale 
    cost = result.cost() / SCALE
    costs.append(cost)

costs = np.array(costs, dtype=np.float32)
print(f"\nDone! Mean cost: {costs.mean():.3f}")
np.savez('/repo/data/cvrp/test/50_sol_pyvrp.npz', costs=costs)
print("Saved BKS solutions to /repo/data/cvrp/test/50_sol_pyvrp.npz")
