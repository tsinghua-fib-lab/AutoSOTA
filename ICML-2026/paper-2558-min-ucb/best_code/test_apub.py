import numpy as np
import pickle
import gurobipy as gp
from apub import APUB
import time
import sys
sys.path.insert(0, '/repo')

# Load pre-generated data
with open("/repo/120.pkl", "rb") as f:
    data = pickle.load(f)

train_samples_list = data["train_samples"]
train_sample = train_samples_list[0]

# Config parameters
n_items = 20
n_machines = 8
c = [-14, -9, -20, -15, -4, -40, -18, -11, -13, -16, -17, -8, -9, -24, -10, -7, -12, -3, -4, -5]
A = np.zeros((n_machines, n_items))
b = np.zeros(n_machines)

# Run APUB with M=100 (small test)
print("Testing APUB with M=100, alpha=0.1...")
start = time.perf_counter()
apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
x_opt, eta, obj_val, num_cuts = apub.solve_two_stage_apub(train_sample, alpha=0.1, M_bootstrap=100)
elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.2f}s, Iterations: {num_cuts}, Objective: {obj_val:.2f}")
print(f"x_opt[:5]: {x_opt[:5]}")
print("SUCCESS!")
