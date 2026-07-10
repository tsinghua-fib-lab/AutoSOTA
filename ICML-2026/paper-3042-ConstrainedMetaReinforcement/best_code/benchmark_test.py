import sys, time, pickle, numpy as np
sys.path.insert(0, ".")
from examples.safe_PCE import *

# Load pre-computed U 
with open("examples/data/U_hatPi.pkl", "rb") as f:
    data = pickle.load(f)
U = data["U"]
print(f"U size: {len(U)}")

# Build safe policy
env_max_noise = make_env(0.5)
pi_s, _, _ = LP(env_max_noise)
print(f"Safe policy computed, shape: {pi_s.shape}")

# Build new hat_Pi using the U set
t0 = time.time()
hat_Pi_new = policy_set(U, pi_s, show_progress=True)
print(f"Policy set built in {time.time()-t0:.1f}s")

# Run a single test
print("Running test_stage with K=5000 to benchmark...")
t0 = time.time()
real_r, real_c = test_stage(K=5000, noise=0.3, pi_s=pi_s, hat_Pi=list(hat_Pi_new))
elapsed = time.time() - t0
print(f"K=5000 done in {elapsed:.1f}s")
print(f"Estimated K=40000 time: {elapsed * 8:.1f}s = {elapsed * 8 / 60:.1f} min")
print(f"Final regret at K=5000: {real_r[-1]:.2f}")
print(f"Final constraint at K=5000: {real_c[-1]:.4f}")
