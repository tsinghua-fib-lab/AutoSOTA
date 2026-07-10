import sys, time, pickle, numpy as np
sys.path.insert(0, ".")
from examples.safe_PCE import *

# Load pre-computed data
with open("examples/data/U_hatPi.pkl", "rb") as f:
    data = pickle.load(f)
U = data["U"]
hat_Pi_pre = data["hat_Pi"]

# Extract pi_s from the first entry (all entries share the same pi_s)
# But actually pi_s is not stored directly in hat_Pi entries...
# Looking at the test_stage, it receives pi_s and hat_Pi separately
# The hat_Pi entries contain (pi, u, v, u_s, v_s) where u_s, v_s are the values of pi_s
# on that training CMDP. pi_s itself needs to be computed.

# Compute pi_s the same way as training
env_max = make_env(0.5)
pi_s, _, _ = LP(env_max)
print(f"Safe policy computed, u_s={value_function_r_c(env_max, pi_s)}")

# Run test with pre-computed hat_Pi
noise = 0.3
print(f"\nRunning test with noise={noise}, K=40000 using pre-computed hat_Pi...")
t0 = time.time()
real_r, real_c = test_stage(K=40000, noise=noise, pi_s=pi_s, hat_Pi=list(hat_Pi_pre))
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Final reward regret at K=40000: {real_r[-1]:.2f}")
print(f"Final constraint value: {real_c[-1]:.4f}")
