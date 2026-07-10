import sys, numpy as np
sys.path.insert(0, ".")
from examples.safe_PCE import *

# Test if LP gives deterministic results
np.random.seed(42)
env1 = make_env(0.3)
p1, u1, v1 = LP(env1)
print(f"Run 1: u={u1}, v={v1}")
print(f"  Policy[0,:] = {p1[0,:]}")
print(f"  Policy[45,:] = {p1[45,:]}")

np.random.seed(42)
env2 = make_env(0.3)
p2, u2, v2 = LP(env2)
print(f"Run 2: u={u2}, v={v2}")
print(f"  Policy[0,:] = {p2[0,:]}")
print(f"  Policy[45,:] = {p2[45,:]}")

print(f"Policies equal: {np.allclose(p1, p2)}")

# Test with different seed
np.random.seed(123)
env3 = make_env(0.3)
p3, u3, v3 = LP(env3)
print(f"Run 3 (seed 123): u={u3}, v={v3}")
print(f"  Policy[0,:] = {p3[0,:]}")
print(f"Policies equal (1 vs 3): {np.allclose(p1, p3)}")
