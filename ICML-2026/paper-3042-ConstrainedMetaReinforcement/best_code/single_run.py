import sys, time, pickle, numpy as np, copy, os
sys.path.insert(0, ".")
from examples.safe_PCE import *

SEED = 42
K = 40000
np.random.seed(SEED)
random.seed(SEED)

print("=== Single Full Run: K=40000 ===")

# Pretraining
t0 = time.time()
U, hat_Pi_size = pretrain_stage(0.1, 0.05)
print(f"Pretraining: {time.time()-t0:.1f}s, U size={len(U)}")

# Safe policy
env_max = make_env(0.5)
pi_s, u_s_opt, v_s_opt = LP(env_max)
print(f"Safe policy: u_s={u_s_opt:.2f}")

# Build hat_Pi
t0 = time.time()
hat_Pi = policy_set(U, pi_s, show_progress=True)
print(f"Policy set: {time.time()-t0:.1f}s, size={len(hat_Pi)}")

# Test
test_noise = 0.3
print(f"\nTest noise={test_noise}, K={K}")
t0 = time.time()
hat_Pi_copy = copy.deepcopy(hat_Pi)
real_r, real_c = test_stage(K=K, noise=test_noise, pi_s=pi_s, hat_Pi=hat_Pi_copy)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Final (K={K}): regret={real_r[-1]:.2f}, constraint={real_c[-1]:.4f}")
print(f"Regret in thousands: {real_r[-1]/1000:.1f}K")
