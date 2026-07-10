import sys, time, pickle, numpy as np, copy, os
sys.path.insert(0, ".")
from examples.safe_PCE import *

SEED = 42
K = 40000
N_TASKS = 5
EPSILON = 0.005
DELTA = 0.1

np.random.seed(SEED)
random.seed(SEED)
os.makedirs("/repo/examples/data/reproduction", exist_ok=True)

print("=" * 60)
print(f"Full Reproduction: K={K}, N_TASKS={N_TASKS}, epsilon={EPSILON}")
print("=" * 60)

# Step 1: Pretraining
print("\n[1/4] Pretraining...")
t0 = time.time()
U, hat_Pi_size = pretrain_stage(DELTA, EPSILON)
print(f"Done in {time.time()-t0:.1f}s. U size={len(U)}")

# Step 2: Safe policy
print("\n[2/4] Computing safe policy...")
env_max = make_env(0.5)
pi_s, u_s_opt, v_s_opt = LP(env_max)
print(f"pi_s: u={u_s_opt:.2f}")

# Step 3: Build policy set
print("\n[3/4] Building policy-value set...")
t0 = time.time()
hat_Pi = policy_set(U, pi_s, show_progress=True)
print(f"Done in {time.time()-t0:.1f}s. Size={len(hat_Pi)}")

# Step 4: Test
print(f"\n[4/4] Running test stage for {N_TASKS} tasks...")
test_noises = list(truncated_gaussian(size=N_TASKS))
all_r, all_c = [], []

for idx, noise in enumerate(test_noises):
    print(f"\n--- Task {idx+1}/{N_TASKS}: noise={noise:.6f} ---")
    t0 = time.time()
    hat_Pi_copy = copy.deepcopy(hat_Pi)
    real_r, real_c = test_stage(K=K, noise=noise, pi_s=pi_s, hat_Pi=hat_Pi_copy)
    elapsed = time.time() - t0
    final_r = real_r[-1]
    final_c = real_c[-1]
    print(f"  Done in {elapsed:.1f}s. Final regret={final_r:.2f} ({final_r/1000:.1f}K), constraint={final_c:.4f}")
    all_r.append(real_r)
    all_c.append(real_c)

# Summary
all_r = np.array(all_r)
all_c = np.array(all_c)
valid = all_r[:, -1] > 100
n_valid = np.sum(valid)
print(f"\n{=*60}")
print(f"SUMMARY: {n_valid}/{N_TASKS} valid runs")
if n_valid > 0:
    vr = all_r[valid]
    vc = all_c[valid]
    mean_r = np.mean(vr[:, -1])
    mean_c = np.mean(vc[:, -1])
    print(f"Mean reward regret: {mean_r:.2f} ({mean_r/1000:.1f}K)")
    print(f"Mean constraint: {mean_c:.4f}")
    print(f"Per-run: {[f{vr[i,
