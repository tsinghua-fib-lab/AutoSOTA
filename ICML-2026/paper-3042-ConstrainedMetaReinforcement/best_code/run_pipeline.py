#!/usr/bin/env python3
"""
Full pipeline runner for paper 3042 optimization.
Generates pre-computed pickle files for eval_reproduction.py.
v2: Supports safe policy fallback for high-noise tasks.
"""
import sys, time, pickle, numpy as np, copy, os, argparse
sys.path.insert(0, ".")
from examples.safe_PCE import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epsilon', type=float, default=0.005)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--n-tasks', type=int, default=20)
    parser.add_argument('--k', type=int, default=40000)
    parser.add_argument('--safe-noise', type=float, default=0.3,
                        help='Noise level for computing safe policy')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs("/repo/examples/data", exist_ok=True)

    print("Pipeline: K=%d, N_TASKS=%d, epsilon=%.4f, safe_noise=%.2f" % (
        args.k, args.n_tasks, args.epsilon, args.safe_noise))

    # Step 1: Pretraining
    print("\n[1/5] Pretraining...")
    t0 = time.time()
    U, hat_Pi_size = pretrain_stage(args.delta, args.epsilon)
    print("Done in %.1fs. U size=%d" % (time.time() - t0, len(U)))

    # Step 2: Primary safe policy (median-noise)
    print("\n[2/5] Computing primary safe policy (noise=%.2f)..." % args.safe_noise)
    t0 = time.time()
    env_safe = make_env(args.safe_noise)
    pi_s, u_s_opt, v_s_opt = LP(env_safe)
    print("pi_s: u=%.4f, v=%.4f, time=%.1fs" % (u_s_opt, v_s_opt, time.time() - t0))

    # Step 3: Build primary policy set
    print("\n[3/5] Building primary policy-value set...")
    t0 = time.time()
    hat_Pi = policy_set(U, pi_s, show_progress=True)
    print("Done in %.1fs. Size=%d" % (time.time() - t0, len(hat_Pi)))

    # Step 4: Fallback safe policy (max-noise) + policy set
    print("\n[4/5] Computing fallback safe policy (noise=0.50) and hat_Pi...")
    t0 = time.time()
    env_fallback = make_env(0.5)
    pi_s_fallback, u_fb, v_fb = LP(env_fallback)
    print("pi_s_fallback LP: u=%.4f, v=%.4f, time=%.1fs" % (u_fb, v_fb, time.time() - t0))
    hat_Pi_fallback = policy_set(U, pi_s_fallback, show_progress=False)
    print("hat_Pi_fallback built in %.1fs" % (time.time() - t0))

    # Save U and primary hat_Pi
    with open("/repo/examples/data/U_hatPi.pkl", "wb") as f:
        pickle.dump({"U": U, "hat_Pi": hat_Pi}, f)

    # Step 5: Test stage
    print("\n[5/5] Running test stage for %d tasks..." % args.n_tasks)
    np.random.seed(args.seed + 1)
    random.seed(args.seed + 1)
    test_noises = list(truncated_gaussian(size=args.n_tasks))
    all_r, all_c = [], []
    n_fallback_used = 0

    for idx, noise in enumerate(test_noises):
        print("\n--- Task %d/%d: noise=%.6f ---" % (idx + 1, args.n_tasks, noise))
        t0 = time.time()

        # Check if primary safe policy is feasible on this test task
        env_test = make_env(noise)
        _, c_check = value_function_r_c(env_test, pi_s)
        if float(c_check) > 1.5:
            _, c_fb = value_function_r_c(env_test, pi_s_fallback)
            if float(c_fb) <= 1.5:
                print("  Fallback: primary c=%.4f > 1.5, using fallback c=%.4f" % (
                    float(c_check), float(c_fb)))
                pi_s_use = pi_s_fallback
                hat_Pi_copy = copy.deepcopy(hat_Pi_fallback)
                n_fallback_used += 1
            else:
                print("  WARNING: both infeasible, using primary")
                pi_s_use = pi_s
                hat_Pi_copy = copy.deepcopy(hat_Pi)
        else:
            pi_s_use = pi_s
            hat_Pi_copy = copy.deepcopy(hat_Pi)

        real_r, real_c = test_stage(K=args.k, noise=noise, pi_s=pi_s_use,
                                     hat_Pi=hat_Pi_copy)
        elapsed = time.time() - t0
        final_r = real_r[-1] if len(real_r) > 0 else -1
        final_c = real_c[-1] if len(real_c) > 0 else -1
        print("  Done in %.1fs. Final regret=%.2f, constraint=%.4f" % (elapsed, final_r, final_c))
        all_r.append(real_r)
        all_c.append(real_c)

    # Convert to fixed-length arrays
    r_all_run = np.array([np.pad(r[:args.k], (0, max(0, args.k - len(r))),
                                 constant_values=0) for r in all_r])
    c_all_run = np.array([np.pad(c[:args.k], (0, max(0, args.k - len(c))),
                                 constant_values=0) for c in all_c])

    # Save results
    with open("/repo/examples/data/r_c_history.pkl", "wb") as f:
        pickle.dump({"r_all_run": r_all_run, "c_all_run": c_all_run}, f)

    # Summary
    valid = r_all_run[:, -1] > 100
    n_valid = int(np.sum(valid))
    print("\n" + "=" * 60)
    print("SUMMARY: %d/%d valid runs (fallback used: %d)" % (
        n_valid, args.n_tasks, n_fallback_used))
    if n_valid > 0:
        vr = r_all_run[valid]
        vc = c_all_run[valid]
        mean_r = float(np.mean(vr[:, -1]))
        mean_c = float(np.mean(vc[:, -1]))
        print("Mean reward regret: %.2f (%.1fK)" % (mean_r, mean_r / 1000))
        print("Mean constraint: %.4f" % mean_c)
        print("Per-run regret: %s" % [float(x) for x in vr[:, -1]])
        print("reward_regret: %.2f" % mean_r)
        print("constraint_value: %.4f" % mean_c)

if __name__ == "__main__":
    main()
