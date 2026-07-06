#!/usr/bin/env python3
"""Evaluate PAVE+TD3 on Pendulum-v1.

Paper hyperparameters: lamS=2.0, lamT=0.005, lamC=2.0, sigma=0.01, delta=1.0
5 training seeds, 10 evaluation episodes per seed.
"""
import sys, os, numpy as np
os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/td3/tests")
sys.path.insert(0, "/repo/td3")

from modules.envs import make_pendulum_env
from modules.action_extractor import calculate_smoothness_np
from models.custom_td3 import CustomTD3

PTH_DIR = "/repo/td3/results/pths/pendulum/"
EVAL_SEEDS = [857751, 968229, 423337, 499844, 985365,
              713160, 643903, 235098, 197317, 212049]

def main():
    env = make_pendulum_env()()
    model_dirs = sorted([d for d in os.listdir(PTH_DIR) if "pave_td3" in d and os.path.isfile(os.path.join(PTH_DIR, d, "final.zip"))])

    if not model_dirs:
        print("ERROR: No PAVE model directories found with final.zip")
        sys.exit(1)

    print("Found %d trained PAVE models" % len(model_dirs))
    all_returns = []
    all_smoothness = []

    for model_dir in model_dirs:
        final_path = os.path.join(PTH_DIR, model_dir, "final.zip")
        seed_str = model_dir.split("_")[-1]
        print("\nEvaluating seed %s..." % seed_str)
        model = CustomTD3.load(final_path, env=env)

        for eval_seed in EVAL_SEEDS:
            obs, _ = env.reset(seed=eval_seed)
            actions = []
            total_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                actions.append(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            sm = calculate_smoothness_np(np.array(actions))
            all_returns.append(total_reward)
            all_smoothness.append(sm)

    env.close()

    re_mean = np.mean(all_returns)
    re_std = np.std(all_returns)
    sm_mean = np.mean(all_smoothness)
    sm_std = np.std(all_smoothness)

    sep = "=" * 60
    print("\n%s" % sep)
    print("PAVE+TD3 on Pendulum-v1 (5 training seeds x 10 eval episodes = 50 episodes)")
    print("  Cumulative Return: %.2f +/- %.2f" % (re_mean, re_std))
    print("  Smoothness Score:  %.4f +/- %.4f" % (sm_mean, sm_std))
    print("  Paper Table 1:     re=-167.6+/-77.5, sm=0.351+/-0.118")
    print(sep)

    # Save to CSV
    csv_path = "/repo/eval_results.csv"
    with open(csv_path, "w") as f:
        f.write("metric,mean,std\n")
        f.write("cumulative_return,%.4f,%.4f\n" % (re_mean, re_std))
        f.write("smoothness_score,%.4f,%.4f\n" % (sm_mean, sm_std))
    print("Results saved to %s" % csv_path)

if __name__ == "__main__":
    main()
