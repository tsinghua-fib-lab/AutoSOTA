"""
Q9: LipsNet + PAVE evaluation (SAC, LunarLander only, 5 seeds)
Eval-only: no training, just loads pretrained models and computes metrics.
Run from PAVE_Merge root: python sac/tests/eval_lipsnet.py

Models: lips_sac (LipsNet only), asap_lips_sac (ASAP+LipsNet), pave_lips_sac (PAVE+LipsNet)
"""
import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.action_extractor import test_lips, load_seeds

from modules.envs import make_lunar_env

seed_file_path = "./sac/tests/validation_seeds.txt"

save_dir_root = "./results/pths/"

# Algorithm configs to evaluate
# lips_sac: LipsNet only (baseline)
# asap_lips_sac: ASAP + LipsNet
# pave_lips_sac default: lamT0.5_lamS0.1_lamC0.05
pth_names = [
    "lips_sac_lips1e-05",
    "asap_lips_sac_lamS10.0_lamT0.5_lips1e-05",
    "pave_lips_sac_lamT0.5_lamS0.1_lamC0.05",
]

# Only lunar has models
envs_to_eval = ["lunar"]

out_dir = os.path.join(save_dir_root, "eval_lipsnet")
os.makedirs(out_dir, exist_ok=True)

sm_mean_rows = [["al_name"] + envs_to_eval]
sm_std_rows = [["al_name"] + envs_to_eval]
re_mean_rows = [["al_name"] + envs_to_eval]
re_std_rows = [["al_name"] + envs_to_eval]

for al_name in pth_names:
    sm_mean_row = [al_name]
    sm_std_row = [al_name]
    re_mean_row = [al_name]
    re_std_row = [al_name]
    for env_name in envs_to_eval:
        print(f"Evaluating {al_name} on {env_name}...")
        result = test_lips(save_dir_root, al_name, env_name, deterministic=True)
        if isinstance(result, tuple) and len(result) == 4:
            sm_mean, sm_std, re_mean, re_std = result
        else:
            sm_mean, sm_std, re_mean, re_std = 0.0, 0.0, 0.0, 0.0
        sm_mean_row.append(sm_mean)
        sm_std_row.append(sm_std)
        re_mean_row.append(re_mean)
        re_std_row.append(re_std)
        print(f"  re={re_mean:.1f}({re_std:.1f}), sm={sm_mean:.4f}({sm_std:.4f})")
    sm_mean_rows.append(sm_mean_row)
    sm_std_rows.append(sm_std_row)
    re_mean_rows.append(re_mean_row)
    re_std_rows.append(re_std_row)

for name, rows in [("sm_mean", sm_mean_rows), ("sm_std", sm_std_rows),
                    ("re_mean", re_mean_rows), ("re_std", re_std_rows)]:
    with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

print(f"\nResults saved to {out_dir}/")
