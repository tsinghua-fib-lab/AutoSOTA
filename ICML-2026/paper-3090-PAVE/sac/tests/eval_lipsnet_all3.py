"""
Q6: LipsNet + PAVE full evaluation (all configs from server3 merge)
SAC, LunarLander, 5 seeds
Run from PAVE_Merge root:
  /home/airlab1tb/anaconda3/envs/gym2/bin/python sac/tests/eval_lipsnet_all3.py
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.action_extractor import test_lips

save_dir_root = "./LipsNet/sac/pths/"

pth_names = [
    # baselines
    "lips_sac_lips1e-05",
    "asap_lips_sac_lamS10.0_lamT0.5_lips1e-05",
    # === All PAVE+LipsNet configs (63 total, sorted by T) ===
    "pave_lips_sac_lamT0.1_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT0.1_lamS2.0_lamC0.05",
    "pave_lips_sac_lamT0.1_lamS5.0_lamC0.05",
    "pave_lips_sac_lamT0.25_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT0.5_lamS0.01_lamC0.05",
    "pave_lips_sac_lamT0.5_lamS0.05_lamC0.05",
    "pave_lips_sac_lamT0.5_lamS0.1_lamC0.01",
    "pave_lips_sac_lamT0.5_lamS0.1_lamC0.025",
    "pave_lips_sac_lamT0.5_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT0.5_lamS0.1_lamC0.1",
    "pave_lips_sac_lamT0.5_lamS0.1_lamC0.5",
    "pave_lips_sac_lamT0.5_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT0.5_lamS1.0_lamC0.05",
    "pave_lips_sac_lamT1.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT1.0_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT1.0_lamS0.5_lamC0.5",
    "pave_lips_sac_lamT1.0_lamS1.0_lamC0.05",
    "pave_lips_sac_lamT2.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT2.0_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT2.0_lamS0.5_lamC0.5",
    "pave_lips_sac_lamT3.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT3.0_lamS0.1_lamC0.1",
    "pave_lips_sac_lamT3.0_lamS0.1_lamC0.5",
    "pave_lips_sac_lamT3.0_lamS0.1_lamC1.0",
    "pave_lips_sac_lamT3.0_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT3.0_lamS1.0_lamC0.05",
    "pave_lips_sac_lamT3.0_lamS2.0_lamC0.05",
    "pave_lips_sac_lamT5.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT5.0_lamS0.1_lamC0.1",
    "pave_lips_sac_lamT5.0_lamS0.1_lamC0.5",
    "pave_lips_sac_lamT5.0_lamS0.1_lamC1.0",
    "pave_lips_sac_lamT5.0_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT5.0_lamS0.5_lamC0.5",
    "pave_lips_sac_lamT5.0_lamS1.0_lamC0.05",
    "pave_lips_sac_lamT5.0_lamS1.0_lamC0.5",
    "pave_lips_sac_lamT5.0_lamS2.0_lamC0.05",
    "pave_lips_sac_lamT7.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT7.0_lamS0.1_lamC0.1",
    "pave_lips_sac_lamT7.0_lamS0.1_lamC0.5",
    "pave_lips_sac_lamT7.0_lamS0.1_lamC1.0",
    "pave_lips_sac_lamT7.0_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT7.0_lamS0.5_lamC0.5",
    "pave_lips_sac_lamT7.0_lamS1.0_lamC0.05",
    "pave_lips_sac_lamT7.0_lamS1.0_lamC0.5",
    "pave_lips_sac_lamT7.0_lamS2.0_lamC0.05",
    "pave_lips_sac_lamT10.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT10.0_lamS0.1_lamC0.1",
    "pave_lips_sac_lamT10.0_lamS0.1_lamC0.5",
    "pave_lips_sac_lamT10.0_lamS0.1_lamC1.0",
    "pave_lips_sac_lamT10.0_lamS0.5_lamC0.05",
    "pave_lips_sac_lamT10.0_lamS0.5_lamC0.5",
    "pave_lips_sac_lamT10.0_lamS1.0_lamC0.05",
    "pave_lips_sac_lamT10.0_lamS1.0_lamC0.5",
    "pave_lips_sac_lamT10.0_lamS2.0_lamC0.05",
    "pave_lips_sac_lamT15.0_lamS0.1_lamC0.05",
    "pave_lips_sac_lamT15.0_lamS0.5_lamC0.5",
    "pave_lips_sac_lamT20.0_lamS0.1_lamC0.05",
]

envs_to_eval = ["lunar"]

out_dir = os.path.join(save_dir_root, "eval_lipsnet_all3")
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
