"""
Q3 + Q7: Ablation evaluation (7 configs × 5 seeds, Lunar + Walker TD3)
Eval-only: no training, just loads pretrained models and computes metrics.
Run from PAVE_Merge root: python td3/tests/eval_ablation.py

Configs (S=MPR, T=VFC, C=Curv):
  S0.0_T0.0_C0.01 = Curv only
  S0.0_T0.1_C0.0  = VFC only
  S0.0_T0.1_C0.01 = VFC + Curv
  S0.1_T0.0_C0.0  = MPR only
  S0.1_T0.0_C0.01 = MPR + Curv
  S0.1_T0.1_C0.0  = MPR + VFC (no Curv)
  S0.1_T0.1_C0.01 = PAVE (full)
"""
from modules.action_extractor import test_some_path
from modules.q_extractor2 import test_some_path as test_some_path_q

save_dir_root = "./Ablation/td3/pths/"

pth_names = [
    "pave_td3_S0.0_T0.0_C0.01_sig0.01_del1.0",   # Curv only
    "pave_td3_S0.0_T0.1_C0.0_sig0.01_del1.0",     # VFC only
    "pave_td3_S0.0_T0.1_C0.01_sig0.01_del1.0",    # VFC + Curv
    "pave_td3_S0.1_T0.0_C0.0_sig0.01_del1.0",     # MPR only
    "pave_td3_S0.1_T0.0_C0.01_sig0.01_del1.0",    # MPR + Curv
    "pave_td3_S0.1_T0.1_C0.0_sig0.01_del1.0",     # MPR + VFC (no Curv)
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del1.0",    # PAVE (full)
]

# reward + smoothness
print("=== Evaluating reward & smoothness (Ablation) ===")
test_some_path(save_dir_root, True, pth_names, "eval_ablation")

# Q-function gradients (M, μ) — needed for Q3 (Does L_Curv preserve μ?)
print("=== Evaluating Q-function gradients (Ablation) ===")
test_some_path_q(save_dir_root, True, pth_names, "eval_ablation_q")
