"""
Q10: Hyperparameter Sensitivity evaluation
21 configs × 5 seeds × 2 envs (Walker + Lunar), TD3 only.
Sweeps: λ₁(S), λ₂(T), λ₃(C), σ, δ — each with 5 values around default.
Default config: S0.1_T0.1_C0.01_sig0.01_del1.0

Eval-only: no training, just loads pretrained models and computes metrics.
Run from PAVE_Merge root: python td3/tests/eval_sensitivity.py
"""
from modules.action_extractor import test_some_path
from modules.q_extractor2 import test_some_path as test_some_path_q

save_dir_root = "./Sensitivity/td3/pths/"

# 21 unique configs (default is shared across all sweeps)
pth_names = [
    # === S sweep (λ₁ MPR): T=0.1, C=0.01, σ=0.01, δ=1.0 fixed ===
    "pave_td3_S0.001_T0.1_C0.01_sig0.01_del1.0",
    "pave_td3_S0.01_T0.1_C0.01_sig0.01_del1.0",
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del1.0",   # default
    "pave_td3_S1.0_T0.1_C0.01_sig0.01_del1.0",
    "pave_td3_S10.0_T0.1_C0.01_sig0.01_del1.0",
    # === T sweep (λ₂ VFC): S=0.1, C=0.01, σ=0.01, δ=1.0 fixed ===
    "pave_td3_S0.1_T0.001_C0.01_sig0.01_del1.0",
    "pave_td3_S0.1_T0.01_C0.01_sig0.01_del1.0",
    # default already listed
    "pave_td3_S0.1_T1.0_C0.01_sig0.01_del1.0",
    "pave_td3_S0.1_T10.0_C0.01_sig0.01_del1.0",
    # === C sweep (λ₃ Curv): S=0.1, T=0.1, σ=0.01, δ=1.0 fixed ===
    "pave_td3_S0.1_T0.1_C0.0001_sig0.01_del1.0",
    "pave_td3_S0.1_T0.1_C0.001_sig0.01_del1.0",
    # default already listed
    "pave_td3_S0.1_T0.1_C0.1_sig0.01_del1.0",
    "pave_td3_S0.1_T0.1_C1.0_sig0.01_del1.0",
    # === σ sweep (MPR noise std): S=0.1, T=0.1, C=0.01, δ=1.0 fixed ===
    "pave_td3_S0.1_T0.1_C0.01_sig0.001_del1.0",
    "pave_td3_S0.1_T0.1_C0.01_sig0.005_del1.0",
    # default already listed (sig0.01)
    "pave_td3_S0.1_T0.1_C0.01_sig0.05_del1.0",
    "pave_td3_S0.1_T0.1_C0.01_sig0.1_del1.0",
    # === δ sweep (Curv margin): S=0.1, T=0.1, C=0.01, σ=0.01 fixed ===
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del0.1",
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del0.5",
    # default already listed (del1.0)
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del2.0",
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del5.0",
]

# reward + smoothness
print("=== Evaluating reward & smoothness (Sensitivity) ===")
test_some_path(save_dir_root, True, pth_names, "eval_sensitivity")

# Q-function gradients (M, μ)
print("=== Evaluating Q-function gradients (Sensitivity) ===")
test_some_path_q(save_dir_root, True, pth_names, "eval_sensitivity_q")
