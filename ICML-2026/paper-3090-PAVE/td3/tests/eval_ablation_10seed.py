"""
Ablation 10-seed evaluation — same protocol as eval_ablation.py but on pths_10seed.
Uses action_extractor.py (validated protocol).
Run from PAVE_Merge root: python td3/tests/eval_ablation_10seed.py
"""
from modules.action_extractor import test_some_path

save_dir_root = "./Ablation/td3/pths_10seed/"

pth_names = [
    "pave_td3_S0.0_T0.0_C0.01_sig0.01_del1.0",   # Curv only
    "pave_td3_S0.0_T0.1_C0.0_sig0.01_del1.0",     # VFC only
    "pave_td3_S0.0_T0.1_C0.01_sig0.01_del1.0",    # VFC + Curv
    "pave_td3_S0.1_T0.0_C0.0_sig0.01_del1.0",     # MPR only
    "pave_td3_S0.1_T0.0_C0.01_sig0.01_del1.0",    # MPR + Curv
    "pave_td3_S0.1_T0.1_C0.0_sig0.01_del1.0",     # MPR + VFC (no Curv)
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del1.0",    # PAVE (full)
]

print("=== Evaluating reward & smoothness (Ablation 10-seed) ===")
test_some_path(save_dir_root, True, pth_names, "eval_ablation_10seed")
