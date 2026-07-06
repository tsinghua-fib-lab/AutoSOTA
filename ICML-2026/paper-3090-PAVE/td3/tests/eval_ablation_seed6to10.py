"""
Ablation eval — seed #6-#10 only (same SLURM hardware).
Includes Base from Full + 7 ablation configs.
Uses validated action_extractor.py protocol.
Run from PAVE_Merge root: python td3/tests/eval_ablation_seed6to10.py
"""
from modules.action_extractor import test_some_path

save_dir_root = "./Ablation/td3/pths_seed6to10/"

pth_names = [
    "base_td3",                                      # Base from Full
    "pave_td3_S0.0_T0.0_C0.01_sig0.01_del1.0",     # Curv only
    "pave_td3_S0.0_T0.1_C0.0_sig0.01_del1.0",       # VFC only
    "pave_td3_S0.0_T0.1_C0.01_sig0.01_del1.0",     # VFC + Curv
    "pave_td3_S0.1_T0.0_C0.0_sig0.01_del1.0",       # MPR only
    "pave_td3_S0.1_T0.0_C0.01_sig0.01_del1.0",     # MPR + Curv
    "pave_td3_S0.1_T0.1_C0.0_sig0.01_del1.0",       # MPR + VFC (no Curv)
    "pave_td3_S0.1_T0.1_C0.01_sig0.01_del1.0",     # PAVE (full)
]

print("=== Evaluating Ablation seed #6-#10 (same SLURM hardware) ===")
test_some_path(save_dir_root, True, pth_names, "eval_ablation_seed6to10")
