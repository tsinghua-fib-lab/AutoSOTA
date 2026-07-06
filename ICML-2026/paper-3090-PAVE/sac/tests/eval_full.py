"""
Q8: Full evaluation (10 seeds × 5 methods × 6 envs) - SAC
Eval-only: no training, just loads pretrained models and computes metrics.
Run from PAVE_Merge root: python sac/tests/eval_full.py
"""
from modules.action_extractor import test_some_path

save_dir_root = "./Full/sac/pths/"

pth_names = ["vanilla", "caps_sac", "grad_sac", "asap_sac", "pave_sac"]

# reward + smoothness
print("=== Evaluating reward & smoothness ===")
test_some_path(save_dir_root, True, pth_names, "eval_full")
