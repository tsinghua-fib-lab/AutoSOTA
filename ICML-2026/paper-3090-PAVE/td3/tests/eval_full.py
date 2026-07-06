"""
Q8: Full evaluation (10 seeds × 5 methods × 6 envs) - TD3
Eval-only: no training, just loads pretrained models and computes metrics.
Run from PAVE_Merge root: python td3/tests/eval_full.py
"""
from modules.action_extractor import test_some_path
from modules.q_extractor2 import test_some_path as test_some_path_q

save_dir_root = "./Full/td3/pths/"

pth_names = ["base_td3", "caps_td3", "grad_td3", "asap_td3", "pave_td3"]

# reward + smoothness
print("=== Evaluating reward & smoothness ===")
test_some_path(save_dir_root, True, pth_names, "eval_full")

# Q-function gradients (M, μ)
print("=== Evaluating Q-function gradients ===")
test_some_path_q(save_dir_root, True, pth_names, "eval_full_q")
