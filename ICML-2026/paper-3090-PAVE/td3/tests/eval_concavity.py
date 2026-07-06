"""
Q9: Concavity satisfaction measurement (5 methods × 6 envs, TD3)
Measures fraction of (s,a) pairs where tr(∇²_aa Q) < -δ.
Follows q_extractor2.py pipeline exactly.

Run from PAVE_Merge root:
  /home/airlab1tb/anaconda3/envs/gym2/bin/python td3/tests/eval_concavity.py
"""
from modules.q_concavity import test_some_path_concavity

save_dir_root = "./Full/td3/pths/"
pth_names = ["base_td3", "caps_td3", "grad_td3", "asap_td3", "pave_td3"]

print("=== Evaluating concavity satisfaction ===")
test_some_path_concavity(save_dir_root, True, pth_names, "eval_concavity")
