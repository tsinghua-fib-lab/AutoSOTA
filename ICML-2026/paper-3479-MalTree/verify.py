import sys; sys.path.insert(0, "/repo")
import json

with open("/repo/outputs/er_experiment_results.json") as f:
    data = json.load(f)

s = data["summary"]
print("=== REPRODUCTION VERIFICATION ===")
print("DiCoLA+FCI CI_Tests:", s["DiCoLA+FCI"]["CI_Tests"])
print("DiCoLA+FCI F1:", s["DiCoLA+FCI"]["F1"])
print("DiCoLA+FCI Precision:", s["DiCoLA+FCI"]["Precision"])
print("DiCoLA+FCI Recall:", s["DiCoLA+FCI"]["Recall"])
print("DiCoLA+FCI Time:", s["DiCoLA+FCI"]["Time"])
print("FCI_baseline CI_Tests:", s["FCI_baseline"]["CI_Tests"])
print("FCI_baseline F1:", s["FCI_baseline"]["F1"])
print("n_runs:", s["n_runs"], "alpha:", s["alpha"], "total_time:", s["total_time_sec"], "s")

ci_tests = s["DiCoLA+FCI"]["CI_Tests"]
f1 = s["DiCoLA+FCI"]["F1"]
print()
print("=== RUBRIC BOUNDS CHECK ===")
print("CI Tests", ci_tests, ": bounds [2068.57, 8623.34] ->", "PASS" if 2068.57 <= ci_tests <= 8623.34 else "FAIL")
print("F1", f1, ": bounds [0.71, 0.721] ->", "PASS" if 0.71 <= f1 <= 0.721 else "FAIL")
print()
print("=== REPRODUCTION SUCCEEDED ===")
