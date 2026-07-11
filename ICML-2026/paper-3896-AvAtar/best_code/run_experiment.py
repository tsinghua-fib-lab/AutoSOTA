import subprocess
import sys
import re
import os

os.environ["PYTHONPATH"] = "/autosota_cache/PlanetAlign:" + os.environ.get("PYTHONPATH", "")

seeds = [0, 1, 2, 3, 4]
results = []

for seed in seeds:
    sep = "=" * 60
    print("\n" + sep)
    print("RUNNING SEED {}".format(seed))
    print(sep + "\n")
    
    cmd = [
        "python3", "active_na.py",
        "--alg", "PARROT",
        "--dataset", "phone-email",
        "--device", "cuda",
        "--query_round", "10",
        "--query_portion", "0.2",
        "--init_train_ratio", "0.2",
        "--outIter", "10",
        "--modes", "sq_l2_adjoint_grad",
        "--anchor_selection_seed", str(seed)
    ]
    
    result = subprocess.run(cmd, cwd="/repo/source", capture_output=True, text=True, timeout=1200)
    
    mrr_values = []
    for line in result.stdout.split("\n"):
        if "MRR:" in line:
            match = re.search(r"MRR:\s*([\d.]+)", line)
            if match:
                mrr_values.append(float(match.group(1)))
    
    final_mrr = mrr_values[-1] if mrr_values else None
    mrr_at_round_10 = mrr_values[-11] if len(mrr_values) >= 11 else (mrr_values[-1] if mrr_values else None)
    
    print("\nSeed {}: Final MRR = {}, MRR at Round 10 = {}".format(seed, final_mrr, mrr_at_round_10))
    results.append({"seed": seed, "final_mrr": final_mrr, "mrr_round_10": mrr_at_round_10})
    
    if result.stderr:
        for line in result.stderr.split("\n"):
            if "UserWarning" not in line and "Triggered internally" not in line:
                print("STDERR: " + line, file=sys.stderr)

sep = "=" * 60
print("\n" + sep)
print("SUMMARY")
print(sep)
for r in results:
    print("Seed {}: Round 10 MRR = {}, Final MRR = {}".format(
        r["seed"], r["mrr_round_10"], r["final_mrr"]))

valid = [r["mrr_round_10"] for r in results if r["mrr_round_10"] is not None]
if valid:
    avg_round_10 = sum(valid) / len(valid)
    print("\nAverage MRR at Round 10: {:.4f}".format(avg_round_10))
