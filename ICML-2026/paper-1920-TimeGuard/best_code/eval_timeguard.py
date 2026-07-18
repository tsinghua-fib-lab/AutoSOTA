#!/usr/bin/env python3
"""TimeGuard evaluation script for PEMS03+BackTime.
Averaged over FEDformer, SimpleTM, TimesNet per paper Section 3.1.
"""
import subprocess
import sys
import re
import os

os.chdir("/repo")

MODELS = ["FEDformer", "SimpleTM", "TimesNet"]
CONFIG_BASE = "./configs/timeguard/PEMS03_backtime_FEDformer_1212"

results_cln = []
results_atk = []

print("=== TimeGuard Defense Evaluation ===")
print("Dataset: PEMS03, Attack: BackTime, Pattern: cone")
print("L_in=12, L_out=12, eta_T=0.03, eta_S=0.3")
print("alpha=0.2, beta=0.5, T_b=10, T_1=10, T_2=90")
print()

for model in MODELS:
    print(f"--- Running TimeGuard for {model} ---")
    config = f"{CONFIG_BASE}/{model}/TimeGuard.yaml"

    result = subprocess.run(
        ["python3", "defense_timeguard.py", "--defense_config_path", config],
        capture_output=True, text=True, cwd="/repo"
    )

    stdout = result.stdout
    stderr = result.stderr
    if stderr:
        print(f"STDERR: {stderr[-500:]}")

    # Extract final results
    match_cln = re.search(r"'cln_mae':\s*([0-9.]+)", stdout)
    match_atk = re.search(r"'atk_mae':\s*([0-9.]+)", stdout)

    if match_cln and match_atk:
        cln = float(match_cln.group(1))
        atk = float(match_atk.group(1))
        results_cln.append(cln)
        results_atk.append(atk)
        print(f"  {model}: MAEc={cln:.6f}, MAEp={atk:.6f}")
    else:
        print(f"  ERROR: Could not parse results for {model}")
        if match_cln:
            print(f"  Found cln_mae: {match_cln.group(1)}")
        if match_atk:
            print(f"  Found atk_mae: {match_atk.group(1)}")
        print(f"  stdout tail: {stdout[-500:]}")

print()
print("=== Results ===")

# Compute averages
avg_cln = sum(results_cln) / len(results_cln)
avg_atk = sum(results_atk) / len(results_atk)

# Compute FDER using undefended baselines from paper Table 3
maec_undef = 17.607
maep_undef = 14.201

rho_maep = max(0, 1 - maep_undef / avg_atk)
rho_maec = max(0, 1 - maec_undef / avg_cln)
fder = (rho_maep - rho_maec + 1) / 2

print(f"Models averaged: {', '.join(MODELS)}")
print(f"Individual MAEc: {', '.join(f'{c:.3f}' for c in results_cln)}")
print(f"Individual MAEp: {', '.join(f'{a:.3f}' for a in results_atk)}")
print(f"Average MAEc: {avg_cln:.3f}")
print(f"Average MAEp: {avg_atk:.3f}")
print(f"FDER: {fder:.3f}")
print()
print(f"Paper reported:    MAEc=18.048, MAEp=39.303, FDER=0.808")
print(f"Our reproduction:  MAEc={avg_cln:.3f}, MAEp={avg_atk:.3f}, FDER={fder:.3f}")
print()
print("=== Evaluation Complete ===")
