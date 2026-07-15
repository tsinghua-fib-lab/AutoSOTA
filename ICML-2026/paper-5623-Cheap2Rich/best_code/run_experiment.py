#!/usr/bin/env python3
"""Multi-run reproduction script for Cheap2Rich paper (Paper 5623)."""
import sys, os, json, subprocess
import numpy as np

os.chdir("/repo")

seeds = [42, 123, 456, 789, 101112]
results = []

for run_idx, seed in enumerate(seeds):
    print(f"\n{'='*80}")
    print(f"RUN {run_idx+1}/{len(seeds)} (seed={seed})")
    print(f"{'='*80}")

    env = os.environ.copy()
    env["RUN_SEED"] = str(seed)

    proc = subprocess.run(
        [sys.executable, "Cheap2Rich.py"],
        capture_output=True, text=True, env=env, timeout=600
    )

    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode != 0:
        print(f"RUN FAILED, rc={proc.returncode}")
        print(f"STDERR: {stderr[-2000:]}")
        continue

    rmse_val = None; rmse_full = None
    rmse_lf_val = None; ssim_val = None; ssim_full = None

    for line in stdout.split("\n"):
        s = line.strip()
        if "LF+HF RMSE:" in s:
            try: rmse_val = float(s.split("LF+HF RMSE:")[1].strip())
            except: pass
        if "Full dataset LF+HF RMSE:" in s:
            try: rmse_full = float(s.split("Full dataset LF+HF RMSE:")[1].strip())
            except: pass
        if "LF-only (with GAN) RMSE:" in s:
            try: rmse_lf_val = float(s.split("LF-only (with GAN) RMSE:")[1].strip())
            except: pass

    ssim_lines = [l for l in stdout.split("\n") if "SSIM between lf_hf predictions and high-fidelity data:" in l]
    if len(ssim_lines) >= 1:
        try: ssim_val = float(ssim_lines[0].split(":")[1].strip())
        except: pass
    if len(ssim_lines) >= 2:
        try: ssim_full = float(ssim_lines[1].split(":")[1].strip())
        except: pass

    print(f"  RMSE_val={rmse_val:.6f}, RMSE_full={rmse_full}, SSIM_val={ssim_val}, SSIM_full={ssim_full}")

    results.append({
        "seed": seed,
        "rmse_val": rmse_val,
        "rmse_full": rmse_full,
        "rmse_lf_val": rmse_lf_val,
        "ssim_val": ssim_val,
        "ssim_full": ssim_full,
    })

print(f"\n{'='*80}")
print(f"FINAL RESULTS (over {len(results)} runs)")
print(f"{'='*80}")

for key, label in [("rmse_val", "RMSE (val)"), ("rmse_full", "RMSE (full)"),
                    ("ssim_val", "SSIM (val)"), ("ssim_full", "SSIM (full)")]:
    vals = [r[key] for r in results if r[key] is not None]
    if vals:
        print(f"{label}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}  (n={len(vals)})")

for r in results:
    print(f"  seed={r['seed']}: RMSE_val={r['rmse_val']:.6f}, RMSE_full={r['rmse_full']}, "
          f"SSIM_val={r['ssim_val']}, SSIM_full={r['ssim_full']}")

# Save
with open("reproduction_results.json", "w") as f:
    json.dump({
        "rmse_val_mean": float(np.mean([r["rmse_val"] for r in results if r["rmse_val"] is not None])),
        "rmse_full_mean": float(np.mean([r["rmse_full"] for r in results if r["rmse_full"] is not None])),
        "ssim_full_mean": float(np.mean([r["ssim_full"] for r in results if r["ssim_full"] is not None])),
        "per_run": results
    }, f, indent=2, default=str)
print("\nResults saved to reproduction_results.json")
