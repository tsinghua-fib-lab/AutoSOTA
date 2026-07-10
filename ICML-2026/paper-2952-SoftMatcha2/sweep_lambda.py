#!/usr/bin/env python3
"""Quick sweep of LAMBDA values for BM25/SoftMatcha interpolation."""
import subprocess, sys

# Read current eval_ir_v2.py
with open("/repo/eval_ir_v2.py") as f:
    original = f.read()

# Test different LAMBDA values
lambdas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
results = []

for lam in lambdas:
    # Modify LAMBDA in the script
    modified = original.replace("LAMBDA = 0.5", f"LAMBDA = {lam}")
    with open("/repo/eval_ir_v2_sweep.py", "w") as f:
        f.write(modified)
    
    print(f"\n=== Testing LAMBDA={lam} ===", flush=True)
    result = subprocess.run(
        ["uv", "run", "python3", "/repo/eval_ir_v2_sweep.py"],
        capture_output=True, text=True, timeout=600,
        cwd="/repo",
        env={**__import__("os").environ, "PATH": f"/root/.local/bin:{__import__(\"os\").environ.get(\"PATH\",\"\")}"}
    )
    
    for line in result.stdout.split("\n"):
        if "P@20=" in line and "R@1000=" in line and "softmatcha2" in result.stdout:
            # Only capture softmatcha2 result
            pass
        if "[softmatcha2]" in result.stdout:
            # The softmatcha2 line itself has the metrics
            pass
    
    # Parse metrics
    for line in result.stdout.split("\n"):
        if "softmatcha2" in line.lower() and "Running" not in line:
            continue
        if "P@20=" in line and "R@1000=" in line:
            # This is a metrics line, find the softmatcha one
            pass
    
    # Just parse all P@20 lines
    bm25_p20 = None
    sm_p20 = None
    sm_r1000 = None
    in_softmatcha = False
    
    for line in result.stdout.split("\n"):
        if "[softmatcha2] Running..." in line:
            in_softmatcha = True
        elif "[bm25] Running..." in line:
            in_softmatcha = False
        elif in_softmatcha and "P@20=" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("P@20="):
                    sm_p20 = float(p.split("=")[1].rstrip(","))
                if p.startswith("R@1000="):
                    sm_r1000 = float(p.split("=")[1].rstrip(","))
    
    if sm_p20 is not None:
        results.append((lam, sm_p20, sm_r1000))
        print(f"  LAMBDA={lam}: P@20={sm_p20:.1f}, R@1000={sm_r1000:.1f}", flush=True)
    else:
        print(f"  LAMBDA={lam}: FAILED to parse", flush=True)

print("\n=== RESULTS ===")
print(f"{LAMBDA:>8} {P@20:>8} {R@1000:>8}")
for lam, p20, r1000 in sorted(results, key=lambda x: -x[1]):
    print(f"{lam:>8.1f} {p20:>8.1f} {r1000:>8.1f}")
