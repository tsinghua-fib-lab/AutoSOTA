import subprocess
import sys
import os
import re

seeds = [0, 1, 2, 3, 4]
results = {}
work_dir = "/repo/Allen-Cahn"

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    
    # Read the patched script and replace the seed line
    with open(f"{work_dir}/TINN-AC-patched.py", "r") as f:
        content = f.read()
    
    # Replace the PRNGKey seed
    content = re.sub(r'random\.PRNGKey\(\d+\)', f'random.PRNGKey({seed})', content)
    
    # Write seed-specific script
    script_path = f"{work_dir}/TINN-AC-seed{seed}.py"
    with open(script_path, "w") as f:
        f.write(content)
    
    # Run it
    result = subprocess.run(
        ["python3", script_path],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=7200  # 2h per seed
    )
    
    # Save output
    with open(f"{work_dir}/seed{seed}.log", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    
    # Extract rel-L2-Error
    for line in result.stdout.split("\n"):
        if "rel-L2-Error:" in line:
            match = re.search(r'rel-L2-Error:\s*([\d.e+\-]+)', line)
            if match:
                results[seed] = float(match.group(1))
                print(f"Seed {seed}: rel-L2-Error = {results[seed]:.5e}")
    
    print(f"Seed {seed} exit code: {result.returncode}")

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for seed in seeds:
    if seed in results:
        print(f"Seed {seed}: {results[seed]:.5e}")
    else:
        print(f"Seed {seed}: FAILED")

if results:
    import numpy as np
    vals = list(results.values())
    print(f"Mean: {np.mean(vals):.5e}")
    print(f"Std:  {np.std(vals):.5e}")
    print(f"Paper: 3.85E-06 +/- 1.48E-06")
