"""Fine phi sweep with Fisher weights - runs full evaluation for each phi."""
import numpy as np, torch, time, sys
sys.path.insert(0, "/repo/clean_synthetic")
import importlib.util
spec = importlib.util.spec_from_file_location("cmp", "/repo/clean_synthetic/power_beta_bernoulli_compare.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Override fit_rasch_beta_torch to accept phi parameter
original_fit = mod.fit_rasch_beta_torch

# Test phi values
phis = [150, 175, 200, 225, 250, 275, 300, 350, 400]
results = {}

for phi in phis:
    # Modify the global call in run_single_simulation
    # The original already has phi=250.0 hardcoded
    # We need to patch the module
    import re
    
    # Read current file
    with open("/repo/clean_synthetic/power_beta_bernoulli_compare.py", "r") as f:
        code = f.read()
    
    # Replace phi value
    code_modified = code.replace("phi=250.0", "phi=%.1f" % phi)
    with open("/repo/clean_synthetic/power_beta_bernoulli_compare.py", "w") as f:
        f.write(code_modified)
    
    # Reload and run
    import importlib
    spec2 = importlib.util.spec_from_file_location("cmp2", "/repo/clean_synthetic/power_beta_bernoulli_compare.py")
    mod2 = importlib.util.module_from_spec(spec2)
    
    # Just run the main script - capture output
    import subprocess
    t0 = time.time()
    result = subprocess.run(
        ["python3", "/repo/clean_synthetic/power_beta_bernoulli_compare.py"],
        capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "HTTP_PROXY": "", "HTTPS_PROXY": ""}
    )
    
    # Extract RMSE
    for line in result.stdout.split("\n"):
        if "M=   2:" in line:
            parts = line.split("Beta RMSE=")
            if len(parts) > 1:
                rmse_str = parts[1].split("±")[0].strip()
                rmse = float(rmse_str)
                results[phi] = rmse
                print("phi=%.1f: RMSE=%.5f (%.1fs)" % (phi, rmse, time.time()-t0))
                break
    
    # Also extract correlation
    import numpy as np
    d = np.load("/repo/clean_synthetic/power_beta_bernoulli_data.npz")
    corr = d["corr_beta_mean"][0]
    print("         Corr=%.5f" % corr)

# Restore phi=250
with open("/repo/clean_synthetic/power_beta_bernoulli_compare.py", "r") as f:
    code = f.read()
code = code.replace("phi=%.1f" % phis[-1], "phi=250.0")
with open("/repo/clean_synthetic/power_beta_bernoulli_compare.py", "w") as f:
    f.write(code)

if results:
    best_phi = min(results, key=results.get)
    print("\nBest phi: %.1f (RMSE=%.5f)" % (best_phi, results[best_phi]))
