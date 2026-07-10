import sys; sys.path.insert(0, "/repo")
from experiment_er import run_single_trial
import time
print("Testing fixed script...")
for seed in [1, 2, 3]:
    t0 = time.time()
    result = run_single_trial(seed)
    elapsed = time.time() - t0
    print(f"Trial {seed} ({elapsed:.1f}s): true_edges={result['n_true_edges']} truth_time={result['truth_time']:.2f}s")
    print(f"  FCI: CI={result['fci']['CI_num']:.0f} t={result['fci']['runtime_sec']:.2f}s P={result['fci']['precision']:.3f} R={result['fci']['recall']:.3f} F1={result['fci']['f1']:.3f}")
    print(f"  DiCoLA: CI={result['dicola']['CI_num']:.0f} t={result['dicola']['runtime_sec']:.2f}s P={result['dicola']['precision']:.3f} R={result['dicola']['recall']:.3f} F1={result['dicola']['f1']:.3f}")
print("DONE")
