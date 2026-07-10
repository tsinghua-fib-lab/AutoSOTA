import sys, time, numpy as np
sys.path.insert(0, ".")
from examples.safe_PCE import *

for eps in [0.05, 0.02, 0.01, 0.005]:
    np.random.seed(42)
    random.seed(42)
    print(f"\n=== epsilon={eps} ===")
    t0 = time.time()
    U, size = pretrain_stage(0.1, eps)
    print(f"U size={size}, noises={[float(x) for x in U]}, time={time.time()-t0:.1f}s")
