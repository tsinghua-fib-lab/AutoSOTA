import numpy as np
np.random.seed(711)
n_repetitions = 100
for seed in np.random.randint(1, np.iinfo(np.int32).max, n_repetitions):
    print(f"python run_estimator_benchmark.py --seed {seed} --knot_heuristic sqrt")
    print(f"python run_estimator_benchmark.py --seed {seed} --knot_heuristic lin")
