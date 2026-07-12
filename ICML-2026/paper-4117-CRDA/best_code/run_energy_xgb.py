import sys
import os
sys.path.insert(0, "/repo")

from src.utils.config import Config
from src.utils.logger import Logger
from src.experiment import Experiment
from time import perf_counter

config = Config(
    baseline="xgboost",
    dataset_path="/repo/data/EnergyEfficiency.csv",
    results_dir="/repo/experiments/EnergyEfficiency",
    sample_sizes=[765],
    save_params=True,
    hyperparam_tune=True,
    ignore_filter=True,
)

print("=" * 80)
print("CRDA EXPERIMENT: Energy Efficiency, XGBoost, sample_size=765")
print("=" * 80)
print(f"Config: baseline={config.baseline}")
print(f"Config: test_size={config.test_size}")
print(f"Config: num_seeds={config.num_seeds}")
print(f"Config: hyperparam_tune={config.hyperparam_tune}")
print(f"Config: ignore_filter={config.ignore_filter}")
print(f"Config: experiment_dir={config.experiment_dir}")
print("=" * 80)

logger = Logger(log_to_file=False, log_to_console=True)
experiment = Experiment(config, logger)
start_time = perf_counter()
results = experiment.run()
end_time = perf_counter()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

# Print results
if results is not None:
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(results.to_string())
    
    # Find the delta_mse for EnergyEfficiency_sample_765
    ee_rows = results[results["dataset"] == "EnergyEfficiency_sample_765"]
    print("\n" + "=" * 80)
    print("ENERGY EFFICIENCY (sample_765) RESULTS:")
    print("=" * 80)
    for _, row in ee_rows.iterrows():
        print(f"  {row[metric]}: mean={row[mean]:.6f}, std_err={row[std]:.6f}")
    
    # Save to a specific location
    results.to_csv("/repo/energy_efficiency_xgb_results.csv", index=False)
    print("\nResults saved to /repo/energy_efficiency_xgb_results.csv")
else:
    print("ERROR: No results produced!")

