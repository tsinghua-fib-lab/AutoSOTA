import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_mil import main as run_mil_experiment


def run_experiment(dataset_name, model_name, trial_seed, epochs=100, batch_size=16, lr=0.001, gamma=0.96):
    test_args = [
        "test_mil.py",
        "--dataset", dataset_name,
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--gamma", str(gamma),
        "--seed", str(trial_seed),
    ]

    with patch.object(sys, 'argv', test_args):
        return run_mil_experiment()


def main():
    dataset_names = ['tiger', 'fox', 'elephant']
    model_names = ['kf_pooling', 'hf_pooling', 'ein_pooling']
    num_trials = 5
    epochs = 100

    result_rows = []

    for dataset_name in dataset_names:
        for model_name in model_names:
            trial_scores = []

            print("\n" + "#" * 70)
            print(f"Evaluating Model={model_name}, Dataset={dataset_name}")
            print("#" * 70)

            for trial_idx in range(num_trials):
                trial_seed = 42 + trial_idx
                print(f"Trial {trial_idx + 1}/{num_trials} (seed={trial_seed})")

                try:
                    mean_auc = run_experiment(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        trial_seed=trial_seed,
                        epochs=epochs,
                        batch_size=16,
                        lr=0.001,
                        gamma=0.96,
                    )
                    trial_scores.append(mean_auc)
                    print(f"Final AUC: {mean_auc:.4f}")
                except Exception as e:
                    print(f"Error in Trial {trial_idx + 1} for {model_name} on {dataset_name}: {e}")

            if trial_scores:
                mean_auc = np.mean(trial_scores)
                std_auc = np.std(trial_scores)

                result_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Mean_AUC": round(mean_auc, 4),
                    "Std_Dev": round(std_auc, 4),
                    "Trials": len(trial_scores)
                })

    df = pd.DataFrame(result_rows)
    output_dir = os.path.join("results", "mil")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mil_benchmark_results.csv")
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("FINAL MIL BENCHMARK")
    print("=" * 40)
    print(f"Saved to: {os.path.abspath(output_file)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()