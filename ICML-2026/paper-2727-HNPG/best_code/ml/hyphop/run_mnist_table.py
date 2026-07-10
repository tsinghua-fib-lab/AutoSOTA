import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_mnist import main as run_mnist_experiment


def run_experiment(model_name, hidden_dim, trial_seed, epochs=14, device="auto"):
    test_args = [
        "test_mnist.py",
        "--model", model_name,
        "--hidden-dim", str(hidden_dim),
        "--epochs", str(epochs),
        "--lr", "0.001",
        "--gamma", "0.96",
        "--seed", str(trial_seed),
        "--log-interval", "100",
        "--device", device,
    ]

    with patch.object(sys, 'argv', test_args):
        return run_mnist_experiment()


def main():
    device = "auto"

    model_names = ["kf_attention", "hf_attention", "ein_attention"]
    hidden_dims = [4, 8, 32]
    num_trials = 5
    epochs = 14

    result_rows = []

    for h_dim in hidden_dims:
        for model_name in model_names:
            trial_scores = []

            print("\n" + "#" * 70)
            print(f"Evaluating Model={model_name}, hidden_dim={h_dim}")
            print("#" * 70)

            for trial_idx in range(num_trials):
                trial_seed = 42 + trial_idx
                print(f"Trial {trial_idx + 1}/{num_trials} (seed={trial_seed})")

                acc = run_experiment(
                    model_name=model_name,
                    hidden_dim=h_dim,
                    trial_seed=trial_seed,
                    epochs=epochs,
                    device=device,
                )
                trial_scores.append(acc)
                print(f"Final Accuracy: {acc:.2f}%")

            mean_acc = np.mean(trial_scores)
            std_acc = np.std(trial_scores)

            result_rows.append({
                "Model": model_name,
                "Hidden Dim": h_dim,
                "Mean Accuracy": f"{mean_acc:.2f}%",
                "Std Dev": f"{std_acc:.2f}%"
            })

    df = pd.DataFrame(result_rows)
    output_dir = os.path.join("results", "mnist")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mnist_benchmark_results.csv")
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("FINAL MNIST ATTENTION BENCHMARK")
    print("=" * 40)
    print(f"Saved to: {os.path.abspath(output_file)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
