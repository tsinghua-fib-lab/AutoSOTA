import subprocess
import sys
from itertools import product
from pathlib import Path

def build_cmd(base_script, params):
    cmd = [sys.executable, str(base_script)]
    for key, value in params.items():
        flag = f"--{key}"
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    return cmd


#sweep lists
seeds_list = [0]
lmbda_s = [1.5]#[0.1, 0.2, 0.3, 0.4, 0.5]#[0.1, 0.5, 1, 1.5, 2, 2.5, 3]
num_important_features_s = [30]#[5, 10, 20, 30, 40, 50,60,70,80,90,100]
epochs_s = [50]#[50, 100, 250, 500, 1000]
model_types = ["l2x"]

val_or_test = "test"
folder_for_pickle = f'ICML_experiments/tcga/{val_or_test}'
stop_on_failure = True

# Choose which hyperparameter to sweep:
# - "lmbda"
# - "num_important_features"
# - "epochs" (for lime)
sweep_mode = "num_important_features"

# Used when the selected sweep mode does not vary this parameter.
fixed_lmbda = 0.3
fixed_num_important_features = 3
fixed_epochs = None


def validate_sweep_mode_for_models():
    lmbda_models = {"hide_and_seek", "realx", "invase"}
    num_important_features_models = {"l2x"}
    epochs_models = {"lime"}

    for model_type in model_types:
        if model_type in lmbda_models and sweep_mode != "lmbda":
            raise ValueError(
                f"Invalid config: model_type='{model_type}' requires sweep_mode='lmbda'"
            )
        if model_type in num_important_features_models and sweep_mode != "num_important_features":
            raise ValueError(
                f"Invalid config: model_type='{model_type}' requires sweep_mode='num_important_features'"
            )
        if model_type in epochs_models and sweep_mode != "epochs":
            raise ValueError(
                f"Invalid config: model_type='{model_type}' requires sweep_mode='epochs'"
            )


def iter_experiments():
    if sweep_mode == "lmbda":
        for model_type, lmbda, seed in product(model_types, lmbda_s, seeds_list):
            yield {
                "model_type": model_type,
                "lmbda": lmbda,
                "num_important_features": fixed_num_important_features,
                "epochs": fixed_epochs,
                "seed": seed,
                "val_or_test": val_or_test,
                "folder-for-pickle": folder_for_pickle,
            }
    elif sweep_mode == "num_important_features":
        for model_type, num_important_features, seed in product(model_types, num_important_features_s, seeds_list):
            yield {
                "model_type": model_type,
                "lmbda": fixed_lmbda,
                "num_important_features": num_important_features,
                "epochs": fixed_epochs,
                "seed": seed,
                "val_or_test": val_or_test,
                "folder-for-pickle": folder_for_pickle,
            }
    elif sweep_mode == "epochs":
        for model_type, epochs, seed in product(model_types, epochs_s, seeds_list):
            yield {
                "model_type": model_type,
                "lmbda": fixed_lmbda,
                "num_important_features": fixed_num_important_features,
                "epochs": epochs,
                "seed": seed,
                "val_or_test": val_or_test,
                "folder-for-pickle": folder_for_pickle,
            }
    else:
        raise ValueError("sweep_mode must be 'lmbda', 'num_important_features', or 'epochs'")


if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent
    base_script = this_dir / "tcga_tools.py"

    validate_sweep_mode_for_models()

    experiments = list(iter_experiments())
    failures = []

    print(f"Running {len(experiments)} TCGA experiment(s)...")
    for idx, params in enumerate(experiments, start=1):
        cmd = build_cmd(base_script, params)
        print(f"[{idx}/{len(experiments)}] RUNNING:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as err:
            failures.append((params, err.returncode))
            print(f"FAILED (exit code {err.returncode}): {params}")
            if stop_on_failure:
                raise

    if failures:
        print("\nSummary of failed experiments:")
        for params, code in failures:
            print(f"- exit={code} params={params}")
    else:
        print("\nAll TCGA experiments completed successfully.")