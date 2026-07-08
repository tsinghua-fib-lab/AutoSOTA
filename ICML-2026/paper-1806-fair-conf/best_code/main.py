import sys
import hydra
import pprint
from omegaconf import OmegaConf
from pathlib import Path

from internal.process.run_conformal import run_conformal
from internal.dataset import DATASET_CLASS_MAP
from internal.validation.chart_heatmap import chart_heatmap
from internal.validation.run_llm_in_loop import run_llm_in_loop
from internal.process.compute_fairness_stats import run_compute_statistics


@hydra.main(config_path=".", config_name="custom_config", version_base="1.3")
def run(custom_config):
    print("=" * 50)
    print(" 📊  Conformal Prediction - Fairness Project Runner")
    print("=" * 50)
    print("Select a step to run:")
    print("[1] Run Conformal")
    print("[2] Run LLM-in-loop")
    print("[3] Chart and Heatmap")
    print("[4] Compute statistics results")

    choice = input("\nEnter your choice: ").strip()

    match choice:
        case "1":
            full_config = choose_dataset(
                custom_config, "\nChoosing Conformal Prediction Dataset...\n"
            )
            run_conformal(full_config)
        case "2":
            full_config = choose_dataset(
                custom_config, "\nChoosing Dataset For LLM-in-loop...\n"
            )
            run_llm_in_loop(full_config, None, None)
        case "3":
            full_config = choose_dataset(
                custom_config, "\nChoosing Dataset For Chart and Heatmap...\n"
            )
            chart_heatmap(full_config)
        case "4":
            full_config = choose_dataset(
                custom_config, "\nChoosing Dataset For computing comprehensive statistics...\n"
            )
            run_compute_statistics(full_config)
        case _:
            print("Exiting.")
            sys.exit(0)


def choose_dataset(custom_config, prompt_text: str = "Select dataset:"):
    print(prompt_text)
    datasets = list(DATASET_CLASS_MAP.keys())
    for i, ds in enumerate(datasets, start=1):
        print(f"[{i}] {ds}")

    choice = input("\nEnter dataset number: ").strip()
    try:
        dataset = datasets[int(choice) - 1]
        print(f"\n✅ Selected dataset: {dataset}")
        full_config = get_full_config(custom_config, dataset)
        return full_config
    except ValueError as e:
        print(f"Error: {e}")
        print("Exiting.")
        sys.exit(0)


def get_full_config(custom_config, dataset):
    dataset_cfg_path = Path(f"src/internal/conf/dataset/{dataset}.yaml")
    if not dataset_cfg_path.exists():
        print(dataset_cfg_path.resolve())
        raise FileNotFoundError(f"Dataset config not found for: {dataset}")
    dataset_cfg = OmegaConf.load(dataset_cfg_path)

    # 2. Load custom config file if exists
    base_config_path = Path("src/substantive/faircp/conf/config.yaml")
    base_config = OmegaConf.load(base_config_path)

    # 3. Merge configs: base < dataset < custom < CLI overrides
    full_cfg = OmegaConf.to_container(
        OmegaConf.merge(base_config, dataset_cfg, custom_config), resolve=True
    )

    # 4. Validate score_fn-specific hyperparams
    score_fn = full_cfg.get("score_fn")
    h_params = full_cfg.get(f"h_params_{score_fn}")
    if not h_params:
        raise ValueError(f"Missing hyperparams for score function '{score_fn}'")
    full_cfg["h_params_conformal"] = h_params

    # 5. Pretty print
    print(10 * "-" + " Final Config " + 10 * "-")
    pprint.PrettyPrinter(indent=4).pprint(full_cfg)

    return full_cfg


if __name__ == "__main__":
    run()
