import json

from src.common_utils import find_latest_file
from src.plotting.plot_ablation import plot_ablation_study


def find_latest_ablation_json():
    patterns = [
        "results/regpepg_ablation_multiseed_*.json",
        "regpepg_ablation_*.json",
        "results/regpepg_ablation_*.json",
    ]

    latest_file = find_latest_file(patterns)
    if not latest_file:
        raise FileNotFoundError(
            "No ablation JSON files found. Please check the file location."
        )

    return latest_file


def plot_value_function_ablation(json_file):
    with open(json_file, "r") as f:
        results = json.load(f)

    regpepg_data = results["regpepg"]

    png_file, pdf_file = plot_ablation_study(
        results=regpepg_data,
        param_name="entropy_reg_lambda",
        param_label="λ",
        output_filename_prefix="regpepg_value_function_ablation",
        title_suffix="Entropy Regularization",
        ylabel="Value Function $V^\\pi$",
    )

    print(f"High-quality plot saved as: {png_file}")
    print(f"PDF version saved as: {pdf_file}")
    print(f"\nLoaded data from: {json_file}")

    return png_file, pdf_file


if __name__ == "__main__":
    json_file = find_latest_ablation_json()
    print(f"Found ablation results: {json_file}")

    png_file, pdf_file = plot_value_function_ablation(json_file)

    print(f"\nPlot generation completed!")
