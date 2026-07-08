import json

from src.common_utils import find_latest_file
from src.plotting.plot_ablation import plot_ablation_study


def find_latest_ablation_json():
    patterns = [
        "results/pepg_eta_ablation_multiseed_*.json",
        "pepg_eta_ablation_*.json",
        "results/pepg_eta_ablation_*.json",
    ]

    latest_file = find_latest_file(patterns)
    if not latest_file:
        raise FileNotFoundError(
            "No learning rate ablation JSON files found. Please check the file location."
        )

    return latest_file


def plot_value_function_ablation(json_file):
    with open(json_file, "r") as f:
        results = json.load(f)

    pepg_data = results["pepg"]

    png_file, pdf_file = plot_ablation_study(
        results=pepg_data,
        param_name="eta",
        param_label="η",
        output_filename_prefix="pepg_value_function_lr_ablation",
        title_suffix="Learning Rate",
        ylabel="$v^\\pi_\\pi$",
    )

    print(f"High-quality plot saved as: {png_file}")
    print(f"PDF version saved as: {pdf_file}")
    print(f"\nLoaded data from: {json_file}")

    return png_file, pdf_file


if __name__ == "__main__":
    try:
        json_file = find_latest_ablation_json()
        print(f"Found ablation results: {json_file}")

        png_file, pdf_file = plot_value_function_ablation(json_file)

        print(f"\nPlot generation completed!")
        print(f"PNG (300 DPI): {png_file}")
        print(f"PDF (300 DPI): {pdf_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure your ablation JSON file is in one of these locations:")
        print("- results/pepg_eta_ablation_multiseed_*.json")
        print("- pepg_eta_ablation_*.json")
        print("- results/pepg_eta_ablation_*.json")
    except Exception as e:
        print(f"Error occurred: {e}")
