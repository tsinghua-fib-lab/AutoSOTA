import json
import sys
from datetime import datetime
from pathlib import Path

from src.common_utils import find_latest_file
from src.plotting.plot_comparison import (
    plot_algorithm_comparison,
    plot_individual_algorithm,
)


def find_latest_comparison_file():
    patterns = [
        "data/outputs_comparison_*.json",
        "data/outputs_comparison.json",
    ]

    latest_file = find_latest_file(patterns)
    if latest_file:
        return latest_file

    try:
        with open("data/latest_output.txt", "r") as f:
            latest_filename = f.read().strip()
        if "comparison" in latest_filename and Path(latest_filename).exists():
            return latest_filename
    except FileNotFoundError:
        pass

    return None


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = find_latest_comparison_file()
        if not filename:
            print("Error: No comparison JSON file found!")
            print("Tried:")
            print("  - data/latest_output.txt reference")
            print("  - data/outputs_comparison_*.json pattern")
            print("  - data/outputs_comparison.json fallback")
            print("\nUsage: python plot_refactored.py [json_file]")
            sys.exit(1)
        print(f"Auto-detected latest comparison file: {filename}")

    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        sys.exit(1)

    output_dir = f"figures/comparison_plots_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"All plots will be saved to: {output_dir}/")

    pg_data = data.get("pg", [])
    pepg_data = data.get("pepg", [])
    mdrr_data = data.get("mdrr", [])
    reg_pepg_data = data.get("reg_pepg", [])

    print(f"Loaded data:")
    print(f"  PG: {len(pg_data)} configurations")
    print(f"  PePG: {len(pepg_data)} configurations")
    print(f"  MDRR: {len(mdrr_data)} configurations")
    print(f"  Reg PePG: {len(reg_pepg_data)} configurations")

    if pg_data or pepg_data or mdrr_data or reg_pepg_data:
        print("\nGenerating algorithm comparison plot...")
        plot_algorithm_comparison(
            pg_data, pepg_data, mdrr_data, reg_pepg_data, output_dir
        )

    if pg_data:
        print("\nGenerating PG individual plot...")
        plot_individual_algorithm(pg_data, "RPO FS", output_dir)

    if pepg_data:
        print("\nGenerating PePG individual plot...")
        plot_individual_algorithm(pepg_data, "PePG", output_dir)

    if mdrr_data:
        print("\nGenerating MDRR individual plot...")
        plot_individual_algorithm(mdrr_data, "MDRR", output_dir)

    if reg_pepg_data:
        print("\nGenerating Reg PePG individual plot...")
        plot_individual_algorithm(reg_pepg_data, "Reg PePG", output_dir)


if __name__ == "__main__":
    main()
