from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_all_real_data import RAW, run_base_experiments, write_manifest
import plotting_boxplots


if __name__ == "__main__":
    if not (RAW / "figure4_imagenet_main_raw.csv").exists():
        run_base_experiments()
    plotting_boxplots.restyle_figure4()
    write_manifest()
