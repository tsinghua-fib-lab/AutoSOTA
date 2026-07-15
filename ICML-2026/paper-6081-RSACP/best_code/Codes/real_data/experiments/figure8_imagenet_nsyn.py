from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_all_real_data import RAW, run_base_experiments, write_manifest
import plotting_paper_style


if __name__ == "__main__":
    if not (RAW / "figure8_imagenet_nsyn_raw.csv").exists():
        run_base_experiments()
    plotting_paper_style.draw_sensitivity(
        "figure8_imagenet_nsyn_raw.csv",
        "n_ref",
        "Synthetic data size (n_cal_maj)",
        "figure8_imagenet_nsyn",
    )
    write_manifest()
