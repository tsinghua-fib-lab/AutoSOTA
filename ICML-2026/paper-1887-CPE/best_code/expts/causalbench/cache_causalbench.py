# cache_causalbench.py
from __future__ import annotations

import os
import subprocess
import argparse
from pathlib import Path


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="weissmann_k562")
    ap.add_argument("--training_regime", type=str, default="observational")
    ap.add_argument("--data_directory", type=str, default="data_causalbench")
    ap.add_argument("--output_directory", type=str, default="cb_tmp_output")
    ap.add_argument("--export_npz", type=str, default="data_causalbench/exports/weissmann_k562_observational.npz")
    ap.add_argument("--subset_data", type=float, default=1.0)
    ap.add_argument("--model_seed", type=int, default=0)
    ap.add_argument("--max_path_length", type=int, default=-1)
    ap.add_argument("--omission_estimation_size", type=int, default=500)
    ap.add_argument("--do_filter", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_directory)
    out_dir = Path(args.output_directory)
    export_npz = Path(args.export_npz)

    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_npz.parent.mkdir(parents=True, exist_ok=True)

    # Tell the export model where to write:
    env = os.environ.copy()
    env["CAUSALBENCH_EXPORT_PATH"] = str(export_npz)

    cmd = [
        "causalbench_run",
        "--dataset_name", args.dataset_name,
        "--output_directory", str(out_dir),
        "--data_directory", str(data_dir),
        "--training_regime", args.training_regime,
        "--model_name", "custom",
        "--inference_function_file_path", "expts/causalbench/causalbench_export_model.py",
        "--subset_data", str(args.subset_data),
        "--model_seed", str(args.model_seed),
        "--max_path_length", str(args.max_path_length),
        "--omission_estimation_size", str(args.omission_estimation_size),
    ]
    if args.do_filter:
        cmd.append("--do_filter")

    print("[cache_causalbench] running:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"[cache_causalbench] done. Exported dataset to: {export_npz}")
    print(f"[cache_causalbench] processed dataset cache should be under: {data_dir} (per causalbench docs).")


if __name__ == "__main__":
    run()


