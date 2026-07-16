from __future__ import annotations

import glob
import json
import os
import tempfile

import pandas as pd
from pathlib import Path

from shapiq import InteractionValues
from shapiq_benchmark.metrics import get_all_metrics
import argparse


def _load_interaction_values(path, raw: bytes | None = None) -> InteractionValues:
    """Load InteractionValues, falling back to a null-byte-strip workaround.

    Some SLURM filesystems append trailing null bytes to JSON files, causing
    ``json.JSONDecodeError: Extra data``.  Try a direct load first; only if
    that raises a JSONDecodeError do we strip null bytes and retry via a
    temporary file — so non-SLURM users see zero overhead.

    If ``raw`` bytes are provided (e.g. already read for another purpose),
    they are used directly instead of re-reading the file.
    """
    if raw is None:
        try:
            return InteractionValues.load(str(path))
        except (json.JSONDecodeError, ValueError):
            with open(path, "rb") as f:
                raw = f.read().rstrip(b"\x00")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return InteractionValues.load(tmp_path)
    finally:
        os.unlink(tmp_path)

parser = argparse.ArgumentParser(
    description="Run benchmark approximations on explanation games."
)
parser.add_argument(
    "--config_approximators",
    type=int,
    default=37,
    help="Configuration ID for approximators: 40 (PAIRING=False, REPLACEMENT=True), "
    "39 (PAIRING=False, REPLACEMENT=False), "
    "38 (PAIRING=True, REPLACEMENT=True), "
    "37 (PAIRING=True, REPLACEMENT=False). Default is 37.",
)
parser.add_argument(
    "--model",
    type=str,
    default="1",
    help="Model ID to evaluate (default: 1). Options are integers corresponding to different trained models.",
)
parser.add_argument(
    "--index",
    type=str,
    default="SV",
    help="Index to compute (default: SV). Options: SV, SII",
)
parser.add_argument(
    "--order", type=int, default=1, help="Order of interaction index (default: 1)."
)
parser.add_argument(
    "--game_type",
    type=str,
    default="exhaustive",
    help="Type of game to compute (default: interventional). Options: exhaustive, pathdependent, interventional",
)

args = parser.parse_args()

if __name__ == "__main__":
    """
    This script loads the approximated Shapley values for all approximators, adds metadata information,
    and computes various metrics. The results are stored in results_benchmark.csv.

    IMPORTANT: Make sure to run the scripts experiments/approximation_baseline.py and/or
    experiments/approximation_pathdependent.py beforehand to generate the approximations and
    ground-truth values.
    """
    GAME_IDS = [
        "ResNet18w14Superpixel",
        "SentimentIMDBDistilBERT14",
        "ViT3by3Patches",
        "ViT4by4Patches",
        # "SOUM",
        "AdultCensusLocalXAI",
        "BikeSharingLocalXAI",
        "CaliforniaHousingLocalXAI",
        "CommunitiesAndCrimeLocalXAI",
        "Corrgroups60LocalXAI",
        "ForestFiresLocalXAI",
        "IndependentLinear60LocalXAI",
        "NHANESILocalXAI",
        "BreastCancerLocalXAI",
        "RealEstateLocalXAI",
    ]
    ORDER = args.order
    INDEX = args.index
    MODELS = [
        args.model
    ]
    RANDOM_STATE = 40
    ID_EXPLANATIONS = range(30)  # ids of test instances to explain
    GAME_TYPES = [
        args.game_type
    ]

    approximation_metrics = pd.DataFrame()

    results = []

    for game_type in GAME_TYPES:
        print(f"Loading {game_type}")
        for game in GAME_IDS:
            print(f"Loading {game}")
            for model in MODELS:
                game_id = game + "_" + model
                for id_explain in ID_EXPLANATIONS:
                    save_path_ground_truth = Path(
                        "ground_truth/"
                        + game_type
                        + "/"
                        + game_id
                        + "_"
                        + str(RANDOM_STATE)
                        + "_"
                        + str(id_explain)
                        + "_"
                        + INDEX
                        + "_"
                        + str(ORDER)
                        + "_exact_values.json"
                    )
                    try:
                        ground_truth = _load_interaction_values(save_path_ground_truth)
                    except Exception:
                        print("Path:", save_path_ground_truth)
                        print(f"Error loading ground truth for {game_type}/{game_id}")
                        continue
                    # File pattern with wildcard for budget
                    file_pattern = f"approximations/{game_type}/{game_id}_*_{id_explain}_*_{INDEX}_{ORDER}.json"
                    # print("Searching for files with pattern:", file_pattern)
                    # Get matching file paths
                    file_paths = glob.glob(file_pattern)
                    for file in file_paths:
                        # print(file.split("_"))
                        # raise NotImplementedError("Parsing filenames for ORDER > 1 not implemented.")
                        id_config_approximator = file.split("_")[2]
                        approximator = file.split("_")[4]
                        budget = int(file.split("_")[-3])
                        result = {
                            "game_type": game_type,
                            "game": game,
                            "model": model,
                            "game_id": game_id,
                            "id_explain": id_explain,
                            "n_players": ground_truth.n_players,
                            "budget": budget,
                            "budget_relative": round(
                                budget / (2**ground_truth.n_players), 6
                            ),
                            "approximator": approximator,
                            "used_budget": budget,
                            "iteration": 1,
                            "id_config_approximator": id_config_approximator,
                        }
                        # get runtime from saved approximated interaction values
                        with open(file, "rb") as f:
                            raw = f.read().rstrip(b"\x00")
                        runtime = json.loads(raw.decode("utf-8")).get("parameters", {})
                        approximated_values = _load_interaction_values(file, raw=raw)
                        # print(game_id, approximator)
                        # print(len(approximated_values.values), ground_truth.n_players**2/2)
                        # assert (
                        #     len(approximated_values.values) - 1
                        #     == ground_truth.n_players**2
                        # )
                        # compute metrics
                        all_metrics = get_all_metrics(ground_truth, approximated_values)
                        # print(all_metrics)
                        # `get_all_metrics` returns a sequence of Metric objects.
                        # Convert to a dict of unique keys -> numeric values so
                        # we can safely update the result mapping.
                        metrics_dict = {}
                        for m in all_metrics:
                            # metric_id is the main identifier (like 'Precision@k', 'MSE')
                            # Disambiguate using computed_k or order when present.
                            key = str(getattr(m, "metric_id", repr(m)))
                            if getattr(m, "computed_k", None) is not None:
                                if "@" not in key:
                                    key = f"{key}@{m.computed_k}"
                                else:
                                    key = f"{key[:-2]}@{m.computed_k}"
                            elif getattr(m, "order", None) is not None:
                                key = f"{key}_order{m.order}"
                            
                            # store numeric value if possible
                            try:
                                metrics_dict[key] = float(m.value)
                            except Exception:
                                metrics_dict[key] = m.value
                        # print(metrics_dict)
                        result.update(metrics_dict)
                        for key in ["total_runtime","evaluations","total_approximation","proxy_fit","extraction","adjustment","sampling","regression","total_Runtime_dealing"]:
                            if key in runtime["kwargs"]:
                                result[key] = runtime["kwargs"][key]
                        #result.update(runtime)
                        results.append(result)
    results_df = pd.DataFrame(results)

    results_df.to_csv(f"results_benchmark_{INDEX}_{ORDER}.csv")
