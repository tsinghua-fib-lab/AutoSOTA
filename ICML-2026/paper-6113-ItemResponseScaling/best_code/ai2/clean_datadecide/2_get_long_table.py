import json
import math
from typing import Iterable, List, Optional
from pathlib import Path
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent / "DataDecide-eval-instances" / "models"
OUTPUT_PATH = Path(__file__).resolve().parent / "2_data_decide_long.parquet"
DRY_RUN_FILE_LIMIT = 3
DRY_RUN_ROW_LIMIT = 5


def get_prediction_files(base_dir: Path, limit_files: Optional[int] = None) -> List[Path]:
    files = sorted(base_dir.rglob("*-predictions.jsonl"))
    if limit_files:
        return files[:limit_files]
    return files


def find_max_choices(prediction_files: Iterable[Path]) -> int:
    max_choices = 0
    for predictions_path in tqdm(prediction_files, desc="Finding max choices"):
        with predictions_path.open() as handle:
            first_line = handle.readline()
        if not first_line:
            continue
        entry = json.loads(first_line)
        model_output = entry.get("model_output") or []
        if len(model_output) > max_choices:
            max_choices = len(model_output)
    return max_choices


def build_column_order(max_choices: int) -> List[str]:
    base_columns = [
        "doc_id",
        "native_id",
        "predicted_index_raw",
        "predicted_index_per_token",
        "predicted_index_per_char",
        "predicted_index_uncond",
        "correct_choice",
        "acc_raw",
        "acc_per_token",
        "acc_per_char",
        "acc_uncond",
    ]

    choice_columns: List[str] = []
    for idx in range(max_choices):
        for suffix in [
            "sum_logits",
            "sum_logits_uncond",
            "logits_per_token",
            "logits_per_char",
            "logits_per_byte",
        ]:
            choice_columns.append(f"choice_{idx}_{suffix}")

    meta_columns = [
        "model_data_mix",
        "model_size",
        "model_seed",
        "model_step",
        "bench_name",
        "question_id",
        "p_correct_choice",
    ]

    return base_columns + choice_columns + meta_columns


def process_predictions(
    base_dir: Path,
    prediction_files: Iterable[Path],
    max_choices: int,
    column_order: List[str],
    limit_rows: Optional[int] = None,
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []

    for predictions_path in tqdm(prediction_files, desc="Processing prediction files"):
        relative_parts = predictions_path.relative_to(base_dir).parts[:-1]

        model_data_mix = relative_parts[0] if relative_parts else None
        model_size = next(
            (part for part in relative_parts if part.endswith("M") or part.endswith("B")),
            None,
        )
        model_seed = next(
            (part.split("seed-")[-1] for part in relative_parts if part.startswith("seed-")),
            None,
        )
        model_step = next(
            (part.split("step-")[-1] for part in relative_parts if part.startswith("step-")),
            None,
        )
        bench_name = predictions_path.name.rsplit("-predictions.jsonl", 1)[0]

        rows = []
        with predictions_path.open() as handle:
            for idx_line, line in enumerate(handle):
                if limit_rows is not None and idx_line >= limit_rows:
                    break

                entry = json.loads(line)
                metrics = entry.get("metrics", {})
                model_output = entry.get("model_output") or []

                row = {
                    "doc_id": entry.get("doc_id"),
                    "native_id": entry.get("native_id"),
                    "predicted_index_raw": metrics.get("predicted_index_raw"),
                    "predicted_index_per_token": metrics.get("predicted_index_per_token"),
                    "predicted_index_per_char": metrics.get("predicted_index_per_char"),
                    "predicted_index_uncond": metrics.get("predicted_index_uncond"),
                    "correct_choice": metrics.get("correct_choice"),
                    "acc_raw": metrics.get("acc_raw"),
                    "acc_per_token": metrics.get("acc_per_token"),
                    "acc_per_char": metrics.get("acc_per_char"),
                    "acc_uncond": metrics.get("acc_uncond"),
                    "model_data_mix": model_data_mix,
                    "model_size": model_size,
                    "model_seed": model_seed,
                    "model_step": model_step,
                    "bench_name": bench_name,
                }

                for idx_choice in range(max_choices):
                    choice = model_output[idx_choice] if idx_choice < len(model_output) else {}
                    row[f"choice_{idx_choice}_sum_logits"] = choice.get("sum_logits")
                    row[f"choice_{idx_choice}_sum_logits_uncond"] = choice.get("sum_logits_uncond")
                    row[f"choice_{idx_choice}_logits_per_token"] = choice.get("logits_per_token")
                    row[f"choice_{idx_choice}_logits_per_char"] = choice.get("logits_per_char")
                    row[f"choice_{idx_choice}_logits_per_byte"] = choice.get("logits_per_byte")

                row["question_id"] = f"{bench_name}_{entry.get('doc_id')}"

                correct_choice = row.get("correct_choice")
                if isinstance(correct_choice, int) and 0 <= correct_choice < max_choices:
                    logits_per_char = row.get(f"choice_{correct_choice}_logits_per_char")
                    row["p_correct_choice"] = math.exp(logits_per_char) if logits_per_char is not None else None
                else:
                    row["p_correct_choice"] = None

                rows.append(row)

        if not rows:
            continue

        frame = pd.DataFrame(rows)
        frame = frame.reindex(columns=column_order)
        frames.append(frame)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_dir = BASE_DIR
    output_path = OUTPUT_PATH
    limit_files = DRY_RUN_FILE_LIMIT if args.dry_run else None
    limit_rows = DRY_RUN_ROW_LIMIT if args.dry_run else None

    prediction_files = get_prediction_files(base_dir, limit_files)

    max_choices = find_max_choices(prediction_files)
    print(f"Max choices found: {max_choices}")
    column_order = build_column_order(max_choices)

    frames = process_predictions(
        base_dir=base_dir,
        prediction_files=prediction_files,
        max_choices=max_choices,
        column_order=column_order,
        limit_rows=limit_rows,
    )

    combined = pd.concat(frames, ignore_index=True)
    # Force native_id to a consistent string dtype so PyArrow does not attempt int conversion.
    combined["native_id"] = combined["native_id"].astype("string")

    if args.dry_run:
        print("Dry run: showing first 5 rows and dtypes")
        print(combined.head())
        print(combined.dtypes)
        return

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, output_path)
    print(f"Wrote {len(combined):,} rows to {output_path}")


if __name__ == "__main__":
    main()
