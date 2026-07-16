import json
from typing import Iterable, List, Optional
from pathlib import Path
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent / "DataDecide-eval-instances" / "models"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "2_datadecide_long.parquet"
MAX_CHOICES = 5
DRY_RUN_FILE_LIMIT = 3
DRY_RUN_ROW_LIMIT = 5

def get_prediction_files(base_dir: Path, limit_files: Optional[int] = None) -> List[Path]:
    files = sorted(base_dir.rglob("*-predictions.jsonl"))
    if limit_files:
        return files[:limit_files]
    return files

def build_column_order(max_choices: int) -> List[str]:
    native_columns = ["doc_id", "correct_choice", "acc_per_char"]
    choice_columns = [
        f"choice_{idx}_{suffix}"
        for idx in range(max_choices)
        for suffix in ["logits_per_char", "logits_per_byte"]
    ]
    new_columns = ["model_data_mix", "model_size", "model_seed", "model_step", "bench_name"]
    return native_columns + choice_columns + new_columns

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
            None
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
                    "correct_choice": metrics.get("correct_choice"),
                    "acc_per_char": metrics.get("acc_per_char"),
                    "model_data_mix": model_data_mix,
                    "model_size": model_size,
                    "model_seed": model_seed,
                    "model_step": model_step,
                    "bench_name": bench_name,
                }
                for idx_choice in range(max_choices):
                    choice = model_output[idx_choice] if idx_choice < len(model_output) else {}
                    row[f"choice_{idx_choice}_logits_per_char"] = choice.get("logits_per_char")
                    row[f"choice_{idx_choice}_logits_per_byte"] = choice.get("logits_per_byte")

                rows.append(row)

        frame = pd.DataFrame(rows)
        frame = frame.reindex(columns=column_order)
        frames.append(frame)

    return frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    limit_files = DRY_RUN_FILE_LIMIT if args.dry_run else None
    limit_rows = DRY_RUN_ROW_LIMIT if args.dry_run else None
    prediction_files = get_prediction_files(BASE_DIR, limit_files)
    column_order = build_column_order(MAX_CHOICES)

    frames = process_predictions(
        base_dir=BASE_DIR,
        prediction_files=prediction_files,
        max_choices=MAX_CHOICES,
        column_order=column_order,
        limit_rows=limit_rows,
    )
    combined = pd.concat(frames, ignore_index=True)

    pd.set_option("display.max_columns", None)
    print(combined.head())
    print(combined.dtypes)
    
    if not args.dry_run:
        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, OUTPUT_PATH)
