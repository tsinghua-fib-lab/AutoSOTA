from pathlib import Path
import random

import pandas as pd
from huggingface_hub import snapshot_download


REPO_ID = "yuhengtu/irsl_datadecide"
SNAPSHOT_DIR = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = SNAPSHOT_DIR / "2_data_decide_long.parquet"
OUTPUT_BINARY = BASE_DIR / "3_binary_matrix.parquet"
OUTPUT_PROB = BASE_DIR / "3_prob_matrix.parquet"
PROB_THRESHOLD = 0.15

### 1. clean
# keep only rows with model_seed == 2
frame = pd.read_parquet(DATA_PATH)
frame = frame[frame["model_seed"].astype(str) == "2"].copy()

# Tag model_is_final_step for each (model_data_mix, model_size)
group_cols = ["model_data_mix", "model_size"]
frame["model_step_numeric"] = pd.to_numeric(frame["model_step"], errors="coerce")
max_steps = (
    frame[group_cols + ["model_step_numeric"]]
        .groupby(group_cols, as_index=False)
        .max()
        .rename(columns={"model_step_numeric": "max_model_step"})
)
frame = frame.merge(max_steps, on=group_cols, how="left")
frame["model_is_final_step"] = frame["model_step_numeric"] == frame["max_model_step"]
# for testing, print 100 rows of the columns: "model_data_mix", "model_size", "model_step", "model_is_final_step"
print(
    frame[["model_data_mix", "model_size", "model_step", "model_is_final_step"]]
    .head(100)
    .to_string(index=False)
)

# Split model_data_mix values into deterministic train/test buckets
mixes = frame["model_data_mix"].dropna().unique().tolist()
assert len(mixes) == 25, f"Expected 25 unique model_data_mix entries, found {len(mixes)}"
random.Random(0).shuffle(mixes)
train_mixes = set(mixes[:5])
test_mixes = set(mixes[5:])
def assign_split(mix: str) -> str:
    if mix in train_mixes:
        return "train"
    elif mix in test_mixes:
        return "test"
frame["model_split"] = frame["model_data_mix"].map(assign_split)

### 2. pivot
index_cols = ["model_data_mix", "model_size", "model_step", "model_split", "model_is_final_step"]
binary_matrix = frame.pivot_table(
    index=index_cols,
    columns="question_id",
    values="acc_per_char",
    aggfunc="first",
)

prob_matrix = frame.pivot_table(
    index=index_cols,
    columns="question_id",
    values="p_correct_choice",
    aggfunc="first",
)

### 3. filter out low score test takers
prob_row_mean = prob_matrix.mean(axis=1, skipna=True)
rows_to_drop = set(prob_row_mean[prob_row_mean < PROB_THRESHOLD].index)
binary_matrix = binary_matrix.drop(index=rows_to_drop)
prob_matrix = prob_matrix.drop(index=rows_to_drop)
print(f"Dropped {len(rows_to_drop)} low-average rows; remaining rows: {len(binary_matrix)}")

for name, matrix in [("binary_matrix", binary_matrix), ("prob_matrix", prob_matrix)]:
    total_cells = matrix.size
    nan_count = int(matrix.isna().sum().sum())
    nan_pct = (nan_count / total_cells * 100) if total_cells else 0.0
    print(f"{name} shape: {matrix.shape}")
    print(f"{name} NaN: {nan_count} cells ({nan_pct:.2f}%)")

binary_matrix.to_parquet(OUTPUT_BINARY)
prob_matrix.to_parquet(OUTPUT_PROB)
