import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import pickle
rng = np.random.default_rng(0)

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT.parent))
from utils import calculate_flops

BUDGET = 50
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=BASE_DIR)
args = parser.parse_args()

data_root = args.data_root
data_root.mkdir(parents=True, exist_ok=True)

input_binary_1pl = data_root / "5_binary_matrix_cated.parquet"
input_beta_1pl = data_root / "5_prob_matrix_cated.parquet"
input_binary_2pl = data_root / "5_binary_matrix_cated_2pl.parquet"
input_beta_2pl = data_root / "5_prob_matrix_cated_2pl.parquet"
input_bpb = data_root / "3_bpb_matrix.parquet"
long_output_path = data_root / "6_long.parquet"
difficulty_output_path = data_root / "6_difficulty.pkl"

binary_1pl_df = pd.read_parquet(input_binary_1pl)
binary_1pl_test_df = binary_1pl_df[binary_1pl_df.index.get_level_values("model_split") == "test"].copy()
beta_1pl_df = pd.read_parquet(input_beta_1pl)
beta_1pl_test_df = beta_1pl_df[beta_1pl_df.index.get_level_values("model_split") == "test"].copy()
binary_2pl_df = pd.read_parquet(input_binary_2pl)
binary_2pl_test_df = binary_2pl_df[binary_2pl_df.index.get_level_values("model_split") == "test"].copy()
beta_2pl_df = pd.read_parquet(input_beta_2pl)
beta_2pl_test_df = beta_2pl_df[beta_2pl_df.index.get_level_values("model_split") == "test"].copy()
bpb_df = pd.read_parquet(input_bpb)
bpb_test_df = bpb_df[bpb_df.index.get_level_values("model_split") == "test"].copy()

question_index_names = ["bench_name", "doc_id"]
binary_1pl_question_index = binary_1pl_test_df.columns.to_frame(index=False)[question_index_names]
beta_1pl_question_index = beta_1pl_test_df.columns.to_frame(index=False)[question_index_names]
binary_2pl_question_index = binary_2pl_test_df.columns.to_frame(index=False)[question_index_names]
beta_2pl_question_index = beta_2pl_test_df.columns.to_frame(index=False)[question_index_names]
bpb_question_index = bpb_test_df.columns.to_frame(index=False)[question_index_names]
assert binary_1pl_question_index.equals(beta_1pl_question_index) \
    and binary_1pl_question_index.equals(binary_2pl_question_index) \
    and binary_1pl_question_index.equals(beta_2pl_question_index) \
    and binary_1pl_question_index.equals(bpb_question_index)

binary_1pl_base_index = binary_1pl_test_df.index.to_frame(index=False)
beta_1pl_base_index = beta_1pl_test_df.index.to_frame(index=False)
binary_2pl_base_index = binary_2pl_test_df.index.to_frame(index=False)
beta_2pl_base_index = beta_2pl_test_df.index.to_frame(index=False)
bpb_base_index = bpb_test_df.index.to_frame(index=False)

model_index_names = ["model_data_mix", "model_size", "model_step"]
assert binary_1pl_base_index[model_index_names].equals(beta_1pl_base_index[model_index_names]) \
    and binary_1pl_base_index[model_index_names].equals(binary_2pl_base_index[model_index_names]) \
    and binary_1pl_base_index[model_index_names].equals(beta_2pl_base_index[model_index_names]) \
    and binary_1pl_base_index[model_index_names].equals(bpb_base_index[model_index_names])

model_index = binary_1pl_base_index[model_index_names].copy()
model_index["model_step"] = model_index["model_step"].astype(int)
max_model_step = (
    model_index.groupby(["model_data_mix", "model_size"], as_index=False)["model_step"]
    .max()
    .rename(columns={"model_step": "max_model_step"})
)
model_index = model_index.merge(max_model_step, on=["model_data_mix", "model_size"], how="left")
final_step_mask = model_index["model_step"] == model_index["max_model_step"]
model_index["FLOP"] = [np.nan] * len(model_index)
model_index.loc[final_step_mask, "FLOP"] = [
    calculate_flops(model_size, step)
    for model_size, step in zip(
        model_index.loc[final_step_mask, "model_size"], model_index.loc[final_step_mask, "model_step"]
    )
]
assert model_index[model_index_names].equals(
    binary_1pl_base_index[model_index_names].assign(model_step=binary_1pl_base_index["model_step"].astype(int))
)

long_df = model_index[["model_data_mix", "model_size", "model_step", "FLOP"]].copy()
ability_index_names = [c for c in binary_1pl_base_index.columns if c.startswith("ability_")]
for ability_index_name in ability_index_names:
    long_df[f"{ability_index_name}_binary_1pl"] = binary_1pl_base_index[ability_index_name]
    long_df[f"{ability_index_name}_binary_2pl"] = binary_2pl_base_index[ability_index_name]
    long_df[f"{ability_index_name}_beta_1pl"] = beta_1pl_base_index[ability_index_name]
    long_df[f"{ability_index_name}_beta_2pl"] = beta_2pl_base_index[ability_index_name]

bench_names = binary_1pl_test_df.columns.get_level_values("bench_name").map(
    lambda b: "mmlu" if b.startswith("mmlu") else b
)
unique_bench_names = sorted(bench_names.unique())
binary_ys = binary_1pl_test_df.to_numpy(dtype=np.float32)
beta_ys = beta_1pl_test_df.to_numpy(dtype=np.float32)
bpb_ys = bpb_test_df.to_numpy(dtype=np.float32)
binary_1pl_zs = binary_1pl_test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
beta_1pl_zs = beta_1pl_test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
binary_2pl_zs = binary_2pl_test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
beta_2pl_zs = beta_2pl_test_df.columns.get_level_values("difficulty").to_numpy(dtype=np.float32)
binary_2pl_as = binary_2pl_test_df.columns.get_level_values("discrimination").to_numpy(dtype=np.float32)
beta_2pl_as = beta_2pl_test_df.columns.get_level_values("discrimination").to_numpy(dtype=np.float32)
difficulty_dict = {}
for bench in unique_bench_names:
    bench_mask = bench_names == bench
    difficulty_dict[bench] = {
        "binary_difficulty_1pl": binary_1pl_zs[bench_mask].tolist(),
        "beta_difficulty_1pl": beta_1pl_zs[bench_mask].tolist(),
        "binary_difficulty_2pl": binary_2pl_zs[bench_mask].tolist(),
        "beta_difficulty_2pl": beta_2pl_zs[bench_mask].tolist(),
        "binary_discrimination_2pl": binary_2pl_as[bench_mask].tolist(),
        "beta_discrimination_2pl": beta_2pl_as[bench_mask].tolist(),
    }
with open(difficulty_output_path, "wb") as f:
    pickle.dump(difficulty_dict, f)

for bench in unique_bench_names:
    bench_mask = bench_names == bench
    bench_binary_ys = binary_ys[:, bench_mask]
    bench_beta_ys = beta_ys[:, bench_mask]
    bench_bpb_ys = bpb_ys[:, bench_mask]
    sampled_idx = rng.choice(bench_binary_ys.shape[-1], size=BUDGET, replace=False)

    long_df[f"acc_full_{bench}"] = np.nanmean(bench_binary_ys, axis=1)
    long_df[f"acc_sub_{bench}"] = np.nanmean(bench_binary_ys[:, sampled_idx], axis=1)
    long_df[f"p_correct_choice_full_{bench}"] = np.nanmean(bench_beta_ys, axis=1)
    long_df[f"p_correct_choice_sub_{bench}"] = np.nanmean(bench_beta_ys[:, sampled_idx], axis=1)
    long_df[f"correct_bpb_full_{bench}"] = np.nanmean(bench_bpb_ys, axis=1)
    long_df[f"correct_bpb_sub_{bench}"] = np.nanmean(bench_bpb_ys[:, sampled_idx], axis=1)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print("long_df shape:", long_df.shape)
print("long_df head:\n", long_df.head(1))
print("long_df dtypes:\n", long_df.dtypes)
na_pct_all = (long_df.isna().mean() * 100).sort_values(ascending=False)
print("NaN percentage per column:")
for col, pct in na_pct_all.items():
    print(f"  {col}: {pct:.2f}%")
correct_bpb_cols = [c for c in long_df.columns if c.startswith("correct_bpb_")]
correct_bpb_nan_mask = long_df[correct_bpb_cols].isna().any(axis=1)
print("Rows with NaN in correct_bpb_* columns (model_data_mix, model_size, model_step, FLOP):")
print(long_df.loc[correct_bpb_nan_mask, ["model_data_mix", "model_size", "model_step", "FLOP"]])
long_df.to_parquet(long_output_path)
