import pandas as pd

# === INPUT FILE ===
path = "invisible_char_token_counts.csv"  
df = pd.read_csv(path)

# Identify columns
char_cols = [c for c in df.columns if c.endswith("_char_tokens")]
diff_cols = [c for c in df.columns if c.endswith("_diff")]

# Map models (so we can align char/diff)
models = [c.replace("_char_tokens", "") for c in char_cols]

# === PER-ROW SUMMARY ===
# Count zeros in _char_tokens => nb of tokenizer that do not recongize the character
row_char_zero = (df[char_cols] == 0).sum(axis=1)

# Count each unique value in _diff 
diff_values = pd.concat([df[col] for col in diff_cols])
unique_diff_values = sorted(pd.to_numeric(diff_values, errors="coerce").dropna().unique())

row_summaries = pd.DataFrame({
    "Unicode": df["Unicode"],
    "Character_Name": df["Character_Name"],
    "Nb_Unrecognized": row_char_zero
})

# For each diff value, count only where _char_tokens != 0
for val in unique_diff_values:
    count_per_row = []
    for i in range(len(df)):
        count = 0
        for c_char, c_diff in zip(char_cols, diff_cols):
            if df.at[i, c_char] != 0 and df.at[i, c_diff] == val:
                count += 1
        count_per_row.append(count)
    row_summaries[f"toke_diff_{int(val)}"] = count_per_row

# Save per-row summary
row_summary_path = "aggregate_char.csv"
row_summaries.to_csv(row_summary_path, index=False)
print(f"Per-row summary saved to: {row_summary_path}")

# === PER-COLUMN SUMMARY ===
column_summaries = []

# Get all possible diff values to ensure consistent columns
all_diff_values = sorted(pd.to_numeric(diff_values, errors="coerce").dropna().unique())

for model, char_col, diff_col in zip(models, char_cols, diff_cols):
    model_df = df[[char_col, diff_col]].copy()

    # Unrecognized (char_tokens == 0)
    unrecog = (model_df[char_col] == 0).sum()

    # For diff counts — ignore rows where char==0
    valid_mask = model_df[char_col] != 0
    diff_counts = model_df.loc[valid_mask, diff_col].value_counts().sort_index()

    # Build a result row
    row = {"model": model, "unrecognized": unrecog}
    for val in all_diff_values:
        row[str(int(val))] = diff_counts.get(val, 0)

    column_summaries.append(row)
col_summary_df = pd.DataFrame(column_summaries)
col_summary_df = col_summary_df.fillna(0).astype({k: int for k in col_summary_df.columns if k != "model"})


# Save per-column summary
col_summary_path = "aggregate_tokenizer.csv"
col_summary_df.to_csv(col_summary_path, index=False)
print(f" Per-column summary saved to: {col_summary_path}")

