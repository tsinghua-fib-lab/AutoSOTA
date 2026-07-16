import pandas as pd
import matplotlib.pyplot as plt

# === INPUT FILES ===
files = [
    ("invisible_char_token_counts.csv", "Invisible"),
    ("emoji_tokenization_analysis.csv", "Emoji"),
]

# Initialize results dictionary
results = {}

for path, label in files:
    print(f"\nProcessing {label} ({path})...")
    df = pd.read_csv(path)

    # Columns by type, excluding those containing 'bert' => bert does not in fact recognize emojis, all of them are translated into the same token
    char_cols = [c for c in df.columns if c.endswith("_char_tokens") and "bert" not in c.lower()]
    diff_cols = [c for c in df.columns if c.endswith("_diff") and "bert" not in c.lower()]

    # --- Collect all diff values ---
    all_diff_values = []
    for col in diff_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        all_diff_values.extend(vals.tolist())

    diff_series = pd.Series(all_diff_values)
    diff_freq = diff_series.value_counts().sort_index()

    # --- Count zero values from _char_tokens ---
    zero_char_count = (df[char_cols] == 0).sum().sum()
    no_token_count = zero_char_count

    # --- Add pseudo-bin for no-token as -1 ---
    diff_freq[-1] = no_token_count

    # --- Normalize counts to percentages ---
    total = diff_freq.sum()
    diff_freq_normalized = diff_freq / total * 100

    # Store normalized results
    results[label] = diff_freq_normalized



# === Align both datasets by diff value ===
all_keys = sorted(set(results["Invisible"].index).union(results["Emoji"].index))
inv_values = [results["Invisible"].get(k, 0) for k in all_keys]
emo_values = [results["Emoji"].get(k, 0) for k in all_keys]

# === Plot normalized histogram ===
plt.figure(figsize=(8, 5))
width = 0.35
x = range(len(all_keys))

plt.bar([i - width/2 for i in x], inv_values, width=width, label="Invisible chars", alpha=0.7)
plt.bar([i + width/2 for i in x], emo_values, width=width, label="Emojis", alpha=0.7)

plt.xticks(x, [str(k) for k in all_keys])
plt.xlabel("Tokenization diff value")
plt.ylabel("Percentage of total characters")
plt.title("Normalized tokenization comparison: Invisible vs Emoji characters")
plt.xticks(x, [("no tokens" if k == -1 else str(k)) for k in all_keys])
plt.legend()
plt.tight_layout()
plt.show()


