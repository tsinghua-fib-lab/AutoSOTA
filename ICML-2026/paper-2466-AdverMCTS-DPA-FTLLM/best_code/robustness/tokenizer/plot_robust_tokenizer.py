import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("invisible_char_token_counts.csv", sep=',')

# Select the 5 Unicode characters you want to visualize
selected_unicodes = ['U+2060', 'U+2061', 'U+2062', 'U+2063', 'U+2064']
df_sel = df[df['Unicode'].isin(selected_unicodes)]

# Keep only *_diff columns except BERT
diff_cols = [col for col in df.columns if col.endswith('_diff') and 'bert' not in col]

# Melt the data for plotting
df_melt = df_sel.melt(
    id_vars=['Unicode'],
    value_vars=diff_cols,
    var_name='Model',
    value_name='Token_Diff'
)


# Clean up model names
df_melt['Model'] = df_melt['Model'].str.replace('_diff', '', regex=False)

# Plot
plt.figure(figsize=(12, 6))
for unicode_val in selected_unicodes:
    subset = df_melt[df_melt['Unicode'] == unicode_val]
    plt.plot(subset['Model'], subset['Token_Diff'], marker='o', label=unicode_val)

# Style and labels
plt.xticks(rotation=45, ha='right')
plt.xlabel("Tokenizer")
plt.ylabel("Additional Tokens")
plt.title("Tokenization Differences Across Tokenizers for Selected Unicode Characters")
plt.legend(title="Unicode")

# ✅ Force y-axis to show only 1, 2, and 3
plt.yticks([1, 2, 3])

plt.tight_layout()
plt.show()
