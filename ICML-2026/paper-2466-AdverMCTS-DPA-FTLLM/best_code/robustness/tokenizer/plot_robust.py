import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("aggregate_char.csv", sep=',')  

# Choose the 5 specific Unicodes you want to plot
selected_unicodes = ['U+2060', 'U+2061', 'U+2062', 'U+2063', 'U+2064']
df_sel = df[df['Unicode'].isin(selected_unicodes)]

# Extract the numeric columns
token_cols = ['toke_diff_0', 'toke_diff_1', 'toke_diff_2', 'toke_diff_3', 'toke_diff_4']

# Set up the plot
plt.figure(figsize=(10, 6))
bar_width = 0.15
x = range(len(token_cols))

# Plot one bar group per Unicode
for i, (_, row) in enumerate(df_sel.iterrows()):
    plt.bar(
        [p + i * bar_width for p in x],
        row[token_cols],
        width=bar_width,
        label=row['Unicode']
    )

# Formatting
plt.xlabel("Tokenization Difference Test")
plt.ylabel("Additional Tokens Count")
plt.title("Additional Tokens for Selected Unicode Characters")
plt.xticks([p + 2 * bar_width for p in x], token_cols)
plt.legend(title="Unicode")
plt.tight_layout()

plt.show()

