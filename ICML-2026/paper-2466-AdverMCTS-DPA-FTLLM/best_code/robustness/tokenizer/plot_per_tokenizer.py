import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("aggregate_tokenizer.csv", sep=",") 
print(df.columns.tolist())

df.set_index("model", inplace=True)

# Plot stacked bar chart
df.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab10")

plt.title("Token number per tokenizer")
plt.ylabel("Count")
plt.xlabel("Tokenizer")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Category")
plt.tight_layout()
plt.show()

