import matplotlib.pyplot as plt

# Data
x = [0, 3, 5, 7]
y = [0.461165049, 0.359223301, 0.412621359, 0.558252427]
ppl = [46.40288162231445, 96.69496154785156, 114.25381469726562, 126.41830444335938]

# Plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(6,4))  # square plot

color = 'tab:blue'
ax1.set_xlabel("# tokens", fontsize=14, fontweight="bold")
ax1.set_ylabel("E2E-ASR", color=color, fontsize=14, fontweight="bold")
ax1.plot(x, y, marker='o', linestyle='-', linewidth=2, color=color, label="E2E-ASR")
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.grid(True, linestyle="--", alpha=0.6)

# Second y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Perplexity", color=color, fontsize=14, fontweight="bold")
ax2.plot(x, ppl, marker='o', linestyle='-', linewidth=2, color=color, label="Perplexity")
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

# Bold x-axis tick labels
ax1.tick_params(axis='x', labelsize=12)

# Save plot
fig.tight_layout()
plt.savefig("length_plot.pdf")
plt.show()