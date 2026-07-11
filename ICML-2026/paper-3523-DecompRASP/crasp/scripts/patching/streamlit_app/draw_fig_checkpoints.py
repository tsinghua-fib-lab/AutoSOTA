import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="white")  # clean background

# ---------------- Data ----------------
acc_curves = [
    (1100.0, (0.263, 0.0, 0.0)),
    (1200.0, (0.925, 0.442, 0.0035)),
    (1300.0, (0.994, 0.937, 0.508)),
    (1400.0, (0.999, 0.992, 0.81)),
    (3000.0, (1.0, 1.0, 0.9905)),
]

num_edges_curves = [
    (1100.0, None),
    (1200.0, None),
    (1300.0, None),
    (1400.0, None),
    (3000.0, 0.21875),
]

# acc_curves = [(500.0, (0.343, 0.002, 0.0)), (700.0, (0.6095, 0.0575, 0.0)), (1400.0, (0.87, 0.274, 0.008)), (1900.0, (0.914, 0.533, 0.078)), (2100.0, (0.969, 0.853, 0.4425)), (2200.0, (0.992, 0.9515, 0.742)), (3300.0, (1.0, 0.999, 0.9975)), (6000, (1.0, 1.0, 0.997))]
# num_edges_curves = [(500.0, None), (700.0, None), (1400.0, None), (1900.0, None), (2100.0, None), (2200.0, None), (3300.0, 0.0948905109489051), (6000, 0.0948905109489051)]

# ---------------- Accuracy ----------------
steps = np.array([s for s, _ in acc_curves])
acc_0_50 = np.array([a[0] for _, a in acc_curves])
acc_50_100 = np.array([a[1] for _, a in acc_curves])
acc_100_plus = np.array([a[2] for _, a in acc_curves])

# ---------------- Edge data ----------------
edge_steps = []
edge_props = []

for step, prop in num_edges_curves:
    edge_steps.append(step)
    edge_props.append(prop)

edge_steps = np.array(edge_steps)
edge_props = np.array(edge_props)

# ---------------- Colors ----------------
acc_palette = sns.color_palette("Blues", 4)
acc_colors = acc_palette[1:]  # light -> dark
value_color = (0.2, 0.7, 0.3)  # green-ish for dots
line_none_color = "red"
line_exist_color = (0.3, 0.6, 0.4)  # green-ish for lines

# ---------------- Plot ----------------
fig, ax1 = plt.subplots(figsize=(8,5))

# Accuracy curves
ax1.plot(steps, acc_0_50, marker="o", color=acc_colors[2], label="Test acc < 50")
ax1.plot(steps, acc_50_100, marker="o", color=acc_colors[1], label="Test acc [51, 100]")
ax1.plot(steps, acc_100_plus, marker="o", color=acc_colors[0], label="Test acc [101, 150]")

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Task Accuracy")
ax1.set_xlim(0, steps.max()*1.05)
ax1.set_ylim(0, 1.05)

# ---------------- Vertical dotted lines ----------------
for step, prop in zip(edge_steps, edge_props):
    color = line_none_color if prop is None else line_exist_color
    ax1.axvline(step, linestyle=":", linewidth=1.5, color=color, alpha=0.7)

# ---------------- Right y-axis for edge values ----------------
ax2 = ax1.twinx()
ax2.set_ylabel("Proportion of necessary edges")
ax2.set_ylim(0, 1.05)

# Plot edge values on right y-axis
for step, prop in zip(edge_steps, edge_props):
    if prop is not None:
        ax2.scatter(step, prop, color=value_color, s=80, zorder=5)

# ---------------- Legend ----------------
lines1, labels1 = ax1.get_legend_handles_labels()
# Add dummy handles for vertical lines and edge dots
ax1.plot([], [], color=line_none_color, linestyle=":", label="LLNA not hold")
ax1.plot([], [], color=line_exist_color, linestyle=":", label="LLNA holds")
ax1.scatter([], [], color=value_color, s=80, label="Edge proportion")
ax1.legend(loc="center right")

plt.tight_layout()
plt.show()