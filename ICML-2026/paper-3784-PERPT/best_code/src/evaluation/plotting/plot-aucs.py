import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE   = "#4287f5"   # seq + phyloP
PURPLE = "#6F2DA8"   # phyloP-only
ORANGE = "#FF8C32"   # seq-only

entries = [
    ("gamba ntp-only\nlikelihood auc", 0.501, ORANGE),
    ("gamba cep-only\nevolutionary auc", 0.677, PURPLE),
    ("gamba ntp+cep\nlikelihood auc", 0.512, BLUE),
    ("gamba ntp+cep\nevolutionary auc", 0.696, BLUE),
    ("bi-gamba mlm-only\nlikelihood auc", 0.501, ORANGE),
    ("bi-gamba mem-only\nevolutionary auc", 0.737, PURPLE),
    ("bi-gamba mlm+mem\nlikelihood auc", 0.528, BLUE),
    ("bi-gamba mlm+mem\nevolutionary auc", 0.740, BLUE),
]

labels = [e[0] for e in entries]
values = [e[1] for e in entries]
colors = [e[2] for e in entries]

plt.figure(figsize=(10, 5))
bars = plt.bar(range(len(values)), values, color=colors)

plt.ylabel("AUC")
plt.ylim(0.48, 0.95)
plt.xticks(range(len(labels)), labels, rotation=30, ha="right")

for b, v in zip(bars, values):
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

legend_handles = [
    Patch(facecolor=ORANGE, label="sequence-only"),
    Patch(facecolor=PURPLE, label="phyloP-only"),
    Patch(facecolor=BLUE, label="sequence + phyloP"),
]

# place legend outside to the right
plt.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, title="pretraining")

plt.title("ClinVar VEP: gamba and bi-gamba")
plt.tight_layout()

out_path_side = "/home/mica/gamba/data_processing/data/240-mammalian/figures/vep_clinvar_gamba_bigamba_bar_sidelegend.svg"
plt.savefig(out_path_side, dpi=300, bbox_inches="tight")
out_path_side
