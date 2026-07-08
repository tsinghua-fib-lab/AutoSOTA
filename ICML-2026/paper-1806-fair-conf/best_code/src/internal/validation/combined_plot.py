from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLORS = ["#3B528B", "#472D7B", "#5EC962", "#21918C"]


def single_run():
    # writer, fairness_input = run_conformal(cfg)
    # run_llm_in_loop(cfg, writer, fairness_input)

    size_df = pd.read_csv("logs/Aug29_11-facet/size_statistics.csv")
    acc_df = pd.read_csv("logs/Aug29_11-facet/accuracy_statistics.csv")

    # Ensure consistent ordering of groups
    unique_groups = sorted(size_df["group_text"].unique())

    # Ensure consistent ordering of treatments (and exclude Control if desired)
    treatment_order = [t for t in size_df["Treatment"].unique() if t != "Control"]
    if "Control" in size_df["Treatment"].unique():
        treatment_order = ["Control"] + treatment_order  # put Control first

    size_df["Treatment"] = pd.Categorical(size_df["Treatment"], treatment_order)
    acc_df["Treatment"] = pd.Categorical(acc_df["Treatment"], treatment_order)

    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Top plot: Set size ---
    sns.barplot(
        data=size_df,
        x="Treatment",
        y="Set size",
        hue="group_text",
        hue_order=unique_groups,
        palette=COLORS[: len(unique_groups)],
        ax=axes[0],
    )
    axes[0].set_title("Average set size by treatment and group")
    axes[0].set_ylabel("Average set size")
    axes[0].legend_.remove()

    # --- Bottom plot: Accuracy ---
    sns.barplot(
        data=acc_df,
        x="Treatment",
        y="Accuracy",
        hue="group_text",
        hue_order=unique_groups,
        palette=COLORS[: len(unique_groups)],
        ax=axes[1],
    )
    axes[1].set_title("Accuracy by treatment and group")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Treatment")
    axes[1].legend_.remove()

    # --- One shared legend outside ---
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Group",
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(unique_groups),
    )

    # Clean layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save output
    output_dir = Path("tmp_plot")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "Combined_size_accuracy.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot at: {output_path}")
    return
