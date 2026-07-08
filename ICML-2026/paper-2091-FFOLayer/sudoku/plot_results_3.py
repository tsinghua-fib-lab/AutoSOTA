import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette()

batch_size = 8
BASE_DIR = f"../sudoku_results_{batch_size}"

# -----------------------
# Split methods into QP / CP
# -----------------------
QP_METHODS = [
    "qpth",
    "dqp",
    "ffoqp_eq",
]
QP_METHODS_LEGEND = {
    "qpth": "qpth",
    "dqp": "dQP",
    "ffoqp_eq": "FFOQP",
}

CP_METHODS = [
    "cvxpylayer",
    "lpgd",
    "bpqp",
    "ffocp_eq",
]
CP_METHODS_LEGEND = {
    "cvxpylayer": "CvxpyLayer",
    "lpgd": "LPGD",
    "bpqp": "BPQP",
    "ffocp_eq": "FFOCP",
}

LINEWIDTH = 1.5


def load_results(base_dir=BASE_DIR, methods=None, legend=None):
    """
    Read all CSVs under base_dir/<method>/*.csv, attach parsed metadata + human-readable method name.
    - methods: folder names (can include *_steps)
    - legend: maps base method name (without *_steps) -> display name
    """
    if methods is None:
        raise ValueError("methods must be provided.")
    if legend is None:
        raise ValueError("legend must be provided.")

    dfs = []
    for folder_method in methods:
        pattern = os.path.join(base_dir, folder_method, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = pd.read_csv(fp)

            base_method = folder_method.removesuffix("_steps")
            if base_method not in legend:
                raise KeyError(f"Method {base_method} not found in legend mapping.")

            df["method"] = legend[base_method]

            fname = os.path.basename(fp)

            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["n"] = grab(r"n(\d+)", int)
            df["lr"] = grab(r"lr([0-9eE\.\-]+)", float)
            dfs.append(df)

        print("loaded folder:", folder_method)

    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir} for methods={methods}.")
    return pd.concat(dfs, ignore_index=True, sort=False)


def plot_time_vs_method(df, time_names=("forward_time", "backward_time"), plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_method = df.groupby("method")[list(time_names)].mean().reset_index()

    df_long = df_avg_method.melt(
        id_vars="method",
        value_vars=list(time_names),
        var_name="Metrics",
        value_name="Time",
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_long, x="method", y="Time", hue="Metrics")
    plt.ylabel("Time")
    plt.title("Forward and Backward Time")
    plt.savefig(f"{plot_path}/{plot_name_tag}_time_vs_method.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_vs_epoch(
    df,
    time_names=("forward_time", "backward_time"),
    iteration_name="epoch",
    plot_path=BASE_DIR,
    plot_name_tag="",
):
    df_avg = df.groupby(["method", iteration_name])[list(time_names)].mean().reset_index()

    # Forward
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_avg,
        x=iteration_name,
        y=time_names[0],
        hue="method",
        marker=None,
        dashes=False,
        linewidth=LINEWIDTH,
    )
    plt.ylabel("Forward Time")
    plt.title(f"Forward Time vs {iteration_name}")
    plt.savefig(f"{plot_path}/{plot_name_tag}_forward_time_vs_{iteration_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Backward
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_avg,
        x=iteration_name,
        y=time_names[1],
        hue="method",
        marker=None,
        dashes=False,
        linewidth=LINEWIDTH,
    )
    plt.ylabel("Backward Time")
    plt.title(f"Backward Time vs {iteration_name}")
    plt.savefig(f"{plot_path}/{plot_name_tag}_backward_time_vs_{iteration_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_total_time_vs_method(df, time_names=("forward_time", "backward_time"), plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_method = df.groupby("method")[list(time_names)].mean().reset_index()

    methods = df_avg_method["method"]
    forward = df_avg_method[time_names[0]]
    backward = df_avg_method[time_names[1]]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, forward, label=time_names[0], color=palette[0])
    plt.bar(methods, backward, bottom=forward, label=time_names[1], color=palette[1])
    plt.ylabel("Time")
    plt.title("Total Time vs Method")
    plt.legend()
    plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_vs_epoch(
    df,
    loss_metric_name,
    iteration_name="epoch",
    plot_path=BASE_DIR,
    plot_name_tag="",
    loss_range=None,
    stride=50,
):
    df_avg = df.groupby(["method", iteration_name])[[loss_metric_name]].mean().reset_index()
    df_avg = df_avg[df_avg[iteration_name] % stride == 0]

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=df_avg,
        x=iteration_name,
        y=loss_metric_name,
        hue="method",
        dashes=False,
        linewidth=LINEWIDTH,
    )
    plt.ylabel(loss_metric_name)
    plt.title(f"{loss_metric_name} vs {iteration_name}")

    if loss_range is not None:
        ax.set_ylim(loss_range)

    plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric_name}_vs_{iteration_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def run_group(methods, legend, group_tag, base_dir=BASE_DIR):
    """
    Produce plots for:
      1) overall timing CSVs under base_dir/<method>/*.csv
      2) per-step CSVs under base_dir/<method>_steps/*.csv
    """
    method_order = [legend[m] for m in methods]

    df = load_results(base_dir=base_dir, methods=methods, legend=legend)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    plot_time_vs_method(df, time_names=("forward_time", "backward_time"), plot_path=base_dir, plot_name_tag=f"sudoku_{group_tag}")
    plot_total_time_vs_method(df, time_names=("forward_time", "backward_time"), plot_path=base_dir, plot_name_tag=f"sudoku_{group_tag}")

    step_methods = [m + "_steps" for m in methods]
    df_steps = load_results(base_dir=base_dir, methods=step_methods, legend=legend)
    df_steps = df_steps.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df_steps["method"] = pd.Categorical(df_steps["method"], categories=method_order, ordered=True)

    plot_time_vs_method(
        df_steps,
        time_names=("iter_forward_time", "iter_backward_time"),
        plot_path=base_dir,
        plot_name_tag=f"sudoku_{group_tag}_steps",
    )
    plot_time_vs_epoch(
        df_steps,
        time_names=("iter_forward_time", "iter_backward_time"),
        iteration_name="iter",
        plot_path=base_dir,
        plot_name_tag=f"sudoku_{group_tag}_steps",
    )
    plot_total_time_vs_method(
        df_steps,
        time_names=("iter_forward_time", "iter_backward_time"),
        plot_path=base_dir,
        plot_name_tag=f"sudoku_{group_tag}_steps",
    )

    for metric, kwargs in [
        ("train_loss", dict(loss_range=(0.07, 0.1))),
        ("train_error", dict(loss_range=None)),
    ]:
        if metric in df_steps.columns:
            plot_loss_vs_epoch(
                df_steps,
                metric,
                iteration_name="iter",
                plot_path=base_dir,
                plot_name_tag=f"sudoku_{group_tag}_steps",
                **kwargs,
            )
        else:
            print(f"[{group_tag}] skip {metric}: column not found in steps CSVs")


if __name__ == "__main__":
    # QP plots
    run_group(QP_METHODS, QP_METHODS_LEGEND, group_tag="qp", base_dir=BASE_DIR)

    # CP plots
    run_group(CP_METHODS, CP_METHODS_LEGEND, group_tag="cp", base_dir=BASE_DIR)
