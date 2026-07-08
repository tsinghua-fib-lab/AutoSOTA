import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TASK = "sudoku"
batch_size = 32
BASE_DIR = f"../{TASK}_results_{batch_size}"
PLOT_PATH=os.path.join(BASE_DIR, "figures")
os.makedirs(PLOT_PATH, exist_ok=True)

# METHODS = [
#     "cvxpylayer",
#     "ffoqp_eq",
#     "ffocp_eq",
#     "lpgd"
# ]

METHODS = [
    "lpgd"
]

def load_results(base_dir=BASE_DIR, methods=METHODS):
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = pd.read_csv(fp)
            df["method"] = m

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["n"] = grab(r"n(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            #df["eps"]  = grab(r"eps([0-9eE\.\-]+)", float)
            dfs.append(df)
        
        print("method: ", m)
        # print(df)
        
    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    return pd.concat(dfs, ignore_index=True, sort=False)




def plot_metric_curve(df, metric_type, fig_path=PLOT_PATH):
    
    sns.set_theme(style="whitegrid", context="talk")
    assert(metric_type in ["error", "loss"])
    fig_name=f"{metric_type}_curve.png"

    metric_cols = [c for c in [f"test_{metric_type}", f"train_{metric_type}"] if c in df.columns]
    id_vars = [c for c in ["epoch","method","seed","ydim"] if c in df.columns]

    ## collapse metric columns into a value column
    long_curves = df.melt(
        id_vars=id_vars,
        value_vars=metric_cols,
        var_name="metric", value_name="value"
    ).dropna(subset=["value","epoch","method"])

    ## for each metric (col) and each dimensionality (ydim), create a plot of value vs epoch for different methods
    g = sns.relplot(
        data=long_curves, x="epoch", y="value",
        hue="method", style="method",
        markers=True, dashes=True,
        kind="line", ci=None,
        linewidth=1.5, alpha=0.9,
        col="metric", col_wrap=2, height=4, aspect=1.3,
        facet_kws=dict(sharey=False)
    )
    
    # g = sns.relplot(
    #     data=long_curves, x="epoch", y="value",
    #     hue="method", style="method",
    #     markers=True, dashes=True,
    #     kind="line", ci=None,
    #     linewidth=1.5, alpha=0.9,
    #     col="metric", row='ydim',  # remove col_wrap
    #     height=4, aspect=1.3,
    #     facet_kws=dict(sharey=False)
    # )

    g.set_titles("{col_name}")
    g.set_xlabels("Epoch"); g.set_ylabels("Loss")
    plt.suptitle("Loss vs Epoch by Method", y=1.02)

    ## reset legend
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if g._legend is not None:
        g._legend.remove()

    g.figure.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=max(1, len(labels)),
        frameon=False
    )
    g.figure.subplots_adjust(bottom=0.50)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, fig_name), dpi=300, bbox_inches="tight")
    
    
    
    
def plot_final_metric(df, metric_type, fig_path=PLOT_PATH):
    keys = ["method","seed"] if "seed" in df.columns else ["method"]
    assert(metric_type in ["loss", "error"])
    fig_name=f"{metric_type}_final.png"

    if "epoch" in df.columns and df["epoch"].notna().any():
        last_rows = df.sort_values("epoch").groupby(keys, dropna=False).tail(1)
    else:
        last_rows = df.groupby(keys, dropna=False).tail(1)
        
    #print(last_rows)

    loss_final = last_rows.melt(
        id_vars=["method","seed","ydim"],
        value_vars=[f"train_{metric_type}",f"test_{metric_type}"],
        var_name="metric", value_name="value"
    ).dropna(subset=["value","method"])
    
    #print(loss_final)

    plt.figure(figsize=(10,5))
    
    ax = sns.barplot(data=loss_final, x="method", y="value", hue="metric", errorbar=("ci",95))
    ax.set_title("Final Loss by Method")
    ax.set_xlabel(""); ax.set_ylabel("Loss")
    ax.tick_params(axis="x", rotation=20)
    
    # g = sns.catplot(
    #     data=loss_final,
    #     x="method", y="value", hue="metric",
    #     col="ydim", kind="bar", ci=95,
    #     height=4, aspect=1.2
    # )
    # g.set_titles("ydim = {col_name}")
    # g.set_axis_labels("", "Loss")
    # g.set_xticklabels(rotation=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, fig_name), dpi=300, bbox_inches="tight")
    
    
    

def plot_forward_backward_time(df, fig_path=PLOT_PATH, fig_name="forward_backward_time.png"):
    keys = ["method","seed"] if "seed" in df.columns else ["method"]
    
    time_median = (
        df.groupby(keys, dropna=False)[["forward_time","backward_time"]]
        .median().reset_index()
    )
    
    # print(time_median)

    time_long = time_median.melt(id_vars=["method","seed"], value_vars=["forward_time","backward_time"],
                             var_name="phase", value_name="seconds").dropna(subset=["seconds","method"])
    
    # print(time_long)

    # method_order_all = [
    #     "cvxpylayer","qpth","ffocp_eq","lpgd"
    # ]
    method_order_all = METHODS
    
    method_order = [m for m in method_order_all if m in time_long["method"].unique()]
    phase_order = ["forward_time", "backward_time"]
    time_long["method"] = pd.Categorical(time_long["method"],
                                        categories=method_order, ordered=True)
    
    # print(time_long)

    plt.figure(figsize=(10,5))
    ax = sns.barplot(
        data=time_long,
        x="method", y="seconds",
        hue="phase",
        order=method_order,
        hue_order=phase_order,
        errorbar=("ci",95)
    )
    
    ax.set_title("Per-Epoch Time by Method")
    ax.set_xlabel(""); ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, fig_name), dpi=300, bbox_inches="tight")



if __name__=="__main__":
    df = load_results()
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df['ydim'] = df['n'] ** 6
    print(df)
    
    plot_metric_curve(df, metric_type="loss")
    plot_metric_curve(df, metric_type="error")
    plot_final_metric(df, metric_type="loss")
    plot_final_metric(df, metric_type="error")
    plot_forward_backward_time(df)

    
