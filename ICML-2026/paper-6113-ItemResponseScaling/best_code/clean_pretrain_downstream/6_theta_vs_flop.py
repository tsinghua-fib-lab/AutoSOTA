from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tueplots import bundles
bundles.icml2024()
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "6_long.parquet"
OUTPUT_DIR = BASE_DIR / "results" / "6_theta_vs_flop"
MAX_STEP_PERCENTAGE = 0.1

df = pd.read_parquet(INPUT_PATH)
ability_cols = [c for c in df.columns if c.startswith("ability_") and c.endswith("_binary_1pl")]
unique_benches = sorted({c[len("ability_") : -len("_binary_1pl")] for c in ability_cols})
unique_data_mixes = sorted(df["model_data_mix"].dropna().unique())

for bench in unique_benches:
    binary_1pl_col = f"ability_{bench}_binary_1pl"
    beta_1pl_col = f"ability_{bench}_beta_1pl"
    binary_2pl_col = f"ability_{bench}_binary_2pl"
    beta_2pl_col = f"ability_{bench}_beta_2pl"
    bench_dir = OUTPUT_DIR / bench
    bench_dir.mkdir(parents=True, exist_ok=True)

    series_specs = [
        ("binary_1pl", binary_1pl_col),
        ("beta_1pl", beta_1pl_col),
        ("binary_2pl", binary_2pl_col),
        ("beta_2pl", beta_2pl_col),
    ]

    for data_mix in tqdm(unique_data_mixes, desc=f"{bench} data_mix", leave=False):
        mix_df = df[df["model_data_mix"] == data_mix]
        for series_label, series_col in tqdm(series_specs, desc=f"{bench} series", leave=False):
            plot_df = mix_df.loc[mix_df["FLOP"].notna(), ["FLOP", series_col]]

            avg_rows = []
            for _, group in tqdm(mix_df.groupby(["model_data_mix", "model_size"]), desc=f"{bench} groups", leave=False):
                group = group.loc[:, ["model_step", "FLOP", series_col]]
                group = group.sort_values("model_step")
                top_n = int(np.ceil(len(group) * MAX_STEP_PERCENTAGE))
                avg_rows.append(
                    {
                        "FLOP": group["FLOP"].iloc[-1],
                        series_col: float(group[series_col].tail(top_n).mean()),
                    }
                )
            avg_df = pd.DataFrame(avg_rows)
            
            # with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
            #     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), sharey=True)
            #     ax_linear, ax_log = axes[0]
            #     ax_avg_linear, ax_avg_log = axes[1]

            #     ax_linear.scatter(plot_df["FLOP"], plot_df[series_col], s=80, alpha=0.7, label=series_label)
            #     ax_linear.set_xlabel("FLOP", fontsize=20)
            #     ax_linear.set_ylabel(r"$\theta$", fontsize=20)
            #     ax_linear.set_title(r"Final $\theta$", fontsize=22)
            #     ax_linear.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            #     ax_linear.tick_params(axis="both", labelsize=18)
            #     ax_linear.legend(fontsize=16)

            #     ax_log.scatter(plot_df["FLOP"], plot_df[series_col], s=80, alpha=0.7)
            #     ax_log.set_xlabel("FLOP (log scale)", fontsize=20)
            #     ax_log.set_xscale("log")
            #     ax_log.set_title(r"Final $\theta$, Log Scale", fontsize=22)
            #     ax_log.tick_params(axis="both", labelsize=18)

            #     ax_avg_linear.scatter(avg_df["FLOP"], avg_df[series_col], s=80, alpha=0.7, label=series_label)
            #     ax_avg_linear.set_xlabel("FLOP", fontsize=20)
            #     ax_avg_linear.set_ylabel(r"$\theta$", fontsize=20)
            #     ax_avg_linear.set_title(r"Avg Final 10\% $\theta$", fontsize=22)
            #     ax_avg_linear.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            #     ax_avg_linear.tick_params(axis="both", labelsize=18)
            #     ax_avg_linear.legend(fontsize=16)

            #     ax_avg_log.scatter(avg_df["FLOP"], avg_df[series_col], s=80, alpha=0.7)
            #     ax_avg_log.set_xlabel("FLOP (log scale)", fontsize=20)
            #     ax_avg_log.set_xscale("log")
            #     ax_avg_log.set_title(r"Avg Final 10\% $\theta$, Log Scale", fontsize=22)
            #     ax_avg_log.tick_params(axis="both", labelsize=18)

            #     fig.suptitle(f"{bench}", fontsize=22)
            #     fig.tight_layout()
            #     out_path = bench_dir / f"{series_label}_{data_mix}_{bench}_theta_vs_flop.png"
            #     fig.savefig(out_path, dpi=150)
            #     plt.close(fig)

            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax_log = plt.subplots(figsize=(6, 4.5))

                ax_log.scatter(plot_df["FLOP"], plot_df[series_col], s=80, alpha=0.7)
                ax_log.set_xlabel("FLOP", fontsize=20)
                ax_log.set_ylabel(r"$\theta$", fontsize=20)
                ax_log.set_xscale("log")
                ax_log.tick_params(axis="both", labelsize=18)

                fig.suptitle(f"{data_mix}, {bench}", fontsize=22)
                fig.tight_layout()
                out_path = bench_dir / f"{series_label}_{data_mix}_{bench}_theta_vs_flop.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
