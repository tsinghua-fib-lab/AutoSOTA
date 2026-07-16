import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tueplots import bundles
import warnings
warnings.filterwarnings("ignore")

# split_method = "random_55split"
split_method = "hardeasy_split"
# split_method = "small_subset_split"
# split_method = "random_28split"
with open(f"downstream_binary_generalize_{split_method}.pkl", "rb") as f:
    results_dict = pickle.load(f)

out_dir = f"../result/downstream_binary_generalize_{split_method}"
os.makedirs(out_dir, exist_ok=True)

# Aggregated Bar Plot
for scenario, model_results in results_dict.items():
    models = list(model_results.keys())

    linear_train_mse = []
    irt_train_mse    = []
    linear_test_mse  = []
    irt_test_mse     = []
    for m in models:
        res = model_results[m]
        linear_train_mse.append(mean_squared_error(res["gt_ctt_train"], res["train_linears"]))
        irt_train_mse.append(   mean_squared_error(res["gt_ctt_train"], res["train_irts"]))
        linear_test_mse.append( mean_squared_error(res["gt_ctt_test"],  res["train_linears"]))
        irt_test_mse.append(    mean_squared_error(res["gt_ctt_test"],  res["test_irts"]))

    # One plot per scenario
    x = np.arange(len(models))
    width = 0.35
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Overlapping bars for Train at x - width/2
        ax.bar(x - width/2, linear_train_mse, width=width, alpha=0.6, label="Linear (Train)")
        ax.bar(x - width/2, irt_train_mse,    width=width, alpha=0.6, label="IRT (Train)")

        # Overlapping bars for Test at x + width/2
        ax.bar(x + width/2, linear_test_mse, width=width, alpha=0.6, label="Linear (Test)")
        ax.bar(x + width/2, irt_test_mse,    width=width, alpha=0.6, label="IRT (Test)")

        ax.set_ylabel("MSE", fontsize=16)
        ax.set_title(f"{scenario}", fontsize=18)
        ax.set_xticks(x)
        models = [m.split("-intermediate-checkpoints")[0] if m.endswith("-intermediate-checkpoints") else m for m in models]
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=12, loc="upper right")
        ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(f"{out_dir}/downstream_mse_{scenario}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
# Single Line Plot
for scenario, model_results in results_dict.items():
    for m, res in model_results.items():
        steps    = res["time_steps"]
        gt_ctt_train  = res["gt_ctt_train"]   # shape (K,)
        gt_ctt_test   = res["gt_ctt_test"]    # shape (K,)
        train_linears = res["train_linears"]
        train_irts    = res["train_irts"]
        test_irts     = res["test_irts"]

        clean_name = m.split("-intermediate-checkpoints")[0]
        save_dir = os.path.join(out_dir, scenario)
        os.makedirs(save_dir, exist_ok=True)
        steps = steps * 300 * 1e9 / 143000 * 12 * 1e9
        
        train_mse_irt    = mean_squared_error(gt_ctt_train, train_irts)
        test_mse_irt = mean_squared_error(gt_ctt_test, test_irts)
        with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax = plt.subplots(figsize=(6, 3))

                ax.plot(steps, gt_ctt_train,
                        color='blue',
                        label="Train GT")
                ax.plot(steps, gt_ctt_test,
                        color='red',
                        label="Test GT")
                
                ax.plot(steps, train_irts,
                        color='blue',
                        linestyle='--',
                        label=f"Train IRT (MSE={train_mse_irt:.4f})")
                ax.plot(steps, test_irts,
                        color='red',
                        linestyle='--',
                        label=f"Test IRT (MSE={test_mse_irt:.4f})")

                # Axes formatting
                ax.set_xscale("log")
                ax.set_xlabel("FLOP", fontsize=20)
                ax.set_ylabel(f"Acc", fontsize=20)
                ax.set_ylim(0, 1)
                ax.tick_params(axis="both", labelsize=14)
                ax.legend(fontsize=12)
                # ax.set_title("Train vs Test", fontsize=16)

                fig.tight_layout()
                fig.savefig(f"{save_dir}/downstream_{clean_name}.png",
                                dpi=300,
                                bbox_inches="tight")
                plt.close(fig)
