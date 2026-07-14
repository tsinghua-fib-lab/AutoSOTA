import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def read_pbt_hps(pbt_dir):
    # Get the initial values from the results files
    result_files = glob.glob(os.path.join(pbt_dir, "run_model_*/result.json"))

    def get_first_result(fpath):
        with open(fpath, "r") as file:
            result = json.loads(file.readline())
        return result

    inits = pd.DataFrame([get_first_result(fpath) for fpath in result_files])
    # Get the perturbations from `pbt_global.txt`
    pbt_global_path = os.path.join(pbt_dir, "pbt_global.txt")
    with open(pbt_global_path, "r") as file:
        perturbs = [json.loads(line) for line in file.read().splitlines()]
    perturbs = pd.DataFrame(
        perturbs,
        columns=[
            "target_tag",
            "clone_tag",
            "target_iteration",
            "cur_epoch",
            "old_config",
            "config",
        ],
    )
    # Use trial_num to match intial values to perturbations
    inits["trial_num"] = inits.trial_id.apply(lambda x: int(x.split("_")[1]))
    perturbs["trial_num"] = perturbs.target_tag.apply(lambda x: int(x.split("_")[0]))
    # Combine initial values and perturbations
    hps_df = pd.concat([inits, perturbs]).reset_index()
    hps_df = hps_df[["trial_num", "cur_epoch", "config"]]
    # Expand the config dictionary into separate columns and recombine
    configs = pd.json_normalize(hps_df.pop("config"))
    hps_df = pd.concat([hps_df, configs], axis=1)
    return hps_df


def plot_pbt_hps(pbt_dir, plot_field, save_dir=None, **kwargs):
    """Plots an HP for all models over the course of PBT.
    This function generates a plot to visualize how an HP
    changes over the course of PBT.
    Parameters
    ----------
    pbt_dir : str
        The path to the PBT run.
    plot_field : str
        The HP to plot. See the HP log headers or lfads_tf2
        source code for options.
    save_dir : str, optional
        The directory for saving the figure, by default None will
        show an interactive plot
    kwargs: optional
        Any keyword arguments to be passed to pandas.DataFrame.plot
    """

    hps_df = read_pbt_hps(pbt_dir)
    plot_df = hps_df.pivot(index="cur_epoch", columns="trial_num", values=plot_field)
    plot_df = plot_df.ffill()
    gen_range = plot_df.index.min(), plot_df.index.max()
    field_range = plot_df.min().min(), plot_df.max().max()
    plot_kwargs = dict(
        drawstyle="steps-post",
        legend=False,
        logy=True,
        c="b",
        alpha=0.2,
        title=f"{plot_field} for PBT run at {pbt_dir}",
        xlim=gen_range,
        ylim=field_range,
        figsize=(10, 5),
    )
    plot_kwargs.update(kwargs)
    plot_df.plot(**plot_kwargs)
    if save_dir is not None:
        filename = plot_field.replace(".", "_").lower()
        fig_path = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
