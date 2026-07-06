import h5py
from typing import Optional
import matplotlib.pyplot as plt


def eeg_plot(
    data_dict: dict, plot_pred_dict: dict, exp_result_folder: Optional[str] = None
):
    """
    data_dict: dict, containing keys "all_X", "train_Y", "train_mask", "test_Y", "test_mask", "means", "stds"
    plot_pred_dict: dict, containing keys "denser_X", "plot_pred_means", "plot_pred_vars"

    Make sure everything is on CPU.
    """
    data_X, train_Y, train_mask, test_Y, test_mask, means, stds = (
        data_dict["all_X"].to("cpu"), data_dict["train_Y"].to("cpu"), data_dict["train_mask"].to("cpu"), data_dict["test_Y"].to("cpu"),
        data_dict["test_mask"].to("cpu"), data_dict["means"].to("cpu"), data_dict["stds"].to("cpu"),
    )

    # Transform back
    train_Y = train_Y * stds + means
    test_Y = test_Y * stds + means

    denser_X, plot_pred_means, plot_pred_vars = (
        plot_pred_dict["denser_X"].to("cpu"), plot_pred_dict["plot_pred_means"].to("cpu"), plot_pred_dict["plot_pred_vars"].to("cpu")
    )

    # Transform back
    plot_pred_means = plot_pred_means * stds + means
    plot_pred_vars = plot_pred_vars * (stds ** 2)

    assert data_X.shape == (256, 1) and train_Y.shape == test_Y.shape == train_mask.shape == test_mask.shape == (256, 7)
    fig = plt.figure(figsize=(9, 16))
    fig.suptitle("EEG")

    col_name_dict = {
        0: "F1", 1: "F2", 6: "FZ",
    }

    # Define x-tick positions and labels
    x_ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]  # Assuming data is normalized between -1 and 1
    x_labels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i, p in enumerate([0, 1, 6]): # F1, F2 and FZ
        ax = plt.subplot(3, 1, (i+1))
        plt.plot(denser_X.squeeze(-1), plot_pred_means[:, p], color="blue", label="pred mean")
        plt.fill_between(
            denser_X.squeeze(-1),
            plot_pred_means[:, p] - 1.96 * plot_pred_vars[:, p].sqrt(),
            plot_pred_means[:, p] + 1.96 * plot_pred_vars[:, p].sqrt(),
            alpha=0.2,
            color="blue"
        )

        plt.scatter(
            data_X.squeeze(-1)[train_mask[:, p]], train_Y[:, p][train_mask[:, p]],
            marker="x", color="black", label="train data", s=15
        )

        plt.scatter(
            data_X.squeeze(-1)[test_mask[:, p]], test_Y[:, p][test_mask[:, p]],
            marker="o", color="red", label="test data", s=15
        )

        # Set x-axis ticks and labels
        plt.xticks(x_ticks, x_labels, rotation=0)

        # Set y-axis to show only 3 ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        plt.xlabel("Time")
        plt.ylabel(col_name_dict[p])
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if exp_result_folder is not None:
        plt.savefig(f"{exp_result_folder}/eeg.pdf")
    plt.close()

    # Save all the necessary information for making this plot again later ...
    with h5py.File(f'{exp_result_folder}/plot_again_info_dict.h5', 'w') as f:
        f.create_dataset('plot_X', data=denser_X.cpu().numpy())
        f.create_dataset('plot_means', data=plot_pred_means.cpu().numpy())
        f.create_dataset('pred_vars', data=plot_pred_vars.cpu().numpy())
        f.create_dataset('scatter_all_X', data=data_X.cpu().numpy())
        f.create_dataset('scatter_train_mask', data=train_mask.cpu().numpy())
        f.create_dataset('scatter_train_Y', data=train_Y.cpu().numpy())
        f.create_dataset('scatter_test_mask', data=test_mask.cpu().numpy())
        f.create_dataset('scatter_test_Y', data=test_Y.cpu().numpy())