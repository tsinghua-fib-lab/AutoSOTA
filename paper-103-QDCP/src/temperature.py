import cvxpy as cx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def tune_temp(
    logits,
    labels,
    binary_search=True,
    lower=0.2,
    upper=5.0,
    eps=0.0001,
):
    logits = np.array(logits)

    if binary_search:
        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        t_guess = torch.FloatTensor([0.5 * (lower + upper)]).requires_grad_()

        while upper - lower > eps:
            if (
                torch.autograd.grad(
                    F.cross_entropy(logits / t_guess, labels),
                    t_guess,
                )[0] > 0
            ):
                upper = 0.5 * (lower + upper)
            else:
                lower = 0.5 * (lower + upper)
            t_guess = t_guess * 0 + 0.5 * (lower + upper)

        t = min(
            [lower, 0.5 * (lower + upper), upper],
            key=lambda x: float(F.cross_entropy(logits / x, labels)),
        )
    else:

        set_size = np.array(logits).shape[0]

        t = cx.Variable()

        expr = sum(
            (
                cx.Minimize(cx.log_sum_exp(logits[i, :] * t) - logits[i, labels[i]] * t)
                for i in range(set_size)
            )
        )
        p = cx.Problem(expr, [lower <= t, t <= upper])

        p.solve()  # p.solve(solver=cx.SCS)
        t = 1 / t.value

    return t


def calc_bins(scores, targets, num_bins=30):
    # Assign each prediction to a bin
    bins = np.linspace(0.01, 1, num_bins)
    binned = np.digitize(scores, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(scores[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (targets[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (scores[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_calibration_metrics(scores, targets, num_bins):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(scores, targets, num_bins=num_bins)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def temp_scale(logits, labels, plot=True):
    temperature = torch.nn.Parameter(
        torch.ones(1)
    )  # .to('cuda' if torch.cuda.is_available() else 'cpu')
    # print(temperature.is_leaf)
    criterion = torch.nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = torch.optim.LBFGS(
        [temperature], lr=0.001, max_iter=10000, line_search_fn="strong_wolfe"
    )

    temps = []
    losses = []

    def _eval():
        loss = criterion(torch.div(logits, temperature), labels)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss.item())
        return loss

    optimizer.step(_eval)

    if plot:
        print("Final T_scaling factor: {:.2f}".format(temperature.item()))

    if plot:
        plt.figure(figsize=(9, 2))
        plt.subplot(121)
        plt.plot(list(range(len(temps))), temps)

        plt.subplot(122)
        plt.plot(list(range(len(losses))), losses)
        plt.show()
    return temperature.detach()


def draw_reliability_graph(logits, targets, use_temp_scale=True, plot=False, title=None, num_bins=10, save=None, fontsize=16):
    if use_temp_scale:
        T = temp_scale(logits, targets, plot=plot)
        temp_scores = torch.softmax(logits / T, 1)

    scores = torch.softmax(logits, 1)
    targets = torch.nn.functional.one_hot(targets)
    
        
    ECE, MCE = get_calibration_metrics(scores, targets, num_bins=num_bins)
    bins, _, bin_accs, _, _ = calc_bins(scores, targets)
    if use_temp_scale:
        temp_ECE, temp_MCE = get_calibration_metrics(temp_scores, targets, num_bins=num_bins)
        temp_bins, _, temp_bin_accs, _, _ = calc_bins(temp_scores, targets)

    fig, ax = plt.subplots(ncols=2, figsize=(4 * (2 if use_temp_scale else 1), 4))
    if title is not None:
        fig.suptitle(title, fontsize=24, x=0.53, y=0.97)

    # x/y limits
    ax[0].set_xlim(0, 1.02)
    ax[0].set_ylim(0, 1)
    if use_temp_scale:
        ax[1].set_xlim(0, 1.02)
        ax[1].set_ylim(0, 1)

    # x/y labels
    ax[0].set_xlabel('Confidence', fontsize=fontsize)
    ax[0].set_ylabel('Accuracy', fontsize=fontsize)
    if use_temp_scale:
        ax[1].set_xlabel('Confidence', fontsize=fontsize)
        ax[1].set_ylabel('Accuracy', fontsize=fontsize)

    # Create grid
    # ax.set_axisbelow(True)
    # ax.grid(color='gray', linestyle='dashed')

    # Error bars
    ax[0].bar(
        bins, bins, width=0.032, alpha=0.3, edgecolor="black", color="r", hatch="\\"
    )
    if use_temp_scale:
        ax[1].bar(
            temp_bins,
            temp_bins,
            width=0.032,
            alpha=0.3,
            edgecolor="black",
            color="r",
            hatch="\\",
        )

    # Draw bars and identity line
    ax[0].bar(bins, bin_accs, width=0.032, alpha=1, edgecolor='black', color='C0')
    ax[0].plot([0,1],[0,1], '--', color='gray', linewidth=2)
    if use_temp_scale:
        ax[1].bar(temp_bins, temp_bin_accs, width=0.032, alpha=1, edgecolor='black', color='C0')
        ax[1].plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    ax[0].set_aspect("equal", adjustable="box")
    if use_temp_scale:
        ax[1].set_aspect("equal", adjustable="box")

    # ECE and MCE legend
    # ECE_patch = mpatches.Patch(color='C1', label='ECE = {:.1f}%'.format(ECE*100),)
    MCE_patch = mpatches.Patch(color='C2', label='MCE = {:.1f}%'.format(MCE*100),)
    # ax[0].legend(handles=[ECE_patch, MCE_patch])
    ax[0].text(0.40, 0.9, f'MCE = {MCE:.0%}', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    if use_temp_scale:
        # temp_ECE_patch = mpatches.Patch(color='C1', label='ECE = {:.1f}%'.format(temp_ECE*100), )
        temp_MCE_patch = mpatches.Patch(color='C2', label='MCE = {:.1f}%'.format(temp_MCE*100), )
        # ax[1].legend(handles=[temp_ECE_patch, temp_MCE_patch])
        ax[1].text(0.40, 0.9, f'MCE = {temp_MCE:.0%}', horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
        
    ax[0].set_title('Before scaling', fontsize=fontsize)
    if use_temp_scale:
        ax[1].set_title('After scaling', fontsize=fontsize)

    ax[0].xaxis.set_tick_params(labelsize=fontsize-2, pad=8)
    ax[0].yaxis.set_tick_params(labelsize=fontsize-2, pad=8)
    ax[1].xaxis.set_tick_params(labelsize=fontsize-2, pad=8)
    ax[1].yaxis.set_tick_params(labelsize=fontsize-2, pad=8)
    
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def client_temp_scale(
    experiments,
    clients_class_map=None,
    use_three_partition_label=False,
    val_df=None,
    test_df=None,
):
    if clients_class_map is None and val_df is not None and test_df is not None:
        if use_three_partition_label:
            partition = "three_partition_label"
        else:
            partition = "aggregated_fitzpatrick_scale"
        val_index_map = {
            str(part): (val_df[partition] == part).values
            for part in sorted(val_df[partition].unique())
        }
        test_index_map = {
            str(part): (test_df[partition] == part).values
            for part in sorted(test_df[partition].unique())
        }
    else:
        # val_index_map = {
        #     k: sum(experiments["tct"]["val_targets"] == k for k in v).bool()
        #     for k, v in clients_class_map.items()
        # }
        # test_index_map = {
        #     k: sum(experiments["tct"]["test_targets"] == k for k in v).bool()
        #     for k, v in clients_class_map.items()
        # }
        val_index_map = {
            k: sum(experiments["fedavg"]["val_targets"] == k for k in v).bool()
            for k, v in clients_class_map.items()
        }
        test_index_map = {
            k: sum(experiments["fedavg"]["test_targets"] == k for k in v).bool()
            for k, v in clients_class_map.items()
        }

    for k, v in experiments.items():
        val_logits = v["val_scores"]
        test_logits = v["test_scores"]
        val_targets = v["val_targets"]
        test_targets = v["test_targets"]

        clients_temp = {}
        for client, index in val_index_map.items():
            client_val_logits = val_logits[index]
            client_val_targets = val_targets[index]
            temp = temp_scale(client_val_logits, client_val_targets, plot=False)
            clients_temp[client] = temp
        # print(k.center(30, '-'))
        mean_temp = torch.mean(torch.cat(list(clients_temp.values())))

        experiments[k]["temp_val_scores"] = torch.softmax(val_logits / mean_temp, 1)
        experiments[k]["temp_test_scores"] = torch.softmax(test_logits / mean_temp, 1)
