import torch
import matplotlib.pyplot as plt

import linkage


@torch.no_grad()
def calc_gradual_shift(
    loss_fun, pred_source, pred_target, y_source, y_target, num_classes, margin_m
):

    source_means = torch.zeros(num_classes, num_classes)
    sorted_dists = []
    for i in range(num_classes):
        source_cond = pred_source[torch.argmax(y_source, dim=1) == i]
        source_means[i] = torch.mean(source_cond, dim=0)
        target_cond = pred_target[torch.argmax(y_target, dim=1) == i]
        dists = torch.norm(target_cond - source_means[i], dim=1)
        sorted_dists.append(torch.sort(dists, descending=False)[0])

    eta = 0.1
    label_counts_source = torch.sum(y_source, dim=0)
    label_counts_target = torch.sum(y_target, dim=0)
    rho = torch.max(label_counts_target / label_counts_source)
    lambda_est_cond = torch.zeros(num_classes)

    # Modeling conditional distances using a log-normal distribution
    for i in range(num_classes):
        s, mu = torch.std_mean(torch.log(sorted_dists[i]))
        mean = torch.exp(mu + 0.5 * s * s)
        std = torch.sqrt((torch.exp(s * s) - 1) * torch.exp(2 * mu + s * s))
        entropy = torch.log(s * torch.exp(mu))
        lambda_est_cond[i] = (
            -entropy * torch.pow((mean + 2 * std) / margin_m, 2) / ((1 - eta) * rho)
        )
    # print(f"Estimated lambda inverse per conditional: {lambda_est_inv_cond}")
    num_source = y_source.shape[0]
    lambda_est = torch.sum(lambda_est_cond * (label_counts_source / num_source))
    return lambda_est

    ##### PLOT DISTANCES FROM THE SOURCE CONDITIONAL MEANS
    # cond_dist, correct_labeled = compute_cond_dist(
    # pred_target, y_target, source_means, num_classes
    # )
    # plot_distances(cond_dist, correct_labeled, num_classes)
    # plt.show()

    ##### LINKAGE METHOD TO CHECK IF THERE IS GRADUAL SHIFT
    """
    Z = linkage.compute_cluster(pred_source_cond, pred_target, method="single")
    # linkage.plot_cluster(Z, num_targets, num_classes)
    y_pseudo = linkage.compute_pseudolabels(Z, num_targets, num_classes, soft=False)
    pseudo_loss = loss_fun(y_pseudo, y_target)
    print(f"Linkage_pseudo_loss: {pseudo_loss.item()}")
    pseudo_acc = (
        torch.sum(torch.argmax(y_pseudo, dim=1) == torch.argmax(y_target, dim=1))
        / num_targets
    )
    print(f"Linkage acc: {pseudo_acc}")
    """


def compute_cond_dist(pred_target, y_target, source_means, num_classes):
    # Note that we don't use dist[i][j], i \neq j for now!
    dists = {i: {j: [] for j in range(num_classes)} for i in range(num_classes)}
    correct = {i: [] for i in range(num_classes)}
    pred_labels = pred_target.argmax(dim=1, keepdim=True)
    for i in range(num_classes):
        mask = torch.argmax(y_target, dim=1) == i
        cond = pred_target[mask]
        for j in range(num_classes):
            dists[i][j].append(torch.norm(cond - source_means[j], dim=1))
        correct[i].append(pred_labels[mask] == i)

    for i in range(num_classes):
        for j in range(num_classes):
            dists[i][j] = torch.concatenate(dists[i][j], axis=0)
        correct[i] = torch.concatenate(correct[i], axis=0)
    return dists, correct


def plot_distances(dist, correct_labeled, num_classes):
    plt.figure(figsize=(10, 8))
    plt.title("Distance of target conditionals to source class mean")
    cmap = plt.get_cmap("tab10")
    for i in range(num_classes):
        sort_dist, sort_idx = torch.sort(dist[i][i], descending=False)
        x_vec = torch.range(1, len(sort_idx))
        # sizes = [50 if value == 0 else 10 for value in correct_labeled[i][sort_idx]]
        # plt.scatter(x_vec, sort_dist, s=sizes, label=str(i))
        mask = correct_labeled[i][sort_idx].squeeze()
        plt.plot(x_vec, sort_dist, "-", label=str(i), c=cmap(i))
        plt.plot(x_vec[mask], sort_dist[mask], "o", c=cmap(i))
        plt.plot(x_vec[~mask], sort_dist[~mask], "x", c=cmap(i))
    plt.ylabel("Euclidean distance")
    plt.xlabel("Target class conditional index")
    plt.legend()

    folder_name = os.path.joint("results", "debug")
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(
        os.path.join(folder_name, "gradual_shift"),
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
