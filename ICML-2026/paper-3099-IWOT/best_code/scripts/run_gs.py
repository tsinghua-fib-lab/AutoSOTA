import torch
import matplotlib.pyplot as plt

import iter_ls


def gen_gauss_covariates(mean, var, num_samples):
    if var.numel() == 1:
        return mean + torch.sqrt(var) * torch.randn(num_samples)
    dim = mean.shape[0]
    return mean[:, torch.newaxis] + torch.linalg.cholesky(var) @ torch.randn(
        dim, num_samples
    )


def check_gradual_shift_in_linear_classification():
    """
    TODO: Test also weighted WRR whenever both source and target cannot be both seperated!
    TODO: Should we feed in the whole set of source points to the clusterer?
    """

    torch.manual_seed(0)
    num_target = 100
    num_source = 100
    dim = 2

    mean1_source = 2 * torch.ones(dim)
    mean2_source = -2 * torch.ones(dim)
    var1_source = 0.4 * torch.eye(dim)
    var2_source = var1_source
    x1_source = gen_gauss_covariates(
        mean1_source, var1_source, num_samples=num_source // 2
    ).T
    x2_source = gen_gauss_covariates(
        mean2_source, var2_source, num_samples=num_source // 2
    ).T
    y1_source = torch.ones(num_source // 2)
    y2_source = -torch.ones(num_source // 2)

    x_source = torch.vstack((x1_source, x2_source))
    X_source = torch.hstack((torch.ones(num_source, 1), x_source))
    y_source = torch.hstack((y1_source, y2_source))
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source_ls = X_source @ theta_ls

    # Check source accuracy!
    loss = torch.nn.CrossEntropyLoss()
    print(f"CE loss: {loss(pred_source_ls, y_source)}")
    y_hat_source_ls = 2 * (pred_source_ls > 0) - 1
    acc = torch.sum(y_hat_source_ls == y_source) / num_source
    print(f"Accuracy % = {acc}")

    # generate gradual shift
    num_shifts = 5
    shift_direction_1 = 10 * torch.tensor([0.0, 1.0])
    shift_direction_2 = 10 * torch.tensor([0.0, 1.0])
    var1_target = var1_source
    var2_target = var2_source
    x_target = torch.zeros(num_target // 2, 2, 2)
    num_gen = num_target // num_shifts
    for i in range(num_shifts):
        mean1_target = mean1_source - (i / num_shifts) * shift_direction_1
        mean2_target = mean2_source + (i / num_shifts) * shift_direction_2
        x_target[i * num_gen // 2 : (i + 1) * num_gen // 2, :, 0] = (
            gen_gauss_covariates(mean1_target, var1_target, num_samples=num_gen // 2).T
        )
        x_target[i * num_gen // 2 : (i + 1) * num_gen // 2, :, 1] = (
            gen_gauss_covariates(mean2_target, var2_target, num_samples=num_gen // 2).T
        )
    x_target = torch.vstack((x_target[:, :, 0], x_target[:, :, 1]))
    y1_target = torch.ones(num_target // 2)
    y2_target = -torch.ones(num_target // 2)
    y_target = torch.hstack((y1_target, y2_target))
    X_target = torch.hstack((torch.ones(num_target, 1), x_target))

    # Check target accuracy, it should be poor for LS!
    pred_target_ls = X_target @ theta_ls
    print(f"CE loss: {loss(pred_target_ls, y_target)}")
    y_hat_target_ls = 2 * (pred_target_ls > 0) - 1
    acc = torch.sum(y_hat_target_ls == y_target) / num_target
    print(f"Accuracy % = {acc}")

    # Viz source and target inputs!
    plt.scatter(x1_source[:, 0], x1_source[:, 1], s=20, c="b", marker="*")
    plt.scatter(x2_source[:, 0], x2_source[:, 1], s=20, c="b", marker="^")
    plt.scatter(x_target[:, 0], x_target[:, 1], s=20, c="r", marker="+")

    # Draw the decision boundary where predictions are zero!
    x_plot = torch.arange(start=-5, end=5, step=0.05)
    y_plot = (-theta_ls[0] - theta_ls[1] * x_plot) / theta_ls[2]
    plt.plot(x_plot, y_plot, c="b", linestyle="--")

    # Test WRR-based optimizer
    theta_wrr, pred_target_wrr = iter_ls.optim_wrr_iter_ls(
        X_source, X_target, y_source, thresh=1e-4
    )
    # Interestingly, the SGD based optimizer does not work well here!
    # Seems to get stuck at local optima?
    # print('Trying WRR opt with SGD...')
    # theta_wrr, pred_target_wrr = optim_wrr_sgd(X_source, X_target, y_source, thresh=1e-4, theta0=theta_ls)

    # Draw the decision boundary where predictions are zero!
    x_plot = torch.arange(start=-1, end=1, step=0.05)
    y_plot = (-theta_wrr[0] - theta_wrr[1] * x_plot) / theta_wrr[2]
    plt.plot(x_plot, y_plot, c="r", linestyle="--")

    # Check target accuracy, it should be good for WRR!
    print(f"CE loss: {loss(pred_target_wrr, y_target)}")
    y_hat_target_wrr = 2 * (pred_target_wrr > 0) - 1
    acc = torch.sum(y_hat_target_wrr == y_target) / num_target
    print(f"Accuracy % = {acc}")

    # Test linkage-clustering based pseudolabeler
    theta_pl, pred_target_pl = iter_ls.optim_linkage(
        X_source,
        X_target,
        y_source,
        thresh=1e-2,
        theta0=theta_ls,
        method="single",
        soft=True,
    )
    # theta_pl, pred_target_pl = iter_ls.optim_input_linkage(X_source, X_target, y_source, method='single', soft=False)
    print(f"CE loss: {loss(pred_target_pl, y_target)}")
    y_hat_target_pl = 2 * (pred_target_pl > 0) - 1
    acc = torch.sum(y_hat_target_pl == y_target) / num_target
    print(f"Accuracy % = {acc}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    check_gradual_shift_in_linear_classification()
