import math
import numpy as np
import torch
import ot
import geomloss
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import matplotlib
import matplotlib.pyplot as plt

import iter_ls
from importance_weighting import apply_importance_weighting as apply_iw

matplotlib.use(backend="QtAgg", force=True)


def regression_fun_Nd(x):
    # Watch out for values close to 0
    x_norm = torch.linalg.norm(x, dim=1)
    return torch.sin(torch.pi * x_norm) / (torch.pi * x_norm)


def regression_fun_1d(x):
    # Watch out for values close to 0
    return torch.sin(torch.pi * x) / (torch.pi * x)


def gen_gauss_covariates(mean, var, num_samples):
    if var.numel() == 1:
        return mean + torch.sqrt(var) * torch.randn(num_samples)
    dim = mean.shape[0]
    return mean[:, torch.newaxis] + torch.linalg.cholesky(var) @ torch.randn(
        dim, num_samples
    )


def gen_labels(fun, x, noise_var):
    num_pts = x.shape[0]
    y_act = fun(x)
    y_noise = y_act + math.sqrt(noise_var) * torch.randn(num_pts)
    return y_noise, y_act


def opt_weighted_wrr_rule_iter_ls(
    X_source, X_target, y_source, thresh, epsilon, theta0=None
):
    num_target = X_target.shape[0]
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = theta0
    else:
        theta = torch.zeros(dim)
    X_pq = torch.cat((X_source, X_target))
    loss_fun = torch.nn.MSELoss(reduction="none")
    prev_loss = 0.0
    total_loss = 1.0
    # TODO: Change the scale back to 0.5!
    scale_bound = 1.0
    I_t = torch.eye(num_target) / num_target
    add_coupling_der = False

    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        Gamma, wrr_value = calc_weighted_wrr(
            pred_source, pred_target, y_source, epsilon, scale_bound
        )
        losses = loss_fun(pred_source, y_source)
        costs = (
            torch.cdist(pred_source[:, None], pred_target[:, None], p=2) ** 2
            + losses[:, None]
        )
        if add_coupling_der is True:
            mat = Gamma * costs - num_target * Gamma * torch.sum(Gamma * costs, dim=0)
            mat = Gamma - mat / epsilon
        else:
            mat = Gamma
        w = torch.sum(mat, dim=1)
        W = torch.diag(w)
        X_pq = torch.cat((X_source, X_target))
        # print(f"Optimized weights: {w}, max: {torch.max(w)}, min: {torch.min(w)}")
        M_pq = torch.cat(
            (
                torch.cat((W, -mat), dim=1),
                torch.cat((-mat.T, I_t), dim=1),
            ),
            dim=0,
        )
        R = X_pq.T @ M_pq @ X_pq
        theta = torch.linalg.solve(
            X_source.T @ W @ X_source + R, X_source.T @ W @ y_source
        )
        total_loss = wrr_value
        print(
            f"Total loss: {total_loss}, source loss: {torch.mean(losses)}, theta: {theta.detach().numpy()}"
        )
    return theta, X_target @ theta, w


def opt_wrr_rule_iter_abs_dev(theta0, X_source, X_target, y_source, thresh):
    dim = len(theta0)
    # theta = theta0.clone().detach()
    theta = torch.zeros(dim)
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    # w_source = torch.ones(num_source) / num_source
    # w_target = torch.ones(num_target) / num_target
    prev_loss = 0.0
    total_loss = 1.0

    # Prepare linear program
    num_gamma = num_source * num_target
    c1 = torch.ones(num_source) / num_source
    c2 = torch.ones(num_gamma)
    c3 = torch.zeros(dim)
    c = torch.hstack((c1, c2, c3))
    X1 = torch.tile(X_source, (1, num_target))
    X2 = torch.tile(X_target.flatten(), (num_source, 1))
    A1 = torch.hstack(
        (-torch.eye(num_source), torch.zeros(num_source, num_gamma), -X_source)
    )
    A2 = torch.hstack(
        (-torch.eye(num_source), torch.zeros(num_source, num_gamma), X_source)
    )
    b_ub = torch.hstack((-y_source, y_source, torch.zeros(2 * num_gamma)))
    lb = torch.zeros((num_source + num_gamma + dim, 1))
    lb[-dim:, 0] = torch.nan
    ub = torch.nan * torch.ones((num_source + num_gamma + dim, 1))
    bounds = torch.hstack((lb, ub))

    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        cost_mat = ot.utils.euclidean_distances(
            pred_source[:, torch.newaxis], pred_target[:, torch.newaxis], squared=False
        )
        Gamma = ot.emd_1d(pred_source, pred_target)
        # Gamma = ot.sinkhorn(w_source, w_target, cost_mat, reg=1e-1)
        # print(Gamma)
        ot_cost = torch.sum(Gamma * cost_mat)

        print("======== Least Absolute Deviations ========")
        M = torch.kron(Gamma, torch.ones(dim)) * (X1 - X2)
        M = M.reshape((num_source * num_target, dim))
        A3 = torch.hstack(
            (torch.zeros(num_gamma, num_source), -torch.eye(num_gamma), M)
        )
        A4 = torch.hstack(
            (torch.zeros(num_gamma, num_source), -torch.eye(num_gamma), -M)
        )
        A_ub = torch.vstack((A1, A2, A3, A4))
        res = linprog(c.numpy(), A_ub.numpy(), b_ub.numpy(), bounds=bounds.numpy())
        theta = torch.tensor(res.x[-dim:])
        pred_abs_dev = X_source @ theta

        source_loss = torch.sum(torch.abs(pred_abs_dev - y_source)) / num_source
        total_loss = source_loss + ot_cost
        print(
            f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_cost}, theta: {theta.detach().numpy()}"
        )
    return (theta, X_target @ theta)


def test_that_wrr_rule_can_be_optimized_in_linear_regression():
    """
    Check WRR optimization in linear regression using
    (a) SGD and 1d optimal transport [squared distance]
    (b) Iterative Least Squares (no derivative of OT coupling computed!) [squared distance]
    (c) Iterative Absolute Deviations [Euclidean distance] using linear program for both OT and regression

    - Try in multiple dim
    - Check linear program in iterative absolute deviations
    """

    torch.manual_seed(0)
    num_source = 100
    num_target = 100
    dim = 1
    fun = regression_fun_Nd
    if dim == 1:
        fun = regression_fun_1d

    s2_noise = torch.tensor([0.01])
    mean_source = -torch.ones(dim)
    var_source = 0.8 * torch.eye(dim)
    x_source = gen_gauss_covariates(mean_source, var_source, num_samples=num_source)[0]
    y_source, _ = gen_labels(fun, x_source, s2_noise)

    X_source = torch.vstack((torch.ones((num_source)), x_source)).T
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source_ls = X_source @ theta_ls

    # generate test samples
    mean_target = torch.ones(dim)
    var_target = 0.8 * torch.eye(dim)
    x_target = gen_gauss_covariates(mean_target, var_target, num_samples=num_target)[0]
    X_target = torch.vstack((torch.ones(num_target), x_target)).T
    pred_target_ls = X_target @ theta_ls

    print("======== SGD ========")
    theta_wrr_sgd, pred_target_wrr_sgd = iter_ls.opt_wrr_sgd(
        X_source, X_target, y_source, thresh=1e-4, theta0=None
    )
    print("======= Iterative LS =======")
    theta_wrr_iter_ls, pred_target_wrr_iter_ls = iter_ls.opt_wrr_iter_ls(
        X_source, X_target, y_source, thresh=1e-4, theta0=None
    )
    # print("======= Iterative Absolute Deviations (L1) =======")
    # theta_wrr_abs_dev, pred_target_wrr_abs_dev = opt_wrr_rule_iter_abs_dev(
    #     theta_ls, X_source, X_target, y_source, thresh=1e-4
    # )

    y_target, _ = gen_labels(fun, x_target, s2_noise)
    loss = torch.nn.MSELoss()
    risk_source_ls = loss(pred_source_ls, y_source)
    risk_target_ls = loss(pred_target_ls, y_target)
    risk_source_wrr_sgd = loss(X_source @ theta_wrr_sgd, y_source)
    risk_target_wrr_sgd = loss(pred_target_wrr_sgd, y_target)
    risk_source_wrr_iter_ls = loss(X_source @ theta_wrr_iter_ls, y_source)
    risk_target_wrr_iter_ls = loss(pred_target_wrr_iter_ls, y_target)
    # risk_source_wrr_abs_dev = loss(X_source @ theta_wrr_abs_dev, y_source)
    # risk_target_wrr_abs_dev = loss(pred_target_wrr_abs_dev, y_target)

    """
    print("Parameters:")
    print(
        f"LS: {theta_ls}, WRR-sgd: {theta_wrr_sgd}, WRR-abs-dev: {theta_wrr_abs_dev}, WRR-iter-ls: {theta_wrr_iter_ls}"
    )
    """

    print(f"LS Source risk: {risk_source_ls}")
    print(f"LS Target risk: {risk_target_ls}")
    print(f"WRR-sgd Source risk: {risk_source_wrr_sgd}")
    print(f"WRR-sgd Target risk: {risk_target_wrr_sgd}")
    print(f"WRR-iter-ls Source risk: {risk_source_wrr_iter_ls}")
    print(f"WRR-iter-ls Target risk: {risk_target_wrr_iter_ls}")
    # print(f"WRR-abs-dev Source risk: {risk_source_wrr_abs_dev}")
    # print(f"WRR-abs-dev Target risk: {risk_target_wrr_abs_dev}")

    plt.scatter(x_source, y_source, c="red", s=15)
    plt.scatter(x_source, pred_source_ls, c="blue", s=15)
    plt.scatter(x_target, pred_target_wrr_sgd.detach().numpy(), c="brown", s=15)
    plt.scatter(x_target, pred_target_wrr_iter_ls.detach().numpy(), c="black", s=15)
    plt.legend(["noisy labels", "lstsq", "WRR-sgd", "WRR-iter-ls"])
    plt.show()


def test_that_dual_variables_in_OT_correspond_to_effect_of_IW():
    num_x = 10
    num_y = 10
    dim = 2
    torch.manual_seed(1)
    x = torch.randn(num_x, dim)
    ot_dist = torch.randn(num_y, dim)
    a = torch.rand(num_x)
    a /= torch.sum(a)
    a.requires_grad = True
    b = torch.rand(num_y)
    b /= torch.sum(b)
    p = 1
    blur = 0.001
    # ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    # ot_cost = ot_loss(a, x, b, ot_dist)
    # ot_cost.backward()

    cost_mat = torch.tensor(cdist(x, ot_dist, metric="euclidean"))
    ot_cost = ot.emd2(a, b, cost_mat)
    ot_cost.backward()

    # Modify p randomly using a vector theta that sums up to 1
    da = torch.randn(num_x)
    da -= da.mean()
    lamb = 0.001
    a2 = a + lamb * da

    diff_auto = torch.dot(a.grad, a2 - a)
    print(f"Auto-diff of OT costs: {diff_auto}")

    # Get dual variables using geomloss
    potentials = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True)
    f, _ = potentials(a, x, b, ot_dist)
    f[0] -= f[0].mean()
    print(f"Potential from geomloss: {f}")

    # Check dual variables using POT library
    # G, log = ot.sinkhorn(a, b, cost_mat, reg=0.001, log=True)
    # f = log['u']
    # f = torch.log(f)
    # f -= f.mean()
    # print(f"Potential from POT: {f}")

    # Get f and dot with theta to calculate the derivative
    diff_deriv = torch.dot(f[0], a2 - a)

    diff_deriv = diff_deriv.detach().numpy()
    diff_auto = diff_auto.detach().numpy()
    print(f"Potential based derivative of OT costs: {diff_deriv}")

    ot_cost2 = ot.emd2(a2, b, cost_mat)
    # ot_cost2 = ot_loss(a2, x, b, ot_dist)
    diff_exact = ot_cost2 - ot_cost
    diff_exact = diff_exact.detach().numpy()
    print(f"Exact difference of OT costs: {diff_exact}")

    # TODO: If we compute the total derivative it should match with diff_auto!
    # assert np.allclose(diff_deriv, diff_auto, rtol=1e-2) is True
    assert np.allclose(diff_auto, diff_exact, rtol=1e-2) is True


def compute_weighted_wrr(
    w_source, loss_fun, pred_source, pred_target, y_source, num_iter, blur
):
    # loss matrix
    num_target = pred_target.shape[0]
    w_target = torch.ones(num_target) / num_target

    # Now minimize the OT using weights
    w_source.requires_grad = True
    opt = torch.optim.SGD([w_source], lr=1e-3, momentum=0.98)
    potentials = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur, potentials=True)
    losses = loss_fun(pred_source[:, 0], y_source)
    for i in range(num_iter):
        f, g = potentials(w_source, pred_source, w_target, pred_target)
        f0 = f - f.mean()
        w_source.grad = f0[0] + losses
        opt.step()
        opt.zero_grad()
        w_source.requires_grad = False
        w_source[w_source < 0.0] = 0.0
        w_source /= w_source.sum()
        w_source.requires_grad = True
        source_loss = losses @ w_source
        ot_dist = f[0] @ w_source + g @ w_target
    return source_loss, ot_dist


def compute_weighted_ot(x_source, x_target, num_iter, blur):
    # loss matrix
    num_source = x_source.shape[0]
    num_target = x_target.shape[0]
    a = torch.ones(num_source) / num_source
    b = torch.ones(num_target) / num_target
    ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur)
    ot_cost = ot_loss(a, x_source, b, x_target)
    potentials = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur, potentials=True)
    a.requires_grad = True
    print(f"Standard OT cost: {ot_cost}")

    # Now minimize the OT using weights
    opt = torch.optim.SGD([a], lr=1e-3, momentum=0.98)
    for i in range(num_iter):
        f, _ = potentials(a, x_source, b, x_target)
        f -= f.mean()
        a.grad = f[0]
        opt.step()
        opt.zero_grad()
        a.requires_grad = False
        a[a < 0.0] = 0.0
        a /= a.sum()
        a.requires_grad = True
        ot_cost_weighted = ot_loss(a, x_source, b, x_target)
    return ot_cost_weighted, a


def test_that_weighted_OT_can_be_computed():
    # Create two distributions
    # using Gaussians such that
    # the W1-distance is significant
    # but the weighted distance converges to zero
    # as the number of samples increase!
    torch.manual_seed(1)
    dim = 2
    num_source = 60
    num_target = 50

    mu_source = torch.rand(1, dim)
    cov_source = 0.1 * torch.eye(dim)

    mu_target = torch.rand(1, dim)
    mat_rand = torch.randn(dim, dim)
    cov_target = 0.1 * mat_rand @ mat_rand.T

    x_source = (
        mu_source.repeat(num_source, 1)
        + torch.randn(num_source, dim) @ torch.linalg.cholesky_ex(cov_source)[0]
    )
    x_target = (
        mu_target.repeat(num_target, 1)
        + torch.randn(num_target, dim) @ torch.linalg.cholesky_ex(cov_target)[0]
    )

    ot_cost_weighted, w = compute_weighted_ot(
        x_source, x_target, num_iter=100, blur=1e-4
    )
    # Check explicit solution
    cost_mat = torch.tensor(cdist(x_source, x_target, metric="euclidean"))
    epsilon = 1e-4
    q = torch.ones(num_target) / num_target
    ot_mat = torch.softmax(-cost_mat / epsilon, dim=0)
    ot_mat /= torch.sum(ot_mat, dim=0)
    ot_mat *= q[None, :]
    # Implicitly optimized weights
    # w = torch.sum(ot_mat, dim=1)
    ot_cost_alt = torch.sum(ot_mat * cost_mat)

    print(f"Optimized solution: {ot_cost_weighted}, explicit solution: {ot_cost_alt}")


def test_that_weighted_wrr_can_be_computed():
    # Repeat the above for optimizing gamma and weights w for the WRR cost function
    torch.manual_seed(0)
    dim = 10

    # generate source inputs and noisy labels
    num_source = 100
    mu_source = torch.rand(1, dim)
    cov_source = 0.1 * torch.eye(dim)
    x_source = (
        mu_source.repeat(num_source, 1)
        + torch.randn(num_source, dim) @ torch.linalg.cholesky_ex(cov_source)[0]
    )
    y_source, _ = gen_labels(regression_fun_Nd, x_source, noise_var=1e-2)

    X_source = torch.hstack((x_source, torch.ones(num_source)[:, None]))
    theta_ls, _, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source = X_source @ theta_ls

    # generate target inputs
    num_target = 50
    mat_rand = torch.randn(dim, dim)
    cov_target = 0.1 * mat_rand @ mat_rand.T
    mu_target = torch.rand(1, dim)
    x_target = (
        mu_target.repeat(num_target, 1)
        + torch.randn(num_target, dim) @ torch.linalg.cholesky_ex(cov_target)[0]
    )
    X_target = torch.hstack((x_target, torch.ones(num_target)[:, None]))
    pred_target = X_target @ theta_ls

    # Optimize solution
    w_source = torch.ones(num_source) / num_source
    loss_fun = torch.nn.MSELoss(reduction="none")
    source_loss_w, ot_dist_w = compute_weighted_wrr(
        w_source,
        loss_fun,
        pred_source[:, torch.newaxis],
        pred_target[:, torch.newaxis],
        y_source,
        num_iter=200,
        blur=1e-4,
    )
    wrr_weighted = source_loss_w + ot_dist_w

    _, wrr_alt = calc_weighted_wrr(pred_source, pred_target, y_source, epsilon=1e-4)
    print(f"Optimized solution: {wrr_weighted}, explicit solution: {wrr_alt}")


def calc_weighted_wrr(
    pred_source, pred_target, y_source, epsilon, scale, unbalanced=False
):
    loss_fun = torch.nn.MSELoss(reduction="none")
    losses = loss_fun(pred_source, y_source)
    num_target = pred_target.shape[0]
    # Calculate explicit solution
    cost_mat = (
        torch.cdist(pred_source[:, torch.newaxis], pred_target[:, torch.newaxis], p=2)
        ** 2
    )
    # losses = loss_fun(pred_source, y_source)
    total_mat = cost_mat + (losses[:, None] / scale)

    num_source = pred_source.shape[0]
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target

    if unbalanced:
        ot_mat = ot.unbalanced.mm_unbalanced(
            w_source, w_target, total_mat, reg_m=(1, 100)
        )
    else:
        # Semi-relaxed
        ot_mat = torch.softmax(-total_mat / epsilon, dim=0) / num_target
    # Implicitly optimized weights
    # w = torch.sum(ot_mat, dim=1)
    cost = torch.sum(ot_mat * total_mat)
    return ot_mat, cost


def compare_weighted_wrr_opt_to_iw_in_linear_regression():
    """
    - Try using negative squared distance as the kernel
    - Try using sq. exp. loss for WRR, that corresponds to the chosen squared exp. kernel
    """

    torch.manual_seed(0)
    num_source = 100
    s2_noise = torch.tensor([1 / 100])
    x_source = gen_gauss_covariates(
        mean=torch.tensor([1.0]), var=torch.tensor([1 / 4.0]), num_samples=num_source
    )
    y_source, y_source_clean = gen_labels(regression_fun_1d, x_source, s2_noise)

    X_source = torch.vstack((torch.ones((num_source)), x_source)).T
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)

    # generate test samples
    num_target = 50
    x_target = gen_gauss_covariates(
        mean=torch.tensor([2.0]), var=torch.tensor([1 / 16.0]), num_samples=num_target
    )
    X_target = torch.vstack((torch.ones(num_target), x_target)).T
    pred_target_ls = X_target @ theta_ls

    theta_iw, pred_target_iw, w_iw = apply_iw(
        x_source, x_target, y_source, method="kmm", max_weight=10
    )

    # Unweighted WRR!
    # theta_wrr, pred_target_wrr = opt_wrr_rule_iter_ls(
    #     theta_ls, X_source, X_target, y_source, thresh=1e-4
    # )
    # w_wrr = torch.ones(num_source) / num_source

    theta_wrr, pred_target_wrr, w_wrr = opt_weighted_wrr_rule_iter_ls(
        X_source, X_target, y_source, thresh=1e-4, epsilon=1e-2, theta0=theta_iw
    )

    loss = torch.nn.MSELoss()
    y_target, _ = gen_labels(regression_fun_1d, x_target, s2_noise)
    risk_target_ls = loss(pred_target_ls, y_target)
    risk_target_wrr = loss(pred_target_wrr, y_target)
    risk_target_iw = loss(pred_target_iw, y_target)

    print(f"LS Test risk: {risk_target_ls}")
    print(f"WRR Test risk: {risk_target_wrr}")
    print(f"IW Test risk: {risk_target_iw}")

    theta_oracle, _, _, _ = torch.linalg.lstsq(X_target, y_target)
    pred_oracle = X_target @ theta_oracle
    risk_oracle = loss(pred_oracle, y_target)
    print(f"Oracle risk: {risk_oracle}")

    ls_mesh = torch.linspace(-0.5, 2.5, steps=100)
    X_mesh = torch.vstack((torch.ones(100), ls_mesh)).T
    pred_ls_mesh = X_mesh @ theta_ls
    pred_wrr_mesh = X_mesh @ theta_wrr
    pred_iw_mesh = X_mesh @ theta_iw
    fig = plt.figure(figsize=(8, 5))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax.scatter(x_source, y_source, c="red", s=5)
    ax.scatter(x_source, y_source_clean, c="black", s=15)
    ax.plot(ls_mesh, pred_ls_mesh, linestyle="dashed", c="blue")
    ax.plot(ls_mesh, pred_wrr_mesh, linestyle="dashed", c="brown")
    ax.plot(ls_mesh, pred_iw_mesh, linestyle="dashed", c="purple")
    plt.legend(["noisy labels", "clean labels", "lstsq", "WRR", "IW"])

    # Add histograms
    ax_hist = fig.add_subplot(gs[1], sharex=ax)
    ax_hist.hist(x_source, bins=20, color="gray")
    ax_hist.hist(x_target, bins=20, color="blue")
    ax_hist.scatter(x_source, w_iw, color="red")
    sum_iw = torch.sum(w_iw)
    ax_hist.scatter(x_source, sum_iw * w_wrr, color="black")

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.show()


def compare_ls_solution_to_least_abs_deviations():
    torch.manual_seed(10)
    num_source = 150
    s2_noise = torch.tensor([1 / 16])
    x_source = gen_gauss_covariates(
        mean=torch.tensor([1.0]), var=torch.tensor([1 / 4.0]), num_samples=num_source
    )
    y_source, y_source_clean = gen_labels(regression_fun_1d, x_source, s2_noise)
    # Add a crazy outlier
    y_source[-1] += 10.0

    X_source = torch.vstack((torch.ones((num_source)), x_source)).T
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_ls = X_source @ theta_ls

    print("======== Least Absolute Deviations ========")
    from scipy.optimize import linprog

    c1 = np.ones(num_source)
    dim = 2
    c2 = np.zeros(dim)
    c = np.hstack((c1, c2))
    A1 = np.hstack((-np.eye(num_source), -X_source))
    A2 = np.hstack((-np.eye(num_source), X_source))
    A_ub = np.vstack((A1, A2))
    b_ub = np.hstack((-y_source, y_source))
    lb = np.zeros((num_source + dim, 1))
    lb[-dim:, 0] = None
    ub = np.nan * np.ones((num_source + dim, 1))
    bounds = np.hstack((lb, ub))
    res = linprog(c, A_ub, b_ub, bounds=bounds)
    theta_abs_dev = res.x[-dim:]
    pred_abs_dev = X_source @ theta_abs_dev

    loss = torch.nn.MSELoss()
    risk_ls = loss(pred_ls, y_source_clean)
    risk_abs_dev = loss(pred_abs_dev, y_source_clean)
    print(f"LS source risk: {risk_ls}")
    print(f"Least abs dev source risk: {risk_abs_dev}")


def check_weighted_wrr_derivatives():
    torch.manual_seed(1)
    dim = 2
    num_source = 50
    num_target = 30
    beta = torch.rand(dim)
    beta.requires_grad = True

    # Create source inputs
    var_source = torch.randn(dim, dim)
    var_source = var_source @ var_source.T
    x_source = gen_gauss_covariates(
        mean=torch.ones(dim), var=var_source, num_samples=num_source
    ).T
    pred_source = x_source @ beta
    y_source, _ = gen_labels(regression_fun_Nd, x_source, noise_var=0.01)

    # Create targets
    var_target = torch.randn(dim, dim)
    var_target = var_target @ var_target.T
    x_target = gen_gauss_covariates(
        mean=2 * torch.ones(dim), var=var_target, num_samples=num_target
    ).T
    pred_target = x_target @ beta

    # Calculate OT distance
    eps = 1e-2
    # TODO: When scale is not 1.0 the above does not work
    scale_bound = 1.0
    Gamma, wrr_value = calc_weighted_wrr(
        pred_source, pred_target, y_source, epsilon=eps, scale=scale_bound
    )
    wrr_value.backward()
    print(f"Auto-diff derivative: {beta.grad}")

    # Calculation using 2-for loops (before adding source risk!)
    loss_fun = torch.nn.MSELoss(reduction="none")
    losses = loss_fun(pred_source, y_source)
    costs = (
        torch.cdist(pred_source[:, None], pred_target[:, None], p=2) ** 2
        + (1 / scale_bound) * losses[:, None]
    )
    # Delta makes a small multiplicative correction, coming from the derivative of the diff. coupling!
    Delta = (num_target * torch.sum(Gamma * costs, dim=0) - costs) / eps
    mat = Gamma * (1 + Delta)
    w = torch.sum(mat, dim=1)
    der = torch.zeros(dim)
    for i in range(num_source):
        der += 2 * (x_source[i] @ beta - y_source[i]) * x_source[i] * w[i]
        for j in range(num_target):
            der += (
                2
                * mat[i, j]
                * (
                    torch.outer(x_source[i] - x_target[j], x_source[i] - x_target[j])
                    @ beta
                )
            )
    assert torch.allclose(beta.grad, der)

    # Faster matrix computation of derivative
    loss_fun = torch.nn.MSELoss(reduction="none")
    losses = loss_fun(pred_source, y_source)
    costs = (
        torch.cdist(pred_source[:, None], pred_target[:, None], p=2) ** 2
        + (1 / scale_bound) * losses[:, None]
    )
    Delta = (num_target * torch.sum(Gamma * costs, dim=0) - costs) / eps
    X_p = x_source
    X_pq = torch.cat((x_source, x_target))
    mat = Gamma + Gamma * Delta
    W = torch.diag(torch.sum(mat, dim=1))
    I_t = torch.eye(num_target) / num_target
    M_pq = torch.cat(
        (
            torch.cat((W, -mat), dim=1),
            torch.cat((-mat.T, I_t), dim=1),
        ),
        dim=0,
    )
    der = (
        2 * X_p.T @ W @ X_p @ beta
        + 2 * (X_pq.T @ M_pq @ X_pq) @ beta
        - 2 * (X_p.T @ W @ y_source)
    )
    print(f"Custom calc. derivative: {der}")
    assert torch.allclose(beta.grad, der)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # test_that_dual_variables_in_OT_correspond_to_effect_of_IW()
    # test_that_weighted_OT_can_be_computed()
    # test_that_weighted_wrr_can_be_computed()
    # compare_ls_solution_to_least_abs_deviations()
    # test_that_wrr_rule_can_be_optimized_in_linear_regression()
    # check_weighted_wrr_derivatives()
    compare_weighted_wrr_opt_to_iw_in_linear_regression()
