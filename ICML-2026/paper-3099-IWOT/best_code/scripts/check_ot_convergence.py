"""
- Create a toy problem with gaussians in 10D
- Parameterize the OT w.r.t linear predictions Wx
- Check gradients w.r.t. W numerically vs. autodiff
- Can we estimate gradients better and if so, would it help?
- Actually apply the stochastic vs. full-batch optimization
and check for convergence

"""

import matplotlib
import numpy as np
import torch
import geomloss
import ot
import matplotlib.pyplot as plt
import time

import shifts
import synthetic_shifts
from models.mlp import MultiLayerPerceptron as MLP
from sinkhorn_uot import fast_uot_sinkhorn, mm_unbalanced

matplotlib.use(backend="QtAgg", force=True)


def calc_emd(f_source, f_target, p):
    num_source = f_source.shape[0]
    num_target = f_target.shape[0]
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target
    cost_mat = torch.cdist(f_source, f_target, p) ** 2
    ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=100000)
    return torch.sqrt(torch.sum(ot_mat * cost_mat))


def calc_ot(f_source, f_target, method, reg=1e-6, p=2):
    if method == "emd":
        return calc_emd(f_source, f_target, p)
    elif method == "sinkhorn_log":
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]
        w_source = torch.ones(num_source) / num_source
        w_target = torch.ones(num_target) / num_target
        cost_mat = torch.cdist(f_source, f_target, p=2) ** 2
        ot_mat = ot.sinkhorn(
            w_source, w_target, cost_mat, method="sinkhorn_log", reg=reg
        )
        return torch.sqrt(torch.sum(ot_mat * cost_mat))
    elif method == "sinkhorn":
        # print(f"Computing OT with geomloss using {method} method")
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=reg, debias=True)
        return ot_loss(f_source, f_target)
    elif method == "sliced":
        return ot.sliced_wasserstein_distance(
            f_source, f_target, p=2, n_projections=1000
        )
    elif method == "debias_ot":
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]
        n_s = int(num_source / 2)
        n_t = int(num_target / 2)
        f_s1 = f_source[:n_s]
        f_s2 = f_source[n_s:]
        f_t1 = f_target[:n_t]
        f_t2 = f_target[n_t:]
        # ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=reg, debias=False)
        # return (ot_loss(f_s1, f_t1) + ot_loss(f_s2, f_t1) + ot_loss(f_s1, f_t2) + ot_loss(f_s2, f_t2) - 2*ot_loss(f_s1, f_s2) - 2*ot_loss(f_t1, f_t2))/2
        # return 2*calc_emd(f_source, f_target, p) - calc_emd(f_s1, f_s2, p) - calc_emd(f_t1, f_t2, p)
        return (
            calc_emd(f_s1, f_t1, p)
            + calc_emd(f_s2, f_t1, p)
            + calc_emd(f_s1, f_t2, p)
            + calc_emd(f_s2, f_t2, p)
            - 2 * calc_emd(f_s1, f_s2, p)
            - 2 * calc_emd(f_t1, f_t2, p)
        ) / 2
    else:
        print("Solver not found!")
        return 0.0


def calc_grad_stats(auto_grad, anal_grad):
    grad_error_norm = torch.linalg.vector_norm(auto_grad - anal_grad, ord=2, dim=None)

    auto_grad_norm = torch.linalg.vector_norm(auto_grad, ord=2, dim=None)
    anal_grad_norm = torch.linalg.vector_norm(anal_grad, ord=2, dim=None)
    inner_prod = torch.sum(auto_grad * anal_grad) / (auto_grad_norm * anal_grad_norm)
    grad_angle_error = 180.0 * torch.acos(inner_prod) / torch.pi

    return grad_error_norm, grad_angle_error


def plot_stats(
    batch_sizes,
    anal_dist,
    samp_dists,
    samp_dist_grad_norm_error,
    samp_dist_grad_angle_error,
    method,
):
    std_dists, mean_dists = torch.std_mean(samp_dists, dim=1)
    # Plot the convergence
    plt.figure()
    plt.errorbar(
        batch_sizes,
        mean_dists.detach().numpy(),
        yerr=std_dists.detach().numpy(),
        fmt="-o",
        capsize=5,
        label=method,
    )
    plt.axhline(
        y=anal_dist.detach().numpy(), color="r", linestyle="--", label="Analytical"
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("W2-Dist")
    plt.title("W2-Dist")
    plt.legend()
    plt.grid(True)

    std_grad_dists, mean_grad_dists = torch.std_mean(samp_dist_grad_norm_error, dim=1)
    plt.figure()
    plt.errorbar(
        batch_sizes,
        mean_grad_dists.detach().numpy(),
        yerr=std_grad_dists.detach().numpy(),
        fmt="-o",
        capsize=5,
        label=method,
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("W2-Dist Grad Norm Diff")
    plt.title("W2-Dist Grad Norm Error")
    plt.legend()
    plt.grid(True)

    std_grad_angles, mean_grad_angles = torch.std_mean(
        samp_dist_grad_angle_error, dim=1
    )
    plt.figure()
    plt.errorbar(
        batch_sizes,
        mean_grad_angles.detach().numpy(),
        yerr=std_grad_angles.detach().numpy(),
        fmt="-o",
        capsize=5,
        label=method,
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("W2-Dist Grad Angle Diff")
    plt.title("W2-Dist Grad Angle Error")
    plt.legend()
    plt.grid(True)


def reset_all(seed):
    # Python & NumPy
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (optional but recommended for fair comps)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_gaussians():
    in_dim, out_dim = 10, 10
    seed = 5
    reset_all(seed)
    model = MLP(layer_sizes=[in_dim, out_dim], f_nonlinear=[])

    method = "emd"  # emd, sinkhorn, sliced, sinkhorn_log
    num_trials = 10
    num_batches = 20
    batch_sizes = torch.arange(start=100, end=1100, step=100, dtype=torch.int)
    samp_dists = torch.zeros(num_trials, num_batches)
    samp_dist_grad_norm_error = torch.zeros(num_trials, num_batches)
    samp_dist_grad_angle_error = torch.zeros(num_trials, num_batches)

    for i in range(num_trials):
        dl_options = {
            "batch_size": batch_sizes[i].item(),
            "shuffle": False,
            "drop_last": True,
        }
        reset_all(seed)
        model.zero_grad()
        scenario = synthetic_shifts.RandomGaussians(
            dim=in_dim, num_batches=num_batches, dataloader_options=dl_options
        )
        anal_dist, anal_grad = scenario.calc_anal_dist(
            model.net[1].weight, model.net[1].bias, calc_grad=True
        )
        print(f"Batch size: {batch_sizes[i].item()}")
        for j, (X_source, X_target) in enumerate(
            zip(scenario.source_dataloader, scenario.target_dataloader)
        ):
            model.zero_grad()
            W_distance = calc_ot(
                model(X_source), model(X_target), method, reg=1e-6, p=2
            )
            W_distance.backward()

            samp_dists[i, j] = W_distance
            auto_grad = model.net[1].weight.grad
            samp_dist_grad_norm_error[i, j], samp_dist_grad_angle_error[i, j] = (
                calc_grad_stats(auto_grad, anal_grad)
            )

    plot_stats(
        batch_sizes,
        anal_dist,
        samp_dists,
        samp_dist_grad_norm_error,
        samp_dist_grad_angle_error,
        method,
    )
    plt.show()


def check_mnist_to_usps():
    in_dim, out_dim = 256, 10
    model = MLP(layer_sizes=[in_dim, out_dim], f_nonlinear=[])

    method = "emd"
    dl_options = {"batch_size": 4096, "shuffle": False, "drop_last": True}
    reset_all(seed=0)
    scenario = shifts.MNIST_to_USPS(dl_options, dl_options)
    for (X_source, y_source), (X_target, y_target) in zip(
        scenario.source_dataloader, scenario.target_dataloader
    ):
        model.zero_grad()
        W_distance = calc_ot(model(X_source), model(X_target), method, reg=1e-6, p=2)
        W_distance.backward()

        big_dist = W_distance
        big_grad = model.net[1].weight.grad
        break

    num_trials = 6
    num_batches = 10
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    samp_dists = torch.zeros(num_trials, num_batches)
    samp_dist_grad_norm_error = torch.zeros(num_trials, num_batches)
    samp_dist_grad_angle_error = torch.zeros(num_trials, num_batches)
    for i in range(num_trials):
        dl_options = {"batch_size": batch_sizes[i], "shuffle": False, "drop_last": True}
        reset_all(seed=i)
        scenario = shifts.MNIST_to_USPS(dl_options, dl_options)
        print(f"Batch size: {batch_sizes[i]}")
        for j, ((X_source, y_source), (X_target, y_target)) in enumerate(
            zip(scenario.source_dataloader, scenario.target_dataloader)
        ):
            model.zero_grad()
            W_distance = calc_ot(
                model(X_source), model(X_target), method, reg=1e-6, p=2
            )
            W_distance.backward()

            samp_dists[i, j] = W_distance
            auto_grad = model.net[1].weight.grad
            samp_dist_grad_norm_error[i, j], samp_dist_grad_angle_error[i, j] = (
                calc_grad_stats(auto_grad, big_grad)
            )
            if j == num_batches - 1:
                break

    plot_stats(
        batch_sizes,
        big_dist,
        samp_dists,
        samp_dist_grad_norm_error,
        samp_dist_grad_angle_error,
        method,
    )
    plt.show()


def compare_ot_solvers():
    # Run various OT methods and check for speed
    # - Sinkhorn divergence via Geomloss
    # - EMD / Sinkhorn via POT
    # - Sliced W-distances via POT
    # Extend later to UOT!
    reset_all(seed=0)
    dim = 10
    dl_options = {"batch_size": 64, "shuffle": False, "drop_last": False}
    num_batches = 100
    scenario = synthetic_shifts.RandomGaussians(
        dim=dim, num_batches=num_batches, dataloader_options=dl_options
    )
    scenario.calc_anal_dist(torch.eye(dim), torch.zeros(dim), calc_grad=False)
    start_time = time.time()
    W_distance = torch.zeros(num_batches)
    for j, (X_source, X_target) in enumerate(
        zip(scenario.source_dataloader, scenario.target_dataloader)
    ):
        # print(f"Iteration {j}")
        W_distance[j] = calc_ot(X_source, X_target, method="emd", reg=1e-6, p=2)
    end_time = time.time()
    print(f"EMD took in total {end_time - start_time} seconds")
    print(f"Avg w_distance: {torch.mean(W_distance)}")

    ot_loss = geomloss.SamplesLoss(
        loss="sinkhorn", p=2, blur=1e-6, diameter=10, debias=False
    )
    start_time = time.time()
    for j, (X_source, X_target) in enumerate(
        zip(scenario.source_dataloader, scenario.target_dataloader)
    ):
        # print(f"Iteration {j}")
        W_distance[j] = ot_loss(X_source, X_target)
        # W_distance[j] = calc_ot(X_source, X_target, method='sinkhorn', reg=1e-6, p=2)
    end_time = time.time()
    print(f"Sinkhorn took in total {end_time - start_time} seconds")
    print(f"Avg w_distance: {torch.mean(W_distance)}")

    start_time = time.time()
    for j, (X_source, X_target) in enumerate(
        zip(scenario.source_dataloader, scenario.target_dataloader)
    ):
        W_distance[j] = ot.sliced_wasserstein_distance(
            X_source, X_target, n_projections=50, p=2
        )
    end_time = time.time()
    print(f"Sliced W2 approx. took in total {end_time - start_time} seconds")
    print(f"Avg w_distance: {torch.mean(W_distance)}")


def run_uot_solvers():
    # Run various UOT methods in POT and check for speed and quality of solution
    reset_all(seed=0)
    dim = 5
    dl_options = {"batch_size": 128, "shuffle": False, "drop_last": False}
    num_batches = 20
    scenario = synthetic_shifts.RandomGaussians(
        dim=dim, num_batches=num_batches, dataloader_options=dl_options
    )
    scenario.calc_anal_dist(A=torch.eye(dim), b=torch.zeros(dim), calc_grad=False)
    gamma = 100
    scenario.calc_unbalanced_anal_dist(gamma, reg=1e-6)

    methods = ["mm", "sinkhorn", "fast_uot"]
    num_methods = len(methods)
    W_distance = torch.zeros(num_methods, num_batches)
    for i in range(num_methods):
        start_time = time.time()
        for j, (X_source, X_target) in enumerate(
            zip(scenario.source_dataloader, scenario.target_dataloader)
        ):
            W_distance[i, j] = calc_uot(
                X_source, X_target, gamma, p=2, method=methods[i]
            )
            # print(W_distance[j].item())
        end_time = time.time()
        print("===================")
        print(
            f"Unbalanced OT using {methods[i]} took in total {end_time - start_time} seconds"
        )
        print(
            f"Avg/std UOT_distance: {torch.mean(W_distance[i])}/{torch.std(W_distance[i])}"
        )


def calc_uot(X_source, X_target, gamma, p, method):
    """
    Possible things to test:

    - Go through the sinkhorn and mm loops
    - Check warmstarting
    - Add GMM based solver
    - Add more dims, change gamma etc.
    """

    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    cost_mat = torch.cdist(X_source, X_target, 2) ** p
    reg_m = (gamma, gamma)
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target
    # For initialization
    # eps = 1e-6
    # ot_mat_init = torch.softmax(-cost_mat / eps, dim=0) * w_target[None, :]
    if method == "fast_uot":
        ot_mat = fast_uot_sinkhorn(
            w_source,
            w_target,
            cost_mat,
            eps=1e-2,
            rho=gamma,
            rho2=gamma,
            n_iter=1000,
            thresh=1e-12,
        )
    elif method == "mm":
        ot_mat = mm_unbalanced(
            w_source, w_target, cost_mat, rho=gamma, rho2=gamma, numItermax=1000
        )
    elif method == "sinkhorn":
        ot_mat = ot.sinkhorn_unbalanced(
            w_source,
            w_target,
            cost_mat,
            reg=1e-2,
            reg_m=reg_m,
            method="sinkhorn",
            stopThr=1e-12,
            numItermax=1000,
        )
    else:
        raise Exception("Not implemented method")
    return torch.sum(ot_mat * cost_mat)


def compare_gromov_wasserstein_to_ot():
    reset_all(seed=0)
    dim = 10
    dl_options = {"batch_size": 64, "shuffle": False, "drop_last": False}
    num_batches = 100
    scenario = synthetic_shifts.RandomGaussians(
        dim=dim, num_batches=num_batches, dataloader_options=dl_options
    )
    scenario.calc_anal_dist(torch.eye(dim), torch.zeros(dim), calc_grad=False)
    start_time = time.time()
    W_distance = torch.zeros(num_batches)
    GW_distance = torch.zeros(num_batches)
    for j, (X_source, X_target) in enumerate(
        zip(scenario.source_dataloader, scenario.target_dataloader)
    ):
        # print(f"Iteration {j}")
        W_distance[j] = calc_ot(X_source, X_target, method="emd", reg=1e-6, p=2)
        dist_source = torch.cdist(X_source, X_source, p=2)
        dist_target = torch.cdist(X_target, X_target, p=2)
        GW_distance[j] = torch.sqrt(
            ot.gromov.gromov_wasserstein2(dist_source, dist_target, symmetric=True)
        )
    end_time = time.time()
    print(f"EMD took in total {end_time - start_time} seconds")
    print(f"Avg w_distance: {torch.mean(W_distance)}")
    print(f"Avg gw_distance: {torch.mean(GW_distance)}")


def check_procrustes_alignment():
    from scipy.spatial import procrustes

    a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], "d")
    b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], "d")
    mtx1, mtx2, disparity = procrustes(a, b)
    round(disparity)
    print(mtx1, mtx2)


if __name__ == "__main__":
    # logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    # logging.getLogger("torch._inductor").setLevel(logging.ERROR)

    # check_procrustes_alignment()
    # compare_gromov_wasserstein_to_ot()
    # run_uot_solvers()
    # compare_ot_solvers()
    check_gaussians()
    # check_mnist_to_usps()
