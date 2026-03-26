import torch
from torch import distributions as dists
from tqdm import tqdm
import sys
import os
from experiment_utils import make_star_prec, tpr, fpr, make_planar_funcs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from MSM.models import density_models as models  # noqa: E402
from MSM.models import variational_models as var_models  # noqa: E402
import MSM.runners as runners  # noqa: E402
from MSM.utils import NaNModel, Prox_SGD  # noqa: E402

save_path = os.path.join(
    os.path.pardir,
    os.path.pardir,
    "simulated_data",
    "GGM",
    "StarPrec",
    "VaryDim",
)
os.makedirs(save_path, exist_ok=True)
nrep = int(sys.argv[1])
dim_iter = int(sys.argv[2])
miss_dims = [5, 10, 20, 50, 100]

dim = miss_dims[dim_iter]
prob_connected = 0.9
min_eval = 0.1
sample_size = 1000
miss_prob = 0.7

training_args = {
    "nepochs": 1000,
    "niters": 500,
    "snapshot_freq": 10,
    "min_loss_val": -1e6,
    "true_loss": False,
}

l1_regs = torch.logspace(1, -4, 101)
niters = torch.tensor([200] + [10] * 100).int()
threshold = 0.002

change_points = (torch.cumsum(niters, 0) / training_args["snapshot_freq"]).int()
new_stored_vals = {
    "TPR": [],
    "FPR": [],
    "Prec_Dist": [],
    "Losses": [],
}
new_stored_vals["Change_Points"] = change_points
# precs = {"True_Precs": [], "Est_Precs": []}

for i in range(nrep):
    # Make distribution
    prec = make_star_prec(dim, prob_connected)
    true_adj = (torch.abs(prec) > 0).int()
    cov = torch.linalg.inv(prec)
    mean = torch.zeros(dim)
    untrunc_dist = dists.MultivariateNormal(mean, cov)

    # Make boundary
    norm_vec = torch.randn(dim) / torch.linalg.vector_norm(torch.randn(dim))
    intercept = -torch.quantile(
        untrunc_dist.sample((10000,)) @ norm_vec, q=torch.tensor(0.2)
    ).item()
    planar_g, planar_g_mask, planar_boundary = make_planar_funcs(norm_vec, intercept)

    # Generate data
    X_untrunc = untrunc_dist.sample((2 * sample_size,))
    X = X_untrunc[planar_g(X_untrunc) > 0][:sample_size]
    if X.shape[0] < sample_size:
        raise ValueError("Not enough samples")

    # Create mask
    mask = torch.bernoulli(torch.ones_like(X) * (1 - miss_prob))
    X_corrupted = X * mask

    full_stored_vals = {"Losses": [], "q_State_dicts": []}
    # Set up models
    q_theta = models.NormalSymDensity(
        dim=dim,
        mean=torch.zeros(dim),
        Precision=torch.eye(dim),
        boundary_func=planar_boundary,
    )
    fixed_phi = var_models.VariationalConstantNormal(
        dim=dim, mean=torch.zeros(dim), std=0.1 * torch.ones(dim)
    )
    # Set up optimisers
    theta_optimizer = Prox_SGD(
        [
            {"params": [q_theta.prec_diag], "l1_reg": 0},
            {"params": [q_theta.prec_off], "l1_reg": 0.02},
        ],
        lr=0.025,
    )
    runner = runners.TruncatedIWScoreMatchingRunner(
        q_theta,
        fixed_phi,
        theta_optimizer,
        planar_g_mask,
        elementwise_trunc=False,
        ncopies=10,
    )
    runner.init_dataset(X_corrupted, mask, batch_size=100)
    runner.set_grad_opts(control_method="clip", max_norm=0.5, norm_type=2)
    for i, l1_reg in tqdm(enumerate(l1_regs)):
        # Update optimiser
        theta_optimizer.param_groups[1]["l1_reg"] = l1_reg.item()
        # Update training args
        training_args["niters"] = niters[i].item()
        try:
            runner.train(**training_args)
        except NaNModel:
            print("Nan model")
        except IndexError:
            print("Index Error")

        for key, val in runner.stored_vals.items():
            # Add stored vals from this run to all stored_vals
            full_stored_vals[key] += val

    est_precs = torch.stack(
        [state_dict["Precision"] for state_dict in full_stored_vals["q_State_dicts"]]
    )
    est_adjs = (torch.abs(est_precs) > threshold).int()
    prec_dists = torch.linalg.matrix_norm(est_precs - prec, dim=(1, 2))
    fprs = fpr(true_adj, est_adjs)
    tprs = tpr(true_adj, est_adjs)

    # precs["True_Precs"].append(prec)
    # precs["Est_Precs"].append(est_precs[change_points - 1])
    new_stored_vals["TPR"].append(tprs)
    new_stored_vals["FPR"].append(fprs)
    new_stored_vals["Prec_Dist"].append(prec_dists)
    new_stored_vals["Losses"].append(full_stored_vals["Losses"])

    torch.save(new_stored_vals, os.path.join(save_path, f"iw_fullauc_repeat_dim_{dim}.pth"))
    # torch.save(
    #     precs, os.path.join(save_path, "iw_fullauc_repeat_prec_LARGE.pth")
    # )
