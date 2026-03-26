import torch
from torch import distributions as dists
from torch import optim
from tqdm import tqdm
import sys
import os
from experiment_utils import make_multistar_prec, tpr, fpr, roc, AUC, make_planar_funcs

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
    "100Dim_StarPrec",
    "VaryCentres",
)
os.makedirs(save_path, exist_ok=True)
nrep = int(sys.argv[1])
centres_iter = int(sys.argv[2])

dim = 100
miss_prob = 0.7
num_centers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
prob_connected_val = [0.8/i for i in num_centers]
min_eval = 0.1
sample_size = 1000
ncentres = num_centers[centres_iter]
prob_connected = prob_connected_val[centres_iter]

training_args = {
    "nepochs": 1000,
    "niters": 500,
    "snapshot_freq": 10,
    "min_loss_val": -1e6,
    "true_loss": False,
}

l1_regs = torch.logspace(-1.7, -4, 101)
niters = torch.tensor([200] + [10] * 100).int()
threshold = 0.002

change_points = (torch.cumsum(niters, 0) / training_args["snapshot_freq"]).int()
new_stored_vals = {
    "TPR": [],
    "FPR": [],
    "TPR_abs": [],
    "FPR_abs": [],
    "AUC": [],
    "Prec_Dist": [],
    "Losses": [],
}
new_stored_vals["Change_Points"] = change_points
# precs = {"True_Precs": [], "Est_Precs": []}

for i in range(nrep):
    # Make distribution
    prec = make_multistar_prec(dim, prob_connected, min_eval, num_stars=ncentres)
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

    # Set up models
    q_theta = models.NormalSymDensity(
        dim=dim,
        mean=torch.zeros(dim),
        Precision=torch.eye(dim),
        boundary_func=planar_boundary,
    )
    p_phi = var_models.VariationalMLPTruncConstVarNormal(
        dim=dim,
        boundary_func=planar_boundary,
        separate_boundary=False,
        std_lb=0.1,
        sampling_factor=10,
        std_increase=0.1,
    )
    # Set up optimisers
    phi_optimizer = optim.Adam(p_phi.parameters(), lr=0.01)
    full_stored_vals = {"Losses": [], "q_State_dicts": []}
    theta_optimizer = Prox_SGD(
        [
            {"params": [q_theta.prec_diag], "l1_reg": 0},
            {"params": [q_theta.prec_off], "l1_reg": 0.02},
        ],
        lr=0.025,
    )
    runner = runners.BiLevelMarginalTrunc(
        q_theta,
        p_phi,
        theta_optimizer,
        phi_optimizer,
        planar_g_mask,
        elementwise_trunc=False,
        do_iw=False,
        inner_loss="fisher",
        n_phi_step=10,
    )
    runner.init_dataset(X_corrupted, mask, batch_size=100)
    runner.set_grad_opts("both", control_method="clip", max_norm=0.5, norm_type=2)
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

        runner.stored_vals.pop("p_State_dicts")
        for key, val in runner.stored_vals.items():
            # Add stored vals from this run to all stored_vals
            full_stored_vals[key] += val

    est_precs = torch.stack(
        [state_dict["Precision"] for state_dict in full_stored_vals["q_State_dicts"]]
    )
    est_adjs_abs = (torch.abs(est_precs) > threshold).int()
    est_adjs = (est_precs > threshold).int()
    prec_dists = torch.linalg.matrix_norm(est_precs - prec, dim=(1, 2))
    fprs = fpr(true_adj, est_adjs)
    tprs = tpr(true_adj, est_adjs)
    fprs_abs = fpr(true_adj, est_adjs_abs)
    tprs_abs = tpr(true_adj, est_adjs_abs)
    aucs = AUC(*roc(true_adj, est_precs))

    # precs["True_Precs"].append(prec)
    # precs["Est_Precs"].append(est_precs[change_points-1])
    new_stored_vals["TPR"].append(tprs)
    new_stored_vals["FPR"].append(fprs)
    new_stored_vals["TPR_abs"].append(tprs_abs)
    new_stored_vals["FPR_abs"].append(fprs_abs)
    new_stored_vals["Prec_Dist"].append(prec_dists)
    new_stored_vals["AUC"].append(aucs)
    new_stored_vals["Losses"].append(full_stored_vals["Losses"])

    torch.save(
        new_stored_vals,
        os.path.join(
            save_path, f"variational_fullauc_repeat_ncentres_{ncentres}.pth"
        ),
    )
    # torch.save(precs, os.path.join(save_path, "variational_fullauc_repeat_prec_LARGE.pth"))
