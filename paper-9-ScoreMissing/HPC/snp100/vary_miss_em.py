# # Imports and data ####################################################################
import torch
import numpy as np
from tqdm import tqdm
import sys
import os
from itertools import product

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from MSM.models import density_models as models  # noqa: E402
from MSM.models import variational_models as var_models  # noqa: E402
import MSM.runners as runners  # noqa: E402
from MSM.utils import NaNModel, Prox_SGD, convert_state_dicts, reg_gridsearch  # noqa: E402
from MSM.utils.data import gen_inverse_ind, fpr, tpr, AUC  # noqa: E402

# Read in the data
data_path = os.path.join(
    os.pardir, os.pardir, "real_world_experiments", "RealData", "snp_100_data_tensor.pt"
)
data_tensor = torch.load(data_path, weights_only=True)

# Set things up
save_path = os.path.join(
    os.path.pardir,
    os.path.pardir,
    "real_world_experiments",
    "Results",
    "snp100",
    "VaryMiss",
)
os.makedirs(save_path, exist_ok=True)

# # General set-up ##############################################################
nrep = int(sys.argv[1])
miss_iter = int(sys.argv[2])
miss_probs = torch.arange(0.2, 0.91, step=0.1)
missing_prob = miss_probs[miss_iter].item()

n, dim = data_tensor.shape
training_args = {"nepochs": 10000, "snapshot_freq": 1, "min_loss_val": -1e6}


type_names = ["NonAbs", "Abs"]
object_names = ["Accuracy", "AUC", "TPR", "FPR"]
abs_threshold = torch.linspace(0.1, 0.5, 5)
nonabs_threshold = torch.linspace(0.05, 0.25, 5)
threshold_list = [nonabs_threshold, abs_threshold]
lower_indices = torch.tril_indices(dim, dim, offset=-1)

# # Run Non-corrupted model #####################################################
true_stored_vals = {"Losses": [], "q_State_dicts": []}

l1_regs_noncorrupted = torch.cat((torch.logspace(-1.7, -4, 100), torch.zeros(1)))
niters = torch.tensor([200] + [10] * 100).int()
q_theta = models.NormalSymDensity(
    dim=dim, mean=torch.zeros(dim), Precision=torch.eye(dim)
)

theta_optimizer = Prox_SGD(
    [{"params": [q_theta.prec_diag], "l1_reg": 0},
     {"params": [q_theta.prec_off], "l1_reg": 0.02}],
    lr=0.1,
)

true_runner = runners.ScoreMatchingRunner(q_theta, theta_optimizer)
true_runner.init_dataset(data_tensor, batch_size=100, drop_last=True)
true_runner.set_grad_opts(control_method="clip", max_norm=0.5, norm_type=2)
for i, l1_reg in enumerate(l1_regs_noncorrupted):
    # Update optimiser
    theta_optimizer.param_groups[1]["l1_reg"] = l1_reg.item()
    # Update training args
    training_args["niters"] = niters[i].item()
    try:
        true_runner.train(**training_args)
    except NaNModel:
        print("Nan model")
    except IndexError:
        print("Index Error")

    for key, val in true_runner.stored_vals.items():
        # Add stored vals from this run to all stored_vals
        true_stored_vals[key] += val

q_theta_params = convert_state_dicts(true_stored_vals["q_State_dicts"])
true_precs = torch.stack(q_theta_params["Precision"])
true_adjs = true_precs*(1-torch.eye(dim)) > 0.002
true_abs_adjs = torch.abs(true_precs*(1-torch.eye(dim))) > 0.002
positive_rates = torch.sum(true_adjs, dim=(1, 2))/(dim*(dim-1))
abs_positive_rates = torch.sum(true_abs_adjs, dim=(1, 2))/(dim*(dim-1))

true_abs_indices = [gen_inverse_ind(abs_positive_rates, threshold, True) for threshold in abs_threshold]
true_indices = [gen_inverse_ind(positive_rates, threshold, True) for threshold in nonabs_threshold]
true_adjs_subset = true_adjs[true_indices]
true_abs_adjs_subset = true_abs_adjs[true_abs_indices]
true_adjs_list = [true_adjs_subset, true_abs_adjs_subset]

nonzero_threshold_quantiles = torch.tensor([0.1, 0.01, 0.])


# # Run corrupted model #########################################################

niters = torch.tensor([200]+[10]*100).int()
change_points = (torch.cumsum(niters, 0) / training_args["snapshot_freq"]).int()

for i in tqdm(range(nrep)):
    mask = torch.bernoulli((1 - missing_prob) * torch.ones_like(data_tensor))
    missing_data = data_tensor * mask
    if i == 0:
        # ## Run Initial non-regularised model for threshold calibration ##################
        q_theta = models.NormalSymDensity(dim=dim, mean=torch.zeros(dim), Precision=torch.eye(dim))
        fixed_phi = var_models.VariationalConstantNormal(
            dim=dim, mean=torch.zeros(dim), std=0.1 * torch.ones(dim)
        )
        theta_optimizer = torch.optim.SGD(q_theta.parameters(), lr=.1)
        temp_runner = runners.EMScoreMatchingRunner(
            q_theta, fixed_phi, theta_optimizer, ncopies=10)
        temp_runner.init_dataset(missing_data, mask, batch_size=100, drop_last=True)
        temp_runner.set_grad_opts(control_method="clip", max_norm=0.5, norm_type=2)
        temp_runner.train(1000, 1000, snapshot_freq=10)
        # Get quantiles of lower diagonal of precision matrix
        lower_indices = torch.tril_indices(dim, dim, offset=-1)
        lower_diag = q_theta.Precision[lower_indices[0], lower_indices[1]]
        em_quantiles = torch.quantile(lower_diag, torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])).detach().numpy()
        em_pos_quantiles = torch.quantile(lower_diag[lower_diag > 0],
                                          nonzero_threshold_quantiles).detach().numpy()
        em_abs_quantiles = torch.quantile(torch.abs(lower_diag), nonzero_threshold_quantiles).detach().numpy()
        print("Value of quantile for all lower diag points", em_quantiles)
        print("Value of quantile for positive lower diag points", em_pos_quantiles)
        print("Value of quantiles for absolute data", em_abs_quantiles)

        # ## Run Regularisation Calibration Model #####################################
        q_theta = models.NormalSymDensity(dim=dim, mean=torch.zeros(dim), Precision=torch.eye(dim))
        fixed_phi = var_models.VariationalConstantNormal(
            dim=dim, mean=torch.zeros(dim), std=0.1 * torch.ones(dim)
        )
        theta_optimizer = Prox_SGD([
            {"params": [q_theta.prec_diag], "l1_reg": 0.},
            {"params": [q_theta.prec_off], "l1_reg": 1e-1}
            ], lr=.1)

        em_runner = runners.EMScoreMatchingRunner(q_theta, fixed_phi, theta_optimizer, ncopies=10)
        em_runner.init_dataset(missing_data, mask, batch_size=100, drop_last=True)
        em_runner.set_grad_opts(control_method="clip", max_norm=0.5, norm_type=2)

        em_levels = reg_gridsearch(em_runner, density_range=(0.0, 0.3),
                                   abs_density_range=(0.001, 0.8), start_range=(1e-4, 1e-1), steps=100,
                                   threshold=em_pos_quantiles[0],
                                   abs_threshold=em_abs_quantiles[0],
                                   max_recur=5, log=True)
        l1_regs = torch.logspace(np.log10(em_levels[0]), np.log10(em_levels[1]), 101)
    # ## Run EM model #########################################################
    em_stored_vals = {"Losses": [], "q_State_dicts": []}
    q_theta = models.NormalSymDensity(dim=dim, mean=torch.zeros(dim), Precision=torch.eye(dim))
    fixed_phi = var_models.VariationalConstantNormal(
            dim=dim, mean=torch.zeros(dim), std=0.1 * torch.ones(dim)
        )

    theta_optimizer = Prox_SGD([
        {"params": [q_theta.prec_diag], "l1_reg": 0.},
        {"params": [q_theta.prec_off], "l1_reg": l1_regs[0].item()}
        ], lr=.1)

    em_runner = runners.EMScoreMatchingRunner(
        q_theta, fixed_phi, theta_optimizer, ncopies=10)

    em_runner.init_dataset(missing_data, mask, batch_size=100, drop_last=True)
    em_runner.set_grad_opts(control_method="clip", max_norm=0.5, norm_type=2)
    for j, l1_reg in enumerate(l1_regs):
        # Update optimiser
        theta_optimizer.param_groups[1]["l1_reg"] = l1_reg.item()
        # Update training args
        training_args["niters"] = niters[j].item()
        try:
            em_runner.train(**training_args)
        except NaNModel:
            print("Nan model")
        except IndexError:
            print("Index Error")
        for key, val in em_runner.stored_vals.items():
            # Add stored vals from this run to all stored_vals
            em_stored_vals[key] += val

    q_theta_params = convert_state_dicts(em_stored_vals["q_State_dicts"])
    em_precs = torch.stack(q_theta_params["Precision"])
    for quant_ind, threshold_quantile in enumerate(nonzero_threshold_quantiles):
        if i == 0:
            # Create empty object dict.
            object_dict = {
                f"{object_name}_{type_name}_{tpr_type}": []
                for object_name, type_name, tpr_type in
                product(object_names, type_names, ["Normal", "Cumulative"])}
            object_dict["Threshold_Abs"] = abs_threshold
            object_dict["Threshold_NonAbs"] = nonabs_threshold
        else:
            # Load object dict.
            object_dict = torch.load(
                os.path.join(
                    save_path,
                    f"em_accuracy_list_threshold={threshold_quantile:.2f},miss_prob={missing_prob:.1f}.pt"),
                weights_only=True)

        for tpr_type in ["Normal", "Cumulative"]:
            em_adjs = em_precs*(1-torch.eye(dim)) > em_pos_quantiles[quant_ind]
            em_abs_adjs = torch.abs(em_precs*(1-torch.eye(dim))) > em_abs_quantiles[quant_ind]
            if tpr_type == "Cumulative":
                em_adjs = torch.cummax(em_adjs, dim=0)[0]
                em_abs_adjs = torch.cummax(em_abs_adjs, dim=0)[0]
            em_adjs_list = [em_adjs, em_abs_adjs]
            for type_count, type_name in enumerate(type_names):
                tprs = []
                fprs = []
                aucs = []
                for k in range(len(true_indices)):
                    temp_tprs = tpr(true_adjs_list[type_count][k], em_adjs_list[type_count][change_points-1])
                    temp_fprs = fpr(true_adjs_list[type_count][k], em_adjs_list[type_count][change_points-1])
                    # Append 0 to beggining and 1 to end
                    temp_tprs = torch.cat((torch.zeros(1), temp_tprs, torch.ones(1)))
                    temp_fprs = torch.cat((torch.zeros(1), temp_fprs, torch.ones(1)))
                    tprs.append(temp_tprs)
                    fprs.append(temp_fprs)
                    aucs.append(AUC(temp_tprs, temp_fprs))

                em_adjs_lower = em_adjs_list[type_count][..., lower_indices[0], lower_indices[1]]
                positive_rates = torch.mean(em_adjs_lower.float(), dim=-1)

                em_indices = [gen_inverse_ind(positive_rates, threshold, True)
                              for threshold in threshold_list[type_count]]

                em_adjs_subset = em_adjs_list[type_count][em_indices]
                accuracy_vec = ((torch.sum(true_adjs_list[type_count] == em_adjs_subset, dim=(1, 2)) - dim)
                                / (dim*(dim-1)))
                object_dict[f"AUC_{type_name}_{tpr_type}"].append(aucs)
                object_dict[f"Accuracy_{type_name}_{tpr_type}"].append(accuracy_vec)
                object_dict[f"TPR_{type_name}_{tpr_type}"].append(tprs)
                object_dict[f"FPR_{type_name}_{tpr_type}"].append(fprs)

        torch.save(
            object_dict,
            os.path.join(
                save_path,
                f"em_accuracy_list_threshold={threshold_quantile:.2f},miss_prob={missing_prob:.1f}.pt"))
