# # Imports and data ####################################################################
import torch
from tqdm import tqdm
import sys
import os
from itertools import product
from sklearn.covariance import GraphicalLasso

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from MSM.models import density_models as models  # noqa: E402
from MSM.models import variational_models as var_models  # noqa: E402
import MSM.runners as runners  # noqa: E402
from MSM.utils import NaNModel, Prox_SGD, convert_state_dicts, threshold_selector  # noqa: E402
from MSM.utils.data import gen_inverse_ind, fpr, tpr, AUC  # noqa: E402

# Read in the data
data_path = os.path.join(
    os.pardir, os.pardir, "real_world_experiments", "RealData", "yeast_data_tensor_transposed.pt"
)
data_tensor = torch.load(data_path, weights_only=True)

# Set things up
save_path = os.path.join(
    os.path.pardir,
    os.path.pardir,
    "real_world_experiments",
    "Results",
    "yeast_transposed",
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

type_names = ["NonAbs", "Abs", "AbsMLE"]
object_names = ["Accuracy", "AUC", "TPR", "FPR"]
abs_threshold = torch.linspace(0.05, 0.25, 5)
nonabs_threshold = torch.linspace(0.05, 0.25, 5)
threshold_list = [nonabs_threshold, abs_threshold, abs_threshold]
lower_indices = torch.tril_indices(dim, dim, offset=-1)

# # Run Non-corrupted model #####################################################
true_stored_vals = {"Losses": [], "q_State_dicts": []}

l1_regs_noncorrupted = torch.cat((torch.logspace(-1.7, -4, 100), torch.zeros(1)))
niters = torch.tensor([200] + [10] * 100).int()
q_theta = models.NormalSymDensity(
    dim=dim, mean=torch.zeros(dim), Precision=torch.eye(dim)
)

theta_optimizer = Prox_SGD(
    [
        {"params": [q_theta.prec_diag], "l1_reg": 0},
        {"params": [q_theta.prec_off], "l1_reg": 0.02},
    ],
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
positive_rates = torch.mean(true_adjs[..., lower_indices[0], lower_indices[1]].float(), dim=-1)
abs_positive_rates = torch.mean(true_abs_adjs[..., lower_indices[0], lower_indices[1]].float(), dim=-1)

# # Add in MLE Model ############################################################
mle_thresholds = [0.49, 0.349, 0.22, 0.128, 0.093]
true_abs_adjs_mle = []
for mle_threshold in mle_thresholds:
    mle_runner = GraphicalLasso(alpha=mle_threshold, tol=1e-4, max_iter=1000)
    mle_runner.fit(data_tensor.numpy())
    true_abs_adjs_mle.append(torch.abs(torch.tensor(mle_runner.precision_)) > 0.)

true_abs_adjs_mle_subset = torch.stack(true_abs_adjs_mle, dim=0)


true_indices = [gen_inverse_ind(positive_rates, threshold, True) for threshold in nonabs_threshold]
true_abs_indices = [gen_inverse_ind(abs_positive_rates, threshold, True) for threshold in abs_threshold]
true_adjs_subset = true_adjs[true_indices]
true_abs_adjs_subset = true_abs_adjs[true_abs_indices]
true_adjs_list = [true_adjs_subset, true_abs_adjs_subset, true_abs_adjs_mle_subset]

nonzero_threshold_quantiles = torch.tensor([0.1, 0.01, 0.])
positive_threshold_quantiles = torch.tensor([0.8, 0.75, 0.7])


# # Run corrupted model #########################################################
niters = torch.tensor([200]+[20]*200).int()
change_points = (torch.cumsum(niters, 0) / training_args["snapshot_freq"]).int()
l1_regs = torch.logspace(1, -4, 201)

for i in tqdm(range(nrep)):
    mask = torch.bernoulli((1 - missing_prob) * torch.ones_like(data_tensor))
    missing_data = data_tensor * mask
    # ## Run variational model #########################################################
    break_var = False
    em_stored_vals = {"Losses": [], "q_State_dicts": []}
    q_theta = models.NormalSymDensity(dim=dim, mean=torch.zeros(dim), Precision=torch.eye(dim))
    fixed_phi = var_models.VariationalConstantNormal(
        dim=dim, mean=torch.zeros(dim), std=0.1 * torch.ones(dim)
    )

    theta_optimizer = Prox_SGD([
        {"params": [q_theta.prec_diag], "l1_reg": 0.},
        {"params": [q_theta.prec_off], "l1_reg": l1_regs[0].item()}
        ], lr=.001)

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
            print("Nan model, moving to next simulation")
            break_var = True
            break
        except IndexError:
            print("Index Error")
            break_var = True
            break
        for key, val in em_runner.stored_vals.items():
            # Add stored vals from this run to all stored_vals
            em_stored_vals[key] += val

    # If model NaN skip storing and continue on to next simulation
    if break_var:
        continue
    q_theta_params = convert_state_dicts(em_stored_vals["q_State_dicts"])
    em_precs = torch.stack(q_theta_params["Precision"])

    # ## Store results ##########

    file_name = (f"em_slowlr_accuracy_list_miss_prob={missing_prob:.1f}.pt")
    if i == 0:
        # Create empty object dict.
        object_dict = {
            f"{object_name}_{type_name}_{tpr_type}": []
            for object_name, type_name, tpr_type in product(object_names, type_names, ["Normal", "Cumulative"])}
        object_dict["Threshold_Abs"] = abs_threshold
        object_dict["Threshold_NonAbs"] = nonabs_threshold
    else:
        # Load object dict.
        object_dict = torch.load(os.path.join(save_path, file_name), weights_only=True)

    for tpr_type in ["Normal", "Cumulative"]:
        lower_diag = em_precs[..., lower_indices[0], lower_indices[1]][change_points-1]
        positive_threshold = threshold_selector(lower_diag, (1e-6, 1e-1), threshold=0.01,
                                                iters=3, steps=100, type="changes", angles=False)
        abs_threshold = threshold_selector(torch.abs(lower_diag), (1e-6, 1e-1), threshold=0.01,
                                           iters=3, steps=100, type="changes", angles=False)
        em_adjs = em_precs*(1-torch.eye(dim)) > positive_threshold
        em_abs_adjs = (torch.abs(em_precs*(1-torch.eye(dim))) > abs_threshold)
        if tpr_type == "Cumulative":
            em_adjs = torch.cummax(em_adjs, dim=0)[0]
            em_abs_adjs = torch.cummax(em_abs_adjs, dim=0)[0]
        em_adjs_list = [em_adjs, em_abs_adjs, em_abs_adjs]
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

    torch.save(object_dict, os.path.join(save_path, file_name))
