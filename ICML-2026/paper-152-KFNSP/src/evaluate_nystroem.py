# Utilities
import torch
import json
from datetime import datetime


# Fairness Measure for continous protected attribute + regression
from utils.hgr import hgr, kde
from utils.gdp import gdp

# Models + benchmarks tools
from models.decomposition_models import RegularizedSVR
from models.special import Dummy

from utils.experimental_setup import cross_validate, dict_conversion
from utils.pairwise_fairness import pairwise_fairness

from data.load_data import load_dataset


# Evaluation handle passed to "cross_validate"
def evaluate_cont_fairness(y_hat, y_true, prot_attr):
    MAE = torch.nn.functional.l1_loss(torch.tensor(y_true), torch.tensor(y_hat))

    # Demographic Parity
    HGR = hgr(torch.tensor(y_hat), torch.tensor(prot_attr), density=kde)

    GDP = gdp(torch.tensor(y_hat), torch.tensor(prot_attr))

    # Equal Opportunity
    PF_EO = pairwise_fairness(
        torch.tensor(y_true),
        torch.tensor(y_hat),
        torch.tensor(prot_attr),
        use_label=True,
    )

    return torch.tensor([MAE, HGR, GDP, PF_EO])


# Load dataset
dataset = "Crime" #ACSIncome #ACSTravelTime
X, y, p = load_dataset(dataset)
inp_dim = X.shape[-1]


# Parameters
FOLDS = 5
SEED = 7341293

# Dictionary to sore results and metadata in
result_dictionary = {}

result_dictionary["SEED"] = SEED
result_dictionary["FOLDS"] = FOLDS
result_dictionary["DATASET"] = dataset


fname = (
    "./results/Benchmark_Nystroem_"
    + dataset
    + "_"
    + str(datetime.now()).replace(":", "_")
    + ".json"
)

models = []
measures = ["HGR [DP]", "GDP [DP]", "PF [EO]"]


if dataset == "Crime":
    nystroem = None
    fixed_param_per_method = {
        "SVR": (0.01, 0.05, 0.75),  # epsilon, gamma, C
    }
    param_dict = {
        "SVR-FKD": [0, 5, 30, 45, 60, 80],
    }

elif dataset == "ACSIncome":
    nystroem = None
    fixed_param_per_method = {
        "SVR": (0.005, 0.05, 0.5),  # epsilon, gamma, C
    }
    param_dict = {
        "SVR-FKD": [0, 45, 60, 80, 100],
    }
    
elif dataset == "ACSTravelTime":
    nystroem = None
    fixed_param_per_method = {
        "SVR": (0.001, 0.01, 0.125),  # epsilon, gamma, C
    }
    param_dict = {
        "SVR-FKD": [0, 45, 60, 100],
    }
else:
    raise Exception("Unknown dataset")


# Benchmark for 25, 50 ,75 and 100% of components in nystroem approx.
for nys_cmp in [0.25, 0.5, 0.75, 1]:
    print("NYSTR: ", nys_cmp)

    param = param_dict["SVR-FKD"]
    EPSILON, GAMMA, C_reg = fixed_param_per_method["SVR"]

    alpha_p = 0.05

    result, run_time = cross_validate(
        RegularizedSVR,
        {
            "alpha_prime": alpha_p,
            "gamma": GAMMA,
            "C": C_reg,
            "eps": EPSILON,
            "nystrom_comp": nys_cmp,
        },
        dict_conversion({"iterations": param}),
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_cont_fairness,
        seed=SEED,
    )

    model_dictionary = {"param": param, "result": result.tolist(), "time": run_time}

    str_model = "SVR-FKD" + "(" + str(nys_cmp * 100) + "%)"

    result_dictionary[str_model] = model_dictionary
    models.append(str_model)


result, run_time = cross_validate(
    Dummy,
    {},
    [{"dummy": 0}],
    X,
    y,
    p,
    folds=FOLDS,
    eval_handle=evaluate_cont_fairness,
)


model_dictionary = {"param": None, "result": result.tolist(), "time": run_time}
result_dictionary["DUMMY"] = model_dictionary
models.append("DUMMY")

result_dictionary["MODELS"] = models
result_dictionary["MEASURES"] = measures


with open(fname, "w") as filepath:
    json.dump(result_dictionary, filepath, indent=4)
