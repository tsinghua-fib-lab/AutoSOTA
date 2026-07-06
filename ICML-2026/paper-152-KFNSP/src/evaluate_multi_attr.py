from data.load_data import load_dataset

from utils.experimental_setup import cross_validate, dict_conversion

from models.decomposition_models import RegularizedSVR

from utils.gdp import gdp

from datetime import datetime

import torch

from models.special import Dummy
import json


# Note that we here use {"single_protected" : True} which is also only aimed for this benchmark
# as this will only protected w.r.t. to the first one of all passed protected attributes.

SEED = 2611864105

X, y, p = load_dataset("CrimeMulti")


def evaluate_double_fairness(y_hat, y_true, prot_attr):
    MAE = torch.nn.functional.l1_loss(torch.tensor(y_true), torch.tensor(y_hat))

    # Measure GDP once w.r.t. to white pop. perc. and once w.r.t. to black pop. perc.

    prot_black = torch.tensor(prot_attr)[:, 0]
    prot_white = torch.tensor(prot_attr)[:, 1]

    GDP_black = gdp(torch.tensor(y_hat), prot_black)
    GDP_white = gdp(torch.tensor(y_hat), prot_white)

    return torch.tensor([MAE, GDP_black, GDP_white])


FOLDS = 5

fixed_param_per_method = {
    "SVR": (0.01, 0.05, 0.75),  # epsilon, gamma
}

param_dict = {
    "SVR-FKD": [0, 5, 12, 25, 35, 45],
}


print("STARTING MULTICRIME")

param = param_dict["SVR-FKD"]
EPSILON, GAMMA, C_reg = fixed_param_per_method["SVR"]

result, run_time = cross_validate(
    RegularizedSVR,
    {"alpha_prime": 0.05, "gamma": GAMMA, "eps": EPSILON, "C": C_reg},
    dict_conversion({"iterations": param}),
    X,
    y,
    p,
    folds=FOLDS,
    eval_handle=evaluate_double_fairness,
    seed=SEED,
)

crime_mlt = {"param": param, "result": result.tolist(), "time": run_time}


param_dict = {
    "SVR-FKD": [0, 10, 35, 50, 70, 85],
}


print("STARTING SINGLE CRIME")
param = param_dict["SVR-FKD"]

EPSILON, GAMMA, C_reg = fixed_param_per_method["SVR"]

result, run_time = cross_validate(
    RegularizedSVR,
    {
        "single_protected": True,
        "alpha_prime": 0.05,
        "gamma": GAMMA,
        "eps": EPSILON,
        "C": C_reg,
    },
    dict_conversion({"iterations": param}),
    X,
    y,
    p,
    folds=FOLDS,
    eval_handle=evaluate_double_fairness,
    seed=SEED,
)

crime_sgl = {"param": param, "result": result.tolist(), "time": run_time}


result, run_time = cross_validate(
    Dummy,
    {},
    [{"dummy": 0}],
    X,
    y,
    p,
    folds=FOLDS,
    eval_handle=evaluate_double_fairness,
)

crime_dummy = {"result": result.tolist(), "time": run_time}

result_dictionary = {
    "CrimeSingle": crime_sgl,
    "CrimeMulti": crime_mlt,
    "DUMMY": crime_dummy,
}


fname = (
    "./results/"
    + "Benchmark_multi_protected_"
    + "_"
    + str(datetime.now()).replace(":", "_")
    + "_"
    + ".json"
)

with open(fname, "w") as filepath:
    json.dump(result_dictionary, filepath, indent=4)
