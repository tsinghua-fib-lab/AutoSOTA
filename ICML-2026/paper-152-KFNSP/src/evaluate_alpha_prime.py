from data.load_data import load_dataset
from datetime import datetime
from models.decomposition_models import RegularizedKRR

import json
import numpy as np

from utils.gdp import gdp
from utils.experimental_setup import cross_validate, dict_conversion

import torch


def evaluate_metrics(y_hat, y_true, prot_attr):
    MAE = torch.nn.functional.l1_loss(torch.tensor(y_true), torch.tensor(y_hat))

    GDP = gdp(torch.tensor(y_hat), torch.tensor(prot_attr))

    return torch.tensor([MAE, GDP])


X, y, p = load_dataset("Crime")

vals = list(np.linspace(0.05, 0.75, 4))

FOLDS = 5

ITER = [3, 5]

result_krr = np.empty((len(vals), FOLDS, len(ITER), 2))  # 2 Measures (MAE + GDP)

for i, alpha_prime in enumerate(vals):
    cv_res, _ = cross_validate(
        RegularizedKRR,
        {"alpha": 0.25, "alpha_prime": alpha_prime, "gamma": 0.05},
        dict_conversion({"iterations": ITER}),
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_metrics,
        seed=345234456,
    )

    result_krr[i, :] = cv_res


res_dict = {"KRR": result_krr.tolist(), "ITER": ITER, "alpha_primes": vals}


with open(
    "./results/Benchmark_alpha_prime_influence_" + str(datetime.now()).replace(":", "_") + ".json",
    "w",
) as filepath:
    json.dump(res_dict, filepath, indent=4)
