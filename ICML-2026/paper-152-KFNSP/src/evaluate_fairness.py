# Utilities Stuff
import torch
import argparse

# Writing Result
from datetime import datetime

# Fairness Measure for continous protected attribute + regression
from utils.hgr import hgr, kde, chi_2, chi_2_cond
from utils.gdp import gdp

# Import models
from models.networks import (
    NeuralWrapper,
    ExampleNeuralNet,
    ExampleNN_FREM,
    FREM
)

from models.decomposition_models import RegularizedKRR, RegularizedSVR
from models.kernel_facil import FairKernelLearningKRR
from models.special import Dummy
from models.linear_inlp import LinearINPWithSVR # Rebuttal
#
from utils.experimental_setup import cross_validate, dict_conversion
from utils.pairwise_fairness import pairwise_fairness

from data.load_data import load_dataset

# Utils
import json


### Penalties for net-HGR
def chi_squared_l1_kde(X, Y, Z):
    # Z is placeholder to keep consistency
    return chi_2(X, Y, kde)


def chi_squared_l1_kde_cond(X, Y, Z):
    return torch.mean(chi_2_cond(X, Y, Z, kde))


def gdp_penalty(X, Y, Z):
    # Z is placeholder to keep consistency
    return gdp(X, Y)


# Helper Function
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


###################################################################################################


#print(torch.cuda.get_device_properties("cuda"))

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, help="Number of epochs", default=500)
parser.add_argument(
    "--folds", type=int, help="Number of cross validation folds", default=5
)
parser.add_argument(
    "--dataset",
    help="File name suffix",
    default="ACSIncome",  # [Crime, ACSIncome, ACSTravelTime]
)
parser.add_argument(
    "--alpha",
    type=float,
    help="Corresponds to alpha'. Reg. for LS step in kernel decomp.",
    default=None,
)
parser.add_argument("--suffix", help="File name suffix", default="")

args = parser.parse_args()

dataset = args.dataset

if args.suffix == "Rebuttal":
    if dataset == "Crime":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.01, 0.05, 0.75),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "SVR-FKD": [0, 5, 30, 45, 60, 80],
            "NN-HGR": [0, 0.025, 0.1, 0.2, 0.5],
            "NN-HGR-L": [0, 0.025, 0.1, 0.2, 0.5],
        }

    elif dataset == "ACSIncome":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.005, 0.05, 0.5),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "SVR-INPL": [0, 60, 100, 160, 180, 200],
            "SVR-FKD": [0, 45, 60, 80, 100, 140], #
            #"NN-HGR": [0, 0.00625, 0.0125, 0.05],
            #"NN-HGR-L": [0, 0.00625, 0.0125, 0.05],
        }     
    elif dataset == "ACSTravelTime":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.001, 0.01, 0.125),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "SVR-FKD": [0, 45, 60, 100],
            "NN-HGR": [0, 0.00625, 0.0125, 0.05],
            "NN-HGR-L": [0, 0.00625, 0.0125, 0.05],
        }
    else:
        raise Exception("Unknown dataset")
elif args.suffix == "Appendix":
    if dataset == "Crime":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.01, 0.05, 0.75),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "SVR-FKD": [0, 5, 30, 45, 60, 80],
            "KRR-FKL": [0, 0.05, 0.25, 0.6, 1],
            "NN-FREM": [0, 0.5, 1.5, 5, 10],
            "NN-HGR-L": [0, 0.025, 0.1, 0.2, 0.5],
        }

    elif dataset == "ACSIncome":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.005, 0.05, 0.5),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "SVR-FKD": [0, 45, 60, 80, 100],
            "KRR-FKL": [0, 0.05, 0.25, 1],
            "NN-FREM": [0, 0.5, 1.5, 5, 10],
            "NN-HGR-L": [0, 0.00625, 0.0125, 0.05],
        }
        
    elif dataset == "ACSTravelTime":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.001, 0.01, 0.125),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "SVR-FKD": [0, 45, 60, 100],
            "KRR-FKL": [0, 0.5, 1, 2],
            "NN-FREM": [0, 0.5, 1.5, 5, 10],
            "NN-HGR-L": [0, 0.00625, 0.0125, 0.05],
        }
    else:
        raise Exception("Unknown dataset")
else:
    if dataset == "Crime":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.01, 0.05, 0.75),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "KRR-FKD": [0, 5, 10, 18],
            "SVR-FKD": [0, 5, 30, 45, 60, 80],
            "KRR-FKL": [0, 0.05, 0.25, 0.6, 1],
            "NN-HGR": [0, 0.025, 0.1, 0.2, 0.5],
        }

    elif dataset == "ACSIncome":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.005, 0.05, 0.5),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "NN-HGR": [0, 0.00625, 0.0125, 0.05],
            "KRR-FKD": [0, 15, 20, 30],
            "SVR-FKD": [0, 45, 60, 80, 100],
            "KRR-FKL": [0, 0.05, 0.25, 1],
        }
        
    elif dataset == "ACSTravelTime":
        nystroem = None
        fixed_param_per_method = {
            "SVR": (0.001, 0.01, 0.125),  # epsilon, gamma, C
            "KRR": (0.25, 0.05),  # alpha, gamma
        }
        param_dict = {
            "NN-HGR": [0, 0.00625, 0.0125, 0.05],
            "KRR-FKD": [0, 15, 20, 30],
            "SVR-FKD": [0, 45, 60, 100],
            "KRR-FKL": [0, 0.5, 1, 2],
        }
    else:
        raise Exception("Unknown dataset")

X, y, p = load_dataset(dataset)

inp_dim = X.shape[-1]


epochs = args.epochs
num_folds = args.folds

result_dictionary = {}

SEED = 7341293

FOLDS = args.folds
EPOCHS = args.epochs
ALPHA_PRIME = args.alpha

fname_suffix = args.suffix

if ALPHA_PRIME is not None:
    result_dictionary["ALPHA_PRIME"] = ALPHA_PRIME

    fname_suffix + "_" + str(ALPHA_PRIME)

    print(ALPHA_PRIME)

fname = (
    "./results/Benchmark_"
    + dataset
    + "_"
    + str(datetime.now()).replace(":", "_")
    + "_"
    + fname_suffix
    + ".json"
)


## Overall Settings

result_dictionary["SEED"] = SEED
result_dictionary["EPOCHS"] = EPOCHS
result_dictionary["FOLDS"] = FOLDS
result_dictionary["DATASET"] = dataset

BATCH_SIZE = 500

## Benchmark
models = []
measures = ["HGR [DP]", "GDP [DP]", "PF [EO]"]



if "SVR-INPL" in param_dict.keys():
    print("STARTING SVR-INPL")
    param = param_dict["SVR-INPL"]

    EPSILON, GAMMA, C_reg = fixed_param_per_method["SVR"]

    if ALPHA_PRIME is not None:
        tmp = ALPHA_PRIME
    else:
        tmp = 0.05

    result, run_time = cross_validate(
        LinearINPWithSVR,
        {
            "alpha_prime": tmp,
            "gamma": GAMMA,
            "C": C_reg,
            "eps": EPSILON,
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
    result_dictionary["SVR-INPL"] = model_dictionary
    models.append("SVR-INPL")

    

if "KRR-FKD" in param_dict.keys():
    print("STARTING KRR-FKD")
    param = param_dict["KRR-FKD"]

    ALPHA, GAMMA = fixed_param_per_method["KRR"]

    if ALPHA_PRIME is not None:
        tmp = ALPHA_PRIME
    else:
        tmp = 0.1

    result, run_time = cross_validate(
        RegularizedKRR,
        {"alpha": ALPHA, "alpha_prime": tmp, "gamma": GAMMA, "nystrom_comp": nystroem},
        dict_conversion({"iterations": param}),
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_cont_fairness,
        seed=SEED,
    )

    model_dictionary = {"param": param, "result": result.tolist(), "time": run_time}
    result_dictionary["KRR-FKD"] = model_dictionary
    models.append("KRR-FKD")


if "KRR-FKL" in param_dict.keys():
    print("STARTING KRR-FKL")
    param = param_dict["KRR-FKL"]

    ALPHA, GAMMA = fixed_param_per_method["KRR"]

    result, run_time = cross_validate(
        FairKernelLearningKRR,
        {"alpha": ALPHA, "gamma": GAMMA},
        dict_conversion({"mu_reg": param}),
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_cont_fairness,
        seed=SEED,
    )

    model_dictionary = {"param": param, "result": result.tolist(), "time": run_time}
    result_dictionary["KRR-FKL"] = model_dictionary
    models.append("KRR-FKL")


if "SVR-FKD" in param_dict.keys():
    print("STARTING SVR-FKD")
    param = param_dict["SVR-FKD"]

    EPSILON, GAMMA, C_reg = fixed_param_per_method["SVR"]

    if ALPHA_PRIME is not None:
        tmp = ALPHA_PRIME
    else:
        tmp = 0.05

    result, run_time = cross_validate(
        RegularizedSVR,
        {
            "alpha_prime": tmp,
            "gamma": GAMMA,
            "C": C_reg,
            "eps": EPSILON,
            "nystrom_comp": nystroem,
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
    result_dictionary["SVR-FKD"] = model_dictionary
    models.append("SVR-FKD")


if "NN-HGR-L" in param_dict.keys():
    param = param_dict["NN-HGR-L"]  

    result, run_time = cross_validate(
        NeuralWrapper,
        {
            "network_cls": ExampleNeuralNet,
            "num_epochs": EPOCHS,
            "network_params": {"input_size": inp_dim, "output_size": 1, "size" : 250},
            "penalty": chi_squared_l1_kde,
            "batch_size": BATCH_SIZE,
        },
        dict_conversion({"lbd_reg": param}), 
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_cont_fairness,
        seed=SEED,
    )

    model_dictionary = {"param": param, "result": result.tolist(), "time": run_time}
    result_dictionary["NN-HGR-L"] = model_dictionary
    models.append("NN-HGR-L")

if "NN-HGR" in param_dict.keys():
    param = param_dict["NN-HGR"]  

    result, run_time = cross_validate(
        NeuralWrapper,
        {
            "network_cls": ExampleNeuralNet,
            "num_epochs": EPOCHS,
            "network_params": {"input_size": inp_dim, "output_size": 1},
            "penalty": chi_squared_l1_kde,
            "batch_size": BATCH_SIZE,
        },
        dict_conversion({"lbd_reg": param}), 
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_cont_fairness,
        seed=SEED,
    )

    model_dictionary = {"param": param, "result": result.tolist(), "time": run_time}
    result_dictionary["NN-HGR"] = model_dictionary
    models.append("NN-HGR")

if "NN-FREM" in param_dict.keys():
    param = param_dict["NN-FREM"]  

    result, run_time = cross_validate(
        FREM,
        {
            "network_cls": ExampleNN_FREM,
            "num_epochs": EPOCHS,
            "network_params": {"input_size": inp_dim, "output_size": 1},
            "batch_size": BATCH_SIZE,
        },
        dict_conversion({"lbd_reg": param}), 
        X,
        y,
        p,
        folds=FOLDS,
        eval_handle=evaluate_cont_fairness,
        seed=SEED,
    )

    model_dictionary = {"param": param, "result": result.tolist(), "time": run_time}
    result_dictionary["NN-FREM"] = model_dictionary
    models.append("NN-FREM")


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
