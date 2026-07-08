import torch
from tqdm import tqdm
from tasks.utils import *
from torch_basic_settings import *
from tasks.problem import taskProblem
from tasks.utils import genOffset


############### Elementary Functions ################
## Basic elementary functions (range roughly bounded in [-1e3, 1e3])


FUNCTIONS = {}
START_GLOBAL_ID = 10


@safe_eval
def ef1(x, b=None):
    b = 0 if FUNCTIONS["ef1"]["bias"] is None else FUNCTIONS["ef1"]["bias"]

    z = x - b
    y = 200 * torch.mean(torch.cos(torch.abs(z)), dim=-1)

    return y


FUNCTIONS["ef1"] = {
    "fid": "ef1",
    "fun": ef1,
    "bias": None,
    "xlb": -10,
    "xub": 10,
}


@safe_eval
def ef2(x, b=None):
    b = 0 if FUNCTIONS["ef2"]["bias"] is None else FUNCTIONS["ef2"]["bias"]

    z = x - b
    y = torch.sum(torch.sin(z), dim=-1)

    return y


FUNCTIONS["ef2"] = {
    "fid": "ef2",
    "fun": ef2,
    "bias": None,
    "xlb": -10,
    "xub": 10,
}


@safe_eval
def ef3(x, b=None):
    b = 0 if FUNCTIONS["ef3"]["bias"] is None else FUNCTIONS["ef3"]["bias"]

    z = x - b
    y = torch.sum(torch.sqrt(torch.abs(z)), dim=-1)

    return y


FUNCTIONS["ef3"] = {
    "fid": "ef3",
    "fun": ef3,
    "bias": None,
    "xlb": -100,
    "xub": 100,
}


@safe_eval
def ef4(x, b=None):
    b = 0 if FUNCTIONS["ef4"]["bias"] is None else FUNCTIONS["ef4"]["bias"]

    z = x - b
    y = torch.log1p(1 + torch.mean(torch.abs(z), dim=-1))

    return y


FUNCTIONS["ef4"] = {
    "fid": "ef4",
    "fun": ef4,
    "bias": None,
    "xlb": -100,
    "xub": 100,
}


@safe_eval
def ef5(x, b=None):
    b = 0 if FUNCTIONS["ef5"]["bias"] is None else FUNCTIONS["ef5"]["bias"]

    z = x - b
    y = 1e-3 * torch.mean(torch.abs(torch.pow(z, 3)), dim=-1)

    return y


FUNCTIONS["ef5"] = {
    "fid": "ef5",
    "fun": ef5,
    "bias": None,
    "xlb": -10,
    "xub": 10,
}


@safe_eval
def ef6(x, b=None):
    b = 0 if FUNCTIONS["ef6"]["bias"] is None else FUNCTIONS["ef6"]["bias"]

    z = x - b
    y = torch.exp2(0.1 * torch.mean(torch.abs(z), dim=-1))

    return y


FUNCTIONS["ef6"] = {
    "fid": "ef6",
    "fun": ef6,
    "bias": None,
    "xlb": -50,
    "xub": 50,
}


@safe_eval
def ef7(x, b=None):
    b = 0 if FUNCTIONS["ef7"]["bias"] is None else FUNCTIONS["ef7"]["bias"]

    z = x - b
    y = torch.sum(torch.pow(z, 2) - z, dim=-1)

    return y


FUNCTIONS["ef7"] = {
    "fid": "ef7",
    "fun": ef7,
    "bias": None,
    "xlb": -15,
    "xub": 15,
}


@safe_eval
def ef8(x, b=None):
    b = 0 if FUNCTIONS["ef8"]["bias"] is None else FUNCTIONS["ef8"]["bias"]

    z = x - b
    z1 = z[..., :5]
    z2 = z[..., 5:]
    y = torch.mean(torch.abs(z1), dim=-1) + torch.mean(torch.abs(z2), dim=-1)

    return y


FUNCTIONS["ef8"] = {
    "fid": "ef8",
    "fun": ef8,
    "bias": None,
    "xlb": -100,
    "xub": 100,
}


@safe_eval
def ef9(x, b=None):
    b = 0 if FUNCTIONS["ef9"]["bias"] is None else FUNCTIONS["ef9"]["bias"]

    z = x - b
    z1 = z[..., :5]
    z2 = z[..., 5:]
    y = torch.mean(torch.abs(z1 + z2), dim=-1) + torch.mean(torch.abs(z), dim=-1)

    return y


FUNCTIONS["ef9"] = {
    "fid": "ef9",
    "fun": ef9,
    "bias": None,
    "xlb": -100,
    "xub": 100,
}


@safe_eval
def ef10(x, b=None):
    b = 0 if FUNCTIONS["ef10"]["bias"] is None else FUNCTIONS["ef10"]["bias"]

    z = x - b
    y = torch.sum(torch.pow(z[..., :2], 2), dim=-1) + torch.sum(
        torch.pow(z[..., 2:], 2), dim=-1
    )

    return y


FUNCTIONS["ef10"] = {
    "fid": "ef10",
    "fun": ef10,
    "bias": None,
    "xlb": -10,
    "xub": 10,
}


@safe_eval
def ef11(x, b=None):
    b = 0 if FUNCTIONS["ef11"]["bias"] is None else FUNCTIONS["ef11"]["bias"]

    z = x - b
    y = 25 * torch.mean(torch.sin(z) + torch.cos(z), dim=-1)

    return y


FUNCTIONS["ef11"] = {
    "fid": "ef11",
    "fun": ef11,
    "bias": None,
    "xlb": -100,
    "xub": 100,
}


for fid, fun in FUNCTIONS.items():
    fun["train_popsize"] = 100
    fun["train_problemdim"] = 10

for i in range(1, 12):
    FUNCTIONS[f"ef{i}"]["global_id"] = START_GLOBAL_ID + i - 1

if __name__ == "__main__":
    # Test the functions
    # n_test = 100
    # task = taskProblem(dim=10)
    # bar = tqdm(range(n_test))
    # for _ in bar:
    #     for k, v in FUNCTIONS.items():
    #         if v["fid"] == "ef10":
    #             genOffset(dim=10, fun=v)
    #             task.setfun(v)
    #             x = task.genRandomPop((1000, 100, 10))
    #             result = task.calfitness(x)[0][..., 0]
    #             result = result.view(-1)
    #             print(f"{k}: min={result.min()}; max={result.max()}")
    #             if any(torch.isnan(result)) or any(torch.isinf(result)):
    #                 print(f"Function {k} has invalid values.")

    x = torch.tensor(
        [
            -1.3859,
            70.0376,
            -89.4913,
            -22.9989,
            -41.9943,
            11.5305,
            -41.9542,
            28.1918,
            45.3584,
            -39.3441,
        ]
    )

    print(ef10(x=x, b=None))
