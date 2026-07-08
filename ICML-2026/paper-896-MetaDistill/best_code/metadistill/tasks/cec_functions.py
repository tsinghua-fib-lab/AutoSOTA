import torch
from tasks.utils import *
from torch_basic_settings import *

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)


################CEC####################


FUNCTIONS = dict()
START_GLOBAL_ID = 1

@safe_eval
def cecfun1(x, b=None, w=None):
    # (xi-bi)^2
    # b,n,dim
    # w (dim,1)
    try:
        batch, n, dim = x.shape
    except ValueError:
        batch = 1
        n = 1
        dim = x.shape[-1]
    z = x if FUNCTIONS["cecf1"]["bias"] is None else x - FUNCTIONS["cecf1"]["bias"].view(-1)
    sc = torch.sin(z)
    sc = sc @ FUNCTIONS["cecf1"]["w"]  # b,n,d @  d ,1 = b,n,1
    sc = torch.pow(sc, 2).view(batch, n)
    return sc


FUNCTIONS["cecf1"] = {
    "fid": "cecf1",
    "fun": cecfun1,
    "bias": None,
    "w": None,
    "xub": 10,
    "xlb": -10,
    "bub": 10,
    "blb": -10,
}


@safe_eval
def cecfun2(x, b=None):
    if not FUNCTIONS["cecf2"]["bias"] is None:
        b = FUNCTIONS["cecf2"]["bias"].view(-1)
        z = x - b
    else:
        z = x
    sc = torch.sum(torch.abs(z), dim=-1)
    return sc


FUNCTIONS["cecf2"] = {
    "fid": "cecf2",
    "fun": cecfun2,
    "bias": None,
    "xub": 10,
    "xlb": -10,
    "bub": 10,
    "blb": -10,
}


@safe_eval
def cecfun3(x, b=None):
    if not FUNCTIONS["cecf3"]["bias"] is None:
        z = x - FUNCTIONS["cecf3"]["bias"]
    else:
        z = x
    z1 = z[..., :-1]
    z2 = z[..., 1:]
    sc = torch.sum(torch.abs(z1 + z2), dim=-1) + torch.sum(torch.abs(z), dim=-1)
    return sc


FUNCTIONS["cecf3"] = {
    "fid": "cecf3",
    "fun": cecfun3,
    "bias": None,
    "xub": 10,
    "xlb": -10,
    "bub": 10,
    "blb": -10,
}


@safe_eval
def cecfun4(x, b=None):  # checked
    if not FUNCTIONS["cecf4"]["bias"] is None:
        z = x - FUNCTIONS["cecf4"]["bias"]
    else:
        z = x
    sc = 1e-4 * torch.sum(torch.pow(z, 2), dim=-1)
    return sc


FUNCTIONS["cecf4"] = {
    "fid": "cecf4",
    "fun": cecfun4,
    "bias": None,
    "xub": 100,
    "xlb": -100,
    "bub": 50,
    "blb": -50,
}


@safe_eval
def cecfun5(x, b=None):  # checked
    if not FUNCTIONS["cecf5"]["bias"] is None:
        z = x - FUNCTIONS["cecf5"]["bias"]
    else:
        z = x
    z = torch.abs(z)
    sc = 0.1 * torch.max(z, dim=-1)[0]
    return sc


FUNCTIONS["cecf5"] = {
    "fid": "cecf5",
    "fun": cecfun5,
    "bias": None,
    "xub": 100,
    "xlb": -100,
    "bub": 50,
    "blb": -50,
}


@safe_eval
def cecfun6(x, b=None):  # checked
    if not FUNCTIONS["cecf6"]["bias"] is None:
        z = x - FUNCTIONS["cecf6"]["bias"]
    else:
        z = x
    x1 = z[..., :-1]
    x2 = z[..., 1:]
    return 1e-9 * torch.sum(
        100 * torch.pow((torch.pow(x1, 2) - x2), 2) + torch.pow((x1 - 1), 2), dim=-1
    )


FUNCTIONS["cecf6"] = {
    "fid": "cecf6",
    "fun": cecfun6,
    "bias": None,
    "xub": 100,
    "xlb": -100,
    "bub": 50,
    "blb": -50,
}


@safe_eval
def cecfun7(x, b=None):  # checked
    if not FUNCTIONS["cecf7"]["bias"] is None:
        z = x - FUNCTIONS["cecf7"]["bias"]
    else:
        z = x
    sc = torch.sum(
        torch.pow(z, torch.tensor(2).to(DEVICE)) - 10 * torch.cos(2 * np.pi * (z)) + 10,
        dim=-1,
    )
    return sc


FUNCTIONS["cecf7"] = {
    "fid": "cecf7",
    "fun": cecfun7,
    "bias": None,
    "xub": 5,
    "xlb": -5,
    "bub": 2.5,
    "blb": -2.5,
}

    
@safe_eval
def cecfun8(x, b=None):  # checked
    if not FUNCTIONS["cecf8"]["bias"] is None:
        z = x - FUNCTIONS["cecf8"]["bias"]
    else:
        z = x
    i = (
        torch.from_numpy(np.array([i + 1 for i in range(x.shape[-1])]))
        .view(-1)
        .to(DEVICE)
        .view(1, 1, x.shape[-1])
    )
    sc = (
        torch.sum(torch.pow(z, 2) / 4000, dim=-1)
        - torch.prod(torch.cos((z) / torch.sqrt(i)), dim=-1)
        + 1
    )
    return sc


FUNCTIONS["cecf8"] = {
    "fid": "cecf8",
    "fun": cecfun8,
    "bias": None,
    "xub": 600,
    "xlb": -600,
    "bub": 300,
    "blb": -300,
}


@safe_eval
def cecfun9(x, b=None):  # checked
    if not FUNCTIONS["cecf9"]["bias"] is None:
        z = x - FUNCTIONS["cecf9"]["bias"]
    else:
        z = x
    sc = (
        -20
        * torch.exp(
            -0.2 * torch.sqrt((1 / x.shape[-1]) * torch.sum(torch.pow(z, 2), dim=-1))
        )
        - torch.exp((1 / x.shape[-1]) * torch.sum(torch.cos(2 * np.pi * (z)), dim=-1))
        + 20
        + np.e
    )
    return sc


FUNCTIONS["cecf9"] = {
    "fid": "cecf9",
    "fun": cecfun9,
    "bias": None,
    "xub": 32,
    "xlb": -32,
    "bub": 16,
    "blb": -16,
}


for fid, fun in FUNCTIONS.items():
    fun["train_popsize"] = 100
    fun["train_problemdim"] = 10

for i in range(1, 10):  # global id for training tasks (1..9)
    FUNCTIONS[f"cecf{i}"]["global_id"] = START_GLOBAL_ID + i - 1
