import torch
import numpy as np
import torch.nn.functional as F
from functools import wraps
from torch_basic_settings import DEVICE, DTYPE


torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

BD = 5


def fsign(x):
    x = torch.sign(x)
    return x


def fx_hat(x):
    x1 = x.clone()
    x1[x1 == 0] = 1
    x1 = torch.log(torch.abs(x1))
    return x1


def fc_1(x):
    x1 = x.clone()
    x1[x1 > 0] = 10
    x1[x1 <= 0] = 5.5
    return x1


def fc_2(x):
    x1 = x.clone()
    x1[x1 > 0] = 7.9
    x1[x1 <= 0] = 3.1
    return x1


def fTosz(x):
    xhat = fx_hat(x)
    c1 = fc_1(x)
    c2 = fc_2(x)
    r = fsign(x) * torch.exp(
        xhat + 0.049 * (torch.sin(c1 * xhat) + torch.sin(c2 * xhat))
    )
    return r


def fTasy(x, beta):
    x1 = x.clone()
    z = torch.ones_like(x1).to(DEVICE)  # z marks entries <= 0
    z[x1 > 0] = 0
    xle0 = x1 * z
    x1[x1 <= 0] = 0
    d = x1.shape[-1]
    exp = torch.tensor([beta * (i / (d - 1)) for i in range(d)]).to(DTYPE).to(DEVICE)
    exp = 1 + exp * torch.pow(x1, 0.5)
    y = torch.pow(x1, exp)
    y = (1 - z) * y + z * xle0
    return y


def fpen(x):
    x1 = x.clone()
    x1 = torch.abs(x1) - 5
    x1[x1 < 0] = 0
    r = torch.sum(torch.pow(x1, 2), dim=-1).to(DEVICE).to(DTYPE)
    return r


def fjianAlpha(a, d):
    exp = torch.tensor([0.5 * (i / (d - 1)) for i in range(d)]).to(DTYPE).to(DEVICE)
    lam = torch.pow(a, exp)
    r = torch.diag(lam).to(DEVICE).to(DTYPE)
    return r


def f1_1(d):
    p = torch.rand(d).to(DEVICE)
    p[p > 0.5] = 1
    p[p <= 0.5] = -1
    return p


def gram_schmidt_tensor(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


def orthogo_tensor(x):
    # m, n = x.size()
    # x_np = x.t().detach().cpu().numpy()

    # matrix = [Matrix(x_np.T[i]) for i in tqdm(range((x_np.T).shape[0]))]
    # gram = GramSchmidt(matrix)

    gram = gram_schmidt_tensor(x)
    # ort_list = []
    # m,n=x.shape
    # for i in range(m):
    #     vector = []
    #     for j in range(n):
    #         vector.append(float(gram[i][j]))
    #     ort_list.append(vector)
    # ort_list = np.mat(ort_list)
    # ort_list = torch.from_numpy(ort_list).to(DEVICE)
    ort_list = F.normalize(gram, dim=1)
    return ort_list


def genQorR(dim):
    r = orthogo_tensor(torch.randn((dim, dim)).to(DEVICE)).to(DTYPE)
    return r


def genY(fid, dim):
    if fid == 21:
        y = torch.rand((101, dim), device=DEVICE) * 10 - 5
        y[0] = torch.rand(dim, device=DEVICE) * 8 - 4
    elif fid == 22:
        y = torch.rand((21, dim), device=DEVICE) * 9.8 - 4.9
        y[0] = torch.rand(dim, device=DEVICE) * 3.92 * 2 - 3.92

    return y.to(DTYPE)


def fjianAlphaforC(a, d):
    exp = torch.tensor([0.5 * (i / (d - 1)) for i in range(d)]).to(DTYPE).to(DEVICE)
    lam = torch.pow(a, exp)
    idx = torch.randperm(lam.nelement())
    lam = lam.view(-1)[idx].view(lam.size())
    r = torch.diag(lam)
    return r


def genC(fid, dim):
    if fid == 22:
        alpha = np.random.choice(
            [np.power(1000, float(2 * i) / 19) for i in range(20)],
            size=20,
            replace=False,
        )
        alpha = np.insert(alpha, 0, 1000**2)
        alpha = torch.from_numpy(alpha).to(DTYPE).to(DEVICE)
        C = []
        for item in alpha:
            diag = fjianAlphaforC(item, dim)
            C.append(torch.unsqueeze(diag / torch.pow(item, 0.25), 0))
        C = torch.cat(C, dim=0)
    elif fid == 21:
        alpha = np.random.choice(
            [np.power(1000, float(2 * i) / 99) for i in range(100)],
            size=100,
            replace=False,
        )
        alpha = np.insert(alpha, 0, 1000)
        alpha = torch.from_numpy(alpha).to(DTYPE).to(DEVICE)
        C = []
        for item in alpha:
            diag = fjianAlphaforC(item, dim)
            C.append(torch.unsqueeze(diag / torch.pow(item, 0.25), 0))
        C = torch.cat(C, dim=0)
    return C.to(DTYPE)


def getFitness(x, fun):
    """
    Compute fitness value for a given function.
    """
    if fun["fid"] in [1, 2, 3, 4, 5, 8, 20]:
        r = fun["fun"](x=x, xopt=fun["xopt"], fopt=fun["fopt"])

    elif fun["fid"] in [6]:
        r = fun["fun"](x=x, xopt=fun["xopt"], fopt=fun["fopt"], Q=fun["Q"])

    elif fun["fid"] in [9, 19]:
        r = fun["fun"](x=x, fopt=fun["fopt"], R=fun["R"])

    elif fun["fid"] in [10, 11, 12, 14]:
        r = fun["fun"](x=x, xopt=fun["xopt"], fopt=fun["fopt"], R=fun["R"])

    elif fun["fid"] in [7, 13, 15, 16, 17, 18, 23, 24]:
        r = fun["fun"](x=x, xopt=fun["xopt"], fopt=fun["fopt"], Q=fun["Q"], R=fun["R"])

    elif fun["fid"] in [21, 22]:
        r = fun["fun"](x=x, y=fun["y"], fopt=fun["fopt"], R=fun["R"], C=fun["C"])

    elif fun["fid"] in ["cecf1"]:
        r = fun["fun"](x=x, b=fun["bias"], w=fun["w"])

    elif fun["fid"] in [
        "cecf2",
        "cecf3",
        "cecf4",
        "cecf5",
        "cecf6",
        "cecf7",
        "cecf8",
        "cecf9",
    ]:
        r = fun["fun"](x=x, b=fun["bias"])

    elif fun["fid"] in [f"ef{i}" for i in range(1, 12)]:
        r = fun["fun"](x=x, b=fun["bias"])

    elif fun["fid"] == "CartPole":
        r = fun["fun"](x=x, state=fun["init_state"])

    return r


def genOffset(dim, fun):
    if fun["fid"] in [1, 2, 3, 4]:
        fun["xopt"] = torch.rand(dim, device=DEVICE) * 10 - 5
        fun["fopt"] = np.random.uniform(-1000, 1000)

    elif fun["fid"] in [6]:
        fun["xopt"] = torch.rand(dim, device=DEVICE) * 10 - 5
        fun["fopt"] = np.random.uniform(-1000, 1000)
        fun["Q"] = genQorR(dim)

    elif fun["fid"] in [9, 19]:
        fun["fopt"] = np.random.uniform(-1000, 1000)
        fun["R"] = genQorR(dim)

    elif fun["fid"] in [10, 11, 12, 14]:
        fun["xopt"] = torch.rand(dim, device=DEVICE) * 10 - 5
        fun["fopt"] = np.random.uniform(-1000, 1000)
        fun["R"] = genQorR(dim)

    elif fun["fid"] in [7, 13, 15, 16, 17, 18, 23, 24]:
        fun["xopt"] = torch.rand(dim, device=DEVICE) * 10 - 5
        fun["fopt"] = np.random.uniform(-1000, 1000)
        fun["Q"] = genQorR(dim)
        fun["R"] = genQorR(dim)
        if fun["fid"] in [24]:
            fun["xopt"] = 1.25 * f1_1(dim)

    elif fun["fid"] in [21, 22]:
        # r=fun['fun'](x,y=fun['y'],fopt=fun['fopt'],R=fun['R'],C=fun['C'])
        fun["y"] = genY(fun["fid"], dim)
        fun["fopt"] = np.random.uniform(-1000, 1000)
        fun["R"] = genQorR(dim)
        fun["C"] = genC(fun["fid"], dim)

    elif fun["fid"] in [5]:
        fun["xopt"] = 5 * f1_1(dim)
        fun["fopt"] = np.random.uniform(-1000, 1000)

    elif fun["fid"] in [8]:
        fun["xopt"] = torch.rand(dim, device=DEVICE) * 6 - 3
        fun["fopt"] = np.random.uniform(-1000, 1000)

    elif fun["fid"] in [20]:
        fun["xopt"] = 4.2096874633 / 2 * f1_1(dim)
        fun["fopt"] = np.random.uniform(-1000, 1000)

    elif fun["fid"] in ["cecf1"]:
        fun["bias"] = (torch.rand(dim, device=DEVICE) - 0.5) * (fun["xub"] - fun["xlb"])
        fun["w"] = torch.randn((dim, 1), device=DEVICE)

    elif fun["fid"] in [
        "cecf2",
        "cecf3",
        "cecf4",
        "cecf5",
        "cecf6",
        "cecf7",
        "cecf8",
        "cecf9",
    ]:
        fun["bias"] = (torch.rand(dim, device=DEVICE) - 0.5) * (fun["xub"] - fun["xlb"])

    elif fun["fid"] in [f"ef{i}" for i in range(1, 12)]:
        fun["bias"] = (torch.rand(dim, device=DEVICE) - 0.5) * (fun["xub"] - fun["xlb"])

    elif fun["fid"] == "CarthPole":  # generate initial state
        loc = np.random.rand((4,)) - 0.5
        scaling = np.array([5, 0.2, 0.4, 0.2])

        init_state = np.dot(loc, scaling)

        fun["init_state"] = init_state


def gen_tf_offsets(dim, fun, epoch=500):
    if fun["fid"] == "cecf1":
        offsets = []
        for _ in range(epoch):
            offset = {}
            offset["bias"] = (
                ((torch.rand(dim, device=DEVICE) - 0.5) * (fun["xub"] - fun["xlb"]))
                .cpu()
                .numpy()
            )
            offset["w"] = torch.randn((dim, 1), device=DEVICE).cpu().numpy()
            offsets.append(offset)

    elif fun["fid"] in [
        "cecf2",
        "cecf3",
        "cecf4",
        "cecf5",
        "cecf6",
        "cecf7",
        "cecf8",
        "cecf9",
    ]:
        offsets = []
        for _ in range(epoch):
            offset = {}
            offset["bias"] = (
                ((torch.rand(dim, device=DEVICE) - 0.5) * (fun["xub"] - fun["xlb"]))
                .cpu()
                .numpy()
            )
            offsets.append(offset)
    elif fun["fid"] in [f"ef{i}" for i in range(1, 12)]:
        offsets = []
        for _ in range(epoch):
            offset = {}
            offset["bias"] = (
                ((torch.rand(dim, device=DEVICE) - 0.5) * (fun["xub"] - fun["xlb"]))
                .cpu()
                .numpy()
            )
            offsets.append(offset)
    else:
        offsets = None

    return offsets


def set_tf_offset(fun, offset):
    if fun["fid"] == "cecf1":
        fun["bias"] = torch.from_numpy(offset["bias"]).to(DEVICE).to(DTYPE)
        fun["w"] = torch.from_numpy(offset["w"]).to(DEVICE).to(DTYPE)
    elif fun["fid"] in [
        "cecf2",
        "cecf3",
        "cecf4",
        "cecf5",
        "cecf6",
        "cecf7",
        "cecf8",
        "cecf9",
    ]:
        fun["bias"] = torch.from_numpy(offset["bias"]).to(DEVICE).to(DTYPE)

    elif fun["fid"] in [f"ef{i}" for i in range(1, 12)]:
        fun["bias"] = torch.from_numpy(offset["bias"]).to(DEVICE).to(DTYPE)
    else:
        raise ValueError("Invalid TF ID")


def setOffset(fun, kwargs):
    if fun["fid"] in [1, 2, 3, 4]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]

    elif fun["fid"] in [6]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]
        fun["Q"] = kwargs["Q"].to(DEVICE).to(DTYPE)

    elif fun["fid"] in [9, 19]:
        fun["fopt"] = kwargs["fopt"]
        fun["R"] = kwargs["R"].to(DEVICE).to(DTYPE)

    elif fun["fid"] in [10, 11, 12, 14]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]
        fun["R"] = kwargs["R"].to(DEVICE).to(DTYPE)

    elif fun["fid"] in [7, 13, 15, 16, 17, 18, 23, 24]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]
        fun["Q"] = kwargs["Q"].to(DEVICE).to(DTYPE)
        fun["R"] = kwargs["R"].to(DEVICE).to(DTYPE)

    elif fun["fid"] in [21, 22]:
        # r=fun['fun'](x,y=fun['y'],fopt=fun['fopt'],R=fun['R'],C=fun['C'])
        fun["y"] = kwargs["y"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]
        fun["R"] = kwargs["R"].to(DEVICE).to(DTYPE)
        fun["C"] = kwargs["C"].to(DEVICE).to(DTYPE)

    elif fun["fid"] in [5]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]

    elif fun["fid"] in [8]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]

    elif fun["fid"] in [20]:
        fun["xopt"] = kwargs["xopt"].to(DEVICE).to(DTYPE)
        fun["fopt"] = kwargs["fopt"]

    elif fun["fid"] in ["cecf1"]:
        fun["bias"] = kwargs["bias"].to(DEVICE).to(DTYPE)
        fun["w"] = kwargs["w"].to(DEVICE).to(DTYPE)

    elif fun["fid"] in [
        "cecf2",
        "cecf3",
        "cecf4",
        "cecf5",
        "cecf6",
        "cecf7",
        "cecf8",
        "cecf9",
    ]:
        fun["bias"] = kwargs["bias"].to(DEVICE).to(DTYPE)


def getOffset(fun):
    if fun["fid"] in [1, 2, 3, 4]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"]}

    elif fun["fid"] in [6]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"], "Q": fun["Q"]}

    elif fun["fid"] in [9, 19]:
        return {"fopt": fun["fopt"], "R": fun["R"]}

    elif fun["fid"] in [10, 11, 12, 14]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"], "R": fun["R"]}

    elif fun["fid"] in [7, 13, 15, 16, 17, 18, 23, 24]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"], "Q": fun["Q"], "R": fun["R"]}

    elif fun["fid"] in [21, 22]:
        return {"y": fun["y"], "fopt": fun["fopt"], "R": fun["R"], "C": fun["C"]}

    elif fun["fid"] in [5]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"]}

    elif fun["fid"] in [8]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"]}

    elif fun["fid"] in [20]:
        return {"xopt": fun["xopt"], "fopt": fun["fopt"]}

    elif fun["fid"] in ["cecf1"]:
        return {"bias": fun["bias"], "w": fun["w"]}

    elif fun["fid"] in [
        "cecf2",
        "cecf3",
        "cecf4",
        "cecf5",
        "cecf6",
        "cecf7",
        "cecf8",
        "cecf9",
    ]:
        return {"bias": fun["bias"]}


# Compatibility helper
def safe_eval(func):
    """
    Ensure input arrays/tensors are converted to a consistent type.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        np_out = False

        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                new_arg = torch.from_numpy(args[i]).to(DEVICE).to(DTYPE)
                new_args.append(new_arg)
                if i == 0:  # x
                    np_out = True
            else:
                new_args.append(args[i])

        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                new_kwargs[key] = torch.from_numpy(value).to(DEVICE).to(DTYPE)
                if key == "x":
                    np_out = True
            else:
                new_kwargs[key] = value

        fitness = func(*new_args, **new_kwargs)
        if np_out:
            fitness = fitness.cpu().numpy()

        return fitness

    return wrapper


if __name__ == "__main__":
    # fset = [CEC.FUNCTIONS[f"cecf{i}"] for i in range(1, 10)]
    # offsets = {}
    # for func in fset:
    #     genOffset(fun=func, dim=10)
    #     offset = getOffset(fun=func)
    #     offsets[func["fid"]] = offset

    # offsets = pickle.dumps(offsets)
    # with open("offsets/cec_offsets.pkl", "wb") as fp:
    #     fp.write(offsets)
    ...
