import math
import torch
from tasks.utils import *
from torch_basic_settings import *


torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)


#######FUNCTIONS#####


FUNCTIONS = dict()


############################# bbob functions #################################


def f1(x=None, xopt=None, fopt=None):
    z = x - xopt
    r = torch.sum(torch.pow(z, 2), dim=-1) + fopt
    return r


FUNCTIONS[1] = {"fid": 1, "fun": f1, "xopt": None, "fopt": None}


def f2(x=None, xopt=None, fopt=None):
    z = fTosz(x - xopt)
    z = torch.pow(z, 2)
    f = (
        torch.tensor([6 * (i / z.shape[-1] - 1) for i in range(z.shape[-1])])
        .float()
        .to(DEVICE)
    )
    f = torch.pow(10.0, f)
    z = z * f
    r = torch.sum(z, dim=-1) + fopt
    return r


FUNCTIONS[2] = {"fid": 2, "fun": f2, "xopt": None, "fopt": None}


def f3(x=None, xopt=None, fopt=None):
    x = x.view(-1, x.shape[-1])
    z = x - xopt
    z = fTosz(z)
    z = fTasy(z, 0.2)
    z = (fjianAlpha(10, z.shape[-1]).to(DTYPE).to(DEVICE) @ z.transpose(-1, -2)).transpose(
        -1, -2
    )  # (n,d)@(d,d)
    r = (
        10 * (z.shape[-1] - torch.sum(torch.cos(2 * np.pi * z), dim=-1))
        + torch.sum(torch.pow(z, 2), dim=-1)
        + fopt
    )
    return r


FUNCTIONS[3] = {"fid": 3, "fun": f3, "xopt": None, "fopt": None}


def f4(x=None, xopt=None, fopt=None):
    z = x - xopt
    z = fTosz(z)
    jiou = []  # 1 for even index, 0 for odd index
    for i in range(x.shape[-1]):
        if i % 2 == 0:
            jiou.append(1)
        else:
            jiou.append(0)
    jiou = torch.tensor(jiou).float().to(DEVICE)
    zl0 = torch.zeros_like(z).to(DEVICE)
    zl0[z > 0] = 1
    case1_indexes = zl0 * jiou
    case2_indexes = 1 - case1_indexes
    case = torch.pow(
        10,
        torch.tensor([0.5 * (i / (x.shape[-1] - 1)) for i in range(z.shape[-1])])
        .float()
        .to(DEVICE),
    )
    case1 = 10 * case
    case2 = case
    s = case1 * case1_indexes + case2 * case2_indexes
    z = s * z
    r = (
        10 * (z.shape[-1] - torch.sum(torch.cos(2 * np.pi * z), dim=-1))
        + torch.sum(torch.pow(z, 2), dim=-1)
        + 100 * fpen(x)
        + fopt
    )
    return r


FUNCTIONS[4] = {"fid": 4, "fun": f4, "xopt": None, "fopt": None}


def f5(x=None, xopt=None, fopt=None):
    """
    xopt=zopt=5*f1_1(d)
    """
    z = x.clone()
    case1_index = torch.zeros_like(x).to(DEVICE)
    b1 = xopt * x
    case1_index[b1 < 25] = 1
    case2_index = 1 - case1_index
    z = x * case1_index + xopt * case2_index
    s = fsign(xopt) * torch.pow(
        10,
        torch.tensor([(i / (x.shape[-1] - 1)) for i in range(z.shape[-1])])
        .float()
        .to(DEVICE),
    )
    r = torch.sum(5 * torch.abs(s) - s * z, dim=-1) + fopt
    return r


FUNCTIONS[5] = {"fid": 5, "fun": f5, "xopt": None, "fopt": None}


def f6(x=None, xopt=None, fopt=None, Q=None):
    z = (Q @ fjianAlpha(10, x.shape[-1]) @ Q @ (x - xopt).transpose(-1, -2)).transpose(
        -1, -2
    )
    s = torch.ones_like(z).to(DEVICE)
    tmp = z * xopt
    s[tmp > 0] = 100
    r = fTosz(torch.pow(torch.sum(torch.pow(s * z, 2), dim=-1), 0.9)) + fopt
    return r


FUNCTIONS[6] = {"fid": 6, "fun": f6, "xopt": None, "fopt": None, "Q": None}


def f7(x=None, xopt=None, fopt=None, Q=None, R=None):
    z_hat = (fjianAlpha(10, x.shape[-1]) @ R @ (x - xopt).transpose(-1, -2)).transpose(
        -1, -2
    )
    z_b1 = torch.floor(z_hat + 0.5)
    z_b2 = torch.floor(0.5 + 10 * z_hat) / 10
    case1_index = torch.zeros_like(z_hat).to(DEVICE)
    case1_index[z_hat > 0.5] = 1
    case2_index = 1 - case1_index
    zb = z_b1 * case1_index + z_b2 * case2_index
    z = (Q @ zb.transpose(-1, -2)).transpose(-1, -2)
    item1 = torch.abs(z_hat[..., 0]) / 10e4
    item2 = torch.sum(
        torch.pow(
            10,
            torch.tensor([2 * i / (x.shape[-1] - 1) for i in range(x.shape[-1])])
            .float()
            .to(DEVICE),
        )
        * torch.pow(z, 2),
        dim=-1,
    )
    mask = item2 - item1
    case1_index = torch.zeros_like(mask).to(DEVICE)
    case1_index[mask > 0] = 1  # item2>item1
    case2_index = 1 - case1_index
    item2 = item2 * case1_index + item1 * case2_index
    r = 0.1 * item2 + fpen(x) + fopt
    return r


FUNCTIONS[7] = {"fid": 7, "fun": f7, "xopt": None, "fopt": None, "Q": None, "R": None}


def f8(x=None, xopt=None, fopt=None):
    """
    zopt=(1,1,...,1)
    """
    z = x - xopt
    if x.shape[-1] >= 64:
        z = (
            torch.sqrt(torch.tensor(x.shape[-1], dtype=torch.float, device=DEVICE))
            / 8
            * z
        )
    z = z + 1
    z1 = z[:, :-1]
    z2 = z[:, 1:]
    r = (
        torch.sum(
            100 * torch.pow(torch.pow(z1, 2) - z2, 2) + torch.pow(z1 - 1, 2), dim=-1
        )
        + fopt
    )

    return r


FUNCTIONS[8] = {"fid": 8, "fun": f8, "xopt": None, "fopt": None}


def f9(x=None, fopt=None, R=None):
    """
    zopt=(1,1,...,1)
    """
    z = x.clone()
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    if x.shape[-1] >= 64:
        z = (
            torch.sqrt(torch.tensor(x.shape[-1], dtype=torch.float, device=DEVICE))
            / 8
            * z
        )
    z = z + torch.ones_like(z).to(DEVICE) / 2
    z1 = z[:, :-1]
    z2 = z[:, 1:]
    r = (
        torch.sum(
            100 * torch.pow(torch.pow(z1, 2) - z2, 2) + torch.pow(z1 - 1, 2), dim=-1
        )
        + fopt
    )
    return r


FUNCTIONS[9] = {"fid": 9, "fun": f9, "fopt": None, "R": None}


def f10(x=None, xopt=None, fopt=None, R=None):
    z = fTosz((R @ (x - xopt).transpose(-1, -2)).transpose(-1, -2))
    case = torch.pow(
        10,
        torch.tensor([6 * (i / (x.shape[-1] - 1)) for i in range(z.shape[-1])])
        .float()
        .to(DEVICE),
    )
    r = torch.sum(case * torch.pow(z, 2), dim=-1) + fopt
    return r


FUNCTIONS[10] = {"fid": 10, "fun": f10, "xopt": None, "fopt": None, "R": None}


def f11(x=None, xopt=None, fopt=None, R=None):
    z = fTosz((R @ (x - xopt).transpose(-1, -2)).transpose(-1, -2))
    z1 = 10e6 * torch.pow(z[:, 0], 2)
    r = z1 + torch.sum(torch.pow(z[:, 1:], 2), dim=-1) + fopt
    return r


FUNCTIONS[11] = {"fid": 11, "fun": f11, "xopt": None, "fopt": None, "R": None}


def f12(x=None, xopt=None, fopt=None, R=None):
    z = x - xopt
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z = fTasy(z, 0.5)
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z1 = torch.pow(z[:, 0], 2)
    r = z1 + 10e6 * torch.sum(torch.pow(z[:, 1:], 2), dim=-1) + fopt
    return r


FUNCTIONS[12] = {"fid": 12, "fun": f12, "xopt": None, "fopt": None, "R": None}


def f13(x=None, xopt=None, fopt=None, Q=None, R=None):
    z = x - xopt
    z = ((Q @ z.transpose(-1, -2)).transpose(-1, -2)) @ R
    r = (
        torch.pow(z[:, 0], 2)
        + 100 * torch.sqrt(torch.sum(torch.pow(z[:, 1:], 2), dim=-1))
        + fopt
    )
    return r


FUNCTIONS[13] = {
    "fid": 13,
    "fun": f13,
    "xopt": None,
    "fopt": None,
    "R": None,
    "Q": None,
}


def f14(x=None, xopt=None, fopt=None, R=None):
    z = x - xopt
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    exp = (
        torch.tensor([4 * i / (x.shape[-1] - 1) for i in range(z.shape[-1])])
        .float()
        .to(DEVICE)
        + 2
    )
    r = torch.sqrt(torch.sum(torch.pow(torch.abs(z), exp), dim=-1)) + fopt
    return r


FUNCTIONS[14] = {
    "fid": 14,
    "fun": f14,
    "xopt": None,
    "fopt": None,
    "R": None,
}


def f15(x=None, xopt=None, fopt=None, Q=None, R=None):
    z = x - xopt
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z = fTosz(z)
    z = fTasy(z, 0.2)
    z = (Q @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (fjianAlpha(10, z.shape[-1]) @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    r = (
        10 * (x.shape[-1] - torch.sum(torch.cos(2 * np.pi * z), dim=-1))
        + torch.sum(torch.pow(z, 2), dim=-1)
        + fopt
    )
    return r


FUNCTIONS[15] = {
    "fid": 15,
    "fun": f15,
    "xopt": None,
    "fopt": None,
    "R": None,
    "Q": None,
}


def f16(x=None, xopt=None, fopt=None, Q=None, R=None):
    z = x - xopt
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z = fTosz(z)
    z = (Q @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (fjianAlpha(0.01, z.shape[-1]) @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    k = torch.tensor([i for i in range(12)]).float().to(DEVICE)
    f = torch.sum(
        torch.pow(0.5, k) * torch.cos(2 * np.pi * torch.pow(3, k) * 0.5), dim=-1
    )
    z = z + 0.5  # n,dim
    z = torch.unsqueeze(z, dim=1)
    z = z.repeat(1, 12, 1)
    k = k.view(-1, 1)
    item = torch.pow(0.5, k) * torch.cos(2 * np.pi * torch.pow(3, k) * (z + 0.5))
    item = torch.sum(item, dim=-2)
    item = torch.sum(item, dim=-1)
    r = (
        10 * (torch.pow(1.0 / z.shape[-1] * item - f, 3))
        + 10.0 / x.shape[-1] * fpen(x)
        + fopt
    )
    return r


FUNCTIONS[16] = {
    "fid": 16,
    "fun": f16,
    "xopt": None,
    "fopt": None,
    "R": None,
    "Q": None,
}


def f17(x=None, xopt=None, fopt=None, Q=None, R=None):
    z = x - xopt
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z = fTasy(z, 0.5)
    z = (Q @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (fjianAlpha(10, z.shape[-1]) @ z.transpose(-1, -2)).transpose(-1, -2)
    s = torch.sqrt(torch.pow(z[:, :-1], 2) + torch.pow(z[:, 1:], 2))
    item = (
        1.0
        / s.shape[-1]
        * torch.sum(
            torch.sqrt(s) * (1 + torch.pow(torch.sin(50 * torch.pow(s, 0.2)), 2)),
            dim=-1,
        )
    )
    r = torch.pow(item, 2) + 10 * fpen(x) + fopt
    return r


FUNCTIONS[17] = {
    "fid": 17,
    "fun": f17,
    "xopt": None,
    "fopt": None,
    "R": None,
    "Q": None,
}


def f18(x=None, xopt=None, fopt=None, Q=None, R=None):
    z = x - xopt
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z = fTasy(z, 0.5)
    z = (Q @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (fjianAlpha(1000, z.shape[-1]) @ z.transpose(-1, -2)).transpose(-1, -2)
    s = torch.sqrt(torch.pow(z[:, :-1], 2) + torch.pow(z[:, 1:], 2))
    item = (
        1.0
        / s.shape[-1]
        * torch.sum(
            torch.sqrt(s) * (1 + torch.pow(torch.sin(50 * torch.pow(s, 0.2)), 2)),
            dim=-1,
        )
    )
    r = torch.pow(item, 2) + 10 * fpen(x) + fopt
    return r


FUNCTIONS[18] = {
    "fid": 18,
    "fun": f18,
    "xopt": None,
    "fopt": None,
    "R": None,
    "Q": None,
}


def f19(x=None, fopt=None, R=None):
    """
    zopt=[1,1,...,1]
    """
    z = x.clone()
    z = (R @ z.transpose(-1, -2)).transpose(-1, -2)
    z = (
        torch.max(
            torch.tensor(
                [
                    1.0,
                    torch.sqrt(
                        torch.tensor(x.shape[-1], dtype=torch.float, device=DEVICE)
                    )
                    / 8,
                ]
            )
            .float()
            .to(DEVICE)
        )
        * z
        + 0.5
    )
    s = 100 * torch.pow(torch.pow(z[:, :-1], 2) - z[:, 1:], 2) + torch.pow(
        z[:, :-1] - 1, 2
    )
    item = s / 4000.0 - torch.cos(s)
    r = 10.0 / item.shape[-1] * torch.sum(item, dim=-1) + 10 + fopt
    return r


FUNCTIONS[19] = {
    "fid": 19,
    "fun": f19,
    "fopt": None,
    "R": None,
}

def f20(x=None, xopt=None, fopt=None):

    x_hat = 2 * f1_1(x.shape[-1]) * x
    xh1 = x_hat[:, :-1]
    xh2 = x_hat[:, 1:]
    zhat = xh2 + 0.25 * (xh1 - 2 * torch.abs(xopt.view(1, -1)[:, :-1]))
    zhat = torch.cat((x_hat[:, 0].view(-1, 1), zhat[:, :]), dim=-1)
    item = (
        fjianAlpha(10, zhat.shape[-1]) @ (zhat - 2 * torch.abs(xopt)).transpose(-1, -2)
    ).transpose(-1, -2)
    z = 100 * (item + 2 * torch.abs(xopt))
    i1 = -1.0 / (100 * x.shape[-1]) * torch.sum(z * torch.sin(torch.sqrt(torch.abs(z))))
    r = i1 + 4.189828872724339 + 100 * fpen(z / 100) + fopt
    return r


FUNCTIONS[20] = {
    "fid": 20,
    "fun": f20,
    "xopt": None,
    "fopt": None,
}

w21 = torch.tensor([i for i in range(101)]).float().to(DEVICE)
w21[0] = 10.0
w21[1:] = 1.1 + 8 * (w21[1:] - 1) / 99


def f21(x=None, y=None, fopt=None, R=None, C=None):
    """Gallagher's Gaussian 101-me peaks.

    C: (101, dim, dim) diagonal matrices
    y: (101, dim)
    R: (dim, dim)

    Notes:
      - The original implementation used chained matmul with broadcasting which may
        materialize huge intermediate tensors at high dimensions.
      - Here we compute the equivalent quadratic form in a memory-efficient way.
    """
    n, d = x.shape[0], x.shape[-1]

    # (n, 101, dim)
    diff = x.unsqueeze(1) - y.unsqueeze(0)

    # Apply shared rotation without broadcasting to (n, 101, dim, dim).
    diff_rot = (diff.reshape(-1, d) @ R.transpose(-1, -2)).reshape(n, diff.shape[1], d)

    # C is diagonal by construction; only take its diagonal.
    C_diag = torch.diagonal(C, dim1=-2, dim2=-1)  # (101, dim)

    quad = torch.sum((diff_rot * diff_rot) * C_diag.unsqueeze(0), dim=-1)  # (n, 101)

    item = -1.0 / (2 * d) * quad
    item = w21 * torch.exp(item)
    item = 10 - torch.max(item, dim=-1)[0]
    item = fTosz(item) ** 2
    r = item + fpen(x) + fopt
    return r


FUNCTIONS[21] = {"fid": 21, "fun": f21, "fopt": None, "R": None, "C": None, "y": None}


w22 = torch.tensor([i for i in range(21)]).float().to(DEVICE)
w22[0] = 10.0
w22[1:] = 1.1 + 8 * (w22[1:] - 1) / 19


def f22(x, y=None, fopt=None, R=None, C=None):
    """Gallagher's Gaussian 21-hi peaks.

    C: (21, dim, dim) diagonal matrices
    y: (21, dim)
    R: (dim, dim)

    Notes: see f21.
    """
    n, d = x.shape[0], x.shape[-1]

    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, 21, dim)
    diff_rot = (diff.reshape(-1, d) @ R.transpose(-1, -2)).reshape(n, diff.shape[1], d)

    C_diag = torch.diagonal(C, dim1=-2, dim2=-1)  # (21, dim)
    quad = torch.sum((diff_rot * diff_rot) * C_diag.unsqueeze(0), dim=-1)  # (n, 21)

    item = -1.0 / (2 * d) * quad
    item = w22 * torch.exp(item)
    item = 10 - torch.max(item, dim=-1)[0]
    item = fTosz(item) ** 2
    r = item + fpen(x) + fopt
    return r


FUNCTIONS[22] = {"fid": 22, "fun": f22, "fopt": None, "R": None, "C": None, "y": None}


def f23(x, xopt=None, fopt=None, Q=None, R=None):
    x = x - xopt
    x0 = x.transpose(-1, -2)
    z = (Q @ fjianAlpha(100, x.shape[-1]) @ R @ x0).transpose(-1, -2)
    d = x.shape[-1]
    index = torch.tensor([i for i in range(1, 32 + 1)]).float().to(DEVICE)
    e = torch.pow(2, index).view(32, 1)
    z = torch.unsqueeze(z, dim=-2).repeat(1, 32, 1)
    i1 = e * z
    i2 = torch.round(i1)
    item = torch.sum(torch.abs(i1 - i2) / e, dim=-2)
    index = torch.tensor([i for i in range(1, d + 1)]).float().to(DEVICE)
    item = 1 + index * item
    item = torch.prod(
        torch.pow(
            torch.tensor(item, dtype=torch.float, device=DEVICE),
            10.0 / (torch.pow(torch.tensor(d, dtype=torch.float, device=DEVICE), 1.2)),
        ),
        dim=-1,
    )
    r = 10.0 / (d**2) * (item - 1) + fpen(x) + fopt
    return r


FUNCTIONS[23] = {
    "fid": 23,
    "fun": f23,
    "xopt": None,
    "fopt": None,
    "R": None,
    "Q": None,
}


def f24(x, xopt=None, fopt=None, Q=None, R=None):
    """
    xopt=miu0/2*f0_1(x.shape[-1])
    """
    global xpt
    D = x.shape[-1]
    # if xpt is None:
    #     xpt=1.25*f1_1(x.shape[-1])
    # xopt=xpt
    s = 1 - 1.0 / (2 * math.sqrt(D + 20) - 8.2)
    miu0 = 2.5
    miu1 = -math.sqrt((miu0**2 - 1) / (s))
    xhat = 2 * fsign(xopt) * x
    xhat1 = xhat.transpose(-1, -2)
    z = (Q @ fjianAlpha(100, D) @ R @ (xhat1 - miu0)).transpose(-1, -2)
    item1 = torch.sum(torch.pow(xhat - miu0, 2), dim=-1)  # n,1
    item2 = D + s * torch.sum(torch.pow(xhat - miu1, 2), dim=-1)  # n,1
    item = torch.cat((item1.view(-1, 1), item2.view(-1, 1)), dim=-1)  # (n,2)
    i1 = torch.min(item, dim=-1)[0]
    i2 = 10 * (D - torch.sum(torch.cos(2 * math.pi * z), dim=-1))
    r = i1 + i2 + 10e4 * fpen(x) + fopt
    return r


FUNCTIONS[24] = {"fid": 24, "fun": f24, "fopt": None, "R": None, "Q": None}


for fid, fun in FUNCTIONS.items():
    fun["train_popsize"] = 100
    fun["train_problemdim"] = 10


if __name__ == "__main__":
    b = torch.randn(3, 5) * 5
    print(
        f24(
            b,
            xopt=torch.randn(
                5,
            )
            * 5,
            fopt=100,
            Q=torch.randn(5, 5),
            R=torch.randn(5, 5),
        )
    )
