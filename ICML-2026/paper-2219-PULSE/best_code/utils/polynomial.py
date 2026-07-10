import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial import Chebyshev as C
from numpy.polynomial import Hermite as H
from numpy.polynomial import Laguerre as La
from numpy.polynomial import Legendre as L


def standard_laguerre(data, degree):
    tvals = np.linspace(0, 5, len(data))
    coeffs = La.fit(tvals, data, degree).coef

    laguerre_poly = La(coeffs)
    reconstructed_data = laguerre_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def laguerre_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(0, 5, T)
    laguerre_polys = np.array([La.basis(i)(tvals) for i in range(degree)])

    laguerre_polys = torch.from_numpy(
        laguerre_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals))
    coeffs_candidate = torch.mm(laguerre_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(laguerre_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, laguerre_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_hermite(data, degree):
    tvals = np.linspace(-5, 5, len(data))
    coeffs = H.fit(tvals, data, degree).coef

    hermite_poly = H(coeffs)
    reconstructed_data = hermite_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def hermite_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-5, 5, T)
    hermite_polys = np.array([H.basis(i)(tvals) for i in range(degree)])

    hermite_polys = torch.from_numpy(
        hermite_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals ** 2))
    coeffs_candidate = torch.mm(hermite_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(hermite_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, hermite_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_leg(data, degree):
    tvals = np.linspace(-1, 1, len(data))
    coeffs = L.fit(tvals, data, degree).coef

    legendre_poly = L(coeffs)
    reconstructed_data = legendre_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def leg_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)  # The Legendre series are defined in t\in[-1, 1]
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(degree)])  # Generate the basis functions which are sampled at tvals.
    # tvals = torch.from_numpy(tvals).to(device)
    legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]

    # This is implemented for 1D series. 
    # For N-D series, here, the data matrix should be transformed as B,T,D -> B,D,T -> BD, T. 
    # The legendre polys should be T,degree
    # Then, the dot should be a matrix multiplication: (BD, T) * (T, degree) -> BD, degree, which is the result of legendre transform.
    coeffs_candidate = torch.mm(legendre_polys, data) / T * 2
    coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(degree)]).to(device)
    coeffs = coeffs.transpose(0, 1)  # shape: [B * D, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_chebyshev(data, degree):
    tvals = np.linspace(-1, 1, len(data))
    coeffs = C.fit(tvals, data, degree).coef

    chebyshev_poly = C(coeffs)
    reconstructed_data = chebyshev_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def chebyshev_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)
    chebyshev_polys = np.array([C.basis(i)(tvals) for i in range(degree)])

    chebyshev_polys = torch.from_numpy(chebyshev_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(1 / torch.sqrt(1 - tvals ** 2))
    coeffs_candidate = torch.mm(chebyshev_polys, data) / torch.pi / T * 2
    # coeffs_candidate = torch.mm(torch.mm(chebyshev_polys, scale), data) / torch.pi * 2
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(chebyshev_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, chebyshev_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


if __name__ == "__main__":
    data = np.linspace(-1, 1, 360)
    data1 = data ** 3 + data * 2
    data2 = data ** 2 + data * 3
    data3 = data ** 5 + data
    data_b1 = np.stack([data1, data2, data3], axis=1)
    data_b2 = data_b1 + 5
    data = np.stack([data_b1, data_b2], axis=0)
    print(data.shape)

    d = 1
    degree = 5
    coeffs_std, reconstructed_std = standard_leg(data[1, :, d], degree)
    print(coeffs_std)

    device = 'cuda:0'
    _data = torch.from_numpy(data).to(device)
    coeffs_torch, reconstructed_torch = leg_torch(_data, degree, rtn_data=True, device=device)
    print(coeffs_torch)

    import matplotlib.pyplot as plt
    plt.plot(data[1, :, d])
    plt.plot(reconstructed_std)
    plt.plot(reconstructed_torch[1, :, d].cpu())
    plt.legend(['data', 'std', 'torch'])