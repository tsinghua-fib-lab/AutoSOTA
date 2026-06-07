import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import math
import warnings
import mcubes
from torch.func import functional_call, vmap, vjp, jvp, jacrev


# produce 2D grid coordinate
def get_mgrid(H, W, dim=2):
    tensors = (torch.linspace(-1, 1, steps=H), torch.linspace(-1, 1, steps=W))
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_coords(H, W, T=None):
    '''
        Get 3D coordinates
    '''
    if T is None:
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))

    return torch.tensor(coords.astype(np.float32))


# convert to image tensor
def get_image_tensor(fig):
    img = Image.fromarray(fig)
    transform = Compose([
        ToTensor()
        # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


# image fitting dataset
class ImageFitting(Dataset):
    def __init__(self, H, W, fig):
        super().__init__()
        img = get_image_tensor(fig)
        self.coords = get_mgrid(H, W, 2)
        self.pixels = img.permute(1, 2, 0).view(-1, 3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


# 1d_function dataset
class CustomDataset(Dataset):
    def __init__(self, min_val, max_val, num_samples):
        self.num_samples = num_samples
        self.x = torch.linspace(min_val, max_val, num_samples)
        self.x = torch.reshape(self.x, (self.num_samples, 1))
        self.y = 2 * torch.round((torch.sin(3 * math.pi * self.x) + torch.sin(5 * math.pi * self.x) + torch.sin(
            7 * math.pi * self.x) + torch.sin(9 * math.pi * self.x)) / 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if idx>0: raise IndexError
        return self.x[idx], self.y[idx]


def fnet_single(params, x):
    return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)


# the realization of emprical NTK matrix
def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False

    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def get_IoU(preds, gt, thres=None):
    intersection, union = get_I_and_U(preds, gt, thres)
    return intersection / union


def get_I_and_U(preds, gt, thres=None):
    if thres is not None:
        preds[preds < thres] = 0.0
        preds[preds >= thres] = 1.0

    if isinstance(preds, np.ndarray):
        intersection = np.logical_and(preds, gt).sum()
        union = np.logical_or(preds, gt).sum()
    else:
        intersection = torch.logical_and(preds.cuda(), gt.cuda()).sum()
        union = torch.logical_or(preds.cuda(), gt.cuda()).sum()
    # intersection = np.logical_and(preds, gt).sum()
    # union =  np.logical_or(preds, gt).sum()
    return intersection, union


def march_and_save(occupancy, mcubes_thres, savename, smoothen=False):
    '''
        Convert volumetric occupancy cube to a 3D mesh

        Inputs:
            occupancy: (H, W, T) occupancy volume with values going from 0 to 1
            mcubes_thres: Threshold for marching cubes algorithm
            savename: DAE file name to save
            smoothen: If True, the mesh is binarized, smoothened, and then the
                marching cubes is applied
        Outputs:
            None
    '''
    if smoothen:
        occupancy = occupancy.copy()
        occupancy[occupancy < mcubes_thres] = 0.0
        occupancy[occupancy >= mcubes_thres] = 1.0
        # from scipy.ndimage import gaussian_filter
        # occupancy = gaussian_filter(occupancy, sigma=1)
        occupancy = mcubes.smooth(occupancy, method='gaussian', sigma=1)
        mcubes_thres = 0

    vertices, faces = mcubes.marching_cubes(occupancy, mcubes_thres)

    # vertices /= occupancy.shape[0]

    mcubes.export_mesh(vertices, faces, savename)