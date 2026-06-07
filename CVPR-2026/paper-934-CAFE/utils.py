import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import os
import scipy.io as sio
import numpy as np


def mse_fn(pred, gt):
    return ((pred - gt) ** 2).mean()


def psnr_fn(pred, gt):
    return -10. * torch.log10(mse_fn(pred, gt))


def get_mgrid(sidelen, dim=2):

    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def img2tensor(path, sidelength, normalize=True):

    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        # 处理MAT文件
        mat_data = sio.loadmat(path)

        img = mat_data['img']

        if img is None:
            raise ValueError(f"Please modify the key.")

        if img.ndim == 2:
            img = img[..., None]
        elif img.ndim == 4:
            img = img[0]

        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')

    else:
        img = Image.open(path).convert('RGB')

    transforms_list = [
        Resize((sidelength, sidelength)),
        ToTensor(),  # 转换为 (C, H, W) 且范围 [0, 1]
    ]

    if normalize:
        transforms_list.append(Normalize(torch.Tensor([0.5,]), torch.Tensor([0.5,])))
        print("Normalize image, please Denormalize image when showing image and calculating psnr metrix.")
    transform = Compose(transforms_list)
    img_tensor = transform(img)

    return img_tensor


class ImageFitting(Dataset):
    def __init__(self, path, sidelength, normalize=True):
        super().__init__()
        img = img2tensor(path, sidelength, normalize=normalize)  # (C, sidelength, sidelength)
        C, H, W = img.shape

        self.pixels = img.permute(1, 2, 0).reshape(H * W, C)  # (sidelength*sidelength, C)
        self.coords = get_mgrid(sidelength, 2)  # (sidelength*sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels


def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x

    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin) / (xmax - xmin)

    return xnormalized


def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2))
    snrval = 10*np.log10(np.max(x)/denom)

    return snrval
