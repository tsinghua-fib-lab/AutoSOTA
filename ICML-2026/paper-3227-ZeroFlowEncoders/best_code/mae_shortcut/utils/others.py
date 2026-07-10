import random
import torch
import numpy as np


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    pass

def log_mae_recon(writer, epoch, img, recon_img, mask_img=None):
    writer.add_image("input", img[0], epoch)
    writer.add_image("reconstruction", recon_img[0], epoch)
    if mask_img is not None:
        writer.add_image("mask", mask_img[0], epoch)