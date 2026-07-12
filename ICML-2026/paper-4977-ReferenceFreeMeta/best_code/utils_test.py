import os
import torch
import numpy as np
from utils import *
import h5py
from numpy import fft
from matplotlib import pyplot as plt

from SIREN_IPOD_train import SirenModel

import time
import torchkbnufft as tkbn 
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.optim import lr_scheduler


# Set random seed for reproducibility
import random
def set_seed(seed=42):
    random.seed(seed)              # Python random
    np.random.seed(seed)           # Numpy random
    torch.manual_seed(seed)        # CPU random
    torch.cuda.manual_seed(seed)   # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure convolution determinism
    torch.backends.cudnn.benchmark = False     # Disable automatic optimization search

# Set random seed
set_seed(35236)

def run_test_siren(mask, GT, GT_ksp_tensor, mask_tensor, csmp_tensor, DEVICE, epoch, 
                   step_size=500, lr=1e-4, checkpoint_path=None, save_dir=None, 
                   save_iterations=None, gamma=0.8, TV_weight=2):
    
    model = SirenModel().to(DEVICE)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('All parameters loaded')
        
    fn = lambda x: normalize01(np.abs(x))
    
    # Create meta optimizer - not directly used in Reptile, but kept for learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    os.makedirs(save_dir, exist_ok=True)
    
    if checkpoint_path is not None:
        if "ablation" in checkpoint_path.lower():  # Case insensitive
            method = 'Siren_Meta_Abla_'
        else:
            method = 'Siren_Meta_'
    else:
        method = 'Siren_'
    
    # Create learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    MAE_loss_function = torch.nn.L1Loss()
    TV_loss_function = MYTVLoss()
    
    nRow, nCol, nCoil = GT_ksp_tensor.shape
    coor_tensor = torch.from_numpy(build_coordinate_train(L_RO=nRow, L_PE=nCol)).to(DEVICE).float()   
    losses = []
    psnrs = []
    ssims = []
    
    normOrg = fn(GT)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normOrg, cmap='gray', vmin=0, vmax=0.7)
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    gt_save_path = os.path.join(save_dir, 'ground_truth.png')
    plt.savefig(gt_save_path, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black', edgecolor='black')

    plt.close()
    print(f"Ground truth saved to: {gt_save_path}")
    
    t_start = time.time()
    
    for e in range(epoch):
        pre_intensity_mag, pre_intensity_phi = model.forward(coor_tensor.view(-1, 2))
        pre_intensity = torch.complex(pre_intensity_mag.view(nRow, nCol, 1), 
                                     pre_intensity_phi.view(nRow, nCol, 1))
        
        # Calculate multi-channel image (multiply with csmp)
        pre_intensity_multi = pre_intensity * csmp_tensor
        
        # Calculate k-space representation
        fft_pre_intensity = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.fftshift(pre_intensity_multi, dim=(0, 1)), 
                dim=(0, 1)
            ), 
            dim=(0, 1)
        )
        
        # Calculate loss (only at sampled locations)
        mae_ksp_loss = MAE_loss_function(
            torch.view_as_real(fft_pre_intensity[mask_tensor==1]).float(), 
            torch.view_as_real(GT_ksp_tensor[mask_tensor==1]).float()
        )
        TV_loss = TV_loss_function(pre_intensity_mag.view(nRow, nCol, 1)) + \
                  TV_loss_function(pre_intensity_phi.view(nRow, nCol, 1))
        loss = mae_ksp_loss + TV_weight * TV_loss
        
        # Backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record and print loss
        print('(TRAIN0) Epoch[{}/{}], Lr:{}, Loss:{:.6f}'.format(
            e+1, epoch, scheduler.get_lr()[0], loss.item()))
        losses.append(loss.item())
        scheduler.step()
       
        with torch.no_grad():
            normRec = fn(pre_intensity.detach().cpu().squeeze().numpy())
            psnrRec = psnr(normOrg * mask, normRec * mask)
            psnrs.append(psnrRec)
            
            ssimRec = ssim(normOrg * mask, normRec * mask)
            ssims.append(ssimRec)
            
            if (e + 1) in save_iterations:
                save_reconstruction_images(normRec * mask, normOrg * mask, e + 1, 
                                         save_dir, method=method)
    
    t_end = time.time()
    
    with torch.no_grad():
        model.eval()
              
        # Forward propagation
        pre_intensity_mag, pre_intensity_phi = model.forward(coor_tensor.view(-1, 2))
        pre_intensity = torch.complex(pre_intensity_mag.view(nRow, nCol, 1), 
                                     pre_intensity_phi.view(nRow, nCol, 1))
        
        # Calculate multi-channel image
        pre_intensity_multi = pre_intensity * csmp_tensor
        
        # FFT transform to k-space
        fft_pre_intensity = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.fftshift(pre_intensity_multi, dim=(0, 1)), 
                dim=(0, 1)
            ), 
            dim=(0, 1)
        )
        
        # Convert to numpy arrays
        pre_ksp = fft_pre_intensity.cpu().detach().numpy().reshape(nRow, nCol, nCoil)
        pre_img = pre_intensity.cpu().detach().numpy().reshape(nRow, nCol)

        with torch.no_grad():
            normRec = fn(pre_img)
            psnrRec = psnr(normOrg * mask, normRec * mask)
            psnrs.append(psnrRec)
            
            ssimRec = ssim(normOrg * mask, normRec * mask)
            ssims.append(ssimRec)
            
            if (e + 1) in save_iterations:
                save_reconstruction_images(normRec * mask, normOrg * mask, e + 1, 
                                         save_dir, method=method)
    
    print(f'Complete in {t_end-t_start}s')
    return losses, pre_img, psnrs, ssims