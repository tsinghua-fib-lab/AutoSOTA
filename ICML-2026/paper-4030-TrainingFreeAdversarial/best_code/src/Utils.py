import re
import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
import matplotlib.patheffects as path_effects
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA/cuDNN more deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def L2_Loss(recon, label):
    return torch.norm(recon-label , p=2) / torch.norm( label, p=2)

def LogL2_Loss(recon, label, eps=1e-8):
    """Log-scaled L2 loss: compresses dynamic range for frequency-balanced gradients.
    log(1+|x-y|) reduces the dominance of high-energy low-frequency components."""
    diff = torch.abs(recon - label)
    ref = torch.abs(label)
    return torch.norm(torch.log(1 + diff + eps), p=2) / (torch.norm(torch.log(1 + ref + eps), p=2) + eps)


def fft_loss(zf, coil, axis):
    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(zf * coil, dim=axis), dim=axis, norm='ortho'), dim=axis)

def FFT(image, axis=[-2,-1]):
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(image,dim=axis), dim=axis, norm = 'ortho'), dim=axis)
    
def IFFT(kspace, axis=[-2,-1]):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(kspace,dim = axis), dim=axis, norm = 'ortho'),dim = axis)

def get_iter_num(fname):
    return int(re.search(r"iter_(\d+)\.png", fname).group(1))

def read_gif(path):
    gif = Image.open(path)
    return [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]

def resize_to_height(img, height):
    w, h = img.size
    return img.resize((int(w * height / h), height), Image.LANCZOS)

def getssim(recon,ref):
    data_range = np.amax(np.abs(ref)) - np.amin(np.abs(ref))
    ssim_      = ssim(np.abs(recon), np.abs(ref), data_range=data_range)
    return ssim_

def getpsnr(recon, ref): 
    data_range = np.amax(np.abs(ref)) - np.amin(np.abs(ref))
    psnr_      = psnr(np.abs(ref), np.abs(recon), data_range=data_range)
    return psnr_

def Reverse_PGD_Projection(zf, zf_p, Grad, config, alpha):
    zf = zf - alpha* Grad.sgn()
    zf = torch.min( torch.real(zf_p + config["Mitigation"]["epsilon_proj"]), torch.max( torch.real(zf_p - config["Mitigation"]["epsilon_proj"]) , torch.real(zf))) + 1j*torch.min( torch.imag(zf_p + config["Mitigation"]["epsilon_proj"]), torch.max( torch.imag(zf_p - config["Mitigation"]["epsilon_proj"]) , torch.imag(zf)))
    zf.retain_grad()    
    return zf

def exp_alpha_scheduler(initial_alpha, current_iter, decay_rate=0.95):
    return initial_alpha * (decay_rate ** current_iter)

def linear_alpha_scheduler(initial_alpha, current_iter, total_iters):
    factor = 1 - (current_iter / total_iters)
    return initial_alpha * factor

def cosine_alpha_scheduler(initial_alpha, current_iter, total_iters):
    """Cosine annealing: alpha = alpha_0 * 0.5 * (1 + cos(pi * t/T))"""
    return initial_alpha * 0.5 * (1 + math.cos(math.pi * current_iter / total_iters))


class CG_EEH(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def EEH(self, kspace, Coil, Mask):
        return FFT(torch.sum(IFFT(kspace * Mask) * torch.conj(Coil), dim=1,keepdim=True) * Coil) * Mask
    
    def forward(self, ksp, Coil, Mask):
        p_now = ksp
        r_now = torch.clone(p_now)
        b_approx = torch.zeros_like(p_now)
        
        for i in range(10):
            q = self.EEH(p_now,Coil,Mask)
            rrOverpq = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))  # rrOverpq = (r'*r)/(p'*q);
            b_next = b_approx + rrOverpq*p_now
            r_next = r_now - rrOverpq*q;   
            p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now # p = r_next + ( (r_next'*r_next)/(r'*r) )*p;
            b_approx = b_next
            p_now = torch.clone(p_next)
            r_now = torch.clone(r_next)
        return b_approx


def ABA_detect(zf_temp, coil, Masks, model, config, jitt):

    Recons_mit = torch.zeros([config["kx"], config["ky"], config["Mitigation"]["cyclic_stages"]]              , dtype = torch.complex64)
    err_ksp    = torch.zeros([config["nc"], config["kx"], config["ky"], config["Mitigation"]["cyclic_stages"]], dtype = torch.complex64)
    
    for i in range(config["Mitigation"]["cyclic_stages"]):
        rec               = model(zf_temp,coil,Masks[i])
        Recons_mit[:,:,i] = rec
        ksp_recon         = fft_loss(rec, coil, config["axis"]) 
        err_ksp [:,:,:,i] = ksp_recon * Masks[0]
        ksp_recon         = ksp_recon + jitt

        if i != (config["Mitigation"]["cyclic_stages"]-1):
            zf_temp = torch.sum(torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp_recon * Masks[i + 1] , dim=config["axis"]), dim=config["axis"],norm='ortho'), dim=config["axis"]) * torch.conj(coil), dim=1, keepdim=True)
        else:
            zf_temp = torch.sum(torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(ksp_recon * Masks[0], dim=config["axis"]), dim=config["axis"],norm='ortho'), dim=config["axis"]) * torch.conj(coil), dim=1, keepdim=True) 
    return Recons_mit, err_ksp


def noise_jiterring(y_omega, Ex1, mask, std_scale=1): 
    _,kx,ky          = mask.shape 
    nc               = y_omega.shape[1]
    jitt             = torch.zeros([1, nc, kx, ky], dtype=torch.complex64)
    diff_1           = y_omega - Ex1
    mask             = mask[0, :, 0]
    non_zero_indices = torch.nonzero(mask).squeeze()
    diff             = diff_1[:, :, non_zero_indices, :]  

    for i in range(nc):
        temp             = diff[:, i, :, :]
        std              = torch.sqrt(torch.norm(temp,2)**2/(temp.numel()))
        noise            = torch.normal(0, std_scale * std.item(), size=(1, kx, ky))  # Termal noise, zero-mean
        jitt[:, i, :, :] = (noise + 1j*noise) * np.sqrt(2)/2
    return jitt

def attack_generation(zero_filled, network, device, Omega_Mask, coil, config):
    
    zf = zero_filled.clone()
    zf.retain_grad()
    alpha = float(config["Attack"]["epsilon"])/5

    if config["Attack"]["mode"] == "PGD":
        zf_eta = zf + (1/math.sqrt(2))*(torch.tensor(np.random.uniform(low=-config["Attack"]["epsilon"], high=config["Attack"]["epsilon"], size=zf.shape), dtype=torch.float32) + 1j*torch.tensor(np.random.uniform(low=-config["Attack"]["epsilon"], high=config["Attack"]["epsilon"], size=zf.shape), dtype=torch.float32)).to(device) 
        zf_eta.retain_grad()
        
        rec = network(zf, coil, Omega_Mask)
        
        for _ in range(config["Attack"]["iterations"]):
            rec_eta = network(zf_eta, coil, Omega_Mask)
            loss = L2_Loss(rec_eta, rec) # Unsupervised attack
            network.zero_grad()
            loss.backward(retain_graph=True)

            grad   = zf_eta.grad.data
            zf_eta = zf_eta + alpha * grad.sgn() 
            zf_eta = torch.min( torch.real(zf+config["Attack"]["epsilon"]), torch.max( torch.real(zf-config["Attack"]["epsilon"]) , torch.real(zf_eta))) + 1j*torch.min( torch.imag(zf+config["Attack"]["epsilon"]), torch.max( torch.imag(zf-config["Attack"]["epsilon"]) , torch.imag(zf_eta)))
            zf_eta = zf_eta.detach().requires_grad_(True)            
        zf_p   = zf_eta.clone().detach()

    return zf_p

def Imshow_Custom(image, path, ssimm=None, psnrr=None, window_scale=0.6, name=None, show_panel_labels=False, panel_labels=("Proposed Mitigation", "Error Map"), metric_color="tab:blue"):
    img = np.abs(np.flipud(image))
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=window_scale * img.max())

    if show_panel_labels:
        t1 = ax.text(0.02, 0.97, panel_labels[0], transform=ax.transAxes, color="white", fontsize=10, fontweight="bold", ha="left", va="top", clip_on=True); t2 = ax.text(0.52, 0.97, panel_labels[1], transform=ax.transAxes, color="white", fontsize=10, fontweight="bold", ha="left", va="top", clip_on=True)
        for t in [t1, t2]: t.set_path_effects([path_effects.Stroke(linewidth=3, foreground="black"), path_effects.Normal()])
    if psnrr is not None and ssimm is not None:
        text = ax.text(0.98, 0.03, f"PSNR: {psnrr:.2f}\nSSIM: {ssimm:.3f}", transform=ax.transAxes, color=metric_color, fontsize=10, fontweight="bold", ha="right", va="bottom", clip_on=True); text.set_path_effects([path_effects.Stroke(linewidth=3, foreground="black"), path_effects.Normal()])
    
    ax.axis("off")
    plt.savefig(os.path.join(path, f"{name}.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def Cyclic_Mitigation(config, network, zf_p, label, coil, Omega_Mask, mask_list, device, jitt):
        
        label         = label.squeeze()
        initial_alpha = config["Mitigation"]["alpha_mitigate"]
        alpha         = initial_alpha
        cg_EEH        = CG_EEH()
        zf_temp       = zf_p.clone().detach().requires_grad_(True)
        
        cg_EEH.to(device)
        zf_temp.retain_grad()
        
        best_loss = torch.inf
        best_psnr = 0.0
        best_ssim = 0.0
        LOSS      = []
        PSNR      = []
        SSIM      = []

        pbar = tqdm(range(config["Mitigation"]["iterations_mitigate"]), desc="Mitigation", dynamic_ncols=True)
        
        for iii in pbar:
            loss=0.0
            
            for maskk  in mask_list:    
                Recons_mit, err_ksp = ABA_detect(zf_temp, coil, [Omega_Mask, maskk], jitt = jitt, config = config, model = network)
                zf_temp_ksp         = cg_EEH(fft_loss(zf_temp, coil, config["axis"]) * Omega_Mask, coil, Omega_Mask)                
                loss_1              = LogL2_Loss(zf_temp_ksp, err_ksp[...,-1].to(device))
                loss               += loss_1/len(mask_list) 

            loss.backward(retain_graph=True)
            LOSS.append(loss.item())    
            
            zf_temp      = Reverse_PGD_Projection(zf_temp, zf_p, zf_temp.grad.data, config, alpha)
            zf_temp.grad = None
            
            if config["Mitigation"]["alpha_scheduler"] == "Exp":
                if iii % 20 == 0:
                    alpha = exp_alpha_scheduler(initial_alpha, iii)
            elif config["Mitigation"]["alpha_scheduler"] == "Cosine":
                alpha = cosine_alpha_scheduler(initial_alpha, iii, config["Mitigation"]["iterations_mitigate"])
            elif config["Mitigation"]["alpha_scheduler"] == "Linear":
                alpha = linear_alpha_scheduler(initial_alpha, iii, config["Mitigation"]["iterations_mitigate"])
            
            with torch.no_grad():
                Recons_mit = Recons_mit.to(device)
                recon_i    = Recons_mit[:,:,0]
                recon_i [torch.abs(label)<config["Saving"]["Threshold"]] = 0
                label   [torch.abs(label)<config["Saving"]["Threshold"]] = 0

                psnr_val = getpsnr(recon_i.cpu().detach().numpy(), label.cpu().detach().numpy())
                ssim_val = getssim(recon_i.cpu().detach().numpy(), label.cpu().detach().numpy())
                PSNR.append(psnr_val)
                SSIM.append(ssim_val)

                pbar.set_postfix({
                    "loss": f"{best_loss:.6f}",
                    "PSNR": f"{best_psnr:.3f}",
                    "SSIM": f"{best_ssim:.3f}",
                    "alpha": f"{alpha:.2e}"}
                )

            
            if loss.item()<best_loss:
                best_psnr  = psnr_val
                best_ssim  = ssim_val                  
                best_loss  = loss.item()
                best_recon = recon_i
                Imshow_Custom(best_recon.cpu().permute(1,0).detach()                  , config["Saving"]["path"], ssimm=best_ssim, psnrr=best_psnr, name="Mitigated"      , window_scale = config["Saving"]["Window_scale"])
                Imshow_Custom((label-best_recon).squeeze().permute(1,0).cpu().detach(), config["Saving"]["path"], ssimm=best_ssim, psnrr=best_psnr, name="Mitigated_Error", window_scale = config["Saving"]["Window_scale"])

            
            error_i = (recon_i-label).permute(1,0).to(device)
            recon_i = recon_i.permute(1,0).to(device)
            cmbined = torch.concat([recon_i, error_i], axis = 1)
            Imshow_Custom (cmbined.cpu().detach()           , f"{config['Saving']['path']}/iterations", ssimm=ssim_val, psnrr=psnr_val, name=f"iter_{iii}", window_scale = config["Saving"]["Window_scale"], show_panel_labels=True)
            Imshow_Custom (label.permute(1,0).cpu().detach(), config["Saving"]["path"]                , name="Reference", window_scale = config["Saving"]["Window_scale"])               

        return PSNR, SSIM, LOSS
