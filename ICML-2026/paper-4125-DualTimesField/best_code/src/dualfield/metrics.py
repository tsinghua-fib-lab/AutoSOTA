import torch
import torch.nn.functional as F
import numpy as np


def mse(pred, target):
    return F.mse_loss(pred, target).item()


def rmse(pred, target):
    return np.sqrt(mse(pred, target))


def mae(pred, target):
    return F.l1_loss(pred, target).item()


def mape(pred, target, eps=1e-8):
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean().item() * 100


def smape(pred, target, eps=1e-8):
    numerator = torch.abs(pred - target)
    denominator = (torch.abs(pred) + torch.abs(target)) / 2 + eps
    return (numerator / denominator).mean().item() * 100


def r2_score(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def psnr(pred, target, max_val=1.0):
    mse_val = mse(pred, target)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse_val)


def spectral_divergence(pred, target):
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    
    pred_power = pred_fft.abs() ** 2
    target_power = target_fft.abs() ** 2
    
    pred_power = pred_power / (pred_power.sum(dim=1, keepdim=True) + 1e-8)
    target_power = target_power / (target_power.sum(dim=1, keepdim=True) + 1e-8)
    
    kl_div = (target_power * torch.log((target_power + 1e-8) / (pred_power + 1e-8))).sum(dim=1)
    return kl_div.mean().item()


def temporal_correlation(pred, target):
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1))
    
    correlation = numerator / (denominator + 1e-8)
    return correlation.mean().item()


class ReconstructionEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def evaluate(self, dataloader):
        self.model.eval()
        
        all_mse = []
        all_mae = []
        all_psnr = []
        all_corr = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, t = batch
                x = x.to(self.device)
                t = t.to(self.device)
                
                output, ctf_output, dgf_output = self.model(t)
                
                all_mse.append(mse(output, x))
                all_mae.append(mae(output, x))
                all_psnr.append(psnr(output, x))
                all_corr.append(temporal_correlation(output, x))
        
        return {
            'mse': np.mean(all_mse),
            'mae': np.mean(all_mae),
            'psnr': np.mean(all_psnr),
            'temporal_correlation': np.mean(all_corr)
        }
    
    def evaluate_decomposition(self, dataloader):
        self.model.eval()
        
        ctf_energy = []
        dgf_energy = []
        total_energy = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, t = batch
                x = x.to(self.device)
                t = t.to(self.device)
                
                output, ctf_output, dgf_output = self.model(t)
                
                ctf_energy.append((ctf_output ** 2).mean().item())
                dgf_energy.append((dgf_output ** 2).mean().item())
                total_energy.append((x ** 2).mean().item())
        
        ctf_ratio = np.mean(ctf_energy) / (np.mean(total_energy) + 1e-8)
        dgf_ratio = np.mean(dgf_energy) / (np.mean(total_energy) + 1e-8)
        
        return {
            'ctf_energy_ratio': ctf_ratio,
            'dgf_energy_ratio': dgf_ratio,
            'ctf_mean_energy': np.mean(ctf_energy),
            'dgf_mean_energy': np.mean(dgf_energy)
        }
