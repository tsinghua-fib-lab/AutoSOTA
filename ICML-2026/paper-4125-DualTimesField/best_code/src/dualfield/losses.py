import torch
import torch.nn as nn
import torch.nn.functional as F


class DualFieldLoss(nn.Module):
    def __init__(
        self,
        sparsity_lambda=0.01,
        smoothness_lambda=0.001,
        freq_lambda=0.0,
        separation_lambda=0.01
    ):
        super().__init__()
        self.sparsity_lambda = sparsity_lambda
        self.smoothness_lambda = smoothness_lambda
        self.freq_lambda = freq_lambda
        self.separation_lambda = separation_lambda
        
    def forward(self, model, x, t, z=None):
        output, ctf_output, dgf_output = model(t, z)
        
        rec_loss = F.mse_loss(output, x)
        
        sparsity_loss = model.dgf.compute_sparsity_loss()
        
        smoothness_loss = self.compute_smoothness(model.ctf, t, z)
        
        separation_loss = self.compute_separation_loss(ctf_output, dgf_output, x)
        
        total_loss = (
            rec_loss + 
            self.sparsity_lambda * sparsity_loss + 
            self.smoothness_lambda * smoothness_loss +
            self.separation_lambda * separation_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': rec_loss,
            'sparsity': sparsity_loss,
            'smoothness': smoothness_loss,
            'separation': separation_loss
        }
    
    def compute_smoothness(self, ctf, t, z):
        t_grad = t.clone().requires_grad_(True)
        output = ctf(t_grad, z)
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=t_grad,
            create_graph=True,
            retain_graph=True
        )[0]
        return (grad ** 2).mean()
    
    def compute_separation_loss(self, ctf_output, dgf_output, target):
        ctf_residual = target - ctf_output
        correlation = (ctf_output * dgf_output).mean()
        dgf_should_fit = F.mse_loss(dgf_output, ctf_residual)
        return correlation.abs() + 0.1 * dgf_should_fit


class GroupLassoLoss(nn.Module):
    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
        
    def forward(self, amplitude):
        group_norms = torch.norm(amplitude, p=2, dim=1)
        return self.lambda_reg * group_norms.sum()


class FrequencyDomainLoss(nn.Module):
    def __init__(self, lambda_fft=0.1):
        super().__init__()
        self.lambda_fft = lambda_fft
        
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        fft_loss = F.l1_loss(pred_fft.abs(), target_fft.abs())
        return self.lambda_fft * fft_loss


class MultiScaleLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4, 8], weights=None):
        super().__init__()
        self.scales = scales
        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)
        self.weights = weights
        
    def forward(self, pred, target):
        total_loss = 0.0
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                scaled_pred = pred
                scaled_target = target
            else:
                scaled_pred = F.avg_pool1d(
                    pred.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
                scaled_target = F.avg_pool1d(
                    target.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
            total_loss += weight * F.mse_loss(scaled_pred, scaled_target)
        return total_loss


class CompositeLoss(nn.Module):
    def __init__(
        self,
        mse_weight=1.0,
        fft_weight=0.1,
        multiscale_weight=0.1,
        sparsity_weight=0.01,
        smoothness_weight=0.001
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.fft_weight = fft_weight
        self.multiscale_weight = multiscale_weight
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
        
        self.fft_loss = FrequencyDomainLoss(lambda_fft=1.0)
        self.multiscale_loss = MultiScaleLoss()
        self.group_lasso = GroupLassoLoss(lambda_reg=1.0)
        
    def forward(self, model, x, t, z=None):
        output, ctf_output, dgf_output = model(t, z)
        
        mse = F.mse_loss(output, x)
        fft = self.fft_loss(output, x)
        multiscale = self.multiscale_loss(output, x)
        sparsity = self.group_lasso(model.dgf.atoms.amplitude)
        
        t_grad = t.clone().requires_grad_(True)
        ctf_out = model.ctf(t_grad, z)
        grad = torch.autograd.grad(
            outputs=ctf_out.sum(),
            inputs=t_grad,
            create_graph=True
        )[0]
        smoothness = (grad ** 2).mean()
        
        total = (
            self.mse_weight * mse +
            self.fft_weight * fft +
            self.multiscale_weight * multiscale +
            self.sparsity_weight * sparsity +
            self.smoothness_weight * smoothness
        )
        
        return {
            'total': total,
            'mse': mse,
            'fft': fft,
            'multiscale': multiscale,
            'sparsity': sparsity,
            'smoothness': smoothness
        }
