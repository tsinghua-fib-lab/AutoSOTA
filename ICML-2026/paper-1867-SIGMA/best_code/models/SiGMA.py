import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from scipy import special as sc_special


class ForecastingModule(nn.Module):
    """Forecasting module for handling temporal and feature transformations."""
    
    def __init__(self, configs):
        """Initialize forecasting module with configuration parameters."""
        super().__init__()
        self.seq_len = configs.seq_len
        self.activation = nn.GELU
        self.scale_independence = configs.scale_independence  
        self.feature_transformation = configs.feature_transformation
        
        if self.scale_independence == 1:
            self.update_mlp = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len),
                self.activation(),
                nn.Linear(self.seq_len, self.seq_len),
            )
        else:
            self.update_mlp = nn.Sequential(
                nn.Linear(self.seq_len * 2, self.seq_len * 2) ,
                self.activation(),
                nn.Linear(self.seq_len * 2, self.seq_len * 2),
            )
        
        if self.feature_transformation == 1:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                self.activation(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize network parameters using Xavier uniform distribution."""
        for layer in self.update_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            
        if self.feature_transformation:
            for layer in self.out_cross_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _apply_time_mlp(self, mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP along temporal dimension by permuting and processing input tensor."""
        x_bct = x.permute(0, 2, 1)
        x_bct = mlp(x_bct)
        return x_bct.permute(0, 2, 1)

    def forward(self, x_list):
        """Process multi-scale representations with optional independence and feature transformation."""
        x0, x1 = x_list
        if self.scale_independence == 1:
            h0 = self._apply_time_mlp(self.update_mlp, x0)
            h1 = self._apply_time_mlp(self.update_mlp, x1)
            if self.feature_transformation == 1:
                h0 = self.out_cross_layer(h0)
                h1 = self.out_cross_layer(h1)
            out0 = h0 + x0
            out1 = h1 + x1
        else:
            h = torch.cat([x0, x1], dim=1)
            h_upd = self._apply_time_mlp(self.update_mlp, h)
            if self.feature_transformation == 1:
                h_upd = self.out_cross_layer(h_upd)
            out0 = h_upd[:, :self.seq_len, :] + x0
            out1 = h_upd[:, self.seq_len:, :] + x1
        
        return [out0, out1]


class MultiScaleGenerator(nn.Module):
    """Generator for multi-scale decomposition using discrete Gaussian kernels."""

    def __init__(self, configs):
        """Initialize multi-scale generator with learnable kernel parameters."""
        super().__init__()
        self.seq_len = configs.seq_len
        self.configs = configs
        self.enc_in = configs.enc_in
        self.t_param = nn.Parameter(torch.empty(self.seq_len, dtype=torch.float32), requires_grad=True)
        nn.init.constant_(self.t_param, 0.0)
        self.eps = 1e-6

    def _discrete_gaussian_kernel(self, index_diff: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        """Compute discrete Gaussian kernel using modified Bessel functions with custom gradient."""
        d = index_diff.to(dtype=torch.long, device=t_vec.device) 


        class _DiscreteGaussianKernelFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, index_diff_long: torch.Tensor, t_vec_in: torch.Tensor):
                device = t_vec_in.device
                T = t_vec_in.shape[0]

                D_np = index_diff_long.detach().cpu().numpy()
                t_np = t_vec_in.detach().cpu().numpy()

                orders = np.arange(T)
                I_n = sc_special.iv(orders, t_np)  # [T]
                K_d = np.exp(-t_np) * I_n  # [T]
                K_np = K_d[D_np]  # [T, T]

                ctx.save_for_backward(index_diff_long, t_vec_in)
                return torch.from_numpy(K_np).to(device=device, dtype=t_vec_in.dtype)

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor):
                index_diff_long, t_vec_in = ctx.saved_tensors
                device = t_vec_in.device
                T = t_vec_in.shape[0]

                t_np = t_vec_in.detach().cpu().numpy()
                orders = np.arange(T)
                I_n = sc_special.iv(orders, t_np)
                I_nm1 = sc_special.iv(orders - 1, t_np)
                I_np1 = sc_special.iv(orders + 1, t_np)
                # I_{-1} = I_{1}
                if T > 0:
                    I_nm1[0] = I_np1[0]
                dId = (I_nm1 + I_np1) / 2.0
                dKd_dt = np.exp(-t_np) * (dId - I_n)  # [T]

                go_np = grad_output.detach().cpu().numpy().ravel()
                D_np = index_diff_long.detach().cpu().numpy().ravel()
                sums_per_d = np.bincount(D_np, weights=go_np, minlength=T)
                grad_t_np = sums_per_d * dKd_dt

                grad_t = torch.from_numpy(grad_t_np).to(device=device, dtype=t_vec_in.dtype)
                return None, grad_t

        index_diff_long = d 
        return _DiscreteGaussianKernelFn.apply(index_diff_long, t_vec)

    def forward(self, x_enc):
        """Generate multi-scale decomposition using learnable Gaussian kernel convolution."""
        if isinstance(x_enc, list):
            x_enc = x_enc[0]
        x_ct = x_enc.permute(0, 2, 1)  
        B, C, T = x_ct.size()

        i_indices = torch.arange(T, device=x_ct.device, dtype=torch.long).view(T, 1)
        j_indices = torch.arange(T, device=x_ct.device, dtype=torch.long).view(1, T)
        index_diff = torch.abs(i_indices - j_indices)

        t_vec = F.softplus(self.t_param)
        K = self._discrete_gaussian_kernel(index_diff, t_vec)

        x_kernel = torch.matmul(x_ct, K)
        x_residual = x_ct - x_kernel

        x_list = [x_kernel.permute(0, 2, 1), x_residual.permute(0, 2, 1)]
        return x_list



class Model(nn.Module):
    """SiGMA: Multi-scale time series forecasting model with Gaussian kernel decomposition."""
    
    def __init__(self, configs):
        """Initialize SiGMA model with multi-scale generators and processors."""
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence
        
        self.forecasting_modules = nn.ModuleList([
            ForecastingModule(configs) for _ in range(configs.e_layers)
        ])

        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layer = Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)

        self.predict_layers = torch.nn.ModuleList(
            [
            torch.nn.Linear(
                configs.seq_len,
                configs.pred_len,
            ),
            torch.nn.Linear(
                configs.seq_len,
                configs.pred_len,
            )
            ]
        )
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

        self.multi_scale_generators = nn.ModuleList(
            [
                MultiScaleGenerator(
                    configs=configs
                ) for _ in range(configs.e_layers)
            ]
        )
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize prediction and projection layer parameters."""
        for layer in self.predict_layers:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 1.0 / self.pred_len)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.02)
        if self.projection_layer.bias is not None:
            nn.init.zeros_(self.projection_layer.bias)


    def forecast(self, x, x_mark, x_dec, x_mark_dec, eval=False):
        """Perform multi-horizon time series forecasting using multi-scale decomposition."""
        B, T, N = x.size()
        x = self.normalize_layer(x, 'norm')
        
        # do not use time features for ETT dataset
        if 'ETT' in self.configs.data:
            x_mark = None
    
        if self.channel_independence == 1:
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            if x_mark is not None:
                Fm = x_mark.size(-1)
                x_mark = x_mark.unsqueeze(1).expand(B, N, T, Fm).reshape(B * N, T, Fm)

        enc_out = self.enc_embedding(x, x_mark) if x_mark is not None else self.enc_embedding(x, None)
        enc_out_list = [enc_out]
        
        for j in range(self.layer):
            enc_out_list = self.multi_scale_generators[j](enc_out_list)
            enc_out_list = self.forecasting_modules[j](enc_out_list)
        
        dec_out_list = []
        for i, enc_out in zip(range(len(enc_out_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1) 
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)
        
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layer(dec_out, 'denorm')
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, eval=False):
        """Main forward pass routing to appropriate task-specific methods."""
        if 'long_term_forecast' in self.task_name or 'short_term_forecast' in self.task_name:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, eval)
            return dec_out
        
        else:
            raise ValueError('Other tasks implemented yet')
