import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# # Simple Moving Average

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x

# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean

# Exponential Moving Average

class ema(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, alpha):
        super(ema, self).__init__()
        # self.alpha = nn.Parameter(alpha)
        self.alpha = alpha

    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)

    # def forward(self, x):
    #     self.alpha.data.clamp_(0, 1)
    #     s = x[:, 0, :]
    #     res = [s.unsqueeze(1)]
    #     for t in range(1, x.shape[1]):
    #         xt = x[:, t, :]
    #         s = self.alpha * xt + (1 - self.alpha) * s
    #         res.append(s.unsqueeze(1))
    #     return torch.cat(res, dim=1)

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, alpha):
        super(series_decomp, self).__init__()
        self.ema = ema(alpha)

    def forward(self, x):
        moving_average = self.ema(x)
        res = x - moving_average
        return res, moving_average

# Double Exponential Moving Average

# class dema(nn.Module):
#     """
#     Double Exponential Moving Average (DEMA) block to highlight the trend of time series
#     """
#     def __init__(self, alpha, beta):
#         super(dema, self).__init__()
#         self.alpha = nn.Parameter(alpha)
#         self.beta = nn.Parameter(beta)

#     def forward(self, x):
#         self.alpha.data.clamp_(0, 1)
#         self.beta.data.clamp_(0, 1)
#         s_prev = x[:, 0, :]
#         b = x[:, 1, :] - s_prev
#         res = [s_prev.unsqueeze(1)]
#         for t in range(1, x.shape[1]):
#             xt = x[:, t, :]
#             s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
#             b = self.beta * (s - s_prev) + (1 - self.beta) * b
#             s_prev = s
#             res.append(s.unsqueeze(1))
#         return torch.cat(res, dim=1)

# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, alpha, beta):
#         super(series_decomp, self).__init__()
#         self.dema = dema(alpha, beta)

#     def forward(self, x):
#         moving_average = self.dema(x)
#         res = x - moving_average
#         return res, moving_average

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
