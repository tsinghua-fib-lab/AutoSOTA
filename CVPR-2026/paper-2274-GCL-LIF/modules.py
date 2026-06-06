import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, max_val):
        x_quant = torch.clamp(torch.round(x * scale), 0, max_val) / scale
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def quantize_4bit(x):
    return Quantizer.apply(x, 4.0, 15.0)


class OnlineNeuron(nn.Module):
    def __init__(self, T, mem_bn=False, num_features=128, parallel_mode=False, mode: Literal['vanilla', 'compression']='vanilla'):
        super(OnlineNeuron, self).__init__()
        self.v_th = nn.Parameter(torch.zeros(T), requires_grad=True)
        self.T = T
        self.mem_bn = mem_bn
        self.parallel_mode = parallel_mode
        self.mode = mode
        self.t = 0
        self.v_mem = None
        if self.parallel_mode is False:
            self.alpha = nn.Parameter(torch.zeros(T), requires_grad=True)
            if mem_bn is True:
                self.bn_ratio = nn.Parameter(torch.zeros(T), requires_grad=True)
                self.mem_ratio = nn.Parameter(torch.zeros(T), requires_grad=True)
                self.bn = nn.ModuleList([nn.BatchNorm2d(num_features) for i in range(T)])
        self.quantizer = quantize_4bit if self.mode == 'compression' else (lambda x: x)
        
    def forward(self, x):
        if self.t == 0 and self.parallel_mode is False:
            self.v_mem = torch.zeros_like(x)
        if self.parallel_mode is True:
            spike = F.relu(x - self.v_th[self.t].sigmoid())
            return self.quantizer(spike)
        else:
            self.v_mem = self.alpha[self.t].sigmoid() * self.v_mem.detach() + x
            if self.mem_bn is True:
                self.v_mem = self.bn_ratio[self.t].sigmoid() * self.bn[self.t](self.v_mem) + self.mem_ratio[self.t].sigmoid() * self.v_mem
            spike = F.relu(self.v_mem - self.v_th[self.t].sigmoid())
            self.v_mem -= spike
            self.t += 1
            return self.quantizer(spike)
        
    def reset(self):
        self.t = 0
        self.v_mem = None


class LIFNeuron(nn.Module):
    def __init__(self, T, online_training=True, is_learnable=False):
        super(LIFNeuron, self).__init__()
        self.act = SurrogateFunction.apply
        self.T = T
        self.t = 0
        self.online_training = online_training
        self.v_mem = None
        self.alpha = nn.Parameter(torch.zeros(T), requires_grad=is_learnable)
        self.v_th = nn.Parameter(torch.zeros(T), requires_grad=is_learnable)
            
    def forward(self, x):
        if self.t == 0:
            self.v_mem = torch.zeros_like(x)
        if self.online_training is True:
            self.v_mem = self.alpha[self.t].sigmoid() * self.v_mem.detach() + x
        else:
            self.v_mem = self.alpha[self.t].sigmoid() * self.v_mem + x
        spike = self.act(self.v_mem - (self.v_th[self.t].sigmoid()+0.5), 1.) * (self.v_th[self.t].sigmoid()+0.5)
        self.v_mem -= spike
        self.t += 1
        return spike
        
    def reset(self):
        self.t = 0
        self.v_mem = None
        

class SurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        ctx.gama = gama
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        gama = ctx.gama
        tmp = (1 / gama) ** 2 * (gama - input.abs()).clamp(min=0)
        grad_input = grad_output * tmp
        return grad_input, None