import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product     

class InferQuantMatMul(nn.Module):
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.A_bit=A_bit
        self.B_bit=B_bit
        self.A_qmax=2**(self.A_bit-1)
        self.B_qmax=2**(self.B_bit-1)
        self.mode=mode
        # self.split=split
        
    
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["A_bit", "B_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        
        for sti in ["n_G_A", "n_V_A", "n_H_A", "n_G_B", "n_V_B", "n_H_B"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s 
    
    def get_parameter(self, A_interval, B_interval, n_G_A, n_V_A, n_H_A, n_G_B, n_V_B, n_H_B, crb_groups_A, crb_groups_B, crb_rows_A, crb_rows_B, crb_cols_A, crb_cols_B, pad_groups_A, pad_groups_B, pad_rows_A, pad_rows_B, pad_cols_A, pad_cols_B):
        self.register_buffer('A_interval', A_interval)
        self.register_buffer('B_interval', B_interval)
        
        self.register_buffer('n_G_A', torch.tensor(n_G_A, dtype=torch.int32))
        self.register_buffer('n_V_A', torch.tensor(n_V_A, dtype=torch.int32))
        self.register_buffer('n_H_A', torch.tensor(n_H_A, dtype=torch.int32))
        self.register_buffer('n_G_B', torch.tensor(n_G_B, dtype=torch.int32))
        self.register_buffer('n_V_B', torch.tensor(n_V_B, dtype=torch.int32))
        self.register_buffer('n_H_B', torch.tensor(n_H_B, dtype=torch.int32))
        self.register_buffer('crb_groups_A', torch.tensor(crb_groups_A, dtype=torch.int32))
        self.register_buffer('crb_groups_B', torch.tensor(crb_groups_B, dtype=torch.int32))
        self.register_buffer('crb_rows_A', torch.tensor(crb_rows_A, dtype=torch.int32))
        self.register_buffer('crb_rows_B', torch.tensor(crb_rows_B, dtype=torch.int32))
        self.register_buffer('crb_cols_A', torch.tensor(crb_cols_A, dtype=torch.int32))
        self.register_buffer('crb_cols_B', torch.tensor(crb_cols_B, dtype=torch.int32))
        self.register_buffer('pad_groups_A', torch.tensor(pad_groups_A, dtype=torch.int32))
        self.register_buffer('pad_groups_B', torch.tensor(pad_groups_B, dtype=torch.int32))
        self.register_buffer('pad_rows_A', torch.tensor(pad_rows_A, dtype=torch.int32))
        self.register_buffer('pad_rows_B', torch.tensor(pad_rows_B, dtype=torch.int32))
        self.register_buffer('pad_cols_A', torch.tensor(pad_cols_A, dtype=torch.int32))
        self.register_buffer('pad_cols_B', torch.tensor(pad_cols_B, dtype=torch.int32))
        # self.register_buffer('split', torch.tensor(split, dtype=torch.float32))
    
    def forward(self, A, B):
        if self.mode=='raw':
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input_A(self, x):
        x = F.pad(x, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A])
        x = x.view(-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
        x = (x/self.A_interval).round_().clamp(-self.A_qmax,self.A_qmax-1).mul_(self.A_interval)
        x = x.view(-1,self.n_G_A*self.crb_groups_A,self.n_V_A*self.crb_rows_A,self.n_H_A*self.crb_cols_A)
        x = x[:,:x.shape[1]-self.pad_groups_A,:x.shape[2]-self.pad_rows_A,:x.shape[3]-self.pad_cols_A]
        return x
    
    def quant_input_B(self, x):
        x = F.pad(x, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B])
        x = x.view(-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        x = (x/self.B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(self.B_interval)
        x = x.view(-1,self.n_G_B*self.crb_groups_B,self.n_V_B*self.crb_rows_B,self.n_H_B*self.crb_cols_B)
        x = x[:,:x.shape[1]-self.pad_groups_B,:x.shape[2]-self.pad_rows_B,:x.shape[3]-self.pad_cols_B]
        return x
    
    def quant_forward(self, A, B):
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        out=A_sim@B_sim
        return out

    
class InferQuantMatMulPost(nn.Module):
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.A_bit=A_bit
        self.B_bit=B_bit
        self.A_qmax=2**(self.A_bit-1)
        self.B_qmax=2**(self.B_bit-1)
        self.mode=mode
        
    
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["A_bit", "B_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        
        for sti in ["n_G_A", "n_V_A", "n_H_A", "n_G_B", "n_V_B", "n_H_B", "split"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s 

    def forward(self, A, B):
        if self.mode=='raw':
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B)
        else:
            raise NotImplementedError
        return out
    
    def get_parameter(self, A_interval, B_interval, n_G_A, n_V_A, n_H_A, n_G_B, n_V_B, n_H_B, crb_groups_A, crb_groups_B, crb_rows_A, crb_rows_B, crb_cols_A, crb_cols_B, pad_groups_A, pad_groups_B, pad_rows_A, pad_rows_B, pad_cols_A, pad_cols_B, split):
        self.register_buffer('A_interval', A_interval)
        self.register_buffer('B_interval', B_interval)
        self.register_buffer('n_G_A', torch.tensor(n_G_A, dtype=torch.int32))
        self.register_buffer('n_V_A', torch.tensor(n_V_A, dtype=torch.int32))
        self.register_buffer('n_H_A', torch.tensor(n_H_A, dtype=torch.int32))
        self.register_buffer('n_G_B', torch.tensor(n_G_B, dtype=torch.int32))
        self.register_buffer('n_V_B', torch.tensor(n_V_B, dtype=torch.int32))
        self.register_buffer('n_H_B', torch.tensor(n_H_B, dtype=torch.int32))
        self.register_buffer('crb_groups_A', torch.tensor(crb_groups_A, dtype=torch.int32))
        self.register_buffer('crb_groups_B', torch.tensor(crb_groups_B, dtype=torch.int32))
        self.register_buffer('crb_rows_A', torch.tensor(crb_rows_A, dtype=torch.int32))
        self.register_buffer('crb_rows_B', torch.tensor(crb_rows_B, dtype=torch.int32))
        self.register_buffer('crb_cols_A', torch.tensor(crb_cols_A, dtype=torch.int32))
        self.register_buffer('crb_cols_B', torch.tensor(crb_cols_B, dtype=torch.int32))
        self.register_buffer('pad_groups_A', torch.tensor(pad_groups_A, dtype=torch.int32))
        self.register_buffer('pad_groups_B', torch.tensor(pad_groups_B, dtype=torch.int32))
        self.register_buffer('pad_rows_A', torch.tensor(pad_rows_A, dtype=torch.int32))
        self.register_buffer('pad_rows_B', torch.tensor(pad_rows_B, dtype=torch.int32))
        self.register_buffer('pad_cols_A', torch.tensor(pad_cols_A, dtype=torch.int32))
        self.register_buffer('pad_cols_B', torch.tensor(pad_cols_B, dtype=torch.int32))
        self.register_buffer('split', torch.tensor(split, dtype=torch.float32))
    
    def quant_input_A(self, x):
        x_high = (x.clamp(self.split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split)/self.A_interval).round_().clamp_(0,self.A_qmax-1)*self.A_interval
        return x_high + x_low
    
    def quant_input_B(self, x):
        x = F.pad(x, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B])
        x = x.view(-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        x = (x/self.B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(self.B_interval)
        x = x.view(-1,self.n_G_B*self.crb_groups_B,self.n_V_B*self.crb_rows_B,self.n_H_B*self.crb_cols_B)
        x = x[:,:x.shape[1]-self.pad_groups_B,:x.shape[2]-self.pad_rows_B,:x.shape[3]-self.pad_cols_B]
        return x
    
    def quant_forward(self, A, B):
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        out=A_sim@B_sim
        return out

    
class InferQuantLinear(nn.Linear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_correction = False):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        self.bias_correction = bias_correction

    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V", "n_a"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_parameter(self, n_V, n_H, n_a, a_interval, w_interval, crb_rows, crb_cols, crb_acts):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))
        self.register_buffer('n_a', torch.tensor(n_a, dtype=torch.int32))
        self.register_buffer('crb_rows', torch.tensor(crb_rows, dtype=torch.int32))
        self.register_buffer('crb_cols', torch.tensor(crb_cols, dtype=torch.int32))
        self.register_buffer('crb_acts', torch.tensor(crb_acts, dtype=torch.int32))
    
    def quant_weight_bias(self):
        w = (self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim = w.mul_(self.w_interval).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim, self.bias
        else:
            return w_sim, None
    
    def quant_input(self, x):
        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_sim=(x_sim.div_(self.a_interval)).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim = x_sim.mul_(self.a_interval).reshape_as(x)
        return x_sim

    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        return out

    
class InferQuantLinearPost(nn.Linear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_correction = False):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        self.bias_correction = bias_correction
        tmp_a_neg_interval = torch.tensor(0.16997124254703522/self.a_qmax)
        self.register_buffer('a_neg_interval', tmp_a_neg_interval)

    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V", "n_a"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_parameter(self, n_V, n_H, n_a, a_interval, w_interval, crb_rows, crb_cols, crb_acts):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))
        self.register_buffer('n_a', torch.tensor(n_a, dtype=torch.int32))
        self.register_buffer('crb_rows', torch.tensor(crb_rows, dtype=torch.int32))
        self.register_buffer('crb_cols', torch.tensor(crb_cols, dtype=torch.int32))
        self.register_buffer('crb_acts', torch.tensor(crb_acts, dtype=torch.int32))
    
    def quant_weight_bias(self):
        w = (self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim = w.mul_(self.w_interval).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim, self.bias
        else:
            return w_sim, None
    
    def quant_input(self, x):
        x_=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        x_pos=(x_/(self.a_interval)).round_().clamp_(0,self.a_qmax-1).mul_(self.a_interval)
        x_neg=(x_/(self.a_neg_interval)).round_().clamp_(-self.a_qmax,0).mul_(self.a_neg_interval)
        return (x_pos + x_neg).reshape_as(x)

    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        return out

    
class InferQuantConv2d(nn.Conv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.mode=mode
        self.w_bit=w_bit
        self.a_bit=a_bit
        # self.bias_bit=bias_bit
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        
    def get_parameter(self, n_V, n_H, a_interval, a_bias, w_interval):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('a_bias', a_bias)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))
        
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
        
    def forward(self, x):
        if self.mode=='raw':
            out=F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out
            
    def quant_weight_bias(self):
        w_sim = (self.weight/self.w_interval).round_().clamp(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval)
        return w_sim, self.bias
    
    def quant_input(self, x):        
        aq = (x - self.a_bias)/ self.a_interval
        aq = aq.round_().clamp_(-self.a_qmax, self.a_qmax-1)
        x_sim = aq * self.a_interval  + self.a_bias
        return x_sim
    
    def quant_forward(self, x):
        w_sim, bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x) if self.a_bit < 32 else x
        out=F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    
class InferQuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation = 1,
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.mode=mode
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        
    def __repr__(self):
        s = '{}('.format(self.__class__.__name__)
        for sti in ["w_bit", "a_bit", "mode"]: 
            s += "{}={}, ".format(sti, self.__dict__[sti])
        for sti in ["n_H", "n_V"]:
            s += "{}={}, ".format(sti, self.__dict__["_buffers"][sti])
        s = s[:-2]
        s +=  ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
                
    def quant_weight_bias(self):
        w_sim = (self.weight/self.w_interval).round_().clamp(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval)
        return w_sim, self.bias
    
    def quant_input(self, x):
        aq = (x - self.a_bias)/ self.a_interval
        aq = aq.round_().clamp_(-self.a_qmax, self.a_qmax-1)
        x_sim = aq * self.a_interval  + self.a_bias
        return x_sim

    def get_parameter(self, n_V, n_H, a_interval, a_bias, w_interval):
        self.register_buffer('a_interval', a_interval)
        self.register_buffer('a_bias', a_bias)
        self.register_buffer('w_interval', w_interval)
        self.register_buffer('n_V', torch.tensor(n_V, dtype=torch.int32))
        self.register_buffer('n_H', torch.tensor(n_H, dtype=torch.int32))

    def quant_forward(self, x):
        w_sim, bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x) if self.a_bit < 32 else x
        out=F.conv_transpose2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return out

    def forward(self, x):
        if self.mode=='raw':
            out=F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        else:
            raise NotImplementedError
        return out