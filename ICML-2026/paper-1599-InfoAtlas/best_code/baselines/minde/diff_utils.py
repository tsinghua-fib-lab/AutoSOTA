import torch
import torch.nn as nn
from copy import deepcopy



class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
def pop_elem_i(encodings , i =[]  ):
    encodings = encodings.copy()
    return{
        key : encodings[key] for key in encodings.keys() if ( key in i ) == False
    } 
    
def deconcat(z,var_list,sizes):
    data = torch.split(z, sizes, dim=1)
    return {var: data[i] for i, var in enumerate(var_list)}



def concat_vect(encodings):
    return torch.cat(list(encodings.values()),dim = -1)
    


def unsequeeze_dict(data):
        for key in data.keys():
            if data[key].ndim == 1 :
                data[key]= data[key].view(data[key].size(0),1)
        return data


def cond_x_data(x_t,data,mod):

    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            x[k]=data[k] 
    return x



def marginalize_data(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k !=mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x


def marginalize_one_var(x_t, mod,fill_zeros =False):
    x = x_t.copy()
    for k in x.keys():
        if k ==mod:
            if fill_zeros:
                x[k]=torch.zeros_like(x_t[k] ) 
            else:
                x[k]=torch.randn_like(x_t[k] )
    return x


def minus_x_data(x_t, mod,fill_zeros=True):
        x = x_t.copy()
        for k in x.keys():
                if k ==mod:
                    if fill_zeros:
                        x[k]=torch.zeros_like(x_t[k] ) 
                    else:
                        x[k]=torch.rand_like(x_t[k] )
        return x


def expand_mask(mask, var_sizes):
        return torch.cat([
            mask[:, i].view(mask.shape[0], 1).expand(mask.shape[0], size) for i, size in enumerate(var_sizes)
        ], dim=1)
        
        
def get_samples(test_loader,device,N=10000):
    
    var_list = list(test_loader.dataset[0].keys())
    
    data = {var: torch.Tensor().to(device) for var in var_list}
    for batch in test_loader:
            for var in var_list:
                data[var] = torch.cat([data[var], batch[var].to(device)])
    return {var: data[var][:N,:] for var in var_list}

