import pickle
import torch
import types
import sys
sys.path.append("..")
sys.path.append("./")

import dnnlib
from omegaconf import OmegaConf
from .ldm_utils import instantiate_from_config

# load model and transform it's output to eps
def load_model(dist, model_name):
    if model_name == "CIFAR10-cond" or model_name == "CIFAR10-uncond":
        model = load_edm(dist, model_name)
        def new_forward(self, x, t, cond, alpha):
            x_model_input = x/(alpha)**(1/2)
            x0_pred = model(x_model_input, t, cond)
            eps = (x - (alpha)**(1/2) * x0_pred)/(1-alpha)**(1/2)
            return eps.reshape(x.shape[0], -1)
        model.apply_model = types.MethodType(new_forward, model)
        return model
    
    elif model_name == "ImageNet64-S" or model_name == "ImageNet64-L" or model_name == "ImageNet512-XS" or model_name == "ImageNet512-XXL":
        model, encoder = load_edm2(dist, model_name)
        def new_forward(self, x, t, cond, alpha):
            x_model_input = x/(alpha)**(1/2)
            x0_pred = model(x_model_input, t, cond)
            eps = (x - (alpha)**(1/2) * x0_pred)/(1-alpha)**(1/2)
            return eps.reshape(x.shape[0], -1)
        model.apply_model = types.MethodType(new_forward, model)
        return model, encoder
    
    elif model_name == "FFHQ" or model_name == "LSUN":
        model = load_ldm(dist, model_name)
        old_forward = model.apply_model
        def new_forward(self, x, t, cond, alpha):
            eps = old_forward(x, t, cond)
            return eps.reshape(eps.shape[0],-1)
        model.apply_model = types.MethodType(new_forward, model)
        return model
    
    else:
        raise ValueError(f"No existing algorithm {model_name}!")
    
    
def load_edm(dist, model_name):
    device = "cuda"
    if model_name == "CIFAR10-cond":
        ckpt = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
    else:
        ckpt = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
    dist.print0(f'Loading network from "{ckpt}"...')
    with dnnlib.util.open_url(ckpt, verbose=(dist.get_rank() == 0)) as f:
        model = pickle.load(f)['ema'].to(device)
    return model


def load_edm2(dist, model_name):
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
    config_presets = {
        'ImageNet512-XS':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),
        'ImageNet512-XXL':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),
        'ImageNet64-S':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),
        'ImageNet64-L':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),
    }
    device = "cuda"
    opts = dnnlib.EasyDict()
    for key, value in config_presets[model_name].items():
        opts[key] = value
    opts.guidance = 1
    opts.gnet = None
    
    net = opts.net
    dist.print0(f'Loading main network from {net} ...')
    with dnnlib.util.open_url(net, verbose=dist.get_rank() == 0) as f:
        data = pickle.load(f)
    net = data['ema'].to(device)
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None
    assert encoder is not None
    model = net
    return model, encoder
    

def load_ldm(dist, model_name):
    if model_name == "FFHQ":
        config_path = "config_ldm/ffhq.yaml"
        ckpt = "ckpt/model_ffhq.ckpt"
    else:
        config_path = "config_ldm/lsun.yaml"
        ckpt = "ckpt/model_lsun.ckpt"
        
    dist.print0(f'Loading main network from {ckpt} ...')
    pl_sd = torch.load(ckpt, map_location="cpu")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model

