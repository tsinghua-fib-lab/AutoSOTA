import torch
import random
import numpy as np
import math
from easydict import EasyDict as edict

from models.transformer import GraphTransformer, GraphTransformer_Mol
from mix import DiffusionMixture
from losses import get_pred_loss_fn
from solver import get_pc_sampler
from utils.ema import ExponentialMovingAverage

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'transformer':
        model = GraphTransformer(**params_)
    elif model_type == 'transformer_mol':
        model = GraphTransformer_Mol(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    if config_train.optimizer=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, amsgrad=True,
                                        weight_decay=config_train.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optimizer} not implemented.')
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data in ['QM9', 'ZINC250k']:
        from utils.data_loader_mol import dataloader
        return dataloader(config, get_graph_list)
    else:
        from utils.data_loader import dataloader
        return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_mix(config_mix):
    mix_type = config_mix.type
    drift_coeff = config_mix.drift_coeff
    sigma_0 = config_mix.sigma_0
    sigma_1 = config_mix.sigma_1
    num_scales = config_mix.num_scales
    mix = DiffusionMixture(bridge=mix_type, drift_coeff=drift_coeff,
                            sigma_0=sigma_0, sigma_1=sigma_1, N=num_scales)
    return mix


def load_mix_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    mix_x = load_mix(config.mix.x)
    mix_adj = load_mix(config.mix.adj)
    get_loss_fn = get_pred_loss_fn
    loss_fn = get_loss_fn(mix_x, mix_adj, train=True, reduce_mean=reduce_mean, 
                            eps=config.train.eps, lambda_train=config.train.lambda_train,
                            loss_type=config.train.loss_type)
    return loss_fn


def load_sampling_fn(config_train, config_sampler, config_sample, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    get_sampler = get_pc_sampler
    mix = edict({'x': load_mix(config_train.mix.x), 'adj': load_mix(config_train.mix.adj)})
    max_node_num  = config_train.data.max_node_num
    batch_size = config_sample.batch_size
    shape = edict({'x': (batch_size, max_node_num, config_train.data.max_feat_num), 
                    'adj': (batch_size, max_node_num, max_node_num)})
    sampling_fn = get_sampler(mix=mix, shape=shape, sampler=config_sampler,
                                denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id,
                                ref=config_sample.ref, gamma=config_sample.gamma, tc=config_sample.tc)
    return sampling_fn


def load_model_params(config):
    input_dims = {'X': config.data.max_feat_num, 'E': config.model.input_dims.E, 
                    'y': config.model.input_dims.y+1} # +1 for time feature
    output_dims = {'X': config.data.max_feat_num, 'E': 1, 'y': 0}
    params = {'model_type': config.model.type, 'n_layers': config.model.num_layers,  
                'hidden_mlp_dims': config.model.hidden_mlp_dims,
                'hidden_dims': config.model.hidden_dims, 
                'input_dims': input_dims, 'output_dims': output_dims}
    if 'mol' in config.model.type:
        params['scale'] = config.model.adj_scale
        output_dims['E'] = 2 # for binary representation
    else:
        params['feat_dict'] = config.data.feat
    return params


def load_ckpt(config, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    path = f'checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt_dict = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model


def load_opt_from_ckpt(config_train, state_dict, model):
    if config_train.optimizer=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, amsgrad=True,
                                        weight_decay=config_train.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optimizer} not implemented.')
    optimizer.load_state_dict(state_dict)
    return optimizer


def load_eval_settings(data, kernel='emd'):
    # Settings for general graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree': kernel, 
                'cluster': kernel, 
                'orbit': kernel,
                'spectral': kernel}
    if data == 'sbm':
        try:
            import graph_tool.all as gt
            methods.append('eval_sbm')
        except:
            pass
    elif data == 'planar':
        methods.append('eval_planar')
    return methods, kernels

