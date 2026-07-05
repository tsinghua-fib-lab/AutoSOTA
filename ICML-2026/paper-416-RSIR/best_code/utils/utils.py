import yaml
import random
import torch
import numpy as np
import os
import importlib
import torch.nn as nn
from copy import deepcopy
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.autograd.set_detect_anomaly(True)

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(config):
    if isinstance(config['device'], int):
        # gpu id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])
        config['device'] = 'cuda'

def setup_environment(config):
    seed_everything(config['seed'])
    set_device(config)

def get_model_class(config):
    path = '.'.join(['model', config['model'].lower()])
    module = importlib.import_module(path, __name__)
    model_class = getattr(module, config['model'])
    return model_class

def prepare_datasets(config):
    model_class = get_model_class(config['model'])
    dataset_class = model_class._get_dataset_class(config)

    train_dataset = dataset_class(config, phase='train')
    # val_dataset = dataset_class(config, phase='val_patterns_beta_than9')
    val_dataset = dataset_class(config, phase='val')
    #tmux 0:2;tmux 1:3;tmux 2:4;tmux 10:5;tmux 11:6;tmux 5:7;tmux 6:8;tmux 7:9;tmux 8:than9(least10);tmux 9:all
    #val_patterns_beta_only2.pth
    #val_patterns_beta_than9.pth
    #conda activate env5090
    #python run.py -m SASRec -d "amazon-toys" --trainfile '' --theta_r_path 256 --device 2
    test_dataset = dataset_class(config, phase='test')

    train_dataset.build()
    val_dataset.build()
    test_dataset.build()

    return train_dataset, val_dataset, test_dataset

def prepare_model(config, dataset_list):
    model_class = get_model_class(config['model'])
    model = model_class(config, dataset_list)
    return model

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def normal_initialization(module, initial_range=0.02):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def get_parameter_list(model, detach=True):
    para_list = [p.detach().cpu() for p in model.parameters()]
    return para_list

def flatten_state_dict(state_dict):
    return torch.cat([p.flatten() for _, p in state_dict.items()])

def load_config(config : dict):
    # dataset config
    path = os.path.join('configs', config['dataset'].lower() + '.yaml')
    with open(path, "r") as stream:
        config['data'] = yaml.safe_load(stream)
    config['data']['dataset'] = deepcopy(config['dataset'])
    config.pop('dataset')
    # basemodel config
    model_name = deepcopy(config['model'])
    path = os.path.join('configs', 'basemodel.yaml')
    with open(path, "r") as stream:
        config.update(yaml.safe_load(stream))
    # model config
    path = os.path.join('configs', model_name.lower() + '.yaml')
    with open(path, "r") as stream:
        model_config = yaml.safe_load(stream)
        for key, value in model_config.items():
            config[key].update(value)
    config['model']['model'] = model_name
    try:
        config['data']['train_file'] = config['trainfile']
    except KeyError:
        pass
    try:
        config['train']['device'] = int(config['device'])
    except KeyError:
        pass
    return config

def load_sweep_config(config):
    path = os.path.join('sweep', config['model']['model'].lower() + '.yaml')
    sweep_config = {}
    with open(path, "r") as stream:
        model_config = yaml.safe_load(stream)
        for key, value in model_config.items():
            sweep_config[key] = value
    return sweep_config

def transform_config_into_sweep_config(sweep_config, config):
    for category_k, category_v in config.items():
        for entry_k, entry_v in category_v.items():
            if sweep_config['parameters'].get(category_k + '.' + entry_k, None) == None:
                sweep_config['parameters'][category_k + '.' + entry_k] = {'value': entry_v}
    return sweep_config

def transform_sweep_config_into_config(sweep_config):
    config = {'data': {}, 'model': {}, 'train': {}, 'eval': {}}
    for k, v in sweep_config.items():
        key = k.split('.')
        config[key[0]][key[1]] = v
    return config

class Hypergrad:
    """Implicit differentiation for auxiliary parameters.
    This implementation follows the Algs. in "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf), with small differences.

    """

    def __init__(self, learning_rate=.1, truncate_iter=3):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter

    def grad(self, loss_val, loss_train, aux_params, params):
        """Calculates the gradients w.r.t \phi dloss_aux/dphi, see paper for details

        :param loss_val:
        :param loss_train:
        :param aux_params:
        :param params:
        :return:
        """
        dloss_val_dparams = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )

        dloss_train_dparams = torch.autograd.grad(
                loss_train,
                params,
                allow_unused=True,
                create_graph=True,
        )

        v2 = self._approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params)

        v3 = torch.autograd.grad(
            dloss_train_dparams,
            aux_params,
            grad_outputs=v2,
            allow_unused=True
        )

        # note we omit dL_v/d_lambda since it is zero in our settings
        return list(-g for g in v3)

    def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
        """

        :param dloss_val_dparams: dL_val/dW
        :param dloss_train_dparams: dL_train/dW
        :param params: weights W
        :return: dl_val/dW * dW/dphi
        """
        p = v = dloss_val_dparams

        for _ in range(self.truncate_iter):
            grad = torch.autograd.grad(
                    dloss_train_dparams,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True
                )

            grad = [g * self.learning_rate for g in grad]  # scale: this a is key for convergence

            v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
            # note: different than the pseudo code in the paper
            p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

        return list(pp for pp in p)

class MetaOptimizer:

    def __init__(self, meta_optimizer, hpo_lr, truncate_iter=3, max_grad_norm=10):
        """Auxiliary parameters optimizer wrapper

        :param meta_optimizer: optimizer for auxiliary parameters
        :param hpo_lr: learning rate to scale the terms in the Neumann series
        :param truncate_iter: number of terms in the Neumann series
        :param max_grad_norm: max norm for grad clipping
        """
        self.meta_optimizer = meta_optimizer
        self.hypergrad = Hypergrad(learning_rate=hpo_lr, truncate_iter=truncate_iter)
        self.max_grad_norm = max_grad_norm

    def step(self, train_loss, val_loss, parameters, aux_params, return_grads=False):
        """

        :param train_loss: train loader
        :param val_loss:
        :param parameters: parameters (main net)
        :param aux_params: auxiliary parameters
        :param return_grads: whether to return gradients
        :return:
        """
        # zero grad
        self.zero_grad()

        # validation loss
        hyper_gards = self.hypergrad.grad(
            loss_val=val_loss,
            loss_train=train_loss,
            aux_params=aux_params,
            params=parameters
        )

        for p, g in zip(aux_params, hyper_gards):
            p.grad = g

        # grad clipping
        if self.max_grad_norm is not None:
            clip_grad_norm_(aux_params, max_norm=self.max_grad_norm)

        # meta step
        self.meta_optimizer.step()
        if return_grads:
            return hyper_gards

    def zero_grad(self):
        self.meta_optimizer.zero_grad()

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, hard=False, eps=1e-10):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.eps = eps

    def forward(self, scores, tau=1):
        device = scores.device
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores, device=device)
        onehot_approx = torch.zeros_like(scores, device=device)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([self.eps], device=device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot, device=device)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res
