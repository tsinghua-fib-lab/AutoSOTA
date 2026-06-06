# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/hparams_registry.py
# The specific hyper-parameters for each algorithm

import numpy as np
from engine.utils import misc
from einops import rearrange, repeat


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(cfg):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms.py / networks / etc. should add entries here.
    """

    random_seed = cfg.seed
    dataset = cfg.data_name
    algorithm = cfg.algorithm

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and returns a random hyperparameter value."""
        assert (name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Define global parameters
    # Note that domain_num is for test only
    _hparam('source_domains', cfg.source_domains, lambda r: cfg.source_domains)
    _hparam('num_classes', cfg.num_classes, lambda r: cfg.num_classes)
    _hparam('data_size', cfg.data_size, lambda r: cfg.data_size)
    _hparam('algorithm', cfg.algorithm, lambda r: cfg.algorithm)
    _hparam('data_name', cfg.data_name, lambda r: cfg.data_name)

    ## rebuttal
    _hparam('weight_decay', cfg.weight_decay, lambda r: r.choice([0, 1e-3, 5e-4, 5e-5]))

    if dataset == 'RMNIST':
        _hparam('lr', 5e-4, lambda r: 10 ** r.uniform(-4.5, -2.5))
    elif dataset == 'Portraits':
        _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))
    elif dataset in ['ToyCircle', 'PowerSupply', 'ONP']:
        _hparam('lr', 5e-6, lambda r: 10 ** r.uniform(-5, -3.5))
    elif dataset in ['ToySine', 'Caltran']:
        _hparam('lr', 1e-5, lambda r: 10 ** r.uniform(-5, -3.5))
    else:
        _hparam('lr', 5e-6, lambda r: 10 ** r.uniform(-5, -3.5))

    if algorithm in ['SYNC']:
        _hparam('zc_dim', cfg.zc_dim, lambda r: cfg.zc_dim)
        _hparam('zd_dim', cfg.zd_dim, lambda r: cfg.zd_dim)
        _hparam('lambda_evolve', cfg.lambda_evolve, lambda r: cfg.lambda_evolve)
        _hparam('lambda_mi', cfg.lambda_mi, lambda r: cfg.lambda_mi)
        _hparam('lambda_causal', cfg.lambda_causal, lambda r: cfg.lambda_causal)

    if algorithm in ['VAE', 'DIVA', 'LSSAE', 'MMD_LSAE',]:
        _hparam('zc_dim', cfg.zc_dim, lambda r: cfg.zc_dim)
        _hparam('zw_dim', cfg.zw_dim, lambda r: cfg.zw_dim)
        _hparam('zdy_dim', cfg.zw_dim, lambda r: cfg.zw_dim)
        _hparam('momentum', 0., lambda r: 0.)

    return hparams


def get_hparams(cfg):
    return {a: b for a, (b, c) in _hparams(cfg).items()}
