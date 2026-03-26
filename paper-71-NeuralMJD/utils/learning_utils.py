import logging
from typing import Tuple, Any, Optional

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from ema_pytorch import EMA
from model.transformer import MJDTransformer
from model.mjd.neural_mjd import NeuralMJD


def get_network(config, dist_helper) -> torch.nn.Module:
    """
    Construct the neural network according to the dataset and model config.

    Returns:
        A torch.nn.Module moved to the appropriate device and adapted for distributed training if enabled.
    """
    data_config = config.dataset
    dataset_nm = config.dataset.name

    if dataset_nm.startswith('sp500'):
        in_seq_dim = 2
        num_static_features = 1
    else:
        raise ValueError(f'Unknown dataset name {dataset_nm}')


    if config.model.name in ['neural_mjd', 'neural_bs']:
        if dataset_nm.startswith('sp500'):
            ffn_embedding_dim = config.model.feature_dims
        else:
            raise ValueError(f'Unknown dataset name {dataset_nm}')

        assert config.model.network == 'transformer', "Only transformer is supported for SDE models"

        in_seq_length = data_config.seqlen - data_config.predlen
        out_seq_length = data_config.predlen

        if config.model.name == 'neural_bs':
            output_dim = 2
        elif config.model.name == 'neural_mjd':
            output_dim = 5
        else:
            raise ValueError(f'Unknown model name {config.model.name}')

        network = MJDTransformer(
                # input/output dimensions
                in_seq_length=in_seq_length,
                in_seq_dim=in_seq_dim,
                out_seq_length=out_seq_length,
                out_seq_dim=output_dim,
                num_static_features=num_static_features,
                # transformer parameters
                num_encoder_layers=config.model.num_layers,
                embedding_dim=config.model.feature_dims,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=config.model.num_attention_heads,
                pre_layernorm=config.model.pre_layernorm,
                activation_fn=config.model.activation_fn,
                dropout=config.model.dropout,
                light_mode='sp500' in dataset_nm,
        )

        if config.model.name in ['neural_mjd', 'neural_bs']:
            model = NeuralMJD(
                model=network,
                w_cond_mean_loss=config.model.w_cond_mean_loss,
                steps_per_unit_time=config.model.steps_per_unit_time,
                jump_diffusion=config.model.name == 'neural_mjd',
                s_0_from_avg='sp500' not in dataset_nm,
                cond_mean_raw_scale='sp500' not in dataset_nm,
            )
        else:
            raise NotImplementedError(f'Unknown diffusion objective {config.model.name}')
    else:
        raise ValueError(f'Unknown model name {config.model.name}')
    
    model = model.to(dist_helper.device)

    # count model parameters
    param_string, total_params, total_trainable_params = count_model_params(model)
    # logging.info(f"Parameters: \n{param_string}")  # print model parameters
    logging.info(f"Parameters Count: {total_params:,}, Trainable: {total_trainable_params:,}")

    # load checkpoint to resume training
    if config.train.resume is not None:
        logging.info("Resuming training from checkpoint: {:s}".format(config.train.resume))
        ckp_data = torch.load(config.train.resume)
        model = load_model(ckp_data, model, 'model')

    # adapt to distributed training
    if dist_helper.is_distributed:
        model = dist_helper.dist_adapt_model(model)
    else:
        logging.info("Distributed OFF. Single-GPU training.")

    return model


def count_model_params(model: torch.nn.Module) -> Tuple[str, int, int]:
    """
    Iterate model params and return a formatted string and counts.
    """
    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()} {param.numel()}")
    param_string = '\n'.join(param_strings)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_string, total_params, total_trainable_params


def get_optimizer(model: torch.nn.Module, config, dist_helper) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """
    Configure the optimizer (AdamW or ZeroRedundancyOptimizer) and LR scheduler.
    """
    if isinstance(model, torch.nn.Module):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = 0
    if num_params:
        if dist_helper.is_ddp:
            optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                                optimizer_class=torch.optim.Adam,
                                                lr=config.train.lr_init,
                                                betas=(0.9, 0.999), eps=1e-8,
                                                weight_decay=config.train.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config.train.lr_init,
                                          betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=config.train.weight_decay)
        if config.train.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_epoch, eta_min=config.train.lr_min)
        elif config.train.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_step_size, gamma=config.train.lr_gamma)
        elif config.train.lr_schedule == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey,)
        elif config.train.lr_schedule == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        else:
            scheduler = None
    else:
        optimizer, scheduler = None, None
    return optimizer, scheduler


def get_ema_helper(config, model: torch.nn.Module):
    """
    Setup exponential moving average training helper(s) if enabled in the config.
    """
    if isinstance(model, torch.nn.Module):
        flag_ema = False
        ema_coef = config.train.ema_coef if hasattr(config.train, 'ema_coef') else 1.0
        if isinstance(ema_coef, list):
            flag_ema = True
        if isinstance(ema_coef, float):
            flag_ema = ema_coef < 1
        if flag_ema:
            ema_coef = [ema_coef] if isinstance(ema_coef, float) else ema_coef
            assert isinstance(ema_coef, list)
            ema_helper = []
            for coef in sorted(ema_coef):
                if coef >= 1:
                    raise ValueError("EMA coefficient must be less than 1!")
                ema = EMA(model=model, beta=coef, update_every=1, update_after_step=0, inv_gamma=1, power=1)
                ema_helper.append(ema)
            logging.info("Exponential moving average (EMA) is ON. Coefficient: {}".format(ema_coef))
        else:
            ema_helper = None
            logging.info("Exponential moving average (EMA) is OFF.")
        return ema_helper
    else:
        return None


def load_model(ckp_data: dict, model: torch.nn.Module, weight_keyword: str) -> torch.nn.Module:
    """
    Load network weights from a checkpoint into the provided model.

    Args:
        ckp_data: Loaded checkpoint dictionary.
        model: Model instance to load parameters into.
        weight_keyword: Which key in the checkpoint to load (e.g., 'model' or 'model_ema_beta_0.9990').
    """
    assert weight_keyword in ckp_data
    cur_keys = sorted(list(model.state_dict().keys()))
    ckp_keys = sorted(list(ckp_data[weight_keyword].keys()))
    if set(cur_keys) == set(cur_keys) & set(ckp_keys):
        model.load_state_dict(ckp_data[weight_keyword], strict=True)
    else:
        to_load_state_dict = {}
        for cur_key, ckp_key in zip(cur_keys, ckp_keys):
            if cur_key == ckp_key:
                pass
            # note: .module prefix is added during the DP training
            elif cur_key.startswith('module.') and not ckp_key.startswith('module.'):
                assert cur_key == 'module.' + ckp_key
            elif ckp_key.startswith('module.') and not cur_key.startswith('module.'):
                assert 'module.' + cur_key == ckp_key
            else:
                raise NotImplementedError
            to_load_state_dict[cur_key] = ckp_data[weight_keyword][ckp_key]
        assert set(cur_keys) == set(list(to_load_state_dict.keys()))
        model.load_state_dict(to_load_state_dict, strict=True)
        del to_load_state_dict
        torch.cuda.empty_cache()
    return model