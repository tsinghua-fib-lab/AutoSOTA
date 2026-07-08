import os

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()
        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)
        if self.lock:
            self.lock.release()
        if verbose:
            print(str)


def set_log(config, is_train=True):
    data = config.data.data
    exp_name = config.train.name

    log_folder_name = os.path.join(*[data, exp_name])
    root = 'logs_train' if is_train else 'logs_sample'
    if not(os.path.isdir(f'./{root}/{log_folder_name}')):
        os.makedirs(os.path.join(f'./{root}/{log_folder_name}'))
    log_dir = os.path.join(f'./{root}/{log_folder_name}/')

    if not(os.path.isdir(f'./checkpoints/{data}')) and is_train:
        os.makedirs(os.path.join(f'./checkpoints/{data}'))
    ckpt_dir = os.path.join(f'./checkpoints/{data}/')

    print('-'*100)
    print("Make Directory {} in Logs".format(log_folder_name))

    return log_folder_name, log_dir, ckpt_dir


def check_log(log_folder_name, log_name):
    return os.path.isfile(f'./logs_sample/{log_folder_name}/{log_name}.log')


def data_log(logger, config):
    if 'feat' not in config.data.keys():
        config.data.feat = config.data.init
    logger.log(f'[{config.data.data}] feat={config.data.feat} ({config.data.max_feat_num}) seed={config.seed} '
                f'batch_size={config.data.batch_size} perm_mix={config.data.perm_mix}')

def mix_log(logger, config_mix):
    mix_x = config_mix.x
    mix_adj = config_mix.adj

    if mix_x.type == 'BB':
        logger.log(f'(x  :{mix_x.type}), [{mix_x.sigma_0:.3e},{mix_x.sigma_1:.3e}]) N={mix_x.num_scales}') 
    else:
        logger.log(f'(x  :{mix_x.type}), {mix_x.drift_coeff}, '
                    f'[{mix_x.sigma_0:.3e},{mix_x.sigma_1:.3e}]) N={mix_x.num_scales}') 
    if mix_adj.type == 'BB':
        logger.log(f'(adj:{mix_adj.type}), [{mix_adj.sigma_0:.3e},{mix_adj.sigma_1:.3e}]) N={mix_adj.num_scales}')
    else:
        logger.log(f'(adj:{mix_adj.type}), {mix_adj.drift_coeff}, '
                    f'[{mix_adj.sigma_0:.3e},{mix_adj.sigma_1:.3e}]) N={mix_adj.num_scales}')
    logger.log('-'*100)


def model_log(logger, params):
    logger.log(f"Input dims={params['input_dims']}  Output dims={params['output_dims']}")
    logger.log(f"Model={params['model_type']}  Layers={params['n_layers']}  MLP_dims={params['hidden_mlp_dims']} ")
    logger.log(f"hid_dims={params['hidden_dims']}")
    logger.log('-'*100)


def start_log(logger, config):
    logger.log('-'*100)
    data_log(logger, config)
    logger.log('-'*100)


def train_log(logger, config, params):
    logger.log(f'lr={config.train.lr:.1e} schedule={config.train.lr_schedule}  '
                f'epochs={config.train.num_epochs} optimizer={config.train.optimizer} '
                f'weight_decay={config.train.weight_decay} grad_norm={config.train.grad_norm} ')
    logger.log(f'eps={config.train.eps:.1e} ema={config.train.ema} lambda_train={config.train.lambda_train} '
                f'loss_type={config.train.loss_type}')
    logger.log('-'*100)
    mix_log(logger, config.mix)
    model_log(logger, params)


def sample_log(logger, config):
    sampler = config.sampler
    sample_log = f"({sampler.predictor})+({sampler.corrector})+(ref={config.sample.ref})+(gamma={config.sample.gamma})+(tc={config.sample.tc})" 
    if sampler.corrector != 'None':
        sample_log += f'snr={sampler.snr:.2f} seps={sampler.scale_eps:.1f} n_steps={sampler.n_steps} '
    if config.data.data in ['QM9', 'ZINC250k']:
        sample_log += f"eps={config.sample.eps}"
    else:
        sample_log += f"eps={config.sample.eps} kernel={config.sample.kernel}"
    logger.log(sample_log)
    logger.log('-'*100)


def resume_log(logger, config, params):
    print(f'lr={config.train.lr:.1e} schedule={config.train.lr_schedule}  '
            f'epochs={config.train.num_epochs} optimizer={config.train.optimizer} '
            f'weight_decay={config.train.weight_decay} grad_norm={config.train.grad_norm} ')
    print(f'eps={config.train.eps:.1e} ema={config.train.ema} '
            f'lambda_train={config.train.lambda_train}')
    print('-'*100)
    mix_x = config.mix.x
    mix_adj = config.mix.adj

    if mix_x.type == 'BB':
        print(f'(x  :{mix_x.type}), [{mix_x.sigma_0:.3e},{mix_x.sigma_1:.3e}]) N={mix_x.num_scales}') 
    else:
        print(f'(x  :{mix_x.type}), {mix_x.drift_coeff}, '
                    f'[{mix_x.sigma_0:.3e},{mix_x.sigma_1:.3e}]) N={mix_x.num_scales}') 
    if mix_adj.type == 'BB':
        print(f'(adj:{mix_adj.type}), [{mix_adj.sigma_0:.3e},{mix_adj.sigma_1:.3e}]) N={mix_adj.num_scales}')
    else:
        print(f'(adj:{mix_adj.type}), {mix_adj.drift_coeff}, '
                    f'[{mix_adj.sigma_0:.3e},{mix_adj.sigma_1:.3e}]) N={mix_adj.num_scales}')
    print('-'*100)
    print(f"Input dims={params['input_dims']}  Output dims={params['output_dims']}")
    print(f"Layers={params['n_layers']}, MLP_dims={params['hidden_mlp_dims']} ")
    print(f"hid_dims={params['hidden_dims']}")
    logger.log('-'*100)
    logger.log('Resume')
    logger.log('-'*100)