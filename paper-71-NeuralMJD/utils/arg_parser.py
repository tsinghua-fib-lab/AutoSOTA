import argparse
import logging
import os
import shutil
import random
import sys
from pprint import pformat

import glob
import numpy as np
import torch
import yaml
from ml_collections import config_dict
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.dist_training import get_ddp_save_flag


def parse_arguments(mode='train'):
    """
    Parse CLI arguments, load YAML config, and apply CLI overrides to config.

    Returns:
        (args, config): argparse.Namespace and an ml_collections.ConfigDict
    """
    parser = argparse.ArgumentParser(description="Running Experiments")

    # logging options
    parser.add_argument('-l', '--log_level', type=str,
                        default='DEBUG', help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="A single line comment for the experiment")

    # distributed training options
    parser.add_argument('--dp', default=False, action='store_true',
                        help='To use DataParallel distributed learning.')
    parser.add_argument('--ddp', default=False, action='store_true',
                        help='To use DistributedDataParallel distributed learning.')
    parser.add_argument('--ddp_gpu_ids', nargs='+', default=None,
                        help="A list of GPU IDs to run DDP distributed learning."
                             "For DP mode, please use CUDA_VISIBLE_DEVICES env. variable to specify GPUs.")
    parser.add_argument('--ddp_init_method', default='env://', type=str,
                        help='torch.distributed.init_process_group options.')

    # model options
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size. Overwrite the loaded config if input is not empty.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed.')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers for dataloader.')

    # data filtering options
    parser.add_argument('--data_norm', type=str, choices=['none', 'avg', 'max', 'sqrt', 'minmax'], default=None,
                        help='Data normalization method.')

    # MJD options
    parser.add_argument('--steps_per_unit_time', type=int, default=None,
                        help='To specify the number of steps per unit time.')

    # mode specific options
    if mode == 'train':
        parser.add_argument('-c', '--config_file', type=str, required=True,
                            help="Path of config file")
        parser.add_argument('--dataset_name', default='sp500', type=str, choices=['sp500'],
                            help='To overwrite the dataset name specified in the config.')
        parser.add_argument('--subset', default=None, type=int,
                            help='To overwrite the dataset subset specified in the config.')
        parser.add_argument('--overfit', default=False, action='store_true',
                            help='To overwrite the overfit flag specified in the config.')

        parser.add_argument('--max_epoch', default=None, type=int,
                            help='To overwrite the training epochs specified in the config.')
        parser.add_argument('--lr_init', default=None, type=float,
                            help='To overwrite the initial learning rate specified in the config.')
        parser.add_argument('--lr_schedule', default=None, type=str,
                            help='To overwrite the learning rate schedule specified in the config.')
        parser.add_argument('--save_interval', type=int, default=None,
                            help='To overwrite the save interval specified in the config.')
        parser.add_argument('--resume', type=str, default=None,
                            help='To resume training from the latest checkpoint.')
        parser.add_argument('--skip_ckpt_saving', default=False, action='store_true',
                            help='To skip saving checkpoints during training.')

        # model options
        parser.add_argument('--feature_dims', type=int, default=None,
                            help='To overwrite the model dimension specified in the config.')
        
        # regularization options
        parser.add_argument('--huber_delta', type=float, default=None,
                            help='To use huber loss instead of MSE loss.')
        parser.add_argument('--weight_decay', type=float, default=None,
                            help='To use weight decay for ADAMW.')
        parser.add_argument('--dropout', type=float, default=None,
                            help='To specify model dropout.')

        # MJD options
        parser.add_argument('--w_cond_mean_loss', type=float, default=None,
                            help='To specify the weight of the conditional mean loss.')

        args = parser.parse_args()
    elif mode == 'eval':
        parser.add_argument('-p', '--model_path', type=str, default=None, required=True,
                            help="Path of the model")
        parser.add_argument('--search_weights', default=False, action='store_true',
                            help='To search for network weights inside the path.')
        parser.add_argument('--min_epoch', type=int, default=None,
                            help='Select network weights with minimum number of training epochs.')
        parser.add_argument('--max_epoch', type=int, default=None,
                            help='Select network weights with maximum number of training epochs.')
        parser.add_argument('--num_ckpts', type=int, default=None,
                            help='Select at most k network weights evenly distributed in the [min_epoch, max_epoch] range.')
        parser.add_argument('--use_ema', default='all', nargs='+',
                            help='To use EMA version weight with specified coefficients.')
        parser.add_argument('--section', default='test', choices=['test', 'val'],
                            help='To specify the dataset section for evaluation.')
        parser.add_argument('-c', '--config_file', type=str, default=None, 
                            help="Path of config file")
        args = parser.parse_args()

        # handle special use_ema keywords 'all' or 'none'
        _use_ema = args.use_ema
        if (isinstance(_use_ema, list) and len(_use_ema) == 1) or isinstance(_use_ema, str):
            # either 'all', 'none' or a single value; it must be a string
            _use_ema = _use_ema[0] if isinstance(_use_ema, list) else _use_ema
            assert isinstance(_use_ema, str)
            if _use_ema in ['all', 'none']:
                args.use_ema = None if _use_ema == 'none' else 'all'
            else:
                args.use_ema = [float(_use_ema)]
        else:
            # specific EMA coefficients
            _use_ema = []
            for item in args.use_ema:
                # store float number except for special keywords 'all' or 'none'
                _use_ema.append(float(item) if item not in ['all', 'none'] else item)
            args.use_ema = _use_ema  # always a list

        # handle model path and its config file
        assert isinstance(args.model_path, str) and os.path.exists(args.model_path)
        if os.path.isfile(args.model_path):
            # single model file
            config_file = os.path.abspath(os.path.join(os.path.dirname(args.model_path), '../config.yaml')) if args.config_file is None else args.config_file
            args.model_path = [args.model_path]
        elif os.path.isdir(args.model_path):
            # multiple model files
            assert args.search_weights, 'Please specify --search_weights to search for model weights.'
            config_file = os.path.abspath(os.path.join(args.model_path, '../config.yaml')) if args.config_file is None else args.config_file

            _model_path_ls = sorted(glob.glob(os.path.join(args.model_path, '*.pth')))
            min_epoch = 0 if args.min_epoch is None else args.min_epoch
            max_epoch = float('inf') if args.max_epoch is None else args.max_epoch
            num_ckpts = len(_model_path_ls) if args.num_ckpts is None else args.num_ckpts
            model_path_ls = []
            for model_path in _model_path_ls:
                _epoch = os.path.basename(model_path).split('_')[-1].replace('.pth', '')
                if _epoch == 'best':
                    continue
                else:
                    _epoch = int(_epoch)
                if min_epoch <= _epoch <= max_epoch:
                    model_path_ls.append(model_path)
            if len(model_path_ls) > num_ckpts:
                model_path_ls = model_path_ls[::len(model_path_ls) // num_ckpts]
            args.model_path = model_path_ls
        else:
            raise NotImplementedError
        assert os.path.exists(config_file), 'Config file not found: {:s}'.format(config_file)
        args.config_file = config_file
    else:
        raise NotImplementedError
    args.mode = mode

    """load config file and overwrite config parameters"""
    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    config = config_dict.ConfigDict(config)
    config.lock()
    args_dict = vars(args)

    # overwrite training parameters
    if mode == 'train':
        # overwrite training parameters
        _train_overwrite_keywords = ['max_epoch', 'lr_init', 'lr_schedule', 'batch_size', 'save_interval', 'workers', 'huber_delta', 'weight_decay']
        for keyword in _train_overwrite_keywords:
            if keyword == 'huber_delta':
                _original_param = config.train[keyword]
                if _original_param == 'inf':
                    with config.unlocked():
                        del config.train[keyword]
                        config.train[keyword] = np.inf

            if args_dict[keyword] is not None:
                _original_param = config.train[keyword]
                config.train[keyword] = args_dict[keyword]
                print("Overwriting config file: @train: {:s}, {} {:s} {}".format(
                    keyword, _original_param, '------>', args_dict[keyword]))

        # resume training from a checkpoint
        with config.unlocked():
            config.train.resume = args_dict['resume']
            if config.train.resume is not None:
                assert os.path.exists(config.train.resume), 'Resume file not found: {:s}'.format(config.train.resume)

        # skip saving checkpoints
        with config.unlocked():
            config.train.skip_ckpt_saving = args_dict['skip_ckpt_saving']
            print("Overwriting config file: @train: {:s}, {} {:s} {}".format("skip_ckpt_saving", False, '------>', args_dict['skip_ckpt_saving']))

        # model parameters
        _model_overwrite_keywords = ['feature_dims', 'dropout']
        for keyword in _model_overwrite_keywords:
            if args_dict[keyword] is not None:
                _original_param = config.model[keyword]
                if keyword == 'feature_dims':
                    if isinstance(config.model[keyword], int):
                        config.model[keyword] = args_dict[keyword]
                    elif isinstance(config.model[keyword], list):
                        assert len(config.model[keyword]) == 1
                        config.model[keyword] = [args_dict[keyword]]
                    else:
                        raise NotImplementedError
                else:
                    config.model[keyword] = args_dict[keyword]
                print("Overwriting config file: @model: {:s}, {} {:s} {}".format(
                    keyword, _original_param, '------>', args_dict[keyword]))
        
        # overwrite mjd parameters (used during training)
        _mjd_overwrite_keywords = ['w_cond_mean_loss']
        for keyword in _mjd_overwrite_keywords:
            if args_dict[keyword] is not None:
                _original_param = config.model[keyword]
                config.model[keyword] = args_dict[keyword]
                print("Overwriting config file: @model: {:s}, {} {:s} {}".format(
                    keyword, _original_param, '------>', args_dict[keyword]))
    elif mode == 'eval':
        # overwrite evaluation dataset section
        if args_dict['section'] is not None:
            _original_val_time = config.dataset.validation
            _original_param = config.dataset.testing
            if args_dict['section'] == 'val':
                config.dataset.testing = _original_val_time
            elif args_dict['section'] == 'test':
                pass
            else:
                raise NotImplementedError

            print("Overwriting config file: @section_{}: {} {:s} {}".format(
                args_dict['section'], _original_param, '------>', config.dataset.testing))

    # overwrite dataset path (used during training or testing)
    _dataset_overwrite_keywords = ['dataset_name', 'subset', 'overfit', 'data_norm']
    for keyword in _dataset_overwrite_keywords:
        if keyword in args_dict and args_dict[keyword] is not None:
            _config_key = keyword if keyword != 'dataset_name' else 'name'
            _original_param = config.dataset[_config_key]
            config.dataset[_config_key] = args_dict[keyword]
            print("Overwriting config file: @dataset: {:s}, {} {:s} {}".format(
                _config_key, _original_param, '------>', args_dict[keyword]))
            
    # overwrite sampling parameters (used during training or testing)
    _sampling_overwrite_keywords_train_test = ['batch_size']
    _sampling_overwrite_keywords_test = []
    for keyword in _sampling_overwrite_keywords_train_test + _sampling_overwrite_keywords_test:
        if keyword in _sampling_overwrite_keywords_train_test:
            pass
        elif keyword in _sampling_overwrite_keywords_test and mode == 'eval':
            pass
        else:
            continue
        if args_dict[keyword] is not None:
            if keyword in config.test:
                _original_param = config.test[keyword]
                config.test[keyword] = args_dict[keyword]
            else:
                _original_param = 'None'
                with config.unlocked():
                    config.test[keyword] = args_dict[keyword]
            print("Overwriting config file: @test: {:s}, {} {:s} {}".format(
                keyword, _original_param, '------>', args_dict[keyword]))
    
    # overwrite MJD parameters (used during both training and testing)
    _mjd_overwrite_keywords = ['steps_per_unit_time']
    for keyword in _mjd_overwrite_keywords:
        if args_dict[keyword] is not None:
            _original_param = config.model[keyword]
            config.model[keyword] = args_dict[keyword]
            print("Overwriting config file: @model: {:s}, {} {:s} {}".format(
                keyword, _original_param, '------>', args_dict[keyword]))

    # overwrite random seed
    if args_dict['seed'] is not None:
        _original_param = config.seed
        config.seed = args_dict['seed']
        print("Overwriting config file: @seed: {} {:s} {}".format(
            _original_param, '------>', args_dict['seed']))

    return args, config


def set_seed_and_logger(config, log_level, comment, dist_helper, eval_mode=False):
    """
    Initialize randomness, set PyTorch numerics flags, construct run directories, dump the final config, and configure logging + tensorboard writer.
    """
    # Setup random seed
    if dist_helper.is_ddp:
        config.seed += dist.get_rank()
    else:
        pass
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # torch numerical accuracy flags
    # reference: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    # add log directory
    str_subset = 'sub_{:03d}'.format(config.dataset.subset) if config.dataset.subset is not None else None
    str_overfit = 'overfit' if config.dataset.overfit else None

    str_bs = str_ep = None

    batch_size = config.test.batch_size if eval_mode else config.train.batch_size

    str_bs = 'bs_{:d}'.format(batch_size) if batch_size is not None else None
    str_ep = 'ep_{:d}'.format(config.train.max_epoch) if config.train.max_epoch is not None else None
    str_comment = comment if len(comment) else None
    # str_eval = "EVAL_{}".format('_'.join(config.dataset.testing).replace('/', '_')) if eval_mode else None
    str_eval = "EVAL" if eval_mode else None

    str_folder_name = [
        str_eval,
        config.dataset.name, config.model.name,
        str_subset, str_overfit, 
        str_bs, str_ep, str_comment,
        # time.strftime('%b-%d-%H-%M-%S')
    ]
    logdir = '_'.join([item for item in str_folder_name if item is not None])
    logdir = os.path.join(config.exp_dir, config.exp_name, logdir)

    with config.unlocked():
        config.logdir = logdir
        config.model_ckpt_dir = os.path.join(logdir, 'models_ckpt')
        config.plot_save_dir = os.path.join(logdir, 'plots')
        if 'dev' in config:
            # reset device if it is already set
            config.dev = None
        config.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.logdir, exist_ok=True)
    os.makedirs(config.plot_save_dir, exist_ok=True)
    if not eval_mode:
        os.makedirs(config.model_ckpt_dir, exist_ok=True)

    # dump config to yaml file
    yaml_save_path = os.path.join(config.logdir, 'config.yaml')
    with open(yaml_save_path, 'w') as f:
        config_dict_ = config.to_dict()
        config_dict_['dev'] = str(config.dev)
        yaml.dump(config_dict_, f)

    # setup logger
    if dist_helper.is_ddp:
        log_file = "ddp_rank_{:02d}_".format(dist.get_rank()) + 'logging' + ".log"
    else:
        log_file = 'logging' + ".log"
    if eval_mode:
        log_file = 'eval_' + log_file
    log_file = os.path.join(logdir, log_file)
    log_format = comment + '| %(asctime)s %(message)s'
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format=log_format,
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[fh, logging.StreamHandler(sys.stdout)])

    # avoid excessive logging messages
    logging.getLogger('PIL').setLevel(logging.WARNING)  # avoid PIL logging pollution
    logging.getLogger('matplotlib').setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)  # remove excessive matplotlib messages

    logging.info('EXPERIMENT BEGIN: ' + comment)
    logging.info('logging into %s', log_file)

    # setup tensorboard logger
    if get_ddp_save_flag() and not eval_mode:
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None

    return writer


def backup_code(config, config_file_path):
    logging.info('Config: \n' + pformat(config))
    if get_ddp_save_flag():
        code_path = os.path.join(config.logdir, 'code_backup')
        dirs_to_save = ['model', 'runner', 'utils']
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        os.makedirs(code_path, exist_ok=True)
        if config_file_path is not None:
            shutil.copy(os.path.abspath(config_file_path), os.path.join(config.logdir, 'config_original.yaml'))
 
        os.system('cp ./*py ' + code_path)
        [shutil.copytree(os.path.join('./', this_dir), os.path.join(code_path, this_dir)) for this_dir in dirs_to_save]