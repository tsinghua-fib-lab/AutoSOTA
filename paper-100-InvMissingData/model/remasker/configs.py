import argparse

def get_args_parser():

    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dataset', default='iris', type=str)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,  help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--norm_field_loss', action='store_true', help='Use (per-patch) normalized field as targets for computing loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
   
    # Model parameters (configurable)
    parser.add_argument('--mask_ratio', default=None, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=None, type=int, help='embedding dimensions')
    parser.add_argument('--depth', default=None, type=int, help='encoder depth')
    parser.add_argument('--decoder_depth', default=None, type=int, help='decoder depth')
    parser.add_argument('--num_heads', default=None, type=int, help='number of heads')
    parser.add_argument('--mlp_ratio', default=None, type=float, help='mlp ratio')
    parser.add_argument('--encode_func', default=None, type=str, help='encoding function')
    
    ###### change this path
    parser.add_argument('--path', default='./', type=str, help='dataset path')
    parser.add_argument('--exp_name', default='', type=str, help='experiment name')
    parser.add_argument('--note', default='', type=str, help='note about the experiment')
    
    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--pin_mem', action='store_false')

    # distributed training parameters    
    return parser



configs = {
    
    'iris' : {
        'embed_dim' : 32,
        'depth' : 4,
        'decoder_depth' : 2,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    'climate' : {
        'embed_dim' : 32,
        'depth' : 4,
        'decoder_depth' : 2,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    'compression' : { 
        'embed_dim' : 32,
        'depth' : 6,
        'decoder_depth' : 4,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    'wine' : { 
        'embed_dim' : 64,
        'depth' : 8,
        'decoder_depth' : 6,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    'yacht' : { 
        'embed_dim' : 32,
        'depth' : 4,
        'decoder_depth' : 2,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },
    
    'spam' : { 
        'embed_dim' : 64,
        'depth' : 8,
        'decoder_depth' : 6,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },
    
        
    'letter' : { 
        'embed_dim' : 64,
        'depth' : 8,
        'decoder_depth' : 6,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },
    
    'credit' : { 
        'embed_dim' : 32,
        'depth' : 6,
        'decoder_depth' : 4,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    'raisin' : { 
        'embed_dim' : 64,
        'depth' : 8,
        'decoder_depth' : 4,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    'bike' : { 
        'embed_dim' : 32,
        'depth' : 8,
        'decoder_depth' : 4,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },
    

    'obesity' : { 
        'embed_dim' : 64,
        'depth' : 8,
        'decoder_depth' : 6,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },

    
    'california' : { 
        'embed_dim' : 32,
        'depth' : 6,
        'decoder_depth' : 4,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },
    
    'diabetes' : { 
        'embed_dim' : 32,
        'depth' : 4,
        'decoder_depth' : 2,
        'num_heads' : 4,
        'mlp_ratio' : 4.,
        'max_epochs' : 600,
        'mask_ratio' : 0.5,
        'encode_func' : 'linear'
    },
}

def get_configs(dataset: str):
    return configs[dataset]

