from __future__ import absolute_import, division, print_function

import os
import argparse
import configargparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def str2bool(v):
     if isinstance(v, bool):
          return v
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise argparse.ArgumentTypeError('Boolean value expected.')

class MonodepthOptions:
     def __init__(self):
          self.parser = configargparse.ArgumentParser(description="Monodepthv2 options", fromfile_prefix_chars='@')
          # PATHS
          self.parser.add_argument('--config', is_config_file=True, help='config file path')
          self.parser.add_argument("--log_dir", type=str, help="log directory", default=os.path.join(os.path.expanduser("~"), "tmp"))

          # Dataset
          self.parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='nyu')
          self.parser.add_argument('--input_height', type=int, help='input height', default=480)
          self.parser.add_argument('--input_width', type=int, help='input width', default=640)
        
          # Preprocessing
          self.parser.add_argument('--aug', action='store_true', default=False)
          self.parser.add_argument('--degree', type=float, default=2.5)  
          self.parser.add_argument('--do_random_rotate', action='store_true', default=False)
          self.parser.add_argument('--random_crop', action='store_true', default=False)
          self.parser.add_argument('--random_translate', action='store_true', default=False)
        
          # Multi-gpu training
          self.parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
          self.parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
          self.parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
          self.parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
          self.parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
          self.parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
          self.parser.add_argument('--distributed',           help='Use distributed training to launch ', action='store_true',)
          self.parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                           'N processes per node, which has N GPUs. This is the '
                                                                           'fastest way to use PyTorch for either single node or '
                                                                           'multi node data parallel training', action='store_true',)
          # Data
          self.parser.add_argument('--data_path_eval',
                                 type=str,
                                 help='path to the data for evaluation')
          self.parser.add_argument('--gt_path_eval',
                                 type=str,
                                 help='path to the groundtruth data for evaluation')
          self.parser.add_argument('--filenames_file_eval',
                                 type=str,
                                 help='path to the filenames text file for evaluation')
          self.parser.add_argument('--data_path',
                                 type=str,
                                 help='path to the data for training')
          self.parser.add_argument('--txt_path',
                                 type=str,
                                 help='path to the text for training')
          self.parser.add_argument('--txt_path_eval',
                                 type=str,
                                 help='path to the text for training')
          self.parser.add_argument('--gt_path',
                                 type=str,
                                 help='path to the groundtruth data for training')
          self.parser.add_argument('--filenames_file',
                                 type=str,
                                 help='path to the filenames text file for evaluation')
        
          # Eval
          self.parser.add_argument('--min_depth_eval',
                                 type=float,
                                 help='minimum depth for evaluation',
                                 default=1e-3)
          self.parser.add_argument('--max_depth_eval',
                                 type=float,
                                 help='maximum depth for evaluation',
                                 default=10)
          self.parser.add_argument('--eigen_crop',
                                 help='if set, crops according to Eigen NIPS14',
                                 action='store_true',
                                 default=True)
          self.parser.add_argument('--garg_crop',
                                 help='if set, crops according to Garg  ECCV16',
                                 action='store_true')
          self.parser.add_argument('--visualize_results',
                                 help='if set, visualize evaluation results',
                                 action='store_true')
        
          # Training
          self.parser.add_argument('--weight_decay',
                                 type=float,
                                 help='weight decay factor for optimization',
                                 default=1e-2)
          self.parser.add_argument('--adam_eps',
                                 type=float,
                                 help='epsilon in Adam optimizer',
                                 default=1e-6)
          self.parser.add_argument('--batch_size',
                                 type=int,
                                 help='batch size',
                                 default=20)
          self.parser.add_argument('--num_epochs',
                                 type=int,
                                 help='number of epochs',
                                 default=50)
          self.parser.add_argument('--learning_rate',
                                 type=float,
                                 help='initial learning rate',
                                 default=1e-5)
          self.parser.add_argument('--end_learning_rate',
                                 type=float,
                                 help='end learning rate',
                                 default=-1)
          self.parser.add_argument('--norm_loss',
                                 action='store_true')
          self.parser.add_argument('--combine_words_no_area',
                                 action='store_true')
          self.parser.add_argument('--use_amp',
                                 action='store_true', default=True)

          self.parser.add_argument('--close_car_percent',
                                 type=float,
                                 default=0.01)
          self.parser.add_argument('--far_car_percent',
                                 type=float,
                                 default=0.001)

          # Log and save
          self.parser.add_argument('--model_name',
                                 type=str,
                                 help='model name', default='scalemap')
          self.parser.add_argument('--log_directory',
                                 type=str,
                                 help='directory to save checkpoints and summaries',
                                 default='./models/')
          self.parser.add_argument('--log_freq',
                                 type=int,
                                 help='Logging frequency in global steps',
                                 default=100)
          self.parser.add_argument('--log_print_freq',
                                 type=int,
                                 help='Logging frequency in global steps',
                                 default=500)
          self.parser.add_argument('--save_freq_ckpt',
                                 type=int,
                                 help='Checkpoint saving frequency in global steps',
                                 default=50000)
          self.parser.add_argument('--eval_freq',
                                 type=int,
                                 help='Eval frequency in global steps',
                                 default=500)
          self.parser.add_argument('--eval_before_train',
                                 action='store_true')
          self.parser.add_argument('--load_ckpt_path',
                                 type=str,
                                 default=None)

          self.parser.add_argument('--depth_model',
                                 type=str,
                                 default='da_b')

          # Void
          self.parser.add_argument('--val_image_path_void',
                                 type=str)
          self.parser.add_argument('--val_sparse_depth_path_void',
                                 type=str)
          self.parser.add_argument('--val_intrinsics_path_void',
                                 type=str)
          self.parser.add_argument('--val_ground_truth_path_void',
                                 type=str)
          self.parser.add_argument('--txt_path_eval_void',
                                 type=str)

          # ScaleMap model
          self.parser.add_argument('--text_encoder',
                                 type=str,
                                 default='RN50')
          self.parser.add_argument('--img_encoder',
                                 type=str,
                                 default='vits')
          self.parser.add_argument('--w_sig',
                                 type=float,
                                 default=10.0)
          self.parser.add_argument('--w_sts',
                                 type=float,
                                 default=2.0)
          self.parser.add_argument('--w_eas',
                                 type=float,
                                 default=0.1)
          self.parser.add_argument('--w_scon',
                                 type=float,
                                 default=0.2)
          self.parser.add_argument('--image_channels',
                                 type=int,
                                 default=1024)
          self.parser.add_argument('--text_channels',
                                 type=int,
                                 default=1024)
          self.parser.add_argument('--out_channels',
                                 type=int,
                                 default=192)
          self.parser.add_argument('--features',
                                 type=int,
                                 default=128)
          self.parser.add_argument('--cross_attn_nhead',
                                 type=int,
                                 default=8) 
          self.parser.add_argument('--coef_scale',
                                 type=float,
                                 default=0.01)
          self.parser.add_argument('--coef_shift',
                                 type=float,
                                 default=0.01)          
          self.parser.add_argument('--lambdad',
                                 type=float,
                                 default=1.0) 
          self.parser.add_argument('--train_1_dataset',
                                 type=str,
                                 default=None,
                                 choices=['nyu', 'kitti', 'void', 'c3vd'])
           # Pseudo depth and Scale Oriented contrastive 
          self.parser.add_argument('--warmup_iter',
                                 type=int,
                                 default=1000) 
          self.parser.add_argument('--scon_start_epoch',
                                 type=int,
                                 default=1) 
          self.parser.add_argument('--momentum',
                                 type=float,
                                 default=0.9)    
          self.parser.add_argument('--weight_decay_con',
                                 type=float,
                                 default=1e-5)  
          self.parser.add_argument('--pixpro_momentum',
                                 type=float,
                                 default=0.99)     
          self.parser.add_argument('--load_from_con_path',
                                 type=str,
                                 default=None)

          # Dataset roots (override hard-coded paths in dataloaders/change_to_*)
          # In-domain (eval_4d.py)
          self.parser.add_argument('--nyu_root', type=str, default=None,
                                 help='root directory of NYU Depth V2 (contains sync/ and official_splits/)')
          self.parser.add_argument('--kitti_root', type=str, default=None,
                                 help='root directory of KITTI raw data')
          self.parser.add_argument('--kitti_gt_root', type=str, default=None,
                                 help='root directory of KITTI ground-truth depth')
          self.parser.add_argument('--void_root', type=str, default=None,
                                 help='root directory of VOID release')
          self.parser.add_argument('--c3vd_root', type=str, default=None,
                                 help='root directory of C3VD (undistorted)')
          # Zero-shot (eval_zero.py)
          self.parser.add_argument('--sunrgbd_root', type=str, default=None,
                                 help='root directory of SUN RGB-D')
          self.parser.add_argument('--ibims_root', type=str, default=None,
                                 help='root directory of iBims-1')
          self.parser.add_argument('--diode_root', type=str, default=None,
                                 help='root directory of DIODE outdoor val split')
          self.parser.add_argument('--hypersim_root', type=str, default=None,
                                 help='root directory of HyperSim')
          self.parser.add_argument('--simcol_root', type=str, default=None,
                                 help='root directory of SimCol')

          # Logging
          self.parser.add_argument('--log_file', type=str, default=None,
                                 help='if set, mirror stdout and stderr to this file in addition to the console; '
                                      'pass "auto" to write to logs/<script>_<timestamp>.log')


     def parse(self):
          self.options = self.parser.parse_args()
          return self.options
