import torch
from tabdiff.main import main as tabdiff_main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of TabDiff')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name dataset, one of those in data/ dir')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--method', type=str, default='tabdiff', help='Currently we only release our model TabDiff. Baselines will be released soon.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name, used to name log directories and the wandb run name')
    parser.add_argument('--deterministic', action='store_true', help='Whether to make the entire process deterministic, i.e., fix global random seeds')
    
    # Configs for tabdiff
    parser.add_argument('--y_only', action='store_true', help='Train guidance model that only model the target column')
    parser.add_argument('--non_learnable_schedule', action='store_true', help='disable learnable noise schedule')
    
    # Configs for testing tabdiff
    parser.add_argument('--num_samples_to_generate', type=int, default=None, help='Number of samples to be generated while testing')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model checkpoint to be tested')
    parser.add_argument('--report', action='store_true', help="Report testing mode: this mode sequentially runs <num_runs> test runs and report the avg and std")
    parser.add_argument('--num_runs', type=int, default=20, help="Number of runs to be averaged in the report testing mode")
    
    # Configs for imputation
    parser.add_argument('--impute', action='store_true')
    parser.add_argument('--trial_start', type=int, default=0)
    parser.add_argument('--trial_size', type=int, default=50)
    parser.add_argument('--resample_rounds', type=int, default=1)
    parser.add_argument('--impute_condition', type=str, default="x_t")
    parser.add_argument('--y_only_model_path', type=str, default=None, help="Path to the y_only model checkpoint that will be used as the unconditional guidance model")
    parser.add_argument('--w_num', type=float, default=0.6)
    parser.add_argument('--w_cat', type=float, default=0.6)

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    tabdiff_main(args)