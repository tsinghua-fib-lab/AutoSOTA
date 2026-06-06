import os
import time
import argparse

def get_options(args=None):
    parser = argparse.ArgumentParser(description="Mamba-DAC")
    
    # General options
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--train_online", action="store_true", help = "Train the model in online manner")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trajectory_file_path", type=str, default='./trajectory_files/trajectories_set_alg0/trajectory_set_0_Unit.pkl')
    parser.add_argument("--model", type=str, default="q_mamba",choices=["q_mamba"])
    parser.add_argument("--model_dir", type=str, default="./model/")
    parser.add_argument("--shuffle", action="store_true", default = False, help="Shuffle the dataset")
    parser.add_argument("--resume",type=str, default = None, help="a txt or json, to resume training")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lambda", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=10.0)
    
    # Q-Mamba options
    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--actions_dim", type=int, default=5)
    parser.add_argument("--action_bins", type=int, default=16)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--num_hidden_mlp", type=int, default=32)
    parser.add_argument("--mamba_num", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--has_conservative_reg_loss", action="store_true", default = False , help="Use conservative reg loss")

    # training options
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=200)
    
    # testing options
    parser.add_argument("--algorithm_id", type=int, default=0)
    parser.add_argument("--load_path", type=str, default=None)
    
    # logging options
    parser.add_argument("--log_path", type=str, default="./log/")
    parser.add_argument("--log_name", type=str, default=None)
    
    opts = parser.parse_args(args)
    
    opts.time_stamp = time.strftime("%Y%m%dT%H%M%S")
    if opts.log_name is None:
        opts.log_name = opts.time_stamp
        
    return opts
