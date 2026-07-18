import os
import random
import yaml
import argparse
import numpy as np
import torch

from dataset_attack import load_raw_data
from attack_backtime_trainer import Trainer
from easydict import EasyDict as edict
from utils import misc

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parser_args():
    current_time = misc.get_current_datetime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", type=str, default="./configs/attacks/PEMS03_backtime_FEDformer_1212_attack.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    print(args)

    default_config = yaml.load(open('./configs/default_config.yaml', 'r'), 
                               Loader=yaml.FullLoader)

    # load training config
    config = yaml.load(open(args.train_config_path), 
                       Loader=yaml.FullLoader)['Train']

    #  Add training config to default config
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']

    config['Surrogate']['token_len'] = config['token_len']
    config['Surrogate']['seq_len'] = config['seq_len']
    config['Surrogate']['label_len'] = config['label_len']
    config['Surrogate']['pred_len'] = config['pred_len']


    args_dict = vars(args)
    config.update(args_dict)
    config['current_time'] = current_time
    return edict(config)



def main(config):
    # Set device
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        DEVICE = torch.device('cuda')
        print("CUDA:", USE_CUDA, DEVICE, "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    else:
        DEVICE = torch.device("cpu")
        print("!!! CUDA IS NOT AVAILABLE, USING", DEVICE)
    
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    ATTACK_SAVE_FOLDER = os.path.join(config.checkpoint_dir, f"{config.dataset}_backtime_{config.surrogate_name}_{config.seq_len}{config.pred_len}")


    config.Surrogate.device = DEVICE
    seed_torch()
    data_config = config.Dataset
    train_data_stamps, test_data_stamps = None, None

    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)

    # set attacked variables
    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)

    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)
    print('Shape of attacked variables', atk_vars.shape)
    print("Attacked variables:", atk_vars)

    # load target pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE)*train_std # oke they multiples it with std of the dataset, new move

    exp_trainer = Trainer(config, atk_vars, target_pattern, 
                          train_mean, train_std, train_data_seq, test_data_seq, 
                          train_data_stamps, test_data_stamps, DEVICE)

    attacker_save_file = os.path.join(ATTACK_SAVE_FOLDER, "attacker.pth") 
    if os.path.exists(attacker_save_file) and config.is_attacker_trained:
        attacker_state = torch.load(attacker_save_file)
        exp_trainer.load_attacker(attacker_state)
        print('Load attacker checkpoint from', attacker_save_file)
    else:
        print('=' * 20, ' [ Stage 1 ] ', '=' * 20)
        print('Start training surrogate model and attacker')
        exp_trainer.train()

        attacker_state = exp_trainer.save_attacker()
        torch.save(attacker_state, attacker_save_file)
        print(f"Save attacker checkpoint at {attacker_save_file}")

    print('=' * 20, ' [ Stage 2 ] ', '=' * 20)
    print('Saving poisoning attack results...')
    seed_torch()
    exp_trainer.save_poisoning_data(attack_save_folder=ATTACK_SAVE_FOLDER) 

if __name__ == "__main__":
    config = parser_args()
    main(config)

