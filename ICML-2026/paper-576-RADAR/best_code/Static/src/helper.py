import json 
import random
import numpy as np
import torch

# 此文件提供了一组与实验运行、IO、字符串清洗、评分与生成停止判定相关的辅助函数与类，供仓库其余模块复用。

def get_log_name(args):
    if args.defense_method == 'none': defense_str = 'none'
    elif args.defense_method == 'voting': defense_str = f'{args.defense_method}-gamma{args.gamma}'
    elif args.defense_method == 'keyword': defense_str = f'{args.defense_method}-{args.alpha}-{args.beta}-gamma{args.gamma}' 
    elif args.defense_method == 'decoding': defense_str = f'{args.defense_method}-{args.eta}-gamma{args.gamma}' 
    elif args.defense_method == 'graph': defense_str = f'{args.defense_method}' 
    elif args.defense_method == 'MIS': defense_str = f'{args.defense_method}' 
    elif args.defense_method == 'greedy': defense_str = f'{args.defense_method}-{args.eta}' 
    elif args.defense_method == 'sampling': defense_str = f'{args.defense_method}-{args.T}-{args.m}-gamma{args.gamma}'
    elif args.defense_method == 'sampling_keyword': defense_str = f'{args.defense_method}-T{args.T}-m{args.m}-gamma{args.gamma}-a{args.alpha}-b{args.beta}' 
    elif args.defense_method == 'astuterag': defense_str = f'{args.defense_method}'
    elif args.defense_method == 'instructrag_icl': defense_str = f'{args.defense_method}'
    elif args.defense_method == 'trustrag': defense_str = f'{args.defense_method}' 
    elif args.defense_method == 'em_based': defense_str = f'{args.defense_method}-mu{args.prior_mu}-kappa{args.prior_kappa}-err{args.err}'
    elif args.defense_method == 'mincut': defense_str = f'{args.defense_method}'
    elif args.defense_method == 'cluster': defense_str = f'{args.defense_method}'
    else: defense_str = ""
    return f'{args.dataset_name}-{args.model_name}-{defense_str}-rep{args.rep}-top{args.top_k}-attack{args.attack_method}-gamma{args.gamma}-attackpos{args.attackpos}-corruptionsz{args.corruption_size}'


def load_jsonl(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:  # This avoids loading all lines into memory at once
            results.append(json.loads(line))
    return results


def save_json(file,file_path):
    with open(file_path, 'w') as g:
        g.write(json.dumps(file, indent=4))

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    #if len(s)>1 and s[-1] == ".":
    #    s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores






from transformers import StoppingCriteriaList, StoppingCriteria
from torch import LongTensor, FloatTensor, eq, device




class StopOnTokens(StoppingCriteria):
    #https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/13
    def __init__(self, stop_token_ids):
        """
        Initializes the stopping criteria with a list of token ID lists,
        each representing a sequence of tokens that should cause generation to stop.
        
        :param stop_token_ids: List of lists of token IDs
        """
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Iterate over each stop token sequence
        for stop_ids in self.stop_token_ids:
            # Ensure input_ids has enough tokens to compare with stop_ids
            if input_ids.shape[1] >= stop_ids.shape[1]:
                # Compare the last tokens of input_ids with the entire stop_ids sequence
                if torch.equal(input_ids[0, -stop_ids.shape[1]:], stop_ids[0]):
                    return True
        return False