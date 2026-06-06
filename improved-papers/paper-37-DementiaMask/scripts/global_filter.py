import sys
sys.path.insert(0, '.')

from src.model import ConfoundM
from transformers import set_seed
import random
import os
import pickle
from tqdm import trange
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
import argparse
import logging
from transformers import AutoModelForSequenceClassification

# setting in (pz0, pz1, alpha_test) format
# always evaluate the reciprocal shift
# SETTINGS = [
#     (0.167, 0.833, 0.2),
#     (0.2, 0.8, 0.25),
#     (0.33, 0.66, 0.5),
#     (0.25, 0.75, 0.33),
#     (0.5, 0.5, 1.0),
#     (0.66, 0.33, 2.),
#     (0.75, 0.25, 3.),
#     (0.8, 0.2, 4.),
#     (0.833, 0.167, 5.)   
# ]
# SETTINGS = [(0.5, 0.5, 1.0),(0.8, 0.2, 4.)]
SEED = 42
data_seed = 24

def setseed():
    np.random.seed(SEED)
    set_seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

def collect_results(input_dict, results, key = '', intact = False):
    """ store evaluation """
    for name, evals in input_dict.items():
        if intact:
            if name not in results:
                results[name] = defaultdict(list)
            for eval, value in evals.items():
                results[name][eval].append(value)
        
        elif name not in results and not intact:
            tag = key + name
            if tag not in results:
                results[tag] = defaultdict(list)
            for eval, value in evals.items():
                results[tag][eval].append(value)

def weights_mask(cnf_obj, changes_primary, changes_confounder, type, ratio, alpha = 0):
    cnf_obj.apply_global_mask(changes1=changes_primary, changes2=changes_confounder, 
                              type=type, ratio=ratio, alpha=alpha)
    cnf_obj.eval_masked_model()

def save(obj, path, filename):
    outfile = os.path.join(path, filename)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outfile, 'wb') as savef:
        pickle.dump(obj, savef)

def load_intact_model(path, global_changes, log_label, cnf_obj, iter):
    dmodel_path = os.path.join(path, f'label_{log_label}.model')
    gmodel_path = os.path.join(path, f'gender_{log_label}.model')
    cnf_obj.model = AutoModelForSequenceClassification.from_pretrained(dmodel_path).to(cnf_obj.device)
    cnf_obj.confounder_model = AutoModelForSequenceClassification.from_pretrained(gmodel_path).to(cnf_obj.device)

    changes_d = global_changes[f'repeat{iter+1}']['dementia_change']
    changes_d = {k:v.to(cnf_obj.device) for k,v in changes_d.items()}
    changes_g = global_changes[f'repeat{iter+1}']['gender_change']
    changes_g = {k:v.to(cnf_obj.device) for k,v in changes_g.items()}

    cnf_obj.num_model_params = np.sum([p.numel() for p in cnf_obj.confounder_model.parameters() if p.requires_grad])
    return changes_d, changes_d

def main():
    parser = argparse.ArgumentParser(description='Run Global Filtering')
    parser.add_argument('-data', '--data_name', type=str)
    parser.add_argument('-model', '--model_name', type=str)
    parser.add_argument('-rep', '--rep_runs', default=2, type=int)
    parser.add_argument('-out', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--n_test', default=150, type=int)
    parser.add_argument('-c', '--config', nargs='+', type = float, required=True)
    parser.add_argument('-l', '--level', default=0, type=int)
    args = parser.parse_args()
    savedir = f'/tmp/checkpoints/global-{args.data_name}'
    level = args.level # masking degree
    
    pz0=args.config[0]
    pz1=args.config[1]
    alpha_test=args.config[2]
    log_label='atrain-{:.2f}'.format(pz1/pz0)
    model_name = os.path.basename(args.model_name)
    logging.basicConfig(filename=f'output/logs/global/{args.data_name}_{model_name}_{log_label}.log',
                    filemode='w',  # Append mode
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setseed()
    results = {}
    global_changes = defaultdict(dict)
    reference = 'direct'



    # run repeats
    for _ in trange(args.rep_runs, desc='repeats runs...'):
        logging.info(f'\nRunning experiments for configuration ALPHA_TRAIN: {pz1/pz0} , ALPHA_TEST: {alpha_test},  repeats:{_+1}\n')
        cnf_obj = ConfoundM(model_name=args.model_name,
                            data_name=args.data_name,
                            pz0=pz0, pz1=pz1, alpha=alpha_test,
                            n_test=args.n_test,data_seed=data_seed+_, device=device)
        
        # else:
        logging.info('duplicate rates: train:{:.2f}, test:{:.2f}'.format(cnf_obj.duplicates['train'], cnf_obj.duplicates['test']))
        # train primary model, not track classifier
        changes_d = cnf_obj.train_primary(num_labels=2, filter = 'global', savedir=savedir)
        changes_d_dump = {k:v.cpu() for k,v in changes_d.items()}
        collect_results(cnf_obj.evaluation, results, key='', intact = True)
        
        logging.info(f'\n {reference} training towards gender')
        changes_g = cnf_obj.filter_global(num_labels=2, reference=reference,savedir=savedir)
        changes_g_dump = {k:v.cpu() for k,v in changes_g.items()}
        
        # save the change matrices of each repeat for further analysis
        global_changes[f'repeat{_+1}'] = {'dementia_change': changes_d_dump,
                                        'gender_change': changes_g_dump}
    
    
        for type in ['intersection', 'compliment', 'all']:
            # only mask up to 60%
            for mask_ratio in np.arange(0.01, 0.6, 0.01):
                logging.info(f'Masking {type} of top {mask_ratio:.2%} changed weights from both models...')
                weights_mask(cnf_obj, changes_primary=changes_d, changes_confounder=changes_g, type=type, ratio=mask_ratio, alpha=level)
                collect_results(cnf_obj.evaluation, results, key=f'{type}.top{mask_ratio:.2f}-', intact = False)

    save(global_changes, args.output_dir, f'change-{args.data_name}/global_{log_label}.pkl')
    save(results, args.output_dir, f'global-{args.data_name}/global_{log_label}.pkl')


if __name__ == '__main__':
    main()