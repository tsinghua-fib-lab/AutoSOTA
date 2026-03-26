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

# run local filtering from layer embedding to layer 12 in bert
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

def collect_results(input_dict, results, key, intact = False):
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

def weights_mask(cnf_obj, n, emb, ratio = 0.15):
    logging.info(f'\n phase 2 training towards gender: \n model trainable beyond layer {n+1} with embedding {emb}')
    changes = cnf_obj.filter_local(n_layer=n, emb=emb)
    cnf_obj.apply_local_mask(changes=changes, ratio=ratio)
    cnf_obj.eval_masked_model()

def save(obj, path, filename):
    outfile = os.path.join(path, filename)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outfile, 'wb') as savef:
        pickle.dump(obj, savef)

def main():
    parser = argparse.ArgumentParser(description='Run Local Filtering')
    parser.add_argument('-data', '--data_name', type=str)
    parser.add_argument('-model', '--model_name', type=str)
    parser.add_argument('-ratio', '--mask_ratio', default=0.15, type=float)
    parser.add_argument('-rep', '--rep_runs', default=2, type=int)
    parser.add_argument('-out', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--n_test', default=150, type=int)
    parser.add_argument('-c', '--config', nargs='+', type = float, required=True)
    args = parser.parse_args()
    
    pz0=args.config[0]
    pz1=args.config[1]
    alpha_test=args.config[2]
    maskratio = args.mask_ratio
    log_label='atrain-{:.2f}'.format(pz1/pz0)
    logging.basicConfig(filename=f'output/logs/local/{args.data_name}_{args.model_name}_{log_label}_mask{maskratio}.log',
                    filemode='w',  # Append mode
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setseed()
    results = {}
    # run repeats
    for _ in trange(args.rep_runs, desc='repeats runs...'):
        logging.info(f'\nRunning experiments for configuration ALPHA_TRAIN: {pz1/pz0} , ALPHA_TEST: {alpha_test},  repeats:{_+1}\n')
        cnf_obj = ConfoundM(model_name=args.model_name,
                                data_name=args.data_name,
                                pz0=pz0, pz1=pz1, alpha=alpha_test,
                                n_test=args.n_test,data_seed=data_seed+_, device=device)
        logging.info('duplicate rates: train:{:.2f}, test:{:.2f}'.format(cnf_obj.duplicates['train'], cnf_obj.duplicates['test']))
        # train primary model, do not need to track weights change in phase 1
        cnf_obj.train_primary(num_labels=2)
        collect_results(cnf_obj.evaluation, results, key='', intact = True)

        for n in trange(11, -2, -1):
            # from only classifier to training all layers including embedding
            weights_mask(cnf_obj=cnf_obj, emb=False, n=n, ratio=maskratio)
            if n==11:
                # the first masking is on classifier head only
                collect_results(cnf_obj.evaluation, results, key=f'classifier.', intact = False)
            else:
                collect_results(cnf_obj.evaluation, results, key=f'layer{n+2}.', intact = False)
            # run it again with emb=True
            if n==-1:
                weights_mask(cnf_obj=cnf_obj, emb=True, n=n, ratio=maskratio)
                collect_results(cnf_obj.evaluation, results, key='emb.', intact = False)


    save(results, args.output_dir, f'local_{log_label}.pkl')



if __name__ == '__main__':
    main()