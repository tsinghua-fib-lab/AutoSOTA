import os
import random
import argparse
import numpy as np
import torch
from datasets import get_all_dataloaders
from utils import *
from sampler import BatchSampler, OnlineSampler
from tqdm import tqdm
from solvers import TransCLIP_solver, StatA_solver, Dirichlet_solver, ZLaP_solver, ADAPT_transductive_solver, GDA_CLIP_solver, OGA_solver
from solvers.MOON import MOON_solver

from loguru import logger

def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    # General arguments
    parser.add_argument('--dataset', default='dtd', help='dataset name', type=str)
    parser.add_argument('--root_path', default='./data', type=str)
    parser.add_argument('--log_path', default='./logs', type=str)
    parser.add_argument('--method', default='MOON', type=str, choices=['StatA', 'TransCLIP', 'Dirichlet', 'ZLaP', 'TDA', 'tent', 'DMN', 'ADAPT', 'GDA_CLIP', 'OGA', 'MOON', 'MOON_online'], help="test-time adaptation method")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='vit_b16', type=str, choices=['rn50', 'rn101', 'vit_b32', 'vit_b16', 'vit_l14'], help="CLIP architecture")
    parser.add_argument('--cache_dir', type = str, default = None, help='where to store visual and textual features if not None')
    parser.add_argument('--load', action='store_true', default=False, help="Load features from cache_dir")
    parser.add_argument('--device', type=str, default='0', help="device to use")

    # Experimental arguments
    parser.add_argument('--n_tasks', type=int, default=1, help="number of tasks to run")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--online', action='store_true', default=False, help='online setting or not')
    parser.add_argument('--num_class_eff', type=int, default=None, help='number of effective classes to sample from per batch')
    parser.add_argument('--num_class_eff_min', type=int, default=None, help='number of effective classes per batch minimum')
    parser.add_argument('--num_class_eff_max', type=int, default=None, help='number of effective classes per batch maximum')
    parser.add_argument('--gamma', type = float, default = 1.0, help = 'Dirichlet parameter used for sampling in the online setting.')
    parser.add_argument('--holdout_ratio', type=float, default=0.0, help='held-out split ratio for online inductive evaluation protocol')
    parser.add_argument('--bank_capacity', type=int, default=128, help='memory bank capacity used by MOON_online')
    parser.add_argument('--tent_episodic', action='store_true', default=False, help='reset Tent model/optimizer before each new online task')
    
    # Solver hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0, help='anchor weighting hyper-parameter')
    parser.add_argument('--lambda_laplacian', type=float, default=1.0, help='Laplacian weighting hyper-parameter')
    parser.add_argument('--lambda_y_hat', type=float, default=1.0, help='y_hat weighting hyper-parameter')
    parser.add_argument('--soft_beta', action='store_true', default=False, help='use soft beta computation')
    parser.add_argument('--n_neighbors', type=int, default=3, help='number of neighbors for Laplacian smoothing')

    args = parser.parse_args()
    return args

def get_hp(args, method_name):
    if method_name == 'StatA':
        return StatA_solver, {
            'alpha': 1.0,
            'lambda_y_hat':1,
            'lambda_laplacian': 1.0,
            'n_neighbors':3,
            'soft_beta': False
        }
    elif method_name == 'TransCLIP':
        return TransCLIP_solver, {'lambda_y_hat':1, 'lambda_laplacian': 1, 'n_neighbors':3}
    elif method_name == 'Dirichlet':
        return Dirichlet_solver, {'T':30}
    elif method_name == 'ZLaP':
        return ZLaP_solver, {'k':5, 'gamma':5.0, 'alpha':0.3, 'scale_sim':False}
    elif method_name == 'TDA':
        return None, None
    elif method_name == 'tent':
        return None, None
    elif method_name == 'DMN':
        return None, None
    elif method_name == 'ADAPT':
        return ADAPT_transductive_solver, {
            'alpha': 0.9,
            'bank_size': 12
        }
    elif method_name == 'GDA_CLIP':
        return GDA_CLIP_solver, {
            'alpha': 5.0
        }
    elif method_name == 'OGA':
        return None, {
            'shot_capacity': 8,
            'tau': 0.01
        }
    elif method_name == 'MOON':
        return MOON_solver, {
            'alpha': args.alpha,
            'lambda_y_hat':args.lambda_y_hat,
            'lambda_laplacian': args.lambda_laplacian,
            'n_neighbors':args.n_neighbors,
            'soft_beta': args.soft_beta
        }
    elif method_name == 'MOON_online':
        return None, None
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented.")



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    args = get_arguments()
    
    if args.device != 'cpu' and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))
        device_str = f"cuda:{int(args.device)}"
    else:
        device_str = 'cpu'
            
    if args.method in ['TDA', 'DMN', 'tent', 'OGA', 'MOON_online'] and not(args.online):
        raise ValueError(f'Got method {args.method} which is only supported for the online setting, but got args.online = {args.online}.')
    set_random_seed(args.seed) # for reproducibility
    
    if not args.cache_dir:
        args.cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Logging setup
    setting = 'online' if args.online else 'batch'
    log_path = os.path.join(args.log_path, setting, args.backbone, f'bs={args.batch_size}', f'min={args.num_class_eff_min}_max={args.num_class_eff_max}' if not args.online else f'gamma={args.gamma}')
    os.makedirs(log_path, exist_ok=True)
    logger.add(os.path.join(log_path, f'{args.method}_{args.dataset}_{{time}}.log'), rotation="500 MB", level="INFO")

    # CLIP model
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}
    clip_model, preprocess = clip.load(backbones[args.backbone], device=device_str)
    clip_model.eval()

    # Prepare dataset
    _, _, test_loader, dataset = get_all_dataloaders(args, preprocess)

    # Load features
    test_features, test_labels, clip_prototypes = get_all_features(args, test_loader, dataset, clip_model)

    # Optional: unload CLIP model from VRAM
    if args.method not in ['tent', 'DMN'] and device_str != 'cpu':
        clip_model = clip_model.to('cpu')
    
    acc_tot = 0
    acc_zs_tot = 0
    
    logger.info("\n============================")
    logger.info("      Final Results         ")
    logger.info("============================")
    logger.info(f"Dataset:         {args.dataset}")
    logger.info(f"Method:          {args.method}")
    logger.info(f"Backbone:        {args.backbone}")
    logger.info(f"Number of Tasks: {args.n_tasks}")
    logger.info(f"Batch Size:      {args.batch_size}")
    logger.info(f"Online Setting:  {'Yes' if args.online else 'No'}")
    logger.info(f"Soft Beta:       {'Yes' if args.soft_beta else 'No'}")
      
    if args.online:
        logger.info(f"Dirichlet Gamma: {args.gamma:.2f}")
        if args.holdout_ratio > 0:
            logger.info(f"Held-out Ratio:   {args.holdout_ratio:.2f}")
    else:
        logger.info(f"Effective Classes Min: {args.num_class_eff_min or 'None'}")
        logger.info(f"Effective Classes Max: {args.num_class_eff_max or 'None'}")
    
    
    ##############################
    # Batch Test-Time Adaptation #
    ##############################
    
    solver, method_args = get_hp(args, args.method)
    
    if not args.online:
        sampler = BatchSampler(test_features, test_labels, args.batch_size, args.num_class_eff, args.num_class_eff_min, args.num_class_eff_max)
    
        for i in tqdm(range(args.n_tasks)):
    
            indices = sampler.generate_indices()
            if indices == None:
                break
            
            preds_zs, preds = solver(test_features[indices,:], test_labels[indices], clip_prototypes,
                                              **method_args)
            
            acc_zs = cls_acc(preds_zs, test_labels[indices])
            acc = cls_acc(preds, test_labels[indices])
            acc_zs_tot += acc_zs
            acc_tot += acc
        
        
        acc_zs_tot /= args.n_tasks
        acc_tot /= args.n_tasks
        
    ###############################
    # Online Test-Time Adaptation #
    ###############################

    if args.online:
        if args.method == 'TDA':
            K = torch.max(test_labels)+1
            d = test_features.shape[1]
            from solvers import TDA_solver
        if args.method == 'tent':
            K = torch.max(test_labels)+1
            from solvers import Tent_solver, get_cfg
        if args.method == 'ADAPT':
            K = torch.max(test_labels) + 1
            d = test_features.shape[1]
            from solvers import ADAPT_online_solver
        if args.method == 'DMN':
            K = torch.max(test_labels)+1
            d = test_features.shape[1]
            from solvers.DMN import DMNClipWrapper, get_cfg_DMN, DMNDualMem
            dmn_args, beta = get_cfg_DMN()
        if args.method == 'MOON_online':
            K = torch.max(test_labels) + 1
            d = test_features.shape[1]
            from solvers import MOON_online_solver
            
        for i in tqdm(range(args.n_tasks)):
            if args.method == 'TDA':
                solver = TDA_solver(K, d) # reinstantiate solver with empty cache
            if args.method == 'tent':
                solver = Tent_solver(get_cfg('tent', episodic=args.tent_episodic),  clip_model.visual, K) # reinstantiate from unchanged model for each task
            if args.method == 'ADAPT':
                solver = ADAPT_online_solver(K, d, alpha=method_args['alpha'], bank_size=method_args['bank_size'],) # reinstantiate solver with empty banks
            if args.method == 'OGA':
                solver = OGA_solver(**method_args)
            if args.method == 'DMN':
                DMN_clip = DMNClipWrapper(clip_model, preprocess, 'cuda', 
                            dataset.classnames, args.batch_size, 
                            arch = backbones[args.backbone],).cuda()
                DMN_clip.reset_classnames(dataset)
                DMN_clip.get_text_features()
                dmn = DMNDualMem(args = dmn_args, feat_dim = test_features.shape[-1], class_num = K) # reinstantiate with empty cache
                dmn = dmn.cuda()
                DMN_clip.eval()
                dmn.eval()
            if args.method == 'MOON_online':
                solver = MOON_online_solver(
                    K=K,
                    d=d,
                    alpha=args.alpha,
                    soft_beta=args.soft_beta,
                    lambda_y_hat=args.lambda_y_hat,
                    lambda_laplacian=args.lambda_laplacian,
                    n_neighbors=args.n_neighbors,
                    bank_capacity=args.bank_capacity,
                )

            if args.method == 'MOON_online' and args.holdout_ratio > 0:
                total_num = test_features.shape[0]
                heldout_num = max(1, int(total_num * args.holdout_ratio))
                rng = np.random.default_rng(args.seed + i)
                perm = rng.permutation(total_num)

                heldout_idx = perm[:heldout_num]
                adapt_idx = perm[heldout_num:]

                adapt_features = test_features[adapt_idx, :]
                adapt_labels = test_labels[adapt_idx]

                if adapt_features.shape[0] > 0:
                    num_batch = max(1, adapt_features.shape[0] // args.batch_size)
                    num_slots = min(num_batch, len(torch.unique(adapt_labels)))
                    adapt_sampler = OnlineSampler(adapt_features, adapt_labels, args.gamma, num_slots, args.batch_size)

                    adapt_indices = adapt_sampler.generate_indices()
                    while adapt_indices is not None:
                        solver(adapt_features[adapt_indices, :], adapt_labels[adapt_indices], clip_prototypes)
                        adapt_indices = adapt_sampler.generate_indices()

                preds_zs, preds = solver.predict_without_update(test_features[heldout_idx, :], clip_prototypes, batch_size=args.batch_size)
                acc_zs = cls_acc(preds_zs, test_labels[heldout_idx])
                acc = cls_acc(preds, test_labels[heldout_idx])

                acc_tot += acc
                acc_zs_tot += acc_zs
                continue

            num_batch = test_features.shape[0]//args.batch_size
            num_slots = min(num_batch, len(torch.unique(test_labels)))
            sampler = OnlineSampler(test_features, test_labels, args.gamma, num_slots, args.batch_size)
            
            indices = sampler.generate_indices()
            all_accs = []
            all_accs_zs = []
            
            while indices is not None:
                if args.method == 'tent':
                    batch_imgs = torch.stack([test_loader.dataset[u][0] for u in indices], dim = 0).cuda()
                    preds = solver(batch_imgs, 
                                            clip_prototypes = clip_prototypes.squeeze().T,
                                            )
                    preds = preds.cpu()
                    preds_zs = (test_features[indices,:].cuda() @ clip_prototypes.squeeze()).cpu()
                elif args.method == 'DMN':
                    all_img_global_pred = torch.zeros((len(indices), K), dtype = torch.float16,  device = 'cuda')
                    with torch.autocast("cuda"), torch.no_grad():
                        image_features_global = test_features[indices,...].cuda()
                        text_logits = DMN_clip.logit_scale.exp()*image_features_global @ clip_prototypes.squeeze()
                        text_probs = text_logits.softmax(1)
                        for ju,u in enumerate(indices):
                            DMN_clip.image_features_global = image_features_global[ju:ju+1,...]
                            dmn.init_pred = text_probs[ju:ju+1,:]
                            dmn.update_memory_bank(DMN_clip)
                        # Predict on the batch with updated memory.
                        with torch.autocast("cuda"), torch.no_grad():
                            for ju,u in enumerate(indices):
                                # get_image_pred is not designed to handle batches.
                                DMN_clip.image_features_global = image_features_global[ju:ju+1,...]
                                all_img_global_pred[ju,...] = dmn.get_image_pred(DMN_clip, return_logit = True)
                        preds = (text_logits + beta * all_img_global_pred).squeeze().cpu()  
                        preds_zs = (test_features[indices,:].cuda() @ clip_prototypes.squeeze()).cpu()
                elif args.method in ['ADAPT', 'TDA', 'OGA', 'MOON_online']:
                    preds_zs, preds = solver(test_features[indices, :], test_labels[indices], clip_prototypes)
                else:
                    preds_zs, preds = solver(test_features[indices,:], test_labels[indices], clip_prototypes, **method_args)
                acc_zs = cls_acc(preds_zs, test_labels[indices])
                acc = cls_acc(preds, test_labels[indices])
                all_accs.append(acc)
                all_accs_zs.append(acc_zs)
                indices = sampler.generate_indices()
                
            acc_tot += sum(all_accs)/len(all_accs)
            acc_zs_tot += sum(all_accs_zs)/len(all_accs_zs)

        acc_tot /= args.n_tasks
        acc_zs_tot /= args.n_tasks
      
    logger.info("----------------------------")
    logger.info(f"ZERO-shot Accuracy: {acc_zs_tot:.4f}")
    logger.info(f"FINAL Accuracy:     {acc_tot:.4f}")
    logger.info("============================\n")



if __name__ == '__main__':
    main()
