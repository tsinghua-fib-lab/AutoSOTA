import os
import random
import argparse
import numpy as np
import torch
import open_clip
from datasets import get_all_dataloaders
from utils_openclip import *
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
    parser.add_argument('--method', default='MOON', type=str,
                        choices=['StatA', 'TransCLIP', 'Dirichlet', 'ZLaP',
                                 'ADAPT', 'GDA_CLIP', 'OGA', 'MOON'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cache_dir', type=str, default=None, help='where to store features if not None')
    parser.add_argument('--load', action='store_true', default=False, help="Load features from cache_dir")
    parser.add_argument('--device', type=str, default='0', help="device to use")

    # OpenCLIP arguments
    parser.add_argument('--openclip_model', type=str, default='ViT-B-16')
    parser.add_argument('--openclip_pretrained', type=str, default='datacomp_xl_s13b_b90k')

    # Experimental arguments
    parser.add_argument('--n_tasks', type=int, default=1, help="number of tasks to run")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--online', action='store_true', default=False, help='online setting or not')
    parser.add_argument('--num_class_eff', type=int, default=None)
    parser.add_argument('--num_class_eff_min', type=int, default=None)
    parser.add_argument('--num_class_eff_max', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=1.0)

    # Solver hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lambda_laplacian', type=float, default=1.0)
    parser.add_argument('--lambda_y_hat', type=float, default=1.0)
    parser.add_argument('--soft_beta', action='store_true', default=False)
    parser.add_argument('--n_neighbors', type=int, default=3)

    return parser.parse_args()


def get_hp(args, method_name):
    if method_name == 'StatA':
        return StatA_solver, {
            'alpha': 1.0,
            'lambda_y_hat': 1,
            'lambda_laplacian': 1.0,
            'n_neighbors': 3,
            'soft_beta': False
        }
    elif method_name == 'TransCLIP':
        return TransCLIP_solver, {'lambda_y_hat': 1, 'lambda_laplacian': 1, 'n_neighbors': 3}
    elif method_name == 'Dirichlet':
        return Dirichlet_solver, {'T': 30}
    elif method_name == 'ZLaP':
        return ZLaP_solver, {'k': 5, 'gamma': 5.0, 'alpha': 0.3, 'scale_sim': False}
    elif method_name == 'ADAPT':
        return ADAPT_transductive_solver, {'alpha': 0.9, 'bank_size': 12}
    elif method_name == 'GDA_CLIP':
        return GDA_CLIP_solver, {'alpha': 5.0}
    elif method_name == 'OGA':
        return None, {'shot_capacity': 8, 'tau': 0.01}
    elif method_name == 'MOON':
        return MOON_solver, {
            'alpha': args.alpha,
            'lambda_y_hat': args.lambda_y_hat,
            'lambda_laplacian': args.lambda_laplacian,
            'n_neighbors': args.n_neighbors,
            'soft_beta': args.soft_beta
        }
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

    if args.method in ['OGA'] and not args.online:
        raise ValueError(f'Got method {args.method} which is only supported for the online setting.')

    set_random_seed(args.seed)

    if not args.cache_dir:
        args.cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(args.cache_dir, exist_ok=True)

    setting = 'online' if args.online else 'batch'
    log_path = os.path.join(args.log_path, setting, f'openclip_{args.openclip_model}_{args.openclip_pretrained}',
                            f'bs={args.batch_size}',
                            f'min={args.num_class_eff_min}_max={args.num_class_eff_max}' if not args.online else f'gamma={args.gamma}')
    os.makedirs(log_path, exist_ok=True)
    logger.add(os.path.join(log_path, f'{args.method}_{args.dataset}_{{time}}.log'),
               rotation="500 MB", level="INFO")

    # OpenCLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.openclip_model, pretrained=args.openclip_pretrained)
    model.eval()
    model = model.to(device_str)
    tokenizer = open_clip.get_tokenizer(args.openclip_model)

    # Prepare dataset
    _, _, test_loader, dataset = get_all_dataloaders(args, preprocess)

    # Load features
    test_features, test_labels, clip_prototypes = get_all_features_openclip(
        args, test_loader, dataset, model, tokenizer)

    if device_str != 'cpu':
        model = model.to('cpu')

    acc_tot = 0
    acc_zs_tot = 0

    solver, method_args = get_hp(args, args.method)

    if not args.online:
        sampler = BatchSampler(test_features, test_labels, args.batch_size,
                               args.num_class_eff, args.num_class_eff_min, args.num_class_eff_max)

        for i in tqdm(range(args.n_tasks)):
            indices = sampler.generate_indices()
            if indices is None:
                break

            preds_zs, preds = solver(test_features[indices, :], test_labels[indices],
                                     clip_prototypes, **method_args)

            acc_zs = cls_acc(preds_zs, test_labels[indices])
            acc = cls_acc(preds, test_labels[indices])
            acc_zs_tot += acc_zs
            acc_tot += acc

        acc_zs_tot /= args.n_tasks
        acc_tot /= args.n_tasks

    if args.online:
        if args.method == 'ADAPT':
            K = torch.max(test_labels) + 1
            d = test_features.shape[1]
            from solvers import ADAPT_online_solver

        for i in tqdm(range(args.n_tasks)):
            if args.method == 'ADAPT':
                solver = ADAPT_online_solver(K, d, alpha=method_args['alpha'], bank_size=method_args['bank_size'])
            if args.method == 'OGA':
                solver = OGA_solver(**method_args)
            num_batch = test_features.shape[0] // args.batch_size
            num_slots = min(num_batch, len(torch.unique(test_labels)))
            sampler = OnlineSampler(test_features, test_labels, args.gamma, num_slots, args.batch_size)

            indices = sampler.generate_indices()
            all_accs = []
            all_accs_zs = []

            while indices is not None:
                if args.method in ['ADAPT', 'OGA']:
                    preds_zs, preds = solver(test_features[indices, :], test_labels[indices],
                                             clip_prototypes)
                else:
                    preds_zs, preds = solver(test_features[indices, :], test_labels[indices],
                                             clip_prototypes, **method_args)
                acc_zs = cls_acc(preds_zs, test_labels[indices])
                acc = cls_acc(preds, test_labels[indices])
                all_accs.append(acc)
                all_accs_zs.append(acc_zs)
                indices = sampler.generate_indices()

            acc_tot += sum(all_accs) / len(all_accs)
            acc_zs_tot += sum(all_accs_zs) / len(all_accs_zs)

        acc_tot /= args.n_tasks
        acc_zs_tot /= args.n_tasks

    logger.info("----------------------------")
    logger.info(f"ZERO-shot Accuracy: {acc_zs_tot:.4f}")
    logger.info(f"FINAL Accuracy:     {acc_tot:.4f}")
    logger.info("============================\n")


if __name__ == '__main__':
    main()
