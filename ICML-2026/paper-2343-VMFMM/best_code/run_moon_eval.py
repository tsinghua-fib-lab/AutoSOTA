"""
Standalone MOON evaluation script using pre-extracted ImageNet features.
Reproduces: ImageNet, Bs=64, Very Low Keff (1-4), MOON, CLIP ViT-B/16, n_runs=1000.
"""
import os, sys, random, argparse
sys.path.insert(0, '/repo')
import numpy as np
import torch
from tqdm import tqdm
from sampler import BatchSampler
from solvers.MOON import MOON_solver
from utils import cls_acc

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_tasks', default=1000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_class_eff_min', default=1, type=int)
    parser.add_argument('--num_class_eff_max', default=4, type=int)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--cache_dir', default='/repo/caches/imagenet', type=str)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--lambda_laplacian', default=1.0, type=float)
    parser.add_argument('--lambda_y_hat', default=1.0, type=float)
    parser.add_argument('--n_neighbors', default=3, type=int)
    parser.add_argument('--soft_beta', action='store_true', default=False)
    parser.add_argument('--max_iter', default=10, type=int)
    parser.add_argument('--e_step_temperature', default=50.0, type=float)
    parser.add_argument('--zs_temperature', default=100.0, type=float)
    args = parser.parse_args()

    set_random_seed(args.seed)

    if args.device != 'cpu' and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))

    print(f"Loading features from {args.cache_dir}")
    test_features = torch.load(os.path.join(args.cache_dir, 'test_f.pt'))
    test_labels = torch.load(os.path.join(args.cache_dir, 'test_l.pt'))
    clip_prototypes = torch.load(os.path.join(args.cache_dir, 'clip_prototypes.pt'))

    print(f"Features: {test_features.shape}")
    print(f"Labels: {test_labels.shape}")
    print(f"Prototypes: {clip_prototypes.shape}")
    print(f"Unique classes: {len(torch.unique(test_labels))}")

    # Setup
    print(f"Setting: batch_size={args.batch_size}, Keff={args.num_class_eff_min}-{args.num_class_eff_max}, n_tasks={args.n_tasks}")
    sampler = BatchSampler(test_features, test_labels, args.batch_size,
                          num_class_eff_min=args.num_class_eff_min,
                          num_class_eff_max=args.num_class_eff_max)

    method_args = {
        'alpha': args.alpha,
        'lambda_y_hat': args.lambda_y_hat,
        'lambda_laplacian': args.lambda_laplacian,
        'n_neighbors': args.n_neighbors,
        'soft_beta': args.soft_beta,
        'max_iter': args.max_iter,
        'e_step_temperature': args.e_step_temperature,
        'zs_temperature': args.zs_temperature,
    }

    acc_tot = 0
    acc_zs_tot = 0
    valid_tasks = 0

    for i in tqdm(range(args.n_tasks)):
        indices = sampler.generate_indices()
        if indices is None:
            break

        preds_zs, preds = MOON_solver(test_features[indices, :], test_labels[indices],
                                       clip_prototypes, **method_args)

        acc_zs = cls_acc(preds_zs, test_labels[indices])
        acc = cls_acc(preds, test_labels[indices])
        acc_zs_tot += acc_zs
        acc_tot += acc
        valid_tasks += 1

    acc_zs_tot /= valid_tasks
    acc_tot /= valid_tasks

    print("=" * 50)
    print(f"RESULTS (n_tasks={valid_tasks})")
    print(f"  Zero-shot Accuracy: {acc_zs_tot:.2f}%")
    print(f"  MOON Accuracy:       {acc_tot:.2f}%")
    print(f"  Improvement:         {acc_tot - acc_zs_tot:+.2f}%")
    print("=" * 50)

    return acc_tot, acc_zs_tot

if __name__ == '__main__':
    main()
