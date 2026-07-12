import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_config
from datasets import get_dataloaders
from utils import get_gpu_memory_usage, reset_gpu_memory_stats, evaluate, evaluate_robustness
from models import (
    Backprop_Network, DMoL_Network, DMoL_NonDiff_Network, NoProp_Network,
    FF_Network, FA_Network, HSIC_Network, DGL_Network
)
from trainers import (
    train_backprop, train_dmol, train_dgl, train_noprop,
    train_ff, train_fa, train_hsic, train_es
)

def main():
    args = get_config()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    DEVICE = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("-" * 60)
    print(f"Starting Benchmark:")
    for k, v in vars(args).items():
        print(f"  - {k}: {v}")
    print(f"  - Device: {DEVICE}")
    print("-" * 60)

    train_loader, test_loader, num_classes, in_channels, img_size = get_dataloaders(args)
    model, optimizers, criterion = None, None, nn.CrossEntropyLoss()

    if args.use_nondiff_module:
        print("INFO: Using network with a non-differentiable module.")
        if args.method in ['backprop', 'fa']:
             raise ValueError(f"Method '{args.method}' cannot train with non-differentiable modules.")
        model = DMoL_NonDiff_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
    else:
        if args.method in ['backprop', 'dmol', 'es']:
             model = Backprop_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
             if args.method == 'dmol':
                 model = DMoL_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        elif args.method == 'noprop':
             model = NoProp_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        elif args.method == 'ff':
            model = FF_Network(args.num_modules, num_classes, in_channels, img_size, 
                               threshold=args.ff_threshold, downstream_lr=args.ff_downstream_lr).to(DEVICE)
        elif args.method == 'fa':
            model = FA_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        elif args.method == 'hsic':
            model = HSIC_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        elif args.method == 'dgl':
            model = DGL_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
            
    sgd_kwargs = {
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'nesterov': True,
    }
    
    if args.method == 'backprop':
        optimizers = optim.SGD(model.parameters(), **sgd_kwargs)
    elif args.method == 'es': 
        optimizers = optim.SGD(model.parameters(), **sgd_kwargs)
    elif args.method in ['dmol', 'dmol_global']:
        optimizers = [optim.SGD(model.feature_extractor.parameters(), **sgd_kwargs)] + \
                     [optim.SGD(m.parameters(), **sgd_kwargs) for m in model.processing_modules]
    elif args.method == 'noprop':
        optimizers = [optim.SGD(model.cnn.parameters(), **sgd_kwargs)] + \
                     [optim.SGD(m.parameters(), **sgd_kwargs) for m in model.mlps]
    elif args.method == 'ff':
        for layer in model.ff_layers:
            layer.optimizer.param_groups[0]['lr'] = args.lr
    elif args.method == 'fa':
        optimizers = optim.Adam(model.parameters(), lr=args.lr)
    elif args.method == 'hsic':
        optimizers = None 
    elif args.method == 'dgl':
        optimizers = []
        optimizers.append(optim.SGD(model.parameters(), **sgd_kwargs))
        for i in range(1, args.num_modules):
            optimizers.append(optim.Adam(model.processing_modules[i].parameters(), lr=args.lr))

    print(f"\nTraining model with method: {args.method.upper()} on {args.dataset.upper()}...")

    # Set up cosine annealing LR scheduler if requested
    schedulers = None
    if args.use_cosine_lr:
        if args.method in ['dmol', 'dmol_global']:
            schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) for opt in optimizers]
        elif optimizers is not None and not isinstance(optimizers, list):
            schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=args.epochs)]

    training_history = []

    for epoch in range(args.epochs):
        start_time = time.time()
        reset_gpu_memory_stats(DEVICE)

        train_loss = 0
        if args.method == 'backprop':
            train_loss = train_backprop(model, train_loader, optimizers, criterion, DEVICE)
        elif args.method == 'dmol':
            train_loss = train_dmol(model, train_loader, optimizers, num_classes, DEVICE, args.alpha, args.label_smoothing)
        elif args.method == 'dmol_global':
            pass # Implement if needed
        elif args.method == 'noprop':
            train_loss = train_noprop(model, train_loader, optimizers, DEVICE)
        elif args.method == 'es': 
            train_loss = train_es(model, train_loader, optimizers, criterion, DEVICE, args.es_sigma, args.es_population, args.lr)
        elif args.method == 'ff':
            train_loss = train_ff(model, train_loader, DEVICE)
        elif args.method == 'fa':
            train_loss = train_fa(model, train_loader, optimizers, criterion, DEVICE)
        elif args.method == 'hsic':
            train_loss = train_hsic(model, train_loader, criterion, DEVICE, args)
        elif args.method == 'dgl':
            train_loss = train_dgl(model, train_loader, optimizers, criterion, num_classes, DEVICE)

        # Step LR schedulers
        if schedulers is not None:
            for sched in schedulers:
                sched.step()

        test_acc = evaluate(model, test_loader, DEVICE, args.method)
        epoch_time = time.time() - start_time
        _, peak_mem = get_gpu_memory_usage(DEVICE)
        
        epoch_results = {
            'epoch': epoch + 1, 'train_loss': train_loss, 'test_acc': test_acc, 
            'time_s': round(epoch_time, 2), 'peak_mem_mb': round(peak_mem, 2)
        }
        training_history.append(epoch_results)
        
        print(f"Epoch [{epoch+1:>{len(str(args.epochs))}}/{args.epochs}] | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:6.2f}% | Time: {epoch_time:5.1f}s | Peak Mem: {peak_mem:6.1f}MB")
    
    print("\nTraining finished.")
    final_results = {'config': vars(args), 'history': training_history}

    if args.run_robustness_test:
        robustness_results = evaluate_robustness(model, test_loader, DEVICE, args.method, criterion, args.fgsm_epsilon)
        final_results['robustness'] = robustness_results
        
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename_parts = [args.method, args.dataset, f"e{args.epochs}", f"d{args.num_modules}"]
    if args.method == 'dmol': filename_parts.append(f"a{args.alpha}")
    if args.k_shot > 0: filename_parts.append(f"k{args.k_shot}")
    if args.label_noise_p > 0: filename_parts.append(f"n{args.label_noise_p}")
    if args.use_nondiff_module: filename_parts.insert(0, "nondiff")
    filename_parts.append(timestamp)
    filename = "_".join(map(str, filename_parts)) + ".json"
    
    filepath = os.path.join(args.log_dir, filename)
    try:
        with open(filepath, 'w') as f: json.dump(final_results, f, indent=4)
        print(f"\nResults successfully saved to {filepath}")
    except Exception as e:
        print(f"\nError saving results to {filepath}: {e}")

    print("\nBenchmark finished.")

if __name__ == '__main__':
    main()
