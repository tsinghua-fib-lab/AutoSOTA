import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Advanced NN Training Methods Benchmark")
    parser.add_argument('--method', type=str, default='backprop', 
                        choices=['dmol', 'dmol_global', 'backprop', 'noprop', 'ff', 'fa', 'hsic', 'es', 'dgl'],
                        help='Training method to use.')
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'], 
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Root directory for datasets')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for main model parts')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_modules', type=int, default=4, help='Number of modules/steps (depth)')
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log_dir', type=str, default='SGD_results', help='Directory to save JSON results')
    
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for SGD (L2 regularization)')
    
    # DMoL specific
    parser.add_argument('--alpha', type=float, default=0.5, help='DMoL loss balance (target vs consistency).')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0.0=disable).')

    # General training
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--use_cosine_lr', action='store_true', help='If set, use cosine annealing LR schedule.')
    
    # DMoL_global specific
    parser.add_argument('--noise_std', type=float, default=0.01, help='Noise std for dmol_global gradient estimation')
    
    # FF specific
    parser.add_argument('--ff_downstream_lr', type=float, default=1e-2, help='Learning rate for the FF downstream classifier')
    parser.add_argument('--ff_threshold', type=float, default=2.0, help='FF goodness threshold for squared activity')

    # HSIC specific
    parser.add_argument('--hsic_lambda', type=float, default=0.1, help='Weight for the HSIC independence loss')
    parser.add_argument('--hsic_decoder_lr', type=float, default=1e-3, help='Learning rate for HSIC decoders')
    parser.add_argument('--hsic_kernel', type=str, default='rbf', choices=['rbf', 'linear'], help='Kernel for HSIC calculation')
    
    # ES specific
    parser.add_argument('--es_sigma', type=float, default=0.1, help='Noise sigma for Evolution Strategy')
    parser.add_argument('--es_population', type=int, default=50, help='Population size for Evolution Strategy')

    parser.add_argument('--run_robustness_test', action='store_true', help='If set, run robustness benchmark after training.')
    parser.add_argument('--fgsm_epsilon', type=float, default=0.1, help='Epsilon for FGSM attack.')
    parser.add_argument('--use_nondiff_module', action='store_true', help='If set, insert a non-differentiable module.')
    parser.add_argument('--k_shot', type=int, default=0, help='Enable K-shot learning with K samples per class (0=disable).')
    parser.add_argument('--label_noise_p', type=float, default=0.0, help='Symmetric label noise probability (0.0=disable).')

    try:
        args = parser.parse_args()
    except (SystemExit, TypeError):
        args = parser.parse_args(args=[])
        
    return args
