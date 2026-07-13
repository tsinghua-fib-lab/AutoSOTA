
#!/usr/bin/env python3
"""Fast CLD head retraining using cached features from baseline model."""
import argparse, pickle, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP
from jaxcld.optimizers.admm import admm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_model', type=str, required=True, help='Path to baseline .pkl model with cached features')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save new model')
    parser.add_argument('--neuron', type=int, default=32)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--admm_iters', type=int, default=6)
    parser.add_argument('--pcg_iters', type=int, default=32)
    parser.add_argument('--gamma_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check_opt', action='store_true', default=False)
    args = parser.parse_args()

    print(f'Loading baseline model from {args.baseline_model}')
    with open(args.baseline_model, 'rb') as f:
        baseline = pickle.load(f)
    
    X = baseline.X
    y = baseline.y
    Xtst = baseline.Xtst
    ytst = baseline.ytst
    n_classes = baseline.n_classes
    print(f'Features: X={X.shape}, y={y.shape}, Xtst={Xtst.shape}, ytst={ytst.shape}')
    print(f'n_classes={n_classes}')
    
    print(f'Creating model: neuron={args.neuron}, beta={args.beta}, rho={args.rho}')
    model = CVX_ReLU_MLP(X, y, n_classes, args.neuron, args.beta, args.rho, jax.random.PRNGKey(args.seed))
    model.init_model()
    model.Xtst = Xtst
    model.ytst = ytst
    
    cronos_params = dict(
        rank=args.rank, neuron=args.neuron, beta=args.beta, rho=args.rho,
        gamma_ratio=args.gamma_ratio, admm_iters=args.admm_iters,
        pcg_iters=args.pcg_iters, check_opt=args.check_opt
    )
    print(f'ADMM params: {cronos_params}')
    
    t0 = time.time()
    for i in range(2):
        _, metrics = admm(model, cronos_params)
        if i == 1:
            elapsed = time.time() - t0
            print(f'ADMM training done in {elapsed:.1f}s')
    
    train_peak = np.max(metrics['train_acc'])
    val_peak = np.max(metrics['val_acc'])
    print(f'Peak train acc: {train_peak:.6f}')
    print(f'Peak val acc: {val_peak:.6f}')
    
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {args.output_path}')
    
    return val_peak

if __name__ == '__main__':
    main()
