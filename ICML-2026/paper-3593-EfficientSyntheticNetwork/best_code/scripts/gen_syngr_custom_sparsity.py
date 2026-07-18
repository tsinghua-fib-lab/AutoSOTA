import sys, pickle, argparse, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syngler.res.bootstrap import bootstrap_latents
from syngler.utils.source import reconstruct_adjacency

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitted_pkl', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--num_samples', type=int, default=200)
    ap.add_argument('--sparsity_override', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    
    with open(args.fitted_pkl, 'rb') as f:
        d = pickle.load(f)
    Z = np.asarray(d['model_Z'])
    alpha = np.asarray(d['model_alpha']).flatten()
    rho = args.sparsity_override if args.sparsity_override is not None else float(d.get('model_sparsity', 0.0))
    
    print(f'Z shape={Z.shape}, alpha shape={alpha.shape}, rho={rho:.6f}')
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    for k, (Zk, ak) in enumerate(bootstrap_latents(Z, alpha, args.num_samples, seed=args.seed)):
        A = reconstruct_adjacency(Zk, ak, rho=rho, seed=args.seed + k + 1)
        np.save(out / f'rep{k}.npy', A.astype(np.uint8))
    
    print(f'Generated {args.num_samples} samples -> {out}')

if __name__ == '__main__':
    main()
