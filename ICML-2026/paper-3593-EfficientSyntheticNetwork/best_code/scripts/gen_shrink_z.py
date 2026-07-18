import sys, pickle, argparse, numpy as np
from pathlib import Path
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syngler.lsm.source import sigmoid
from syngler.res.bootstrap import bootstrap_latents
from syngler.utils.source import reconstruct_adjacency

def calibrate_sparsity(A_ref, Z, alpha, r0=-7.0):
    n = A_ref.shape[0]
    ZZT = Z @ Z.T
    alpha_outer = np.outer(alpha, np.ones(n)) + np.outer(np.ones(n), alpha)
    Theta_no_rho = ZZT + alpha_outer
    Theta_no_rho = np.triu(Theta_no_rho, 1) + np.triu(Theta_no_rho, 1).T
    observed_density = np.triu(A_ref, k=1).sum() / (n * (n - 1) / 2)
    
    def density_gap(rho_val):
        P_trial = sigmoid(Theta_no_rho + rho_val)
        P_trial = np.triu(P_trial, 1) + np.triu(P_trial, 1).T
        return np.triu(P_trial, k=1).sum() / (n * (n - 1) / 2) - observed_density
    
    try:
        return float(brentq(density_gap, -20.0, 20.0, xtol=1e-6))
    except ValueError:
        return float(np.log(observed_density / (1 - observed_density)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitted_pkl', required=True)
    ap.add_argument('--ref_adj', default='data/real/polblogs/generator/seed=0.npy')
    ap.add_argument('--output', required=True)
    ap.add_argument('--num_samples', type=int, default=200)
    ap.add_argument('--shrink_z', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    
    with open(args.fitted_pkl, 'rb') as f:
        d = pickle.load(f)
    Z = np.asarray(d['model_Z']).copy()
    alpha = np.asarray(d['model_alpha']).flatten().copy()
    
    if args.shrink_z > 0:
        Z = Z * (1.0 - args.shrink_z)
        print(f'Z shrunk by factor {args.shrink_z}: new std={Z.std():.4f}')
    
    A_ref = np.load(args.ref_adj).astype(np.float32)
    rho = calibrate_sparsity(A_ref, Z, alpha)
    print(f'Calibrated rho={rho:.6f}')
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    for k, (Zk, ak) in enumerate(bootstrap_latents(Z, alpha, args.num_samples, seed=args.seed)):
        A = reconstruct_adjacency(Zk, ak, rho=rho, seed=args.seed + k + 1)
        np.save(out / f'rep{k}.npy', A.astype(np.uint8))
    
    print(f'Generated {args.num_samples} samples -> {out}')

if __name__ == '__main__':
    main()
