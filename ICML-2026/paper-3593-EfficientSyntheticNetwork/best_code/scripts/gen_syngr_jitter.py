"""Generate SyNG-R samples with jittered bootstrap to reduce duplicate-latent clustering."""
import sys, pickle, argparse, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syngler.utils.source import bootstrap_alpha_Z, reconstruct_adjacency

def generate_graphs_jitter(model_Z, model_alpha, n_reps, rho=0.0, seed=0, jitter_std=0.0):
    """SyNG-R with jittered bootstrap latents.
    
    jitter_std: std of Gaussian noise added to bootstrapped Z and alpha.
    If jitter_std=0, this is identical to standard SyNG-R.
    """
    model_alpha = np.asarray(model_alpha).reshape(-1, 1)
    model_Z = np.asarray(model_Z)
    
    for k in range(n_reps):
        np.random.seed(seed + k)
        a_raw, Z_raw = bootstrap_alpha_Z(model_alpha, model_Z, batch=1)
        Z = Z_raw.squeeze(0)
        a = a_raw.squeeze(0)
        
        # Add jitter
        if jitter_std > 0:
            rng = np.random.default_rng(seed + k + 100000)
            Z = Z + rng.normal(0, jitter_std, size=Z.shape).astype(Z.dtype)
            a = a + rng.normal(0, jitter_std, size=a.shape).astype(a.dtype)
        
        yield reconstruct_adjacency(Z, a, rho=rho, seed=seed + k + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitted_pkl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--jitter_std", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    with open(args.fitted_pkl, "rb") as f:
        d = pickle.load(f)
    Z = np.asarray(d["model_Z"])
    alpha = np.asarray(d["model_alpha"]).flatten()
    rho = float(d.get("model_sparsity", 0.0))
    
    Z_std = float(np.std(Z))
    jitter_abs = args.jitter_std * Z_std
    print(f"Z std={Z_std:.4f}, jitter_std_factor={args.jitter_std}, jitter_abs={jitter_abs:.6f}")
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    
    for k, A in enumerate(generate_graphs_jitter(Z, alpha, args.num_samples,
                                                   rho=rho, seed=args.seed,
                                                   jitter_std=jitter_abs)):
        np.save(out / f"rep{k}.npy", A.astype(np.uint8))
    print(f"SyNG-R+jitter: {args.num_samples} samples -> {out}")

if __name__ == "__main__":
    main()
