import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import argparse
import pickle
import numpy as np
from tqdm import tqdm

def load_result(base_dir: str, r: int, seed: int):
    file_path = os.path.join(base_dir, f"r={r}", f"seed={seed}.pkl")
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    Z = np.asarray(results["model_Z"], dtype=np.float32)
    alpha = np.asarray(results["model_alpha"], dtype=np.float32).reshape(-1, 1)
    sparsity = results.get("model_sparsity", 0.0)
    return Z, alpha, sparsity, file_path

from ForestDiffusion import ForestDiffusionModel as ForestFlowModel

def build_forest_model(Z: np.ndarray,
                       alpha: np.ndarray,
                       seed: int,
                       n_t: int = 100,
                       duplicate_K: int = 100,
                       xgb_params: dict | None = None):
    if xgb_params is None:
        xgb_params = {}
    X = np.hstack([Z, alpha])
    y_dummy = np.zeros(X.shape[0], dtype=np.int64)
    model = ForestFlowModel(
        X,
        label_y=y_dummy,
        n_t=n_t,
        duplicate_K=duplicate_K,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        diffusion_type="vp",
        n_jobs=-1,
        seed=int(seed),
        **xgb_params,
    )
    return model, X

def generate_and_save(model: ForestFlowModel,
                      X: np.ndarray,
                      r: int,
                      out_dir: str,
                      reps: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    n = X.shape[0]
    for b in range(reps):
        Xy_fake = model.generate(batch_size=n)
        x_fake = Xy_fake[:, :-1]
        Z_fake = x_fake[:, :r]
        alpha_fake = x_fake[:, r:r+1]
        np.savez(os.path.join(out_dir, f"rep{b}.npz"), Z=Z_fake, alpha=alpha_fake)

def process_once(dataset: str,
                 r: int,
                 seed: int,
                 data_root: str,
                 out_root: str,
                 reps: int,
                 xgb_params: dict,
                 model_cfg: dict):
    input_base = os.path.join(data_root, dataset, "run")
    try:
        Z, alpha, sparsity, src_path = load_result(input_base, r, seed)
    except FileNotFoundError:
        tqdm.write(f"[WARN] missing input: dataset={dataset}, r={r}, seed={seed}, base_dir={input_base}")
        return False

    model, X = build_forest_model(
        Z, alpha, seed,
        n_t=model_cfg.get("n_t", 100),
        duplicate_K=model_cfg.get("duplicate_K", 100),
        xgb_params=xgb_params
    )

    out_dir = os.path.join(out_root, dataset, "Diff-sample", f"r={r}", f"seed={seed}")
    os.makedirs(out_dir, exist_ok=True)

    meta = dict(dataset=dataset, r=r, seed=seed, n=int(X.shape[0]), reps=reps, src_path=src_path, sparsity=sparsity)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    generate_and_save(model, X, r, out_dir, reps=reps)
    tqdm.write(f"[OK] dataset={dataset}, r={r} -> {out_dir} (reps={reps})")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dblp", "youtube", "yelp", "polblogs"])
    parser.add_argument("--data-root", default="../../datasets")
    parser.add_argument("--out-root", default="../../synthetic")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--n_t", type=int, default=100)
    parser.add_argument("--duplicate_K", type=int, default=100)
    parser.add_argument("--r_values", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    args = parser.parse_args()

    xgb_params = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        eta=args.eta,
        tree_method=args.tree_method,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
    )
    model_cfg = dict(n_t=args.n_t, duplicate_K=args.duplicate_K)

    ok = 0
    total = 0
    for r in args.r_values:
        total += 1
        ok += bool(process_once(args.dataset, r, args.seed, args.data_root, args.out_root, args.reps, xgb_params, model_cfg))

    print(f"Complete {ok}/{total} r-values for dataset={args.dataset}.")

if __name__ == "__main__":
    main()
