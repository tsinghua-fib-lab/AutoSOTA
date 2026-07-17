import argparse, os, shutil, json
import numpy as np
import pandas as pd

def _load_frame(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")

def _save_frame(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _ensure_delta_cols(df, delta_col):
    if delta_col not in df.columns:
        raise ValueError(f"Delta column '{delta_col}' not found. Available: {list(df.columns)[:20]}")
    df = df.copy()
    df["delta_abs"] = df[delta_col].abs()
    # sign: +1 for >=0, -1 for <0 (ties to +1)
    df["y_sign"] = np.where(df[delta_col] >= 0, 1, -1)
    return df

def _prob_upsample_by_abs_delta_stratified(df, epoch_size=None, seed=1337):
    """Sample with replacement with probability ∝ |Δ| **within each sign bucket**,
    then concatenate buckets to preserve sign balance."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in (-1, 1):
        d = df[df["y_sign"] == s].copy()
        if len(d) == 0:
            continue
        w = d["delta_abs"].to_numpy()
        if np.all(w == 0):
            # fallback to uniform if all zeros
            p = None
        else:
            wsum = w.sum()
            if wsum == 0:
                p = None
            else:
                p = w / wsum
        n_take = len(d) if epoch_size is None else int(round(epoch_size * (len(d) / len(df))))
        idx = rng.choice(len(d), size=n_take, replace=True, p=p)
        parts.append(d.iloc[idx])
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def _topk_by_abs_delta_stratified(df, keep_frac=0.5):
    parts = []
    for s in (-1, 1):
        d = df[df["y_sign"] == s].copy()
        if len(d) == 0:
            continue
        thr = d["delta_abs"].quantile(1 - keep_frac)
        parts.append(d[d["delta_abs"] >= thr])
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out

def _copy_eval_splits(src_dir, dst_dir, val_name, test_name):
    os.makedirs(dst_dir, exist_ok=True)
    for split_name in (val_name, test_name):
        src = os.path.join(src_dir, split_name)
        dst = os.path.join(dst_dir, split_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def _summarize(df, name):
    by_sign = df.groupby("y_sign")["y_sign"].count().to_dict()
    return {
        "name": name,
        "n": int(len(df)),
        "mean_abs_delta": float(df["delta_abs"].mean()) if "delta_abs" in df else None,
        "by_sign_counts": {str(k): int(v) for k, v in by_sign.items()},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True, choices=["jigsaw", "turkey"],
                    help="Name used for output folder structure.")
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing the base splits.")
    ap.add_argument("--train_file", required=True, help="Path to train split (.csv or .parquet).")
    ap.add_argument("--val_file", required=True, help="Path to val split (.csv or .parquet).")
    ap.add_argument("--test_file", required=True, help="Path to test split (.csv or .parquet).")
    ap.add_argument("--delta_col", default="delta", help="Column name for Δ.")
    ap.add_argument("--out_root", default="data", help="Root output dir for variant splits.")
    ap.add_argument("--format", choices=["csv", "parquet"], default=None,
                    help="Override output format; if omitted, inferred from input train_file.")
    ap.add_argument("--keep_fracs", nargs="+", type=float, default=[0.5, 0.3, 0.7],
                    help="Fractions for Tdown variants (top-k by |Δ|), e.g., 0.5 0.3 0.7")
    args = ap.parse_args()

    train = _load_frame(args.train_file)
    val = _load_frame(args.val_file)
    test = _load_frame(args.test_file)

    # Infer extension
    if args.format is None:
        ext = os.path.splitext(args.train_file)[1].lower().lstrip(".")
        if ext == "pq": ext = "parquet"
    else:
        ext = args.format

    train = _ensure_delta_cols(train, args.delta_col)
    # For eval splits we do NOT touch labels; but we add delta_abs/y_sign for convenience if present
    try:
        val = _ensure_delta_cols(val, args.delta_col)
    except Exception:
        pass
    try:
        test = _ensure_delta_cols(test, args.delta_col)
    except Exception:
        pass

    # Output base dirs
    base_out = os.path.join(args.out_root, args.dataset_name)
    os.makedirs(base_out, exist_ok=True)

    logs = []

    # U (uniform, unchanged)
    outU = os.path.join(base_out, "U")
    os.makedirs(outU, exist_ok=True)
    _save_frame(train, os.path.join(outU, f"train.{ext}"))
    _save_frame(val,   os.path.join(outU,   f"val.{ext}"))
    _save_frame(test,  os.path.join(outU,  f"test.{ext}"))
    logs.append(_summarize(train, "U"))

    # P_up (probabilistic upsampling by |Δ| within sign)
    outP = os.path.join(base_out, "P_up")
    os.makedirs(outP, exist_ok=True)
    train_P = _prob_upsample_by_abs_delta_stratified(train, epoch_size=len(train))
    _save_frame(train_P, os.path.join(outP, f"train.{ext}"))
    _save_frame(val,     os.path.join(outP, f"val.{ext}"))
    _save_frame(test,    os.path.join(outP, f"test.{ext}"))
    logs.append(_summarize(train_P, "P_up"))

    # Tdown variants (top-k% |Δ| within sign)
    for k in args.keep_fracs:
        tag = f"Tdown{int(k*100)}"
        outT = os.path.join(base_out, tag)
        os.makedirs(outT, exist_ok=True)
        train_T = _topk_by_abs_delta_stratified(train, keep_frac=k)
        _save_frame(train_T, os.path.join(outT, f"train.{ext}"))
        _save_frame(val,     os.path.join(outT, f"val.{ext}"))
        _save_frame(test,    os.path.join(outT, f"test.{ext}"))
        logs.append(_summarize(train_T, tag))

    # Write summary log
    with open(os.path.join(base_out, "sampling_summary.json"), "w") as f:
        json.dump(logs, f, indent=2)
    print(json.dumps(logs, indent=2))

if __name__ == "__main__":
    main()

