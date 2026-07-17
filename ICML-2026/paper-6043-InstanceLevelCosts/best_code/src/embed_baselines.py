"""
embed_baselines.py

Beyond TF-IDF: frozen Transformer embeddings + linear heads.

Pipeline:
  • Load processed Jigsaw data (comment_text, y_star, abs_delta, delta_signed).
  • Optional subsample (--sample N).
  • Encode with HF model (mean/CLS pooling) to fixed-size vectors.
  • Heads:
      - LogisticRegression for classification (y*).
      - Ridge for signed Δ regression (L2 loss).
  • Report:
      - cls_unweighted_acc, cls_weighted_acc
      - reg_mae, reg_rmse, reg_weighted_mae, reg_weighted_rmse
      - reg_sign_acc, reg_sign_weighted_acc
  • Save CSV under results/bert/.

Usage:
  python -m src.embed_baselines --sample 10000
  python -m src.embed_baselines --sample 50000 --use_weights
  python -m src.embed_baselines --model unitary/unbiased-toxic-roberta --pool mean
"""

import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from src.utils import load_processed

# Quieter tokenizers when forking/etc.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESULTS = pathlib.Path("results/bert")
RESULTS.mkdir(parents=True, exist_ok=True)


# ----------------------- device -----------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------- embed ------------------------
def embed_texts(
    texts,
    tokenizer,
    model,
    device,
    max_length=128,
    batch_size=64,
    pool="mean",  # "mean" or "cls"
):
    """Encode a list of strings -> numpy array [N, D]."""
    model.eval()
    out_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i : i + batch_size]
            toks = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            outputs = model(**toks)
            h = getattr(outputs, "last_hidden_state", None)
            if h is None:  # e.g., if we accidentally loaded a head-only model
                h = model.base_model(**toks).last_hidden_state

            if pool == "cls":
                vec = h[:, 0, :]
            else:
                mask = toks["attention_mask"].unsqueeze(-1)  # [B,T,1]
                summed = (h * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                vec = summed / counts

            out_vecs.append(vec.detach().cpu().numpy())

    return np.concatenate(out_vecs, axis=0) if out_vecs else np.zeros((0, model.config.hidden_size))


# --------------------- metrics ------------------------
def weighted_acc(y_true, y_pred, w):
    return float((w * (y_true == y_pred)).sum() / w.sum())


def weighted_mae(y_true, y_pred, w):
    return float(np.sum(w * np.abs(y_pred - y_true)) / np.sum(w))


def weighted_rmse(y_true, y_pred, w):
    return float(np.sqrt(np.sum(w * (y_pred - y_true) ** 2) / np.sum(w)))


# ----------------------- main -------------------------
def main(args):
    # Load processed Jigsaw (utils reconstructs delta_signed)
    df = load_processed()  # expects columns: comment_text, y_star, abs_delta, delta_signed

    # Optional subsample
    if args.sample is not None and args.sample < len(df):
        df = df.sample(int(args.sample), random_state=42)

    texts = df["comment_text"].astype(str).tolist()
    y = df["y_star"].astype(int).values                 # 0/1 majority label (y*)
    w = df["abs_delta"].astype(float).values            # |Δ|
    d = df["delta_signed"].astype(float).values         # signed Δ

    X_tr_txt, X_te_txt, y_tr, y_te, w_tr, w_te, d_tr, d_te = train_test_split(
        texts, y, w, d, test_size=0.2, stratify=y, random_state=42
    )

    # HF model
    from transformers import AutoTokenizer, AutoModel
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    # Embed
    X_tr_vec = embed_texts(
        X_tr_txt, tokenizer, model, device,
         max_length=args.max_length, batch_size=args.batch_size, pool=args.pool
    )
    X_te_vec = embed_texts(
        X_te_txt, tokenizer, model, device,
         max_length=args.max_length, batch_size=args.batch_size, pool=args.pool
    )

    # ---------------- classification (LogReg) ----------------
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs" if X_tr_vec.shape[0] < 100_000 else "saga",
        n_jobs=-1,
    )
    if args.use_weights:
        clf.fit(X_tr_vec, y_tr, sample_weight=w_tr)
    else:
        clf.fit(X_tr_vec, y_tr)
    cls_pred = clf.predict(X_te_vec)

    cls_unw_acc = accuracy_score(y_te, cls_pred)
    cls_w_acc = weighted_acc(y_te, cls_pred, w_te)

    # ---------------- regression (Ridge on signed Δ) ----------
    reg = Ridge(alpha=1.0)
    if args.use_weights:
        reg.fit(X_tr_vec, d_tr, sample_weight=w_tr)
    else:
        reg.fit(X_tr_vec, d_tr)
    d_hat = reg.predict(X_te_vec)

    reg_mae = mean_absolute_error(d_te, d_hat)
    reg_rmse = np.sqrt(mean_squared_error(d_te, d_hat))
    reg_w_mae = weighted_mae(d_te, d_hat, w_te)
    reg_w_rmse = weighted_rmse(d_te, d_hat, w_te)

    # Sign metrics (predict positive vs negative Δ)
    d_hat_lbl = (d_hat >= 0).astype(int)
    reg_sign_acc = accuracy_score(y_te, d_hat_lbl)
    reg_sign_w_acc = weighted_acc(y_te, d_hat_lbl, w_te)

    # ---------------- save ----------------
    rows = [
        ("cls_unweighted_acc", cls_unw_acc),
        ("cls_weighted_acc",   cls_w_acc),
        ("reg_mae",            reg_mae),
        ("reg_rmse",           reg_rmse),
        ("reg_weighted_mae",   reg_w_mae),
        ("reg_weighted_rmse",  reg_w_rmse),
        ("reg_sign_acc",       reg_sign_acc),
        ("reg_sign_weighted_acc", reg_sign_w_acc),
    ]
    df_metrics = pd.DataFrame(rows, columns=["Metric", "Value"])

    tag = args.model.split("/")[-1]
    if args.use_weights:
        tag += "_w"
    if args.sample:
        tag += f"_n{int(args.sample)}"

    out_csv = RESULTS / f"embed_metrics_{tag}.csv"
    df_metrics.to_csv(out_csv, index=False)
    print(df_metrics.to_string(index=False))
    print(f"[+] wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="unitary/unbiased-toxic-roberta")
    ap.add_argument("--sample", type=float, default=None, help="Rows to sample BEFORE split.")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean","cls"])
    ap.add_argument("--use_weights", action="store_true", help="Use |Δ| as sample_weight during training.")
    args = ap.parse_args()
    main(args)

