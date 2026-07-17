# src/turkey_image_baselines.py
import argparse, pathlib, os, math
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

def weighted_acc(y_true, y_pred, w):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); w = np.asarray(w)
    return float((w * (y_true == y_pred)).sum() / w.sum())

class TurkeyImageDataset(Dataset):
    def __init__(self, root: pathlib.Path, df: pd.DataFrame, transform):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.paths = [root / rel for rel in self.df["rel"].astype(str)]
        self.y_star = self.df["y_star"].astype(int).to_numpy()
        self.delta = self.df["delta"].astype(float).to_numpy()
        self.abs_delta = self.df["abs_delta"].astype(float).to_numpy()

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), self.y_star[i], self.delta[i], self.abs_delta[i]

def build_loader(root, df, batch_size):
    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = TurkeyImageDataset(root, df, tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

@torch.no_grad()
def embed_resnet50(loader, device):
    # torchvision >= 0.13
    try:
        weights = tvm.ResNet50_Weights.DEFAULT
        model = tvm.resnet50(weights=weights)
    except Exception:
        model = tvm.resnet50(pretrained=True)

    model.fc = torch.nn.Identity()  # 2048-d features
    model.eval().to(device)

    feats, y_star, delta, abs_delta = [], [], [], []
    for x, ys, d, w in loader:
        x = x.to(device)
        z = model(x)                      # [B, 2048]
        feats.append(z.cpu().numpy())
        y_star.append(ys.numpy())
        delta.append(d.numpy())
        abs_delta.append(w.numpy())
    F = np.concatenate(feats, axis=0)
    y = np.concatenate(y_star, axis=0)
    d = np.concatenate(delta, axis=0)
    w = np.concatenate(abs_delta, axis=0)
    return F, y, d, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to Turkey root containing fold1..fold5 and annotations.json")
    ap.add_argument("--sample", type=int, default=0, help="Optional train+test sample size for speed")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--use_weights", action="store_true", help="Use |Δ| as sample_weight during classification training")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out_dir = pathlib.Path("results/turkey"); out_dir.mkdir(parents=True, exist_ok=True)

    cost_csv = out_dir / "turkey_cost_table.csv"
    if not cost_csv.exists():
        raise SystemExit(f"Missing {cost_csv}. Run: python -m src.turkey_prelim --root {root}")

    df = pd.read_csv(cost_csv)
    # keep only images that exist
    df["exists"] = df["rel"].apply(lambda r: (root / r).exists())
    df = df[df["exists"]].copy()

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    # stratified split on y_star to keep class balance
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y_star"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # embed
    tr_loader = build_loader(root, tr, args.batch)
    te_loader = build_loader(root, te, args.batch)
    Xtr, ytr, dtr, wtr = embed_resnet50(tr_loader, device)
    Xte, yte, dte, wte = embed_resnet50(te_loader, device)

    # ===== Classification on y* =====
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    fit_kw = {}
    if args.use_weights:
        fit_kw["sample_weight"] = wtr
    clf.fit(Xtr, ytr, **fit_kw)
    yhat = clf.predict(Xte)
    cls_unw = accuracy_score(yte, yhat)
    cls_w   = weighted_acc(yte, yhat, wte)

    # ===== Δ regression (signed) =====
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr, dtr)
    d_hat = reg.predict(Xte)

    mae  = mean_absolute_error(dte, d_hat)
    rmse = math.sqrt(mean_squared_error(dte, d_hat))
    w_mae  = mean_absolute_error(dte, d_hat, sample_weight=wte)
    w_rmse = math.sqrt(mean_squared_error(dte, d_hat, sample_weight=wte))

    sign_pred = (d_hat >= 0).astype(int)
    sign_acc  = accuracy_score(yte, sign_pred)
    sign_wacc = weighted_acc(yte, sign_pred, wte)

    # ===== save metrics =====
    tag = f"_w_n{len(tr)}" if args.use_weights else f"_n{len(tr)}"
    out_csv = out_dir / f"image_embed_metrics_resnet50{tag}.csv"
    rows = [
        ("cls_unweighted_acc", cls_unw),
        ("cls_weighted_acc",   cls_w),
        ("reg_mae",            mae),
        ("reg_rmse",           rmse),
        ("reg_weighted_mae",   w_mae),
        ("reg_weighted_rmse",  w_rmse),
        ("reg_sign_acc",       sign_acc),
        ("reg_sign_weighted_acc", sign_wacc),
        ("n_train",            len(tr)),
        ("n_test",             len(te)),
        ("feature_dim",        Xtr.shape[1]),
    ]
    pd.DataFrame(rows, columns=["Metric","Value"]).to_csv(out_csv, index=False)
    print(pd.DataFrame(rows, columns=["Metric","Value"]))
    print("[+] wrote", out_csv)

if __name__ == "__main__":
    main()

