"""Train one (variant, fold) RNATransformer. Same inputs/outputs as ov.trainer (Keras Nullrecurrent).

Usage: python -m ov.trainer_tfm --run runs/<name> --head tfm --fold 0
Config section `tfm` (see configs/t1_tfm.yaml)."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml

from ov.trainer import Batches, _rows, load_sets
from ov.tfm import RNATransformer


def loss_fn(p, y, m, w, den, col_w):
    se = (m * (y - p) ** 2).sum(1) / den[:, None]
    rmse = torch.sqrt(se + 1e-12)
    per_col = (rmse * w[:, None]).sum(0) / w.sum()
    return (per_col * col_w).sum()


@torch.no_grad()
def predict(model, X, A, dev, batch_size=64):
    model.eval()
    out = []
    for s in range(0, len(X), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size])).to(dev)
        a = torch.from_numpy(np.ascontiguousarray(A[s:s + batch_size])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p1 = model(x, a).float()
            p2 = model(x.flip(1), a.flip(1, 2)).float().flip(1)
        out.append(((p1 + p2) / 2).cpu().numpy())
    return np.concatenate(out)


def val_score(model, X, A, idx, y, sn, dev):
    p = predict(model, X[idx], _rows(A, idx), dev)[:, :y.shape[1]]
    yt, keep = y[idx], sn[idx]
    errs = []
    for c in range(3):
        a, b = yt[keep, :, c].ravel(), p[keep, :, c].ravel()
        ok = ~np.isnan(a)
        errs.append(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    return float(np.mean(errs)), p


def variant_config(cfg, head):
    """-> (training config with variant overrides, model kwargs) for one variant."""
    tc = dict(cfg["tfm"])
    variant = dict(tc.get("model", {}))
    for k, v in tc.get("variants", {}).get(head, {}).items():
        if k in ("lr", "weight_decay", "batch_size", "epochs"):
            tc[k] = v
        else:
            variant[k] = v
    return tc, variant


def pretrained_path(cfg, run, head):
    tc = cfg["tfm"]
    if tc.get("pretrain_from"):
        return os.path.join(os.path.dirname(run.rstrip("/")), tc["pretrain_from"], "pretrain", "%s.pt" % head)
    if tc.get("pretrain"):
        return os.path.join(run, "pretrain", "%s.pt" % head)
    return None


def to_dev(batch, dev):
    return [torch.from_numpy(np.ascontiguousarray(b)).to(dev) for b in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    run = args.run
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    tc, variant = variant_config(cfg, args.head)
    prep = np.load(os.path.join(run, "prep.npz"))
    sets = load_sets(run)
    seed = cfg["seed"] * 1000 + args.fold * 10 + sorted(cfg["heads"]).index(args.head)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    tr = sets["train"]
    folds = prep["folds"]
    tr_idx, va_idx = np.where(folds != args.fold)[0], np.where(folds == args.fold)[0]
    y, err, w = prep["y"], prep["err"], prep["w"]
    sn = prep["sn_filter"].astype(bool)
    train = Batches(tr["X"], tr["A"], y, err, w, tr_idx)
    pls = []
    if cfg.get("pl"):
        for name in cfg["pl"]["sets"]:
            d = np.load(os.path.join(run, "pl_%s.npz" % name))
            pls.append(Batches(sets[name]["X"], sets[name]["A"], d["y"], d["err"], d["w"]))

    model = RNATransformer(tr["X"].shape[2], tr["A"].shape[3], **variant).to(dev)
    pre = pretrained_path(cfg, run, args.head)
    if pre:
        model.load_state_dict(torch.load(pre, map_location=dev))
        print("%s_f%d loaded %s" % (args.head, args.fold, pre), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    col_w = torch.tensor(cfg["col_weights"], device=dev)
    bs = tc["batch_size"]
    epochs = tc["epochs"]
    pl_frac = cfg["pl"].get("tfm_pl_frac", 1.0) if pls else 0.0
    pl_end = int(epochs * cfg["pl"].get("tfm_pl_until", 0.8)) if pls else 0
    steps_per_epoch = math.ceil(len(train) / bs) + (int(sum(len(p) for p in pls) * pl_frac / bs) if pls else 0)
    total = epochs * steps_per_epoch
    warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    rev = 0.5 if cfg["reverse_aug"] else 0.0

    tag = "%s_f%d" % (args.head, args.fold)
    log, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        batches = [(None, ids) for ids in train.epoch(bs, rng)]
        if pls and ep < pl_end:
            for k, ds in enumerate(pls):
                n = int(len(ds) * pl_frac)
                sub = rng.choice(len(ds), n, replace=False)
                batches += [(k, sub[s:s + bs]) for s in range(0, n, bs)]
        losses = []
        for j in rng.permutation(len(batches)):
            k, ids = batches[j]
            if k is None:
                b = train.get(ids, rng, perturb=cfg["perturb"], reverse_p=rev)
            else:
                b = pls[k].get(ids, rng, perturb=cfg["pl"]["perturb"], reverse_p=rev)
            x, a, yy, m, ww, den = to_dev(b, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x, a).float()
            loss = loss_fn(p, yy, m, ww, den, col_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if ep % 5 == 4 or ep == epochs - 1:
            s, _ = val_score(model, tr["X"], tr["A"], va_idx, y, sn, dev)
            log.append(dict(epoch=ep, loss=float(np.mean(losses)), val=s, t=time.time() - t0))
            print(tag, log[-1], flush=True)

    out = os.path.join(run, "models")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "%s.pt" % tag))
    s, oof = val_score(model, tr["X"], tr["A"], va_idx, y, sn, dev)
    preds = {"oof_idx": va_idx, "oof": oof.astype(np.float32)}
    for name, st in sets.items():
        if name != "train":
            preds[name] = predict(model, st["X"], st["A"], dev).astype(np.float32)
    np.savez_compressed(os.path.join(out, "%s_preds.npz" % tag), **preds)
    json.dump(dict(head=args.head, fold=args.fold, val=s, log=log), open(os.path.join(out, "%s.json" % tag), "w"), indent=1)
    print(tag, "FINAL val %.5f  (%.0fs)" % (s, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
