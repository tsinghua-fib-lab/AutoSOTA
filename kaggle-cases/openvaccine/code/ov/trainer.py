"""Train one (head, fold) Nullrecurrent model.

Usage: python -m ov.trainer --run runs/<name> --head gru --fold 0
Reads <run>/config.yaml and <run>/prep.npz (written by scripts/train.py)."""
import argparse
import json
import os
import time

import numpy as np
import yaml


def load_sets(run):
    """Feature arrays registered by scripts/train.py in <run>/sets.json."""
    with open(os.path.join(run, "sets.json")) as f:
        meta = json.load(f)
    sets = {}
    for name, m in meta.items():
        sets[name] = dict(m, X=np.load(m["X"]), A=np.load(m["A"], mmap_mode="r"))
    return sets


class Batches:
    """A labelled dataset: features + padded targets/mask/error + per-sample weight."""

    def __init__(self, X, A, y=None, err=None, w=None, idx=None):
        self.idx = np.arange(len(X)) if idx is None else np.asarray(idx)
        self.X, self.A = X, A
        self.y, self.err, self.w = y, err, w

    def __len__(self):
        return len(self.idx)

    def get(self, ids, rng=None, perturb=0.0, reverse_p=0.0):
        x = self.X[ids]
        a = _rows(self.A, ids)
        L = x.shape[1] - 2
        n = self.y.shape[1]
        y = np.zeros((len(ids), L, 5), np.float32)
        m = np.zeros((len(ids), L, 5), np.float32)
        yy = self.y[ids]
        valid = ~np.isnan(yy)
        yy = np.nan_to_num(yy)
        if perturb > 0 and rng is not None:
            yy = yy + rng.standard_normal(yy.shape).astype(np.float32) * self.err[ids] * perturb
        y[:, :n], m[:, :n] = yy, valid
        den = np.full(len(ids), n, np.float32)
        if reverse_p > 0 and rng is not None:
            flip = rng.random(len(ids)) < reverse_p
            x[flip] = x[flip, ::-1]
            a[flip] = a[flip, ::-1, ::-1]
            y[flip] = y[flip, ::-1]
            m[flip] = m[flip, ::-1]
        return x, a, y, m, self.w[ids].astype(np.float32), den

    def epoch(self, batch_size, rng):
        order = rng.permutation(self.idx)
        for s in range(0, len(order), batch_size):
            yield order[s:s + batch_size]


def build(cfg, n_feat, n_adj, head):
    import tensorflow as tf
    from ov import nullrec

    tf.keras.backend.clear_session()
    base = nullrec.get_base(n_feat, n_adj)
    model = nullrec.get_model(base, head, n_feat, n_adj)
    return base, model


def make_step(model, col_w, lr):
    import tensorflow as tf

    opt = tf.keras.optimizers.Adam(lr)
    col_w = tf.constant(col_w, tf.float32)

    def loss_fn(y, m, w, den, p):
        se = tf.reduce_sum(m * tf.square(y - p), axis=1) / den[:, None]  # [B, 5]
        rmse = tf.sqrt(se + 1e-12)
        per_col = tf.reduce_sum(rmse * w[:, None], axis=0) / tf.reduce_sum(w)
        return tf.reduce_sum(per_col * col_w)

    sig = [tf.TensorSpec([None, None, None], tf.float32), tf.TensorSpec([None, None, None, None], tf.float32),
           tf.TensorSpec([None, None, 5], tf.float32), tf.TensorSpec([None, None, 5], tf.float32),
           tf.TensorSpec([None], tf.float32), tf.TensorSpec([None], tf.float32)]

    @tf.function(input_signature=sig)
    def step(x, a, y, m, w, den):
        with tf.GradientTape() as tape:
            p = model([x, a], training=True)
            loss = loss_fn(y, m, w, den, p)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    return step, opt


def predict(model, X, A, batch_size=128, tta=True):
    import tensorflow as tf

    if not hasattr(model, "_ov_fwd"):
        @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32),
                                      tf.TensorSpec([None, None, None, None], tf.float32)])
        def _fwd(x, a):
            return model([x, a], training=False)
        model._ov_fwd = _fwd
    fwd = model._ov_fwd

    out = []
    for s in range(0, len(X), batch_size):
        x = X[s:s + batch_size]
        a = np.asarray(A[s:s + batch_size])
        p = fwd(x, a).numpy()
        if tta:
            p2 = fwd(np.ascontiguousarray(x[:, ::-1]), np.ascontiguousarray(a[:, ::-1, ::-1])).numpy()[:, ::-1]
            p = (p + p2) / 2
        out.append(p)
    return np.concatenate(out)


def val_score(model, val, sn_mask):
    """MCRMSE over the 3 scored columns, SN_filter=1 validation rows (Kaggle-like CV)."""
    p = predict(model, val.X[val.idx], _rows(val.A, val.idx))
    y = val.y[val.idx]
    n = y.shape[1]
    p = p[:, :n]
    keep = sn_mask[val.idx]
    errs = []
    for c in range(3):
        yt, yp = y[keep, :, c].ravel(), p[keep, :, c].ravel()
        ok = ~np.isnan(yt)
        errs.append(np.sqrt(np.mean((yt[ok] - yp[ok]) ** 2)))
    return float(np.mean(errs)), p


def _rows(A, idx):
    order = np.argsort(idx)
    rows = np.asarray(A[np.asarray(idx)[order]])
    return rows[np.argsort(order)]


def run_section(step, ds, epochs, batch_size, rng, cfg):
    for _ in range(epochs):
        for ids in ds.epoch(batch_size, rng):
            x, a, y, m, w, den = ds.get(ids, rng, perturb=cfg["perturb"], reverse_p=0.5 if cfg["reverse_aug"] else 0)
            step(x, a, y, m, w, den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()

    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    run = args.run
    with open(os.path.join(run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    prep = np.load(os.path.join(run, "prep.npz"))
    sets = load_sets(run)
    seed = cfg["seed"] * 1000 + args.fold * 10 + ["gru", "lstm", "forward", "wave"].index(args.head)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    tr = sets["train"]
    folds = prep["folds"]
    tr_idx, va_idx = np.where(folds != args.fold)[0], np.where(folds == args.fold)[0]
    y, err, w = prep["y"], prep["err"], prep["w"]
    train = Batches(tr["X"], tr["A"], y, err, w, tr_idx)
    val = Batches(tr["X"], tr["A"], y, err, w, va_idx)
    sn = prep["sn_filter"].astype(bool)

    n_feat, n_adj = tr["X"].shape[2], tr["A"].shape[3]
    base, model = build(cfg, n_feat, n_adj, args.head)
    if cfg.get("ae") and os.path.exists(os.path.join(run, "base_ae.h5")):
        base.load_weights(os.path.join(run, "base_ae.h5"))
    step, opt = make_step(model, cfg["col_weights"], cfg["lr"])

    log = []
    t0 = time.time()
    tag = "%s_f%d" % (args.head, args.fold)
    pl = cfg.get("pl")
    if not pl:
        for ep, bs in cfg["schedule"]:
            run_section(step, train, ep, bs, rng, cfg)
            s, _ = val_score(model, val, sn)
            log.append(dict(section="train", epochs=ep, bs=bs, val=s, t=time.time() - t0))
            print(tag, log[-1], flush=True)
    else:
        pls = []
        for name in pl["sets"]:
            d = np.load(os.path.join(run, "pl_%s.npz" % name))
            pls.append(Batches(sets[name]["X"], sets[name]["A"], d["y"], d["err"], d["w"]))
        pl_ds = pls
        for cyc in range(pl["cycles"]):
            run_section(step, train, pl["train_epochs"], pl["batch_size"], rng, cfg)
            s_tr, _ = val_score(model, val, sn)
            snap = [v.numpy() for v in model.weights]
            opt_snap = [v.numpy() for v in opt.variables]
            _pl_section(step, pl_ds, pl["pl_epochs"], pl["batch_size"], rng, cfg)
            s_pl, _ = val_score(model, val, sn)
            rolled = s_pl > s_tr + pl["rollback_tol"]
            if rolled:
                for v, a in zip(model.weights, snap):
                    v.assign(a)
                for v, a in zip(opt.variables, opt_snap):
                    v.assign(a)
            log.append(dict(section="cycle", cycle=cyc, val_after_train=s_tr, val_after_pl=s_pl,
                            rolled_back=bool(rolled), t=time.time() - t0))
            print(tag, log[-1], flush=True)
        for ep, bs in pl.get("final_schedule", []):
            run_section(step, train, ep, bs, rng, cfg)
            s, _ = val_score(model, val, sn)
            log.append(dict(section="final", epochs=ep, bs=bs, val=s, t=time.time() - t0))
            print(tag, log[-1], flush=True)

    out = os.path.join(run, "models")
    os.makedirs(out, exist_ok=True)
    model.save_weights(os.path.join(out, "%s.h5" % tag))
    s, oof = val_score(model, val, sn)
    res = dict(head=args.head, fold=args.fold, val=s, log=log)
    preds = {"oof_idx": va_idx, "oof": oof.astype(np.float32)}
    for name, st in sets.items():
        if name == "train":
            continue
        preds[name] = predict(model, st["X"], st["A"]).astype(np.float32)
    np.savez_compressed(os.path.join(out, "%s_preds.npz" % tag), **preds)
    with open(os.path.join(out, "%s.json" % tag), "w") as f:
        json.dump(res, f, indent=1)
    print(tag, "FINAL val %.5f  (%.0fs)" % (s, time.time() - t0), flush=True)


def _pl_section(step, pl_ds, epochs, batch_size, rng, cfg):
    """PL datasets may have different lengths; interleave their batches in random order."""
    for _ in range(epochs):
        batches = [(k, ids) for k, ds in enumerate(pl_ds) for ids in ds.epoch(batch_size, rng)]
        for j in rng.permutation(len(batches)):
            k, ids = batches[j]
            x, a, y, m, w, den = pl_ds[k].get(ids, rng, perturb=cfg["pl"]["perturb"],
                                             reverse_p=0.5 if cfg["reverse_aug"] else 0)
            step(x, a, y, m, w, den)


if __name__ == "__main__":
    main()
