"""End-to-end Nullrecurrent training: data prep -> AE pre-train -> (head x fold) jobs on GPUs -> blend + score.

  python scripts/train.py --config configs/s1_nopl.yaml --gpus 0,1,2,3,4,5,6,7

Test labels are read only in the final `score` step (data/test_labels.json)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ov import data as D  # noqa: E402
from ov.folding import eternafold_mfe  # noqa: E402
from ov.metrics import load_labels, score_test  # noqa: E402
from ov.paths import DATA, MODEL_COLS, RUNS  # noqa: E402

PY = sys.executable
TORCH_PY = "/usr/local/bin/python"  # system interpreter with PyTorch 2.0 (venv has TensorFlow)
OFFICIAL_SETS = ("test107", "test130")


def prepare_frames(cfg):
    train = D.load_train()
    test = D.load_test()
    # structure: kaggle = as provided (train/test107 Vienna MFE, test130 EternaFold MFE);
    #            eternafold = EternaFold MFE everywhere (matches the private 130-mers)
    if cfg.get("structure", "kaggle") == "eternafold" or cfg.get("train_structure") == "eternafold":
        from multiprocessing import Pool
        with Pool(100) as p:
            train["structure"] = p.map(eternafold_mfe, list(train.sequence))
            if cfg.get("structure") == "eternafold":
                test["structure"] = p.map(eternafold_mfe, list(test.sequence))
    train = D.prepare_frame(train, "bprna")
    test = D.prepare_frame(test, cfg.get("test_loop", "bprna"))
    frames = {"train": train, "test107": test[test.seq_length == 107], "test130": test[test.seq_length == 130]}
    for spec in cfg.get("random_sets", []):
        path = os.path.join(DATA, "random_%s.json" % spec["name"])
        if os.path.exists(path):
            df = pd.read_json(path, lines=True)
        else:
            templates = list(frames["test130"].sequence) if spec["length"] == 130 else list(train.sequence)
            df = D.make_random_frame(spec["n"], spec["length"], templates, seed=spec["seed"])
            df.to_json(path, orient="records", lines=True)
        frames[spec["name"]] = df
    return frames


def prepare(cfg, run):
    frames = prepare_frames(cfg)
    meta = {}
    for name, df in frames.items():
        D.build_features(df, name)
        fx, fa = D.feature_paths(df, name)
        meta[name] = dict(X=fx, A=fa, ids=list(df.id), length=int(df.seq_length.iloc[0]))
    with open(os.path.join(run, "sets.json"), "w") as f:
        json.dump(meta, f)

    train = frames["train"]
    y, err = D.train_targets(train, **cfg["mask"])
    labels = D.edit_distance_clusters(list(train.sequence), cfg["cluster_threshold"])
    w = D.cluster_weights(labels) if cfg.get("cluster_weight", True) else np.ones(len(train))
    if cfg.get("snr_weight"):
        w = w * np.clip(np.log(train.signal_to_noise.values + 1.1) / 2, 0.05, None)
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(labels.max() + 1)
    folds = perm[labels] % cfg["n_folds"]
    np.savez(os.path.join(run, "prep.npz"), y=y, err=err, w=w.astype(np.float32), folds=folds,
             sn_filter=train.SN_filter.values, clusters=labels)
    return frames, meta


def launch(cmds, gpus, per_gpu, logdir):
    """Run shell commands with at most per_gpu concurrent jobs per GPU."""
    os.makedirs(logdir, exist_ok=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running, queue, failed = [], list(cmds), []
    while queue or running:
        while queue and slots:
            name, cmd = queue.pop(0)
            g = slots.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TF_CPP_MIN_LOG_LEVEL="2",
                       OMP_NUM_THREADS="4", TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="2")
            log = open(os.path.join(logdir, name + ".log"), "w")
            running.append((name, g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), log))
        time.sleep(5)
        for item in list(running):
            name, g, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                slots.append(g)
                if p.returncode != 0:
                    failed.append(name)
                    print("FAILED", name, "see", os.path.join(logdir, name + ".log"), flush=True)
                else:
                    print("done", name, flush=True)
    if failed:
        raise SystemExit("failed jobs: %s" % failed)


def official_preds(meta, names, gpu):
    """Predictions of the released 20 Nullrecurrent models on given sets (cached)."""
    out = {}
    for name in names:
        path = meta[name]["X"].replace("_X.npy", "_official.npy")
        if not os.path.exists(path):
            cmd = [PY, os.path.join(ROOT, "scripts", "predict_official_sets.py"), meta[name]["X"], meta[name]["A"], path]
            subprocess.run(cmd, check=True, cwd=ROOT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TF_CPP_MIN_LOG_LEVEL="2"))
        out[name] = np.load(path)  # [20, N, L, 5]
    return out


def teacher_preds(cfg, meta, gpu):
    """Stack of teacher model predictions per PL set: {name: [M, N, L, 5]}."""
    """Returns ({name: [M, N, L, 5]}, model weights [M]); each teacher run gets total weight
    pl.teacher_weights[i] (default 1), split evenly over its models."""
    pl = cfg["pl"]
    stacks = {n: [] for n in pl["sets"]}
    weights = []
    tw = pl.get("teacher_weights") or [1.0] * len(pl.get("teachers", []))
    for t, wt in zip(pl.get("teachers", []), tw):
        tdir = os.path.join(RUNS, t, "models")
        files = [f for f in sorted(os.listdir(tdir)) if f.endswith("_preds.npz")]
        for f in files:
            d = np.load(os.path.join(tdir, f))
            for n in stacks:
                stacks[n].append(d[n])
            weights.append(wt / len(files))
    if pl.get("use_official"):
        for n, p in official_preds(meta, pl["sets"], gpu).items():
            stacks[n].extend(list(p))
        weights += [1.0 / 20] * 20
    w = np.array(weights) / np.sum(weights)
    return {n: np.clip(np.stack(v), -0.5, 6) for n, v in stacks.items()}, w


def make_pl(cfg, run, meta, gpu):
    pl = cfg["pl"]
    stacks, w = teacher_preds(cfg, meta, gpu)
    for name, p in stacks.items():
        n = pl["positions"][str(meta[name]["length"])]
        p = p[:, :, :n]
        mean = np.tensordot(w, p, 1)
        std = np.sqrt(np.tensordot(w, (p - mean) ** 2, 1))
        thr = np.quantile(std.reshape(-1, 5), pl["std_mask_quantile"], axis=0)
        y = np.where(std > thr, np.nan, mean).astype(np.float32)
        np.savez(os.path.join(run, "pl_%s.npz" % name), y=y, err=std.astype(np.float32),
                 w=np.full(len(y), pl.get("weight", 1.0), np.float32))
        print("PL %s: %d seqs x %d pos from %d teachers, masked %.1f%%" % (
            name, len(y), n, len(p), 100 * np.isnan(y).mean()), flush=True)


def gather(cfg, run, meta, frames):
    mdir = os.path.join(run, "models")
    prep = np.load(os.path.join(run, "prep.npz"))
    y = prep["y"]
    oof = np.full(y.shape, np.nan, np.float32)
    per_head = {}
    test = {"test107": [], "test130": []}
    vals = []
    for head in cfg["heads"]:
        oof_h = np.zeros(y.shape, np.float32)
        for fold in range(cfg["n_folds"]):
            tag = "%s_f%d" % (head, fold)
            d = np.load(os.path.join(mdir, tag + "_preds.npz"))
            oof_h[d["oof_idx"]] = d["oof"]
            for n in test:
                test[n].append(d[n])
            vals.append(json.load(open(os.path.join(mdir, tag + ".json")))["val"])
        per_head[head] = oof_h
    oof = np.mean([per_head[h] for h in cfg["heads"]], axis=0)
    sn = prep["sn_filter"].astype(bool)

    def cv(p):
        e = []
        for c in range(3):
            yt, yp = y[sn, :, c].ravel(), p[sn, :, c].ravel()
            ok = ~np.isnan(yt)
            e.append(np.sqrt(np.mean((yt[ok] - yp[ok]) ** 2)))
        return float(np.mean(e))

    res = {"cv_blend": cv(oof), "cv_heads": {h: cv(p) for h, p in per_head.items()},
           "val_mean_single": float(np.mean(vals))}
    rows = []
    for n in test:
        p = np.mean(np.clip(np.stack(test[n]), -0.5, 6), axis=0)
        for uid, v in zip(meta[n]["ids"], p):
            rows.append(dict(id=uid, **{c: v[:, k].tolist() for k, c in enumerate(MODEL_COLS)}))
    sub = pd.DataFrame(rows)
    sub.to_json(os.path.join(run, "test_pred.json"), orient="records", lines=True)

    labels = load_labels(os.path.join(DATA, "test_labels.json"))
    s = score_test(sub, labels)
    res.update(public=s["public"][0], private=s["private"][0], private_cols=[float(v) for v in s["private"][1]])
    with open(os.path.join(run, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--jobs-per-gpu", type=int, default=2)
    ap.add_argument("--name", default=None)
    ap.add_argument("--only", choices=["prepare", "train", "gather"], default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    name = args.name or cfg["name"]
    run = os.path.join(RUNS, name)
    os.makedirs(run, exist_ok=True)
    shutil.copy(args.config, os.path.join(run, "config.yaml"))
    gpus = [int(g) for g in args.gpus.split(",")]

    t0 = time.time()
    frames, meta = prepare(cfg, run)
    print("prepared sets: %s  (%.0fs)" % ({k: len(v) for k, v in frames.items()}, time.time() - t0), flush=True)
    if args.only == "prepare":
        return
    if args.only != "gather":
        if cfg.get("pl"):
            make_pl(cfg, run, meta, gpus[0])
        if cfg.get("ae_from") and not os.path.exists(os.path.join(run, "base_ae.h5")):
            shutil.copy(os.path.join(RUNS, cfg["ae_from"], "base_ae.h5"), os.path.join(run, "base_ae.h5"))
        if cfg.get("ae") and not os.path.exists(os.path.join(run, "base_ae.h5")):
            launch([("ae", [PY, "-m", "ov.ae", "--run", run])], gpus[:1], 1, os.path.join(run, "logs"))
            print("ae done (%.0fs)" % (time.time() - t0), flush=True)
        if cfg.get("model") == "tfm" and cfg["tfm"].get("pretrain") and not cfg["tfm"].get("pretrain_from"):
            pre = [("pretrain_%s" % h, [TORCH_PY, "-m", "ov.pretrain_tfm", "--run", run, "--head", h])
                   for h in cfg["heads"] if not os.path.exists(os.path.join(run, "pretrain", "%s.pt" % h))]
            launch(pre, gpus, args.jobs_per_gpu, os.path.join(run, "logs"))
            print("pretrain done (%.0fs)" % (time.time() - t0), flush=True)
        trainer = [TORCH_PY, "-m", "ov.trainer_tfm"] if cfg.get("model") == "tfm" else [PY, "-m", "ov.trainer"]
        cmds = [("%s_f%d" % (h, f), trainer + ["--run", run, "--head", h, "--fold", str(f)])
                for h in cfg["heads"] for f in range(cfg["n_folds"])
                if not os.path.exists(os.path.join(run, "models", "%s_f%d_preds.npz" % (h, f)))]
        launch(cmds, gpus, args.jobs_per_gpu, os.path.join(run, "logs"))
        print("training done (%.0fs)" % (time.time() - t0), flush=True)
    gather(cfg, run, meta, frames)


if __name__ == "__main__":
    main()
