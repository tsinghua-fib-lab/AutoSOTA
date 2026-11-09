"""Masked-node pre-training of one RNATransformer variant on unlabelled inputs (train + test + random sets).

A fraction of positions get their node features zeroed; the model reconstructs them (MSE on masked positions).
Usage: python -m ov.pretrain_tfm --run runs/<name> --head <variant>   -> <run>/pretrain/<variant>.pt
Config: tfm.pretrain {epochs, lr, mask_frac, batch_size, sets}."""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from ov.trainer import _rows, load_sets
from ov.tfm import RNATransformer
from ov.trainer_tfm import variant_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    cfg = yaml.safe_load(open(os.path.join(args.run, "config.yaml")))
    tc, variant = variant_config(cfg, args.head)
    pc = tc["pretrain"]
    sets = load_sets(args.run)
    names = [n for n in pc.get("sets", list(sets)) if n in sets]
    seed = cfg["seed"] * 1000 + 999
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    n_feat, n_adj = sets["train"]["X"].shape[2], sets["train"]["A"].shape[3]
    model = RNATransformer(n_feat, n_adj, **variant).to(dev)
    recon = nn.Linear(model.ln.normalized_shape[0], n_feat).to(dev)
    params = list(model.parameters()) + list(recon.parameters())
    opt = torch.optim.AdamW(params, lr=pc.get("lr", 1e-3), weight_decay=tc["weight_decay"])
    bs, epochs, frac = pc.get("batch_size", 32), pc["epochs"], pc.get("mask_frac", 0.15)
    steps = epochs * sum(math.ceil(len(sets[n]["X"]) / bs) for n in names)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        batches = []
        for n in names:
            order = rng.permutation(len(sets[n]["X"]))
            batches += [(n, order[s:s + bs]) for s in range(0, len(order), bs)]
        losses = []
        for j in rng.permutation(len(batches)):
            n, ids = batches[j]
            ids = np.sort(ids)
            x = torch.from_numpy(sets[n]["X"][ids]).to(dev)
            a = torch.from_numpy(np.ascontiguousarray(_rows(sets[n]["A"], ids))).to(dev)
            if rng.random() < 0.5:
                x, a = x.flip(1), a.flip(1, 2)
            m = torch.rand(x.shape[:2], device=dev) < frac
            m[:, 0] = m[:, -1] = False
            xin = x.masked_fill(m[..., None], 0.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = recon(model.encode(xin, a)).float()
            loss = ((out - x) ** 2).mean(-1)[m].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        print("pretrain %s epoch %d loss %.5f (%.0fs)" % (args.head, ep, np.mean(losses), time.time() - t0), flush=True)

    out = os.path.join(args.run, "pretrain")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "%s.pt" % args.head))


if __name__ == "__main__":
    main()
