#!/usr/bin/env python3
"""Run the runner-up's 82M parameter checkpoint with original FP32 math."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from atomic_model import build_model, TYPES
from metrics import group_log_mae, validate_submission

ROOT = Path(__file__).resolve().parents[1]


def load_arrays(split, base=None):
    base = base or ROOT / "data/atomic"
    arrays = {name: np.load(base / (split + "_" + name + ".npy"), mmap_mode="r")
              for name in ("xyz", "atom", "type", "pair_group", "pair_atom", "pair_id", "pair_type")}
    with np.load(base / "normalization.npz") as data:
        arrays["mean"] = data["mean"]
        arrays["std"] = data["std"]
    return arrays


def model_inputs(arrays, indices, device):
    xyz = (np.array(arrays["xyz"][indices]) - arrays["mean"]) / arrays["std"]
    atom = np.array(arrays["atom"][indices], dtype=np.int64)
    coupling_type = np.array(arrays["type"][indices], dtype=np.int64)
    valid = atom >= 0
    xyz *= valid[:, None, :]
    xyz = torch.as_tensor(xyz, device=device)
    atom = torch.as_tensor(atom[:, None, :], device=device)
    coupling_type = torch.as_tensor(coupling_type[:, None, :], device=device)
    valid = torch.as_tensor(valid, device=device)
    # Only the zero/nonzero Coulomb mask is consumed by the original model.
    coulomb = (valid[:, :, None] & valid[:, None, :]).float()
    unused = torch.zeros_like(atom, dtype=torch.float32)
    return (xyz, coupling_type, unused, atom, unused, coulomb,
            valid[:, None, :], valid.sum(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke-groups", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--arrays-dir", type=Path, default=ROOT / "data/atomic")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/atomic_inference")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--submission-file", type=Path, default=ROOT / "submissions/runnerup_single_reproduced.csv")
    args = parser.parse_args()
    arrays = load_arrays(args.split, args.arrays_dir)
    groups = len(arrays["xyz"])
    # Symmetry supplies input context, but the original submission predicts
    # only source groups containing original (not added reverse) pairs.
    active_groups = np.unique(arrays["pair_group"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.combine:
        predictions = np.full((groups, 29), np.nan, dtype=np.float32)
        for shard in range(args.shards):
            start, stop = len(active_groups) * shard // args.shards, len(active_groups) * (shard + 1) // args.shards
            predictions[active_groups[start:stop]] = np.load(output_dir / (args.split + "_shard%d.npy" % shard))
        row_predictions = predictions[arrays["pair_group"], arrays["pair_atom"]]
        frame = pd.DataFrame({"id": np.array(arrays["pair_id"]), "scalar_coupling_constant": row_predictions})
        reference = pd.read_csv(args.reference_dir / (args.split + ".csv"))
        report = validate_submission(frame, reference)
        if args.split == "train":
            report["training_set_metric_not_oof"] = group_log_mae(reference.scalar_coupling_constant.values, row_predictions, reference.type.values)
        else:
            args.submission_file.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.submission_file, index=False, float_format="%.9g")
        report.update({"mode": "checkpoint_inference", "checkpoint": "loss-5.6396.pth", "test_labels_used": False,
                       "precision": "FP32, TF32 disabled", "shards": args.shards,
                       "input_directory": str(args.reference_dir), "submission_file": str(args.submission_file)})
        (output_dir / (args.split + "_verification.json")).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(ROOT / "artifacts/runnerup_checkpoint/model_state.pt", map_location="cpu", weights_only=True)
    model = build_model(state).cuda().eval()
    start, stop = len(active_groups) * args.shard // args.shards, len(active_groups) * (args.shard + 1) // args.shards
    indices = active_groups[start:stop]
    if args.smoke_groups:
        indices = indices[np.linspace(0, len(indices) - 1, min(args.smoke_groups, len(indices)), dtype=np.int64)]
        predictions = np.empty((len(indices), 29), dtype=np.float32)
    else:
        predictions = open_memmap(output_dir / (args.split + "_shard%d.npy" % args.shard), mode="w+", dtype=np.float32, shape=(len(indices), 29))
    started, last = time.time(), time.time()
    with torch.inference_mode():
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset:offset + args.batch_size]
            inputs = model_inputs(arrays, selected, "cuda")
            result = model(*inputs)[2].sum(dim=1).float().cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite inference output")
            predictions[offset:offset + len(selected)] = result
            if time.time() - last > 25:
                print(json.dumps({"split": args.split, "shard": args.shard, "groups": offset + len(selected),
                                  "total": len(indices), "seconds": round(time.time() - started, 1)}), flush=True)
                last = time.time()
    if args.smoke_groups:
        if args.split == "train":
            target = np.load(args.arrays_dir / "train_target.npy", mmap_mode="r")[indices]
            selected_pairs = np.isin(arrays["pair_group"], indices)
            local_group = np.searchsorted(indices, arrays["pair_group"][selected_pairs])
            pair_atom = arrays["pair_atom"][selected_pairs]
            names = np.array(TYPES)[arrays["pair_type"][selected_pairs]]
            report = group_log_mae(target[local_group, pair_atom], predictions[local_group, pair_atom], names)
            report["evaluation_scope"] = "training examples; not OOF and not a leaderboard score"
            report["groups"] = len(indices)
            (output_dir / "train_smoke.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
    else:
        predictions.flush()
    print(json.dumps({"split": args.split, "shard": args.shard, "complete": True,
                      "groups": len(indices), "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
