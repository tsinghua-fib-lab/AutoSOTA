#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
from cleanup_ssps.sspspace import RandomSSPSpace

def generate_and_save(ssp_space, total, batch_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    batches = total // batch_size + int(total % batch_size > 0)
    for b in range(batches):
        start = b * batch_size
        end   = min(start + batch_size, total)
        count = end - start

        ssps, _ = ssp_space.get_sample_pts_and_ssps(count, method='sobol')
        for i in range(count):
            idx = start + i
            path = os.path.join(out_dir, f'random_ssp_{idx:06d}.npy')
            np.save(path, ssps[i])

        print(f"[{b+1}/{batches}] Wrote {count} files → {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate RandomSSPSpace SSPs matching existing hex‑SSP counts."
    )
    p.add_argument("--input-root",  type=str, required=True,
                   help="Root of existing hex‑SSP tree (e.g. …/data)")
    p.add_argument("--output-root", type=str, required=True,
                   help="Where to write random SSPs")
    p.add_argument("--batch-size",  type=int, default=2000,
                   help="How many to generate per batch")
    args = p.parse_args()

    # fixed domain settings
    domain_dim = 2
    sides = 1
    bounds = np.array([[2, sides+2], [2, sides+2]])

    # find every config folder: dim_*_scale_*
    pattern = os.path.join(args.input_root, "dim_*_scale_*")
    configs = sorted(glob.glob(pattern))
    if not configs:
        raise RuntimeError(f"No configs found under {args.input_root}")

    for cfg in configs:
        base = os.path.basename(cfg)  # e.g. "dim_295_scale_0.7"
        parts = base.split("_")
        dim   = int(parts[1])
        scale = float(parts[3])
        print(f"\n=== {base} ===")

        # count existing SSPs
        train_dir = os.path.join(cfg, "train", "coordinate_ssps")
        test_dir  = os.path.join(cfg, "test",  "coordinate_ssps")
        n_train = len(glob.glob(os.path.join(train_dir, "*.npy")))
        n_test  = len(glob.glob(os.path.join(test_dir,  "*.npy")))
        print(f" Found {n_train} train, {n_test} test")

        # make the RandomSSPSpace
        space = RandomSSPSpace(
            domain_dim    = domain_dim,
            ssp_dim       = dim,
            domain_bounds = bounds,
            length_scale  = scale,
            rng           = np.random.default_rng()
        )

        # generate train
        out_train = os.path.join(args.output_root, base, "train", "random_ssps")
        print("→ Generating TRAIN…")
        generate_and_save(space, n_train, args.batch_size, out_train)

        # generate test
        out_test = os.path.join(args.output_root, base, "test", "random_ssps")
        print("→ Generating TEST…")
        generate_and_save(space, n_test, args.batch_size, out_test)

    print("\n✅ All done!")
