#!/usr/bin/env python3
import os
import numpy as np
from cleanup_ssps.sspspace import RandomSSPSpace

def generate_and_save_random_ssps_batched(ssp_space, total_samples, batch_size, output_dir):
    """
    Generate and save Sobol‑sampled SSPs in batches.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_batches = total_samples // batch_size + (1 if total_samples % batch_size else 0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx   = min(start_idx + batch_size, total_samples)
        count     = end_idx - start_idx

        # 1) Sobol sample domain points & encode
        pts_ssp, pts = ssp_space.get_sample_pts_and_ssps(count, method='sobol')
        # pts_ssp shape = (count, ssp_dim)

        # 2) save each SSP
        for i in range(count):
            idx = start_idx + i
            out_path = os.path.join(output_dir, f'random_ssp_{idx:06d}.npy')
            np.save(out_path, pts_ssp[i])

        print(f"[{batch_idx+1}/{num_batches}] Saved {count} SSPs → {output_dir}")

if __name__ == "__main__":
    #  ── User‑tweakable settings ──────────────────────────────────
    base_data_dir  = '/u1/khabashy/CleanupSSP/data'
    total_samples  = 20000
    batch_size     = 2000
    train_ratio    = 0.8

    n_values       = [3, 5, 7, 9, 11, 13]
    length_scales  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    sides = 1
    bounds = np.array([[2, sides+2], [2, sides+2]])
    # ─────────────────────────────────────────────────────────────

    for n in n_values:
        for length_scale in length_scales:
            ssp_dim = n * n * 6 + 1
            print(f"\n=== Config n={n}, ssp_dim={ssp_dim}, length_scale={length_scale} ===")

            # build the random SSP space
            space = RandomSSPSpace(
                domain_dim    = 2,
                ssp_dim       = ssp_dim,
                domain_bounds = bounds,
                length_scale  = length_scale,
                rng           = np.random.default_rng(42)
            )

            # compute splits
            n_train = int(train_ratio * total_samples)
            n_test  = total_samples - n_train

            # prepare dirs
            cfg_dir  = os.path.join(base_data_dir, f'dim_{ssp_dim}_scale_{length_scale}')
            train_dir = os.path.join(cfg_dir, 'train', 'random_ssps')
            test_dir  = os.path.join(cfg_dir, 'test',  'random_ssps')

            # generate!
            print("→ Generating TRAIN SSPs...")
            generate_and_save_random_ssps_batched(space, n_train, batch_size, train_dir)

            print("→ Generating TEST SSPs...")
            generate_and_save_random_ssps_batched(space, n_test, batch_size, test_dir)

    print("\nAll done!")
