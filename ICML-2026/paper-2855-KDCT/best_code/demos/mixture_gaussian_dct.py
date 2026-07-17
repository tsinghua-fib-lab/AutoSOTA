"""Gaussian-mixture DCT demo.

The synthetic problem is deliberately small enough to be inspectable.  The
reference pair uses a balanced perturbation of a uniform Gaussian mixture, and
the alternative pair uses the same perturbation direction with a modestly larger
contrast on a more concentrated base distribution.  Kernel bandwidths are
selected from independent pilot samples.  The DCT threshold is not calibrated
from reference samples; it is epsilon plus the asymptotic standard error
estimated from each test sample.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm


def regular_simplex_centers(num_components, radius):
    eye = np.eye(num_components)
    centered = eye - np.ones((num_components, num_components)) / num_components
    return centered * (radius / np.sqrt(2.0))


def contrast_vector(num_components):
    if num_components % 2 != 0:
        raise ValueError("num_components must be even.")
    contrast = np.concatenate(
        [np.ones(num_components // 2), -np.ones(num_components // 2)]
    )
    return contrast / np.linalg.norm(contrast)


def common_weights(num_components, contrast, contrast_scale, concentrated):
    min_mass = contrast_scale * np.max(np.abs(contrast)) + 1e-4
    if min_mass * num_components >= 1.0:
        raise ValueError("contrast_scale is too large for a valid mixture.")
    if not concentrated:
        return np.ones(num_components) / num_components

    weights = np.ones(num_components) * min_mass
    weights[0] += 1.0 - weights.sum()
    return weights


def mixture_weights(common, contrast, contrast_scale, sign):
    weights = common + sign * contrast_scale * contrast
    if np.min(weights) < -1e-12:
        raise ValueError("Invalid mixture weights.")
    weights = np.maximum(weights, 0.0)
    return weights / weights.sum()


def component_kernel(centers, bandwidth, component_std):
    dim = centers.shape[1]
    sq_dists = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    scale = (bandwidth**2 / (bandwidth**2 + 4.0 * component_std**2)) ** (dim / 2.0)
    return scale * np.exp(-sq_dists / (bandwidth**2 + 4.0 * component_std**2))


def sample_component_counts(rng, common, contrast, contrast_scale, sign, sample_size):
    weights = mixture_weights(common, contrast, contrast_scale, sign)
    return rng.multinomial(sample_size, weights)


def sample_component_labels(rng, common, contrast, contrast_scale, sign, sample_size):
    weights = mixture_weights(common, contrast, contrast_scale, sign)
    return rng.choice(len(weights), size=sample_size, p=weights)


def mmd_nammd_from_counts(count_x, count_y, kernel):
    n = int(count_x.sum())
    diag = np.diag(kernel)
    xx = (count_x @ kernel @ count_x - np.sum(count_x * diag)) / (n * (n - 1.0))
    yy = (count_y @ kernel @ count_y - np.sum(count_y * diag)) / (n * (n - 1.0))
    xy = (count_x @ kernel @ count_y) / (n * n)
    mmd = xx - 2.0 * xy + yy
    nammd = mmd / (4.0 - xx - yy)
    return mmd, nammd


def draw_pair_counts(rng, common, contrast, contrast_scale, sample_size):
    count_x = sample_component_counts(
        rng, common, contrast, contrast_scale, 1.0, sample_size
    )
    count_y = sample_component_counts(
        rng, common, contrast, contrast_scale, -1.0, sample_size
    )
    return count_x, count_y


def statistic_sample(rng, common, contrast, contrast_scale, sample_size, kernel):
    labels_x = sample_component_labels(
        rng, common, contrast, contrast_scale, 1.0, sample_size
    )
    labels_y = sample_component_labels(
        rng, common, contrast, contrast_scale, -1.0, sample_size
    )
    stats = statistics_from_labels(labels_x, labels_y, kernel)
    return stats["mmd"], stats["nammd"]


def statistics_from_labels(labels_x, labels_y, kernel, eps=1e-12):
    n = len(labels_x)
    if n != len(labels_y):
        raise ValueError("labels_x and labels_y must have the same length.")
    if n < 4:
        raise ValueError("sample_size must be at least 4.")

    num_components = kernel.shape[0]
    count_x = np.bincount(labels_x, minlength=num_components)
    count_y = np.bincount(labels_y, minlength=num_components)

    k_x_count_x = kernel[labels_x] @ count_x
    k_y_count_y = kernel[labels_y] @ count_y
    k_x_count_y = kernel[labels_x] @ count_y
    k_y_count_x = kernel[labels_y] @ count_x

    k_xx_diag = kernel[labels_x, labels_x]
    k_yy_diag = kernel[labels_y, labels_y]
    k_xy_diag = kernel[labels_x, labels_y]

    h_sums = (
        (k_x_count_x - k_xx_diag)
        + (k_y_count_y - k_yy_diag)
        - (k_x_count_y - k_xy_diag)
        - (k_y_count_x - k_xy_diag)
    )
    d_sums = 4.0 * (n - 1.0) - (k_x_count_x - k_xx_diag) - (k_y_count_y - k_yy_diag)

    numerator = h_sums.sum()
    denominator = d_sums.sum()

    mmd = numerator / (n * (n - 1.0))
    nammd = numerator / max(denominator, eps)
    reg = denominator / (n * (n - 1.0))

    h_rows = h_sums / (n - 1.0)
    d_rows = d_sums / (n - 1.0)
    se_mmd = 2.0 * np.std(h_rows, ddof=1) / np.sqrt(n)
    nammd_influence = (h_rows - nammd * d_rows) / max(reg, eps)
    se_nammd = 2.0 * np.std(nammd_influence, ddof=1) / np.sqrt(n)

    return {
        "mmd": mmd,
        "nammd": nammd,
        "se_mmd": max(se_mmd, eps),
        "se_nammd": max(se_nammd, eps),
    }


def population_statistics(common, contrast, contrast_scale, kernel):
    weights_x = mixture_weights(common, contrast, contrast_scale, 1.0)
    weights_y = mixture_weights(common, contrast, contrast_scale, -1.0)
    xx = weights_x @ kernel @ weights_x
    yy = weights_y @ kernel @ weights_y
    xy = weights_x @ kernel @ weights_y
    mmd = xx - 2.0 * xy + yy
    nammd = mmd / (4.0 - xx - yy)
    return {"mmd": mmd, "nammd": nammd}


def select_bandwidths(args, centers, contrast, common_ref, common_alt):
    rng = np.random.default_rng(args.seed + 17)
    alt_contrast_scale = args.contrast_scale * args.alt_contrast_multiplier
    candidate_bandwidths = np.logspace(
        args.log10_bandwidth_min, args.log10_bandwidth_max, args.num_bandwidths
    )

    rows = []
    best_mmd = None
    best_nammd = None
    for bandwidth in candidate_bandwidths:
        kernel = component_kernel(centers, bandwidth, args.component_std)
        mmd_gaps = []
        nammd_gaps = []
        for _ in range(args.selection_reps):
            ref_mmd, ref_nammd = statistic_sample(
                rng,
                common_ref,
                contrast,
                args.contrast_scale,
                args.sample_size,
                kernel,
            )
            alt_mmd, alt_nammd = statistic_sample(
                rng,
                common_alt,
                contrast,
                alt_contrast_scale,
                args.sample_size,
                kernel,
            )
            mmd_gaps.append(alt_mmd - ref_mmd)
            nammd_gaps.append(alt_nammd - ref_nammd)

        mmd_gaps = np.asarray(mmd_gaps)
        nammd_gaps = np.asarray(nammd_gaps)
        mmd_score = mmd_gaps.mean() / (mmd_gaps.std(ddof=1) + 1e-12)
        nammd_score = nammd_gaps.mean() / (nammd_gaps.std(ddof=1) + 1e-12)
        row = {
            "bandwidth": bandwidth,
            "mmd_gap": mmd_gaps.mean(),
            "nammd_gap": nammd_gaps.mean(),
            "mmd_score": mmd_score,
            "nammd_score": nammd_score,
        }
        rows.append(row)
        if best_mmd is None or row["mmd_score"] > best_mmd["mmd_score"]:
            best_mmd = row
        if best_nammd is None or row["nammd_score"] > best_nammd["nammd_score"]:
            best_nammd = row

    return best_mmd, best_nammd, rows


def reference_level(args, centers, contrast, common_ref, bandwidth, stat_index):
    kernel = component_kernel(centers, bandwidth, args.component_std)
    stat_name = "mmd" if stat_index == 0 else "nammd"
    return population_statistics(
        common_ref,
        contrast,
        args.contrast_scale,
        kernel,
    )[stat_name]


def asymptotic_test(args, centers, contrast, common_ref, common_alt, bandwidth, epsilon, stat_index):
    rng = np.random.default_rng(args.seed + 101 + stat_index)
    kernel = component_kernel(centers, bandwidth, args.component_std)
    z_threshold = norm.ppf(1.0 - args.alpha)
    stat_name = "mmd" if stat_index == 0 else "nammd"
    se_name = "se_mmd" if stat_index == 0 else "se_nammd"
    alt_contrast_scale = args.contrast_scale * args.alt_contrast_multiplier

    def one_trial(common, contrast_scale):
        labels_x = sample_component_labels(
            rng, common, contrast, contrast_scale, 1.0, args.sample_size
        )
        labels_y = sample_component_labels(
            rng, common, contrast, contrast_scale, -1.0, args.sample_size
        )
        stats = statistics_from_labels(labels_x, labels_y, kernel)
        statistic = stats[stat_name]
        se = stats[se_name]
        threshold = epsilon + z_threshold * se
        return statistic, threshold, int(statistic > threshold)

    null_trials = np.array(
        [one_trial(common_ref, args.contrast_scale) for _ in range(args.test_reps)]
    )
    alt_trials = np.array(
        [one_trial(common_alt, alt_contrast_scale) for _ in range(args.test_reps)]
    )

    return {
        "epsilon": epsilon,
        "z_threshold": z_threshold,
        "type_i": float(null_trials[:, 2].mean()),
        "power": float(alt_trials[:, 2].mean()),
        "null_mean": float(null_trials[:, 0].mean()),
        "alt_mean": float(alt_trials[:, 0].mean()),
        "null_threshold_mean": float(null_trials[:, 1].mean()),
        "alt_threshold_mean": float(alt_trials[:, 1].mean()),
    }


def write_outputs(output_dir, selection_rows, summary_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mixture_gaussian_kernel_selection.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bandwidth",
                "mmd_gap",
                "nammd_gap",
                "mmd_score",
                "nammd_score",
            ],
        )
        writer.writeheader()
        writer.writerows(selection_rows)

    with (output_dir / "mixture_gaussian_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "bandwidth",
                "epsilon",
                "z_threshold",
                "type_i",
                "power",
                "null_mean",
                "alt_mean",
                "null_threshold_mean",
                "alt_threshold_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reference-sample kernel selection demo for MMD and NAMMD."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--num_components", type=int, default=20)
    parser.add_argument("--component_std", type=float, default=0.02)
    parser.add_argument("--simplex_radius", type=float, default=3.0)
    parser.add_argument("--contrast_scale", type=float, default=0.1)
    parser.add_argument("--alt_contrast_multiplier", type=float, default=1.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num_bandwidths", type=int, default=17)
    parser.add_argument("--log10_bandwidth_min", type=float, default=-0.4)
    parser.add_argument("--log10_bandwidth_max", type=float, default=0.8)
    parser.add_argument("--selection_reps", type=int, default=300)
    parser.add_argument("--test_reps", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=Path("Results/demos"))
    return parser.parse_args()


def main():
    args = parse_args()
    centers = regular_simplex_centers(args.num_components, args.simplex_radius)
    contrast = contrast_vector(args.num_components)

    common_ref = common_weights(
        args.num_components, contrast, args.contrast_scale, concentrated=False
    )
    alt_contrast_scale = args.contrast_scale * args.alt_contrast_multiplier
    common_alt = common_weights(
        args.num_components, contrast, alt_contrast_scale, concentrated=True
    )

    best_mmd, best_nammd, selection_rows = select_bandwidths(
        args, centers, contrast, common_ref, common_alt
    )
    epsilon_mmd = reference_level(
        args, centers, contrast, common_ref, best_mmd["bandwidth"], stat_index=0
    )
    epsilon_nammd = reference_level(
        args, centers, contrast, common_ref, best_nammd["bandwidth"], stat_index=1
    )
    mmd_result = asymptotic_test(
        args,
        centers,
        contrast,
        common_ref,
        common_alt,
        best_mmd["bandwidth"],
        epsilon_mmd,
        stat_index=0,
    )
    nammd_result = asymptotic_test(
        args,
        centers,
        contrast,
        common_ref,
        common_alt,
        best_nammd["bandwidth"],
        epsilon_nammd,
        stat_index=1,
    )

    summary_rows = [
        {"method": "MMD", "bandwidth": best_mmd["bandwidth"], **mmd_result},
        {"method": "NAMMD", "bandwidth": best_nammd["bandwidth"], **nammd_result},
    ]
    write_outputs(args.output_dir, selection_rows, summary_rows)

    print(
        "Reference contrast="
        f"{args.contrast_scale:.4g}, alternative contrast="
        f"{alt_contrast_scale:.4g}"
    )
    print("Epsilon source: population value of the synthetic reference pair")
    print("Reference-sample selected bandwidths")
    print(
        f"MMD   : sigma={best_mmd['bandwidth']:.6g}, "
        f"selection score={best_mmd['mmd_score']:.3f}"
    )
    print(
        f"NAMMD : sigma={best_nammd['bandwidth']:.6g}, "
        f"selection score={best_nammd['nammd_score']:.3f}"
    )
    print("")
    for row in summary_rows:
        print(
            f"{row['method']:6s} type-I={row['type_i']:.3f}, "
            f"power={row['power']:.3f}, epsilon={row['epsilon']:.6g}, "
            f"mean null threshold={row['null_threshold_mean']:.6g}"
        )
    print(f"\nSaved CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
