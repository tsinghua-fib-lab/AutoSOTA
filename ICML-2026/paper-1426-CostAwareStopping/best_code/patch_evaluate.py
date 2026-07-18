#!/usr/bin/env python3
"""Patch evaluate.py to support gaussian_likelihood and dataset subset."""
import sys

with open(sys.argv[1], "r") as f:
    content = f.read()

# Edit 1: Add gaussian_likelihood param to run_one_experiment
old_sig = 'def run_one_experiment(bench, dataset_name, seed, acq="PBGI(1e-4)",'
new_sig = 'def run_one_experiment(bench, dataset_name, seed, acq="PBGI(1e-4)", gaussian_likelihood=False,'
content = content.replace(old_sig, new_sig)

# Edit 2: Change fit_gp_model calls to use gaussian_likelihood
content = content.replace(
    "old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize)",
    "old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize, gaussian_likelihood=gaussian_likelihood)"
)
content = content.replace(
    "model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)",
    "model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize, gaussian_likelihood=gaussian_likelihood)"
)

# Edit 3: Add --gaussian-likelihood and --datasets flags to arg parser
old_parser = 'parser.add_argument("--n_seeds", type=int, default=50,'
new_parser = '''parser.add_argument("--gaussian_likelihood", action="store_true",
                        help="Use learned Gaussian likelihood instead of fixed noise")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of datasets to run (default: all)")
    ''' + old_parser
content = content.replace(old_parser, new_parser)

# Edit 4: Pass gaussian_likelihood to run_one_experiment calls in full mode
content = content.replace(
    "results[d][s] = run_one_experiment(bench, d, s, acq=acq)",
    "results[d][s] = run_one_experiment(bench, d, s, acq=acq, gaussian_likelihood=args.gaussian_likelihood)"
)

# Edit 5: Support --datasets in full mode (filter datasets)
old_all_ds = "all_datasets = bench.get_dataset_names()"
new_all_ds = """all_datasets = bench.get_dataset_names()
        if args.datasets:
            requested = [d.strip() for d in args.datasets.split(",")]
            all_datasets = [d for d in all_datasets if d in requested]
            print(f"Filtered to {len(all_datasets)} datasets: {all_datasets}")"""
content = content.replace(old_all_ds, new_all_ds)

with open(sys.argv[1], "w") as f:
    f.write(content)

print("Patched successfully")
