# Paper Code Map

This release keeps only code needed to reproduce the camera-ready experiments.

## Certifiers

| Paper method | Implementation | Used by |
| --- | --- | --- |
| `(E,C,G)+M` bounded higher-order certificate | `src/regression_certifiers/certify/bounded_fn_certifier_with_mean.py` | MNIST rotation, UTKFace age regression |
| `(E,C)+M` bounded ablation | `src/regression_certifiers/certify/bounded_fn_certifier_variance_mean.py` | MNIST rotation, UTKFace postprocessing |
| Unbounded `(C,G)` certificate | `src/regression_certifiers/certify/variance_gradient_certifier.py` | Synthetic experiment, MNIST estimate postprocessing |
| Alpha-smoothing baseline | `src/regression_certifiers/certify/alpha_trimming_certifier.py` | Synthetic, MNIST rotation, UTKFace age regression |

`src/regression_certifiers/certify/bounded_fn_certifier.py` is compatibility support for the MNIST convergence helper. It is not a main result method in the paper.

## Experiments

| Paper experiment | Entry points |
| --- | --- |
| Synthetic functions | `experiments/synthetic/run_unbounded_synthetic.py`, `experiments/synthetic/summarize_unbounded_grid.py` |
| MNIST rotation regression | `experiments/mnist_rotation/mnist_rotation_full_certification.py`, `experiments/mnist_rotation/mnist_alpha_trimming_certification.py`, `experiments/mnist_rotation/compute_ec_radii_from_estimates.py`, `experiments/mnist_rotation/compute_ecg_radii_from_estimates.py`, `experiments/mnist_rotation/plot_cdf_best_sigma_updated.py` |
| UTKFace age regression | `experiments/utkface_age/utkface_bounded_vs_alpha_experiment.py`, `experiments/utkface_age/postprocess_utkface_ec_from_saved_estimates.py`, `experiments/utkface_age/analyze_utkface_split_mode_results.py`, `experiments/utkface_age/plot_utkface_appendix_convergence.py` |

## Deliberately Excluded

This release omits development-only certifiers, old submission folders, review-response notes, raw datasets, checkpoints, generated grids, and cluster-specific scripts with private paths.
