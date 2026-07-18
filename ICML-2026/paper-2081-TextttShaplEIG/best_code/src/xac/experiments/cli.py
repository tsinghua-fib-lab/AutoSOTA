from __future__ import annotations

import json
import logging
from pathlib import Path

import gpytorch
import hydra
import threadpoolctl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.distributions import Categorical, MixtureSameFamily, Normal
from torch.nn.functional import binary_cross_entropy

from xac.acquisition_functions import SHAPIQAcquisitionFunction
from xac.applications import TabRepoBenchmarkApplication
from xac.experimental_designs.experiment_runner import run_experiment
from xac.surrogates import NUTSConfig
from xac.utils.metrics import (compute_accuracy, compute_ce_loss, compute_mae,
                               compute_mse, compute_nlpd, compute_msll, compute_calibration)
from xac.utils.plotting import plot_trajectory
from xac.utils.random_utils import set_seed

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------  #
#  Main entry-point (for one specific seed and config)                           #
# -----------------------------------------------------------------------------  #
@hydra.main(
    config_path="conf", config_name="shapley_example", version_base="1.3"
)  # benchmark_example, pdp_example, shapley_example, shapley_tree_example, shapley_initial_design_size
def main(cfg: DictConfig) -> None:
    """Hydra entry-point – runs exactly ONE job (seed × config combination for sequential experiment)."""

    set_seed(cfg.meta.seed)

    # ------------------------------------------------------------------
    # Instantiate objects from sub-configs
    # ------------------------------------------------------------------
    application = hydra.utils.instantiate(cfg.application)
    surrogate_cfg = hydra.utils.instantiate(cfg.surrogate)
    acquisition_fn = hydra.utils.instantiate(cfg.acquisition)
    acquisition_optimizer = hydra.utils.instantiate(cfg.acquisition_optimizer)
    blackbox_fn = hydra.utils.instantiate(cfg.blackbox)
    ed_cfg = hydra.utils.instantiate(cfg.experimental_design)
    meta_cfg = hydra.utils.instantiate(cfg.meta)

    # ------------------------------------------------------------------
    # Run the sequential loop
    # ------------------------------------------------------------------
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    try:
        run_id = int(HydraConfig.get().runtime.output_dir.rsplit("/", 1)[-1])
    except ValueError:
        run_id = 0

    (
        (prop_posts, prop_posts_noisy, test_predictions, test_labels, #train_labels, 
         prop_gt, sv_approximations),
        (archive_x, archive_y),
        (hp_fit_durations, prop_post_durations, acq_fun_durations, peak_rss_mb),
    ) = run_experiment(
        application=application,
        blackbox_fn=blackbox_fn,
        surrogate_cfg=surrogate_cfg,
        acquisition_fn=acquisition_fn,
        acquisition_optimizer=acquisition_optimizer,
        ed_cfg=ed_cfg,
        meta_cfg=meta_cfg,
        run_dir=run_dir,
    )

    metrics = {}

    if meta_cfg.debug_mode and blackbox_fn.is_pseudo_expensive:
        # ------------------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------------------
        if not isinstance(acquisition_fn, SHAPIQAcquisitionFunction):
            metrics["mae"] = torch.tensor(
                [compute_mae(temp_prop_post, prop_gt) for temp_prop_post in prop_posts]
            )

            metrics["mse"] = torch.tensor(
                [compute_mse(temp_prop_post, prop_gt) for temp_prop_post in prop_posts]
            )

            metrics["mse_normalized"] = torch.tensor(
                [compute_mse(temp_prop_post, prop_gt, normalized=True) for temp_prop_post in prop_posts]
            )

            metrics["nlpd"] = torch.tensor(
                [compute_nlpd(temp_prop_post, prop_gt) for temp_prop_post in prop_posts]
            )

            metrics["nlpd_noisy"] = torch.tensor(
                [
                    compute_nlpd(temp_prop_post, prop_gt)
                    for temp_prop_post in prop_posts_noisy
                ]
            )

            for calibration_level in [0.683, 0.955, 0.997]:
                metrics[f"surr_calibration_{int(calibration_level*100)}"] = torch.tensor(
                    [compute_calibration(test_predictions[temp_i], 
                                  test_labels[temp_i], 
                                  calibration_level) 
                                  for temp_i in range(len(test_predictions))]
                )

            # Benchmarking as a classification problem
            if isinstance(application, TabRepoBenchmarkApplication):
                #     and not isinstance(
                #     surrogate_cfg.fit_config, NUTSConfig
                # ):  # Not possible for fully Bayesian

                # Binary target (indicating whether competitor is better than incumbent)
                prop_gt_binary = (prop_gt.squeeze() > 0).double()

                def get_predicted_prob(temp_prop_post):
                    if temp_prop_post.covariance_matrix.ndim == 3:
                        # Gaussian mixture case
                        mixture_components = Normal(
                            loc=temp_prop_post.mean.squeeze(),
                            scale=torch.sqrt(
                                temp_prop_post.covariance_matrix.squeeze(-1, -2)
                            ),
                        )

                        mixture_weights = Categorical(
                            probs=torch.ones(mixture_components.loc.shape[0])
                            / mixture_components.loc.shape[0]
                        )

                        gmm = MixtureSameFamily(mixture_weights, mixture_components)

                        return 1 - gmm.cdf(torch.tensor(0))

                    else:
                        return 1 - Normal(
                            loc=temp_prop_post.mean.squeeze(),
                            scale=torch.sqrt(
                                temp_prop_post.covariance_matrix.squeeze()
                            ),
                        ).cdf(torch.tensor(0))

                    ## Only works for 1 difference in benchmarking

                # Predicted probability per iteration
                predicted_probs = [
                    get_predicted_prob(temp_prop_post) for temp_prop_post in prop_posts
                ]

                # Predicted probability computed using noisy GP predictions per iteration
                predicted_probs_noisy = [
                    get_predicted_prob(temp_prop_post)
                    for temp_prop_post in prop_posts_noisy
                ]

                metrics["ce_loss"] = torch.tensor(
                    [
                        compute_ce_loss(temp_predicted_prob, prop_gt_binary)
                        for temp_predicted_prob in predicted_probs
                    ]
                )

                metrics["ce_loss_noisy"] = torch.tensor(
                    [
                        compute_ce_loss(temp_predicted_prob, prop_gt_binary)
                        for temp_predicted_prob in predicted_probs_noisy
                    ]
                )

                metrics["accuracy"] = torch.tensor(
                    [
                        compute_accuracy(temp_predicted_prob, prop_gt_binary)
                        for temp_predicted_prob in predicted_probs
                    ]
                )

                metrics["accuracy_noisy"] = torch.tensor(
                    [
                        compute_accuracy(temp_predicted_prob, prop_gt_binary)
                        for temp_predicted_prob in predicted_probs_noisy
                    ]
                )

            if meta_cfg.time_ops:
                metrics["hp_fit_duration"] = torch.tensor(hp_fit_durations)
                metrics["prop_post_duration"] = torch.tensor(prop_post_durations)
                metrics["acq_fun_duration"] = torch.tensor(acq_fun_durations)
                metrics["peak_rss_mb"] = torch.tensor(peak_rss_mb)



        else:

            class SV_Approx:
                def __init__(self, mean):
                    self.mean = torch.tensor(mean)
                    self.covariance_matrix = torch.zeros(mean.shape[0], mean.shape[0])

            metrics["mae"] = torch.tensor(
                [
                    compute_mae(SV_Approx(temp_sv_approx), prop_gt)
                    for temp_sv_approx in sv_approximations
                ]
            )

            metrics["mse"] = torch.tensor(
                [
                    compute_mse(SV_Approx(temp_sv_approx), prop_gt)
                    for temp_sv_approx in sv_approximations
                ]
            )

            metrics["mse_normalized"] = torch.tensor(
                [
                    compute_mse(SV_Approx(temp_sv_approx), prop_gt, normalized=True)
                    for temp_sv_approx in sv_approximations
                ]
            )

            if meta_cfg.time_ops and acq_fun_durations is not None:
                metrics["acq_fun_duration"] = torch.tensor(acq_fun_durations)

    # ------------------------------------------------------------------
    # Add further metrics to be persisted
    # ------------------------------------------------------------------
    metrics["application"] = application.__class__.__name__
    metrics["blackbox"] = blackbox_fn.plot_name
    metrics["run_id"] = run_id  # Unique identifier
    metrics["seed"] = int(torch.asarray(meta_cfg.seed))  # needs a python int
    metrics["acquisition"] = acquisition_fn.plot_name

    metrics["initial_design_size"] = application.X0.shape[0]

    if not isinstance(acquisition_fn, SHAPIQAcquisitionFunction):
        metrics["archive_x"] = archive_x.tolist()
        metrics["archive_y"] = archive_y.tolist()

        metrics["prop_post_means"] = torch.stack(
            [temp_prop_post.mean for temp_prop_post in prop_posts]
        ).tolist()
        metrics["prop_post_covars"] = torch.stack(
            [temp_prop_post.covariance_matrix for temp_prop_post in prop_posts]
        ).tolist()

        metrics["prop_post_means_noisy"] = torch.stack(
            [temp_prop_post.mean for temp_prop_post in prop_posts_noisy]
        ).tolist()
        metrics["prop_post_covars_noisy"] = torch.stack(
            [temp_prop_post.covariance_matrix for temp_prop_post in prop_posts_noisy]
        ).tolist()

        if meta_cfg.debug_mode and blackbox_fn.is_pseudo_expensive:
            metrics["mae"] = metrics["mae"].tolist()
            metrics["mse"] = metrics["mse"].tolist()
            metrics["mse_normalized"] = metrics["mse_normalized"].tolist()

            # if sv_approximations is not None:
            #     metrics["mae_siq"]= metrics["mae_siq"].tolist()

            metrics["nlpd"] = metrics["nlpd"].tolist()
            metrics["nlpd_noisy"] = metrics["nlpd_noisy"].tolist()

            #metrics["surr_msll"] = metrics["surr_msll"].tolist()


            for calibration_level in [0.683, 0.955, 0.997]:
                metrics[f"surr_calibration_{int(calibration_level*100)}"] = metrics[f"surr_calibration_{int(calibration_level*100)}"].tolist()

            if isinstance(application, TabRepoBenchmarkApplication):
                #     and not isinstance(
                #     surrogate_cfg.fit_config, NUTSConfig
                # ):
                metrics["ce_loss"] = metrics["ce_loss"].tolist()
                metrics["ce_loss_noisy"] = metrics["ce_loss_noisy"].tolist()
                metrics["accuracy"] = metrics["accuracy"].tolist()
                metrics["accuracy_noisy"] = metrics["accuracy_noisy"].tolist()

            if meta_cfg.time_ops:
                metrics["hp_fit_duration"] = metrics["hp_fit_duration"].tolist()
                metrics["prop_post_duration"] = metrics["prop_post_duration"].tolist()
                metrics["acq_fun_duration"] = metrics["acq_fun_duration"].tolist()
                metrics["peak_rss_mb"] = metrics["peak_rss_mb"].tolist()

    else:
        metrics["mae"] = metrics["mae"].tolist()
        metrics["mse"] = metrics["mse"].tolist()
        metrics["mse_normalized"] = metrics["mse_normalized"].tolist()

        if metrics["acq_fun_duration"] is not None:
            metrics["acq_fun_duration"] = metrics["acq_fun_duration"].tolist()

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


# -----------------------------------------------------------------------------  #
#  Entrypoint for `python -m`                                                   #
# -----------------------------------------------------------------------------  #
if __name__ == "__main__":
    """
    To run a single deterministic job:

        python -m xac.experiments.cli -m \
            meta.seed=1 \
            acquisition._target_=xac.acquisition_functions.EIGExecutionPath

    To sweep two seeds × two acquisition functions (serial on one core):

        python -m xac.experiments.cli -m \
            meta.seed=1,2 \
            acquisition._target_=xac.acquisition_functions.EIGExecutionPath,\
                                  xac.acquisition_functions.EIGFunctionProperty
    """
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        main()

