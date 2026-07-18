from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import gpytorch
from gpytorch.distributions import MultivariateNormal
import torch

from xac.acquisition_functions import (KernelSHAPSampler, LeverageGPSampler,
                                       SHAPIQAcquisitionFunction, SVARMSampler)
from xac.acquisition_optimizers import (BaseAcquisitionOptimizer, Exhaustive,
                                        Subset)
from xac.applications import BaseApplication, ShapleyApplication
from xac.applications.applications import ShapiqShapleyApplication, YahpoShapleyApplication, DummyShapleyApplication
from xac.blackbox_functions import BaseBlackboxFunction, ShapleyTreeGameBBF
from xac.experiments import MetaExperimentConfig
from xac.surrogates import ConstantNoiseConfig
from xac.surrogates.gp_surrogate import (GPSurrogate, GPSurrogateConfig,
                                         NUTSConfig)
from xac.utils.plotting import plot_data_hists

log = logging.getLogger(__name__)


def _get_peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------
def run_experiment(
    application: BaseApplication,
    blackbox_fn: BaseBlackboxFunction,
    surrogate_cfg: GPSurrogateConfig,
    acquisition_fn: Callable[[torch.Tensor, GPSurrogate], torch.Tensor],
    acquisition_optimizer: BaseAcquisitionOptimizer,
    ed_cfg: ExperimentalDesignConfig,
    meta_cfg: MetaExperimentConfig,
    run_dir: str,
) -> Tuple[torch.distributions.MultivariateNormal, Tuple[torch.Tensor, torch.Tensor]]:
    """Run a sequential experiment.

    Parameters
    ----------
    application :
        Provides execution-path ``Z``, affine matrix ``A``, initial design
        ``X0`` and the termination criterion.
    blackbox_fn :
        The expensive function being queried.
    surrogate_cfg :
        Settings for :class:`GPSurrogate`.
    acquisition_fn :
        Callable ``acq(x_candidates, surrogate) → utility`` that must accept a
        `(M × D)` tensor of *candidate points* and return a `(M,)` utility
        tensor.

    Returns
    -------
    property_posterior :
        Posterior over the property of interest after termination.
    archive :
        Tuple ``(archive_X, archive_Y)`` containing all evaluated points.
    """

    # -----------------------------------------------------------------
    if application.lazy_setup:
        application.run_lazy_setup(
            blackbox_fn,
            seed=meta_cfg.seed,
            amount_iterations=ed_cfg.iterations if ed_cfg.iterations else None,
            scalable_mode=meta_cfg.scalability_mode,
            acquisition_optimizer=acquisition_optimizer,
        )

    # 0) evaluate initial design
    archive_X = application.X0
    archive_Y = (
        application.Y0[0] if hasattr(application, "Y0") else blackbox_fn(archive_X)[0]
    )

    # build a partial constructor with frozen config
    PartialGPSurrogate = partial(
        GPSurrogate,
        config=surrogate_cfg,
        cat_dims=blackbox_fn.cat_dims,  # Add for PDP, previously None for PDP
        log_trafo_dims=blackbox_fn.log_trafo_dims,
        bounds=blackbox_fn.bounds,
        shapley_configs=(
            None
            if not isinstance(application, ShapleyApplication)
            else (application.baseline_config, application.candidate_config)
        ),
    )

    #Set prop GT based on GP surrogate for 
    #Check if this is true
    if (isinstance(application, YahpoShapleyApplication) or isinstance(application, ShapiqShapleyApplication) or isinstance(application, DummyShapleyApplication)) and meta_cfg.scalability_mode and application.amount_players < 20:
        temp_gp = PartialGPSurrogate(archive_X.clone(), archive_Y.clone())

        exact_siq_values = application.get_exact_siq_values(
            archive_X.shape[0], blackbox_fn, surrogate=temp_gp
        )

        object.__setattr__(application, "prop_gt", torch.tensor(exact_siq_values).unsqueeze(dim=1))


    # -----------------------------------------------------------------
    # 1) determine candidate set (only points in Z not yet evaluated)
    if application.eval_bb_only_on_Z:
        candidate_set = application.candidate_set

        if not meta_cfg.scalability_mode:
            candidate_idx_Z = application.candidate_idx_Z

    else:
        raise NotImplementedError("General case not implemented yet.")

    # Tracka NOT: candidate_idx_Z

    iterations = (
        ed_cfg.iterations if ed_cfg.iterations else (candidate_set.shape[0] - 1)
    )
    assert iterations <= candidate_set.shape[0]

    # -----------------------------------------------------------------
    # 2) debug mode
    if meta_cfg.debug_mode and blackbox_fn.is_pseudo_expensive:
        prop_gt = application.prop_gt

    # -----------------------------------------------------------------
    # 3) sequential loop
    prop_posts = []
    prop_posts_noisy = []
    hp_fit_durations = []
    prop_post_durations = []
    acq_fun_durations = []
    peak_rss_mb = []

    #Rebuttal extras
    test_predictions = []
    test_labels = []
    train_labels = []

    if isinstance(acquisition_fn, SHAPIQAcquisitionFunction):
        sv_approximations = []

        # if isinstance(acquisition_fn, KernelSHAPSampler):
        #     siq_approximator= 'KernelSHAPSampler'

    assert not gpytorch.settings.fast_computations.covar_root_decomposition.on()
    assert not gpytorch.settings.fast_computations.log_prob.on()
    assert not gpytorch.settings.fast_computations.solves.on()
    assert not (gpytorch.settings.max_cholesky_size._global_value < 4096)
    assert not gpytorch.settings.fast_pred_var.on()

    for iteration_idx in range(iterations):
        #log.info(f"""Iteration {str(iteration_idx)}""")

        #Handle whether GP HPs are refitted (and by extension if ShaplEIG computation is conducted from scratch)
        is_no_refit_step = False

        if surrogate_cfg.fit_config.refit_schedule is not None:
            #If refit_schedule is not None, is_no_refit_step is ignored
            if surrogate_cfg.fit_config.refit_schedule == "init_64_factor_4":
                #In the first 64 iterations, refit every iteration, then refit every 8th iteration in the next 128 iterations, then refit every 16th iteration in the next 256 iterations, and then refit every 32th iteration afterwards.
                if iteration_idx < 64:
                    is_no_refit_step = False
                elif iteration_idx < 64 + 128:
                    is_no_refit_step = (iteration_idx % 8 != 0)
                elif iteration_idx < 64 + 128 + 256:
                    is_no_refit_step = (iteration_idx % 16 != 0)
                else:
                    is_no_refit_step = (iteration_idx % 32 != 0)

            else:
                raise ValueError(f"Unknown refit schedule.")
            
        else:
            #Only is refit_schedule is None
            is_no_refit_step= (hasattr(surrogate_cfg.fit_config, "refit_interval") and 
                                (surrogate_cfg.fit_config.refit_interval > 1) and 
                                (iteration_idx % surrogate_cfg.fit_config.refit_interval != 0))

        # print(iteration_idx)
        # print(is_no_refit_step)
        if not is_no_refit_step:
            log.info(f"""Iteration {str(iteration_idx)}""")

        if not isinstance(acquisition_fn, SHAPIQAcquisitionFunction):
            if iteration_idx == 0:
                gp = PartialGPSurrogate(archive_X, archive_Y)

                if (
                    isinstance(application, ShapleyApplication)
                    and not meta_cfg.scalability_mode
                ):
                    assert torch.equal(
                        application._candidate_set_binary,
                        gp._model.input_transform.transform(candidate_set),
                    ), "Error in cont to binary mapping."

                    assert torch.equal(
                        application._Z_binary,
                        gp._model.input_transform.transform(application._Z),
                    ), "Error in cont to binary mapping."
                    # print("hi")

            else:
                # if (
                #     hasattr(gp.config.fit_config, "warmstart")
                #     and gp.config.fit_config.warmstart
                # ):
                if (
                    hasattr(gp.config.fit_config, "warmstart")
                    and gp.config.fit_config.warmstart
                ) or (is_no_refit_step):
                    # Warmstart: Update data in existing model (leads to warmstart of hyperparameters)
                    gp.update_data(archive_X, archive_Y)
                    #Caution: After updating_data, gp._model.train_targets is not standardized anymore
                    #Standardization module is not updated (gp._model.outcome_transform.means)
                    #ALso surrogate._model.likelihood.noise is not updated
                    #=> Solved

                    #Alternative:
                    # temp_noise= gp._model.likelihood.noise[0].unsqueeze(dim=0).unsqueeze(dim=0)
                    # temp_model= gp._model.condition_on_observations(X= new_x, Y= new_y, noise= temp_noise)
                    # But this does not update standardization module

                else:
                    # Coldstart: Reinitialize model with new data (leads to random hyperparameters)
                    gp = PartialGPSurrogate(archive_X, archive_Y)

            # if not meta_cfg.skip_fitting:
            if (not meta_cfg.skip_fitting) and not (
                is_no_refit_step
            ):

                if meta_cfg.time_ops:
                    start_hp_fit = time.perf_counter()

                gp.fit()  # Returns model in eval-mode;  # Applies common random numbers in AF optimization

                if meta_cfg.time_ops:
                    end_hp_fit = time.perf_counter()
                    hp_fit_durations.append(end_hp_fit - start_hp_fit)

                # Ensure that GP is function interpolator
                # gp.forward(gp._model.train_inputs[0]).mean
                # gp._model.train_targets

                # Ensure that learnable hyperparameters are within specified bounds
                # (Default behavior of BoTorch does not guarantee this. See https://github.com/meta-pytorch/botorch/issues/2542))
                noise = gp._model.likelihood.noise
                lengthscales = gp._model.covar_module.base_kernel.lengthscale

                # Assert that noise and lengthscales are within bounds
                # First compare the datatypes, if they are different, cast the bound to the same type as noise.data
                if not isinstance(
                    surrogate_cfg.fit_config, NUTSConfig
                ) and not isinstance(surrogate_cfg.noise_config, ConstantNoiseConfig):
                    if (
                        noise.data.dtype
                        != torch.tensor(
                            surrogate_cfg.noise_config.min_inferred_noise_level
                        ).dtype
                    ):
                        bound_casted = (
                            torch.tensor(
                                surrogate_cfg.noise_config.min_inferred_noise_level
                            )
                            .to(noise.device)
                            .to(noise.data.dtype)
                        )
                    else:
                        bound_casted = torch.tensor(
                            surrogate_cfg.noise_config.min_inferred_noise_level
                        ).to(noise.device)
                    assert torch.all(
                        noise.data >= bound_casted
                    ), "Noise below lower bound after fitting."

                    if (
                        lengthscales.data.dtype
                        != torch.tensor(
                            surrogate_cfg.kernel_config.min_lengthscale
                        ).dtype
                    ):
                        bound_casted_ls = (
                            torch.tensor(surrogate_cfg.kernel_config.min_lengthscale)
                            .to(lengthscales.device)
                            .to(lengthscales.data.dtype)
                        )
                    else:
                        bound_casted_ls = torch.tensor(
                            surrogate_cfg.kernel_config.min_lengthscale
                        ).to(lengthscales.device)
                    assert torch.all(
                        lengthscales.data >= bound_casted_ls
                    ), "Lengthscale below lower bound after fitting."

                # Caution: No bounds could be specified for fully bayesian variant

            else:
                hp_fit_durations.append(0)



            # -----------------------------------------------------------------
            # compute acquisition function
            # (move this before property posterior for better tracking of acquisition function runtimes
            if meta_cfg.time_ops:
                start_acq_func = time.perf_counter()


            util = acquisition_fn(
                candidate_set,
                candidate_idx_Z if not meta_cfg.scalability_mode else None,
                gp,
                application,
                iteration_idx,
                meta_cfg.scalability_mode,
                is_no_refit_step,
                new_x if is_no_refit_step else None, #Pass selected candidate from previous iteration if HPs are not refitted
                best_idx if is_no_refit_step else None,
            )  # (M,)

            # gp._model.covar_module(candidate_set[0:1], candidate_set[1:2]).to_dense()
            # gp._model.covar_module(candidate_set[0:1], candidate_set[2:3]).to_dense()
            # gp._model.covar_module(candidate_set[1:2], candidate_set[2:3]).to_dense()

            if meta_cfg.time_ops:
                end_acq_func = time.perf_counter()
                acq_fun_durations.append(end_acq_func - start_acq_func)
                peak_rss_mb.append(_get_peak_rss_mb())

            # -----------------------------------------------------------------
            # compute property posterior
            if meta_cfg.time_ops:
                start_prop_post = time.perf_counter()

            prop_post = application.property_posterior(gp, noisy_variant=False)
            #Caution: Prop post timing not accurate, as some components are recycled and HP fitting is not included

            if meta_cfg.time_ops:
                end_prop_post = time.perf_counter()
                prop_post_durations.append(end_prop_post - start_prop_post)
                peak_rss_mb.append(_get_peak_rss_mb())

            prop_posts.append(prop_post)

            prop_posts_noisy.append(
                prop_post
            )  

            #Rebuttal extras: Track test predictions and labels for each iteration
            test_labels.append(blackbox_fn.evaluate(candidate_set)[0])
            #train_labels.append(blackbox_fn.evaluate(archive_X)[0])
            temp_preds_= gp.forward(candidate_set)

            test_predictions.append(
                (temp_preds_.mean.squeeze(),
                 temp_preds_.variance.squeeze())
            ) #Add as tuple and not as distribution to avoid memory issues (as covariance of MVN would be diagonal)
            # test_predictions.append(MultivariateNormal(
            #     mean= temp_preds_.mean.squeeze(),
            #     covariance_matrix= temp_preds_.variance.squeeze().diag()
            # ))


            # -----------------------------------------------------------------
            # termination check
            if application.termination_criterion(prop_post) and not meta_cfg.debug_mode:
                break
                # In this case, the GP does not need to be refitted again (as no new data is added)
                # In contrast, if the last iteration is reached it should be fitted again

            # -----------------------------------------------------------------
            # optimize acquisition function
            best_idx = torch.argmax(util)

            new_x = candidate_set[best_idx : best_idx + 1]  # keep shape (1 × D)
            new_y = blackbox_fn(new_x)[0]

            # templ= []
            # for i in range(1000):
            #     util = acquisition_fn(candidate_set, candidate_idx_Z, gp, application)  # (M,)
            #     best_idx = torch.argmax(util)
            #     templ.append(gp._model.input_transform.transform(candidate_set[best_idx : best_idx + 1]).sum().item())
            # freq = {}
            # for v in templ:
            #     freq[v] = freq.get(v, 0) + 1

            # # Print table sorted by key
            # for k in sorted(freq):
            #     print(f"{k}: {freq[k]}")

            # update archive
            # gp._model.posterior(archive_X).mean - archive_Y
            archive_X = torch.cat([archive_X, new_x], dim=0)
            archive_Y = torch.cat([archive_Y, new_y], dim=0)

            # remove chosen candidate from the set
            if candidate_set.shape[0] > 0:
                candidate_set = torch.cat(
                    [candidate_set[:best_idx], candidate_set[best_idx + 1 :]]
                )

                if not meta_cfg.scalability_mode:
                    candidate_idx_Z = torch.cat(
                        [
                            candidate_idx_Z[:best_idx]
                            .detach()
                            .clone(),  # torch.tensor(candidate_idx_Z[:best_idx]),
                            candidate_idx_Z[best_idx + 1 :]
                            .detach()
                            .clone(),  # torch.tensor(candidate_idx_Z[best_idx + 1 :])
                        ]
                    )

        else:
            # Compute shapley estimates with SHAPIQ acquisition function
            gp = PartialGPSurrogate(archive_X, archive_Y)
            # Requires GP object for mapping between continuous and binary inputs

            if isinstance(acquisition_fn, LeverageGPSampler):
                # Hybrid case (Fit GP surrogate on samples from LeverageSHAP)
                sv_approximations.append(
                    application.get_levgp_siq_value(
                        application.init_design_size + iteration_idx,
                        blackbox_fn,
                        partial_gp=PartialGPSurrogate,
                        acquisition_fn_name=acquisition_fn.__class__.__name__,
                        temp_gp=gp,
                    )
                )

            else:

                if meta_cfg.time_ops:
                    start_acq_func = time.perf_counter()

                sv_approximations.append(
                    application.get_siq_values(
                        application.init_design_size + iteration_idx,
                        blackbox_fn,
                        surrogate=gp,
                        acquisition_fn_name=acquisition_fn.__class__.__name__,
                    )
                )

                if meta_cfg.time_ops:
                    end_acq_func = time.perf_counter()
                    acq_fun_durations.append(end_acq_func - start_acq_func)

    # -----------------------------------------------------------------
    if iteration_idx == (iterations - 1):
        if not isinstance(acquisition_fn, SHAPIQAcquisitionFunction) and not (
            application.termination_criterion(prop_post) and not meta_cfg.debug_mode
        ):
            # Case: Procedure has not been terminated due to criterion, but as all iterations have been reached.
            # Compute prop post once again for newly added data.
            # gp = PartialGPSurrogate(archive_X, archive_Y)

            if (
                hasattr(gp.config.fit_config, "warmstart")
                and gp.config.fit_config.warmstart
            ) or (is_no_refit_step):
                # Warmstart: Update data in existing model (leads to warmstart of hyperparameters)
                gp.update_data(archive_X, archive_Y)

            else:
                # Coldstart: Reinitialize model with new data (leads to random hyperparameters)
                gp = PartialGPSurrogate(archive_X, archive_Y)

            if (not meta_cfg.skip_fitting) and not (
                is_no_refit_step
            ):
                gp.fit()

            else:
                print("waut")

            if meta_cfg.time_ops:
                start_prop_post = time.perf_counter()

            prop_post_final = application.property_posterior(gp, noisy_variant=False, recycle= False)
            #Do not recycle, as acquisition optimization has not been conduced and hence some components of the property posterior are not up to date

            if meta_cfg.time_ops:
                end_prop_post = time.perf_counter()
                prop_post_durations.append(end_prop_post - start_prop_post)
                peak_rss_mb.append(_get_peak_rss_mb())

            prop_posts.append(prop_post_final)

            # Quick fix: ignore noisy
            prop_posts_noisy.append(prop_post_final)  # True))

            # if isinstance(application, ShapleyApplication):
            #     sv_approximations.append(application.get_siq_values(archive_X.shape[0], blackbox_fn, surrogate= gp))

            # if meta_cfg.debug_mode and blackbox_fn.is_pseudo_expensive:
            #     if isinstance(application, ShapleyApplication):
            #         #Compute ground-truth Shapley values according to ShapIQ
            #         exact_siq_values= application.get_exact_siq_values(archive_X.shape[0], blackbox_fn, surrogate= gp)

            #         assert torch.allclose(prop_gt.squeeze(), torch.tensor(exact_siq_values), atol=1e-3), "Shapley linear-functional ground truth does not match exact ShapIQ computation."

        else:
            if isinstance(acquisition_fn, LeverageGPSampler):
                # Hybrid case (Fit GP surrogate on samples from LeverageSHAP)
                sv_approximations.append(
                    application.get_levgp_siq_value(
                        application.init_design_size + iteration_idx + 1,
                        blackbox_fn,
                        partial_gp=PartialGPSurrogate,
                        acquisition_fn_name=acquisition_fn.__class__.__name__,
                        temp_gp=gp,
                    )
                )

            else:
                sv_approximations.append(
                    application.get_siq_values(
                        application.init_design_size
                        + iteration_idx
                        + 1,  # archive_X.shape[0] + iteration_idx + 1,
                        blackbox_fn,
                        surrogate=gp,
                        acquisition_fn_name=acquisition_fn.__class__.__name__,
                    )
                )

            if meta_cfg.debug_mode and blackbox_fn.is_pseudo_expensive:
                # Compute ground-truth Shapley values according to ShapIQ
                if (
                    not meta_cfg.scalability_mode
                ):  # Only compute in non-scalability mode
                    exact_siq_values = application.get_exact_siq_values(
                        archive_X.shape[0], blackbox_fn, surrogate=gp
                    )

                    assert torch.allclose(
                        prop_gt.squeeze(), torch.tensor(exact_siq_values), atol=1e-3
                    ), "Shapley linear-functional ground truth does not match exact ShapIQ computation."

    if not torch.allclose(gp.forward(archive_X).mean, archive_Y.squeeze(), atol=1e-3):
        log.warning(
            f"Final GP surrogate is not interpolating the observed data points within tolerance. Abs diff: {(gp.forward(archive_X).mean - archive_Y.squeeze()).abs().max()}"
        )

    if isinstance(application, ShapleyApplication) and not (isinstance(
        blackbox_fn, ShapleyTreeGameBBF
    ) or meta_cfg.scalability_mode
    ):  #        #Does not really make sense for tree, as there shapiq lienar functional is taken as ground truth
        # if not meta_cfg.scalability_mode: #in scalability mode do not overwrite exact SVs
        exact_siq_values = application.get_exact_siq_values(
            archive_X.shape[0], blackbox_fn, surrogate=gp
        )
        assert torch.allclose(
            prop_gt.squeeze(),
            torch.tensor(exact_siq_values).unsqueeze(dim=0),
            atol=1e-5,
        ), "Shapley linear-functional ground truth does not match exact ShapIQ computation."

        # Set ground truth to Shapiq variant
        prop_gt = torch.tensor(exact_siq_values).unsqueeze(dim=0)

    # -----------------------------------------------------------------

    return (
        (
            prop_posts,
            prop_posts_noisy,
            test_predictions,
            test_labels, #train_labels,
            (
                prop_gt
                if (meta_cfg.debug_mode and blackbox_fn.is_pseudo_expensive)
                else None
            ),
            (
                sv_approximations
                if isinstance(acquisition_fn, SHAPIQAcquisitionFunction)
                else None
            ),
        ),
        (archive_X, archive_Y),
        (
            (hp_fit_durations, prop_post_durations, acq_fun_durations, peak_rss_mb)
            if meta_cfg.time_ops
            else (None, None, None, None)
        ),
    )


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentalDesignConfig:
    iterations: Optional[int] = None
