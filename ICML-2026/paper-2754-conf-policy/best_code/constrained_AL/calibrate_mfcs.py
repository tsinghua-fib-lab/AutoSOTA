"""
Classes and functions for conformal policy control and computing prediction sets
"""

import numpy as np
import time

from abc import ABC

import pandas as pd
from copy import deepcopy


def get_f_std(y_std, gpr):

    params = gpr.kernel_.get_params()

    normalized_noise_var = params["k2__noise_level"]
    y_train_var = gpr._y_train_std**2

    y_pred_var = y_std**2
    f_pred_var = y_pred_var - (y_train_var * normalized_noise_var)
    f_std = np.sqrt(f_pred_var)
    return f_std


def constrained_pdf_gpr_lik_ratio(
    X,
    var_pool_min_max_norm,
    exp_pool_sum_norm,
    gpr_model,
    lmbda,
    pc_param,
    safe_densities_pool,
    safe_densities_cal_test,
    pc_densities_pool,
):
    """
    X                      : dim (n + t, d), array where each row is a data input of dimension d
    var_pool_min_max_norm  : scalar, max(var_pool) - min(var_pool)
    exp_pool_sum_norm      : scalar, np.exp(var_pool_normed * lmbda)
    gpr_model              : Fitted sklearn GPRegression model for time T
    lmbda                  : scalar, temperature/shift magnitude
    """
    if X.ndim != 2:
        raise Exception(
            "X must be array of dimension 2; if X.ndim == 1 and each entry is a different sample, do: X = X.reshape(-1,1); if is one datapoint, do X = X.reshape(1,-1)"
        )

    np.shape(X)[0]

    #### Get *Current, Unconstrained* Densities on inputs X
    ## Get posterior variances on X
    _, std_pred_ = gpr_model.predict(X, return_std=True)
    std_pred = get_f_std(std_pred_, gpr_model)
    var_pred = std_pred**2
    var_pred_normed = var_pred / var_pool_min_max_norm
    pdf_vals_cal_test_unconstrained = (
        np.exp(var_pred_normed * lmbda) / exp_pool_sum_norm
    )

    #### Get *Current, Constrained* Densities on inputs X
    # print(f'pdf_vals_cal_test_unconstrained : {pdf_vals_cal_test_unconstrained}')
    # print(f'safe_densities_cal_test : {safe_densities_cal_test}')

    opt_safe_log_lik_ratios_cal_test = np.log(
        pdf_vals_cal_test_unconstrained / safe_densities_cal_test
    )  ## Log lik-ratios
    pdf_vals_cal_test_constrained = np.where(
        opt_safe_log_lik_ratios_cal_test < pc_param,
        pdf_vals_cal_test_unconstrained,
        np.exp(pc_param) * np.array(safe_densities_cal_test),
    )  ## constrained PDFs (not normalized yet)
    pdf_vals_cal_test_constrained_normalized = pdf_vals_cal_test_constrained / np.sum(
        pc_densities_pool
    )  ## Normalized constrained PDFs

    return pdf_vals_cal_test_constrained_normalized


def conformal_policy_control(
    X_cal,
    Feasible_cal,
    mixture_weights,
    var_pool_min_max_norm,
    exp_var_preds_pool_split,
    source_densities_cal,
    source_densities_pool,
    gpr_model,
    lmbda,
    constrained_densities_cal_running_list,
    constrained_densities_pool_running_list,
    alpha=0.5,
    constrain_vs_init="Y",
    max_weight_test=True,
    n_grid=200,
):
    """
    X_cal                     : dim (n + t - 1, d), array where each row is a data input of dimension d
    Feasible_cal              : dim (n + t - 1), array of Feasible set indicators for cal data (not test data)
    var_pool_min_max_norm     : scalar, max(var_pool) - min(var_pool)
    exp_var_preds_pool_split  : dim(n_pool), array of 'np.exp(var_pool_normed * lmbda)' values for pool data
    gpr_model                 : Fitted sklearn GPRegression model for time T
    lmbda                     : scalar, temperature/shift magnitude
    source_pdf                : source density function
    """
    if X_cal.ndim != 2:
        raise Exception(
            "X must be array of dimension 2; if X.ndim == 1 and each entry is a different sample, do: X = X.reshape(-1,1); if is one datapoint, do X = X.reshape(1,-1)"
        )

    np.shape(X_cal)[0]

    if constrain_vs_init == "Y":
        ## Constrain relative to initial safe policy
        idx_safe = 0
        test_pt_factor = 1
    else:
        ## Constrain relative to most recent policy
        ## NOTE: Current guarantees may break in this case, as is assuming that most recent policy is safe asymptotically
        idx_safe = -1
        test_pt_factor = 2

    ## Check risk level in source distribution:
    source_densities_pool_max = np.max(source_densities_pool)
    source_weights_cal_pool_max = np.concatenate(
        (source_densities_cal, [source_densities_pool_max])
    )
    source_weights_cal_pool_max_normalized = source_weights_cal_pool_max / np.sum(
        source_weights_cal_pool_max
    )

    if (
        np.sum(source_densities_cal[Feasible_cal == 0]) + source_densities_pool_max
        > alpha
    ):
        # raise Exception("Feasible set risk must be controlled in source distribution to begin")
        print(
            "Warning: Feasible set risk not controlled in source distribution to begin"
        )
        return source_weights_cal_pool_max_normalized, 1

    #### On Pool Set: ####
    ## 1. Get *Current, Unconstrained* (for time t) densities and normalization constant
    norm_constant_pool = np.sum(exp_var_preds_pool_split)
    opt_densities_pool = exp_var_preds_pool_split / norm_constant_pool

    ## 2. Get Mixture for Previous (for times 1:(t-1)) Constrained Densities
    constrained_densities_pool_running_arr = np.array(
        constrained_densities_pool_running_list
    )
    Tmin1_mixture_pool = mixture_pdf_from_densities_mat(
        constrained_densities_pool_running_arr, mixture_weights
    )

    #### On Cal Set: ####
    ## 1. Get *Current, Unconstrained* (for time t) Densities
    _, std_cal_ = gpr_model.predict(X_cal, return_std=True)
    std_cal = get_f_std(std_cal_, gpr_model)
    var_cal = std_cal**2
    var_cal_normed = var_cal / var_pool_min_max_norm
    opt_densities_cal = np.exp(var_cal_normed * lmbda) / norm_constant_pool

    ## 2. Get *Previous, Constrained Mixture* (for times 1:(t-1)) Densities
    constrained_densities_cal_running_arr = np.array(
        constrained_densities_cal_running_list
    )
    Tmin1_mixture_cal = mixture_pdf_from_densities_mat(
        constrained_densities_cal_running_arr, mixture_weights
    )

    ## 3. Get lik-ratios for optimized/safe policies.
    opt_safe_log_lik_ratios_pool = np.log(
        opt_densities_pool / constrained_densities_pool_running_list[idx_safe]
    )
    opt_safe_log_lik_ratios_cal = np.log(
        opt_densities_cal / np.array(constrained_densities_cal_running_list[idx_safe])
    )

    ## 4. Initialization for policy control search
    max_log_lik_ratio = max(opt_safe_log_lik_ratios_pool)
    min_log_lik_ratio = min(opt_safe_log_lik_ratios_pool)
    pc_params_grid = np.linspace(
        min_log_lik_ratio, max_log_lik_ratio, num=n_grid
    )  # [::-1]

    ## 5. Loop to find the most aggressive risk-controlling parameter
    for p_i, pc_param in enumerate(pc_params_grid):
        ## For Unknown Test Point: Compute max(Current Constrained Density / Previous Mixture Density)
        constrained_pool_unnormalized = np.where(
            opt_safe_log_lik_ratios_pool < pc_param,
            opt_densities_pool,
            np.exp(pc_param)
            * np.array(constrained_densities_pool_running_list[idx_safe]),
        )
        constrained_pool_sum = np.sum(constrained_pool_unnormalized)
        constrained_densities_pool = (
            constrained_pool_unnormalized / constrained_pool_sum
        )
        constrained_over_mixture_lik_ratio_pool = (
            constrained_densities_pool / Tmin1_mixture_pool
        )

        if max_weight_test:
            ## Default condition (assumed in theory), conservatively over-estimate weight of unknown test point by maximum
            max_constrained_over_mixture_lik_ratio_pool = test_pt_factor * np.max(
                constrained_over_mixture_lik_ratio_pool
            )

        else:
            max_constrained_over_mixture_lik_ratio_pool = test_pt_factor * np.sum(
                constrained_over_mixture_lik_ratio_pool * constrained_densities_pool
            )

        ## For Cal Points: Compute (Current Constrained Density / Previous Mixture Density)
        constrained_cal_unnormalized = np.where(
            opt_safe_log_lik_ratios_cal < pc_param,
            opt_densities_cal,
            np.exp(pc_param)
            * np.array(constrained_densities_cal_running_list[idx_safe]),
        )
        constrained_densities_cal = constrained_cal_unnormalized / constrained_pool_sum
        constrained_over_mixture_lik_ratio_cal = (
            constrained_densities_cal / Tmin1_mixture_cal
        )

        ## Compute Conformal Weights (For Cal and Test Data)
        # Unnormalized conformal weights:
        constrained_over_mixture_lik_ratio_cal_test = np.concatenate(
            (
                constrained_over_mixture_lik_ratio_cal,
                [max_constrained_over_mixture_lik_ratio_pool],
            )
        )

        # Normalized conformal weights:
        pc_weights_cal_test_normalized = (
            constrained_over_mixture_lik_ratio_cal_test
            / np.sum(constrained_over_mixture_lik_ratio_cal_test)
        )

        ## If (weighted) risk is not controlled, then stop searching and return most recent safe policy control parameter (and associated quantities)
        if (
            np.sum(pc_weights_cal_test_normalized[:-1][Feasible_cal == 0])
            + pc_weights_cal_test_normalized[-1]
            > alpha
            or p_i == len(pc_params_grid) - 1
        ):
            pc_param_selected = (
                pc_params_grid[p_i - 1] if p_i > 0 else pc_params_grid[0]
            )  # p_i

            ## For Test Point: Compute max(Current Constrained Density / Previous Mixture Density)
            constrained_pool_unnormalized = np.where(
                opt_safe_log_lik_ratios_pool < pc_param_selected,
                opt_densities_pool,
                np.exp(pc_param_selected)
                * np.array(constrained_densities_pool_running_list[idx_safe]),
            )
            constrained_pool_sum = np.sum(constrained_pool_unnormalized)
            constrained_densities_pool = (
                constrained_pool_unnormalized / constrained_pool_sum
            )
            constrained_over_mixture_lik_ratio_pool = (
                constrained_densities_pool / Tmin1_mixture_pool
            )
            if max_weight_test:
                max_constrained_over_mixture_lik_ratio_pool = np.max(
                    constrained_over_mixture_lik_ratio_pool
                )

            else:
                max_constrained_over_mixture_lik_ratio_pool = np.sum(
                    constrained_over_mixture_lik_ratio_pool * constrained_densities_pool
                )

            ## For Cal Points: Compute (Current Constrained Density / Previous Mixture Density)
            constrained_cal_unnormalized = np.where(
                opt_safe_log_lik_ratios_cal < pc_param_selected,
                opt_densities_cal,
                np.exp(pc_param_selected)
                * np.array(constrained_densities_cal_running_list[idx_safe]),
            )
            constrained_densities_cal = (
                constrained_cal_unnormalized / constrained_pool_sum
            )
            constrained_over_mixture_lik_ratio_cal = (
                constrained_densities_cal / Tmin1_mixture_cal
            )

            ## Compute Normalized Weights (For Cal and Test Data)
            constrained_over_mixture_lik_ratio_cal_test = np.concatenate(
                (
                    constrained_over_mixture_lik_ratio_cal,
                    [max_constrained_over_mixture_lik_ratio_pool],
                )
            )

            pc_weights_cal_test_normalized = (
                constrained_over_mixture_lik_ratio_cal_test
                / np.sum(constrained_over_mixture_lik_ratio_cal_test)
            )

            print(f"selected policy control param : {pc_param_selected}")
            break

    return (
        constrained_densities_pool,
        constrained_densities_cal,
        constrained_over_mixture_lik_ratio_cal_test[:-1],
        Tmin1_mixture_pool,
        pc_param_selected,
    )


def compute_risk_control_weights_lik_ratio_preset_beta(
    X_cal,
    Feasible_cal,
    mixture_weights,
    var_pool_min_max_norm,
    exp_var_preds_pool_split,
    source_densities_cal,
    source_densities_pool,
    gpr_model,
    lmbda,
    constrained_densities_cal_running_list,
    constrained_densities_pool_running_list,
    pc_param,
    alpha=0.5,
    constrain_vs_init="Y",
    max_weight_test=True,
    n_grid=200,
):
    """
    For CDT (Lekeufack, et al., 2024) method; assumes policy-control param was updated online outside of loop (retroactively)

    X_cal                     : dim (n + t - 1, d), array where each row is a data input of dimension d
    Feasible_cal              : dim (n + t - 1), array of Feasible set indicators for cal data (not test data)
    var_pool_min_max_norm     : scalar, max(var_pool) - min(var_pool)
    exp_var_preds_pool_split  : dim(n_pool), array of 'np.exp(var_pool_normed * lmbda)' values for pool data
    gpr_model                 : Fitted sklearn GPRegression model for time T
    lmbda                     : scalar, temperature/shift magnitude
    source_pdf                : source density function
    """
    if X_cal.ndim != 2:
        raise Exception(
            "X must be array of dimension 2; if X.ndim == 1 and each entry is a different sample, do: X = X.reshape(-1,1); if is one datapoint, do X = X.reshape(1,-1)"
        )

    np.shape(X_cal)[0]

    if constrain_vs_init == "Y":
        ## Constrain relative to initial safe policy
        idx_safe = 0
        test_pt_factor = 1
    else:
        ## Constrain relative to most recent policy
        idx_safe = -1
        test_pt_factor = 2

    ## Check risk level in source distribution:
    source_densities_pool_max = np.max(source_densities_pool)
    source_weights_cal_pool_max = np.concatenate(
        (source_densities_cal, [source_densities_pool_max])
    )
    source_weights_cal_pool_max_normalized = source_weights_cal_pool_max / np.sum(
        source_weights_cal_pool_max
    )

    if (
        np.sum(source_densities_cal[Feasible_cal == 0]) + source_densities_pool_max
        > alpha
    ):
        # raise Exception("Feasible set risk must be controlled in source distribution to begin")
        print(
            "Warning: Feasible set risk not controlled in source distribution to begin"
        )
        return source_weights_cal_pool_max_normalized, 1

    #### On Pool Set:
    ## 1. Get *Current, Unconstrained* (for time t) Values: (unnormalized) max val, normalization constant, & densities
    norm_constant_pool = np.sum(exp_var_preds_pool_split)
    exp_var_preds_pool_split_normed = exp_var_preds_pool_split / norm_constant_pool

    ## 2. Get *Previous, Constrained Mixture* (for times 1:(t-1)) Densities
    constrained_densities_pool_running_arr = np.array(
        constrained_densities_pool_running_list
    )
    Tmin1_mixture_pool = mixture_pdf_from_densities_mat(
        constrained_densities_pool_running_arr, mixture_weights
    )

    #### On Cal Set:
    ## 1. Get *Current, Unconstrained* (for time t) Densities
    _, std_cal_ = gpr_model.predict(X_cal, return_std=True)
    std_cal = get_f_std(std_cal_, gpr_model)
    var_cal = std_cal**2
    var_cal_normed = var_cal / var_pool_min_max_norm
    opt_densities_cal = np.exp(var_cal_normed * lmbda) / norm_constant_pool

    ## 2. Get *Previous, Constrained Mixture* (for times 1:(t-1)) Densities
    constrained_densities_cal_running_arr = np.array(
        constrained_densities_cal_running_list
    )
    Tmin1_mixture_cal = mixture_pdf_from_densities_mat(
        constrained_densities_cal_running_arr, mixture_weights
    )

    ## Initialization for policy control search
    opt_safe_log_lik_ratios_pool = np.log(
        exp_var_preds_pool_split_normed
        / constrained_densities_pool_running_list[idx_safe]
    )
    opt_safe_log_lik_ratios_cal = np.log(
        opt_densities_cal / np.array(constrained_densities_cal_running_list[idx_safe])
    )

    max_log_lik_ratio = max(opt_safe_log_lik_ratios_pool)
    min_log_lik_ratio = min(opt_safe_log_lik_ratios_pool)
    np.linspace(min_log_lik_ratio, max_log_lik_ratio, num=n_grid)  # [::-1]

    ## For Test Point: Compute max(Current Constrained Density / Previous Mixture Density)
    constrained_pool_unnormalized = np.where(
        opt_safe_log_lik_ratios_pool < pc_param,
        exp_var_preds_pool_split_normed,
        np.exp(pc_param) * np.array(constrained_densities_pool_running_list[idx_safe]),
    )
    constrained_pool_sum = np.sum(constrained_pool_unnormalized)
    constrained_densities_pool = constrained_pool_unnormalized / constrained_pool_sum
    constrained_over_mixture_lik_ratio_pool = (
        constrained_densities_pool / Tmin1_mixture_pool
    )

    if max_weight_test:
        max_constrained_over_mixture_lik_ratio_pool = test_pt_factor * np.max(
            constrained_over_mixture_lik_ratio_pool
        )

    else:
        max_constrained_over_mixture_lik_ratio_pool = test_pt_factor * np.sum(
            constrained_over_mixture_lik_ratio_pool * constrained_densities_pool
        )

    ## For Cal Points: Compute (Current Constrained Density / Previous Mixture Density)
    constrained_cal_unnormalized = np.where(
        opt_safe_log_lik_ratios_cal < pc_param,
        opt_densities_cal,
        np.exp(pc_param) * np.array(constrained_densities_cal_running_list[idx_safe]),
    )
    constrained_densities_cal = constrained_cal_unnormalized / constrained_pool_sum
    constrained_over_mixture_lik_ratio_cal = (
        constrained_densities_cal / Tmin1_mixture_cal
    )

    ## Compute Normalized Weights (For Cal and Test Data)
    constrained_over_mixture_lik_ratio_cal_test = np.concatenate(
        (
            constrained_over_mixture_lik_ratio_cal,
            [max_constrained_over_mixture_lik_ratio_pool],
        )
    )

    (
        constrained_over_mixture_lik_ratio_cal_test
        / np.sum(constrained_over_mixture_lik_ratio_cal_test)
    )

    pc_param_selected = pc_param  # pc_params_grid[p_i - 1] if p_i > 0 else p_i

    ## For Test Point: Compute max(Current Constrained Density / Previous Mixture Density)
    constrained_pool_unnormalized = np.where(
        opt_safe_log_lik_ratios_pool < pc_param_selected,
        exp_var_preds_pool_split_normed,
        np.exp(pc_param_selected)
        * np.array(constrained_densities_pool_running_list[idx_safe]),
    )
    constrained_pool_sum = np.sum(constrained_pool_unnormalized)
    constrained_densities_pool = constrained_pool_unnormalized / constrained_pool_sum
    constrained_over_mixture_lik_ratio_pool = (
        constrained_densities_pool / Tmin1_mixture_pool
    )
    if max_weight_test:
        max_constrained_over_mixture_lik_ratio_pool = np.max(
            constrained_over_mixture_lik_ratio_pool
        )

    else:
        max_constrained_over_mixture_lik_ratio_pool = np.sum(
            constrained_over_mixture_lik_ratio_pool * constrained_densities_pool
        )

    ## For Cal Points: Compute (Current Constrained Density / Previous Mixture Density)
    constrained_cal_unnormalized = np.where(
        opt_safe_log_lik_ratios_cal < pc_param_selected,
        opt_densities_cal,
        np.exp(pc_param_selected)
        * np.array(constrained_densities_cal_running_list[idx_safe]),
    )
    constrained_densities_cal = constrained_cal_unnormalized / constrained_pool_sum
    constrained_over_mixture_lik_ratio_cal = (
        constrained_densities_cal / Tmin1_mixture_cal
    )

    ## Compute Normalized Weights (For Cal and Test Data)
    constrained_over_mixture_lik_ratio_cal_test = np.concatenate(
        (
            constrained_over_mixture_lik_ratio_cal,
            [max_constrained_over_mixture_lik_ratio_pool],
        )
    )

    (
        constrained_over_mixture_lik_ratio_cal_test
        / np.sum(constrained_over_mixture_lik_ratio_cal_test)
    )

    return (
        constrained_densities_pool,
        constrained_densities_cal,
        constrained_over_mixture_lik_ratio_cal_test[:-1],
        Tmin1_mixture_pool,
        pc_param_selected,
    )


def mixture_pdf_from_densities_mat(
    constrained_densities_cal_test_all_steps, mixture_weights
):
    """
    constrained_densities_cal_test_all_steps : dim (t_cal, n_cal + 1) Note: rows correspond to t=0, ..., T-1
    mixture_weights         : dim (T), array of relative weights to put on each of *prior* distributions, from t=0, ..., T-1
                       Note : mixture_weights[0] = n_cal_initial
    """
    mixture_weights_normed = mixture_weights / np.sum(mixture_weights)

    mixture_pdfs = constrained_densities_cal_test_all_steps.T @ mixture_weights_normed

    return mixture_pdfs


## Utilities for computing the factorized likelihood for MFCS Split CP, i.e., the numerator of Eq. (9);
## the recursive implementation here corresponds to Eq. (16) in Appendix B.2.


def compute_w_ptest_split_active_replacement(cal_test_vals_mat, depth_max):
    """
    Computes the estimated MFCS Split CP weights for calibration and test points
    (i.e., numerator in Eq. (9) in main paper or Eq. (16) in Appendix B.2)

    @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
            ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
            ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1

    @param : depth_max          : (int) indicating the maximum recursion depth

    :return: Unnormalized weights on calibration and test points, computed for recursion depth depth_max
    """
    if depth_max < 1:
        raise ValueError(
            "Error: depth_max should be an integer >= 1. Currently, depth_max="
            + str(depth_max)
        )

    if depth_max == 1:
        ##
        return cal_test_vals_mat[-1]

    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(cal_test_vals_mat[-1])
    idx_include = np.repeat(True, n_cal_test)

    for i in range(n_cal_test):
        idx_include[i] = False
        idx_include[i - 1] = True
        summation = compute_w_ptest_split_active_replacement_helper(
            cal_test_vals_mat[:-1, idx_include], depth_max - 1
        )
        # if (i == 0):
        #     print(depth_max, adjusted_vals[i], summation)
        adjusted_vals[i] = adjusted_vals[i] * summation
    return adjusted_vals


def compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat, depth_max):
    """
    Helper function for "compute_w_ptest_split_active_replacement". Computes a summation such as the two sums in the numerator in equation (7) in paper

    @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
            ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
            ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1

    @param : depth_max          : (int) indicating the maximum recursion depth

    :return: Summation such as the two sums in the numerator in equation (7) in paper
    """
    if depth_max == 1:
        return np.sum(cal_test_vals_mat)

    else:
        summation = 0
        n_cal_test = np.shape(cal_test_vals_mat)[1]
        idx_include = np.repeat(True, n_cal_test)
        for i in range(n_cal_test):
            idx_include[i] = False
            idx_include[i - 1] = True
            summation += cal_test_vals_mat[
                -1, i
            ] * compute_w_ptest_split_active_replacement_helper(
                cal_test_vals_mat[:-1, idx_include], depth_max - 1
            )
        return summation


# ========== utilities ==========


def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]

    return [v_sorted, w_sorted]


def weighted_quantile(v, w_normalized, q):
    if len(v) != len(w_normalized):
        raise ValueError(
            "Error: v is length "
            + str(len(v))
            + ", but w_normalized is length "
            + str(len(w_normalized))
        )

    if np.sum(w_normalized) > 1.01 or np.sum(w_normalized) < 0.99:
        raise ValueError("Error: w_normalized does not add to 1")

    if q < 0 or 1 < q:
        raise ValueError("Error: Invalid q")

    len(v)

    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)

    w_sorted_cum = np.cumsum(w_sorted)

    #     cum_w_sum = w_sorted[0]
    i = 0
    while w_sorted_cum[i] < q:
        i += 1

    if q > 0.5:  ## If taking upper quantile: ceil
        #         print("w_sorted_cum[i]",i, v_sorted[i], w_sorted_cum[i])
        return v_sorted[i]

    elif q < 0.5:  ## Elif taking lower quantile:
        if i > 0 and w_sorted_cum[i] == q:
            return v_sorted[i]
        elif i > 0:
            #             print("w_sorted_cum[i-1]",i-1, v_sorted[i-1], w_sorted_cum[i-1])
            return v_sorted[i - 1]
        else:
            return v_sorted[0]

    else:  ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if w_sorted_cum[i] == 0.5:
            return v_sorted[i]

        elif i > 0:
            return (v_sorted[i] * w_sorted[i] + v_sorted[i - 1] * w_sorted[i - 1]) / (
                w_sorted[i] + w_sorted[i - 1]
            )

        else:
            return v_sorted[0]


class SplitConformal(ABC):
    """
    Abstract base class for Split Conformal experiments with black-box predictive model.
    """

    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        """
        :param model: object with predict() method
        :param ptrain_fn: function that outputs likelihood of input under training input distribution, p_X
        :param Xuniv_uxp: (u, p) numpy array encoding all sequences in domain (e.g., all 2^13 sequences
            in Poelwijk et al. 2019 data set), needed for computing normalizing constant
        """
        self.model = model
        self.ptrain_fn = ptrain_fn
        self.Xuniv_uxp = Xuniv_uxp
        self.p = Xuniv_uxp.shape[1]

    def get_normalizing_constant(self, beta_p, lmbda):
        predall_u = self.Xuniv_uxp.dot(beta_p)
        Z = np.sum(np.exp(lmbda * predall_u))
        return Z

    #### Active learning ####
    def compute_confidence_sets_active(
        self,
        Xtrain_split,
        Xcal_split,
        ytrain_split,
        ycal_split,
        Xtest_n1xp_split,
        ytest_n1_split,
        Xpool_split,
        w_split_mus_prev_steps,
        exp_vals_pool_list_of_vecs_all_steps,
        constrained_densities_cal_test_running_list,
        constrained_densities_pool_running_list,
        pc_weights_cal_test_normalized,
        var_pool_min_max_norms,
        exp_pool_sum_norms,
        method_names,
        pc_densities_cal_test,
        pc_densities_pool,
        t_cal,
        X_dataset,
        n_cal_initial,
        alpha_aci_curr,
        weight_bounds,
        source_pdf=None,
        pc_params_list=[None],
        weight_depth_maxes=[1, 2],
        lmbda=1 / 10,
        bandwidth=1.0,
        alpha: float = 0.1,
        n_initial_all=100,
        n_dataset=None,
        replacement=True,
        record_weights=False,
    ):
        # , add_to_cal=True
        if self.p != Xtrain_split.shape[1]:
            raise ValueError(
                "Feature dimension {} differs from provided Xuniv_uxp {}".format(
                    Xtrain_split.shape[1], self.Xuniv_uxp.shape
                )
            )
        #         Xaug_n1xp = np.vstack([Xtrain_nxp, Xtest_1xp])
        Xaug_cal_test_split = np.vstack([Xcal_split, Xtest_n1xp_split])
        # n = ytrain_n.size
        #         n1 = len(Xaug_n1xp) - n ### Temp removed this 20240111, as chnaged split test points to those queried
        n1 = len(Xtest_n1xp_split)
        #         n1 = len(ytest_n1_split)

        ###############################
        # split conformal
        ###############################
        n_cal = len(ycal_split)
        muh_split = self.model.fit(Xtrain_split, ytrain_split)
        muh_split_vals = muh_split.predict(
            np.r_[Xcal_split, Xtest_n1xp_split]
        )  # , std_split_vals
        resids_split = np.abs(ycal_split - muh_split_vals[:n_cal])
        muh_split_vals_testpoint = muh_split_vals[n_cal:]
        ind_split = (np.ceil((1 - alpha) * (n_cal + 1))).astype(int)

        np.concatenate(
            [
                resids_split,
            ]
        )

        PIs_dict = {
            "split": pd.DataFrame(
                np.c_[
                    muh_split_vals_testpoint - np.sort(resids_split)[ind_split - 1],
                    muh_split_vals_testpoint + np.sort(resids_split)[ind_split - 1],
                ],
                columns=["lower", "upper"],
            )
        }

        ###############################
        # Weighted split conformal methods for (one-step and multistep) FCS
        ###############################

        ## Append current ML model (muh function) to list of past models used for querying actively selected calibration points
        ## This list 'w_split_mus_prev_and_curr_steps' will thus allow us to compute the needed query functions p(x | Z_{train}^{(t)})
        deepcopy(w_split_mus_prev_steps[:-1])
        # w_split_mus_prev_steps.append(deepcopy(self.model))
        deepcopy(w_split_mus_prev_steps)

        source_densities_cal_test = source_pdf(Xaug_cal_test_split)

        constrained_density_ratios_cal_test_all_steps = np.array(
            constrained_densities_cal_test_running_list
        ) / np.tile(
            source_densities_cal_test,
            (len(constrained_densities_cal_test_running_list), 1),
        )
        np.array(constrained_densities_cal_test_running_list)
        # print(f'constrained_density_ratios_cal_test_all_steps shape : {constrained_density_ratios_cal_test_all_steps.shape}')

        ## Compute the numerator of Eq. (16) in Appendix B.2
        weights_normalized_wsplit_all = []  ## For recording weights, if want to plot how they change over time

        for depth_max in weight_depth_maxes:
            depth_max_curr = min(depth_max, t_cal)

            ## Construct empirical distribution of scores (initially unweighted)
            positive_infinity = np.array([float("inf")])  ## Conservative adjustment
            unweighted_split_vals = np.concatenate([resids_split, positive_infinity])

            wsplit_quantiles = np.zeros(n1)

            time.time()

            for j in range(0, n1):
                if replacement:
                    ## If sampling with replacement
                    # Z = pool_weights_totals_prev_steps[-1]

                    ## Note: For replacement case, easier to normalize ahead of time by dividing by Z
                    constrained_density_ratios_cal_test_all_steps_curr = np.concatenate(
                        (
                            constrained_density_ratios_cal_test_all_steps[:, :n_cal],
                            constrained_density_ratios_cal_test_all_steps[
                                :, n_cal + j
                            ].reshape(-1, 1),
                        ),
                        axis=1,
                    )
                    split_weights_vec = compute_w_ptest_split_active_replacement(
                        constrained_density_ratios_cal_test_all_steps_curr,
                        depth_max=depth_max_curr,
                    )

                else:
                    raise NotImplementedError(
                        "Sampling without replacement is not yet implemented"
                    )

                # print("Time elapsed for depth ", depth_max_curr, " (min) : ", (time.time() - time_begin_w) / 60)

                split_weights_vec = split_weights_vec.flatten()

                weights_normalized_wsplit = np.zeros((n_cal + 1, n1))
                sum_cal_weights = np.sum(split_weights_vec[:n_cal])

                for j in range(0, n1):
                    for i in range(0, n_cal + 1):
                        if i < n_cal:
                            weights_normalized_wsplit[i, j] = split_weights_vec[i] / (
                                sum_cal_weights + split_weights_vec[-1]
                            )
                        else:
                            weights_normalized_wsplit[i, j] = split_weights_vec[-1] / (
                                sum_cal_weights + split_weights_vec[-1]
                            )

                wsplit_quantiles[j] = weighted_quantile(
                    unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha
                )

            PIs_dict["wsplit_" + str(depth_max)] = pd.DataFrame(
                np.c_[
                    muh_split_vals_testpoint - wsplit_quantiles,
                    muh_split_vals_testpoint + wsplit_quantiles,
                ],
                columns=["lower", "upper"],
            )

        ###### Mixture-likelihood-ratio-weighted split CP ######
        wsplit_mix_quantiles = np.zeros(n1)

        for j in range(0, n1):
            # print(f"pc_weights_cal_test_normalized shape : {np.shape(pc_weights_cal_test_normalized)}")
            # print(f"unweighted_split_vals shape : {np.shape(unweighted_split_vals)}")
            # print(f"pc_weights_cal_test_normalized sum : {np.sum(pc_weights_cal_test_normalized)}")
            wsplit_mix_quantiles[j] = weighted_quantile(
                unweighted_split_vals, pc_weights_cal_test_normalized[j], 1 - alpha
            )

        PIs_dict["wsplit_mixture"] = pd.DataFrame(
            np.c_[
                muh_split_vals_testpoint - wsplit_mix_quantiles,
                muh_split_vals_testpoint + wsplit_mix_quantiles,
            ],
            columns=["lower", "upper"],
        )

        ###### ACI ######

        q_aci = np.quantile(unweighted_split_vals, 1 - alpha_aci_curr)

        PIs_dict["aci"] = pd.DataFrame(
            np.c_[muh_split_vals_testpoint - q_aci, muh_split_vals_testpoint + q_aci],
            columns=["lower", "upper"],
        )

        return PIs_dict, w_split_mus_prev_steps, weights_normalized_wsplit_all


class SplitConformalMFCS(SplitConformal):
    """
    Class for MFCS Split Conformal experiments
    """

    def __init__(self, model, ptrain_fn, Xuniv_uxp):
        super().__init__(model, ptrain_fn, Xuniv_uxp)
