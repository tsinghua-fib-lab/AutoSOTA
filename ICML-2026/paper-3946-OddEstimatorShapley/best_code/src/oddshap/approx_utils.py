from __future__ import annotations

import numpy as np
from scipy.special import binom, comb

from shapiq import (
    SVARM,
    PermutationSamplingSV,
    UnbiasedKernelSHAP,
    SVARMIQ,
    SHAPIQ,
    PermutationSamplingSII,
    KernelSHAPIQ,
    ProxySPEX,
    KernelSHAP
)
from oddshap.polyshap import ExplanationFrontierGenerator, PolySHAP
from oddshap.regressionMSR import RegressionMSR
from oddshap.oddshap import OddSHAP
from oddshap.ffd import FFD_RD, FFD_RD_Corrected
from oddshap.fouriershap import FourierSHAP_Approximator


def get_approximators(APPROXIMATORS, n_players, RANDOM_STATE, PAIRING, REPLACEMENT):
    # Create the approximators for the game
    frontier_generator = ExplanationFrontierGenerator(N=set(range(n_players)))

    # initialize the weights for LeverageSHAP Order 1
    sampling_weights_leverage_1 = np.ones(n_players + 1)

    approximator_list = []
    if "KernelSHAP" in APPROXIMATORS:
        # initialize the weights for KernelSHAP
        sampling_weights_kernelshap = np.zeros(n_players + 1)
        for size in range(1, n_players):
            sampling_weights_kernelshap[size] = 1 / (size * (n_players - size))

        explanation_frontier = frontier_generator.generate_kadd(max_order=1)
        kernel_shap = KernelSHAP(
            n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        kernel_shap.name = "KernelSHAP"
        approximator_list.append(kernel_shap)
    if "LeverageSHAP" in APPROXIMATORS:
        # LeverageSHAP
        explanation_frontier = frontier_generator.generate_kadd(max_order=1)
        leverage_shap = PolySHAP(
            n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        leverage_shap.name = "LeverageSHAP"
        approximator_list.append(leverage_shap)
    if "PolySHAP-3ADD" in APPROXIMATORS:
        # ShapleyGAX with k-add explanation basis
        explanation_frontier = frontier_generator.generate_kadd(max_order=3)
        polyshap_3add = PolySHAP(
            n=n_players,
            explanation_frontier=explanation_frontier,
            random_state=RANDOM_STATE,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
        )
        polyshap_3add.name = "PolySHAP-3ADD"
        approximator_list.append(polyshap_3add)
    if "PermutationSampling" in APPROXIMATORS:
        # Permutation Sampling
        permutation_sampling = PermutationSamplingSV(
            n=n_players, pairing_trick=PAIRING, random_state=RANDOM_STATE
        )
        permutation_sampling.name = "PermutationSampling"
        approximator_list.append(permutation_sampling)

    if "MSR" in APPROXIMATORS:
        # MSR = SHAPIQ order 1
        msr = UnbiasedKernelSHAP(
            n=n_players,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        msr.name = "MSR"
        approximator_list.append(msr)
    if "SVARM" in APPROXIMATORS:
        # SVARM
        svarm = SVARM(
            n=n_players,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        svarm.name = "SVARM"
        approximator_list.append(svarm)
    if "RegressionMSR" in APPROXIMATORS:
        # RegressionMSR
        regression_msr = RegressionMSR(
            n=n_players,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        regression_msr.name = "RegressionMSR"
        approximator_list.append(regression_msr)
    if "RegressionMSR-LGBM" in APPROXIMATORS:
        # RegressionMSR
        regression_msr_lgbm = RegressionMSR(
            n=n_players,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            ablation_kwargs= {
                "value_model": "lightgbm",
                "model_params": {"max_depth":10},
                "sampling_weights": "default",
                "residual_approximator": "shapiq",
            }
        )
        regression_msr_lgbm.name = "RegressionMSR-LGBM"
        approximator_list.append(regression_msr_lgbm)
    if "Proxy-LGBM" in APPROXIMATORS:
        # RegressionMSR
        proxy_lgbm = RegressionMSR(
            n=n_players,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            regression_adjustment=False,
            ablation_kwargs= {
                "value_model": "lightgbm",
                "model_params": {"max_depth":10},
                "sampling_weights": "default",
                "residual_approximator": "shapiq",
            }
        )
        proxy_lgbm.name = "Proxy-LGBM"
        approximator_list.append(proxy_lgbm)
    if "Proxy-XGBoost" in APPROXIMATORS:
        # RegressionMSR
        proxy_xgb = RegressionMSR(
            n=n_players,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            regression_adjustment=False,
            ablation_kwargs= {
                "value_model": "xgb",
                "model_params": {},
                "sampling_weights": "default",
                "residual_approximator": "shapiq",
            }
        )
        proxy_xgb.name = "Proxy-XGBoost"
        approximator_list.append(proxy_xgb)
    if "OddSHAP-Fourier-ProxySPEX" in APPROXIMATORS:
        odd_fourier = OddSHAP(
            n=n_players,
            sampling_weights=sampling_weights_leverage_1,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
        )
        odd_fourier.name = "OddSHAP-Fourier-ProxySPEX"
        approximator_list.append(odd_fourier)
    if "OddSHAP-Fourier-Random" in APPROXIMATORS:
        odd_fourier_random = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="Random",
            interaction_factor=10,
            odd_only=True,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False
        )
        odd_fourier_random.name = "OddSHAP-Fourier-Random"
        approximator_list.append(odd_fourier_random)
    if "OddSHAP-Fourier-ProxySPEX-NoCV" in APPROXIMATORS:
        odd_fourier_nocv = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False
        )
        odd_fourier_nocv.name = "OddSHAP-Fourier-ProxySPEX-NoCV"
        approximator_list.append(odd_fourier_nocv)
    if "OddSHAP-Fourier-ProxySPEX-NoCV-15" in APPROXIMATORS:
        odd_fourier_nocv = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False,
            interaction_factor=15
        )
        odd_fourier_nocv.name = "OddSHAP-Fourier-ProxySPEX-NoCV-15"
        approximator_list.append(odd_fourier_nocv)
    if "OddSHAP-Fourier-ProxySPEX-NoCV-30" in APPROXIMATORS:
        odd_fourier_nocv = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False,
            interaction_factor=30
        )
        odd_fourier_nocv.name = "OddSHAP-Fourier-ProxySPEX-NoCV-30"
        approximator_list.append(odd_fourier_nocv)
    if "OddSHAP-Fourier-ProxySPEX-NoCV-20" in APPROXIMATORS:
        odd_fourier_nocv = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False,
            interaction_factor=20
        )
        odd_fourier_nocv.name = "OddSHAP-Fourier-ProxySPEX-NoCV-20"
        approximator_list.append(odd_fourier_nocv)
    if "OddSHAP-Fourier-ProxySPEX-NoCV-5" in APPROXIMATORS:
        odd_fourier_nocv = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False,
            interaction_factor=5
        )
        odd_fourier_nocv.name = "OddSHAP-Fourier-ProxySPEX-NoCV-5"
        approximator_list.append(odd_fourier_nocv)
    if "FFD-RD" in APPROXIMATORS:
        ffd_rd = FFD_RD(n=n_players, random_state=RANDOM_STATE)
        ffd_rd.name = "FFD-RD"
        approximator_list.append(ffd_rd)
    if "FFD-RD-Corrected" in APPROXIMATORS:
        ffd_rd_corrected = FFD_RD_Corrected(n=n_players, random_state=RANDOM_STATE)
        ffd_rd_corrected.name = "FFD-RD-Corrected"
        approximator_list.append(ffd_rd_corrected)
    if "FourierSHAP" in APPROXIMATORS:
        fourier_shap = FourierSHAP_Approximator(n=n_players, T=5, random_state=RANDOM_STATE)
        fourier_shap.name = "FourierSHAP"
        approximator_list.append(fourier_shap)
    if "All-Fourier-ProxySPEX-NoCV" in APPROXIMATORS:
        all_fourier_nocv = OddSHAP(
            n=n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=False,
            sampling_weights=sampling_weights_leverage_1,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=RANDOM_STATE,
            grid_search=False,
            interaction_factor=10
        )
        all_fourier_nocv.name = "All-Fourier-ProxySPEX-NoCV"
        approximator_list.append(all_fourier_nocv)
    return approximator_list
