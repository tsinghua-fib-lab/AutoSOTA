from benchmark.benchmark_utils import (
    get_benchmark_results_synthetic_experiment,
    get_benchmark_results_central_experiment,
)

from data.data_loading import (
    load_synthetic_data,
    load_gbsg,
    load_lung,
    load_vlc,
    load_whas,
    load_colon,
    load_sacadmin,
    load_sac3,
    load_nwtco,
    load_metabric,
    load_support,
)

from benchmark.traditional_methods import (
    fit_CoxPHFitter,
    fit_WeibullAFTFitter,
    fit_RandomSurvivalForest,
    fit_FastSurvivalSVM,
)

from benchmark.dl_methods import (
    fit_CoxTime,
    fit_PMF,
    fit_MTLR,
    fit_BCESurv,
    fit_DeepHitSingle,
    fit_CoxCC,
    fit_Deepsurv,
    fit_PCHazard,
    fit_LogisticHazard,
    fit_Dysurv,
    fit_sumo_net,
    fit_dqs,
)


if __name__ == "__main__":

    output_dir = "/Users/melodiemonod/projects/2025/neuralsurv/benchmark"

    traditional_fit_fns = [
        fit_CoxPHFitter,
        fit_WeibullAFTFitter,
        fit_RandomSurvivalForest,
        fit_FastSurvivalSVM,
    ]
    dl_fit_fns = [
        fit_dqs,
        fit_sumo_net,
        fit_Dysurv,
        fit_LogisticHazard,
        fit_PCHazard,
        fit_Deepsurv,
        fit_CoxCC,
        fit_DeepHitSingle,
        fit_BCESurv,
        fit_MTLR,
        fit_PMF,
        fit_CoxTime,
    ]

    #
    #
    # SYNTHETIC DATA EXPERIMENT
    #

    load_data_fns = [
        load_synthetic_data,
    ]

    for load_data_fn in load_data_fns:

        for fit_fn in dl_fit_fns:
            get_benchmark_results_synthetic_experiment(load_data_fn, fit_fn, output_dir)

        for fit_fn in traditional_fit_fns:
            get_benchmark_results_synthetic_experiment(load_data_fn, fit_fn, output_dir)

    #
    #
    # REAL DATA EXPERIMENT
    #

    # Data set
    load_data_fns = [
        load_gbsg,
        load_support,
        load_sacadmin,
        load_sac3,
        load_nwtco,
        load_metabric,
        load_lung,
        load_vlc,
        load_whas,
        load_colon,
    ]

    for load_data_fn in load_data_fns:

        for fit_fn in traditional_fit_fns:
            get_benchmark_results_central_experiment(load_data_fn, fit_fn, output_dir)

        for fit_fn in dl_fit_fns:
            get_benchmark_results_central_experiment(load_data_fn, fit_fn, output_dir)

    #
    #
    # REAL DATA ABLATION EXPERIMENT
    #

    # Data set
    load_data_fns = [
        load_gbsg,
        load_metabric,
        load_colon,
    ]

    for load_data_fn in load_data_fns:

        for fit_fn in traditional_fit_fns:
            get_benchmark_results_central_experiment(
                load_data_fn,
                fit_fn,
                output_dir,
                subsample_n=250,
                jobid="sub_250_layers_2_hidden_16",
            )

        for fit_fn in dl_fit_fns:
            get_benchmark_results_central_experiment(
                load_data_fn,
                fit_fn,
                output_dir,
                subsample_n=250,
                jobid="sub_250_layers_2_hidden_16",
            )
