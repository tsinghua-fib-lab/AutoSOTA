"""An experiment is a list of dictionaries each containing parameter settings.

Each experiment has an `ExperimentName`.
Use the `get_experiment()` function to get the list of dictionaries associated with an experiment name.
"""

try:
    from enum import StrEnum
except ImportError:
    # Python < 3.11 compatibility
    from enum import Enum
    class StrEnum(str, Enum):
        pass
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4
import numpy as np
import pandas as pd
from .constants import (
    BAS_DRO_EPSILON_SET,
    BAS_NUM_REPLICATIONS,
    BAS_TOTAL_MODEL_SAMPLES,
    CONTAMINATION_LEVEL,
    LV_PORTFOLIO_T_SCALE,
    NUM_CERTIFY,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
    PORTFOLIO_EPSILON_SET,
    IN_SAMPLE_TIME_WINDOW,
    OUT_OF_SAMPLE_TIME_WINDOW,
    ROBAS_DRO_EPSILON_SET,
    LV_USE_PC_BULK_GEOMETRY,
    SMALL_BAS_DRO_EPSILON_SET,
    LV_PORTFOLIO_EPSILON_SET,
    KL_FROM_LV_PORTFOLIO_EPSILON_SET,
    PORTFOLIO_SYN_NUM_OBSERVATIONS_SET,
    NEWSVENDOR_NUM_OBSERVATIONS,
    NEWSVENDOR_NUM_TEST_OBSERVATIONS,
    NEWSVENDOR_EPSILON_SET,
    LV_NEWSVENDOR_T_SCALE,
    RW_GAMMA_GRID,
    RW_EPSILON_GRID,
    DO_THRESHOLD_CALIBRATION,
    NEWSVENDOR_KL_EPSILON_SET,
)
from .parameter_settings import RW_NUM_REPLICATIONS, RW_TEXT_N_FEATURES, NEWSVENDOR_NUM_REPLICATIONS, NEWSVENDOR_CONTAMINATION_LEVEL, CALIFORNIA_HOUSING_GAP_RATIO, CALIFORNIA_HOUSING_NUM_REPLICATIONS
from . import constants as _constants
from .dataset import get_num_time_windows, get_portfolio_returns_df

RW_EXPERIMENT_DEFAULTS = dict(
    rw_cal_fraction=0.2,
    rw_split_seed=0,
    rw_cache_dir=None,
    rw_device="auto",

    rw_head="linear",
    rw_mlp_hidden=256,

    rw_train_batch_size=256,
    rw_train_num_workers=0,
    rw_epochs=20,
    rw_lr=1e-3,
    rw_weight_decay=1e-4,
    rw_smoothmax_temperature=0.1,
    rw_groupdro_step_size=0.01,
    rw_max_grad_norm=None,

    rw_text_n_features=RW_TEXT_N_FEATURES,
    rw_text_ngram_min=1,
    rw_text_ngram_max=1,
)


RW_ALGORITHMS = [
    "rw_erm",
    "rw_lv_empirical",
    "rw_lv_empirical_fair",
    "rw_cvar",
    "rw_groupdro",
    "rw_chi2_dro",
]

RW_ALGORITHMS_BULK_ABLATION = [
    *RW_ALGORITHMS,
    "rw_erm_b",
    "rw_cvar_b",
    "rw_chi2_dro_b",
]


class ExperimentName(StrEnum):
    """Names of experiments"""

    kl_newsvendor_1d = "kl_newsvendor_1d"
    kl_newsvendor_5d = "kl_newsvendor_5d"
    lv_newsvendor = "lv_newsvendor"
    mmd_newsvendor_1d = "mmd_newsvendor_1d"
    mmd_newsvendor_1d_missp = "mmd_newsvendor_1d_missp"
    compare_solve = "compare_solve"
    mmd_newsvendor_5d = "mmd_newsvendor_5d"
    mmd_portfolio = "mmd_portfolio"
    mmd_portfolio_synthetic = "mmd_portfolio_synthetic"
    mmd_portfolio_crash = "mmd_portfolio_crash"
    kl_portfolio = "kl_portfolio"
    kl_portfolio_crash = "kl_portfolio_crash"
    kl_portfolio_synthetic = "kl_portfolio_synthetic"
    kl_newsvendor_exp_1d = "kl_newsvendor_exp_1d"
    mmd_newsvendor_exp_1d = "mmd_newsvendor_exp_1d"
    lv_portfolio = "lv_portfolio"
    lv_portfolio_syn = "lv_portfolio_syn"
    lv_newsvendor_student_t = "lv_newsvendor_student_t"
    lv_newsvendor_student_t_se = "lv_newsvendor_student_t_se"
    lv_newsvendor_student_t_gamma_bulk = "lv_newsvendor_student_t_gamma_bulk"
    lv_newsvendor_student_t_lv_tv = "lv_newsvendor_student_t_lv_tv"
    lv_california_housing = "lv_california_housing"
    lv_california_housing_val = "lv_california_housing_val"
    lv_california_housing_extra_f_div_baseline = "lv_california_housing_extra_f_div_baseline"
    lv_california_housing_val_extra_f_div_baseline = "lv_california_housing_val_extra_f_div_baseline"
    rw_civilcomments = "rw_civilcomments"
    rw_civilcomments_bulk_ablation = "rw_civilcomments_bulk_ablation"


    def is_portfolio(self) -> bool:
        return self in (
            ExperimentName.kl_portfolio,
            ExperimentName.mmd_portfolio,
            ExperimentName.kl_portfolio_crash,
            ExperimentName.mmd_portfolio_crash,
            ExperimentName.lv_portfolio,
        )



def get_experiment(experiment_name: ExperimentName, dataset_dir: Optional[Path] = None) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.kl_newsvendor_1d: kl_newsvendor_1d,
        ExperimentName.kl_newsvendor_5d: kl_newsvendor_5d,
        ExperimentName.mmd_newsvendor_1d: mmd_newsvendor_1d,
        ExperimentName.mmd_newsvendor_1d_missp: mmd_newsvendor_1d_missp,
        ExperimentName.compare_solve: compare_solve,
        ExperimentName.mmd_newsvendor_5d: mmd_newsvendor_5d,
        ExperimentName.mmd_portfolio: mmd_portfolio,
        ExperimentName.mmd_portfolio_crash: mmd_portfolio_crash,
        ExperimentName.kl_portfolio: kl_portfolio,
        ExperimentName.kl_portfolio_crash: kl_portfolio_crash,
        ExperimentName.kl_portfolio_synthetic: kl_portfolio_synthetic,
        ExperimentName.lv_portfolio: lv_portfolio,
        ExperimentName.lv_portfolio_syn: lv_portfolio_syn,
        ExperimentName.kl_newsvendor_exp_1d: kl_newsvendor_exp_1d,
        ExperimentName.mmd_newsvendor_exp_1d: mmd_newsvendor_exp_1d,
        ExperimentName.mmd_portfolio_synthetic: mmd_portfolio_synthetic,
        ExperimentName.lv_newsvendor: lv_newsvendor,
        ExperimentName.lv_newsvendor_student_t: lv_newsvendor_student_t,
        ExperimentName.lv_newsvendor_student_t_se: lv_newsvendor_student_t_se,
        ExperimentName.lv_newsvendor_student_t_gamma_bulk: lv_newsvendor_student_t_gamma_bulk,
        ExperimentName.lv_newsvendor_student_t_lv_tv: lv_newsvendor_student_t_lv_tv,
        ExperimentName.lv_california_housing: lv_california_housing,
        ExperimentName.lv_california_housing_val : lv_california_housing_val,
        ExperimentName.lv_california_housing_extra_f_div_baseline: lv_california_housing_extra_f_div_baseline,
        ExperimentName.lv_california_housing_val_extra_f_div_baseline: lv_california_housing_val_extra_f_div_baseline,
        ExperimentName.rw_civilcomments: rw_civilcomments,
        ExperimentName.rw_civilcomments_bulk_ablation: rw_civilcomments_bulk_ablation,
    }

    try:
        if experiment_name.is_portfolio():
            # NOTE portfolio setup requires a dataset_dir argument
            return function_lookup[experiment_name](dataset_dir)
        return function_lookup[experiment_name]()
    except KeyError as e:
        raise KeyError(
            f"Please add {experiment_name} as a key in the function lookup dictionary"
        ) from e


def get_num_likelihood_samples(dataset: str, num_observations: int, num_total_samples: int, algorithm: str) -> int:
    if algorithm == "kl_bdro" and dataset == "portfolio":
        return 1
    if algorithm == "kl_bdro":
        return int(np.sqrt(num_total_samples))
    if algorithm in ("kl_dro_bas", "kl_pp", "lv_bas", "lv_reverse", "tv_ball"):
        return num_total_samples
    if algorithm == "kl_empirical":
        return num_observations
    raise NotImplementedError()


def get_num_posterior_samples(dataset: str, num_total_samples: int, algorithm: str) -> int:
    if algorithm == "kl_bdro" and dataset == "portfolio":
        return num_total_samples
    if algorithm == "kl_bdro":
        return int(np.sqrt(num_total_samples))
    if algorithm in ("kl_dro_bas", "kl_pp", "lv_bas", "lv_reverse", "tv_ball"):
        return 1
    if algorithm == "kl_empirical":
        return 1
    raise NotImplementedError()



def kl_newsvendor_5d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for algorithm, num_observations, (dgp, likelihood, posterior), epsilon in itertools.product(
        ["kl_pp", "kl_dro_bas", "kl_bdro", "kl_empirical"],
        [NUM_OBSERVATIONS],
        [
            ("multivariate_normal", "multivariate_normal", "normal_inverse_wishart"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            total_model_samples_list = BAS_TOTAL_MODEL_SAMPLES
            inference = "bayes"
        for total_model_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "newsvendor",
                "dgp": dgp,
                "dim": 5,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "njobs": 1,
                "num_likelihood_samples": get_num_likelihood_samples("newsvendor", num_observations, total_model_samples, algorithm),
                "num_observations": num_observations,
                "num_posterior_samples": get_num_posterior_samples("newsvendor", total_model_samples, algorithm),
                "num_replications": BAS_NUM_REPLICATIONS,
                "num_test_observations": NUM_TEST_OBSERVATIONS,
                "posterior": posterior,
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment


def kl_newsvendor_1d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for algorithm, num_observations, (dgp, likelihood, posterior), epsilon in itertools.product(
        ["kl_pp", "kl_dro_bas", "kl_bdro", "kl_empirical"],
        [NUM_OBSERVATIONS],  # [5, 20, 100],
        [
            ("normal", "normal", "normal_gamma"),
            # ("truncated_normal", "normal", "normal_gamma"),
            ("exponential", "exponential", "gamma"),
            # ("contaminated_exp", "exponential", "gamma"),
        ],
        BAS_DRO_EPSILON_SET,
        # SMALL_BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            # total_model_samples_list = BAS_TOTAL_MODEL_SAMPLES
            total_model_samples_list = [3600, 10000]
            inference = "bayes"
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        for total_model_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": contamination,
                "dataset": "newsvendor",
                "dgp": dgp,
                "dim": 1,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "njobs": 1,
                "num_likelihood_samples": get_num_likelihood_samples("newsvendor", num_observations, total_model_samples, algorithm),
                "num_observations": num_observations,
                "num_posterior_samples": get_num_posterior_samples("newsvendor", total_model_samples, algorithm),
                "num_replications": BAS_NUM_REPLICATIONS,
                "num_test_observations": NUM_TEST_OBSERVATIONS,
                "posterior": posterior,
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment

def lv_newsvendor() -> List[Dict]:
    """LV newsvendor with non-regular Uniform(0,θ) likelihood and Pareto conjugate prior.

    This is *sampling-only*: we run kl_pp / kl_bdro (+ empirical baseline).
    kl_dro_bas is not included because it requires exponential-family quantities.
    """
    experiment: List[Dict] = []
    dgp = "unif_0_theta"
    dim = 1

    for algorithm, num_observations, epsilon in itertools.product(
        ["lv_bas","kl_pp", "kl_bdro", "kl_empirical"],
        [NEWSVENDOR_NUM_OBSERVATIONS],
        NEWSVENDOR_EPSILON_SET,
    ):
        if algorithm == "lv_bas" and epsilon > 1.0:
            # LV-BAS only defined for epsilon in (0,1] in newsvendor setting
            continue
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            total_model_samples_list = [3600]
            likelihood = "uniform_0_theta"
            posterior = "pareto"
            inference = "bayes"
        
        if algorithm == "lv_bas":
            # Set lengthscale according to reference t_ref
            epsilon = epsilon
        else:
            epsilon = 0.5 * (epsilon * LV_NEWSVENDOR_T_SCALE) ** 2 

        for total_model_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "newsvendor",
                "dgp": dgp,
                "dim": dim,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "njobs": 1,
                "num_likelihood_samples": get_num_likelihood_samples(
                    "newsvendor", num_observations, total_model_samples, algorithm
                ),
                "num_observations": num_observations,
                "num_posterior_samples": get_num_posterior_samples(
                    "newsvendor", total_model_samples, algorithm
                ),
                "num_replications": NEWSVENDOR_NUM_REPLICATIONS,
                "num_test_observations": NEWSVENDOR_NUM_TEST_OBSERVATIONS,
                "posterior": posterior,
                "uuid": str(uuid4()),
                "lv_use_pc_bulk_geometry": LV_USE_PC_BULK_GEOMETRY,
            }
            experiment.append(params)

    return experiment

def lv_newsvendor_student_t() -> List[Dict]:
    """
    Clean multivariate Student-t (df=3) newsvendor setting.

    Key point for your chosen option:
      - LV-BAS uses DKW-calibrated ellipsoid bulk set with radius t_hat (not sqrt(t_hat)).

    Budgets (total model samples = 2500):
      - kl_pp:       M_post=1,   M_pred=2500
      - lv_bas:      M_post=1,   M_pred=2500 (then truncated; optimisation uses n_trunc=0.5*M_pred)
      - kl_bdro:     M_post=50,  M_pred=50
      - kl_empirical: empirical (uses the n training points)
    """
    from .constants import (
        NEWSVENDOR_CONTAMINATION_TYPE,
        NESWVENDOR_VARY_RHO,
        NEWSVENDOR_KL_EPSILON_SET,
    )
    num_observations = NEWSVENDOR_NUM_OBSERVATIONS
    num_test_observations = NEWSVENDOR_NUM_TEST_OBSERVATIONS
    num_replications = NEWSVENDOR_NUM_REPLICATIONS
    dim = 5

    epsilon_set_1 = NEWSVENDOR_EPSILON_SET
    epsilon_set_2 = NEWSVENDOR_KL_EPSILON_SET
    #algorithm_set = ["or_wdro","lv_bas"]
    #algorithm_set = ["lv_bas", "kl_bdro"]
    vary_rho=NESWVENDOR_VARY_RHO
    experiment_set: List[Dict] = []
    #for algorithm, eps_raw in itertools.product(algorithm_set, epsilon_set):
    algorithm_set_1 = ["or_wdro","lv_bas"]
    algorithm_set_2 = ["kl_pp", "kl_bdro", "kl_empirical"]
    num_total_samples = 2500
    for eps_raw, algorithm in itertools.product(epsilon_set_1, algorithm_set_1):
        # Map LV-style epsilon to KL radius (keeps consistency with existing experiments.py conventions).
        eps = eps_raw
        if algorithm == "lv_bas" and eps > 1.0:
            # LV-BAS only defined for epsilon in (0,1] 
            continue
        if algorithm == "or_wdro"  and vary_rho:
            eps = 0.5 * (eps_raw * LV_NEWSVENDOR_T_SCALE) ** 2
        elif eps>=0.5 and algorithm == "or_wdro":
        # OR-WDRO only defined for epsilon in (0,0.5]
            continue
        inference = "bayes"
        posterior = "student_t_niw"
        likelihood = "multivariate_student_t"

        if algorithm == "kl_empirical":
            inference = "empirical"
            likelihood = "empirical"

        if algorithm in ("or_wdro","kl_empirical"):
            num_posterior_samples = 1
            num_likelihood_samples = num_observations
        else: 
            num_likelihood_samples = get_num_likelihood_samples(
                dataset="newsvendor",
                num_observations=num_observations,
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
            num_posterior_samples = get_num_posterior_samples(
                dataset="newsvendor",
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
        # Keep your existing fields exactly, only add the missing ones expected by the runner.
        params = dict(
            algorithm=algorithm,
            contamination=NEWSVENDOR_CONTAMINATION_LEVEL,
            contamination_type = NEWSVENDOR_CONTAMINATION_TYPE,          
            dataset="newsvendor",
            dgp="student_t",
            dim=dim,
            epsilon=eps,
            ignore_dpp=True,            
            inference=inference,
            lengthscale=-1.0,           
            likelihood=likelihood,
            njobs=1,                   
            num_observations=num_observations,
            num_test_observations=num_test_observations,
            num_replications=num_replications,
            num_posterior_samples=num_posterior_samples,
            num_likelihood_samples=num_likelihood_samples,
            posterior=posterior,
            uuid=str(uuid4()),
            lv_use_pc_bulk_geometry=LV_USE_PC_BULK_GEOMETRY,
            vary_rho=NESWVENDOR_VARY_RHO,          
        )

        experiment_set.append(params)
    for eps_raw, algorithm in itertools.product(epsilon_set_2, algorithm_set_2):
        # Map LV-style epsilon to KL radius (keeps consistency with existing experiments.py conventions).
        eps = eps_raw
        inference = "bayes"
        posterior = "student_t_niw"
        likelihood = "multivariate_student_t"

        if algorithm == "kl_empirical":
            inference = "empirical"
            likelihood = "empirical"

        if algorithm in ("or_wdro","kl_empirical"):
            num_posterior_samples = 1
            num_likelihood_samples = num_observations
        else: 
            num_likelihood_samples = get_num_likelihood_samples(
                dataset="newsvendor",
                num_observations=num_observations,
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
            num_posterior_samples = get_num_posterior_samples(
                dataset="newsvendor",
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
        # Keep your existing fields exactly, only add the missing ones expected by the runner.
        params = dict(
            algorithm=algorithm,
            contamination=NEWSVENDOR_CONTAMINATION_LEVEL,
            contamination_type = NEWSVENDOR_CONTAMINATION_TYPE,          
            dataset="newsvendor",
            dgp="student_t",
            dim=dim,
            epsilon=eps,
            ignore_dpp=True,            
            inference=inference,
            lengthscale=-1.0,           
            likelihood=likelihood,
            njobs=1,                   
            num_observations=num_observations,
            num_test_observations=num_test_observations,
            num_replications=num_replications,
            num_posterior_samples=num_posterior_samples,
            num_likelihood_samples=num_likelihood_samples,
            posterior=posterior,
            uuid=str(uuid4()),
            lv_use_pc_bulk_geometry=LV_USE_PC_BULK_GEOMETRY,
            vary_rho=NESWVENDOR_VARY_RHO,          
        )

        experiment_set.append(params)


    return experiment_set

def lv_newsvendor_student_t_se() -> List[Dict]:
    """
    Sample-efficiency sweep for the multivariate Student-t (df=3) newsvendor setting.

    We keep the same epsilon grid as lv_newsvendor_student_t, but for Monte-Carlo-based
    methods we additionally vary a "total model samples" budget.

    Total model samples definition:
      - lv_bas, kl_pp:  M_total := M_lik     (since M_post = 1)
      - kl_bdro:        M_total := M_post * M_lik

    Implementation:
      - Loop over total_model_samples_list and set (num_posterior_samples, num_likelihood_samples)
        using the existing helpers:
            get_num_posterior_samples(...)
            get_num_likelihood_samples(...)

    Notes (deliberate modelling choices):
      - kl_empirical does not use model sampling; we include it once per epsilon only.
      - or_wdro is excluded: it has no MC sampling knobs, and it is not meaningful in a
        "sample efficiency" comparison defined in terms of (M_post, M_lik).
    """
    from .constants import (
        NEWSVENDOR_CONTAMINATION_TYPE,
    )

    num_observations = NEWSVENDOR_NUM_OBSERVATIONS
    num_test_observations = NEWSVENDOR_NUM_TEST_OBSERVATIONS
    num_replications = NEWSVENDOR_NUM_REPLICATIONS
    dim = 5

    epsilon_set_1 = NEWSVENDOR_EPSILON_SET

    # IMPORTANT: keep this list modest, otherwise the Cartesian product
    # (epsilon_set × total_model_samples_list × algorithms × replications) explodes.
    #
    # Using perfect squares is intentional: for kl_bdro we set M_post=M_lik=int(sqrt(M_total)),
    # so perfect squares avoid "rounding down" artefacts in the realised product.
    total_model_samples_list = [25, 100, 400, 900, 1600, 2500, 3600, 4900]
    #total_model_samples_list = [25, 100, 900]

    experiment_set: List[Dict] = []
    for eps_raw in epsilon_set_1: # more even spread across batches
    #for algorithm, eps_raw in itertools.product(algorithm_set, epsilon_set):
        # Map LV-style epsilon to KL radius (keeps consistency with existing experiments.py conventions).
        algorithm = "lv_bas"
        eps = float(eps_raw)
        if eps > 1.0:
            # LV-BAS only defined for epsilon in (0,1] 
            continue
        inference = "bayes"
        posterior = "student_t_niw"
        likelihood = "multivariate_student_t"

        if algorithm == "kl_empirical":
            inference = "empirical"
            likelihood = "empirical"

        # kl_empirical: no MC sampling sweep (it uses the empirical distribution with n samples).
        if algorithm == "kl_empirical":
            total_samples_iter = [num_observations]
        else:
            total_samples_iter = total_model_samples_list

        for num_total_samples in total_samples_iter:
            num_likelihood_samples = get_num_likelihood_samples(
                dataset="newsvendor",
                num_observations=num_observations,
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
            num_posterior_samples = get_num_posterior_samples(
                dataset="newsvendor",
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )

            params = dict(
                algorithm=algorithm,
                contamination=NEWSVENDOR_CONTAMINATION_LEVEL,
                contamination_type=NEWSVENDOR_CONTAMINATION_TYPE,
                dataset="newsvendor",
                dgp="student_t",
                dim=dim,
                epsilon=eps,
                ignore_dpp=True,
                inference=inference,
                lengthscale=-1.0,
                likelihood=likelihood,
                njobs=1,
                num_observations=num_observations,
                num_test_observations=num_test_observations,
                num_replications=num_replications,
                num_posterior_samples=num_posterior_samples,
                num_likelihood_samples=num_likelihood_samples,
                posterior=posterior,
                uuid=str(uuid4()),
                lv_use_pc_bulk_geometry=LV_USE_PC_BULK_GEOMETRY,
            )
            experiment_set.append(params)
    
    algorithm_set = ["kl_bdro", "kl_pp"]
    epsilon_set_2 = NEWSVENDOR_KL_EPSILON_SET
    for eps_raw, algorithm in itertools.product(epsilon_set_2, algorithm_set): # more even spread across batches
    #for algorithm, eps_raw in itertools.product(algorithm_set, epsilon_set):
        # Map LV-style epsilon to KL radius (keeps consistency with existing experiments.py conventions).
        eps = float(eps_raw)

        inference = "bayes"
        posterior = "student_t_niw"
        likelihood = "multivariate_student_t"

        if algorithm == "kl_empirical":
            inference = "empirical"
            likelihood = "empirical"

        # kl_empirical: no MC sampling sweep (it uses the empirical distribution with n samples).
        if algorithm == "kl_empirical":
            total_samples_iter = [num_observations]
        else:
            total_samples_iter = total_model_samples_list

        for num_total_samples in total_samples_iter:
            num_likelihood_samples = get_num_likelihood_samples(
                dataset="newsvendor",
                num_observations=num_observations,
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
            num_posterior_samples = get_num_posterior_samples(
                dataset="newsvendor",
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )

            params = dict(
                algorithm=algorithm,
                contamination=NEWSVENDOR_CONTAMINATION_LEVEL,
                contamination_type=NEWSVENDOR_CONTAMINATION_TYPE,
                dataset="newsvendor",
                dgp="student_t",
                dim=dim,
                epsilon=eps,
                ignore_dpp=True,
                inference=inference,
                lengthscale=-1.0,
                likelihood=likelihood,
                njobs=1,
                num_observations=num_observations,
                num_test_observations=num_test_observations,
                num_replications=num_replications,
                num_posterior_samples=num_posterior_samples,
                num_likelihood_samples=num_likelihood_samples,
                posterior=posterior,
                uuid=str(uuid4()),
                lv_use_pc_bulk_geometry=LV_USE_PC_BULK_GEOMETRY,
            )
            experiment_set.append(params)

    return experiment_set

def lv_newsvendor_student_t_gamma_bulk() -> List[Dict]:
    """
    Gamma-bulk sensitivity sweep for the multivariate Student-t newsvendor.

    LV-BAS is run over a grid of gamma_bulk values.
    The other baselines are included once each (they do not depend on gamma_bulk)
    so they can be overlaid as dashed reference frontiers.
    """
    from .constants import (
        NEWSVENDOR_CONTAMINATION_TYPE,
        NESWVENDOR_VARY_RHO,
    )

    num_observations = NEWSVENDOR_NUM_OBSERVATIONS
    num_test_observations = NEWSVENDOR_NUM_TEST_OBSERVATIONS
    num_replications = NEWSVENDOR_NUM_REPLICATIONS
    dim = 5

    epsilon_set = NEWSVENDOR_EPSILON_SET
    algorithm = "lv_bas"
    num_total_samples = 2500

    delta_dkw = 0.05
    n_selection = int(num_observations - (num_observations // 2))
    gamma_min_cert = 0.043
    gamma_bulk_set = sorted({
        gamma_min_cert,
        0.05, 0.06, 0.07, 0.08, 0.09,
        0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
    })

    inference = "bayes"
    posterior = "student_t_niw"
    likelihood = "multivariate_student_t"

    num_likelihood_samples = get_num_likelihood_samples(
        dataset="newsvendor",
        num_observations=num_observations,
        num_total_samples=int(num_total_samples),
        algorithm=algorithm,
    )
    num_posterior_samples = get_num_posterior_samples(
        dataset="newsvendor",
        num_total_samples=int(num_total_samples),
        algorithm=algorithm,
    )

    experiment_set: List[Dict] = []

    baseline_gamma_bulk = float(gamma_bulk_set[0])
    for params in lv_newsvendor_student_t():
        if str(params.get("algorithm")) == "lv_bas":
            continue
        params = dict(params)
        params["gamma_bulk"] = baseline_gamma_bulk
        experiment_set.append(params)

    for gamma_bulk, eps in itertools.product(gamma_bulk_set, epsilon_set):
        eps = float(eps)
        if eps > 1.0:
            continue

        params = dict(
            algorithm=algorithm,
            contamination=NEWSVENDOR_CONTAMINATION_LEVEL,
            contamination_type=NEWSVENDOR_CONTAMINATION_TYPE,
            dataset="newsvendor",
            dgp="student_t",
            dim=dim,
            epsilon=eps,
            ignore_dpp=True,
            inference=inference,
            lengthscale=-1.0,
            likelihood=likelihood,
            njobs=1,
            num_observations=num_observations,
            num_test_observations=num_test_observations,
            num_replications=num_replications,
            num_posterior_samples=num_posterior_samples,
            num_likelihood_samples=num_likelihood_samples,
            posterior=posterior,
            uuid=str(uuid4()),
            lv_use_pc_bulk_geometry=LV_USE_PC_BULK_GEOMETRY,
            vary_rho=NESWVENDOR_VARY_RHO,
            gamma_bulk=float(gamma_bulk),
        )
        experiment_set.append(params)

    return experiment_set


def lv_newsvendor_student_t_lv_tv() -> List[Dict[str, Any]]:
    from .constants import (
        NEWSVENDOR_CONTAMINATION_TYPE,
    )
    num_observations = NEWSVENDOR_NUM_OBSERVATIONS
    num_test_observations = NEWSVENDOR_NUM_TEST_OBSERVATIONS
    num_replications = NEWSVENDOR_NUM_REPLICATIONS
    dim = 5

    epsilon_set = NEWSVENDOR_EPSILON_SET
    algorithm_set = ["lv_bas", "tv_ball", "lv_reverse"]
    #algorithm_set = ["lv_bas", "kl_pp"]

    experiment_set: List[Dict] = []
    for algorithm, eps_raw in itertools.product(algorithm_set, epsilon_set):
        # Map LV-style epsilon to KL radius (keeps consistency with existing experiments.py conventions).
        eps = eps_raw
        if algorithm in ("lv_bas", "tv_ball", "lv_reverse") and eps > 1.0:
            # LV-BAS, TV-BALL, LV-REVERSE only defined for epsilon in (0,1] 
            continue
        if algorithm == "or_wdro" and eps >= 0.5:
            # OR-WDRO only defined for epsilon in (0,0.5] 
            continue
        if algorithm in ("kl_bdro", "kl_pp","kl_empirical"):
            eps = 0.5 * (eps_raw * LV_NEWSVENDOR_T_SCALE) ** 2
        num_total_samples = 2500
        inference = "bayes"
        posterior = "student_t_niw"
        likelihood = "multivariate_student_t"

        if algorithm == "kl_empirical":
            inference = "empirical"
            likelihood = "empirical"

        if algorithm in ("or_wdro","kl_empirical"):
            num_posterior_samples = 1
            num_likelihood_samples = num_observations
        else: 
            num_likelihood_samples = get_num_likelihood_samples(
                dataset="newsvendor",
                num_observations=num_observations,
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
            num_posterior_samples = get_num_posterior_samples(
                dataset="newsvendor",
                num_total_samples=int(num_total_samples),
                algorithm=algorithm,
            )
        # Keep your existing fields exactly, only add the missing ones expected by the runner.
        params = dict(
            algorithm=algorithm,
            contamination=NEWSVENDOR_CONTAMINATION_LEVEL,
            contamination_type = NEWSVENDOR_CONTAMINATION_TYPE,          
            dataset="newsvendor",
            dgp="student_t",
            dim=dim,
            epsilon=eps,
            ignore_dpp=True,            
            inference=inference,
            lengthscale=-1.0,           
            likelihood=likelihood,
            njobs=1,                   
            num_observations=num_observations,
            num_test_observations=num_test_observations,
            num_replications=num_replications,
            num_posterior_samples=num_posterior_samples,
            num_likelihood_samples=num_likelihood_samples,
            posterior=posterior,
            uuid=str(uuid4()),
            lv_use_pc_bulk_geometry=LV_USE_PC_BULK_GEOMETRY,          
        )

        experiment_set.append(params)

    return experiment_set



def lv_california_housing() -> List[Dict]:
    """California Housing LV-BAS experiment grid (LV-BAS + ERM LAD + CVaR LAD)."""
    epsilon_set = getattr(_constants, "CALIFORNIA_HOUSING_EPSILON_SET", [0.0, 0.05, 0.10, 0.20, 0.30])
    ridge_set = getattr(_constants, "CALIFORNIA_HOUSING_RIDGE_LAMBDA_SET", [0.01, 0.1, 1.0, 10.0])
    epsilon_wass_set = getattr(_constants, "CALIFORNIA_HOUSING_WASS_SET", [0.0, 0.05, 0.10, 0.20])
    gamma_set = getattr(_constants, "CALIFORNIA_HOUSING_GAMMA_SET", [0.01, 0.05, 0.10])
    num_replications = CALIFORNIA_HOUSING_NUM_REPLICATIONS
    standardise_y = getattr(_constants,"CALIFORNIA_HOUSING_STANDARDISE_Y",False)
    M = int(getattr(_constants, "CALIFORNIA_HOUSING_LV_MC_SAMPLES", 5000))
    Ch_gap_ratio = CALIFORNIA_HOUSING_GAP_RATIO
    clip_extreme_y = getattr(_constants, "CALIFORNIA_HOUSING_CLIP_Y_EXTREME", False)
    experiment_set = []
    for gamma_bulk in gamma_set:
        for epsilon in epsilon_set:
            for algorithm in ("lv_bas_ch", "cvar_lad"):
                if epsilon > 1.0:
                    # LV-BAS and CVaR-LAD only defined for epsilon in [0,1]
                    continue
                experiment_set.append(
                    {
                        "algorithm": algorithm,
                        "dataset": "california_housing",
                        "dgp": "california_housing",
                        "num_observations": 0,
                        "num_test_observations": 0,
                        "dim": 8,
                        "contamination": 0.0,
                        "contamination_type": None,
                        "gamma_bulk": float(gamma_bulk),
                        "num_likelihood_samples": int(M),
                        "num_posterior_samples": 0,
                        "num_certify_points": 0,
                        "posterior": "frequentist",
                        "inference": "frequentist",
                        "likelihood": "gaussian_copula_ridge",
                        "epsilon": float(epsilon),
                        "uuid": str(uuid4()),
                        "num_replications": int(num_replications),
                        "verbose": False,
                        "standardise_y": standardise_y,
                        "Ch_gap_ratio":Ch_gap_ratio,
                        "clip_extreme_y":clip_extreme_y,
                    }
                )
    for gamma_bulk in gamma_set:
        for epsilon in epsilon_wass_set:
            algorithm = "wass_lad"
            experiment_set.append(
                {
                    "algorithm": algorithm,
                    "dataset": "california_housing",
                    "dgp": "california_housing",
                    "num_observations": 0,
                    "num_test_observations": 0,
                    "dim": 8,
                    "contamination": 0.0,
                    "contamination_type": None,
                    "gamma_bulk": float(gamma_bulk),
                    "num_likelihood_samples": int(M),
                    "num_posterior_samples": 0,
                    "num_certify_points": 0,
                    "posterior": "frequentist",
                    "inference": "frequentist",
                    "likelihood": "gaussian_copula_ridge",
                    "epsilon": float(epsilon),
                    "uuid": str(uuid4()),
                    "num_replications": int(num_replications),
                    "verbose": False,
                    "standardise_y": standardise_y,
                    "Ch_gap_ratio":Ch_gap_ratio,
                    "clip_extreme_y":clip_extreme_y,
                }
            )
    for gamma_bulk in gamma_set:
        epsilon = 0.1 #placeholder, not used in erm_lad
        algorithm = "erm_lad"
        experiment_set.append(
            {
                "algorithm": algorithm,
                "dataset": "california_housing",
                "dgp": "california_housing",
                "num_observations": 0,
                "num_test_observations": 0,
                "dim": 8,
                "contamination": 0.0,
                "contamination_type": None,
                "gamma_bulk": float(gamma_bulk),
                "num_likelihood_samples": int(M),
                "num_posterior_samples": 0,
                "num_certify_points": 0,
                "posterior": "frequentist",
                "inference": "frequentist",
                "likelihood": "gaussian_copula_ridge",
                "epsilon": float(epsilon),
                "uuid": str(uuid4()),
                "num_replications": int(num_replications),
                "verbose": False,
                "standardise_y": standardise_y,
                "Ch_gap_ratio":Ch_gap_ratio,
                "clip_extreme_y":clip_extreme_y,
            }
        )
    for gamma_bulk in gamma_set:
        for epsilon in ridge_set:
            algorithm = "erm_ridge"
            experiment_set.append(
                {
                    "algorithm": algorithm,
                    "dataset": "california_housing",
                    "dgp": "california_housing",
                    "num_observations": 0,
                    "num_test_observations": 0,
                    "dim": 8,
                    "contamination": 0.0,
                    "contamination_type": None,
                    "gamma_bulk": float(gamma_bulk),
                    "num_likelihood_samples": int(M),
                    "num_posterior_samples": 0,
                    "num_certify_points": 0,
                    "posterior": "frequentist",
                    "inference": "frequentist",
                    "likelihood": "gaussian_copula_ridge",
                    "epsilon": float(epsilon),
                    "uuid": str(uuid4()),
                    "num_replications": int(num_replications),
                    "verbose": False,
                    "standardise_y": standardise_y,
                    "Ch_gap_ratio":Ch_gap_ratio,
                    "clip_extreme_y":clip_extreme_y,
                }
            )
    return experiment_set

def lv_california_housing_val() -> List[Dict]:
    """
    California Housing LV-BAS experiment with validation calibration.

    This is a two-stage procedure per replication:
      (1) tune epsilon / ridge_lambda on VAL,
      (2) report TEST metrics at the chosen value.

    We keep epsilon in the config for compatibility, but set it to 0.0 as a dummy.
    """
    gamma_set = getattr(_constants, "CALIFORNIA_HOUSING_GAMMA_SET", [0.01, 0.05, 0.10])
    num_replications = CALIFORNIA_HOUSING_NUM_REPLICATIONS
    standardise_y = getattr(_constants, "CALIFORNIA_HOUSING_STANDARDISE_Y", False)
    M = int(getattr(_constants, "CALIFORNIA_HOUSING_LV_MC_SAMPLES", 5000))
    Ch_gap_ratio = CALIFORNIA_HOUSING_GAP_RATIO
    clip_extreme_y = getattr(_constants, "CALIFORNIA_HOUSING_CLIP_Y_EXTREME", False)
    experiment_set = []
    for gamma_bulk in gamma_set:
        for algorithm in ("lv_bas_ch", "erm_lad", "erm_ridge", "wass_lad", "cvar_lad"):
            experiment_set.append(
                {
                    "algorithm": algorithm,
                    "dataset": "california_housing",
                    "dgp": "california_housing",
                    "num_observations": 0,
                    "num_test_observations": 0,
                    "dim": 8,
                    "contamination":0.0,
                    "contamination_type": None,
                    "gamma_bulk": float(gamma_bulk),
                    "num_likelihood_samples": int(M),
                    "num_posterior_samples": 0,
                    "num_certify_points": 0,
                    "posterior": "frequentist",
                    "inference": "frequentist",
                    "likelihood": "gaussian_copula_ridge",
                    "epsilon": 0.1,  # dummy; tuned on validation if calibrate_on_validation=True
                    "calibrate_on_validation": True,
                    "uuid": str(uuid4()),
                    "num_replications": int(num_replications),
                    "verbose": False,
                    "standardise_y": standardise_y,
                    "Ch_gap_ratio": Ch_gap_ratio,
                    "clip_extreme_y":clip_extreme_y,
                }
            )
    return experiment_set

def lv_california_housing_extra_f_div_baseline() -> List[Dict]:
    """California Housing sweep with additional chi-squared and KL DRO baselines."""
    experiment_set = lv_california_housing()

    chi2_set = getattr(_constants, "CALIFORNIA_HOUSING_CHI2_SET", [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0])
    kl_set = getattr(_constants, "CALIFORNIA_HOUSING_KL_SET", [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0])
    gamma_set = getattr(_constants, "CALIFORNIA_HOUSING_GAMMA_SET", [0.01, 0.05, 0.10])
    num_replications = CALIFORNIA_HOUSING_NUM_REPLICATIONS
    standardise_y = getattr(_constants, "CALIFORNIA_HOUSING_STANDARDISE_Y", False)
    M = int(getattr(_constants, "CALIFORNIA_HOUSING_LV_MC_SAMPLES", 5000))
    Ch_gap_ratio = CALIFORNIA_HOUSING_GAP_RATIO
    clip_extreme_y = getattr(_constants, "CALIFORNIA_HOUSING_CLIP_Y_EXTREME", False)

    for gamma_bulk in gamma_set:
        for epsilon in chi2_set:
            experiment_set.append(
                {
                    "algorithm": "chi2_lad",
                    "dataset": "california_housing",
                    "dgp": "california_housing",
                    "num_observations": 0,
                    "num_test_observations": 0,
                    "dim": 8,
                    "contamination": 0.0,
                    "contamination_type": None,
                    "gamma_bulk": float(gamma_bulk),
                    "num_likelihood_samples": int(M),
                    "num_posterior_samples": 0,
                    "num_certify_points": 0,
                    "posterior": "frequentist",
                    "inference": "frequentist",
                    "likelihood": "gaussian_copula_ridge",
                    "epsilon": float(epsilon),
                    "uuid": str(uuid4()),
                    "num_replications": int(num_replications),
                    "verbose": False,
                    "standardise_y": standardise_y,
                    "Ch_gap_ratio": Ch_gap_ratio,
                    "clip_extreme_y": clip_extreme_y,
                }
            )

        for epsilon in kl_set:
            experiment_set.append(
                {
                    "algorithm": "kl_lad",
                    "dataset": "california_housing",
                    "dgp": "california_housing",
                    "num_observations": 0,
                    "num_test_observations": 0,
                    "dim": 8,
                    "contamination": 0.0,
                    "contamination_type": None,
                    "gamma_bulk": float(gamma_bulk),
                    "num_likelihood_samples": int(M),
                    "num_posterior_samples": 0,
                    "num_certify_points": 0,
                    "posterior": "frequentist",
                    "inference": "frequentist",
                    "likelihood": "gaussian_copula_ridge",
                    "epsilon": float(epsilon),
                    "uuid": str(uuid4()),
                    "num_replications": int(num_replications),
                    "verbose": False,
                    "standardise_y": standardise_y,
                    "Ch_gap_ratio": Ch_gap_ratio,
                    "clip_extreme_y": clip_extreme_y,
                }
            )

    return experiment_set


def lv_california_housing_val_extra_f_div_baseline() -> List[Dict]:
    """California Housing geo-block CV experiment with additional chi-squared and KL DRO baselines."""
    experiment_set = lv_california_housing_val()

    gamma_set = getattr(_constants, "CALIFORNIA_HOUSING_GAMMA_SET", [0.01, 0.05, 0.10])
    num_replications = CALIFORNIA_HOUSING_NUM_REPLICATIONS
    standardise_y = getattr(_constants, "CALIFORNIA_HOUSING_STANDARDISE_Y", False)
    M = int(getattr(_constants, "CALIFORNIA_HOUSING_LV_MC_SAMPLES", 5000))
    Ch_gap_ratio = CALIFORNIA_HOUSING_GAP_RATIO
    clip_extreme_y = getattr(_constants, "CALIFORNIA_HOUSING_CLIP_Y_EXTREME", False)

    for gamma_bulk in gamma_set:
        for algorithm in ("chi2_lad", "kl_lad"):
            experiment_set.append(
                {
                    "algorithm": algorithm,
                    "dataset": "california_housing",
                    "dgp": "california_housing",
                    "num_observations": 0,
                    "num_test_observations": 0,
                    "dim": 8,
                    "contamination": 0.0,
                    "contamination_type": None,
                    "gamma_bulk": float(gamma_bulk),
                    "num_likelihood_samples": int(M),
                    "num_posterior_samples": 0,
                    "num_certify_points": 0,
                    "posterior": "frequentist",
                    "inference": "frequentist",
                    "likelihood": "gaussian_copula_ridge",
                    "epsilon": 0.1,
                    "calibrate_on_validation": True,
                    "uuid": str(uuid4()),
                    "num_replications": int(num_replications),
                    "verbose": False,
                    "standardise_y": standardise_y,
                    "Ch_gap_ratio": Ch_gap_ratio,
                    "clip_extreme_y": clip_extreme_y,
                }
            )

    return experiment_set

def kl_newsvendor_exp_1d() -> List[Dict]:
    experiment = []
    total_model_samples = 900
    num_replications = 100
    num_test_observations = NUM_TEST_OBSERVATIONS
    num_certify_points = 200
    for contamination, num_observations, (algorithm, dgp, likelihood, inference, posterior, dim), epsilon in itertools.product(
        [0.0],  # , 0.1, 0.2],
        [20],
        [
            # ("kl_dro_bas", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5),
            # ("kl_pp", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5),
            ("kl_bdro", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5)
            # ("kl_dro_bas", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_pp", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "npl_mmd", "npl", 5),
            # ("kl_dro_bas", "contaminated_normal", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_pp", "contaminated_normal", "normal_known_var", "bayes",  "normal_known_var", 1),
            # ("kl_bdro", "contaminated_normal", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "contaminated_normal", "normal_known_var", "npl_mmd", "npl", 1),
            # ("kl_pp", "contaminated_exp", "exponential", "bayes", "gamma", 1),
            # ("kl_dro_bas", "contaminated_exp", "exponential", "bayes", "gamma", 1),
            # ("kl_bdro", "contaminated_exp", "exponential", "bayes", "gamma", 1),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm in ["kl_bdro"]:
            num_posterior_samples = 90  # int(np.sqrt(total_model_samples))
            num_likelihood_samples = 10  # int(np.sqrt(total_model_samples))
        if algorithm in ["kl_dro_bas", "kl_pp"]:
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = total_model_samples
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": dim,
            "njobs": 1,
            "epsilon": epsilon,
            "ignore_dpp": True,
            "inference": inference,
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": num_test_observations,
            "num_certify_points": num_certify_points,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def mmd_newsvendor_exp_1d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 10
    num_posterior_samples = 90
    num_observations = 20
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference, posterior, dim), contamination, epsilon in itertools.product(
        [
            # ("dro_bas_mmd", "contaminated_normal", "normal_known_var", "npl_mmd", "npl", 1),
            # ("empirical_mmd", "contaminated_normal", "empirical", "empirical", "npl", 1)
            # ("dro_bas_mmd", "bimodal_univariate_gaussian", "normal_known_var", "npl_mmd", "npl", 1),
            # ("empirical_mmd", "bimodal_univariate_gaussian", "empirical", "empirical", "npl", 1)
            # ("dro_bas_mmd", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "npl_mmd", "npl", 5),
            # ("empirical_mmd", "bimodal_multivariate_gaussian", "empirical", "empirical", "npl", 5),
            ("dro_bas_mmd", "contaminated_exp", "exponential", "npl_mmd", "npl", 1),
            ("empirical_mmd", "contaminated_exp", "exponential", "empirical", "empirical", 1),
        ],
        [0.0, 0.1, 0.2],
        ROBAS_DRO_EPSILON_SET,
    ):
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "inference": inference,
            "kernel_name": "k_jax",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_certify_points": 200,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": 100,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def mmd_newsvendor_1d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 30
    num_posterior_samples = 30
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference, posterior), contamination, num_observations, epsilon in itertools.product(
        [
            ("kl_pp", "contaminated_exp", "exponential", "bayes", "gamma"),
            ("kl_pp", "contaminated_exp_large_outliers", "exponential", "bayes", "gamma"),
            ("kl_pp", "contaminated_exp_small_outliers", "exponential", "bayes", "gamma"),
        ],
        [0.0, 0.1, 0.2],
        [100],
        ROBAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = 900
            num_posterior_samples = 1
        if algorithm == "kl_pp":
            num_likelihood_samples = 900
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 1,
            "epsilon": epsilon,
            "inference": inference,
            "kernel_name": "k_jax",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_certify_points": 200,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def mmd_newsvendor_1d_missp() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 20
    num_posterior_samples = 20
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference), epsilon in itertools.product(
        [
            ("kl_bdro", "student_t", "normal", "npl_mmd"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if inference == "bayes":
            posterior = "normal_gamma"
        else:
            posterior = "npl"
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = 0.1
        if algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = 400
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 1,
            "epsilon": epsilon,
            "inference": inference,
            "kernel_name": "k_jax",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def mmd_newsvendor_5d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 20
    num_posterior_samples = 20
    num_observations = 400
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, contamination), epsilon in itertools.product(
        [
            ("dro_bas_mmd", "cont_multivariate_normal", "multivariate_normal_known_cov", 0.05),
            ("empirical_mmd", "cont_multivariate_normal", "empirical", 0.05),
            ("dro_bas_mmd", "cont_multivariate_normal", "multivariate_normal_known_cov", 0.1),
            ("empirical_mmd", "cont_multivariate_normal", "empirical", 0.1),
            ("dro_bas_mmd", "multivariate_normal_known_cov", "multivariate_normal_known_cov", 0.0),
            ("empirical_mmd", "multivariate_normal_known_cov", "empirical", 0.0),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if likelihood == "empirical":
            inference = "empirical"
        else:
            inference = "npl_mmd"
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
            "dim": 5,
            "epsilon": epsilon,
            "inference": inference,
            "kernel_name": "k_jax",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": "npl",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def kl_portfolio_synthetic() -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "portfolio_contaminated_multivariate_normal"
    num_replications = 100
    dim = 5
    epsilon_set = BAS_DRO_EPSILON_SET
    for algorithm, contamination, epsilon in itertools.product(
        [
            "kl_dro_bas",
            "kl_pp",
            "kl_bdro",
        ],
        [0.0, 0.1, 0.2],
        epsilon_set,
    ):
        inference = "bayes"
        likelihood = "multivariate_normal"
        posterior = "normal_inverse_wishart"
        if algorithm == "kl_dro_bas":
            num_likelihood_samples = 1
            num_posterior_samples = 1
        elif algorithm == "kl_pp":
            num_likelihood_samples = 900
            num_posterior_samples = 1
        elif algorithm == "kl_bdro":
            num_likelihood_samples = 10
            num_posterior_samples = 90
        else:
            raise ValueError(algorithm)
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "portfolio_synthetic",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "eta": np.nan,
            "ignore_dpp": True,
            "inference": inference,
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "normalise": False,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": 100,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": 100,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def mmd_portfolio_synthetic() -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "portfolio_contaminated_multivariate_normal"
    num_likelihood_samples = 10
    num_posterior_samples = 90
    num_replications = 100
    dim = 5
    epsilon_set = ROBAS_DRO_EPSILON_SET
    for (algorithm, likelihood), contamination, epsilon in itertools.product(
        [
            ("dro_bas_mmd", "multivariate_normal"),
            ("empirical_mmd", "empirical"),
        ],
        [0.0, 0.1, 0.2],
        epsilon_set,
    ):
        if algorithm == "empirical_mmd":
            inference = "empirical"
            posterior = "empirical"
            eta = np.nan
        elif algorithm == "dro_bas_mmd":
            inference = "npl_mmd"
            posterior = "npl"
            eta = 0.1
        else:
            raise ValueError(algorithm)
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "portfolio_synthetic",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "eta": eta,
            "inference": inference,
            "kernel_name": "k_comp",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "normalise": False,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": 100,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": 100,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def mmd_portfolio(mmc2_dir: Path) -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "DowJones"
    num_likelihood_samples = 10
    num_posterior_samples = 90
    epsilon_set = []
    for epsilon in ROBAS_DRO_EPSILON_SET:
        if epsilon <= 0.2:
            epsilon_set.append(epsilon)
    returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
    num_time_windows = get_num_time_windows(len(returns_df))
    num_stocks = len(returns_df.columns)
    # NOTE when using empirical, set likelihood to 'empirical'
    for (algorithm, likelihood), epsilon in itertools.product(
        [
            ("dro_bas_mmd", "multivariate_normal"),
            ("empirical_mmd", "empirical"),
        ],
        epsilon_set,
    ):
        if likelihood == "empirical":
            inference = "empirical"
            eta_set = [np.nan]
        else:
            inference = "npl_mmd"
            eta_set = [0.1]
        for eta in eta_set:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": num_stocks,
                "epsilon": epsilon,
                "eta": eta,
                "inference": inference,
                "kernel_name": "k_comp",
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "normalise": False,
                "num_certify_points": NUM_CERTIFY,
                "num_likelihood_samples": num_likelihood_samples,
                "num_observations": IN_SAMPLE_TIME_WINDOW,
                "num_posterior_samples": num_posterior_samples,
                "num_replications": num_time_windows,
                "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
                "posterior": "npl",
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment


def mmd_portfolio_crash(mmc2_dir: Path) -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "DowJones"
    num_likelihood_samples = 10
    num_posterior_samples = 90
    dgp = "DowJones-crash"
    epsilon_set = []
    for epsilon in ROBAS_DRO_EPSILON_SET:
        if epsilon <= 0.2:
            epsilon_set.append(epsilon)
    returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
    num_stocks = len(returns_df.columns)
    # NOTE when using empirical, set likelihood to 'empirical'
    for (algorithm, likelihood), epsilon in itertools.product(
        [
            ("dro_bas_mmd", "multivariate_normal"),
            ("empirical_mmd", "empirical"),
        ],
        epsilon_set,
    ):
        if likelihood == "empirical":
            inference = "empirical"
        else:
            inference = "npl_mmd"
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dataset": "portfolio",
            "dgp": dgp,
            "dim": num_stocks,
            "epsilon": epsilon,
            "inference": inference,
            "kernel_name": "k_comp",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": IN_SAMPLE_TIME_WINDOW,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": 1,
            "num_test_observations": IN_SAMPLE_TIME_WINDOW,
            "posterior": "npl",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def kl_portfolio_crash(mmc2_dir: Path) -> List[Dict]:
    """Portfolio experiment with a stock crash"""
    experiment = []
    dgp = "DowJones-crash"
    num_samples = 900
    for algorithm, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro", "kl_pp"],
        PORTFOLIO_EPSILON_SET,
    ):
        returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
        num_stocks = len(returns_df.columns)
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dataset": "portfolio",
            "dgp": dgp,
            "dim": num_stocks,
            "epsilon": epsilon,
            "ignore_dpp": True,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": "multivariate_normal",
            "njobs": 1,
            "num_likelihood_samples": get_num_likelihood_samples("portfolio", IN_SAMPLE_TIME_WINDOW, num_samples, algorithm),
            "num_observations": IN_SAMPLE_TIME_WINDOW,
            "num_posterior_samples": get_num_posterior_samples("portfolio", num_samples, algorithm),
            "num_replications": 1,
            "num_test_observations": IN_SAMPLE_TIME_WINDOW,
            "posterior": "normal_inverse_wishart",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def kl_portfolio(mmc2_dir: Path) -> List[Dict]:
    """KL Portfolio experiment with DRO-BAS vs BDRO"""
    experiment = []
    for algorithm, dgp, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro", "kl_pp", "kl_empirical"],
        ["DowJones"],
        PORTFOLIO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            if algorithm == "kl_dro_bas":
                total_model_samples_list = [1]
            elif algorithm == "kl_pp":
                total_model_samples_list = [900, 3600]
            elif algorithm == "kl_bdro":
                total_model_samples_list = [900]
            else:
                raise ValueError("Provide a supported algorithm")
            likelihood = "multivariate_normal"
            posterior = "normal_inverse_wishart"
            inference = "bayes"
        for num_samples in total_model_samples_list:
            returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
            num_time_windows = get_num_time_windows(len(returns_df))
            num_stocks = len(returns_df.columns)
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": num_stocks,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "likelihood": likelihood,
                "njobs": 1,
                "normalise": False,
                "num_likelihood_samples": get_num_likelihood_samples("portfolio", IN_SAMPLE_TIME_WINDOW, num_samples, algorithm),
                "num_observations": IN_SAMPLE_TIME_WINDOW,
                "num_posterior_samples": get_num_posterior_samples("portfolio", num_samples, algorithm),
                "num_replications": num_time_windows,
                "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
                "posterior": posterior,
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment


def lv_portfolio(mmc2_dir: Path) -> List[Dict]:
    """LV-BAS portfolio experiment: compare LV-BAS with KL-DRO-BAS, BDRO, KL-PP, and empirical under the same settings."""
    experiment: List[Dict] = []
    for algorithm, dgp, epsilon in itertools.product(
        #["kl_dro_bas", "kl_bdro", "kl_pp", "kl_empirical", "lv_bas"],
        ["lv_bas"],
        ["DowJones"],
        PORTFOLIO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            if algorithm == "kl_dro_bas":
                total_model_samples_list = [1]
            elif algorithm == "kl_pp":
                total_model_samples_list = [900, 3600]
            elif algorithm == "kl_bdro":
                total_model_samples_list = [900]
            elif algorithm == "lv_bas":
                # Use the same total-model-sample grid as KL-PP for LV-BAS.
                total_model_samples_list = [900, 3600]
            else:
                raise ValueError(f"Unsupported algorithm {algorithm!r} in lv_portfolio.")

            likelihood = "multivariate_normal"
            posterior = "normal_inverse_wishart"
            inference = "bayes"

        for num_samples in total_model_samples_list:
            returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
            num_time_windows = get_num_time_windows(len(returns_df))
            num_stocks = len(returns_df.columns)

            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": num_stocks,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "njobs": 1,
                "normalise": False,
                "num_likelihood_samples": get_num_likelihood_samples(
                    "portfolio", IN_SAMPLE_TIME_WINDOW, num_samples, algorithm
                ),
                "num_observations": IN_SAMPLE_TIME_WINDOW,
                "num_posterior_samples": get_num_posterior_samples(
                    "portfolio", num_samples, algorithm
                ),
                "num_replications": num_time_windows,
                "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
                "posterior": posterior,
                "uuid": str(uuid4()),
            }
            experiment.append(params)

    return experiment

def lv_portfolio_syn() -> List[Dict]:
    """Synthetic portfolio experiment for LV-BAS vs KL-BAS vs BDRO.

    - Dataset: 'portfolio_synthetic'
    - DGP:     'portfolio_gaussian_5d'
    - dim:     5
    - n_train: 1000, n_test: 250
    - Algorithms: lv_bas, kl_dro_bas, kl_pp, kl_bdro
    """
    from .constants import (
        PORTFOLIO_SYN_DIM,
        PORTFOLIO_SYN_NUM_OBSERVATIONS,
        PORTFOLIO_SYN_NUM_TEST_OBSERVATIONS,
        PORTFOLIO_SYN_NUM_REPLICATIONS,
        CONTAMINATION_LEVEL,
        CONTAMINATION_TYPE,
    )

    experiment: List[Dict] = []
    dgp = "portfolio_gaussian_5d"
    dim = PORTFOLIO_SYN_DIM
    #num_observations = PORTFOLIO_SYN_NUM_OBSERVATIONS
    num_test_observations = PORTFOLIO_SYN_NUM_TEST_OBSERVATIONS
    num_replications = PORTFOLIO_SYN_NUM_REPLICATIONS

    # Same epsilon grid as the real portfolio experiments
    epsilon_set = PORTFOLIO_EPSILON_SET
    contamination = CONTAMINATION_LEVEL
    contamination_type =CONTAMINATION_TYPE
    for algorithm, epsilon, num_observations in itertools.product(
        ["lv_bas", "kl_pp","kl_dro_bas", "kl_bdro"],
        epsilon_set,
        PORTFOLIO_SYN_NUM_OBSERVATIONS_SET,
    ):
        inference = "bayes"
        likelihood = "multivariate_normal"
        posterior = "normal_inverse_wishart"

        # Match sampling regimes used elsewhere (kl_portfolio_synthetic)
        if algorithm == "kl_dro_bas":
            num_likelihood_samples = 1    # closed-form in xi
            num_posterior_samples = 1
        elif algorithm == "kl_pp":
            num_likelihood_samples = 900  # SAA over posterior predictive
            num_posterior_samples = 1
        elif algorithm == "kl_bdro":
            num_likelihood_samples = 10   # BDRO: split budget between θ and ξ
            num_posterior_samples = 90
        elif algorithm == "lv_bas":
            # LV-BAS does not use the cvxpy problem; numbers here are unused
            num_likelihood_samples = 0
            num_posterior_samples = 0
        else:
            raise ValueError(f"Unexpected algorithm {algorithm!r} in lv_portfolio_syn")

        if algorithm == "lv_bas":
            # Set lengthscale according to reference t_ref
            epsilon = epsilon
        else:
            epsilon = 0.5 * (epsilon * LV_PORTFOLIO_T_SCALE) ** 2 
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "contamination_type": contamination_type,
            "dataset": "portfolio_synthetic",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "eta": 0.0,
            "ignore_dpp": True,
            "inference": inference,
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "normalise": False,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": num_test_observations,
            "posterior": posterior,
            "uuid": str(uuid4()),
            "lv_use_pc_bulk_geometry": LV_USE_PC_BULK_GEOMETRY,
        }
        experiment.append(params)

    return experiment


def rw_civilcomments() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for algorithm in RW_ALGORITHMS:
        eps_grid = [0.0] if algorithm in ("rw_erm","rw_groupdro") else RW_EPSILON_GRID
        for eps in eps_grid:
            if algorithm == "rw_chi2_dro" and float(eps) >= 1.0:
                continue  # chi2-DRO requires epsilon < 1
            for gamma in RW_GAMMA_GRID:
                params = dict(
                    uuid=str(uuid4()),
                    algorithm=algorithm,
                    dataset="rw_civilcomments",
                    epsilon=float(eps),
                    gamma=float(gamma),
                    num_replications=RW_NUM_REPLICATIONS,
                    dgp="NA",
                    contamination=0.0,
                    dim=1,
                    num_observations=NUM_OBSERVATIONS,
                    num_test_observations=NUM_TEST_OBSERVATIONS,
                    ignore_dpp=True,
                    njobs=1,
                    normalise=False,
                    verbose=False,
                    do_threshold_calibration = DO_THRESHOLD_CALIBRATION
                )
                params.update(RW_EXPERIMENT_DEFAULTS)
                out.append(params)
    return out

def rw_civilcomments_bulk_ablation() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for algorithm in RW_ALGORITHMS_BULK_ABLATION:
        eps_grid = [0.0] if algorithm in ("rw_erm", "rw_groupdro", "rw_erm_b") else RW_EPSILON_GRID
        for eps in eps_grid:
            if algorithm in ("rw_chi2_dro", "rw_chi2_dro_b") and float(eps) >= 1.0:
                continue  # chi2-DRO requires epsilon < 1
            for gamma in RW_GAMMA_GRID:
                params = dict(
                    uuid=str(uuid4()),
                    algorithm=algorithm,
                    dataset="rw_civilcomments",
                    epsilon=float(eps),
                    gamma=float(gamma),
                    num_replications=RW_NUM_REPLICATIONS,
                    dgp="NA",
                    contamination=0.0,
                    dim=1,
                    num_observations=NUM_OBSERVATIONS,
                    num_test_observations=NUM_TEST_OBSERVATIONS,
                    ignore_dpp=True,
                    njobs=1,
                    normalise=False,
                    verbose=False,
                    do_threshold_calibration=DO_THRESHOLD_CALIBRATION,
                )
                params.update(RW_EXPERIMENT_DEFAULTS)
                out.append(params)
    return out


def compare_solve() -> List[Dict]:
    """Compares the original grid-search algorithm and cvxpy algorithms"""
    experiment = []
    for algorithm, dgp, epsilon, (posterior, likelihood) in itertools.product(
        ["kl_bdro", "bdro_grid_search", "kl_dro_bas"],
        ["truncated_normal"],
        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        [("gamma", "exponential"), ("normal_gamma", "normal")],
    ):
        num_posterior_samples = NUM_POSTERIOR_SAMPLES
        if algorithm == "kl_dro_bas" and posterior != "normal_gamma":
            continue  # skip if the posterior doesn't match our algorithm
        elif algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 1,
            "epsilon": epsilon,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_likelihood_samples": NUM_LIKELIHOOD_SAMPLES,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment
