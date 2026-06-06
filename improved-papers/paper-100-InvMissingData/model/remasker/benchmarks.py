# stdlib
import copy
from time import time
from typing import Any
import warnings

# third party
from IPython.display import display
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

# hyperimpute absolute
import hyperimpute.logger as log
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.distributions import enable_reproducible_results
from hyperimpute.utils.metrics import generate_score, print_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


enable_reproducible_results()

# warnings.filterwarnings("ignore")

imputers = Imputers()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ampute(
    x: pd.DataFrame,
    mechanism: str,
    p_miss: float,
    column_limit: int = 8,
    sample_columns: bool = True,
) -> tuple:
    columns = x.columns
    column_limit = min(len(columns), column_limit)

    if sample_columns:
        sampled_columns = columns[
            np.random.choice(len(columns), size=column_limit, replace=False)
        ]
    else:
        sampled_columns = columns[list(range(column_limit))]

    x_simulated = simulate_nan(
        x[sampled_columns].values, p_miss, mechanism, sample_columns=sample_columns
    )

    isolated_mask = pd.DataFrame(x_simulated["mask"], columns=sampled_columns)
    isolated_x_miss = pd.DataFrame(x_simulated["X_incomp"], columns=sampled_columns)

    mask = pd.DataFrame(np.zeros(x.shape), columns=columns)
    mask[sampled_columns] = pd.DataFrame(isolated_mask, columns=sampled_columns)

    x_miss = pd.DataFrame(x.copy(), columns=columns)
    x_miss[sampled_columns] = isolated_x_miss

    return (
        pd.DataFrame(x, columns=columns),
        x_miss,
        mask,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def scale_data(X: pd.DataFrame) -> pd.DataFrame:
    preproc = MinMaxScaler()
    cols = X.columns
    return pd.DataFrame(preproc.fit_transform(X), columns=cols)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def simulate_scenarios(
    X: pd.DataFrame, column_limit: int = 8, sample_columns: bool = True
) -> pd.DataFrame:
    X = scale_data(X)

    datasets: dict = {}

    mechanisms = ["MAR", "MNAR", "MCAR"]
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

    for ampute_mechanism in mechanisms:
        for p_miss in percentages:
            if ampute_mechanism not in datasets:
                datasets[ampute_mechanism] = {}

            datasets[ampute_mechanism][p_miss] = ampute(
                X,
                ampute_mechanism,
                p_miss,
                column_limit=column_limit,
                sample_columns=sample_columns,
            )

    return datasets


def ws_score(imputed: pd.DataFrame, ground: pd.DataFrame) -> pd.DataFrame:
    res = 0
    for col in range(ground.shape[1]):
        res += wasserstein_distance(
            np.asarray(ground)[:, col], np.asarray(imputed)[:, col]
        )
    return res


# @validate_arguments(config=dict(arbitrary_types_allowed=True))
def benchmark_model(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_miss: pd.DataFrame,
    mask: pd.DataFrame,
) -> tuple:
    
    start = time()
    imputed = model.fit_transform(X_miss.copy())
    
    if len(np.unique(y)) > 20:
        auroc_score = 0
    else:
        clf = LogisticRegression(solver="liblinear", random_state=0).fit(np.asarray(imputed), np.asarray(y))
        if len(np.unique(np.asarray(y))) > 2:
            auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(imputed)), multi_class='ovr')
        else:
            auroc_score = roc_auc_score(np.asarray(y), clf.predict_proba(np.asarray(imputed))[:,1])

    distribution_score = ws_score(imputed, X)
    rmse_score = RMSE(np.asarray(imputed), np.asarray(X), np.asarray(mask))
    
    log.info(f"benchmark {model.name()} took {time() - start}")
    return rmse_score, distribution_score, auroc_score


def benchmark_standard(
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_miss: pd.DataFrame,
    mask: pd.DataFrame,
) -> tuple:

    imputer = imputers.get(model_name)
    return benchmark_model(model_name, imputer, X, y, X_miss, mask)


#  @validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_dataset(
    name: str,
    evaluated_model: Any,
    X_raw: pd.DataFrame,
    y: pd.DataFrame,
    ref_methods: list = ["mean", "missforest", "ice", "gain", "sinkhorn", "softimpute"],
    scenarios: list = ["MAR", "MCAR", "MNAR"],
    miss_pct: list = [0.1, 0.3, 0.5],
    sample_columns: bool = True,
) -> tuple:

    imputation_scenarios = simulate_scenarios(X_raw, sample_columns=sample_columns)

    rmse_results: dict = {}
    distr_results: dict = {}
    auroc_results: dict = {}
    
    for scenario in scenarios:
        rmse_results[scenario] = {}
        distr_results[scenario] = {}
        auroc_results[scenario] = {}

        for missingness in miss_pct:
            log.debug(f"  > eval {scenario} {missingness}")
            rmse_results[scenario][missingness] = {}
            distr_results[scenario][missingness] = {}
            auroc_results[scenario][missingness] = {}

            try:
                
                x, x_miss, mask = imputation_scenarios[scenario][missingness]
                rmse_score, distribution_score, auroc_score = benchmark_model(name, copy.deepcopy(evaluated_model), x, y, x_miss, mask)
                rmse_results[scenario][missingness]["our"] = rmse_score
                distr_results[scenario][missingness]["our"] = distribution_score
                auroc_results[scenario][missingness]["our"] = auroc_score 

                for method in ref_methods:
                    x, x_miss, mask = imputation_scenarios[scenario][missingness]
                    rmse_score, distribution_score, auroc_score = benchmark_standard(method, x, y, x_miss, mask)
                    rmse_results[scenario][missingness][method] = rmse_score
                    distr_results[scenario][missingness][method] = distribution_score
                    auroc_results[scenario][missingness][method] = auroc_score 

            except BaseException as e:
                log.error(f"scenario failed {str(e)}")
                print(e)
                continue

    return rmse_results, distr_results, auroc_results


def compare_models(
    name: str,
    evaluated_model: Any,
    X_raw: pd.DataFrame,
    y: pd.DataFrame,
    ref_methods: list = ["mean", "missforest", "ice", "gain", "sinkhorn", "softimpute"],
    scenarios: list = ["MNAR"],
    miss_pct: list = [0.1, 0.3, 0.5, 0.7],
    n_iter: int = 2,
    sample_columns: bool = True,
    display_results: bool = True,
    n_jobs: int = 1,
) -> dict:

    dispatcher = Parallel(n_jobs=n_jobs)
    start = time()

    def add_metrics(
        store: dict, scenario: str, missingness: float, method: str, score: float
    ) -> None:
        if scenario not in store:
            store[scenario] = {}
        if missingness not in store[scenario]:
            store[scenario][missingness] = {}
        if method not in store[scenario][missingness]:
            store[scenario][missingness][method] = []

        store[scenario][missingness][method].append(score)

    rmse_results_dict: dict = {}
    distr_results_dict: dict = {}
    auroc_results_dict: dict = {}

    def eval_local(it: int) -> Any:
        enable_reproducible_results(it)
        log.debug(f"> evaluation trial {it}")
        return evaluate_dataset(
            name=name,
            evaluated_model=evaluated_model,
            X_raw=X_raw,
            y=y,
            ref_methods=ref_methods,
            scenarios=scenarios,
            miss_pct=miss_pct,
            sample_columns=sample_columns,
        )

    repeated_evals_results = dispatcher(delayed(eval_local)(it) for it in range(n_iter))

    for (
        local_rmse_results,
        local_distr_results,
        local_auroc_results
    ) in repeated_evals_results:
        for scenario in local_rmse_results:
            for missingness in local_rmse_results[scenario]:
                for method in local_rmse_results[scenario][missingness]:
                    add_metrics(
                        rmse_results_dict,
                        scenario,
                        missingness,
                        method,
                        local_rmse_results[scenario][missingness][method],
                    )
                    add_metrics(
                        distr_results_dict,
                        scenario,
                        missingness,
                        method,
                        local_distr_results[scenario][missingness][method],
                    )
                    add_metrics(
                        auroc_results_dict,
                        scenario,
                        missingness,
                        method,
                        local_auroc_results[scenario][missingness][method],
                    )


    rmse_results = []
    distr_results = []
    auroc_results = []

    rmse_str_results = []
    distr_str_results = []
    auroc_str_results = []

    for scenario in rmse_results_dict:

        for missingness in rmse_results_dict[scenario]:

            local_rmse_str_results = [scenario, missingness]
            local_distr_str_results = [scenario, missingness]
            local_auroc_str_results = [scenario, missingness]
            
            local_rmse_results = [scenario, missingness]
            local_distr_results = [scenario, missingness]
            local_auroc_results = [scenario, missingness]
            
            for method in ["our"] + ref_methods:
                rmse_mean, rmse_std = generate_score(
                    rmse_results_dict[scenario][missingness][method]
                )
                rmse_str = print_score((rmse_mean, rmse_std))
                
                distr_mean, distr_std = generate_score(
                    distr_results_dict[scenario][missingness][method]
                )
                distr_str = print_score((distr_mean, distr_std))

                auroc_mean, auroc_std = generate_score(
                    auroc_results_dict[scenario][missingness][method]
                )
                auroc_str = print_score((auroc_mean, auroc_std))

                local_rmse_str_results.append(rmse_str)
                local_rmse_results.append((rmse_mean, rmse_std))

                local_distr_str_results.append(distr_str)
                local_distr_results.append((distr_mean, distr_std))
                
                local_auroc_str_results.append(auroc_str)
                local_auroc_results.append((auroc_mean, auroc_std))
                
            rmse_str_results.append(local_rmse_str_results)
            rmse_results.append(local_rmse_results)
            distr_str_results.append(local_distr_str_results)
            distr_results.append(local_distr_results)
            auroc_str_results.append(local_auroc_str_results)
            auroc_results.append(local_auroc_results)
            
    if display_results:
        log.info(f"benchmark took {time() - start}")
        headers = (
            ["Scenario", "miss_pct [0, 1]"]
            + [f"Evaluated: {evaluated_model.name()}"]
            + ref_methods
        )

        sep = "\n==========================================================\n\n"
        print("RMSE score")
        data = pd.DataFrame(rmse_str_results, columns=headers)
        display(data)

        print(sep + "Wasserstein score")
        data = pd.DataFrame(distr_str_results, columns=headers)
        display(data)

        print(sep + "AUROC score")
        data = pd.DataFrame(auroc_str_results, columns=headers)
        display(data)


    return {
        "headers": headers,
        "rmse": rmse_results,
        "wasserstein": distr_results,
        'auroc': auroc_results
    }