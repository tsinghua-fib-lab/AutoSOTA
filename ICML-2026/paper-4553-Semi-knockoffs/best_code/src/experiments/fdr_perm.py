import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import argparse
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone, BaseEstimator
from utils import GenSynthDataset
from regression_sampler import RegressionSamplerSingle
from hidimstat import D0CRT
from pyhrt.hrt import hrt
from loco import LOCO
from semi_KO import Semi_KO
from sobol_CPI import Sobol_CPI
from sklearn.ensemble import StackingRegressor
from hidimstat.knockoffs import model_x_knockoff
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold




# ------------------------------
# Argument parser
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Comparison of Semi_KO vs dCRT, HRT")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--setting", type=str, choices=['adjacent','spaced','sinusoidal','hidim','nongauss','poly', 'sin', 'cos', 'interact_sin', 'interact_pairwise', 'interact_highorder', 'interact_oscillatory', 'gauss_mixt', 'log', 'square'], required=True)
    parser.add_argument("--model", type=str, choices=['lasso','RF','NN','GB','SL'], required=True)
    return parser.parse_args()

# ------------------------------
# Base model getter
# ------------------------------
def get_base_model(model_name, random_state, n_jobs=1):
    if model_name == 'lasso':
        return LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=random_state)
    elif model_name == 'RF':
        return RandomForestRegressor(n_estimators=200, max_depth=None, random_state=random_state)
    elif model_name == 'NN':
        return MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=random_state)
    elif model_name == 'GB':
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=random_state)
    elif model_name == 'SL':
        estimators = [
            ('lasso', LassoCV(alphas=np.logspace(-3, 3, 10), cv=3, random_state=random_state)),
            ('rf', RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=n_jobs)),
            ('gb', GradientBoostingRegressor(max_depth=3, random_state=random_state)),
        ]
        return StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), n_jobs=n_jobs)

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ------------------------------
# Main function
# ------------------------------
def main(args):
    n = 300
    n_r2 = 1000
    seed = args.seed
    setting = args.setting
    base_model_name = args.model
    rng = np.random.RandomState(seed)

    # Determine number of features depending on setting
    if setting == 'hidim':
        p = 200
    else:
        p = 50

    # Generate dataset large enough to split into training and R2 test
    X_comp, y_comp, true_imp = GenSynthDataset(n=n+n_r2, d=p, setting=setting, seed=seed)

    # Split
    X = X_comp[:n]
    y = y_comp[:n]

    X_r2 = X_comp[n:]
    y_r2 = y_comp[n:]

    # Initialize storage
    num_methods = 9
    estim_imp = np.zeros((num_methods, p))
    execution_time = np.zeros(num_methods)
    r2_values = np.zeros(num_methods)

    # Base model
    base_model = get_base_model(base_model_name, seed)

    # DCRT
    model = clone(base_model)
    start_time = time.time()
    d0crt = D0CRT(
        estimator=model,
        screening_threshold=None,
        random_state=seed,
    )
    d0crt.fit_importance(X, y)
    execution_time[3] = time.time() - start_time
    estim_imp[3,:p]= d0crt.pvalues_.reshape((p,))

    dcrt_model = LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed)
    dcrt_model.fit(X, y)
    r2_values[3] = r2_score(y_r2, dcrt_model.predict(X_r2))

    # CPI-KO
    model = clone(base_model)
    start_time = time.time()
    model.fit(X, y)
    tr_KO_time = time.time()-start_time

    start_time = time.time()
    cpi_knockoffs= Semi_KO(
        estimator=model,
        imputation_model=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5),
        imputation_model_y=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5),
        random_state=seed,
        n_jobs=1)
    cpi_knockoffs.fit(X, y)
    imp_time_Semi_KO = time.time()-start_time
    r2_values[1:2] = r2_score(y_r2, model.predict(X_r2))

    start_time = time.time()
    Semi_KO_importance = cpi_knockoffs.score(X, y, n_perm=1,  p_val='wilcox')
    execution_time[2] = time.time() - start_time + tr_KO_time + imp_time_Semi_KO
    estim_imp[2, :p]= Semi_KO_importance["pval"].reshape((p,))

    execution_time[1] = execution_time[2]
    estim_imp[1, :p]= Semi_KO_importance["importance"].reshape((p,))

    # CPI-KO more permutations

    r2_values[5:8] = r2_values[1]

    start_time = time.time()
    Semi_KO_importance = cpi_knockoffs.score(X, y, n_perm=5,  p_val='wilcox')
    execution_time[6] = time.time() - start_time + tr_KO_time + imp_time_Semi_KO
    estim_imp[6, :p]= Semi_KO_importance["pval"].reshape((p,))

    execution_time[5] = execution_time[6]
    estim_imp[5, :p]= Semi_KO_importance["importance"].reshape((p,))

    start_time = time.time()
    Semi_KO_importance = cpi_knockoffs.score(X, y, n_perm=10,  p_val='wilcox')
    execution_time[8] = time.time() - start_time + tr_KO_time + imp_time_Semi_KO
    estim_imp[8, :p]= Semi_KO_importance["pval"].reshape((p,))

    execution_time[7] = execution_time[8]
    estim_imp[7, :p]= Semi_KO_importance["importance"].reshape((p,))


    # HRT
    model = clone(base_model)
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model.fit(X_train, y_train)
    tr_time = time.time()-start_time

    start_time = time.time()
    tstat_fn = lambda X_eval: ((y_test - model.predict(X_eval))**2).mean()
    for j in range(p):
        # Run the HRT
        sampler = RegressionSamplerSingle(
            estimator=RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5),
            X_train=X_train,
            X_test=X_test,
            covariate_index=j
        )
        sampler.fit()

        def conditional(X_eval=None):
            sample = sampler.sampler().ravel()  # shape (n_test,)
            # HRT expects length(len(lower)+len(upper)) -> here 2
            probs = np.ones(2)                   # dummy CI
            return sample, probs
        result = hrt(j,tstat_fn,X_train,X_test=X_test,conditional=conditional)
        estim_imp[4, j]= result['p_value']
    execution_time[4] = time.time() - start_time + tr_time
    r2_values[4] = r2_score(y_r2, model.predict(X_r2))

    # MX KNOCKOFFS

    start_time = time.time()
    selected, test_scores, threshold, X_tildes = model_x_knockoff(
        X,
        y,
        estimator=LassoCV(
            n_jobs=1,
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            random_state=seed,
        ),
        n_bootstraps=1,
        random_state=seed,
    )
    execution_time[0] = time.time() - start_time
    estim_imp[0, :p]= test_scores.reshape((p,))
    r2_values[0] = r2_values[3]




    methods = [
                "Knockoff","Semi_KO","Semi_KO_Wilcox",
                'dCRT', 'HRT', "Semi_KO_perm5", "Semi_KO_Wilcox_perm5", "Semi_KO_perm10", "Semi_KO_Wilcox_perm10"
            ]
    f_res = pd.DataFrame()
    for i, method in enumerate(methods):
        row = {"method": method, "n": n, "tr_time": execution_time[i], "r2_test": r2_values[i]}
        for j in range(p):
            row[f"tr_V{j}"] = true_imp[j]
            row[f"pval{j}"] = estim_imp[i, j]
        f_res = pd.concat([f_res, pd.DataFrame([row])], ignore_index=True)

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/res_csv/KO_perm2_{setting}_{base_model_name}_seed{seed}.csv"))

    f_res.to_csv(csv_path, index=False)

    print(f_res.head())


if __name__ == "__main__":
    args = parse_args()
    main(args)

    