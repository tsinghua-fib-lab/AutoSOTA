import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import argparse
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone, BaseEstimator
from utils import GenSynthDataset, add_correlated_feature, get_real_dataset
from regression_sampler import RegressionSamplerSingle
from hidimstat import D0CRT
from pyhrt.hrt import hrt
from loco import LOCO
from semi_KO import Semi_KO
from sobol_CPI import Sobol_CPI
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV, LinearRegression


# ------------------------------
# Argument parser
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Comparison of Semi_KO vs LOCO, dCRT, HRT, CPI")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--setting", type=str, choices=['wdbc','california','','diabetes','wine-red','wine-white'], required=True)
    parser.add_argument("--model", type=str, choices=['lasso','RF','NN','GB','SL'], required=True)
    parser.add_argument("--imputer", type=str, choices=['lasso','elasticnet','ridge', 'RF','NN','GB','SL'], default=None)
    parser.add_argument("--n", type=int, default=300)
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
    

def get_imputation_model(model_name, random_state, n_jobs=1):
    if model_name == 'lasso':
        return LassoCV(
            alphas=np.logspace(-3, 3, 10),
            cv=5,
            random_state=random_state
        )

    elif model_name == 'elasticnet':
        return ElasticNetCV(
            alphas=np.logspace(-3, 3, 10),
            l1_ratio=[0.1, 0.5, 0.9, 1.0],   
            cv=5,
            random_state=random_state
        )

    elif model_name == 'ridge':
        return RidgeCV(
            alphas=np.logspace(-3, 3, 10),
            cv=5
        )

    elif model_name == 'RF':
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=n_jobs
        )

    elif model_name == 'NN':
        return MLPRegressor(
            hidden_layer_sizes=(50, 50),
            max_iter=1000,
            random_state=random_state
        )

    elif model_name == 'GB':
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=random_state
        )

    elif model_name == 'SL':
        estimators = [
            ('lasso', LassoCV(alphas=np.logspace(-3, 3, 10), cv=3, random_state=random_state)),
            ('enet', ElasticNetCV(alphas=np.logspace(-3, 3, 10),
                                  l1_ratio=[0.1, 0.5, 0.9], cv=3,
                                  random_state=random_state)),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 10), cv=3)),
            ('gb', GradientBoostingRegressor(max_depth=3, random_state=random_state)),
        ]

        return StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            n_jobs=n_jobs
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ------------------------------
# Main function
# ------------------------------
def main(args):
    n = args.n
    seed = args.seed
    setting = args.setting
    base_model_name = args.model
    imputer_model_name = args.imputer
    rng = np.random.RandomState(seed)


    # Generate dataset large enough to split into training and R2 test
    X_orig, y, names = get_real_dataset(setting)
    X = add_correlated_feature(X_orig, target_corr=0.6, seed=args.seed)
    n, p = X.shape

    # Initialize storage
    num_methods = 33
    p_val = np.zeros((num_methods, p))
    execution_time = np.zeros(num_methods)
    #r2_values = np.zeros(num_methods)

    # Base model
    base_model = get_base_model(base_model_name, seed)
    if imputer_model_name == None: 
        imputer = get_imputation_model("ridge", seed)
    else:
        imputer = get_imputation_model(imputer_model_name, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    # ------------------------------
    # Method 0: LOCO-Williamson
    # ------------------------------
    start_time = time.time()
    for j in range(p):
        model_j = clone(base_model)
        import vimpy
        vimp = vimpy.vim(y=y, x=X, s=j, pred_func=model_j, measure_type="r_squared")
        vimp.get_point_est()
        vimp.get_influence_function()
        vimp.get_se()
        vimp.get_ci()
        vimp.hypothesis_test(alpha=0.05, delta=0)
        p_val[0, j] = vimp.p_value_
    execution_time[0] = time.time() - start_time
   

    # ------------------------------
    # Method 1-15: Sobol-CPI variations
    # ------------------------------
    start_time = time.time()
    model_sobol = clone(base_model) # Model trained with train/test split
    model_sobol.fit(X_train, y_train)
    train_time = time.time() - start_time
    start_time = time.time()
    sobol_CPI_obj = Sobol_CPI(estimator=model_sobol, imputation_model=LassoCV(alphas=np.logspace(-3,3,10), cv=5, random_state=seed),
                               n_permutations=1, random_state=seed, n_jobs=1)
    sobol_CPI_obj.fit(X_train, y_train)
    sobol_imputation_time = time.time() - start_time
    ptype_list = ['emp_var', 'corrected_sqrt', 'corrected_n', 'corrected_n', 'corrected_sqd']
    n_cal_list = [1, 10, 100]

    for group_idx, n_cal in enumerate(n_cal_list):
        for i, ptype in enumerate(ptype_list):
            bootstrap = False
            if ptype == 'corrected_n' and i >= 3:
                bootstrap = True
            if ptype == 'corrected_sqd' and i >= 3:
                bootstrap = True
            idx = group_idx*5 + i + 1  # +1 because method 0 is LOCO-W
            start_time = time.time()
            score_dict = sobol_CPI_obj.score(X_test, y_test, n_cal=n_cal, p_val=ptype, bootstrap=bootstrap)
            p_val[idx, :] = score_dict["pval"].reshape(p)
            execution_time[idx] = time.time() - start_time + sobol_imputation_time + train_time
    # R2
    #r2_values[0:25] = r2_score(y_r2, model_sobol.predict(X_r2)) # Also for LOCO_W, LOCO, S-CPI_W/ST

    # ------------------------------
    # Method 16-20: LOCO variants
    # ------------------------------
    start_time = time.time()
    loco_obj = LOCO(estimator=model_sobol, random_state=seed, loss=mean_squared_error, n_jobs=1)
    loco_obj.fit(X_train, y_train)
    loco_imputation_time = time.time() - start_time
    ptype_list = [
        ('emp_var', False),
        ('corrected_sqrt', False),
        ('corrected_n', False),
        ('corrected_n', True),
        ('corrected_sqd', True),
    ]

    for idx, (ptype, bootstrap) in enumerate(ptype_list):
        start_time = time.time()
        loco_importance = loco_obj.score(X_test, y_test, p_val=ptype, bootstrap=bootstrap)
        p_val[16+idx, :] = loco_importance["pval"].reshape(p)
        execution_time[16+idx] = time.time() - start_time + train_time + loco_imputation_time


    # ------------------------------
    # Method 21-22: Sobol-CPI n_cal variants
    # ------------------------------
    for idx, ptype in enumerate(['sign_test','wilcox']):
        start_time = time.time()
        p_val[21+idx, :] = sobol_CPI_obj.score(X_test, y_test, n_cal=1, p_val=ptype)["pval"].reshape(p)
        execution_time[21+idx] = time.time() - start_time + sobol_imputation_time + train_time

    # ------------------------------
    # Method 23-24: LOCO sign_test/wilcox
    # ------------------------------
    for idx, ptype in enumerate(['sign_test','wilcox']):
        start_time = time.time()
        p_val[23+idx, :] = loco_obj.score(X_test, y_test, p_val=ptype)["pval"].reshape(p)
        execution_time[23:25] = time.time() - start_time + loco_imputation_time + train_time 

    # ------------------------------
    # Method 25-26: Semi-KO
    # ------------------------------
    model_ko = clone(base_model)
    nu_j = clone(imputer)
    rho_j = clone(imputer)
    start_time = time.time()
    model_ko.fit(X, y)
    cpi_knockoffs = Semi_KO(
        estimator=model_ko,
        imputation_model=nu_j,
        imputation_model_y=rho_j,
        random_state=seed,
        n_jobs=1
    )
    cpi_knockoffs.fit(X, y)
    cpi_knockoffs_time = time.time() - start_time
    for idx, ptype in enumerate(['sign_test','wilcox']):
        start_time = time.time()
        p_val[25+idx, :] = cpi_knockoffs.score(X, y, n_perm=1, p_val=ptype)["pval"].reshape(p)
        execution_time[25+idx] = time.time() - start_time + cpi_knockoffs_time
    #r2_values[25:27] = r2_score(y_r2, model_ko.predict(X_r2))

    for idx, ptype in enumerate(['sign_test','wilcox']):
        start_time = time.time()
        p_val[29+idx, :] = cpi_knockoffs.score(X, y, n_perm=5, p_val=ptype)["pval"].reshape(p)
        execution_time[29+idx] = time.time() - start_time + cpi_knockoffs_time
    #r2_values[29:31] = r2_score(y_r2, model_ko.predict(X_r2))

    for idx, ptype in enumerate(['sign_test','wilcox']):
        start_time = time.time()
        p_val[31+idx, :] = cpi_knockoffs.score(X, y, n_perm=10, p_val=ptype)["pval"].reshape(p)
        execution_time[31+idx] = time.time() - start_time + cpi_knockoffs_time
    #r2_values[31:33] = r2_score(y_r2, model_ko.predict(X_r2))


    # ------------------------------
    # Method 27: dCRT
    # ------------------------------
    start_time = time.time()
    d0crt = D0CRT(estimator=clone(base_model), screening_threshold=None, random_state=seed)
    d0crt.fit_importance(X, y)
    p_val[27, :] = d0crt.pvalues_.reshape(p)
    execution_time[27] = time.time() - start_time
    dcrt_model = LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed)
    dcrt_model.fit(X, y)
    #r2_values[27] = r2_score(y_r2, dcrt_model.predict(X_r2))

    # ------------------------------
    # Method 28: HRT
    # ------------------------------
    start_time = time.time()
    for j in range(p):
        sampler = RegressionSamplerSingle(estimator=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed), X_train=X_train, X_test=X_test, covariate_index=j)
        sampler.fit()
        def conditional(X_eval=None):
            sample = sampler.sampler().ravel()
            probs = np.ones(2)
            return sample, probs
        result = hrt(j, lambda X_eval: ((y_test - model_sobol.predict(X_eval))**2).mean(),
                     X_train, X_test=X_test, conditional=conditional)
        p_val[28, j] = result['p_value']
    execution_time[28] = time.time() - start_time + train_time
    #r2_values[28] = r2_score(y_r2, model_sobol.predict(X_r2))

    # ------------------------------
    # Save results to CSV
    # ------------------------------
    methods = [
        "LOCO-W","CPI","CPI_sqrt","CPI_n","CPI_bt","CPI_sqd",
        "S-CPI","S-CPI_sqrt","S-CPI_n","S-CPI_bt","S-CPI_sqd",
        "S-CPI2","S-CPI2_sqrt","S-CPI2_n","S-CPI2_bt","S-CPI2_sqd",
        "LOCO","LOCO_sqrt","LOCO_n","LOCO_bt","LOCO_sqd",
        "S-CPI_ST","S-CPI_Wilcox","LOCO_ST","LOCO_Wilcox",
        "Semi_KO_ST","Semi_KO_Wilcox","dCRT","HRT", "Semi_KO_ST_perm5","Semi_KO_Wilcox_perm5", 
        "Semi_KO_ST_perm10","Semi_KO_Wilcox_perm10"
    ]
    f_res = pd.DataFrame()
    for i, method in enumerate(methods):
        row = {"method": method, "n": n, "tr_time": execution_time[i]}
        for j in range(p):
            row[f"pval{j}"] = p_val[i, j]
        f_res = pd.concat([f_res, pd.DataFrame([row])], ignore_index=True)


    # base name
    base = f"p_values_{setting}_{base_model_name}"

    # add imputer if needed
    if imputer_model_name is not None:
        base += f"_imp{imputer_model_name}"

    # add seed
    base += f"_seed{seed}.csv"

    # full path
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"../../results/res_csv/real_data/{base}")
    )
    f_res.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
