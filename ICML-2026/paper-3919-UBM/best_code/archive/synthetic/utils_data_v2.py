from utils_v2 import *

import numpy as np
import pandas as pd
import itertools

from numpy.random import choice
from numpy.random import binomial
from numpy.random import uniform as unif

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from collections import defaultdict

class dummy_model:
    def __init__(self): 
        pass 

    def fit(self):
        pass 

    def predict_proba(self, X): 
        return np.ones((X.shape[0],2))

    def predict(self, X): 
        return np.ones(X.shape)

def fit_model(df, predictors, target, filter_fit='index==index', model='LR'):
    if model == 'LR': 
        model = LogisticRegression(max_iter=1000)
        model.fit(df.query(filter_fit)[predictors], df.query(filter_fit)[target])
    elif model == 'RF': 
        # model = RandomForestClassifier(random_state=42)
        # param_dist = {
        #     'n_estimators': [100, 500], 
        #     'max_depth': [None, 10, 20],
        #     # 'max_features': ['sqrt', 'log2'],
        #     'min_samples_split': [2, 10]
        # }
        # random_search = RandomizedSearchCV(
        #     estimator=model,
        #     param_distributions=param_dist,
        #     n_iter=5,
        #     cv=5,
        #     scoring='roc_auc',
        #     random_state=42,
        #     n_jobs=-1
        # )
        # random_search.fit(df.query(filter_fit)[predictors], df.query(filter_fit)[target])
        # model = random_search.best_estimator_
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=2,
            random_state=42
        )
        model.fit(df.query(filter_fit)[predictors], df.query(filter_fit)[target])
    elif model == 'GB':
        model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            random_state=42
        )
        model.fit(df.query(filter_fit)[predictors], df.query(filter_fit)[target])
        
    return model

def fit_models(df, predictors, is_rct=False, model="LR"):
    flt_str = {"S": "index==index", "A": "S==1", "Y0": "S==1 & A==0", "Y1": "S==1 & A==1"}
    models = {}

    for key in flt_str:
        if is_rct and key == "S": 
            DM = dummy_model()
            models[key] = DM
        else: 
            if model == "LR": 
                models[key] = fit_model(df, predictors, key, flt_str[key], model=model)
            elif model == "RF": 
                if key == "Y1" or key == "Y0":
                    models[key] = fit_model(df, predictors, key, flt_str[key], model="RF")
                else: 
                    models[key] = fit_model(df, predictors, key, flt_str[key], model="LR")

    return models

def make_preds(df, predictors, models):
    flt_str = {"S": "index==index", "A": "S==1", "Y0": "S==1 & A==0", "Y1": "S==1 & A==1"}

    for key, model in models.items():
        df[f"hat_P({key}=1)"] = model.predict_proba(df[predictors])[:,-1]
        df[f"SE_{key}"] = (df.query(flt_str[key])[key] - df[f"hat_P({key}=1)"]) ** 2

def merge_df_val(df_rct, df_obs, predictors, pr_model, rct_models, obs_models):
    df = pd.concat([df_rct, df_obs]).reset_index(drop=True)

    df["hat_P(R=1)"] = pr_model.predict_proba(df[predictors])[:,-1]
    df["SE_R"] = (df["R"] - df[f"hat_P(R=1)"]) ** 2

    df["mu_0_rct"] = rct_models["Y0"].predict_proba(df[predictors])[:,-1]
    df["mu_1_rct"] = rct_models["Y1"].predict_proba(df[predictors])[:,-1]
    df["mu_0_obs"] = obs_models["Y0"].predict_proba(df[predictors])[:,-1]
    df["mu_1_obs"] = obs_models["Y1"].predict_proba(df[predictors])[:,-1]

    df["b1(X)"] = (df["mu_1_rct"] - df["mu_0_rct"]) - (df["mu_1_obs"] - df["mu_0_obs"])
    df["abs(b1(X))"] = abs(df["b1(X)"])

    return df

def covs_to_prob(row, covs, p):
    group_index = 0
    for i, cov in enumerate(covs[::-1]):
        group_index += (2 ** i) * row[cov]

    return p[int(group_index)]


def sample_probs(d, pl_range, ph_range, bias_flag):
    p = defaultdict()
    for k in range(2 ** d):
        p[k] = choice([unif(*pl_range), unif(*ph_range)])

    p = list(p.values())
    if not bias_flag:
        p[1::2] = p[::2]

    return p

def sample_all_probs(d, pl_range, ph_range, scenario):
    probs = defaultdict(list)

    for key, bias_flag in scenario.items():
        probs[key] = sample_probs(d, pl_range, ph_range, bias_flag)

    return probs


def init_df(n, d, d_meas, r, probs, x_probs, u_probs):
    covs = [f"X{i + 1}" for i in range(d)]
    meas_covs = [f"X{i + 1}" for i in range(d_meas)]
    
    X_meas = choice([0, 1], size=(n,d_meas), p=x_probs[f"R={r}"])
    U = choice([0, 1], size=(n,1), p=u_probs[f"R={r}"])
    X = np.concatenate((X_meas, U), axis=1)

    df = pd.DataFrame({**{'R': r}, **{cov: X[:,i] for i, cov in enumerate(covs)}})

    for i, c in enumerate(list(itertools.product([0, 1], repeat=d_meas))):
        df[f"Xp{i + 1}"] = (df[meas_covs] == c).all(axis=1).astype(int)

    for key, prob in probs.items():
        df[f"P({key}=1)"] = df.apply(lambda row: covs_to_prob(row, covs, prob), axis=1)
        df[key] = df.apply(lambda row: binomial(1, row[f"P({key}=1)"]), axis=1)

    df["Y"] = df["A"] * df["Y1"] + (1 - df["A"]) * df["Y0"]   

    return df
