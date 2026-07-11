from utils import *

import numpy as np
import pandas as pd


from numpy.random import uniform as unif


class DataModule():
    def __init__(self, ...):
        pass

def sample_bernoulli(row, covs, coefs=None, p=0.95):
    if coefs is not None:
        logit = row[covs] @ coefs[1:] + coefs[0]
        p = 1 / (1 + np.exp(-logit))
        # p = max(min(p, 0.9), 0.1)

    return np.random.binomial(1, p)


def sample_TOS(df, covs, coefs, probs):
    df["Y0"] = df.apply(lambda row: sample_bernoulli(row, covs, coefs["Y0"], probs["Y0"]), axis=1)
    df["Y1"] = df.apply(lambda row: sample_bernoulli(row, covs, coefs["Y1"], probs["Y1"]), axis=1)
    df["A"]  = df.apply(lambda row: sample_bernoulli(row, covs, coefs["A"], probs["A"]), axis=1)
    df["Y"]  = df["A"] * df["Y1"] + (1 - df["A"]) * df["Y0"]
    df["S"]  = df.apply(lambda row: sample_bernoulli(row, covs, coefs["S"], probs["S"]), axis=1)


def fit_model(df, covs, target, filter_fit='index==index', model_name='LogReg'):
    if model_name == "LogReg":
        # model = XGBClassifier(
        #     n_estimators=5,  
        #     max_depth=3,     
        # )

        model = LogisticRegression()
        # model = MLPClassifier(hidden_layer_sizes=(32,), activation='logistic')

    model.fit(df.query(filter_fit)[covs], df.query(filter_fit)[target])

    return model


def fit_all_models(df, covs, model_names = {"S": "LogReg", "A": "LogReg", "Y0": "LogReg", "Y1": "LogReg"}):
    PS_model  = fit_model(df, covs, "S", "index==index", model_names["S"])
    PA_model  = fit_model(df, covs, "A", "S==1", model_names["A"])
    PY0_model = fit_model(df, covs, "Y", "S==1 & A==0", model_names["Y0"])
    PY1_model = fit_model(df, covs, "Y", "S==1 & A==1", model_names["Y1"])
    
    return {"S": PS_model, "A": PA_model, "Y0": PY0_model, "Y1": PY1_model}


def log_model_preds(df, covs, models):
    df[f"hat_P(S=1)"] = models["S"].predict_proba(df[covs])[:,-1]
    df[f"SE_S"] = (df["S"] - df["hat_P(S=1)"]) ** 2

    df[f"hat_P(A=1)"] = models["A"].predict_proba(df[covs])[:,-1]
    df[f"SE_A"] = (df["A"] - df["hat_P(A=1)"]) ** 2

    df[f"mu_0"] = models["Y0"].predict_proba(df[covs])[:,-1]
    df[f"SE_Y0"] = (df.query("A==0")["Y"] - df["mu_0"] ) ** 2

    df[f"mu_1"] = models["Y1"].predict_proba(df[covs])[:,-1]
    df[f"SE_Y1"] = (df.query("A==1")["Y"] - df["mu_1"] ) ** 2


def psi_R(row, R):
    if row["R"] != R or row["S"] == 0:
        return 0
    else:
        a_ind = row["A"]
        p_r1, p_a1, p_s1 = row["hat_P(R=1)"], row["hat_P(A=1)"], row["hat_P(S=1)"]
        y, mu0, mu1 = row["Y"], row["mu_0"], row["mu_1"]

        term_1 = mu1 - mu0
        term_2 = a_ind * (y - mu1) / p_a1
        term_3 = (1 - a_ind) * (y - mu0) / (1 - p_a1)

        p_r = R * p_r1 + (1 - R) * (1 - p_r1)

        return (term_1 + term_2 - term_3) / (p_r * p_s1)
        # return term_1 / (p_r * p_s1)


def calc_psi(df):
    df["psi0"] = df.apply(lambda row: psi_R(row, 0), axis=1)
    df["psi1"] = df.apply(lambda row: psi_R(row, 1), axis=1)
    df["psi"] = df["psi1"] - df["psi0"]


def merge_df_train(df_rct, df_obs, covs, pr_model_name="LogReg"):
    df = pd.concat([df_rct, df_obs]).reset_index(drop=True)
    pr_model = fit_model(df, covs, "R", model_name=pr_model_name)

    df["hat_P(R=1)"] = pr_model.predict_proba(df[covs])[:,-1]
    df["SE_R"] = (df["R"] - df[f"hat_P(R=1)"]) ** 2

    calc_psi(df)

    return df, pr_model


def merge_df_val(df_rct, df_obs, covs, pr_model, rct_models, obs_models, df_merged_train):
    df = pd.concat([df_rct, df_obs]).reset_index(drop=True)
    # df = df_obs.copy()

    df["hat_P(R=1)"] = pr_model.predict_proba(df[covs])[:,-1]
    df["SE_R"] = (df["R"] - df[f"hat_P(R=1)"]) ** 2

    df["mu_0_rct"] = rct_models["Y0"].predict_proba(df[covs])[:,-1]
    df["mu_1_rct"] = rct_models["Y1"].predict_proba(df[covs])[:,-1]
    df["mu_0_obs"] = obs_models["Y0"].predict_proba(df[covs])[:,-1]
    df["mu_1_obs"] = obs_models["Y1"].predict_proba(df[covs])[:,-1]
    df["w1(X)"] = (df["mu_1_rct"] - df["mu_0_rct"]) - (df["mu_1_obs"] - df["mu_0_obs"])

    K = laplace_kernel_two_matrices(df[covs], df_merged_train[covs])
    df['w2(X)'] = K @ df_merged_train['psi'] / len(df_merged_train['psi'])

    return df


def init_df(n, d, r, px_dist=None, mean=None, cov=None):
    if px_dist == "mvn":
        vector = np.random.multivariate_normal(mean, cov, size=n)
        vector = np.clip(vector, -1, 1)
        X_rct = vector[:,0]
        U_rct = vector[:,1]    
    else:
        X_rct = unif(-1, 1, n)
        U_rct = unif(-1, 1, n)

    X_rct = legendre_mat(X_rct, d)
    U_rct = legendre_mat(U_rct, d)

    meas_covs = [f"X{i + 1}" for i in range(d)]
    unmeas_covs = [f"U{i + 1}" for i in range(d)]

    df = pd.DataFrame({**{cov: X_rct[:,i] for i, cov in enumerate(meas_covs)},
                        **{u_cov: U_rct[:,i] for i, u_cov in enumerate(unmeas_covs)},
                        **{'R': r}})
    
    return df