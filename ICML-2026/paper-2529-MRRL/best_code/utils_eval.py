import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ivmodels.models import KClass
from mdcrl import LitAutoEncoder
from pycomets.gcm import GCM
from pycomets.regression import LM, DefaultMultiRegression
from sklearn.linear_model import LinearRegression


def get_checkpoint_from_outputs(exp_name, sim_id, selection_strategy="best"):
    """
    Looks in the Hydra output structure: outputs/{exp_name}/{sim_id}/checkpoints/

    Args:
        exp_name: Name of the experiment folder.
        sim_id: The simulation ID (folder name).
        selection_strategy: "best" (default) to look for best-*.ckpt,
                           "last" to look for last.ckpt.
    """
    base_dir = os.path.join("outputs", exp_name, str(sim_id), "checkpoints")

    if selection_strategy == "last":
        last_ckpt = os.path.join(base_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt
        print(
            f"Warning: strategy='last' requested but {last_ckpt} not found. Falling back."
        )

    best_pattern = os.path.join(base_dir, "best-*.ckpt")
    best_files = glob.glob(best_pattern)

    if best_files:
        best_files.sort(key=os.path.getmtime, reverse=True)
        return best_files[0]

    last_ckpt = os.path.join(base_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    all_ckpts = glob.glob(os.path.join(base_dir, "*.ckpt"))
    if all_ckpts:
        all_ckpts.sort(key=os.path.getmtime, reverse=True)
        return all_ckpts[0]

    print(f"Error: No checkpoints found in {base_dir}")
    return None


def load_model(exp_name, sim_id, selection_strategy):
    ckpt_path = get_checkpoint_from_outputs(
        exp_name=exp_name, sim_id=sim_id, selection_strategy=selection_strategy
    )
    print(f"Loading model from {ckpt_path}")

    mod = LitAutoEncoder.load_from_checkpoint(
        ckpt_path, map_location=torch.device("cpu")
    )

    return mod


def simpleIV(Y, T, Z, C=None):
    """2SLS estimator"""
    if C is None:
        lm = LinearRegression()
        lm.fit(y=T, X=Z)
        hT = lm.predict(X=Z)
        lm = LinearRegression()
        lm.fit(y=Y, X=hT)
    else:
        lm = LM()
        lm.fit(Y=Y, X=C)
        res_Y = lm.residuals(Y=Y, X=C)
        lm.fit(Y=T, X=C)
        res_T = lm.residuals(Y=T, X=C)
        mlm = DefaultMultiRegression(LM(), dim=Z.shape[1])
        mlm.fit(Y=Z, X=C)
        res_Z = mlm.residuals(Y=Z, X=C)
        lm = LinearRegression()
        lm.fit(y=res_T, X=res_Z)
        hres_T = lm.predict(X=res_Z)
        lm = LinearRegression()
        lm.fit(y=res_Y, X=hres_T)

    return lm.coef_[0][range(T.shape[1])]


def limlIV(Y, T, Z, C=None):
    """LIML estimator via ivmodels KClass with kappa='liml'."""
    y = Y.to_numpy() if hasattr(Y, "to_numpy") else np.asarray(Y)
    X = T.to_numpy() if hasattr(T, "to_numpy") else np.asarray(T)
    Zn = Z.to_numpy() if hasattr(Z, "to_numpy") else np.asarray(Z)
    Cn = C.to_numpy() if hasattr(C, "to_numpy") else None if C is None else np.asarray(C)

    mod = KClass(kappa="liml", fit_intercept=True).fit(
        Z=Zn, X=X, y=y.ravel(), C=Cn
    )
    return mod.coef_[: T.shape[1]]


def compute_estimates(dfs, batch_nums=None, plot=True, theta=1.0, ax=None, method="tsls"):

    estimator_fn = {"tsls": simpleIV, "liml": limlIV}[method]
    run_all = method == "tsls"

    df_lst = []

    for pop_num in range(len(dfs)):

        df_pop = dfs[pop_num]
        if batch_nums is None:
            batch_nums = df_pop["batch_num"].unique()

        est0 = []  # Z (observables)
        est1 = []  # VW
        est2 = []  # V
        est3 = []  # W
        est4 = []  # hVW (hvws)
        est5 = []  # hV
        est6 = []  # hW
        est7 = []  # W but include V as additional covariates in T to Y
        est8 = []

        for batch_num in batch_nums:

            df_tmp = df_pop[df_pop["batch_num"] == batch_num]

            if run_all:
                # Using Z
                est0.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^Z", axis=1),
                    )
                )

                # Using VW
                est1.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^V|^W", axis=1),
                    )
                )

                # Using V
                est2.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^V", axis=1),
                    )
                )

                # Using W
                est3.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^W", axis=1),
                    )
                )

                # Using hVhW
                est4.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^hV|^hW", axis=1),
                    )
                )

                est5.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^hV", axis=1),
                    )
                )

                est7.append(
                    estimator_fn(
                        Y=df_tmp.filter(regex="^Y", axis=1),
                        T=df_tmp.filter(regex="^D", axis=1),
                        Z=df_tmp.filter(regex="^W", axis=1),
                        C=df_tmp.filter(regex="^V", axis=1),
                    )
                )

            # Always compute hW and hWchV
            est6.append(
                estimator_fn(
                    Y=df_tmp.filter(regex="^Y", axis=1),
                    T=df_tmp.filter(regex="^D", axis=1),
                    Z=df_tmp.filter(regex="^hW", axis=1),
                )
            )

            est8.append(
                estimator_fn(
                    Y=df_tmp.filter(regex="^Y", axis=1),
                    T=df_tmp.filter(regex="^D", axis=1),
                    Z=df_tmp.filter(regex="^hW", axis=1),
                    C=df_tmp.filter(regex="^hV", axis=1),
                )
            )

        if run_all:
            est_all = np.column_stack(
                (est0, est1, est2, est3, est4, est5, est6, est7, est8)
            )
            est_df = pd.DataFrame(
                est_all,
                columns=["Z", "VW", "V", "W", "hVhW", "hV", "hW", "WcV", "hWchV"],
            )
        else:
            est_all = np.column_stack((est6, est8))
            est_df = pd.DataFrame(
                est_all,
                columns=["hW", "hWchV"],
            )

        est_df["pop_num"] = pop_num
        df_lst.append(est_df)

    est_df = pd.concat(df_lst, ignore_index=True)
    est_df_long = est_df.melt(
        id_vars=["pop_num"], var_name="instrument", value_name="estimate"
    )
    if plot and (ax is None):
        plt.figure()
        sns.boxplot(
            est_df_long,
            x="instrument",
            y="estimate",
            hue="pop_num",
            ax=ax,
            order=["Z", "VW", "hVhW", "V", "hV", "W", "hW", "WcV", "hWchV"],
        )
        plt.axhline(y=theta, color="red", linestyle="--", linewidth=1)
        plt.suptitle(f"Pop {pop_num}", y=1.02)
        plt.show()
    elif plot:
        sns.boxplot(
            est_df_long,
            x="instrument",
            y="estimate",
            hue="pop_num",
            ax=ax,
            order=["Z", "VW", "hVhW", "V", "hV", "W", "hW", "WcV", "hWchV"],
        )
        ax.axhline(y=theta, color="red", linestyle="--", linewidth=1)
        ax.legend_.remove()

    return est_df_long


def compute_recon_err(dfs, dim_z=5, batch_nums=None, combine_batches=True):

    df_lst = []
    z_cols = [f"Z_{i}" for i in range(dim_z)]
    hz_cols = [f"hZ_{i}" for i in range(dim_z)]

    for pop_num in range(len(dfs)):

        df_pop = dfs[pop_num]

        if batch_nums is None:
            batch_nums = df_pop["batch_num"].unique()

        if combine_batches:
            df_tmp = df_pop[df_pop["batch_num"].isin(batch_nums)]
            squared_errors = (df_tmp[z_cols].values - df_tmp[hz_cols].values) ** 2
            mse = squared_errors.mean()
            df_lst.append(
                pd.DataFrame(
                    np.array([pop_num, mse]).reshape(1, 2),
                    columns=["pop", "mse"],
                )
            )
        else:
            mses = []
            for batch_num in batch_nums:
                df_tmp = df_pop[df_pop["batch_num"] == batch_num]
                squared_errors = (
                    df_tmp[z_cols].values - df_tmp[hz_cols].values
                ) ** 2
                mses.append(squared_errors.mean())

            mse_df = pd.DataFrame(mses, columns=["mse"])
            mse_df["pop"] = pop_num
            df_lst.append(mse_df)

    mse_df = pd.concat(df_lst, ignore_index=True)
    return mse_df


def compute_gcm(dfs):
    """
    Computes Generalized Covariance Measure (GCM) independence tests
    for each population in the list of DataFrames.
    """
    res = []

    for pop_num, df in enumerate(dfs):
        hV = df.filter(regex="^hV_").to_numpy()
        hW = df.filter(regex="^hW_").to_numpy()
        V = df.filter(regex="^V_").to_numpy()
        W = df.filter(regex="^W_").to_numpy()

        # H0: hV \indep W | V
        gcm_v = GCM()
        gcm_v.test(
            X=hV, Y=W, Z=V,
            reg_yz=LM(), reg_xz=LM(),
            test_type="max", B=999, show_summary=False,
        )

        # H0: hW \indep V | W
        gcm_w = GCM()
        gcm_w.test(
            X=hW, Y=V, Z=W,
            reg_yz=LM(), reg_xz=LM(),
            test_type="max", B=999, show_summary=False,
        )

        res.append(pd.DataFrame([{
            "pop": pop_num,
            "gs_w": gcm_w.stat,
            "gp_w": gcm_w.pval,
            "gs_v": gcm_v.stat,
            "gp_v": gcm_v.pval,
        }]))

    # Combined population
    df_combined = pd.concat(dfs, ignore_index=True)
    hV = df_combined.filter(regex="^hV_").to_numpy()
    hW = df_combined.filter(regex="^hW_").to_numpy()
    V = df_combined.filter(regex="^V_").to_numpy()
    W = df_combined.filter(regex="^W_").to_numpy()

    gcm_v = GCM()
    gcm_v.test(
        X=hV, Y=W, Z=V,
        reg_yz=LM(), reg_xz=LM(),
        test_type="max", B=999, show_summary=False,
    )
    gcm_w = GCM()
    gcm_w.test(
        X=hW, Y=V, Z=W,
        reg_yz=LM(), reg_xz=LM(),
        test_type="max", B=999, show_summary=False,
    )
    res.append(pd.DataFrame([{
        "pop": -1,
        "gs_w": gcm_w.stat,
        "gp_w": gcm_w.pval,
        "gs_v": gcm_v.stat,
        "gp_v": gcm_v.pval,
    }]))

    return pd.concat(res, ignore_index=True)
