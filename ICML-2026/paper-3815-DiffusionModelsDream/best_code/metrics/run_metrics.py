import argparse
import numpy as np
import pandas as pd
from mmd import mmd
from c2st import c2st_auc, check_c2st_convergence
from jax import random
import argparse

def mse_between_common_columns(df1, df2):

    # Identify common columns
    common_cols = df1.columns.intersection(df2.columns)
    
    # Ensure alignment of indices
    df1_common = df1[common_cols].sort_index()
    df2_common = df2[common_cols].sort_index()
    
    # Compute MSE for each column
    mse_values = ((df1_common - df2_common)**2).mean()
    
    return mse_values

#WARNING: FILESTR MAY BE SUPERFLUOUS
def main(data,samples,seed,filestr,corr_threshold=0.9):
    key = random.PRNGKey(seed)
    # Compute metrics
    mmd_value = mmd(data, samples)

    # Separate into correlated and uncorrelated according to the data
    corr = np.corrcoef(data, rowvar=False)

    # Correlated if highly correlated with any variable beside itself
    correlated_mask = np.any(np.abs(corr - np.eye(corr.shape[0])) > corr_threshold, axis=0)

    uncorrelated_mask = ~correlated_mask

    # Split data
    ddata    = np.asarray(data[:, uncorrelated_mask])
    dsamples = np.asarray(samples[:, uncorrelated_mask])

    cdata    = np.asarray(data[:, correlated_mask])   # correlated variables
    csamples = np.asarray(samples[:, correlated_mask])

    if ddata.shape[1] > 0:
        converged = check_c2st_convergence(ddata, dsamples)
        joint_c2st = c2st_auc(ddata, dsamples)
        joint_c2st = np.asarray(joint_c2st).squeeze()

        marginal = []
        for i in range(ddata.shape[1]):
            m = c2st_auc(ddata[:, i][:, None], dsamples[:, i][:, None])
            marginal.append(m)

        mean_marginal_c2st   = float(np.mean(marginal))
        median_marginal_c2st = float(np.median(marginal))
        max_marginal_c2st    = float(np.max(marginal))
        min_marginal_c2st    = float(np.min(marginal))
    else:
        # No uncorrelated variables
        converged = False
        joint_c2st = np.nan
        mean_marginal_c2st = np.nan
        median_marginal_c2st = np.nan
        max_marginal_c2st = np.nan
        min_marginal_c2st = np.nan

    # Compute median absolute error for correlated variables
    if cdata.size > 0:
        correlated_medAE = np.median(np.abs(cdata - csamples))
    else:
        correlated_medAE = np.nan

    metrics = {
        "mmd_value": float(mmd_value),

        "joint_c2st": float(joint_c2st),
        "mean_marginal_c2st": mean_marginal_c2st,
        "median_marginal_c2st": median_marginal_c2st,
        "max_marginal_c2st": max_marginal_c2st,
        "min_marginal_c2st": min_marginal_c2st,

        "n_uncorrelated_variables": int(np.sum(uncorrelated_mask)),
        "n_correlated_variables": int(np.sum(correlated_mask))
    }
    print("C2ST convergence:",converged)
    print(metrics)
    return metrics

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--corr_threshold", type=float, default=0.9)

    args = parser.parse_args()

    # Data reading
    data_raw = pd.read_csv(args.data).filter(like="res_metrics").to_numpy()
    sample_raw = pd.read_csv(args.samples).filter(like="res_metrics").to_numpy()

    # Normalize (ignoring contant values)
    mean = np.nanmean(data_raw,axis=0)
    raw_stdev = np.nanstd(data_raw,axis=0)
    stdev = np.where(raw_stdev==0.,1.,raw_stdev)
    data = (data_raw-mean)/stdev
    samples = (sample_raw-mean)/stdev

    # Call metrics function
    metrics = main(
        data,
        samples,
        seed=args.seed,
        filestr=args.output,
        corr_threshold=args.corr_threshold,
    )

    np.save(args.output,metrics)

if __name__ == "__main__":
    cli()