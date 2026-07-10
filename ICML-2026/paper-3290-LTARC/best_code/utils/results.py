import numpy as np
import pandas as pd

from utils import save_utils


def save_results(out_dir, args, true_obj_all, true_constraint_all):
    mean_true_obj = np.mean(true_obj_all, axis=0)
    mean_true_constraint = np.mean(true_constraint_all, axis=0)

    result_df = {
        "beta": args.constraint_values,
        "mean obj": mean_true_obj,
        "mean constr": mean_true_constraint,
    }

    df = pd.DataFrame(result_df)
    obj_df = pd.DataFrame(true_obj_all)
    constr_df = pd.DataFrame(true_constraint_all)

    # Compute quantiles for "mean obj" and "mean constr"
    quantiles = obj_df.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=0).T
    df[["obj_q10", "obj_q25", "obj_q50", "obj_q75", "obj_q90"]] = quantiles.values

    quantiles = constr_df.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=0).T
    df[["constr_q10", "constr_q25", "constr_q50", "constr_q75", "constr_q90"]] = quantiles.values

    df.to_csv(save_utils.get_full_path(out_dir, '{}.csv'.format(args.name)), index=False)
    obj_df.to_csv(save_utils.get_full_path(out_dir, 'obj_all.csv'.format(args.name)), index=False)
    constr_df.to_csv(save_utils.get_full_path(out_dir, 'constr_all.csv'.format(args.name)), index=False)

    return mean_true_obj, mean_true_constraint
