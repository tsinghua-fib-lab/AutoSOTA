#!/usr/bin/env python3
"""Helper to merge PP and CPD CSVs into features CSV for CV evaluation."""
import sys
import pandas as pd

def main():
    if len(sys.argv) < 5:
        print("Usage: build_features_helper.py <pp_csv> <cpd_csv> <out_csv> <window_sizes...>")
        sys.exit(1)

    pp_path = sys.argv[1]
    cpd_path = sys.argv[2]
    out_path = sys.argv[3]
    windows = [int(w) for w in sys.argv[4:]]

    pp = pd.read_csv(pp_path)
    cpd = pd.read_csv(cpd_path)

    keep_cpd = ["row_index", "online_max_W_plus", "online_max_two_sided", "online_kendall_tau"]
    merged = pp.merge(cpd[keep_cpd], on="row_index", how="inner")
    # Use two-sided CUSUM score (max of W_plus and W_minus) for cpd_online
    cpd_score_col = "online_max_two_sided" if "online_max_two_sided" in merged.columns else "online_max_W_plus"
    out = pd.DataFrame({
        "row_index": merged["row_index"],
        "is_adversarial": merged["is_adversarial"],
        "algorithm": merged["algorithm"],
        "cpd_online": merged[cpd_score_col],
        "cpd_kendall_tau": merged["online_kendall_tau"],
        "pp_global": merged["global_mean_nll"],
    })
    for w in windows:
        col = "window_mean_nll_w{}".format(w)
        out["window_pp_w{}".format(w)] = merged[col]

    out.to_csv(out_path, index=False)
    n_adv = int(out["is_adversarial"].sum())
    n_ben = int((out["is_adversarial"] == 0).sum())
    print("Features: {} rows, {} adversarial, {} benign".format(len(out), n_adv, n_ben))


if __name__ == "__main__":
    main()
