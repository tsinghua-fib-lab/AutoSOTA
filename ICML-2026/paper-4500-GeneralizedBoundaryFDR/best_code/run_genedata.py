import sys
import os

sys.path.insert(0, "/repo/functions")
os.chdir("/repo/RealData")

from utils import *
import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

print("Loading CRISPR gene effect data...")
scores = load_single_cell_line("CRISPR_gene_effect.csv", cell_line_index=5)
scores.index = [x.split(" ")[0] for x in scores.index]

print("Labeling data...")
df_labeled = build_labeled_dataframe(scores, FILE_CEG, FILE_NEG)

scores = df_labeled["Score"]
df_labels = df_labeled[["Label"]]
nseeds = 100
test_size = 2/3  # PARAM-2: try 67/33 split
all_results_records = []
alphalist = [0.05, 0.1, 0.2, 0.3]

print("Starting resampling experiment (100 seeds x 4 alphas)...")
for alpha in alphalist:
    for seed in range(1, 1+nseeds):
        np.random.seed(seed)
        p_values, e_values, test_labels = sample_split_testing(scores, df_labels, test_size, seed)

        rej_bh = BH(p_values, alpha)
        if len(rej_bh) > 0:
            max_p_local_idx = np.argmax(p_values[rej_bh])
            bh_boundary_index = rej_bh[max_p_local_idx]
            is_bh_boundary = 1 if test_labels[bh_boundary_index] == 0 else 0
        else:
            bh_boundary_index = []

        # Domino-P (harmonic mean p-value, k=1)
        rej_dominop, dominop_boundary_index, _ = harmonicpbfdr(p_values, alpha)
        # ALGO-6: Dual-threshold (HMP intersection with Bonferroni k=1)
        rej_dual, dual_boundary_index, _ = dual_threshold_bfdr(p_values, alpha)
        # Domino k=2,3,4,5 (generalized Bonferroni)
        reject_indices_2domino, boundary_indices_2domino = bfdr_k_domino(p_values, k=2, alpha=alpha)
        reject_indices_3domino, boundary_indices_3domino = bfdr_k_domino(p_values, k=3, alpha=alpha)
        reject_indices_4domino, boundary_indices_4domino = bfdr_k_domino(p_values, k=4, alpha=alpha)
        reject_indices_5domino, boundary_indices_5domino = bfdr_k_domino(p_values, k=5, alpha=alpha)

        # Domino-E (e-value closure, k=1,2,3,4,5)
        rej_dominope, dominoe_boundary_index = bfdr_k_eclosure(e_values, k=1, alpha=alpha)
        rej_2dominope, domino2e_boundary_index = bfdr_k_eclosure(e_values, k=2, alpha=alpha)
        rej_3dominope, domino3e_boundary_index = bfdr_k_eclosure(e_values, k=3, alpha=alpha)
        rej_4dominope, domino4e_boundary_index = bfdr_k_eclosure(e_values, k=4, alpha=alpha)
        rej_5dominope, domino5e_boundary_index = bfdr_k_eclosure(e_values, k=5, alpha=alpha)

        # SL procedure
        rej_sl, sl_boundary_index, _ = sl_procedure(p_values, alpha)

        # Simes local test (k=1)
        rej_simes, simes_boundary_index, _ = simesdomino(p_values, alpha)
        # Cauchy combination local test (k=1)
        rej_cauchy, cauchy_boundary_index, _ = cauchypbfdr(p_values, alpha)
        # Average p-value local test (k=1)
        rej_avgp, avgp_boundary_index, _ = averagepbfdr(p_values, alpha)

        # Boundary error evaluation
        is_boundary_false_2domino = kbfdr_evaluate(test_labels, boundary_indices_2domino, k=2)
        is_boundary_false_3domino = kbfdr_evaluate(test_labels, boundary_indices_3domino, k=3)
        is_boundary_false_4domino = kbfdr_evaluate(test_labels, boundary_indices_4domino, k=4)
        is_boundary_false_5domino = kbfdr_evaluate(test_labels, boundary_indices_5domino, k=5)
        is_dominop_boundary = kbfdr_evaluate(test_labels, dominop_boundary_index, k=1)
        is_dual_boundary = kbfdr_evaluate(test_labels, dual_boundary_index, k=1)
        is_sl_boundary = kbfdr_evaluate(test_labels, sl_boundary_index, k=1)
        is_dominoe_boundary = kbfdr_evaluate(test_labels, dominoe_boundary_index, k=1)
        is_domino2_boundary = kbfdr_evaluate(test_labels, domino2e_boundary_index, k=2)
        is_domino3_boundary = kbfdr_evaluate(test_labels, domino3e_boundary_index, k=3)
        is_domino4_boundary = kbfdr_evaluate(test_labels, domino4e_boundary_index, k=4)
        is_domino5_boundary = kbfdr_evaluate(test_labels, domino5e_boundary_index, k=5)
        is_simes_boundary = kbfdr_evaluate(test_labels, simes_boundary_index, k=1)
        is_cauchy_boundary = kbfdr_evaluate(test_labels, cauchy_boundary_index, k=1)
        is_avgp_boundary = kbfdr_evaluate(test_labels, avgp_boundary_index, k=1)

        # FDR/Power evaluation
        bh_fdr, bh_power = evaluate_fdrpower(rej_bh, test_labels)
        dominop_fdr, dominop_power = evaluate_fdrpower(rej_dominop, test_labels)
        dual_fdr, dual_power = evaluate_fdrpower(rej_dual, test_labels)
        dominope_fdr, dominope_power = evaluate_fdrpower(rej_dominope, test_labels)
        sl_fdr, sl_power = evaluate_fdrpower(rej_sl, test_labels)
        domino2_fdr, domino2_power = evaluate_fdrpower(reject_indices_2domino, test_labels)
        domino3_fdr, domino3_power = evaluate_fdrpower(reject_indices_3domino, test_labels)
        domino4_fdr, domino4_power = evaluate_fdrpower(reject_indices_4domino, test_labels)
        domino5_fdr, domino5_power = evaluate_fdrpower(reject_indices_5domino, test_labels)
        domino2e_fdr, domino2e_power = evaluate_fdrpower(rej_2dominope, test_labels)
        domino3e_fdr, domino3e_power = evaluate_fdrpower(rej_3dominope, test_labels)
        domino4e_fdr, domino4e_power = evaluate_fdrpower(rej_4dominope, test_labels)
        domino5e_fdr, domino5e_power = evaluate_fdrpower(rej_5dominope, test_labels)
        simes_fdr, simes_power = evaluate_fdrpower(rej_simes, test_labels)
        cauchy_fdr, cauchy_power = evaluate_fdrpower(rej_cauchy, test_labels)
        avgp_fdr, avgp_power = evaluate_fdrpower(rej_avgp, test_labels)

        all_results_records.append({
            "Seed": seed, "alpha": alpha,
            "BH_FDR": 1-bh_fdr, "BH_Power": bh_power, "BH_Boundary_Error": is_bh_boundary,
            "DominoP_FDR": 1-dominop_fdr, "DominoP_Power": dominop_power, "DominoP_Boundary_Error": is_dominop_boundary,
            "Dual_FDR": 1-dual_fdr, "Dual_Power": dual_power, "Dual_Boundary_Error": is_dual_boundary,
            "SL_FDR": 1-sl_fdr, "SL_Power": sl_power, "SL_Boundary_Error": is_sl_boundary,
            "DominoE_FDR": 1-dominope_fdr, "DominoE_Power": dominope_power, "DominoE_Boundary_Error": is_dominoe_boundary,
            "Domino2_FDR": 1-domino2_fdr, "Domino2_Power": domino2_power, "Domino2_Boundary_Error": is_boundary_false_2domino,
            "Domino3_FDR": 1-domino3_fdr, "Domino3_Power": domino3_power, "Domino3_Boundary_Error": is_boundary_false_3domino,
            "Domino4_FDR": 1-domino4_fdr, "Domino4_Power": domino4_power, "Domino4_Boundary_Error": is_boundary_false_4domino,
            "Domino5_FDR": 1-domino5_fdr, "Domino5_Power": domino5_power, "Domino5_Boundary_Error": is_boundary_false_5domino,
            "Domino2E_FDR": 1-domino2e_fdr, "Domino2E_Power": domino2e_power, "Domino2E_Boundary_Error": is_domino2_boundary,
            "Domino3E_FDR": 1-domino3e_fdr, "Domino3E_Power": domino3e_power, "Domino3E_Boundary_Error": is_domino3_boundary,
            "Domino4E_FDR": 1-domino4e_fdr, "Domino4E_Power": domino4e_power, "Domino4E_Boundary_Error": is_domino4_boundary,
            "Domino5E_FDR": 1-domino5e_fdr, "Domino5E_Power": domino5e_power, "Domino5E_Boundary_Error": is_domino5_boundary,
            "Simes_FDR": 1-simes_fdr, "Simes_Power": simes_power, "Simes_Boundary_Error": is_simes_boundary,
            "Cauchy_FDR": 1-cauchy_fdr, "Cauchy_Power": cauchy_power, "Cauchy_Boundary_Error": is_cauchy_boundary,
            "AverageP_FDR": 1-avgp_fdr, "AverageP_Power": avgp_power, "AverageP_Boundary_Error": is_avgp_boundary,
        })

        if seed % 10 == 0:
            print("  alpha={}, seed={}/{}".format(alpha, seed, nseeds))

df_raw = pd.DataFrame(all_results_records)
df_summary = df_raw.groupby("alpha").agg({
    "BH_FDR": "mean", "BH_Power": "mean", "BH_Boundary_Error": "mean",
    "DominoP_FDR": "mean", "DominoP_Power": "mean", "DominoP_Boundary_Error": "mean",
    "Dual_FDR": "mean", "Dual_Power": "mean", "Dual_Boundary_Error": "mean",
    "SL_FDR": "mean", "SL_Power": "mean", "SL_Boundary_Error": "mean",
    "DominoE_FDR": "mean", "DominoE_Power": "mean", "DominoE_Boundary_Error": "mean",
    "Domino2_FDR": "mean", "Domino2_Power": "mean", "Domino2_Boundary_Error": "mean",
    "Domino3_FDR": "mean", "Domino3_Power": "mean", "Domino3_Boundary_Error": "mean",
    "Domino4_FDR": "mean", "Domino4_Power": "mean", "Domino4_Boundary_Error": "mean",
    "Domino5_FDR": "mean", "Domino5_Power": "mean", "Domino5_Boundary_Error": "mean",
    "Domino2E_FDR": "mean", "Domino2E_Power": "mean", "Domino2E_Boundary_Error": "mean",
    "Domino3E_FDR": "mean", "Domino3E_Power": "mean", "Domino3E_Boundary_Error": "mean",
    "Domino4E_FDR": "mean", "Domino4E_Power": "mean", "Domino4E_Boundary_Error": "mean",
    "Domino5E_FDR": "mean", "Domino5E_Power": "mean", "Domino5E_Boundary_Error": "mean",
    "Simes_FDR": "mean", "Simes_Power": "mean", "Simes_Boundary_Error": "mean",
    "Cauchy_FDR": "mean", "Cauchy_Power": "mean", "Cauchy_Boundary_Error": "mean",
    "AverageP_FDR": "mean", "AverageP_Power": "mean", "AverageP_Boundary_Error": "mean",
}).reset_index()

print("\n========== RESULTS (Table 2 format) ==========")
print(df_summary.to_string())

print("\n========== Key metrics at alpha=0.05 ==========")
row = df_summary[df_summary["alpha"] == 0.05].iloc[0]
dom_b = row["DominoP_Boundary_Error"]
dom_t = row["DominoP_FDR"] * 100
dom_p = row["DominoP_Power"] * 100
sl_b = row["SL_Boundary_Error"]
sl_t = row["SL_FDR"] * 100
sl_p = row["SL_Power"] * 100
bh_b = row["BH_Boundary_Error"]
bh_t = row["BH_FDR"] * 100
bh_p = row["BH_Power"] * 100
dual_b = row["Dual_Boundary_Error"]
dual_t = row["Dual_FDR"] * 100
dual_p = row["Dual_Power"] * 100

print("Domino bFDR (k=1):       {:.4f}".format(dom_b))
print("Domino TDR (k=1):         {:.2f}%".format(dom_t))
print("Domino Power (k=1):       {:.2f}%".format(dom_p))
print("Dual bFDR (k=1):         {:.4f}".format(dual_b))
print("Dual TDR (k=1):           {:.2f}%".format(dual_t))
print("Dual Power (k=1):         {:.2f}%".format(dual_p))
print("SL bFDR:                   {:.4f}".format(sl_b))
print("SL TDR:                    {:.2f}%".format(sl_t))
print("SL Power:                  {:.2f}%".format(sl_p))
print("BH bFDR:                   {:.4f}".format(bh_b))
print("BH TDR:                    {:.2f}%".format(bh_t))
print("BH Power:                  {:.2f}%".format(bh_p))
si_b = row["Simes_Boundary_Error"]
si_t = row["Simes_FDR"] * 100
si_p = row["Simes_Power"] * 100
print("Simes bFDR (k=1):         {:.4f}".format(si_b))
print("Simes TDR (k=1):           {:.2f}%".format(si_t))
print("Simes Power (k=1):         {:.2f}%".format(si_p))
ca_b = row["Cauchy_Boundary_Error"]
ca_t = row["Cauchy_FDR"] * 100
ca_p = row["Cauchy_Power"] * 100
print("Cauchy bFDR (k=1):        {:.4f}".format(ca_b))
print("Cauchy TDR (k=1):          {:.2f}%".format(ca_t))
print("Cauchy Power (k=1):        {:.2f}%".format(ca_p))
av_b = row["AverageP_Boundary_Error"]
av_t = row["AverageP_FDR"] * 100
av_p = row["AverageP_Power"] * 100
print("AverageP bFDR (k=1):      {:.4f}".format(av_b))
print("AverageP TDR (k=1):        {:.2f}%".format(av_t))
print("AverageP Power (k=1):      {:.2f}%".format(av_p))

df_raw.to_csv("/repo/RealData/genedata_raw_results.csv", index=False)
df_summary.to_csv("/repo/RealData/genedata_summary.csv", index=False)
print("\nResults saved to genedata_raw_results.csv and genedata_summary.csv")
