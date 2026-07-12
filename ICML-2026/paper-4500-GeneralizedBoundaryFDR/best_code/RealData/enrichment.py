from utils import *
import pandas as pd
import requests
import json
import time
# Read your data file
df = pd.read_csv(file_path, sep='\t')
# data cleaning
df_clean = df.dropna(subset=['Gene.symbol', 'P.Value']).copy()
bfdr_data = df_clean[['Gene.symbol', 'P.Value']]
p_values_array = bfdr_data['P.Value'].values
alpha = 0.05
all_results_records_pi1 = []
# p-domino       
reject_indices_1pdomino, boundary_indices_1pdomino,_ = simesdomino(p_values_array,  alpha=alpha)

# sl-method
reject_indices_sl, boundary_indices_sl ,_ = sl_procedure(p_values_array,  alpha=alpha)

# 2p-domino
reject_indices_2pdomino, boundary_indices_2pdomino = bfdr_k_domino(p_values_array, k = 2,  alpha=alpha)

# 3p-domino
reject_indices_3pdomino, boundary_indices_3pdomino = bfdr_k_domino(p_values_array, k = 3,  alpha=alpha)

# BH
reject_indices = BH(p_values_array, alpha)

genes_array = bfdr_data['Gene.symbol'].values
gene_lists = {
    '1p-domino': genes_array[reject_indices_1pdomino].tolist(),
    '2p-domino': genes_array[reject_indices_2pdomino].tolist(),
    '3p-domino': genes_array[reject_indices_3pdomino].tolist(),
    'SL-method': genes_array[reject_indices_sl].tolist(),
    'BH-method': genes_array[reject_indices].tolist()
}


target_pathways = [
    "PPAR signaling pathway", 
    "AMPK signaling pathway", 
    "Fatty acid degradation", 
    "Regulation of lipolysis in adipocytes", 
    "Insulin signaling pathway",
    "Glycerolipid metabolism",
    "ECM-receptor interaction",
    "Fc gamma R-mediated phagocytosis"
]


targeted_results = []
database = 'KEGG_2021_Human'

for method_name, genes in gene_lists.items():
    if len(genes) > 0:
        method_scores = {'Method': f"{method_name} (n={len(genes)})"}
        
        df_enrich = get_enrichr_scores(genes, library_name=database)
        
        if df_enrich is not None and not df_enrich.empty:
            for pathway in target_pathways:
                matched_rows = df_enrich[df_enrich['Term'].str.contains(pathway, case=False, na=False, regex=False)]
                if not matched_rows.empty:
                    score = matched_rows.iloc[0]['Combined_Score']
                    method_scores[pathway] = round(score, 2)
                else:
                    method_scores[pathway] = 0.0
        else:
            for pathway in target_pathways:
                method_scores[pathway] = 0.0
                
        targeted_results.append(method_scores)
        time.sleep(1) 

df_targeted = pd.DataFrame(targeted_results)
df_targeted.set_index('Method', inplace=True)
df_table6_style = df_targeted.T
print(df_table6_style.to_string())
