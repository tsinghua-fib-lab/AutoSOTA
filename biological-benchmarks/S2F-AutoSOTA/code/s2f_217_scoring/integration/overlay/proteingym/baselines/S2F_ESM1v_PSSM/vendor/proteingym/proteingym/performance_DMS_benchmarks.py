import pandas as pd
import numpy as np
import os
import argparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import warnings
import json
warnings.simplefilter(action='ignore', category=FutureWarning)
            
def minmax(x):
    return ( (x - np.min(x)) / (np.max(x) - np.min(x)) ) 

def calc_ndcg(y_true, y_score, **kwargs):
    '''
    Inputs:
        y_true: an array of the true scores where higher score is better
        y_score: an array of the predicted scores where higher score is better
    Options:
        quantile: If True, uses the top k quantile of the distribution
        top: under the quantile setting this is the top quantile to
            keep in the gains calc. This is a PERCENTAGE (i.e input 10 for top 10%)
    Notes:
        Currently we're calculating NDCG on the continuous value of the DMS
        I tried it on the binary value as well and the metrics seemed mostly
        the same.
    '''
    if 'quantile' not in kwargs:
        kwargs['quantile'] = True
    if 'top' not in kwargs:
        kwargs['top'] = 10
    if kwargs['quantile']:
        k = np.floor(y_true.shape[0]*(kwargs['top']/100)).astype(int)
    else:
        k = kwargs['top']
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    gains = minmax(y_true)
    ranks = np.argsort(np.argsort(-y_score)) + 1
    
    if k == 'all':
        k = len(ranks)
    #sub to top k
    ranks_k = ranks[ranks <= k]
    gains_k = gains[ranks <= k]
    #all terms with a gain of 0 go to 0
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]
    
    #if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return (0)
    
    #discounted cumulative gains
    dcg = np.sum([g/np.log2(r+1) for r,g in zip(ranks_fil, gains_fil)])
    
    #ideal dcg - calculated based on the top k actual gains
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= k]
    ideal_gains_k = gains[ideal_ranks <= k]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum([g/np.log2(r+1) for r,g in zip(ideal_ranks_fil, ideal_gains_fil)])
    
    #normalize
    ndcg = dcg/idcg
    
    return (ndcg)
def calc_toprecall(true_scores, model_scores, top_true=10, top_model=10):  
    top_true = (true_scores >= np.percentile(true_scores, 100-top_true))
    top_model = (model_scores >= np.percentile(model_scores, 100-top_model))
    
    TP = (top_true) & (top_model)
    recall = TP.sum() / (top_true.sum()) if top_true.sum() > 0 else 0
    
    return (recall)

def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()

def compute_bootstrap_standard_error(df, number_assay_reshuffle=10000):
    """
    Computes the non-parametric bootstrap standard error for the mean estimate of a given performance metric (eg., Spearman, AUC) across DMS assays (ie., the sample standard deviation of the mean across bootstrap samples)
    """
    model_names = df.columns
    mean_performance_across_samples = []
    for sample in range(number_assay_reshuffle):
        mean_performance_across_samples.append(df.sample(frac=1.0, replace=True).mean(axis=0)) #Resample a dataset of the same size (with replacement) then take the sample mean
    mean_performance_across_samples=pd.DataFrame(data=mean_performance_across_samples,columns=model_names)
    return mean_performance_across_samples.std(ddof=1)

def compute_bootstrap_standard_error_functional_categories(df, number_assay_reshuffle=10000):
    """
    Computes the non-parametric bootstrap standard error for the mean estimate of a given performance metric (eg., Spearman, AUC) across DMS assays (ie., the sample standard deviation of the mean across bootstrap samples)
    """
    model_names = df.columns
    mean_performance_across_samples = {}
    for category, group in df.groupby("Selection Type"):
        mean_performance_across_samples[category] = []
        for sample in range(number_assay_reshuffle):
            mean_performance_across_samples[category].append(group.sample(frac=1.0, replace=True).mean(axis=0)) #Resample a dataset of the same size (with replacement) then take the sample mean
        mean_performance_across_samples[category]=pd.DataFrame(data=mean_performance_across_samples[category])
    categories = list(mean_performance_across_samples.keys())
    combined_averages = mean_performance_across_samples[categories[0]].copy()
    for category in categories[1:]:
        combined_averages += mean_performance_across_samples[category]
    combined_averages /= len(categories)
    return combined_averages.std(ddof=1)


proteingym_folder_path = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser(description='ProteinGym performance analysis')
    parser.add_argument('--input_scoring_files_folder', type=str, help='Name of folder where all input scores are present (expects one scoring file per DMS)')
    parser.add_argument('--output_performance_file_folder', default='./outputs/tranception_performance', type=str, help='Name of folder where to save performance analysis files')
    parser.add_argument('--DMS_reference_file_path', type=str, help='Reference file with list of DMSs to consider')
    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS datasets')
    parser.add_argument('--indel_mode', action='store_true', help='Whether to score sequences with insertions and deletions')
    parser.add_argument('--performance_by_depth', action='store_true', help='Whether to compute performance by mutation depth')
    parser.add_argument('--config_file', default=f'{os.path.dirname(proteingym_folder_path)}/config.json', type=str, help='Path to config file containing model information')
    args = parser.parse_args()
    
    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    mapping_protein_seq_DMS["MSA_Neff_L_category"] = mapping_protein_seq_DMS["MSA_Neff_L_category"].apply(lambda x: x[0].upper() + x[1:] if type(x) == str else x)
    num_DMS=len(mapping_protein_seq_DMS)
    print("There are {} DMSs in mapping file".format(num_DMS))
    
    with open(args.config_file) as f:
        config = json.load(f)
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/constants.json") as f:
        constants = json.load(f)
    uniprot_function_lookup = mapping_protein_seq_DMS[["UniProt_ID","coarse_selection_type"]]
    uniprot_function_lookup.columns = ["UniProt_ID", "Selection Type"]
    uniprot_Neff_lookup = mapping_protein_seq_DMS[['UniProt_ID','MSA_Neff_L_category']].drop_duplicates()
    uniprot_Neff_lookup.columns=['UniProt_ID','MSA_Neff_L_category']
    uniprot_taxon_lookup = mapping_protein_seq_DMS[['UniProt_ID','taxon']].drop_duplicates()
    uniprot_taxon_lookup.columns=['UniProt_ID','Taxon']
    if args.indel_mode:
        args.performance_by_depth = False

    score_variables = list(config["model_list_zero_shot_substitutions_DMS"].keys()) if not args.indel_mode else list(config["model_list_zero_shot_indels_DMS"].keys())
    if not os.path.isdir(args.output_performance_file_folder):
        os.mkdir(args.output_performance_file_folder)
    for metric in ['Spearman','AUC','MCC',"NDCG","Top_recall"]:
        if not os.path.isdir(args.output_performance_file_folder+os.sep+metric):
            os.mkdir(args.output_performance_file_folder+os.sep+metric)
    
    model_types={}
    for model in score_variables:
        model_types[model]=config["model_list_zero_shot_substitutions_DMS"][model]["model_type"] if not args.indel_mode else config["model_list_zero_shot_indels_DMS"][model]["model_type"]
    model_types=pd.DataFrame.from_dict(model_types,columns=['Model type'],orient='index')
    model_details=pd.DataFrame.from_dict(constants["model_details"],columns=['Model details'],orient='index')
    model_references=pd.DataFrame.from_dict(constants["model_references"],columns=['References'],orient='index')
    clean_names = constants["clean_names"]
    performance_all_DMS={}
    output_filename={}
    for metric in ['Spearman','AUC','MCC', "NDCG", "Top_recall"]:
        performance_all_DMS[metric]={}
        mutation_type = "substitutions" if not args.indel_mode else "indels"
        output_filename[metric]="DMS_" + mutation_type + "_" + metric
        for i, score in enumerate(score_variables):
            performance_all_DMS[metric][score]=i
            if not args.indel_mode and args.performance_by_depth:
                for depth in ['1','2','3','4','5+']:
                    performance_all_DMS[metric][score+'_'+depth] = i
        performance_all_DMS[metric]['number_mutants']=-1
        performance_all_DMS[metric]["Selection Type"] = -1 
        performance_all_DMS[metric]["UniProt_ID"] = -1 
        performance_all_DMS[metric]['MSA_Neff_L_category']=-1
        performance_all_DMS[metric]['Taxon']=-1
        performance_all_DMS[metric]=pd.DataFrame.from_dict(performance_all_DMS[metric],orient='index').reset_index()
        performance_all_DMS[metric].columns=['score','score_index']

    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    i = 0
    for DMS_id in list_DMS:
        try:
            print(DMS_id)    
            UniProt_ID = mapping_protein_seq_DMS["UniProt_ID"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
            DMS_filename = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
            selection_type = mapping_protein_seq_DMS["coarse_selection_type"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
            MSA_Neff_L_category	= mapping_protein_seq_DMS["MSA_Neff_L_category"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
            Taxon = mapping_protein_seq_DMS["taxon"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]

            DMS_file = pd.read_csv(args.DMS_data_folder+os.sep+DMS_filename)
            print("Length DMS: {}".format(len(DMS_file)))
            merged_scores = pd.read_csv(args.input_scoring_files_folder + os.sep + DMS_id + ".csv") #We assume no missing value (all models were enforced to score all mutants)
            if 'mutant' not in merged_scores: merged_scores['mutant'] = merged_scores['mutated_sequence'] #if mutant not in DMS file we default to mutated_sequence (eg., for indels)
        except:
            print(f"Scoring file for {DMS_id} missing")
            continue

        if not args.indel_mode and args.performance_by_depth:
            merged_scores['mutation_depth']=merged_scores['mutant'].apply(lambda x: len(x.split(":")))
            merged_scores['mutation_depth_grouped']=merged_scores['mutation_depth'].apply(lambda x: '5+' if x >=5 else str(x))
        performance_DMS = {}
        for metric in ['Spearman','AUC','MCC','NDCG','Top_recall']:
            performance_DMS[metric]={}
        for score in score_variables:
            if score not in merged_scores:
                print("Model scores for {} not in merged scores for DMS {}".format(score,DMS_id))
                performance_DMS["Spearman"][score] = np.nan
                performance_DMS["AUC"][score] = np.nan
                performance_DMS["MCC"][score] = np.nan
                performance_DMS["NDCG"][score] = np.nan 
                performance_DMS["Top_recall"][score] = np.nan
                continue
            performance_DMS['Spearman'][score] = spearmanr(merged_scores['DMS_score'], merged_scores[score])[0]
            performance_DMS["NDCG"][score] = calc_ndcg(merged_scores['DMS_score'], merged_scores[score])
            performance_DMS["Top_recall"][score] = calc_toprecall(merged_scores['DMS_score'], merged_scores[score])
            try:
                performance_DMS['AUC'][score] = roc_auc_score(y_true=merged_scores['DMS_score_bin'], y_score=merged_scores[score])
            except:
                print("AUC issue with: {} for model: {}".format(DMS_id,score))
                performance_DMS['AUC'][score] = np.nan
            try:
                median_cutoff=merged_scores[score].median()
                merged_scores[score+"_bin"]=merged_scores[score].map(lambda x: 1 if x >= median_cutoff else 0)
                performance_DMS['MCC'][score] = matthews_corrcoef(y_true=merged_scores['DMS_score_bin'], y_pred=merged_scores[score+"_bin"])
            except:
                print("MCC issue with: {} for model: {}".format(DMS_id,score))
                performance_DMS['MCC'][score] = np.nan

        if not args.indel_mode and args.performance_by_depth:
            for score in score_variables:
                if score not in merged_scores:
                    print("Model scores for {} not in merged scores for DMS {}".format(score,DMS_id))
                    for depth in ['1','2','3','4','5+']:
                        performance_DMS["Spearman"][score+'_'+depth] = np.nan
                        performance_DMS["AUC"][score+'_'+depth] = np.nan
                        performance_DMS["MCC"][score+'_'+depth] = np.nan 
                        performance_DMS["NDCG"][score+'_'+depth] = np.nan
                        performance_DMS["Top_recall"][score+'_'+depth] = np.nan
                    continue
                for depth in ['1','2','3','4','5+']:
                    merged_scores_depth = merged_scores[merged_scores.mutation_depth_grouped==depth]
                    if len(merged_scores_depth) > 0:
                        performance_DMS['Spearman'][score+'_'+depth] = spearmanr(merged_scores_depth['DMS_score'], merged_scores_depth[score])[0]
                        performance_DMS["NDCG"][score+'_'+depth] = calc_ndcg(merged_scores_depth['DMS_score'], merged_scores_depth[score])
                        performance_DMS["Top_recall"][score+'_'+depth] = calc_toprecall(merged_scores_depth['DMS_score'], merged_scores_depth[score])
                        try:
                            performance_DMS['AUC'][score+'_'+depth] = roc_auc_score(y_true=merged_scores_depth['DMS_score_bin'], y_score=merged_scores_depth[score])
                        except:
                            performance_DMS['AUC'][score+'_'+depth] = np.nan
                        try:
                            performance_DMS['MCC'][score+'_'+depth] = matthews_corrcoef(y_true=merged_scores_depth['DMS_score_bin'], y_pred=merged_scores_depth[score+"_bin"])
                        except:
                            performance_DMS['MCC'][score+'_'+depth] = np.nan
                    else:
                        performance_DMS['Spearman'][score+'_'+depth] = np.nan
                        performance_DMS['AUC'][score+'_'+depth] = np.nan
                        performance_DMS['MCC'][score+'_'+depth] = np.nan
                        performance_DMS["NDCG"][score+'_'+depth] = np.nan
                        performance_DMS["Top_recall"][score+'_'+depth] = np.nan
        print("Number of mutants: {}".format(len(merged_scores['DMS_score'].values)))
        for metric in ['Spearman','AUC','MCC','NDCG','Top_recall']:
            performance_DMS[metric]['number_mutants']=len(merged_scores['DMS_score'].values)
            performance_DMS[metric]['UniProt_ID'] = UniProt_ID
            performance_DMS[metric]["Selection Type"] = selection_type
            performance_DMS[metric]['MSA_Neff_L_category'] = MSA_Neff_L_category
            performance_DMS[metric]['Taxon'] = Taxon
            performance_DMS[metric] = pd.DataFrame.from_dict(performance_DMS[metric],orient='index').reset_index()
            performance_DMS[metric].columns=['score',DMS_id]
            performance_all_DMS[metric]=pd.merge(performance_all_DMS[metric],performance_DMS[metric],on='score',how='left')
    for metric in ['Spearman','AUC','MCC','NDCG','Top_recall']:
        performance_all_DMS[metric]=performance_all_DMS[metric].set_index('score')
        del performance_all_DMS[metric]['score_index']
        performance_all_DMS[metric]=performance_all_DMS[metric].transpose()
        for var in performance_all_DMS[metric]:
            if var not in ['UniProt_ID','MSA_Neff_L_category','Taxon',"Selection Type"]:
                performance_all_DMS[metric][var]=performance_all_DMS[metric][var].astype(float).round(3)
            if var in ['number_mutants']:
                performance_all_DMS[metric][var]=performance_all_DMS[metric][var].astype(int)
        if not args.indel_mode and args.performance_by_depth:
            all_columns = performance_all_DMS[metric].columns
            performance_all_DMS_html=performance_all_DMS[metric].copy()
            performance_all_DMS_html.columns=performance_all_DMS_html.columns.map(lambda x: clean_names[x] if x in clean_names else x)
            all_not_depth_columns = all_columns[[all_columns[x].split("_")[-1] not in ['1','2','3','4','5+'] for x in range(len(all_columns))]]
            all_not_depth_columns_clean = all_not_depth_columns.map(lambda x: clean_names[x] if x in clean_names else x)
            performance_all_DMS_html[all_not_depth_columns_clean].to_html(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.html')
            DMS_perf_to_save = performance_all_DMS[metric].copy()[all_not_depth_columns]
            DMS_perf_to_save.columns = DMS_perf_to_save.columns.map(lambda x: clean_names[x] if x in clean_names else x)
            DMS_perf_to_save.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.csv', index_label="DMS ID")
        else:
            performance_all_DMS_html=performance_all_DMS[metric].copy()
            performance_all_DMS_html.columns = performance_all_DMS_html.columns.map(lambda x: clean_names[x] if x in clean_names else x)
            performance_all_DMS_html.to_html(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.html')
            DMS_perf_to_save = performance_all_DMS[metric].copy()
            DMS_perf_to_save.columns = DMS_perf_to_save.columns.map(lambda x: clean_names[x] if x in clean_names else x)
            DMS_perf_to_save.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_DMS_level.csv', index_label="DMS ID")
        
        if not args.indel_mode:
            uniprot_metric_performance = performance_all_DMS[metric].groupby(['UniProt_ID']).mean(numeric_only=True)
            uniprot_function_metric_performance = performance_all_DMS[metric].groupby(['UniProt_ID',"Selection Type"]).mean(numeric_only=True)
            uniprot_metric_performance = uniprot_metric_performance.reset_index()
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_Neff_lookup,on='UniProt_ID', how='left')
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_taxon_lookup,on='UniProt_ID', how='left')
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_function_lookup,on="UniProt_ID",how="left")
            del uniprot_metric_performance['number_mutants']
            del uniprot_function_metric_performance["number_mutants"]
            uniprot_level_average = uniprot_metric_performance.mean(numeric_only=True)
            uniprot_function_level_average = uniprot_function_metric_performance.groupby("Selection Type").mean(numeric_only=True)
            # bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance['TranceptEVE_L'],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_function_level_average = uniprot_function_level_average.reset_index()
            final_average = uniprot_function_level_average.mean(numeric_only=True) 
            if args.performance_by_depth:
                cols = [column for column in all_not_depth_columns if column not in ["number_mutants","Taxon","MSA_Neff_L_category","Selection Type","UniProt_ID"]]
                top_model = final_average.loc[cols].idxmax()
            else:
                top_model = final_average.idxmax()
            bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance[top_model],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_metric_performance.loc['Average'] = uniprot_level_average
            uniprot_function_level_average.loc['Average'] = final_average
            uniprot_metric_performance=uniprot_metric_performance.round(3)
            uniprot_function_level_average=uniprot_function_level_average.round(3)
            if args.performance_by_depth:
                uniprot_metric_performance[[column for column in all_not_depth_columns if column != "number_mutants"]].to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_Uniprot_level.csv', index=False)
                performance_by_depth = {}
                all_not_depth_columns = [x for x in all_not_depth_columns if x not in ['number_mutants',"UniProt_ID","MSA_Neff_L_category","Taxon"]]
                for depth in ['1','2','3','4','5+']:
                    depth_columns = all_columns[[all_columns[x].split("_")[-1]==depth for x in range(len(all_columns))]]
                    performance_by_depth[depth] = uniprot_function_metric_performance[depth_columns].mean(numeric_only=True).reset_index()
                    performance_by_depth[depth]['model_name'] = performance_by_depth[depth]['score'].map(lambda x: '_'.join(x.split('_')[:-1]))
                    performance_by_depth[depth]=performance_by_depth[depth][['model_name',0]]
                    performance_by_depth[depth].columns = ['model_name','Depth_'+depth]
                    performance_by_depth[depth].set_index('model_name', inplace=True)
                uniprot_function_level_average = uniprot_function_level_average[all_not_depth_columns]
            else:
                uniprot_metric_performance.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + '_Uniprot_level.csv', index=False)
            uniprot_function_level_average.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + output_filename[metric] + "_Uniprot_Selection_Type_level.csv",index=False)
            if args.performance_by_depth:
                performance_by_MSA_depth = performance_all_DMS[metric].groupby(["UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["MSA_Neff_L_category"]).mean(numeric_only=True)[[col for col in all_not_depth_columns if col != "Selection Type"]].transpose()
            else:
                performance_by_MSA_depth = performance_all_DMS[metric].groupby(["UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["MSA_Neff_L_category"]).mean(numeric_only=True).transpose()
            performance_by_MSA_depth = performance_by_MSA_depth[['Low','Medium','High']]
            performance_by_MSA_depth.columns = ['Low_MSA_depth','Medium_MSA_depth','High_MSA_depth']
            if args.performance_by_depth:
                performance_by_taxon = performance_all_DMS[metric].groupby(["UniProt_ID","Taxon"]).mean(numeric_only=True).groupby(["Taxon"]).mean(numeric_only=True)[[col for col in all_not_depth_columns if col != "Selection Type"]].transpose()
            else:
                performance_by_taxon = performance_all_DMS[metric].groupby(["UniProt_ID","Taxon"]).mean(numeric_only=True).groupby(["Taxon"]).mean(numeric_only=True).transpose()
            performance_by_taxon = performance_by_taxon[['Human','Eukaryote','Prokaryote','Virus']]
            performance_by_taxon.columns = ['Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus']
            performance_by_function = uniprot_function_level_average.drop(labels="Average",axis=0).set_index("Selection Type").transpose()
            performance_by_function.columns = ["Function_"+x for x in performance_by_function.columns]
            
            summary_performance = pd.merge(pd.DataFrame(final_average,columns=['Average_'+metric]), performance_by_MSA_depth,left_index=True, right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_taxon,left_index=True, right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_function,left_index=True, right_index=True, how='inner')
            if args.performance_by_depth:
                for depth in ['1','2','3','4','5+']:
                    summary_performance = pd.merge(summary_performance, performance_by_depth[depth],left_index=True, right_index=True,how='inner')
            final_column_order = ['Model_name','Model type','Average_'+metric,'Bootstrap_standard_error_'+metric,'Function_Activity','Function_Binding','Function_Expression','Function_OrganismalFitness','Function_Stability','Low_MSA_depth','Medium_MSA_depth','High_MSA_depth','Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus','Depth_1','Depth_2','Depth_3','Depth_4','Depth_5+','Model details','References']

        else:
            performance_all_DMS[metric].loc["Average"] = performance_all_DMS[metric].mean(numeric_only=True)
            uniprot_metric_performance = performance_all_DMS[metric].groupby(['UniProt_ID']).mean(numeric_only=True)
            uniprot_function_metric_performance = performance_all_DMS[metric].groupby(['UniProt_ID',"Selection Type"]).mean(numeric_only=True)
            uniprot_metric_performance = pd.merge(uniprot_metric_performance,uniprot_function_lookup,on="UniProt_ID",how="left")
            del uniprot_metric_performance['number_mutants']
            uniprot_level_average = uniprot_metric_performance.mean(numeric_only=True)
            del uniprot_function_metric_performance["number_mutants"]
            uniprot_function_level_average = uniprot_function_metric_performance.groupby("Selection Type").mean(numeric_only=True)
            # bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance['TranceptEVE_M'],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_function_level_average = uniprot_function_level_average.reset_index()
            final_average = uniprot_function_level_average.mean(numeric_only=True) 
            top_model = final_average.idxmax()
            bootstrap_standard_error = pd.DataFrame(compute_bootstrap_standard_error_functional_categories(uniprot_function_metric_performance.subtract(uniprot_function_metric_performance[top_model],axis=0)),columns=["Bootstrap_standard_error_"+metric])
            uniprot_metric_performance.loc['Average'] = uniprot_level_average
            uniprot_function_level_average.loc['Average'] = final_average
            uniprot_metric_performance=uniprot_metric_performance.round(3)
            uniprot_function_level_average=uniprot_function_level_average.round(3)
            
            performance_by_MSA_depth = performance_all_DMS[metric].groupby(["UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["MSA_Neff_L_category"]).mean(numeric_only=True).transpose()
            performance_by_MSA_depth = performance_by_MSA_depth[['Low','Medium','High']]
            performance_by_MSA_depth.columns = ['Low_MSA_depth','Medium_MSA_depth','High_MSA_depth']
            performance_by_taxon = performance_all_DMS[metric].groupby(["UniProt_ID","Taxon"]).mean(numeric_only=True).groupby(["Taxon"]).mean(numeric_only=True).transpose()
            performance_by_taxon = performance_by_taxon[['Human','Eukaryote','Prokaryote','Virus']]
            performance_by_taxon.columns = ['Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus']
            performance_by_function = uniprot_function_level_average.drop(labels="Average",axis=0).set_index("Selection Type").transpose()
            performance_by_function.columns = ["Function_"+x for x in performance_by_function.columns]
            
            summary_performance = pd.merge(pd.DataFrame(final_average,columns=['Average_'+metric]), performance_by_MSA_depth,left_index=True,right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_taxon,left_index=True, right_index=True,how='inner')
            summary_performance = pd.merge(summary_performance, performance_by_function,left_index=True, right_index=True, how='inner')
            final_column_order = ['Model_name','Model type','Average_'+metric,'Bootstrap_standard_error_'+metric,'Function_Activity','Function_Binding','Function_Expression','Function_OrganismalFitness','Function_Stability','Low_MSA_depth','Medium_MSA_depth','High_MSA_depth','Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus','Model details','References']
        summary_performance.sort_values(by='Average_'+metric,ascending=False,inplace=True)
        summary_performance.index.name = 'Model_name'
        summary_performance.reset_index(inplace=True)
        summary_performance.index = range(1,len(summary_performance)+1)
        summary_performance.index.name = 'Model_rank'
        summary_performance = pd.merge(summary_performance, bootstrap_standard_error, left_on='Model_name', right_index=True, how='left')
        summary_performance = pd.merge(summary_performance, model_types, left_on='Model_name', right_index=True, how='left')
        summary_performance = pd.merge(summary_performance, model_details, left_on='Model_name', right_index=True, how='left')
        summary_performance = pd.merge(summary_performance, model_references, left_on='Model_name', right_index=True, how='left')
        summary_performance=summary_performance.round(3)
        summary_performance['Model_name']=summary_performance['Model_name'].map(lambda x: clean_names[x] if x in clean_names else x)
        summary_performance=summary_performance.reindex(columns=final_column_order)
        summary_performance.to_csv(args.output_performance_file_folder + os.sep + metric + os.sep + 'Summary_performance_'+output_filename[metric]+'.csv')
        summary_performance.to_html(args.output_performance_file_folder + os.sep + metric + os.sep + 'Summary_performance_'+output_filename[metric]+'.html')

if __name__ == '__main__':
    main()