import pandas as pd 
import numpy as np 
import os 
import sys 
import datatable as dt
import itertools
from datatable import f, g, by, sort, join, update, ifelse
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

class DataModuleWHI: 

    def __init__(self, root='/data/whi/data/main_study/', \
                       os_end_day = None, 
                       fetch_ctos = False, 
                       fetch_merged = False, 
                       miss_thresh = 0.20, 
                       drug_thresh = 0.0005, 
                       params = {}): 
        self.root = root 
        self.ctos_table = self.load_ctos_table(fetch = fetch_ctos, os_end_day = os_end_day)
        self.merged_table = self.load_covariates(fetch = fetch_merged, \
                        miss_thresh = miss_thresh, drug_thresh = drug_thresh)
        self.params = params
        assert len(self.params.keys()) != 0
    
    def load_ctos_table(self, fetch, os_end_day):
        if fetch:
            return self.fetch_ctos_table(os_end_day)
        else:
            return pd.read_csv(os.path.join(self.root, 'processed/ctos_table.csv'))
    
    def fetch_ctos_table(self, os_end_day): 
        # Table with outcome information
        out = dt.fread("/data/whi/data/main_study/csv/outc_adj_bio.csv")
        # Table with CT follow-up information
        ct_fu = dt.fread("/data/whi/data/main_study/csv/adh_ht_pub.csv")[:, ['ID','ADHRATE','ENDDY']]
        # Table with study participation and treatment arm information
        std_trt = dt.fread("/data/whi/data/main_study/csv/dem_ctos_bio.csv")[:, ['ID','HRTARM','OSFLAG']]
        
        out.key = 'ID'  
        
        # Set end of follow-up for CT patients as the last day of the year 
        # with follow-up (set as non-missing ADHRATE)
        ct_end = ct_fu[f.ADHRATE != None,:][0, 'ENDDY', by(f.ID), sort(-f.ENDDY)]
        ct_end.names = {'ENDDY': 'END_DY'}
        ct_end.key = 'ID'
        
        # List of outcomes for the global index
        glbl_list = ['CHD', 'BREAST', 'STROKE', 'PE', \
            'ENDMTRL', 'COLORECTAL', 'BKHIP', 'DEATH']    
        # List of other outcomes of interest
        other_list = ['PTCA', 'DVT']

        ## Construct ct_df: data table for CT patients
        
        # Select one row for each ID -> 
        # Select patient in the CT (HRTARM in either arm) -> 
        # Join with END_DY and outcome information
        ct_df = std_trt[0, 'HRTARM', by(f.ID)][\
                    (f.HRTARM == 'E+P intervention') | (f.HRTARM == 'E+P control'), :][\
                    :,:, join(ct_end), join(out)]
        # Code OS = 0 for future merging, HRT to 1/0
        ct_df[:, update(OS = 0, HRTARM = ifelse(f.HRTARM == 'E+P intervention', 1, 0))]    
        
        
        for i in glbl_list + other_list:
            # Event occurred if event was recorded (f[i] == 1) within the follow-up period (f[i+'DY'] <= f['END_DY'])
            ct_df[:, update(**{i+'_E' : ifelse(f[i] == 1, ifelse(f[i+'DY'] <= f['END_DY'], 1, 0), 0)})]    
            # Recode the follow-up time according to event occurence
            ct_df[:, update(**{i+'_DY' : ifelse(f[i+'_E'] == 1, f[i+'DY'], f['END_DY'])})]
            # CONDTION_EDY is the observed time of event (None if censored), used for global index compliation
            ct_df[:, update(**{i+'_EDY' : ifelse(f[i+'_E'] == 1, f[i+'_DY'], None)})]
            
        
        # Global index occured when any of the listed conditions occurred
        ct_df[:, update(GLBL_E = ifelse(dt.rowsum([f[j+'_E'] for j in glbl_list]) > 0, 1, 0))]
        # Time for global index is the smallest observed time of event if any event occurred, else the follow-up time
        ct_df[:, update(GLBL_DY = ifelse(f.GLBL_E == 1, dt.rowmin([f[j+'_EDY'] for j in glbl_list]), \
                                                        dt.rowmin([f[j+'_DY'] for j in glbl_list])))]
        
        # Retain needed variables
        ct_df = ct_df[:,['ID', 'OS', 'HRTARM'] + \
                      [j+'_E' for j in glbl_list + other_list + ['GLBL']] + \
                      [j+'_DY' for j in glbl_list + other_list +['GLBL']]]
        
        ## Construct os_df: data table for OS patients
        
        # Table with hysterectomy information
        hyst = dt.fread("/data/whi/data/main_study/csv/f2_ctos_bio.csv")[:,['ID','HYST']]
        # Table with information for pre-study exposure of HRT
        pre_hrt = dt.fread("/data/whi/data/main_study/csv/f43_ctos_bio.csv")[:,['ID','TOTPSTAT']]
        # Table with information for HRT exposure at one-year follow-up
        post_hrt = dt.fread("/data/whi/data/main_study/csv/f48_av1_os_pub.csv")[:,['ID','ELSTYR','PLSTYR','HRTCMBP']]   
        
        hyst.key = 'ID'
        pre_hrt.key = 'ID'
        post_hrt.key = 'ID'
        
        # Select one row for each ID -> 
        # Select patients in the OS (Did not use HRTARM to identify since some patients opted out of the OS) ->
        # Select patients without hyterectomy history ->
        # Select patients without current use of HRT ->
        # Merge with post-recruitment HRT exposure and outcome information
        os_df = std_trt[0, 'OSFLAG', by(f.ID)][\
                    f.OSFLAG == 'Yes',:][\
                    :,:, join(hyst)][f.HYST == 'No',:][\
                    :,:, join(pre_hrt)][(f.TOTPSTAT == 'Never used') | (f.TOTPSTAT == 'Past user'),:][\
                    :,:, join(post_hrt), join(out)]
        
        # HRTGRP = 1 if patient confirmed usage of combined estrogen and progesterone
        # HRTGRP = 0 if patient denied usage of both estrogen and progesterone
        # HRTGRP = -1 if patient confirmed usage of one and not the other
        # HRTGRP = -2 if non of the above (eg. answered no on HRTCMBP but at least one missing for ELSTYYR/PLSTYR)
        os_df[:, update(OS = 1,\
                        HRTGRP = ifelse(((f.ELSTYR == 'Yes') & (f.PLSTYR == 'Yes')) | (f.HRTCMBP == 'Yes'), 1,\
                                  ifelse((f.ELSTYR == 'No') & (f.PLSTYR == 'No'), 0, \
                                  ifelse(((f.ELSTYR == 'Yes') & (f.PLSTYR == 'No')) | ((f.ELSTYR == 'No') & (f.PLSTYR == 'Yes')), -1, -2))))]
        os_df = os_df[f.HRTGRP != -2,:]
        os_df[:, update(HRTARM = ifelse(f.HRTGRP == 1, 1, 0), END_DY = ifelse(os_end_day == None, f.ENDFOLLOWDY, os_end_day))]
        
        for i in glbl_list + other_list:
            # Event occurred if event was recorded (f[i] == 1) within the follow-up period (f[i+'DY'] <= f['END_DY'])
            os_df[:, update(**{i+'_E' : ifelse(f[i] == 1, ifelse(f[i+'DY'] <= f['END_DY'], 1, 0), 0)})]    
            # Recode the follow-up time according to event occurence
            os_df[:, update(**{i+'_DY' : ifelse(f[i+'_E'] == 1, f[i+'DY'], f['END_DY'])})]
            # CONDTION_EDY is the observed time of event (None if censored), used for global index compliation
            os_df[:, update(**{i+'_EDY' : ifelse(f[i+'_E'] == 1, f[i+'_DY'], None)})]
            
        # Global index occured when any of the listed conditions occurred
        os_df[:, update(GLBL_E = ifelse(dt.rowsum([f[j+'_E'] for j in glbl_list]) > 0, 1, 0))]
        # Time for global index is the smallest observed time of event if any event occurred, else the follow-up time
        os_df[:, update(GLBL_DY = ifelse(f.GLBL_E == 1, dt.rowmin([f[j+'_EDY'] for j in glbl_list]), \
                                                        dt.rowmin([f[j+'_DY'] for j in glbl_list])))]
        
        # Retain needed variables
        os_df = os_df[:,['ID', 'OS', 'HRTARM'] + \
                      [j+'_E' for j in glbl_list + other_list + ['GLBL']] + \
                      [j+'_DY' for j in glbl_list + other_list +['GLBL']] + ['HRTGRP']]
        
        
        # Row bind the two data tables (force = True for os_df having an extra HRTGRP variable)
        ctos_df = dt.rbind(ct_df, os_df, force = True).to_pandas()
        
        ctos_df.to_csv(os.path.join(self.root, 'processed/ctos_table.csv'), index = False)
        
        return ctos_df

    def load_covariates(self, fetch, miss_thresh, drug_thresh): 
        covariate_files = os.listdir(os.path.join(self.root, 'csv/'))
        skip_files = ['f153_medications_ctos_bio.csv', 'f153_barriers_ctos_bio.csv', \
            'f151_ctos_bio.csv', 'f38_ctos_fu_bio.csv', 'fust_ctos_bio.csv', \
            'f33x_ctos_bio.csv', 'bmd_hip_ctos_bio.csv', 'bmd_wbody_ctos_bio.csv', \
            'bmd_spine_ctos_bio.csv', 'ext_ctos_bio.csv', 'f134_ctos_bio.csv',\
            'f154_ctos_bio.csv', 'f155_ctos_bio.csv', 'f156_ctos_bio.csv',\
            'f157_ctos_bio.csv', 'f158_ctos_bio.csv']
        res_files = [f for f in covariate_files \
                    if 'ctos' in f and 'f60' not in f \
                    and 'outc' not in f and f not in skip_files]
        res_files.append('non_commercial_cbc_bio.csv') # add leftover csv file
        res_files = sorted(res_files)

        # List column names identifying the nature of visits and the coding for
        # screeing visits for future filtering
        visit_col = {
            'f80_ctos_bio.csv': ('F80VTYP', 'Screening'), 
            'f45_ctos_bio.csv': ('F45VTYP', 'Screening Visit'),
            'f37_ctos_bio.csv': ('F37VTYP', 'Screening'),
            'non_commercial_cbc_bio.csv': ('CBCVTYP', 'Screening'),
            'f44_ctos_bio.csv': ('F44VTYP', 'Screening')
        }

        # Iterate through files and do pre-processing
        if fetch:
            merged_table = self.ctos_table.copy()
            for i in tqdm(range(len(res_files))): 
                file = res_files[i]
                # Read the whole table to avoid mixed column types, which leads to confusion 
                # in imputation later. May consider prefiltering the dataset by reading only 
                # the ID column and finding row positions matching our cohort...
                df = pd.read_csv(os.path.join(self.root, 'csv/'+file), low_memory = False)
                # Filter tables with multiple visits to only screening visits
                if file in visit_col.keys(): 
                    col, val = visit_col[file]
                    df = df[df[col] == val]
                # Pivot table for baseline medication history
                if 'f44' in file: 
                    # Threshold = 0.005: 119 medications
                    # Threshold = 0.001: 560 medications
                    # Threshold = 0.0005: 937 medications
                    # Threshold = 0.0001: 2457 medications

                    ## remove rows with nan medication codes 
                    df = df.dropna(subset = ['MEDNDC'])

                    ## remove med codes with low prevalence 
                    pt_no = len(pd.unique(df['ID']))
                    med_cnt = df['MEDNDC'].value_counts()
                    drug_list = med_cnt[np.where((med_cnt/pt_no >= drug_thresh).values)[0]].index
                    df = df[df['MEDNDC'].isin(drug_list)]

                    ## apply pivot table 
                    df['DRUG'] = 1
                    df = pd.pivot_table(df, values=['DRUG', 'ADULTDY'], index=['ID'], columns=['MEDNDC'], aggfunc=np.sum)
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    df.fillna(0.)
                    df.reset_index(inplace=True)
                if 'HRTARM' in df.columns.values:
                    df = df.drop(columns = 'HRTARM')
                merged_table = merged_table.merge(df, how='left', on='ID').reset_index(drop=True)
            
            # Drop columns with missing rate greater than miss_thresh
            not_miss_pos = np.where(merged_table.isnull().sum()/len(merged_table) <= miss_thresh)[0]
            merged_table = merged_table.iloc[:,not_miss_pos]

            # 1. Drop columns with only one possible value, has 'FLAG' in name (identifier of randomization into studies)
            #    or HRTGRP (identifier of os patients using only one of estrogen and progesterone)
            # 2. Get dummy of categorical variables (operational definition: number of unique values > 8 or strings)
            out_cols = ['ID', 'OS', 'HRTARM'] + [f for f in merged_table.columns.values if ('_E' in f) or ('_DY' in f)]
            cat_cols = []
            cont_cols = []

            for col in [f for f in merged_table.columns.values if f not in out_cols]:
                val = merged_table[col].dropna().values
                unique_num = len(pd.unique(val))
                if (unique_num <= 1) or ('VY' in col) or ('FLAG' in col) or (col == 'HRTGRP'): 
                    continue
                elif (unique_num <= 8) or (type(val[0]) == str):
                    cat_cols.append(col)
                else:
                    cont_cols.append(col)

            out_table = merged_table[out_cols]
            cat_table = merged_table[cat_cols]
            cont_table = merged_table[cont_cols]

            # Do mean impuation for numerics and most-prevelant imputation for categoricals
            cat_table =  pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(cat_table), columns = cat_cols)
            cont_table = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(cont_table), columns = cont_cols)
            cont_table.to_csv('/data/whi/data/main_study/processed/cont_table.csv')
            self.pre_norm_cont_table = cont_table
            for col in cont_table.columns.values: 
                cont_table[col] = (cont_table[col] - cont_table[col].mean()) / cont_table[col].std()

            # Get dummies for categorical variables and merge the tables
            # Question: do we need to set drop_first = True? May increase model complexity if tree-based methods are used?
            cat_table = pd.get_dummies(cat_table)
            merged_table = pd.concat([out_table, cont_table, cat_table], axis=1)
            merged_table.to_csv('/data/whi/data/main_study/processed/merged.csv', index=False)
        else: 
            merged_table = pd.read_csv('/data/whi/data/main_study/processed/merged.csv', low_memory = False)
            self.pre_norm_cont_table = pd.read_csv('/data/whi/data/main_study/processed/cont_table.csv', low_memory = False)
        return merged_table
    
    def get_normalized_cutoff(self, col_name, cutoff): 
        mean_ = self.pre_norm_cont_table[col_name].mean()
        std_  = self.pre_norm_cont_table[col_name].std()
        return (cutoff - mean_) / std_
    
    def _check_covariate_groups(self): 
        covariates = np.loadtxt('/home/research/rct_obs_causal/experiments/p_covariates.csv', \
            delimiter=';', dtype='str')
        
        for covariate in covariates:
            cov_name, cutoff = covariate.split(',')
            cutoff = float(cutoff)
            if cutoff == 1 or cutoff == 0: 
                n1 = self.merged_table[self.merged_table[cov_name] == 1].shape[0]
                n0 = self.merged_table[self.merged_table[cov_name] == 0].shape[0]
            else: 
                norm_cutoff = self.get_normalized_cutoff(cov_name, cutoff)
                n1 = self.merged_table[self.merged_table[cov_name] >= norm_cutoff].shape[0]
                n0 = self.merged_table[self.merged_table[cov_name] < norm_cutoff].shape[0]
            assert n1 > 0
            assert n0 > 0

    def process_whi(self, 
                    outcome = 'GLBL', 
                    followup = 7*365, 
                    obs_type='confounder_concealment', 
                    downsize=False, 
                    downsize_proportion=0.5,
                    split='train',
                    bootstrap_seed=None, 
                    check_groups=False):
        # obs_type: confounding, selection, estimation 
        self.merged_table['EVENT'] = (self.merged_table['GLBL_DY'] <= followup)*self.merged_table['GLBL_E']
        if downsize: 
            print('[downsizing].')
            ids = self.merged_table['ID'].values
            train_ids, test_ids = train_test_split(ids, test_size=1-downsize_proportion, stratify=self.merged_table['OS'], random_state=self.params['grand_seed'])
            if split == 'train': 
                print('[using training split].')
                self.merged_table = self.merged_table[self.merged_table['ID'].isin(train_ids)]
            else: 
                print('[using testing split].')
                self.merged_table = self.merged_table[self.merged_table['ID'].isin(test_ids)]
        
        if bootstrap_seed is not None: 
            print(f'[generating bootstrap WHI dataset, seed: {bootstrap_seed}]')
            self.merged_table = self.merged_table.sample(n=self.merged_table.shape[0], \
                                    replace=True, random_state=bootstrap_seed).reset_index(drop=True)
            if check_groups:
                print(f['[checking covariate groups in merged table.]']) 
                self._check_covariate_groups()
                print(f['[all groups supported!]']) 
            
        assert obs_type in ['confounder_concealment', 'selection_bias', None]
        if obs_type == None:
            self.obs_tables = [self.merged_table[self.merged_table['OS'] == 1]]
        else: 
            # get our observational datasets (i.e. apply confounder concealment, etc.)
            group_to_name = {
                'age': ['AGE'], 
                'menstrual': ['ANYMENSA', 'MENOPSEA', 'MENO', 'LSTPAPDY'], 
                'dep': ['PSHTDEP'], 
                'prior_hrt': ['TOTE_', 'TOTESTAT_', 'TOTHCAT_'], 
                'body': ['BMI', 'TMINWK', 'HEIGHT', 'WEIGHT', 'WAIST', 'HIP', 'WHR'], 
                'lab': ['HEMOGLBN', 'PLATELET', 'WBC', 'HEMATOCR'], 
                'healthcare': ['MEDICARE_', 'TIMELAST']
            }
            obs_configs = self.params['obs_dict'][obs_type]
            num_obs = len(obs_configs)
            self.obs_tables = []

            # np.random.seed(self.params['selection_seed'])
            for i in range(num_obs):
                obs_config = obs_configs[i]
                if obs_config == None:
                    obs_table = self.merged_table[self.merged_table['OS'] == 1]
                    obs_table = obs_table.sample(n=obs_table.shape[0], \
                                    replace=True, random_state=self.params['obs_dict']['resample_seed'][i])
                    self.obs_tables.append(obs_table)
                    continue

                if obs_type == 'confounder_concealment': 
                    confound_list = list(itertools.chain(*[group_to_name[x] for x in obs_config.split('+')]))
                    updated_confound_list = []
                    for confounder in confound_list: 
                        if '_' in confounder or confounder == 'AGE': 
                            updated_confound_list += [x for x in self.merged_table.columns.values \
                                            if confounder in x]
                        else: 
                            updated_confound_list.append(confounder)
                    obs_table = self.merged_table[self.merged_table['OS'] == 1].drop(columns=updated_confound_list)
                else: 
                    obs_table_init = self.merged_table[self.merged_table['OS'] == 1]
                    sub_ids = obs_table_init[(obs_table_init['EVENT'] == 0) & (obs_table_init['HRTARM'] == 0)]['ID']     
                    rejected_ids    = np.random.choice(sub_ids.values, \
                            size=int(np.floor(len(sub_ids.values)*obs_config)), replace=False)
                    obs_table_inter = obs_table_init[~obs_table_init['ID'].isin(rejected_ids)]
                    obs_table = obs_table_inter.sample(n=obs_table_init.shape[0], \
                                replace=True, random_state=self.params['selection_seed'])

                self.obs_tables.append(obs_table)
        
    def get_datasets(self): 
        self.rct_table = self.merged_table[self.merged_table['OS'] == 0]
        #self.rct_table_partial = self.rct_table[self.rct_table['PREG_Yes'] == 1]
        import copy
        self.rct_table_partial = copy.deepcopy(self.rct_table)
        #self.obs_tables = [self.merged_table[self.merged_table['OS'] == 1]]
    
        return { 
            'rct-partial': self.rct_table_partial, 
            'rct-full': self.rct_table, 
            'obs': self.obs_tables
        }
        
if __name__ == '__main__': 
    whi = DataModuleWHI()
    whi.process_whi()
    whi.get_datasets()
    
