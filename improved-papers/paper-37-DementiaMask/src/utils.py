import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


import itertools
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    )

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='/tmp/dementia-model/', device = 'cuda:1'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_metric_max = -float('inf')
        self.val_metric_min = float('inf')
        self.delta = delta
        self.path = path
        self.device = device

    def __call__(self, val_metrics, model, great_is_better = True):
        if great_is_better:
            if val_metrics > self.val_metric_max:
                self.val_metric_max = val_metrics
                self.counter = 0
                self.save_checkpoint(val_metrics, model)
            elif val_metrics <= self.val_metric_max - self.delta:
                self.counter += 1
                #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('Early Stopping...')
        
        elif not great_is_better:
            if val_metrics < self.val_metric_min:
                self.val_metric_min = val_metrics
                self.counter = 0
                self.save_checkpoint(val_metrics, model)
            elif val_metrics >= self.val_metric_min - self.delta:
                self.counter += 1
                #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('Early Stopping...')
    

    def save_checkpoint(self, val_metric, model):
        """Saves model when metric improve."""
        if self.verbose:
            print(f'Model performance improves to {val_metric:.6f}.  Saving model ...')
        model.save_pretrained(self.path, from_pt=True)

    def load_checkpoint(self):
        if self.verbose:
            print("Load the best checkpoint...")
        model = AutoModelForSequenceClassification.from_pretrained(self.path).to(self.device)
        return model


def confoundSplit(p_pos_train_z1, p_pos_train_z0, p_mix_z1, alpha_test):
    """Calculate probability constraint given some priors"""

    assert 0 <= p_pos_train_z1 <= 1
    assert 0 <= p_pos_train_z0 <= 1
    assert 0 <= p_mix_z1 <= 1
    assert alpha_test >= 0

    C_z = p_mix_z1

    p_mix_z0 = 1 - p_mix_z1

    # C_y = p_train(y=1) = p_train(z=0) * p_train(y=1|z=0) + p_train(z=1) * p_train(y=1|z=1)
    # C_y = p_test(y=1) = p_test(z=0) * p_test(y=1|z=0) + p_test(z=1) * p_test(y=1|z=1)
    C_y = p_mix_z0 * p_pos_train_z0 + p_mix_z1 * p_pos_train_z1

    p_pos_test_z0 = C_y / (1 - (1 - alpha_test) * C_z)
    p_pos_test_z1 = alpha_test * p_pos_test_z0

    # alpha_test = p_pos_test_z1 / p_pos_test_z0
    alpha_train = p_pos_train_z1 / p_pos_train_z0

    return {
        "p_pos_train_z0": p_pos_train_z0,
        "p_pos_train_z1": p_pos_train_z1,
        "p_pos_train": C_y,
        "p_pos_test": C_y,
        "p_mix_z0": p_mix_z0,
        "p_mix_z1": p_mix_z1,
        "alpha_train": alpha_train,
        "alpha_test": alpha_test,
        "p_pos_test_z0": p_pos_test_z0,
        "p_pos_test_z1": p_pos_test_z1,
        "C_y": C_y,
        "C_z": C_z,
    }

def tune_proportion(df, dfkey, setting, sample, seed):
    uniq_id = df['id'].unique()
    n_target = setting[f'{dfkey}_train'] + setting[f'{dfkey}_test']
    
    # EDGE CASE
    # WARNING: if there is only one id in the subset, reuse training set for validation
    if len(uniq_id) == 1:
        df_train, df_test = train_test_split(df,
                                            test_size=setting[f'{dfkey}_test']/n_target,
                                            shuffle = True, random_state=seed)

    elif len(uniq_id) >= n_target:
        uniq_id = df['id'].drop_duplicates(keep = 'last')
        train_id, test_id = train_test_split(uniq_id,
                                            train_size=setting[f'{dfkey}_train'],
                                            test_size=setting[f'{dfkey}_test'],
                                            shuffle = True, random_state=seed)
        df_train, df_test = df.loc[train_id.index], df.loc[test_id.index]
    elif sample:
        train_id,test_id = train_test_split(uniq_id,
                                            test_size=setting[f'{dfkey}_test']/n_target,
                                            shuffle = True, random_state=seed)
        df_train = df[df.id.isin(train_id)]
        df_test = df[df.id.isin(test_id)]
    else:
        warnings.warn("Set sample equals to True or augment current dataset.")
        return None, None

    if sample:
        if len(df_train) < setting[f'{dfkey}_train']:
            train_extra = df_train.sample(n = setting[f'{dfkey}_train'] - len(df_train), replace = True)
            df_train = pd.concat([df_train,train_extra], axis = 0, ignore_index=True)
        else:
            df_train = df_train.sample(n = setting[f'{dfkey}_train'], replace = False)
        
        if len(df_test) < setting[f'{dfkey}_test']:
            test_extra = df_test.sample(n = setting[f'{dfkey}_test'] - len(df_test), replace = True)
            df_test = pd.concat([df_test,test_extra], axis = 0, ignore_index=True)
        else:
            df_test = df_test.sample(n = setting[f'{dfkey}_test'], replace = False)

        
    return df_train, df_test

def create_mix(df1, df0, target, setting, sample = False, seed = 2023):
    """ Create a mixture dataset from two source based on pre-set constraints"""
    n_total = len(df1) + len(df0)
    # 03/26/2024 change log: if sample, make sure it is sampled after split so one record won't occur in both train and test
    # check if there is enough positive samples in each dataset
    # 10/06/2024 change log: in case one id has multiple records, make sure to separate them in train and test
    df0_pos = df0[df0[target] == 1]
    df0_neg = df0[df0[target] == 0]
    
    df1_pos = df1[df1[target] == 1]
    df1_neg = df1[df1[target] == 0]

    # for z0 positive
    df0_train_pos, df0_test_pos = tune_proportion(df0_pos,'n_z0_pos', setting, sample, seed)
    # for z0 negative
    df0_train_neg, df0_test_neg = tune_proportion(df0_neg, 'n_z0_neg', setting, sample, seed)
    # for z1 positive
    df1_train_pos, df1_test_pos = tune_proportion(df1_pos, 'n_z1_pos', setting, sample, seed)
     # for z1 negative
    df1_train_neg, df1_test_neg = tune_proportion(df1_neg, 'n_z1_neg', setting, sample, seed)

    # assemble mixed train and test
    df_train = pd.concat([df0_train_pos, df0_train_neg, df1_train_pos, df1_train_neg], axis = 0, ignore_index=True)
    df_test = pd.concat([df0_test_pos, df0_test_neg, df1_test_pos, df1_test_neg], axis = 0, ignore_index=True)
    
    # in case there is only one id in the subset, data leakage will guarantee to occur
    if 1 not in [df0_pos.id.nunique(), df0_neg.id.nunique(), df1_pos.id.nunique(), df1_neg.id.nunique()]:
        assert ~df_train['id'].isin(df_test['id']).any(), 'Data Leakage!'
    return {'train':df_train, 'test':df_test, 'setting': setting}



def number_split(p_pos_train_z1,
    p_pos_train_z0,
    p_mix_z1,
    alpha_test,
    train_test_ratio=5,
    n_test = 100, # set the number for tests
    verbose = True
      ):
    """Get required number of samples for each category"""
    assert isinstance(train_test_ratio, int)
    assert isinstance(n_test, int)
    
    mix_param_dict = confoundSplit(
        p_pos_train_z0=p_pos_train_z0,
        p_pos_train_z1=p_pos_train_z1,
        p_mix_z1=p_mix_z1,
        alpha_test=alpha_test,
    )

    if all(0 < mix_param_dict[key] < 1 for key in mix_param_dict.keys() if key not in ['alpha_test','alpha_train']): # assert all probability between 0 and 1

        n_train = n_test * train_test_ratio

        n_z1_train = round(n_train * mix_param_dict["C_z"])
        n_z1_test = round(n_test * mix_param_dict["C_z"])

        n_z0_train = n_train - n_z1_train
        n_z0_test = n_test - n_z1_test

        n_z1_p_train = round(n_z1_train * mix_param_dict["p_pos_train_z1"])
        n_z0_p_train = round(n_z0_train * mix_param_dict["p_pos_train_z0"])

        n_z1_p_test = round(n_z1_test * mix_param_dict['p_pos_test_z1'])
        n_z0_p_test = round(n_z0_test * mix_param_dict['p_pos_test_z0'])


        n_z1_n_train = n_z1_train - n_z1_p_train
        n_z0_n_train = n_z0_train - n_z0_p_train

        n_z1_n_test = n_z1_test - n_z1_p_test
        n_z0_n_test = n_z0_test - n_z0_p_test

        ans = {
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_z0_pos_train": n_z0_p_train,
                    "n_z0_neg_train": n_z0_n_train,
                    "n_z0_pos_test": n_z0_p_test,
                    "n_z0_neg_test": n_z0_n_test,
                    "n_z1_pos_train": n_z1_p_train,
                    "n_z1_neg_train": n_z1_n_train,
                    "n_z1_pos_test": n_z1_p_test,
                    "n_z1_neg_test": n_z1_n_test,
                    "mix_param_dict": mix_param_dict
                }

        if all(ans[key] > 0 for key in ans.keys() if key != 'mix_param_dict'):

            return ans

        elif verbose:
            print("Invalid sample numbers ", [(key, val) for key, val in ans.items() if key != 'mix_param_dict'])
            return None

    elif verbose:
        print(f"Invalid test set probability P(Y=1|Z=0):{mix_param_dict['p_pos_test_z0']}, P(Y=1|Z=1):{mix_param_dict['p_pos_test_z1']}")

    return None


    

def generate_theoretical_settings(p_pos_train_z0, p_pos_train_z1, alpha_test, n_test = 150):

    # THEORETICAL SETTING
    #configure the provenance-specific parameters for train/test splits
    # use n = 600 for pitts corpus
    train_test_ratio = 4
    # n_test = 150 #test set size
    # fraction of both train/test data drawn from domain z1 irrespective of class label
    # fraction drawn from z0 would be 1-p_mix_z1_l[s
    # this is a list of distributions, experiments will be conducted at each of these settings 
    p_mix_z1 = [0.5]

    # alpha_test is p(y=1|z=1) / p(y=1|z=0): the ratio between test set probabilities of the positive class by site/domain
    
    # make sure the input are lists
    if not isinstance(p_pos_train_z0, list):
        p_pos_train_z0 = [p_pos_train_z0]
    if not isinstance(p_pos_train_z1, list):
        p_pos_train_z1 = [p_pos_train_z1]
    if not isinstance(alpha_test, list):
        alpha_test = [alpha_test]


    theoretical_full_settings = []
    for combination in itertools.product(p_pos_train_z0, 
                                        p_pos_train_z1, 
                                        p_mix_z1,
                                        alpha_test
                                        ):


        number_setting = number_split(p_pos_train_z0=combination[0], 
                            p_pos_train_z1 = combination[1], 
                            p_mix_z1 = combination[2], alpha_test = combination[3],
                            train_test_ratio = train_test_ratio, 
                            n_test=n_test,
                            verbose=True #set verbose to True to see which invalid combinations are ignored and why
                                    )

        # # enforce each group*label has at least 10 samples
        if (number_setting is not None):
            if np.all([number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]):
                theoretical_full_settings.append(number_setting)
        # theoretical_full_settings.append(number_setting)

    return theoretical_full_settings


def load_data(data_name = 'pitts'):
    if data_name == 'pitts':
        df_pitts = pd.read_csv("/repo/data/processed_pitts_last.csv")
        gender_map = {'female': 1, 'male': 0} # code female as 1
        df_pitts['gender'] = df_pitts['gender'].replace(gender_map)
        # split original dataset into two
        df_pitts_male = df_pitts.query("gender==0").reset_index(drop=True)
        df_pitts_female = df_pitts.query("gender==1").reset_index(drop=True)
        
        return df_pitts_male, df_pitts_female

    
    elif data_name == 'wls':
        df_wls = pd.read_csv("/repo/data/processed_wls.csv", index_col = 0)

        # split original dataset into two
        df_wls_male = df_wls[df_wls['gender']==0]
        df_wls_female = df_wls[df_wls['gender']==1]
    
        return df_wls_male, df_wls_female
    
    elif data_name == 'ccc':
        df_ccc = pd.read_csv("/repo/data/processed_ccc.csv", index_col = 0)

        # split original dataset into two
        df_ccc_male = df_ccc[df_ccc['gender']==0]
        df_ccc_female = df_ccc[df_ccc['gender']==1]

        return df_ccc_male, df_ccc_female

def make_figures_cf(rframe, save_label, nr = 3, nc = 4):
    fig, axes = plt.subplots(nc, nr, figsize=(18, 20))
    all_alpha_train =  np.unique(rframe['alpha_train'])
    all_alpha_train = sorted(all_alpha_train[all_alpha_train>=1])
    i,j = 0, 0
    for alpha_train in all_alpha_train:
        reciprocal = round(1 / alpha_train, 2)
        sns.scatterplot(data=rframe[(rframe['alpha_train'] == alpha_train) | (rframe['alpha_train'] == reciprocal)],x='alpha_test',y='roc', hue='alpha_train', style='C_y', ax=axes[i,j])#, x_estimator=np.mean)
        axes[i,j].set_xscale('log')
        axes[i,j].axvline(alpha_train,color='black', linestyle = '--')
        axes[i,j].axvline(reciprocal, color='mistyrose', linestyle = '--')
        axes[i,j].set_title('alpha_train: ' + str(alpha_train))
        # make y axis consistent
        # axes[i,j].set_ylim(np.min(rframe['roc']), np.max(rframe['roc']))
        j+=1
        if j == nc-1:
            j = 0
            i +=1
            
    plt.tight_layout()
    plt.savefig(f'output/test-{save_label}_pitts.pdf', format="pdf", bbox_inches="tight")        
    plt.show()

def binarize_tensor(change_mat, ratio):
    global_changes = torch.cat([t.view(-1) for t in change_mat.values()]).cpu().numpy()
    quantile_value = np.quantile(global_changes, 1 - ratio)
    # param to mask is 1, param to keep is 0
    binarized_mat = {k:torch.where(v > quantile_value, 1.0, 0.0) for k,v in change_mat.items()}
    return binarized_mat

def get_mask(change_mat1, change_mat2, type, alpha):
    """change_mat1 is dementia weight change, change_mat2 is gender weight change"""
    assert type in ['intersection', 'compliment','all'], 'Invalid set type'
    # mask list of length of tracked layers
    mask_dict = {}
    # Iterate over both lists simultaneously
    for (k1, tensor1), (k2, tensor2) in zip(change_mat1.items(), change_mat2.items()):
        assert k1 == k2, 'not same layer'

        if type == 'intersection': # Find positions where both are significantly changed
            mask = (tensor1 == 1) & (tensor2 == 1)
        elif type == 'compliment': # Find positions where only confounder model is significantly changed but not primary model
            mask = (tensor1 == 0) & (tensor2 == 1)
        elif type == 'all': # Find positions where confounder model is significantly changed
            mask =  (tensor2 == 1)       
        # Append the mask to the list
        assert mask.size() == tensor1.size() and  tensor1.size() == tensor2.size(), 'tensor or mask are not the same shape'     
        # convert 1 to 0 for matrix multiplication
        mask_dict[k1] = ~mask
        # add alpha effects
        mask_dict[k1][mask_dict[k1]== 0.] = alpha
    return mask_dict

def get_masks(changes_mat1, changes_mat2, ratio, type, alpha):
    """ find global cutoff points """
    change1_bi = binarize_tensor(changes_mat1, ratio = ratio)
    change2_bi = binarize_tensor(changes_mat2, ratio = ratio)
    weight_mask_dict = get_mask(change1_bi, change2_bi, type = type, alpha = alpha)
    
    return weight_mask_dict

def calculate_cosine_similarity(change_mats1, change_mats2):
    similarities = []
    for mat1, mat2 in zip(change_mats1, change_mats2):
        # Flatten the tensors to make them vectors
        vector_a = mat1.flatten()
        vector_b = mat2.flatten()
        # Calculate cosine similarity
        similarity = np.dot(vector_a, vector_b)/(np.linalg.norm(vector_a)*np.linalg.norm(vector_b))       
        similarities.append(similarity)  # Convert to Python scalar and store
    
    return similarities



def iid_split(df, pz0, pz1, target, data_seed):
    # get a evaluation dataset identical to training set
    #df = df.drop_duplicates(subset='id')
    setts = generate_theoretical_settings(p_pos_train_z0 = [pz0], p_pos_train_z1 = [pz1], 
                                              alpha_test=[pz1/pz0], n_test = 120)
    # choose 1 setting
    sett = np.random.choice(setts)
    df1 = df[df['gender']==1]
    df0 = df[df['gender']==0]
    dfs = create_mix(df0=df0, df1=df1, target=target, setting=sett, sample=True, 
                         seed=data_seed)
    train_df = dfs['train']
    eval_df = dfs['test']
    return train_df, eval_df
