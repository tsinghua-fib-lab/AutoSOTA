import torch
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def jacard_similarity(mat1, mat2):
    intersection = np.logical_and(mat1, mat2)
    union = np.logical_or(mat1, mat2)
    return intersection.sum() / float(union.sum())
def binarize_mat(change_mat, ratio):
    global_changes = torch.cat([t.view(-1) for t in change_mat.values()]).cpu().numpy()
    quantile_value = np.quantile(global_changes, 1 - ratio)
    # param to mask is 1, param to keep is 0
    binarized_mat = {k:torch.where(v > quantile_value, 1.0, 0.0) for k,v in change_mat.items()}
    return binarized_mat

def make_jaccard(alpha_train, path = '/home/sheng136/DeconDTN/code/output/change_mats/'):
    
    with open(os.path.join(path,f"global_atrain-{alpha_train:.2f}.pkl"), 'rb') as f:
        res = pickle.load(f)
    changesd = res['repeat4']['dementia_change']
    changesg = res['repeat4']['gender_change']

    jaccrad_index = []
    changesd = binarize_mat(changesd, 0.15)
    changesg = binarize_mat(changesg, 0.15)
    for key in changesd.keys():
        if key not in ['bert.embeddings.word_embeddings.weight']:
            weight_d = changesd[key]
            weight_g = changesg[key]
            layer = int(key.split('.')[3])+1
            weight_type = key.split('.')[4:-1]
            weight_type = '.'.join(weight_type)
            jaccard = jacard_similarity(weight_d, weight_g).item()
            jaccrad_index.append({'layer': layer, 'type': weight_type, 'jaccard': jaccard})
            df = pd.DataFrame(jaccrad_index)
    plt.figure(figsize=(20, 8))
    # Plot the data
    ax = sns.barplot(x='layer', y='jaccard', hue='type', data=df, palette='viridis', alpha = 0.90)
    xmin, xmax = ax.get_xlim()
    # pad bars on both sides a bit and draw the grid behind the bars
    ax.set(xlim=(xmin-0.25, xmax+0.25), axisbelow=True);
    plt.title('')
    plt.xlabel('')
    plt.ylabel('Jaccard Index', fontdict={'fontsize': 18})
    plt.xticks(rotation=0, fontsize=15)
    plt.legend(title='', loc='upper right', fontsize=17)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(f'/home/sheng136/DeconDTN/code/misc/figs/jaccard/jaccard_atrain-{alpha_train:.2f}.pdf', 
                format='pdf', bbox_inches='tight', dpi=300)


def make_counts(alpha_train, path = '/home/sheng136/DeconDTN/code/output/change_mats/'):
        
    with open(os.path.join(path,f"global_atrain-{alpha_train:.2f}.pkl"), 'rb') as f:
        res = pickle.load(f)
    changesd = res['repeat4']['dementia_change']
    changesg = res['repeat4']['gender_change']
    counts = []
    changesd = binarize_mat(changesd, 0.03)
    changesg = binarize_mat(changesg, 0.03)
    for key in changesd.keys():
        if key not in ['bert.embeddings.word_embeddings.weight']:
            weight_d = changesd[key]
            weight_g = changesg[key]
            layer = int(key.split('.')[3])+1
            mask_sum_intersection = torch.sum(weight_d * weight_g).item()
            mask_sum_difference = torch.sum(weight_g * (1 - weight_d)).item()
            counts.append({'layer': layer, 'type': 'intersection', 'count': mask_sum_intersection})
            counts.append({'layer': layer, 'type': 'difference', 'count': mask_sum_difference})
    df = pd.DataFrame(counts)
    plt.figure(figsize=(20, 8))
    # Plot the data
    ax = sns.barplot(x='layer', y='count', hue='type', data=df, palette='Set2', alpha = 0.90)
    xmin, xmax = ax.get_xlim()
    # pad bars on both sides a bit and draw the grid behind the bars
    ax.set(xlim=(xmin-0.25, xmax+0.25), axisbelow=True);
    plt.title('')
    plt.xlabel('')
    plt.ylabel('# of Masks', fontdict={'fontsize': 20})
    plt.xticks(rotation=0, fontsize=20)
    plt.legend(title='', loc='upper right', prop={'size' : 25})
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(f'/home/sheng136/DeconDTN/code/misc/figs/num_masks/nmasks_atrain-{alpha_train:.2f}.pdf', 
                format='pdf', bbox_inches='tight', dpi=300)


def main():
    # alpha_trains = [0.20, 0.25, 0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
    # for alpha_train in alpha_trains:
    #     make_jaccard(alpha_train)

    #alpha_trains = [0.20, 0.33, 1.00, 3.00, 5.00]
    alpha_trains = [0.33]
    for alpha_train in alpha_trains:
        #make_jaccard(alpha_train)
        make_counts(alpha_train)
if __name__ == '__main__':
    main()