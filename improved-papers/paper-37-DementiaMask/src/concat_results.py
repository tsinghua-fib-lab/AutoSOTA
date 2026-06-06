# Author: Zhecheng Sheng
# load all local filtering results to a dataframe for plot


import os
import glob
import re
import pandas as pd
import pickle
import argparse


# TODO: remove redundancy when loading the same object for different metric
def load_metric(alpha_train = 4.0, met = 'accuracy', directory = ''):
    # Define the directory and pattern
    pattern = 'c_.*-(.*?)_.*_.*'  # Adjust the extension if necessary
    # List all matching files
    file_paths = glob.glob(os.path.join(directory, f'c_{alpha_train}-*'))  # Adjust the pattern as needed
    # Initialize a set for unique Cy values
    unique_Cy = set()
    # Compile the regex pattern
    regex_pattern = re.compile(pattern)
    # Extract unique Cy values
    for file_path in file_paths:
        match = regex_pattern.match(os.path.basename(file_path))
        if match:
            Cy_value = match.group(1)
            unique_Cy.add(Cy_value)
    out = {}
    for Cy in unique_Cy:
        after_filter_layers = []
        for n in range(-1,11):
            after_layer_runs = []
            if n == -1:
                with open(os.path.join(directory,f'c_{alpha_train}-{Cy}_-1_1'), 'rb') as f:
                    emb_results = pickle.load(f)
                for i in range(len(emb_results['after_filter'])):
                    acc = emb_results['after_filter'][i].metrics[f'test_{met}']
                    after_layer_runs.append(acc)
                after_filter_layers.append(after_layer_runs)
            after_layer_runs = []    
            with open(os.path.join(directory,f'c_{alpha_train}-{Cy}_{n}_0'), 'rb') as f:
                results = pickle.load(f)
                for i in range(len(emb_results['after_filter'])):
                    acc = results['after_filter'][i].metrics[f'test_{met}']
                    after_layer_runs.append(acc)
                after_filter_layers.append(after_layer_runs)
        before_layers = []
        for i in range(len(emb_results['before_filter'])):
            before_layers.append(results['before_filter'][i].metrics[f'test_{met}'])
        after_filter_layers.append(before_layers)
        out[str(Cy)] = after_filter_layers

    dat = pd.DataFrame(out)
    dat  = pd.melt(dat, var_name='Cy', value_name=f'{met}')
    layer_names = ['Emb']
    layer_names.extend(['Layer '+str(n+1) for n in range(12)])
    layer_names.append('Intact')
    layer_names = layer_names*len(unique_Cy)
    final_df = dat.assign(layer = layer_names).explode(f'{met}').assign(alpha_train=f'{alpha_train}')
    final_df['repeats'] = final_df.groupby(['alpha_train','layer', 'Cy']).cumcount() + 1
    return final_df

def main():
    train_alphas = set([0.2, 0.25, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    parser = argparse.ArgumentParser(description='Train a binary classification model for dementia')
    parser.add_argument('-out', '--output_dir', type=str, required=True)
    args = parser.parse_args()
    outdir = args.output_dir

    met = 'aps'
    for met in ['accuracy','aps','roc','f1']:    
        all_results = []
        for ta in train_alphas:
            res = load_metric(alpha_train=ta, met = met, directory=os.path.join(outdir,'healthpts_results/changes'))
            all_results.append(res)
        with open(os.path.join(outdir,'joint_results_{met}.pkl'), 'wb') as f:
            pickle.dump(pd.concat(all_results, axis = 0), f)


if __name__ == '__main__':
    main()