import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle
import torch
import torch.nn as nn

color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
label_dict = {
    0: 'FF-Real', 1: 'FF-Fake', 2: 'CDF-Real', 3: 'CDF-Fake', 4: 'FF-Fsh'
}




import argparse
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument("--test_dataset", nargs="+", default=['FF-DF'])
parser.add_argument("--model_name", nargs="+")
args = parser.parse_args()



def plot_explained_variance(explained_variance_ratio, cumulative_explained_variance_ratio, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(explained_variance_ratio[:200], marker='o', alpha=0.4, markersize=12)
    plt.title('Explained Variance Ratio of Principal Components', fontsize=24, weight='bold', y=1.05)
    plt.xlabel('Principal Component Index', fontsize=24, weight='bold')
    plt.ylabel('Explained Variance Ratio', fontsize=24, weight='bold')
    plt.tick_params(axis='both', labelsize=20)  # 或者设置成24，如果希望和轴标签一样大
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'explained_variance_ratio.png'), dpi=300)

    plt.figure(figsize=(10, 8))
    plt.plot(cumulative_explained_variance_ratio[:200], marker='o', alpha=0.4, markersize=12)
    plt.title('Cumulative Explained Variance Ratio', fontsize=24, weight='bold', y=1.05)
    plt.xlabel('Number of Principal Components', fontsize=24, weight='bold')
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=24, weight='bold')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'cumulative_explained_variance_ratio.png'), dpi=300)

def main():
    base_dir = './zhiyuanyan/DeepfakeBenchv2/features'
    model_name = args.model_name[0]
    base_dir = os.path.join(base_dir, model_name)
    test_dataset = args.test_dataset
    all_testing_data = []
    all_testing_label = []
    pool = nn.AdaptiveAvgPool2d((1, 1))

    for name in os.listdir(base_dir):
        if name in test_dataset:
            with open(os.path.join(base_dir, name, 'tsne.pkl'), 'rb') as f:
                data = pickle.load(f)
                feat = data['feat']
                print(f'shape of {name}:', feat.shape)
                if feat.ndim == 4: # pool
                    print(f'shape of {name} before pooling:', feat.shape)
                    feat = torch.from_numpy(feat)
                    feat = pool(feat).squeeze()
                    feat = feat.numpy()
                    print(f'shape of {name} after pooling:', feat.shape)
                label = data['label']
                if name == 'Celeb-DF-v2':
                    label = label + 2
                elif name == 'FaceShifter':
                    label_mask = (label == 1)
                    feat = feat[label_mask]
                    label = label[label_mask] * 4
                all_testing_data.append(feat)
                all_testing_label.append(label)
    all_testing_data = np.concatenate(all_testing_data, axis=0)
    all_testing_label = np.concatenate(all_testing_label, axis=0)
    print('Total number of samples:', len(all_testing_label))
    print('Label distribution:', np.unique(all_testing_label, return_counts=True))

    # Perform PCA analysis
    feat = all_testing_data
    pca = PCA(n_components=min(feat.shape[0], feat.shape[1]))
    pca.fit(feat)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    # Plot explained variance ratio and cumulative explained variance ratio
    save_path = './zhiyuanyan/DeepfakeBenchv2/pca_results/' + model_name
    os.makedirs(save_path, exist_ok=True)
    plot_explained_variance(explained_variance_ratio, cumulative_explained_variance_ratio, save_path)

    # Output the number of principal components needed to explain 90% variance
    num_components_90 = np.argmax(cumulative_explained_variance_ratio >= 0.9) + 1
    print(f'Number of principal components explaining 90% variance: {num_components_90}')
    print(f'Feature dimension: {feat.shape[1]}')

    # 计算特征值衰减系数
    eigenvalues = pca.explained_variance_
    decay_rate = np.diff(np.log(eigenvalues)) / np.diff(np.arange(len(eigenvalues)))
    print("平均衰减率:", np.mean(decay_rate[:100]))

    # Optional: Plot scatter plot of the first two principal components
    feat_transformed = pca.transform(feat)[:, :2]
    numerical_labels = all_testing_label
    labels = [label_dict[label] for label in numerical_labels]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=feat_transformed[:, 0],
        y=feat_transformed[:, 1],
        hue=labels,
        palette=color_map[:len(np.unique(numerical_labels))],
        alpha=0.4
    )
    plt.title('Scatter Plot of the First Two Principal Components', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=20)
    plt.ylabel('Principal Component 2', fontsize=20)
    plt.legend(title='Classes', loc='best')
    plt.savefig(os.path.join(save_path, 'pca_scatter.png'), dpi=300)

if __name__ == '__main__':
    main()