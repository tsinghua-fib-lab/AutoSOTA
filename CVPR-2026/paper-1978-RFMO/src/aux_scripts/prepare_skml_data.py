import os
import numpy as np
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split


DATASETS = [
    'scene',  # image n_samples: 2407, n_features: 294, n_labels: 6
    'yeast',  # biology n_samples: 2417, n_features: 103, n_labels: 14
    'enron',  # text n_samples: 1702, n_features: 1001, n_labels: 53
    'mediamill',  # video n_samples: 43907, n_features: 120, n_labels: 101
    'medical', # text n_samples: 978, n_features: 1449, n_labels: 45
    'bibtex',  # text n_samples: 7395, n_features: 1836, n_labels: 159
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '../data')


def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for dataset_name in DATASETS:
        train_X, train_y, _, _ = load_dataset(dataset_name, 'train')
        test_X, test_y, _, _ = load_dataset(dataset_name, 'test')

        try:
            val_X, val_y, _, _ = load_dataset(dataset_name, 'val')
        except ValueError:
            print(f'No validation set for {dataset_name}, split train set instead')
            train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)

        np.savez_compressed(
            os.path.join(DATA_DIR, f'{dataset_name}_train.npz'),
            X=train_X.toarray(),
            y=train_y.toarray(),
        )
        np.savez_compressed(
            os.path.join(DATA_DIR, f'{dataset_name}_val.npz'),
            X=val_X.toarray(),
            y=val_y.toarray(),
        )
        np.savez_compressed(
            os.path.join(DATA_DIR, f'{dataset_name}_test.npz'),
            X=test_X.toarray(),
            y=test_y.toarray(),
        )

if __name__ == '__main__':
    prepare_data()
