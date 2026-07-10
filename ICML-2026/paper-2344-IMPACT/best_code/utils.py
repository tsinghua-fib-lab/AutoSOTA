import os
import numpy as np

def get_data(dataset, data_root, setting='general', anomaly_class_idx=0):
    if setting == 'general':
        train_path = os.path.join(data_root, dataset, f'{dataset}_train_{setting}.npz')
    else:
        train_path = os.path.join(data_root, dataset, f'{dataset}_train_{setting}_seen{anomaly_class_idx}.npz')
    test_path = os.path.join(data_root, dataset, f'{dataset}_test.npz')
    train_df = np.load(train_path)
    test_df = np.load(test_path)
    train_data, train_label = train_df['data'], train_df['label']
    test_data, test_label = test_df['data'], test_df['label']
    return train_data, train_label, test_data, test_label