import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

import sys

sys.path.extend(['./', '../'])
from dataset.BioFMSeries import create_bioflowmatching_datamodule_from_bio
from dataset.BioSeries import MultiBioSeriesDataset, MultiBioSeriesDataset_Subset

def get_train_val_test_splits_ood(data_dir, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42, 
                              load_name: str='SysBio-Traj_index.csv', armd_train: bool=False, **kwargs):
    """
    Splits the dataset files in data_dir into training, validation, and test sets.

    Args:
        data_dir (str): Directory containing dataset files.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
    Returns:
        train_list (list): List of file paths for training data.
        val_list (list): List of file paths for validation data.
        test_list (list): List of file paths for testing data.
    """
    # List all dataset files in the directory
    system_info_path = os.path.join(data_dir, load_name)
    system_info = pd.read_csv(system_info_path)
    all_files = [
        os.path.join(data_dir, 'Data', row['model_id'], row['model_name'] + '.csv')
        for _, row in system_info.iterrows()
    ]

    # First split off the test set
    train_val_files, test_list = train_test_split(
        all_files, test_size=test_ratio, random_state=seed
    )

    # Then split the remaining data into training and validation sets
    val_size_adjusted = val_ratio / (1 - test_ratio)  # Adjust val size relative to remaining data
    train_list, val_list = train_test_split(
        train_val_files, test_size=val_size_adjusted, random_state=seed
    )

    print(f"Training set size: {len(train_list)}")
    print(f"Validation set size: {len(val_list)}")
    print(f"Test set size: {len(test_list)}")

    print('Building training datasets...')
    if armd_train: train_dataset = MultiBioSeriesDataset(dataset_dirs=train_list, 
                                                         input_window=kwargs.get('input_window'), 
                                                         output_window=kwargs.get('input_window'), # fixed output window (similar with input)
                                                         stride=kwargs.get('stride'),
                                                         variable_independent=kwargs.get('variable_independent'),
                                                         transform_type=kwargs.get('transform_type'),
                                                         condition_names=kwargs.get('condition_names'))
    else: train_dataset = MultiBioSeriesDataset(dataset_dirs=train_list, **kwargs)
    print('Building validation datasets...')
    val_dataset = MultiBioSeriesDataset(dataset_dirs=val_list, **kwargs)
    print('Building test datasets...')
    test_dataset = MultiBioSeriesDataset(dataset_dirs=test_list, **kwargs)

    return train_dataset, val_dataset, test_dataset

def get_train_val_test_splits(data_dir, 
                              val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42, 
                              load_name: str='SysBio-Traj_index.csv', **kwargs):
    """
    Splits the dataset files in data_dir into training, validation, and test sets.

    Args:
        data_dir (str): Directory containing dataset files.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
    """
    # List all dataset files in the directory
    system_info_path = os.path.join(data_dir, load_name)
    system_info = pd.read_csv(system_info_path)
    all_files = [
        os.path.join(data_dir, 'Data', row['model_id'], row['model_name'] + '.csv')
        for _, row in system_info.iterrows()
    ]
    all_dataset = MultiBioSeriesDataset(dataset_dirs=all_files, **kwargs)
    all_list = list(range(len(all_dataset)))
    # First split off the test set
    train_val_indices, test_indices = train_test_split(
        all_list, test_size=test_ratio, random_state=seed
    )
    # Then split the remaining data into training and validation sets
    val_size_adjusted = val_ratio / (1 - test_ratio)  # Adjust val size relative to remaining data
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size_adjusted, random_state=seed
    )

    train_dataset = MultiBioSeriesDataset_Subset(all_dataset, train_indices)
    val_dataset = MultiBioSeriesDataset_Subset(all_dataset, val_indices)
    test_dataset = MultiBioSeriesDataset_Subset(all_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
