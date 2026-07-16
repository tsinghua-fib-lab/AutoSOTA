import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
import pyarrow as pa
import pyarrow.ipc

# --- 1. Configuration Parameters ---

# Root directory of the dataset
DATA_ROOT_PATH = './LowLat'

# Output directory for Arrow files
OUTPUT_PATH = './LowLat_arrow'

# Feature columns to be extracted
FEATURE_COLUMNS = ['TMP', 'DEW', 'WND_ANGLE', 'WND_RATE', 'SLP']

# Length of the target time series sequence
TARGET_LENGTH = 4096

# Shape of the target data [Features, TimeSteps]
TARGET_SHAPE = [len(FEATURE_COLUMNS), TARGET_LENGTH]

# --- Sampling Configuration ---

# Sliding Window Stride (Step Size) for each split.
# - If stride < TARGET_LENGTH: Windows overlap (Data Augmentation).
# - If stride == TARGET_LENGTH: Non-overlapping windows.
# - If stride > TARGET_LENGTH: Downsampling.
TRAIN_STRIDE = 1 
VAL_STRIDE = 4096        
TEST_STRIDE = 1024 

# Subset Ratios (Keys are folder suffixes, Values are ratios [0.0 - 1.0])
# Example: {'10percent': 0.1} generates a 'train_10percent' folder with 10% of files.
# Set to None or empty dict {} if no subsets are needed.
TRAIN_SUBSET_RATIOS = {'10percent': 0.1} 
TEST_SUBSET_RATIOS = {'small': 0.05}

# Number of parallel workers
NUM_WORKERS = cpu_count()


def process_file(file_path, feature_columns):
    """
    Process a single CSV file: load data and partition by year.
    
    Args:
        file_path (str): Path to the CSV file.
        feature_columns (list): List of column names to extract.
        
    Returns:
        tuple: ((train_data, train_dates), (val_data, val_dates), (test_data, test_dates))
    """
    try:
        df = pd.read_csv(file_path)
        # Select relevant columns
        df = df[['DATE'] + feature_columns]
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Temporal Partitioning
        train_df = df[(df['DATE'].dt.year >= 2020) & (df['DATE'].dt.year <= 2021)]
        val_df = df[df['DATE'].dt.year == 2022]
        test_df = df[df['DATE'].dt.year == 2023]
        
        # Extract values and timestamps
        def extract_numpy(dataframe):
            if not dataframe.empty:
                return dataframe[feature_columns].values, dataframe['DATE'].values
            return None, None

        train_data = extract_numpy(train_df)
        val_data = extract_numpy(val_df)
        test_data = extract_numpy(test_df)
        
        return train_data, val_data, test_data
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return (None, None), (None, None), (None, None)


def save_arrow_file(file_path, start_date, target_data):
    """
    Serialize a single time-series sample into an Arrow file.
    
    Args:
        file_path (str): Destination path.
        start_date (np.datetime64): Timestamp of the first step in the sequence.
        target_data (np.ndarray): Data array of shape [TimeSteps, Features].
    """
    # Convert numpy datetime to string format
    start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
    
    # Transpose data to match [Features, TimeSteps] and flatten
    # Original: [L, C] -> Transpose: [C, L] -> Flatten
    flattened_target = target_data.T.flatten()

    # Create PyArrow Table
    data_dict = {
        'start': [start_str],
        'target._np_shape': [TARGET_SHAPE],
        'target': [flattened_target]
    }
    table = pa.Table.from_pydict(data_dict)
    
    # Write to disk
    with pa.ipc.new_file(file_path, table.schema) as writer:
        writer.write_table(table)


def generate_dataset_split(data_list, output_base_path, split_name, stride, subset_ratio=None):
    """
    Generate Arrow files for a specific dataset split (e.g., train, val, test_small).
    
    Args:
        data_list (list): List of tuples (data_array, date_array).
        output_base_path (str): Root directory for output.
        split_name (str): Name of the subdirectory (e.g., 'train', 'test_small').
        stride (int): Sliding window step size.
        subset_ratio (float, optional): Ratio of files to use. If None, use all files.
    """
    if not data_list:
        print(f"Warning: No data available for split '{split_name}'. Skipping.")
        return

    # Handle Subsetting
    selected_data_list = data_list
    if subset_ratio is not None:
        num_files = len(data_list)
        # Shuffle indices to ensure random selection
        shuffled_indices = np.random.permutation(num_files)
        num_to_select = int(num_files * subset_ratio)
        if num_to_select < 1 and subset_ratio > 0:
            num_to_select = 1
        
        selected_indices = shuffled_indices[:num_to_select]
        selected_data_list = [data_list[i] for i in selected_indices]
        print(f"\n--- Generating Split: {split_name} (Subset: {subset_ratio*100:.1f}%, Files: {len(selected_data_list)}) ---")
    else:
        print(f"\n--- Generating Split: {split_name} (Full Dataset, Files: {len(selected_data_list)}) ---")

    # Create output directory
    split_output_dir = os.path.join(output_base_path, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    sample_counter = 1
    
    # Iterate over files
    for data, dates in tqdm(selected_data_list, desc=f"Processing {split_name}"):
        total_timesteps = len(data)
        
        # Sliding Window Generation
        # Range ensures we do not go out of bounds
        for i in range(0, total_timesteps - TARGET_LENGTH + 1, stride):
            data_chunk = data[i : i + TARGET_LENGTH]
            start_date = dates[i]
            
            output_filename = os.path.join(split_output_dir, f"{sample_counter}.arrow")
            save_arrow_file(output_filename, start_date, data_chunk)
            sample_counter += 1
            
    print(f"Completed: Saved {sample_counter - 1} samples to '{split_output_dir}'.")


def main():
    """
    Main execution pipeline.
    """
    # Ensure reproducibility for shuffling
    np.random.seed(42)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(DATA_ROOT_PATH, '*.csv'))
    if not csv_files:
        print(f"Error: No .csv files found in directory '{DATA_ROOT_PATH}'.")
        return

    print(f"Detected {len(csv_files)} weather station CSV files.")
    print(f"Initiating parallel data loading with {NUM_WORKERS} workers...")

    # --- 2. Parallel Data Loading and Splitting ---
    train_data_list = []
    val_data_list = []
    test_data_list = []
    
    with Pool(processes=NUM_WORKERS) as pool:
        task = functools.partial(process_file, feature_columns=FEATURE_COLUMNS)
        # imap_unordered is faster as it yields results as soon as they are ready
        results = list(tqdm(pool.imap_unordered(task, csv_files), total=len(csv_files), desc="Loading & Partitioning Data"))

    # Aggregate results
    for train_res, val_res, test_res in results:
        # Only add data if it is long enough for at least one sequence
        if train_res[0] is not None and len(train_res[0]) >= TARGET_LENGTH:
            train_data_list.append(train_res)
        if val_res[0] is not None and len(val_res[0]) >= TARGET_LENGTH:
            val_data_list.append(val_res)
        if test_res[0] is not None and len(test_res[0]) >= TARGET_LENGTH:
            test_data_list.append(test_res)

    print("Data loading complete. Proceeding to Arrow generation.")

    # --- 3. Generate Full Datasets ---
    
    # 3.1 Generate Full Training Set
    generate_dataset_split(
        train_data_list, 
        OUTPUT_PATH, 
        split_name='train', 
        stride=TRAIN_STRIDE, 
        subset_ratio=None
    )

    # 3.2 Generate Full Validation Set
    generate_dataset_split(
        val_data_list, 
        OUTPUT_PATH, 
        split_name='val', 
        stride=VAL_STRIDE, 
        subset_ratio=None
    )

    # 3.3 Generate Full Test Set
    generate_dataset_split(
        test_data_list, 
        OUTPUT_PATH, 
        split_name='test', 
        stride=TEST_STRIDE, 
        subset_ratio=None
    )

    # --- 4. Generate Subset Datasets (Optional) ---

    # 4.1 Train Subsets (e.g., for Few-shot learning)
    if TRAIN_SUBSET_RATIOS:
        for suffix, ratio in TRAIN_SUBSET_RATIOS.items():
            generate_dataset_split(
                train_data_list,
                OUTPUT_PATH,
                split_name=f'train_{suffix}',
                stride=TRAIN_STRIDE,
                subset_ratio=ratio
            )

    # 4.2 Test Subsets (e.g., for quick debugging)
    if TEST_SUBSET_RATIOS:
        for suffix, ratio in TEST_SUBSET_RATIOS.items():
            generate_dataset_split(
                test_data_list,
                OUTPUT_PATH,
                split_name=f'test_{suffix}',
                stride=TEST_STRIDE,
                subset_ratio=ratio
            )

    print("\nAll data processing tasks completed successfully.")


if __name__ == '__main__':
    main()