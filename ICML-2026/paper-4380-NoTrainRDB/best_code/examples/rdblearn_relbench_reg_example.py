import pandas as pd
from sklearn.metrics import mean_absolute_error
from tabpfn import TabPFNRegressor

from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnRegressor
from rdblearn.constants import TABPFN_DEFAULT_CONFIG

from loguru import logger
logger.enable("rdblearn")

def main():
    # 1. Load Dataset
    print("Loading 'rel-avito' dataset...")
    dataset = RDBDataset.from_relbench("rel-avito")
    
    # Select the 'ad-ctr' task
    task_name = "ad-ctr"
    if task_name not in dataset.tasks:
        raise ValueError(f"Task '{task_name}' not found in dataset. Available tasks: {list(dataset.tasks.keys())}")
    
    task = dataset.tasks[task_name]
    print(f"Loaded task: {task.name}")
    print(f"Train shape: {task.train_df.shape}")
    print(f"Test shape: {task.test_df.shape}")

    # 2. Initialize Model
    print("Initializing TabPFNRegressor...")
    # Note: You might need to adjust 'device' in TABPFN_DEFAULT_CONFIG if you don't have a GPU
    base_model = TabPFNRegressor(**TABPFN_DEFAULT_CONFIG)

    # Configure RDBLearn
    reg = RDBLearnRegressor(
        base_estimator=base_model,
    )

    # 3. Train
    print("Training model...")
    # Separate features and target
    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]

    reg.fit(
        X=X_train, 
        y=y_train, 
        rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col
    )
    print("Training complete.")

    # 4. Predict
    print("Predicting on test set...")
    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col]

    # Predict with output_type="median"
    y_pred = reg.predict(X=X_test, output_type="median")
    
    # 5. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
