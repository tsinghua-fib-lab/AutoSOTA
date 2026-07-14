import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.constants import TABPFN_DEFAULT_CONFIG

from loguru import logger
logger.enable("rdblearn")

def main():
    # 1. Load Dataset
    # This uses the DBInferAdapter internally to load the RDB and tasks
    print("Loading 'diginetica' dataset from 4DBInfer...")
    dataset = RDBDataset.from_4dbinfer("diginetica")
    
    # Select the 'ctr' task
    task_name = "ctr"
    if task_name not in dataset.tasks:
        raise ValueError(f"Task '{task_name}' not found in dataset. Available tasks: {list(dataset.tasks.keys())}")
    
    task = dataset.tasks[task_name]
    print(f"Loaded task: {task.name}")
    print(f"Train shape: {task.train_df.shape}")
    print(f"Test shape: {task.test_df.shape}")

    # 2. Initialize Model
    # Initialize the base estimator (TabPFN) with default config
    print("Initializing TabPFNClassifier...")
    # Note: You might need to adjust 'device' in TABPFN_DEFAULT_CONFIG if you don't have a GPU
    base_model = TabPFNClassifier(**TABPFN_DEFAULT_CONFIG)

    # Configure RDBLearn
    # Use DFS depth 1 for testing
    clf = RDBLearnClassifier(
        base_estimator=base_model
    )

    # 3. Train
    print("Training model...")
    # Separate features and target
    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]

    clf.fit(
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

    # Predict probabilities for AUC
    y_pred_proba = clf.predict_proba(X=X_test)
    
    # 5. Evaluate
    # TabPFN returns probabilities for all classes. For binary classification, we usually take the probability of the positive class (index 1).
    if y_pred_proba.shape[1] == 2:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba[:, 1] # Assuming index 1 is the positive class

    auc = roc_auc_score(y_test, y_scores)
    print(f"Test AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
