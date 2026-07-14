import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download

from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.utils import LimiXWrapperClassifier

from loguru import logger
logger.enable("rdblearn")

# --- LimiX Setup ---
# Assume user provides the path to the cloned LimiX repository
# e.g. export LIMIX_ROOT=/path/to/LimiX
LIMIX_ROOT = os.environ.get("LIMIX_ROOT")

if not LIMIX_ROOT or not os.path.exists(LIMIX_ROOT):
    print("Please set the LIMIX_ROOT environment variable to the path of your LimiX repository.")
    print("Example: export LIMIX_ROOT=/root/projects/LimiX")
    print("The repository can be found at: https://github.com/limix-ldm/LimiX")
    sys.exit(1)

if LIMIX_ROOT not in sys.path:
    sys.path.insert(0, LIMIX_ROOT)

try:
    from inference.predictor import LimiXPredictor
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to import LimiXPredictor from {LIMIX_ROOT}.")
    print("Please ensure LIMIX_ROOT points to the valid root of the LimiX repo.")
    sys.exit(1)

def main():
    # 1. Load Dataset
    print("Loading 'rel-f1' dataset...")
    dataset = RDBDataset.from_relbench("rel-f1")
    
    # Select the 'driver-dnf' task
    task_name = "driver-dnf"
    if task_name not in dataset.tasks:
        raise ValueError(f"Task '{task_name}' not found. Available: {list(dataset.tasks.keys())}")
    
    task = dataset.tasks[task_name]
    print(f"Loaded task: {task.name}")
    print(f"Train shape: {task.train_df.shape}")
    print(f"Test shape: {task.test_df.shape}")

    # 2. Check for GPU (LimiX usually requires CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Download/Locate Model Weights
    print("Locating LimiX model weights...")
    try:
        model_file = hf_hub_download(
            repo_id="stableai-org/LimiX-16M", 
            filename="LimiX-16M.ckpt", 
            local_dir="./limix_cache"
        )
        print(f"Model found at: {model_file}")
    except Exception as e:
        print(f"Failed to download model weights: {e}")
        sys.exit(1)

    # 4. Initialize LimiX Predictor
    # Note: config path should be relative to LIMIX_ROOT or absolute
    config_path = os.path.join(LIMIX_ROOT, 'config/cls_default_noretrieval.json')
    if not os.path.exists(config_path):
        print(f"Configuration file not found at: {config_path}")
        sys.exit(1)

    print("Initializing LimiXPredictor...")
    limix_predictor = LimiXPredictor(
        device=device, 
        model_path=model_file, 
        inference_config=config_path
    )

    # 5. Wrap with RDBLearn Adapter
    limix_wrapper = LimiXWrapperClassifier(limix_predictor)

    # 6. Initialize RDBLearn Estimator
    clf = RDBLearnClassifier(base_estimator=limix_wrapper)

    # 7. Train
    print("Training (Extracting features and storing context)...")
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

    # 8. Predict
    print("Predicting on test set...")
    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col]

    # Predict probabilities
    y_pred_proba = clf.predict_proba(X=X_test)
    
    # 9. Evaluate
    # For binary classification, we usually take the probability of the positive class (index 1).
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"Test AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
