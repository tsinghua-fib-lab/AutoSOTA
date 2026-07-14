TABPFN_DEFAULT_CONFIG = {
    "n_estimators": 8,
    "ignore_pretraining_limits": True,  # Allow datasets larger than 10K samples
    "device": "cuda",
    "n_preprocessing_jobs": -1
}

RDBLEARN_DEFAULT_CONFIG = {
    "dfs": {
        "max_depth": 2,
        "agg_primitives": ["max", "min", "mean", "count", "mode", "std"],
        "engine": "dfs2sql"
    },
    "max_train_samples": 10000,
    "stratified_sampling": False,
    "balanced_sampling": False,
    "predict_batch_size": 5000
}

TARGET_HISTORY_TABLE_NAME = "_RDBL_target_history"