#!/usr/bin/env python3

from utils import TableData
import argparse
import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from relbench.base import TaskType
from torch_frame import stype


# Add parent directory to sys.path
# sys.path.append(str(Path(__file__).resolve().parent.parent))


def preprocess_dataframe_label_encoding(table_data,
                                        max_features=500,
                                        numerical_impute_strategy='median',
                                        categorical_impute_strategy='most_frequent',
                                        train_sample_indices=None,
                                        verbose=False):
    """
    Preprocess dataframes using TableData metadata with Label Encoding:
    1. Fit imputers/encoders/PCA on full training data
    2. Transform only sampled training data (if train_sample_indices provided)
    3. Transform full val/test data
    4. Drop non-tabular columns (text_embedded, timestamp, embedding, etc.)
    5. Return categorical feature indices for models that need them

    Args:
        train_sample_indices: If provided, only transform these indices from training data
                             (but still fit on full training data)
    """
    # Get dataframes
    train_df = table_data.train_df.copy()
    val_df = table_data.val_df.copy()
    test_df = table_data.test_df.copy()

    # Get target column and separate features
    target_col = table_data.target_col
    X_train_df_full = train_df.drop(columns=[target_col])
    X_val_df = val_df.drop(columns=[target_col])
    X_test_df = test_df.drop(columns=[target_col])

    y_train_full = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # Determine which training samples to transform
    if train_sample_indices is not None:
        if verbose:
            print(f"Will fit on {len(X_train_df_full)} samples, transform {len(train_sample_indices)} samples")
        X_train_df_transform = X_train_df_full.iloc[train_sample_indices]
        y_train = y_train_full[train_sample_indices]
    else:
        X_train_df_transform = X_train_df_full
        y_train = y_train_full

    # Identify column types using col_to_stype
    numerical_cols = []
    categorical_cols = []
    cols_to_drop = []

    for col in X_train_df_full.columns:
        col_stype = table_data.col_to_stype.get(col)
        if col_stype == stype.numerical:
            numerical_cols.append(col)
        elif col_stype == stype.categorical:
            categorical_cols.append(col)
        else:
            cols_to_drop.append(col)

    if verbose:
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        print(
            f"Dropped columns: {len(cols_to_drop)} (types: {set([table_data.col_to_stype.get(c) for c in cols_to_drop])})")

    # Drop non-tabular columns
    if cols_to_drop:
        X_train_df_full = X_train_df_full.drop(columns=cols_to_drop)
        X_train_df_transform = X_train_df_transform.drop(columns=cols_to_drop)
        X_val_df = X_val_df.drop(columns=cols_to_drop)
        X_test_df = X_test_df.drop(columns=cols_to_drop)

    # Keep numerical data and impute NaN values
    numerical_imputer = None
    if numerical_cols:
        # Fit on full training data
        X_train_num_full = X_train_df_full[numerical_cols]

        nan_count = X_train_num_full.isna().sum().sum()
        if nan_count > 0:
            if verbose:
                print(
                    f"Found {nan_count} NaN values in numerical columns, imputing with {numerical_impute_strategy}...")
            numerical_imputer = SimpleImputer(
                strategy=numerical_impute_strategy)
            numerical_imputer.fit(X_train_num_full)

        # Transform only sampled training data (no copy needed)
        if numerical_imputer is not None:
            X_train_num = pd.DataFrame(
                numerical_imputer.transform(X_train_df_transform[numerical_cols]),
                columns=numerical_cols,
                index=X_train_df_transform.index
            )
            X_val_num = pd.DataFrame(
                numerical_imputer.transform(X_val_df[numerical_cols]),
                columns=numerical_cols,
                index=X_val_df.index
            )
            X_test_num = pd.DataFrame(
                numerical_imputer.transform(X_test_df[numerical_cols]),
                columns=numerical_cols,
                index=X_test_df.index
            )
        else:
            X_train_num = X_train_df_transform[numerical_cols]
            X_val_num = X_val_df[numerical_cols]
            X_test_num = X_test_df[numerical_cols]
    else:
        X_train_num = pd.DataFrame()
        X_val_num = pd.DataFrame()
        X_test_num = pd.DataFrame()

    # Handle categorical data: impute NaN first, then label encode
    categorical_imputer = None
    label_encoders = {}
    if categorical_cols:
        # Fit on full training data (no copy needed for read-only operation)
        X_train_cat_raw_full = X_train_df_full[categorical_cols]

        cat_nan_count = X_train_cat_raw_full.isna().sum().sum()
        if cat_nan_count > 0:
            if verbose:
                print(
                    f"Found {cat_nan_count} NaN values in categorical columns, imputing with {categorical_impute_strategy}...")
            categorical_imputer = SimpleImputer(
                strategy=categorical_impute_strategy, fill_value='missing')
            categorical_imputer.fit(X_train_cat_raw_full)
            # Apply imputation for label encoder fitting
            X_train_cat_raw_full_imputed = pd.DataFrame(
                categorical_imputer.transform(X_train_cat_raw_full),
                columns=categorical_cols,
                index=X_train_cat_raw_full.index
            )
        else:
            X_train_cat_raw_full_imputed = X_train_cat_raw_full

        # Fit label encoders on full training data
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X_train_cat_raw_full_imputed[col])
            label_encoders[col] = le

        # Transform only sampled training data (no copy needed)
        if categorical_imputer is not None:
            X_train_cat_raw = pd.DataFrame(
                categorical_imputer.transform(X_train_df_transform[categorical_cols]),
                columns=categorical_cols,
                index=X_train_df_transform.index
            )
            X_val_cat_raw = pd.DataFrame(
                categorical_imputer.transform(X_val_df[categorical_cols]),
                columns=categorical_cols,
                index=X_val_df.index
            )
            X_test_cat_raw = pd.DataFrame(
                categorical_imputer.transform(X_test_df[categorical_cols]),
                columns=categorical_cols,
                index=X_test_df.index
            )
        else:
            X_train_cat_raw = X_train_df_transform[categorical_cols]
            X_val_cat_raw = X_val_df[categorical_cols]
            X_test_cat_raw = X_test_df[categorical_cols]

        # Label encode each categorical column using pandas map (fast vectorized operation)
        X_train_cat = pd.DataFrame(index=X_train_cat_raw.index)
        X_val_cat = pd.DataFrame(index=X_val_cat_raw.index)
        X_test_cat = pd.DataFrame(index=X_test_cat_raw.index)

        for col in categorical_cols:
            le = label_encoders[col]
            # Create mapping dict once per column (faster than repeated le.transform calls)
            mapping = dict(zip(le.classes_, range(len(le.classes_))))

            # Use pandas map with fillna for unseen categories (vectorized, very fast)
            X_train_cat[col] = X_train_cat_raw[col].map(mapping).fillna(-1).astype(int)
            X_val_cat[col] = X_val_cat_raw[col].map(mapping).fillna(-1).astype(int)
            X_test_cat[col] = X_test_cat_raw[col].map(mapping).fillna(-1).astype(int)

        X_train_combined = pd.concat([X_train_num, X_train_cat], axis=1)
        X_val_combined = pd.concat([X_val_num, X_val_cat], axis=1)
        X_test_combined = pd.concat([X_test_num, X_test_cat], axis=1)
    else:
        X_train_combined = X_train_num
        X_val_combined = X_val_num
        X_test_combined = X_test_num

    if verbose:
        print(f"Total features before PCA: {X_train_combined.shape[1]}")
        print(
            f"Numerical features: {len(numerical_cols)}, Categorical features: {len(categorical_cols)}")

    # Check if categorical features alone exceed max_features
    if len(categorical_cols) > max_features:
        raise ValueError(
            f"Cannot support: Categorical features ({len(categorical_cols)}) exceed max_features ({max_features}). "
            f"PCA is only applied to numerical features. Please increase max_features or reduce categorical features."
        )

    # Apply PCA only on numerical features if they exceed available budget
    pca = None
    scaler = None
    available_budget_for_numerical = max_features - len(categorical_cols)

    if len(numerical_cols) > available_budget_for_numerical:
        if verbose:
            print(
                f"Applying PCA to numerical features only: {len(numerical_cols)} -> {available_budget_for_numerical}")

        # Fit scaler and PCA on full numerical training data
        X_train_num_full_for_pca = X_train_df_full[numerical_cols].copy()
        if numerical_imputer is not None:
            X_train_num_full_for_pca = pd.DataFrame(
                numerical_imputer.transform(X_train_num_full_for_pca),
                columns=numerical_cols
            )

        scaler = StandardScaler()
        X_train_num_full_scaled = scaler.fit_transform(X_train_num_full_for_pca)

        pca = PCA(n_components=available_budget_for_numerical, random_state=42)
        pca.fit(X_train_num_full_scaled)

        # Transform only sampled training data
        X_train_num_scaled = scaler.transform(X_train_num)
        X_train_num_pca = pca.transform(X_train_num_scaled)

        X_val_num_scaled = scaler.transform(X_val_num)
        X_val_num_pca = pca.transform(X_val_num_scaled)

        X_test_num_scaled = scaler.transform(X_test_num)
        X_test_num_pca = pca.transform(X_test_num_scaled)

        if verbose:
            print(
                f"Explained variance ratio (numerical): {pca.explained_variance_ratio_.sum():.4f}")

        # Convert PCA results to DataFrames
        num_feature_names = [
            f'num_PC{i + 1}' for i in range(available_budget_for_numerical)]
        X_train_num_final = pd.DataFrame(
            X_train_num_pca, columns=num_feature_names, index=X_train_num.index)
        X_val_num_final = pd.DataFrame(
            X_val_num_pca, columns=num_feature_names, index=X_val_num.index)
        X_test_num_final = pd.DataFrame(
            X_test_num_pca, columns=num_feature_names, index=X_test_num.index)

        # Concatenate with categorical features
        if categorical_cols:
            X_train_final = pd.concat([X_train_num_final, X_train_cat], axis=1)
            X_val_final = pd.concat([X_val_num_final, X_val_cat], axis=1)
            X_test_final = pd.concat([X_test_num_final, X_test_cat], axis=1)
        else:
            X_train_final = X_train_num_final
            X_val_final = X_val_num_final
            X_test_final = X_test_num_final

        # Get categorical feature indices (they're at the end after PCA numerical features)
        categorical_indices = list(
            range(available_budget_for_numerical, X_train_final.shape[1]))
        feature_names = list(X_train_final.columns)

    else:
        # No PCA needed - use original combined features
        X_train_final = X_train_combined
        X_val_final = X_val_combined
        X_test_final = X_test_combined
        feature_names = list(X_train_combined.columns)

        # Get categorical feature indices
        categorical_indices = []
        for i, col in enumerate(X_train_final.columns):
            if col in categorical_cols:
                categorical_indices.append(i)

    # Convert to numpy arrays
    X_train_final = X_train_final.values
    X_val_final = X_val_final.values
    X_test_final = X_test_final.values

    if verbose:
        print(f"Final feature shape: {X_train_final.shape}")
        print(
            f"Categorical feature indices: {len(categorical_indices)} features at positions {categorical_indices[:5]}{'...' if len(categorical_indices) > 5 else ''}")

    results = {
        'X_train': X_train_final,
        'X_val': X_val_final,
        'X_test': X_test_final,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'categorical_indices': categorical_indices,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TabPFN Baseline for Table Data")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data directory.")

    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging.")

    parser.add_argument("--max_features", type=int, default=500,
                        help="Maximum number of features (apply PCA if exceeded).")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to TabPFN model checkpoint. If not provided, uses task-specific default.")

    parser.add_argument("--device", type=str, default="cuda:5",
                        help="Device to use for TabPFN (e.g., 'cuda:0', 'cpu').")

    parser.add_argument("--numerical_impute_strategy", type=str, default="median",
                        choices=["mean", "median", "most_frequent"],
                        help="Strategy for imputing numerical NaN values.")

    parser.add_argument("--categorical_impute_strategy", type=str, default="most_frequent",
                        choices=["most_frequent", "constant"],
                        help="Strategy for imputing categorical NaN values.")

    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for test set inference. If None, infer all at once.")

    args = parser.parse_args()

    verbose = args.verbose
    batch_size = args.batch_size
    # Initialize logger

    # Load data
    print("Loading Data")
    table_data = TableData.load_from_dir(args.data_dir)

    # Display task information
    print(f"Task: {table_data.task_type.value}")
    task_info = f"Dataset: {args.data_dir}\n"
    task_info += f"Model: TabPFN\n"
    task_info += f"Max Features: {args.max_features}\n"
    task_info += f"Device: {args.device}"
    task_info += f"Batch Size: {args.batch_size}\n"
    print("Configuration", task_info)

    # Determine training sample indices (TabPFN limit: 10,000 samples)
    train_sample_indices = None
    if table_data.train_df.shape[0] > 10000:
        if verbose:
            print(
                f"Will sample training data from {table_data.train_df.shape[0]} to 10,000 samples")
        np.random.seed(42)
        train_sample_indices = np.random.choice(
            table_data.train_df.shape[0], size=10000, replace=False)

    # Preprocess data (fit on full training, transform sampled subset)
    print("Preprocessing Data")
    start_time = time.time()
    processed = preprocess_dataframe_label_encoding(
        table_data,
        max_features=args.max_features,
        numerical_impute_strategy=args.numerical_impute_strategy,
        categorical_impute_strategy=args.categorical_impute_strategy,
        train_sample_indices=train_sample_indices,
        verbose=verbose
    )
    preprocessing_time = time.time() - start_time

    if verbose:
        print(f"Preprocessing completed in {preprocessing_time:.2f}s")

    X_train = processed['X_train']
    X_val = processed['X_val']
    X_test = processed['X_test']
    y_train = processed['y_train']
    y_val = processed['y_val']
    y_test = processed['y_test']
    categorical_indices = processed['categorical_indices']

    # Import TabPFN
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError:
        print("TabPFN not installed. Please install it first.")
        sys.exit(1)

    # Determine task type and metrics
    if table_data.task_type == TaskType.REGRESSION:
        evaluate_metric_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
        model_cls = TabPFNRegressor
        default_model_path = "./tabPFN/tabpfn-v2-regressor.ckpt"
    elif table_data.task_type == TaskType.BINARY_CLASSIFICATION:
        evaluate_metric_func = roc_auc_score
        higher_is_better = True
        is_regression = False
        model_cls = TabPFNClassifier
        default_model_path = "./tabPFN/tabpfn-v2-classifier.ckpt"
    else:
        print(f"Unsupported task type: {table_data.task_type}")
        sys.exit(1)

    # Initialize TabPFN model
    print("Initializing TabPFN Model")

    model_kwargs = {
        "device": args.device,
    }

    # Determine which model path to use
    model_path = args.model_path if args.model_path else default_model_path

    # Only add model_path if it exists
    if Path(model_path).exists():
        model_kwargs["model_path"] = model_path
        if verbose:
            print(f"Using model checkpoint: {model_path}")
    else:
        if verbose:
            print(
                f"Model checkpoint not found at {model_path}, using default")

    # Add categorical indices if not using PCA
    if len(categorical_indices) > 0:
        model_kwargs["categorical_features_indices"] = categorical_indices
        if verbose:
            print(
                f"Using {len(categorical_indices)} categorical features")

    clf = model_cls(**model_kwargs)

    # Train model
    print("Training TabPFN")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f}s")

    # Evaluate on validation set
    # print("Evaluating on Validation Set")
    # start_time = time.time()

    # if is_regression:
    #     val_pred = clf.predict(X_val)
    #     val_metric = evaluate_metric_func(y_val, val_pred)
    # else:
    #     val_pred_proba = clf.predict_proba(X_val)
    #     val_metric = evaluate_metric_func(y_val, val_pred_proba[:, 1])

    # val_inference_time = time.time() - start_time

    # print(
    #     f"Val {evaluate_metric_func.__name__}: {val_metric:.6f} | Inference Time: {val_inference_time:.2f}s"
    # )

    # Evaluate on test set
    print("Evaluating on Test Set")
    start_time = time.time()

    test_pred = []
    for i in range(0, X_test.shape[0], batch_size):
        X_test_batch = X_test[i:i + batch_size]
        if is_regression:
            batch_pred = clf.predict(X_test_batch)
        else:
            batch_pred = clf.predict_proba(X_test_batch)[:, 1]
        test_pred.append(batch_pred)

    test_pred = np.concatenate(test_pred, axis=0)
    test_metric = evaluate_metric_func(y_test, test_pred)

    test_inference_time = time.time() - start_time

    print(
        f"Final Test {evaluate_metric_func.__name__}: {test_metric:.6f} | Inference Time: {test_inference_time:.2f}s"
    )

    # Print summary
    print("Summary")
    summary = f"Preprocessing: {preprocessing_time:.2f}s\n"
    summary += f"Training: {training_time:.2f}s\n"
    # summary += f"Val {evaluate_metric_func.__name__}: {val_metric:.6f}\n"
    summary += f"Test {evaluate_metric_func.__name__}: {test_metric:.6f}\n"
    summary += f"Total Time: {preprocessing_time + training_time + test_inference_time:.2f}s"
    print("Results", summary)
