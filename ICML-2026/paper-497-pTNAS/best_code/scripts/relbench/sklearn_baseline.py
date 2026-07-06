import argparse
import time
import numpy as np

from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from relbench.base import TaskType
from utils.table_data import TableData
from utils.logger import ModernLogger


parser = argparse.ArgumentParser(description="Sklearn baseline: Choose between LogisticRegression/Ridge or RandomForest for any task type.")


parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the data directory.")
parser.add_argument("--method", type=str, choices=["logistic", "randomforest"],
                    required=True, help="Method to use: 'logistic' or 'randomforest'.")
parser.add_argument("--disable_msg", action="store_false", default=True,
                    help="Enable verbose logging.")

# Random Forest specific arguments
parser.add_argument("--n_estimators", type=int, default=100,
                    help="Number of trees in Random Forest.")
parser.add_argument("--max_depth", type=int, default=30,
                    help="Maximum depth of trees.")
parser.add_argument("--min_samples_split", type=int, default=2,
                    help="Minimum samples required to split a node.")

# Logistic/Ridge specific arguments
parser.add_argument("--max_iter", type=int, default=500,
                    help="Maximum iterations for Logistic Regression or Ridge.")
parser.add_argument("--C", type=float, default=1.0,
                    help="Inverse regularization strength for Logistic Regression.")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="Regularization strength for Ridge Regression.")

args = parser.parse_args()

verbose = args.disable_msg

# Initialize logger
logger = ModernLogger(
    name="Sklearn_Baseline",
    level="info" if verbose else "critical"
)

data_dir = args.data_dir

logger.info("Loading table data...")
table_data = TableData.load_from_dir(data_dir)
logger.success("Table data loaded successfully")

# Display task information
logger.section(f"Task: {table_data.task_type.value}")

# Determine which method to use based on task type and user choice
if table_data.task_type == TaskType.REGRESSION:
    is_regression = True
    if args.method == "logistic":
        method_name = "Ridge Regression"
        task_info = f"Dataset: {data_dir}\n"
        task_info += f"Method: {method_name}\n"
        task_info += f"Alpha: {args.alpha}\n"
        task_info += f"Max Iterations: {args.max_iter}"
    else:  # randomforest
        method_name = "Random Forest Regressor"
        task_info = f"Dataset: {data_dir}\n"
        task_info += f"Method: {method_name}\n"
        task_info += f"N Estimators: {args.n_estimators}\n"
        task_info += f"Max Depth: {args.max_depth}\n"
        task_info += f"Min Samples Split: {args.min_samples_split}"
else:
    is_regression = False
    if args.method == "logistic":
        method_name = "Logistic Regression"
        task_info = f"Dataset: {data_dir}\n"
        task_info += f"Method: {method_name}\n"
        task_info += f"C: {args.C}\n"
        task_info += f"Max Iterations: {args.max_iter}"
    else:  # randomforest
        method_name = "Random Forest Classifier"
        task_info = f"Dataset: {data_dir}\n"
        task_info += f"Method: {method_name}\n"
        task_info += f"N Estimators: {args.n_estimators}\n"
        task_info += f"Max Depth: {args.max_depth}\n"
        task_info += f"Min Samples Split: {args.min_samples_split}"

logger.info_panel("Configuration", task_info)

# Determine task type and metrics
if table_data.task_type == TaskType.REGRESSION:
    eval_metric = mean_absolute_error
    higher_is_better = False
else:
    eval_metric = roc_auc_score
    higher_is_better = True

# Get dataframes and preprocess
logger.info("Loading dataframes and preprocessing...")

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Get dataframes
train_df = table_data.train_df.copy()
val_df = table_data.val_df.copy()
test_df = table_data.test_df.copy()

# Get target column
target_col = table_data.target_col

# Separate features and target
X_train_df = train_df.drop(columns=[target_col])
y_train = train_df[target_col].values

X_val_df = val_df.drop(columns=[target_col])
y_val = val_df[target_col].values

X_test_df = test_df.drop(columns=[target_col])
y_test = test_df[target_col].values

# Use table_data.col_names_dict to identify column types
from torch_frame import stype
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

col_names_dict = table_data.col_names_dict

# Identify numerical and categorical columns based on col_names_dict
numerical_cols = []
categorical_cols = []
timestamp_cols = []
text_cols_to_drop = []

for col in X_train_df.columns:
    if col in col_names_dict.get(stype.numerical, []):
        numerical_cols.append(col)
    elif col in col_names_dict.get(stype.categorical, []):
        categorical_cols.append(col)
    elif col in col_names_dict.get(stype.timestamp, []):
        timestamp_cols.append(col)
    elif col in col_names_dict.get(stype.text_embedded, []):
        text_cols_to_drop.append(col)
    elif col in col_names_dict.get(stype.embedding, []):
        text_cols_to_drop.append(col)
    elif col in col_names_dict.get(stype.multicategorical, []):
        text_cols_to_drop.append(col)
    elif col in col_names_dict.get(stype.sequence_numerical, []):
        text_cols_to_drop.append(col)
    else:
        # Unknown type, drop it
        text_cols_to_drop.append(col)

logger.info(f"Found {len(numerical_cols)} numerical, {len(categorical_cols)} categorical, and {len(timestamp_cols)} timestamp columns")

# Drop unsupported columns
if text_cols_to_drop:
    logger.warning(f"Dropping {len(text_cols_to_drop)} text/embedding/unsupported columns: {text_cols_to_drop}")
    X_train_df = X_train_df.drop(columns=text_cols_to_drop)
    X_val_df = X_val_df.drop(columns=text_cols_to_drop)
    X_test_df = X_test_df.drop(columns=text_cols_to_drop)


# Custom transformer for timestamp features
class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Extract numerical features from datetime columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        result = []

        for col in X.columns:
            # Convert to datetime
            dt_col = pd.to_datetime(X[col], errors='coerce')

            # Extract features
            result.append(dt_col.dt.year.values.reshape(-1, 1))
            result.append(dt_col.dt.month.values.reshape(-1, 1))
            result.append(dt_col.dt.day.values.reshape(-1, 1))
            result.append(dt_col.dt.dayofweek.values.reshape(-1, 1))
            result.append(dt_col.dt.hour.values.reshape(-1, 1))

        return np.hstack(result) if result else np.empty((len(X), 0))

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Create preprocessing pipeline
transformers = []

# Numerical columns pipeline
if len(numerical_cols) > 0:
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    transformers.append(('num', num_transformer, numerical_cols))

# Categorical columns pipeline
if len(categorical_cols) > 0:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

# Timestamp columns pipeline
if len(timestamp_cols) > 0:
    logger.info(f"Adding timestamp transformer for {len(timestamp_cols)} columns")
    timestamp_transformer = Pipeline([
        ('datetime', DatetimeTransformer()),
        ('imputer', SimpleImputer(strategy='median')),  # Handle NaT values
        ('scaler', StandardScaler())
    ])
    transformers.append(('timestamp', timestamp_transformer, timestamp_cols))

if not transformers:
    raise ValueError("No valid features found")

preprocessor = ColumnTransformer(transformers=transformers)

# Fit preprocessor on training data and transform all splits
logger.info("Fitting preprocessor on training data...")
X_train = preprocessor.fit_transform(X_train_df)
X_val = preprocessor.transform(X_val_df)
X_test = preprocessor.transform(X_test_df)

logger.success(f"Data preprocessed: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

# Create and train model based on task type and method choice
logger.info("Training model...")
train_start = time.time()

if args.method == "logistic":
    if is_regression:
        # REGRESSION: Use Ridge Regression
        logger.info(f"Initializing Ridge Regression with alpha={args.alpha}...")
        model = Ridge(alpha=args.alpha, max_iter=args.max_iter, random_state=42)
        model.fit(X_train, y_train)
    else:
        # CLASSIFICATION: Use Logistic Regression
        logger.info(f"Initializing Logistic Regression with C={args.C}...")
        model = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=42,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        model.fit(X_train, y_train)

elif args.method == "randomforest":
    if is_regression:
        # REGRESSION: Use Random Forest Regressor
        logger.info("Initializing Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        model.fit(X_train, y_train)
    else:
        # CLASSIFICATION: Use Random Forest Classifier
        logger.info("Initializing Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        model.fit(X_train, y_train)

train_time = time.time() - train_start
logger.success(f"Training completed in {train_time:.2f}s")

# Validation
logger.info("Evaluating on validation set...")
if is_regression:
    val_pred = model.predict(X_val)
else:
    val_pred = model.predict_proba(X_val)[:, 1]

val_score = eval_metric(y_val, val_pred)
logger.info(f"Validation {eval_metric.__name__}: {val_score:.4f}")

# Testing
logger.info("Running inference on test set...")
inference_start = time.time()

if is_regression:
    test_pred = model.predict(X_test)
else:
    test_pred = model.predict_proba(X_test)[:, 1]

inference_time = time.time() - inference_start

test_score = eval_metric(y_test, test_pred)

logger.success(
    f"Final Test {eval_metric.__name__}: {test_score:.4f} | Inference Time: {inference_time:.2f}s")

# Model-specific analysis
if args.method == "randomforest" and verbose:
    # Feature importance for Random Forest
    logger.section("Feature Importance")
    feature_importance = model.feature_importances_
    top_n = min(10, len(feature_importance))
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]

    logger.info(f"Top {top_n} most important features:")
    for i, idx in enumerate(top_indices, 1):
        logger.info(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")

elif args.method == "logistic" and verbose:
    # Coefficient analysis for Logistic/Ridge
    logger.section("Coefficient Analysis")
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) == 2:
            coef = coef[0]  # For logistic regression with binary classification
        n_features = len(coef)
        n_nonzero = np.sum(np.abs(coef) > 1e-10)
        logger.info(f"Total features: {n_features}")
        logger.info(f"Non-zero coefficients: {n_nonzero}")

        # Show top coefficients
        top_n = min(10, n_features)
        top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
        logger.info(f"\nTop {top_n} features by coefficient magnitude:")
        for i, idx in enumerate(top_indices, 1):
            logger.info(f"  {i}. Feature {idx}: {coef[idx]:.4f}")
