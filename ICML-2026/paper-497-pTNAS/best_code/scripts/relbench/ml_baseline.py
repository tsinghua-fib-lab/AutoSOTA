import torch_frame
import argparse
import time
import pandas as pd
from torch_frame.gbdt import LightGBM, CatBoost
from torch_frame.typing import Metric
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from utils.table_data import TableData
from utils.logger import ModernLogger


parser = argparse.ArgumentParser(description="ML baseline for RelBench tasks.")


parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the data directory.")
parser.add_argument("--trials", type=int, default=10,
                    help="Number of trials for model training.")
parser.add_argument("--method", type=str, choices=["lgb", "catboost", "xgboost"],
                    default="lgb", help="Method to use for training.")
parser.add_argument("--verbose", action="store_true", default=False,
                    help="Enable verbose logging.")
args = parser.parse_args()

# Initialize logger
logger = ModernLogger(
    name="ML_Baseline",
    level="info" if args.verbose else "critical",
    rich_tracebacks=False
)

data_dir = args.data_dir

table_data = TableData.load_from_dir(data_dir)



# Display task information
logger.section(f"Task: {table_data.task_type.value}")
task_info = f"Dataset: {data_dir}\n"
task_info += f"Method: {args.method.upper()}\n"
task_info += f"Trials: {args.trials}"
logger.info_panel("Configuration", task_info)

if table_data.task_type == TaskType.REGRESSION:
    tune_metrics = Metric.MAE
    task_type_ = torch_frame.TaskType.REGRESSION
    tm = mean_absolute_error
else:
    tune_metrics = Metric.ROCAUC
    task_type_ = torch_frame.TaskType.BINARY_CLASSIFICATION
    tm = roc_auc_score


if args.method == "lgb":
    model = LightGBM(
        task_type=task_type_,
        metric=tune_metrics,
    )
elif args.method == "catboost":
    model = CatBoost(
        task_type=task_type_,
        metric=tune_metrics,
    )
else:
    raise ValueError(f"Unsupported method: {args.method}")

logger.info("Starting hyperparameter tuning...")
start_time = time.time()

model.tune(tf_train=table_data.train_tf,
           tf_val=table_data.val_tf, num_trials=args.trials)

tune_time = time.time() - start_time
logger.success(f"Tuning completed in {tune_time:.2f}s")

logger.info("Running inference on test set...")
inference_start = time.time()

pred = model.predict(tf_test=table_data.test_tf).numpy()
y = table_data.test_tf.y.numpy()

inference_time = time.time() - inference_start

eval_score = tm(y, pred)

logger.success(
    f"Final Test {tm.__name__}: {eval_score:.4f} | Inference Time: {inference_time:.2f}s")
