
from utils.logger import ModernLogger
from utils.table_data import TableData
from utils.resource import get_text_embedder_cfg
from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
import torch
import math
import torch_frame
import argparse
import copy
import time

from DTM import construct_tabular_model
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType

import sys
import os
import random
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
DEFAULT_OUTPUT_DIR = "run_outputs/data/relbench/baselines/deep_tabular"


parser = argparse.ArgumentParser(description="Model configuration parser")


parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the data directory.")

parser.add_argument("--verbose", action="store_true", default=False,
                    help="Enable verbose logging.")

parser.add_argument("--device", type=str, default="auto",
                    help="Device to use for training. Use 'auto' to randomly select from available GPUs.")

parser.add_argument("--channels", type=int, default=64,
                    help="Number of input channels.")
parser.add_argument("--out_channels", type=int, default=1,
                    help="Number of output channels.")
parser.add_argument("--num_layers", type=int, default=2,
                    help="Number of layers in the model.")
parser.add_argument("--dropout_prob", type=float,
                    default=0.2, help="Dropout probability.")
parser.add_argument("--norm", type=str, choices=[
                    "layer_norm", "batch_norm", "none"], default="layer_norm", help="Normalization type.")
parser.add_argument("--model", type=str,
                    choices=["MLP", "FTTrans", "ResNet", "DFM", "TabM", "ARMNet"], default="MLP", help="Model architecture type.")

# --- training parameters

parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for training.")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate.")
parser.add_argument("--num_epochs", type=int, default=200,
                    help="Number of training epochs.")
parser.add_argument("--early_stop_threshold", type=int, default=10,
                    help="Number of epochs to wait for improvement before early stopping.")
parser.add_argument("--max_round_epoch", type=int,
                    default=20, help="Maximum number of epochs per round.")
parser.add_argument("--log_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Directory to save log and CSV results.")

args = parser.parse_args()

# Setup log file
import datetime
os.makedirs(args.log_dir, exist_ok=True)
dataset_name = Path(args.data_dir).name
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(args.log_dir, f"dnn_{args.model}_{dataset_name}_{timestamp}.log")
csv_filename = os.path.join(args.log_dir, f"dnn_results.csv")
_log_file = open(log_filename, "w")
_orig_stdout = sys.stdout
sys.stdout = type(sys.stdout)(
    *([sys.stdout] if False else [])
) if False else sys.stdout

import io
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self._streams:
            s.flush()

sys.stdout = _Tee(_orig_stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

verbose = args.verbose
# Initialize logger
logger = ModernLogger(
    name="DNN_Baseline",
    level="info" if verbose else "critical",
    rich_tracebacks=False
)

# Device selection with auto-random feature
if args.device == "auto":
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        selected_gpu = random.randint(0, num_gpus - 1)
        device = torch.device(f"cuda:{selected_gpu}")
        logger.info(f"Auto-selected GPU {selected_gpu} from {num_gpus} available GPUs")
    else:
        device = torch.device("cpu")
        logger.warning("No GPUs available, using CPU")
else:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        logger.warning(f"CUDA not available, falling back to CPU")

table_data = TableData.load_from_dir(args.data_dir)

# Display task information
logger.section(f"Task: {table_data.task_type.value}")
task_info = f"Dataset: {args.data_dir}\n"
task_info += f"Device: {device}\n"
task_info += f"Model: {args.model}\n"
task_info += f"Channels: {args.channels}, Layers: {args.num_layers}, Dropout: {args.dropout_prob}\n"
task_info += f"Batch Size: {args.batch_size}, Learning Rate: {args.lr}, Epochs: {args.num_epochs}"
logger.info_panel("Configuration", task_info)

if not table_data.is_materialize:
    text_cfg = get_text_embedder_cfg(
        device="cpu"
    )
    table_data.materilize(
        col_to_text_embedder_cfg=text_cfg,
    )

stype_encoder_dict = construct_stype_encoder_dict(
    default_stype_encoder_cls_kwargs,
)

model_args = {
    "channels": args.channels,
    "out_channels": args.out_channels,
    "num_layers": args.num_layers,
    "dropout_prob": args.dropout_prob,
    "normalization": args.norm,
    "col_names_dict": table_data.col_names_dict,
    "stype_encoder_dict": stype_encoder_dict,
    "col_stats": table_data.col_stats,
}

net = construct_tabular_model(args.model, model_args)

if table_data.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
    evaluate_matric_func = mean_absolute_error
    higher_is_better = False
    is_regression = True
elif table_data.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
    evaluate_matric_func = roc_auc_score
    higher_is_better = True
    is_regression = False

batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
early_stop_threshold = args.early_stop_threshold
max_round_epoch = args.max_round_epoch

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


data_loaders = {
    idx: torch_frame.data.DataLoader(
        getattr(table_data, f"{idx}_tf"),
        batch_size=batch_size,
        shuffle=idx == "train",
        pin_memory=True,
    )
    for idx in ["train", "val", "test"]
}


def deactivate_dropout(net: torch.nn.Module):
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def test(net: torch.nn.Module, loader: torch.utils.data.DataLoader, early_stop: int = -1, is_regression: bool = False):
    pred_list = []
    y_list = []
    early_stop = early_stop if early_stop > 0 else len(loader.dataset)

    if not is_regression:
        net.eval()

    if verbose:
        progress, task_id = logger.tmp_progress(
            total=len(loader), description="Testing")
        progress.start()

    for idx, batch in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())

        if verbose:
            progress.update(task_id, advance=1)

        if idx > early_stop:
            break

    if verbose:
        progress.stop()

    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    pred_list = pred_logits.numpy() if is_regression else pred_list.numpy()
    return pred_list, y_list


#  deactivate dropout layer in regression task
if is_regression:
    deactivate_dropout(net)

net.to(device)
patience = 0
best_epoch = 0
best_val_metric = -math.inf if higher_is_better else math.inf
best_model_state = None
start_time = time.time()
for epoch in range(num_epochs):
    loss_accum = count_accum = 0
    net.train()

    if verbose:
        progress, task_id = logger.tmp_progress(
            total=min(len(data_loaders["train"]), max_round_epoch + 1),
            description=f"Epoch {epoch}"
        )
        progress.start()

    for idx, batch in enumerate(data_loaders["train"]):

        if idx > max_round_epoch:
            break

        optimizer.zero_grad()
        batch = batch.to(device)
        pred = net(batch)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        y = batch.y.float()
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        count_accum += 1

        if verbose:
            progress.update(task_id, advance=1)

    if verbose:
        progress.stop()

    train_loss = loss_accum / count_accum
    val_logits, val_pred_hat = test(
        net, data_loaders["val"], is_regression=is_regression)
    val_metric = evaluate_matric_func(val_pred_hat, val_logits)

    if verbose:
        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val {evaluate_matric_func.__name__}: {val_metric:.6f}")
    if (higher_is_better and val_metric > best_val_metric) or \
       (not higher_is_better and val_metric < best_val_metric):
        best_val_metric = val_metric
        best_epoch = epoch
        best_model_state = copy.deepcopy(net.state_dict())
        patience = 0

        test_logits, test_pred_hat = test(
            net, data_loaders["test"], is_regression=is_regression)
        test_metric = evaluate_matric_func(test_pred_hat, test_logits)

        if verbose:
            logger.info(
                f"Updated best scores | Test {evaluate_matric_func.__name__}: {test_metric:.6f}")
    else:
        patience += 1
        if patience > early_stop_threshold:
            if verbose:
                logger.warning(f"Early stopping at epoch {epoch}")
            break
end_time = time.time()
training_duration = end_time - start_time

logger.success(
    f"Training completed in {training_duration:.2f}s | Best Val {evaluate_matric_func.__name__}: {best_val_metric:.6f} at epoch {best_epoch}"
)

start_time = time.time()
net.load_state_dict(best_model_state)
test_logits, test_pred_hat = test(
    net, data_loaders["test"], is_regression=is_regression)
test_metric = evaluate_matric_func(test_pred_hat, test_logits)
end_time = time.time()
inference_time = end_time - start_time

logger.success(
    f"Final Test {evaluate_matric_func.__name__}: {test_metric:.6f} | Inference Time: {inference_time:.2f}s")

# Save results to CSV
import csv as _csv
csv_exists = os.path.exists(csv_filename)
with open(csv_filename, "a", newline="") as _f:
    _writer = _csv.DictWriter(_f, fieldnames=[
        "timestamp", "dataset", "model", "channels", "num_layers",
        "best_val_metric", "test_metric", "train_time_seconds",
        "inference_time_seconds", "metric", "device",
    ])
    if not csv_exists:
        _writer.writeheader()
    _writer.writerow({
        "timestamp": timestamp,
        "dataset": dataset_name,
        "model": args.model,
        "channels": args.channels,
        "num_layers": args.num_layers,
        "best_val_metric": round(best_val_metric, 6),
        "test_metric": round(test_metric, 6),
        "train_time_seconds": round(training_duration, 2),
        "inference_time_seconds": round(inference_time, 2),
        "metric": evaluate_matric_func.__name__,
        "device": str(device),
    })
print(f"📁 Log: {log_filename}")
print(f"📊 CSV: {csv_filename}")

sys.stdout = _orig_stdout
_log_file.close()
