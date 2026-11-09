"""Local adapter for GreySnow's original TensorFlow Squeezeformer.

Model layers and get_model are executed from the archived public notebook.
This adapter replaces Kaggle/TPU paths and IO; it does not reimplement layers.
"""
import ast
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/greysnow"


def load_pickle(folder, name):
    with (DATA / folder / (name + ".p")).open("rb") as handle:
        return pickle.load(handle)


class GreySnowRuntime:
    def __init__(self, notebook="leap-0p0-384dim-nost-3epochs-e30-inference-p"):
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        folder = "leap-msm-ds"
        mean_y = np.asarray(load_pickle(folder, "mean_y"))[0]
        std_y = np.asarray(load_pickle(folder, "std_y"))[0]
        stds = np.where(std_y == 0, 0, 1 / np.where(std_y > 0, std_y, 1))
        self.feature_columns = list(load_pickle("leap-sample-submission-stds-ds", "FEAT_COLS"))
        self.target_columns = list(load_pickle("leap-sample-submission-stds-ds", "TARGET_COLS"))
        self.weights = np.asarray(load_pickle("leap-sample-submission-stds-ds", "stds_new"), dtype=np.float64).reshape(-1)
        profiles = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"]
        self.profile_indices = [self.feature_columns.index(f"{name}_{level}") for name in profiles for level in range(60)]
        self.scalar_indices = [i for i in range(len(self.feature_columns)) if i not in self.profile_indices]
        self.mean_y = mean_y
        self.stds = stds
        namespace = {
            "__name__": "greysnow_public_notebook",
            "tf": tf, "np": np, "strategy": tf.distribute.get_strategy(),
            "col_len": 1680, "col_not_len": 16, "col_num_y": 374,
            "stds_new": self.weights, "stds": stds, "mean_y": mean_y,
        }
        for prefix in ["col_not", "x_col", "X_total"]:
            mean_file = {"col_not": "mean_col_not", "x_col": "x_col_mean", "X_total": "x_total_mean"}[prefix]
            std_file = {"col_not": "std_col_not", "x_col": "x_col_std", "X_total": "x_total_std"}[prefix]
            min_file = {"col_not": "min_col_not", "x_col": "x_col_min", "X_total": "X_total_min"}[prefix]
            mean = np.asarray(load_pickle(folder, mean_file))[0]
            std = np.asarray(load_pickle(folder, std_file))[0]
            minimum = np.minimum(load_pickle(folder, min_file), load_pickle(folder, min_file + "_test"))[0]
            namespace[mean_file] = mean
            namespace[std_file] = std
            norm_min_name = {"col_not": "min_col_not_norm", "x_col": "x_col_norm_min", "X_total": "X_total_norm_min"}[prefix]
            namespace[norm_min_name] = (minimum - mean) / std
        # Notebook names X_total_mean/std with upper-case X after loading.
        namespace["X_total_mean"] = namespace["x_total_mean"]
        namespace["X_total_std"] = namespace["x_total_std"]
        source = ROOT / "solutions/greysnow/notebooks" / (notebook + ".py")
        # Preserve line numbers: TensorFlow AutoGraph reads method source by filename.
        cleaned = "\n".join("# " + line if line.lstrip().startswith(("!", "%")) or " = !" in line else line
                            for line in source.read_text().splitlines())
        tree = ast.parse(cleaned)
        wanted_functions = {"get_model", "get_normed", "decode_tfrec", "loss_fn", "metric_mae"}
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) or
                 isinstance(node, ast.FunctionDef) and node.name in wanted_functions]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        mask = (self.weights != 0).astype(np.float32)[None]
        mask[:, 132:146] = 0
        namespace["loss_mask"] = tf.constant(np.pad(mask, ((0, 0), (0, 6)), constant_values=1), dtype=tf.float32)
        self.namespace = namespace

    def build_model(self, checkpoint, dim=384, head_dim=2048):
        model = self.namespace["get_model"](dim=dim, head_dim=head_dim)
        model.load_weights(str(checkpoint))
        self.model = model
        return model

    def normalize_record(self, x):
        result = self.namespace["get_normed"](dict(x))
        return tf.concat([result["x_total_norm"], result["x_col_norm"], result["x_col_norm_log"],
                          result["wind"], result["x_col_not_norm"]], axis=0)

    def dataset(self, paths):
        options = tf.data.Options()
        options.threading.private_threadpool_size = 8
        options.experimental_deterministic = True
        ds = tf.data.TFRecordDataset([str(p) for p in paths], compression_type="GZIP", num_parallel_reads=1)
        return ds.with_options(options).map(self.namespace["decode_tfrec"], num_parallel_calls=8)

    def physical_predictions(self, normalized, q0002, bad_targets=None):
        normalized = np.asarray(normalized[:, :368], dtype=np.float64).copy()
        if bad_targets is not None:
            normalized[:, bad_targets] = 0
        result = normalized + self.mean_y[None] * self.stds[None]
        result[:, self.weights == 0] = 0
        result /= np.where(self.stds > 0, self.stds, 1)[None]
        result[:, 132:148] = -np.asarray(q0002[:, 12:28], dtype=np.float64) / 1200
        return result


def macro_r2(labels, predictions, weights):
    """sklearn R2 uniform_average on weighted solution, including constant columns."""
    from sklearn.metrics import r2_score
    labels = np.asarray(labels, np.float64) * weights
    predictions = np.asarray(predictions, np.float64) * weights
    per_target = r2_score(labels, predictions, multioutput="raw_values", force_finite=True)
    return float(per_target.mean()), per_target
