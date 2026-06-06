# Adapted from https://github.com/kumuji/stu_dataset
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_curve
from tqdm import tqdm

from utils.common import convert_to_builtin_types, load_labels, load_point_cloud

"""
# Example usage in a model
metrics = PointOODMetricsCalculator()

for batch in dataloader:
    inputs, labels = preprocess(batch)
    outputs = model(inputs)

    # Assume anomaly_scores is derived from model outputs
    anomaly_scores = compute_anomaly_scores(outputs)

    # Update metrics (convert tensors to numpy if needed)
    metrics.update(inputs.pcd.numpy(), anomaly_scores.numpy(), labels.numpy())

# Get final metrics
final_metrics = metrics.compute_metrics()
"""


class PointOODMetricsCalculator:
    min_eval_distance = 2.5
    max_eval_distance = 50
    min_num_points_to_eval = 5

    def __init__(
        self, nuscenes=False, anomal_class=None
    ):
        self.all_scores = []
        self.all_labels = []
        if anomal_class is not None:
            self.anom_class = anomal_class
        else:
            self.anom_class = 100 if nuscenes else 2

    def update(self, points, scores, target):
        """Update the stored scores and labels with new data.

        Args:
            points (np.ndarray): Point cloud coordinates
            scores (np.ndarray): Anomaly scores (higher means more anomalous).
            target (np.ndarray): Ground truth labels
        """
        distances = np.linalg.norm(points, axis=1)
        # Process labels and apply distance mask
        inlier_labels = np.where(target != 0, 0, -1)
        processed_labels = np.where(target == self.anom_class, 1, inlier_labels)
        processed_labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            processed_labels,
        )
        ignore_mask = processed_labels != -1
        labels = processed_labels[ignore_mask]

        # Only evaluate if sufficient anomaly points
        if np.sum(labels) < self.min_num_points_to_eval:
            return
        if len(scores) != len(target):
            raise ValueError("Prediction and label count mismatch")

        prediction = scores[ignore_mask]
        self.all_scores.append(prediction)
        self.all_labels.append(labels)

    def compute_metrics(self):
        """Compute OOD detection metrics on accumulated data.

        Returns:
            dict: Metrics including AP, FPR95, AUROC, and optimal threshold.
        """
        if not self.all_scores:
            return {}

        targets = np.concatenate(self.all_labels, axis=0)
        predictions = np.concatenate(self.all_scores, axis=0)

        AP = average_precision_score(y_true=targets, y_score=predictions)
        roc_auc, fpr, threshold = self._calculate_auroc(predictions, targets)

        return {
            "AP": AP * 100,
            "FPR95": fpr * 100,
            "AUROC": roc_auc * 100,
            "threshold": threshold,
        }

    @staticmethod
    def _calculate_auroc(predictions, targets):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        optimal_threshold = 0

        for tpr_val, fpr_val, thr in zip(tpr, fpr, thresholds):
            if tpr_val > 0.95:
                fpr_best = fpr_val
                optimal_threshold = thr
                break

        return roc_auc, fpr_best, optimal_threshold


def main(args):
    metrics_calculator = PointOODMetricsCalculator(nuscenes=args.nuscenes)

    for seq_path in tqdm(sorted(list(args.data_dir.glob("1[0-9][0-9]")))):
        pred_seq = args.pred_dir / "sequences" / seq_path.name
        if not pred_seq.is_dir():
            continue
        if seq_path.is_dir():
            lidar_files = sorted((seq_path / "velodyne").glob("*.bin"))

            for pcd_file in tqdm(lidar_files, leave=False, position=1):
                points, _ = load_point_cloud(pcd_file)

                label_file = seq_path / "labels" / f"{pcd_file.stem}.label"
                gt_sem, _ = load_labels(label_file)

                pred_file = args.pred_dir / "sequences" / seq_path.name / "scores" / f"{pcd_file.stem}.txt"
                metrics_calculator.update(
                    points, np.loadtxt(pred_file).astype(np.float32), gt_sem
                )

    metrics = metrics_calculator.compute_metrics()
    print(metrics)
    # with open(args.output, "w") as f:
    #     json.dump(metrics, f, indent=4, default=convert_to_builtin_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Detection Metrics")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="detection_metrics.json")
    parser.add_argument("--nuscenes", action="store_true", default=False, help="Set if evaluating on nuScenes dataset")

    args = parser.parse_args()
    main(args)
