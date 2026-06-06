import os.path as osp
from typing import Dict, Sequence

import numpy as np
from mmengine.logging import MMLogger, print_log
from PIL import Image

from mmseg.registry import METRICS
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from collections import defaultdict


@METRICS.register_module()
class DGIoUMetric(IoUMetric):
    def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                res1, res2, res3, res4 = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )
                dataset_key = "unknown"
                for key in self.dataset_keys:
                    if key in data_samples[0]["seg_map_path"]:
                        dataset_key = key
                        break
                self.results.append([dataset_key, res1, res2, res3, res4])
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get("reduce_zero_label", False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])
        metrics_type2mean = defaultdict(list)
        for key, key_result in dataset_results.items():
            logger: MMLogger = MMLogger.get_current_instance()
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = super().compute_metrics(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")
            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)
        for k, v in metrics_type2mean.items():
            metrics[f"mean_{k}"] = sum(v) / len(v)
        return metrics
