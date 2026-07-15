import torch
import numpy as np
from typing import Dict, Any
from appfl.misc.data_readiness import BaseCADREModule


class CADREModuleOutliers(BaseCADREModule):
    """
    CADREModuleOutliers specializing in outlier detection using the IQR method.
    """

    def __init__(self, train_dataset, **kwargs):
        """
        Initialize the CADREModuleOutliers with a training dataset.

        Args:
            train_dataset: The dataset used for training.
            **kwargs: Additional arguments for the base class.
        """
        super().__init__(train_dataset, **kwargs)

    def metric(self, **kwargs) -> Dict[str, Any]:
        """
        Identifies outlier images in the dataset using the IQR method based on image statistics.

        Returns:
            A dictionary containing the proportion of outliers and their indices.
        """
        # Extract images from the dataset
        if hasattr(self.train_dataset, "data_input"):
            images = self.train_dataset.data_input
        else:
            images = torch.stack(
                [img for img, _ in self.train_dataset]
            )  # Convert dataset to tensor batch

        # Compute mean and standard deviation per image
        image_means = images.view(images.shape[0], -1).mean(dim=1).cpu().numpy()
        image_stds = images.view(images.shape[0], -1).std(dim=1).cpu().numpy()

        def find_outliers(data: np.ndarray) -> np.ndarray:
            """
            Identify outliers using the IQR method.

            Args:
                data: A numpy array of data points.

            Returns:
                Indices of outliers.
            """
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = max(q3 - q1, 1e-6)  # Prevent zero division
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return np.where((data < lower_bound) | (data > upper_bound))[0]

        # Find outliers based on mean intensity and standard deviation
        outlier_indices_mean = find_outliers(image_means)
        outlier_indices_std = find_outliers(image_stds)

        # Combine both criteria
        outlier_indices = set(outlier_indices_mean) | set(outlier_indices_std)

        # Compute proportion of outlier images
        proportion_outliers = len(outlier_indices) / len(images)

        return {
            "proportion_outliers": round(proportion_outliers, 4),
            "outlier_indices": list(outlier_indices),
        }

    def rule(self, metric_result: Dict[str, Any], threshold: float = 0.1) -> bool:
        """
        Determine if the proportion of outliers exceeds the threshold.

        Args:
            metric_result: The result of the metric function.
            threshold: The threshold for outlier proportion.

        Returns:
            True if the proportion of outliers exceeds the threshold, False otherwise.
        """
        return metric_result["proportion_outliers"] > threshold

    def remedy(self, metric_result: Dict[str, Any], logger, **kwargs) -> Dict[str, Any]:
        """
        Removes outliers from the dataset based on identified indices.

        Args:
            metric_result: The result of the metric function.
            logger: Logger for logging information.
            **kwargs: Additional arguments.

        Returns:
            A dictionary containing the updated dataset and metadata.
        """
        ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}

        # Get outlier indices from metric result
        outlier_indices = metric_result.get("outlier_indices", [])

        if self.rule(metric_result) and outlier_indices:
            # Remove outlier images by filtering the dataset
            self.train_dataset = [
                (img, label)
                for i, (img, label) in enumerate(self.train_dataset)
                if i not in outlier_indices
            ]

            logger.info(f"Removed {len(outlier_indices)} outlier images.")

        ai_ready_data["ai_ready_dataset"] = self.train_dataset
        return ai_ready_data
