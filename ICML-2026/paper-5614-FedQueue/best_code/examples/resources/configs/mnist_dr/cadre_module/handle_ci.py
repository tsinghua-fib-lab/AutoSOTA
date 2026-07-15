import torch
import random
import numpy as np
from appfl.misc.data_readiness import BaseCADREModule


class CADREModuleCI(BaseCADREModule):
    """
    CADREModuleCI specializing in noise detection with configurable parameters.
    This agent provides methods to compute class imbalance metrics, apply rules,
    and remedy imbalances in binary classification datasets.
    """

    def __init__(self, train_dataset, **kwargs):
        """
        Initialize the CADREModuleCI.

        Args:
        - train_dataset: The training dataset.
        - kwargs: Additional arguments for the base class.
        """
        super().__init__(train_dataset, **kwargs)

    def metric(self, **kwargs):
        """
        Compute the class imbalance metric for the dataset.

        Returns:
        - A dictionary containing the class imbalance metric.
        """
        # Retrieve labels from the dataset
        try:
            if hasattr(self.train_dataset, "data_label"):
                data_labels = self.train_dataset.data_label.tolist()
            else:
                data_labels = [
                    label.item() if hasattr(label, "item") else label
                    for _, label in self.train_dataset
                ]
        except Exception as e:
            raise ValueError("Failed to extract labels from the dataset.") from e

        # Count occurrences of each class
        counts = {}
        for label in data_labels:
            counts[label] = counts.get(label, 0) + 1

        # Check if only one class exists
        num_classes = len(counts)
        if num_classes == 1:
            return {"class_imbalance": float("inf")}

        # Calculate imbalance degree
        total_elements = len(data_labels)
        actual_proportions = np.array(
            [counts[label] / total_elements for label in sorted(counts.keys())]
        )
        balanced_proportions = np.array([1 / num_classes] * num_classes)
        euclidean_distance = np.linalg.norm(actual_proportions - balanced_proportions)

        # Normalize for binary classification
        if num_classes == 2:
            max_imbalance = np.linalg.norm(
                np.array([1 / total_elements, (total_elements - 1) / total_elements])
                - np.array([0.5, 0.5])
            )
            normalized_distance = euclidean_distance / max_imbalance
            return {"class_imbalance": round(normalized_distance, 2)}

        return {"class_imbalance": round(euclidean_distance, 2)}

    def rule(self, metric_result: dict, threshold=0):
        """
        Check if the rule condition is met based on the metric result.

        Args:
        - metric_result: A dictionary containing computed metric results.
        - threshold: The threshold for class imbalance.

        Returns:
        - True if the rule condition is met, False otherwise.
        """
        return metric_result.get("class_imbalance", 0) > threshold

    def remedy(self, metric_result: dict, logger):
        """
        Apply the remedy by undersampling the majority class to balance the dataset.

        Args:
        - metric_result: Metric result that triggers the rule.
        - logger: Logger instance for logging modifications.

        Returns:
        - A dictionary containing the modified dataset and metadata.
        """
        ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}

        if not self.rule(metric_result):
            return ai_ready_data

        # Extract features and labels
        features, labels = [], []
        for data in self.train_dataset:
            x, y = data
            features.append(x)
            labels.append(int(y))  # Ensure labels are integers

        # Count class occurrences
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        if len(class_counts) != 2:
            logger.warning(
                "Expected binary classification but found more than two classes."
            )
            return ai_ready_data

        # Identify majority and minority classes
        sorted_classes = sorted(class_counts, key=class_counts.get)
        minority_class, majority_class = sorted_classes
        minority_count = class_counts[minority_class]

        # Group samples by class
        minority_samples = [
            (x, y) for x, y in zip(features, labels) if y == minority_class
        ]
        majority_samples = [
            (x, y) for x, y in zip(features, labels) if y == majority_class
        ]

        # Undersample majority class
        random.shuffle(majority_samples)
        majority_samples = majority_samples[:minority_count]

        # Combine and shuffle
        balanced_dataset = minority_samples + majority_samples
        random.shuffle(balanced_dataset)

        # Convert to tensors
        new_features = torch.stack([x for x, _ in balanced_dataset])
        new_labels = torch.tensor([y for _, y in balanced_dataset])

        # Replace with TensorDataset
        self.train_dataset = torch.utils.data.TensorDataset(new_features, new_labels)

        logger.info(
            "Dataset modified: Majority class undersampled to balance class distribution."
        )
        ai_ready_data["ai_ready_dataset"] = self.train_dataset

        return ai_ready_data
