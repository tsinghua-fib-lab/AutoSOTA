import torch
from appfl.misc.data_readiness import BaseCADREModule


class CADREModuleDuplicates(BaseCADREModule):
    def __init__(self, train_dataset, **kwargs):
        """
        CADREModuleDuplicates specializing in duplicate detection with configurable parameters.
        """
        super().__init__(train_dataset, **kwargs)

    def metric(self, **kwargs):
        """
        Compute the proportion of duplicate samples in a dataset.

        Parameters:
        - dataset: A PyTorch Dataset or list of (input_data, label) tuples.

        Returns:
        - duplicate_proportion: The fraction of duplicate samples (0.0 to 1.0).
        """

        # Determine how to retrieve data input and labels based on dataset attributes
        if hasattr(self.train_dataset, "data_input"):
            data_input = self.train_dataset.data_input
        else:
            data_input = torch.stack(
                [input_data for input_data, _ in self.train_dataset]
            )

        # Count occurrences of each sample
        counts = {}
        for sample in data_input:
            sample_str = str(sample.tolist())
            counts[sample_str] = counts.get(sample_str, 0) + 1

        # Calculate proportion of duplicate samples
        total_samples = len(data_input)
        num_duplicates = total_samples - len(counts)
        duplicate_proportion = num_duplicates / total_samples

        return {"duplicates": round(duplicate_proportion, 2)}

    def rule(self, metric_result: float, threshold=0):
        """
        Check if the rule condition is met.

        Args:
        - metric_result: A dictionary containing computed metric results.

        Returns:
        - True if the rule condition is met, False otherwise.
        """
        return metric_result["duplicates"] >= threshold

    def remedy(self, metric_result, logger, **kwargs):
        """
        Apply the remedy by removing duplicate samples.

        Returns:
        - Modified dataset.
        """
        ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}
        if self.rule(metric_result):
            self.train_dataset = list(set(self.train_dataset))
            ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}

        return ai_ready_data
