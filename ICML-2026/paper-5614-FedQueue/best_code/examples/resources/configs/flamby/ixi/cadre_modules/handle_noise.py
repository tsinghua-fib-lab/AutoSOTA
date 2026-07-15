import torch
from appfl.misc.data_readiness import BaseCADREModule


class CADREModuleNoise(BaseCADREModule):
    def __init__(self, train_dataset, **kwargs):
        """
        Initialize the CADREModuleNoise class, specializing in noise detection with configurable parameters.

        Args:
        - train_dataset: The dataset used for training.
        - kwargs: Additional keyword arguments for customization.
        """
        super().__init__(train_dataset, **kwargs)

    def metric(self, **kwargs):
        """
        Compute the mean magnitude of the data samples.

        Returns:
        - A dictionary containing:
            - "mean": The mean magnitude of the dataset samples.
            - "magnitudes": A list of magnitudes for each sample in the dataset.
        """
        # Check if the dataset has a 'data_input' attribute; otherwise, process the dataset.
        if hasattr(self.train_dataset, "data_input"):
            data_input = self.train_dataset.data_input
        else:
            # Stack input data from the dataset.
            data_input = torch.stack(
                [input_data for input_data, _ in self.train_dataset]
            )

        # Compute the magnitude of each sample.
        magnitudes = [torch.mean(sample[0] ** 2).item() for sample in data_input]
        # Calculate the mean of the magnitudes.
        mean = sum(magnitudes) / len(magnitudes)

        return {"mean": round(mean, 2), "magnitudes": magnitudes}

    def rule(self, metric_result, noise_threshold=2):
        """
        Apply a rule to identify noisy samples based on a noise threshold.

        Args:
        - metric_result: A dictionary containing computed metric results.
        - noise_threshold: Threshold below which samples are considered non-noisy.

        Returns:
        - A list of indices of samples considered noisy.
        """
        # Identify indices of samples with magnitudes below the noise threshold.
        noisy_indices = [
            i
            for i, mag in enumerate(metric_result["magnitudes"])
            if mag < noise_threshold
        ]
        return noisy_indices

    def remedy(self, metric_result, logger, **kwargs):
        """
        Apply a remedy by filtering out samples without noise.

        Args:
        - metric_result: Dictionary with computed magnitudes.
        - logger: Logger instance for logging modifications.
        - kwargs: Additional keyword arguments for customization.

        Returns:
        - A dictionary containing:
            - "ai_ready_dataset": The filtered dataset containing only samples not noisy.
            - "metadata": Metadata associated with the filtering process (currently None).
        """
        # Initialize the AI-ready data structure.
        ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}

        # Get the indices of noisy samples based on the rule.
        noisy_indices = self.rule(metric_result)

        # Filter the dataset to retain only noisy samples.
        self.train_dataset = [self.train_dataset[i] for i in noisy_indices]

        # Log the filtering process.
        logger.info(f"Filtered dataset to remove {len(noisy_indices)} noisy samples.")

        # Update the AI-ready dataset.
        ai_ready_data["ai_ready_dataset"] = self.train_dataset

        return ai_ready_data
