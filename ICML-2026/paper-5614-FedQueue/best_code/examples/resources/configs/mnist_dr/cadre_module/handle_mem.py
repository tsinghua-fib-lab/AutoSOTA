import torch
from typing import Any, Dict
from appfl.misc.data_readiness import BaseCADREModule


class CADREModuleMem(BaseCADREModule):
    """
    CADREModuleMem specializing in memory optimization with configurable parameters.
    """

    def __init__(self, train_dataset: Any, **kwargs: Any) -> None:
        """
        Initialize the CADREModuleMem with a training dataset and additional parameters.

        Args:
            train_dataset (Any): The training dataset.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(train_dataset, **kwargs)

    def metric(self, **kwargs: Any) -> Dict[str, float]:
        """
        Calculate memory usage of the training dataset.

        Args:
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Dict[str, float]: A dictionary containing memory usage in MB.
        """
        if hasattr(self.train_dataset, "data_input"):
            data_input = self.train_dataset.data_input
        else:
            data_input = torch.stack(
                [input_data for input_data, _ in self.train_dataset]
            )

        mem_usage = data_input.element_size() * data_input.nelement() / (1024**2)
        return {"mem_usage": round(mem_usage, 2)}

    def rule(self, metric_result: Dict[str, float], threshold: float = 100.0) -> bool:
        """
        Check if the memory usage exceeds the given threshold.

        Args:
            metric_result (Dict[str, float]): The result of the metric calculation.
            threshold (float): The memory usage threshold in MB. Default is 100 MB.

        Returns:
            bool: True if memory usage exceeds the threshold, False otherwise.
        """
        return metric_result["mem_usage"] > threshold

    def remedy(
        self, metric_result: Dict[str, float], logger: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Remedy the dataset based on the metric result and rule.
        If the rule is met, remove duplicates from the dataset.

        Args:
            metric_result (Dict[str, float]): The result of the metric calculation.
            logger (Any): Logger for logging information.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the AI-ready dataset and metadata.
        """
        ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}

        if self.rule(metric_result):
            logger.info(
                "Memory usage exceeds threshold. Removing duplicates from the dataset."
            )
            self.train_dataset = list(set(self.train_dataset))
            ai_ready_data = {"ai_ready_dataset": self.train_dataset, "metadata": None}
        else:
            logger.info("Memory usage is within acceptable limits. No action taken.")

        return ai_ready_data
