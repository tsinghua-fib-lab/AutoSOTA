import numpy as np
from appfl.agent import ClientAgent


class CADREModule(ClientAgent):
    def generate_mnist_readiness_report(self):
        if hasattr(self.client_agent_config, "dr_metrics"):
            dr_metrics = self.client_agent_config.dr_metrics
            results = {}
            if "ci" in dr_metrics:
                labels = self.train_dataset.data_label.tolist()
                imb_deg = self._imbalance_degree(labels)
                results["ci"] = imb_deg
            if "ss" in dr_metrics:
                sample_size = len(self.train_dataset.data_label.tolist())
                results["ss"] = sample_size
            return results
        else:
            return {}

    def _imbalance_degree(self, lst):
        """
        Calculate the proportions of each class and the proportions for a balanced distribution and return the euclidean distance between the two distributions.
        """
        counts = {}
        for elem in lst:
            counts[elem] = counts.get(elem, 0) + 1
        total_elements = len(lst)
        actual_proportions = np.array(
            [counts[elem] / total_elements for elem in sorted(counts.keys())]
        )
        num_classes = len(counts)
        balanced_proportions = np.array([1 / num_classes] * num_classes)
        euclidean_distance = np.linalg.norm(actual_proportions - balanced_proportions)
        return euclidean_distance
