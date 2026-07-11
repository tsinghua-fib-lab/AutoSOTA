import pandas as pd
import numpy as np
from typing import Union, List


class gaussian_MB_learn:
    """
    Implementation of the Gaussian Markov Blanket algorithm.
    References:
        - https://github.com/ban-epfl/rcd/blob/main/rcd/utilities/utils.py#L101
    """
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], alpha: float = None, **kwargs):
        from scipy import stats
        self.num_samples, self.num_nodes = data.shape
        if self.num_samples <= self.num_nodes:
            raise ValueError("Number of samples must be greater than number of nodes.")
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
            self.column_names = list(data.columns)

        elif isinstance(data, np.ndarray):
            self.data = data
            self.column_names = list(range(data.shape[1]))
        else:
            raise ValueError("Unsupported data type. Please provide a pandas DataFrame or a numpy array.")
        
        crr = np.corrcoef(self.data, rowvar=False)
        prec = np.linalg.pinv(crr)
        norm_vec = np.sqrt(np.diag(prec))
        mb_mat = np.abs(prec / norm_vec[:, None] / norm_vec[None, :])

        sig_level = 1 / self.num_nodes ** 2 if alpha is None else alpha

        thresh = np.tanh(stats.norm.ppf(1 - sig_level / 2) / np.sqrt(self.num_samples - self.num_nodes - 1))

        mb_mat = np.where(mb_mat > thresh, 1, -1)  # 1 means MB, -1 means not MB
        # set diagonal to 0
        np.fill_diagonal(mb_mat, 0)

        self.bool_mb_df = pd.DataFrame(
            mb_mat, index=self.column_names, columns=self.column_names
        )
        # print(f"bool_mb_df: {self.bool_mb_df}")

    def __call__(self, target: Union[int, str]) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = [col for col in self.bool_mb_df.columns if self.bool_mb_df.loc[target, col] == 1]
        return mb
    


def MB_learn(data: Union[pd.DataFrame, np.ndarray], alpha: float = None, **kwargs):
    """
    :param data: Input dataset (pd.DataFrame or np.ndarray).
    :param alpha: Significance level for CI tests.
    :param kwargs: Additional arguments for specific learner methods.
    """
    mb_method_type = kwargs.get("mb_method_type", "gaussian_MB")

    if mb_method_type == "gaussian_MB":
        return gaussian_MB_learn(data, alpha, **kwargs)
    else:
        raise ValueError(f"Unknown method type: {mb_method_type}")
    



