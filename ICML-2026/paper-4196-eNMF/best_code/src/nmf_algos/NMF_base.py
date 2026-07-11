import os
import shutil
import logging
from abc import ABC
from typing import Tuple
import numpy as np

from nmf_algos.utils.utils import assert_shape, flush_dict_into_log

logger = logging.getLogger(__name__)


class NMFBase(ABC):
    # ---------- Convenience views ----------
    @property
    def basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return factor matrices (U, V)."""
        if self.U is None or self.V is None:
            raise RuntimeError(
                "Factors are not initialized. Call `factor_init`/`fit` first."
            )
        return self.U, self.V

    @property
    def target(self) -> np.ndarray:
        """Return the matrix to be factorized."""
        if self.X is None:
            raise RuntimeError("Target matrix X is not set.")
        return self.X

    def __init__(self, method_name, params):
        """
        method_name:
        params:
            "X": data matrix
            "r": latent dimension
            "U": optional initial left factor matrix
            "V": optional initial right factor matrix
        """

        assert ("X" in params) and ("r" in params)
        self.X = params["X"]
        self.r = params["r"]
        self.model_name = method_name
        self.method_name = method_name
        self.cur_run_id = 1  # tracking number of runs
        logger.info(
            f"Loaded target matrix with shape {self.X.shape}! Going to factorize it with latent dimension {self.r}"
        )
        if (self.X < 0).any():
            raise ValueError(
                "The input matrix contains negative elements; NMF requires nonnegative data."
            )
        if "U" in params and "V" in params:
            self.U = params["U"]
            self.V = params["V"]
            assert_shape(self.U, [self.X.shape[0], self.r])
            assert_shape(self.V, [self.X.shape[1], self.r])
            logger.info("Loaded initial factors U and V.")

    def method_config_init(self, params):
        # [default, fixed_tim, target_error,]
        if "run_mode" in params:
            self.run_mode = params["run_mode"]
        if "dataset_name" in params:
            self.dataset_name = params["dataset_name"]

        if "save_dir" in params:
            self.save_dir = params["save_dir"]
        else:
            result_dir = os.path.join(os.getcwd(), "Results")
            self.save_dir = os.path.join(
                result_dir,
                self.dataset_name,
                self.method_name,
                f"latent_dim_{self.r}",
                f"{self.cur_run_id}",
            )

        if "iter_save_dir" in params:
            self.iter_save_dir = params["iter_save_dir"]
        else:
            self.iter_save_dir = os.path.join(self.save_dir, "Iters")
        if "rerun_times" in params:
            self.rerun_times = params["rerun_times"]

        self.params = params
        self.set_params(params)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if os.path.exists(self.iter_save_dir):
            shutil.rmtree(self.iter_save_dir)
        os.makedirs(self.iter_save_dir)

    def factor_init(self, params):
        raise NotImplementedError("Subclasses must implement factor_init().")

    def reset_status(self, params):
        # prepare for rerun
        self.method_config_init(params)
        self.factor_init(params)

    def tracker(self, U, V, save_dir, n_iter, added_info=None, prefix=""):
        """
        Save data:
        prefix can be used for dataset name
        """
        f_name_res = (
            prefix
            + self.method_name
            + "_"
            + str(self.r)
            + "_factors_"
            + str(n_iter)
            + ".npy"
        )
        result = {"U": U, "V": V}
        # save time, error info inside factor and into log txt
        if added_info is not None:
            result.update(added_info)
            log_path = os.path.join(save_dir, "log.txt")
            flush_dict_into_log(log_path, n_iter, added_info)
        np.save(os.path.join(save_dir, f_name_res), result)

    def save_factors(self, file_name, info=None):
        """Save final factors and optional metadata."""
        if self.X is None or self.U is None or self.V is None:
            raise RuntimeError("X, U, V must be set before saving factors.")
        result = {"X": self.X, "U": self.U, "V": self.V}
        if info:
            result.update(info)
        save_path = os.path.join(self.save_dir, file_name)
        np.save(save_path, result)

    # ---------- Param helper ----------
    def set_params(self, params):
        """Update instance attributes from a dict (only known fields are changed)."""
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
                logger.debug("Set %s=%r", key, val)
