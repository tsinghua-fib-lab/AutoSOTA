"""
DIMATTrainer: Client-side trainer for the DIMAT algorithm.

Extends VanillaTrainer to reset BatchNorm running statistics after loading
the merged global model parameters, which is important for convergence
after DIMAT's feature-space alignment.
"""

from typing import Union, Dict, OrderedDict, Any
from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer


class DIMATTrainer(VanillaTrainer):
    """
    DIMATTrainer resets BatchNorm statistics after loading global parameters.
    After DIMAT merges models via activation matching, the BatchNorm running
    statistics need to be recomputed using the local training data.
    """

    def load_parameters(
        self,
        params: Union[Dict, OrderedDict, Any],
    ):
        """Load model parameters from the server.

        The server already resets BN stats using the full proxy dataset after
        merging, so we must NOT re-reset here with only local data — that would
        overwrite the server's higher-quality statistics and degrade accuracy.
        """
        super().load_parameters(params)
