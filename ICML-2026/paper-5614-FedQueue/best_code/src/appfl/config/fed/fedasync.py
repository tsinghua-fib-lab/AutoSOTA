"""Configuration for numerous asynchronous global update techniques:
(1) ServerFedAsynchronous   : update the global model once receiving one local model with staleness factor applied
(2) ServerFedBuffer:        : store gradients from clients in a buffer until K gradients are received
"""

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


@dataclass
class FedAsync:
    type: str = "fedasync"
    servername: str = "ServerFedAsynchronous"
    clientname: str = "ClientOptim"
    args: DictConfig = OmegaConf.create(
        {
            ## Server update
            "server_learning_rate": 0.01,
            "server_adapt_param": 0.001,
            "server_momentum_param_1": 0.9,
            "server_momentum_param_2": 0.99,
            ## Clients optimizer
            "optim": "SGD",
            "num_local_epochs": 10,
            "optim_args": {
                "lr": 0.001,
            },
            ## Differential Privacy
            ##  use_dp: False  (non-private)
            ##  epsilon: 1      (stronger privacy as the value decreases)
            ##  epsilon: 0.05
            "use_dp": False,
            "epsilon": 1,
            ## Gradient Clipping
            ## clip_value: False (no-clipping)
            ## clip_value: 10    (clipping)
            ## clip_value: 1
            "clip_grad": False,
            "clip_value": 1,
            "clip_norm": 1,
            ## Fed Asynchronous Parameters
            ### Staleness factor
            "alpha": 0.9,
            "staleness_func": {"name": "constant", "args": {"a": 0.5, "b": 4}},
            ### FedBuf: Buffer size
            "K": 3,
            ### FedCompass
            "q_ratio": 0.2,
            "lambda_val": 1.5,
            ### whether the client sends the gradient or the model
            "gradient_based": False,
        }
    )
