import torch
from omegaconf import OmegaConf

from .server import ServerAgent


class SimServerAgent(ServerAgent):
    """Simulation-oriented server agent backed by the standard ServerAgent."""

    def __init__(self, server_agent_config, num_clients=None, **kwargs):
        config = OmegaConf.create(server_agent_config)

        ## Inevitable part due to backward compatibility
        self.sim_config = config.copy()  # note: this is never used
        if "client_configs" not in config:
            algorithm = config.get("algorithm_configs", {})
            log_configs = config.get("log_configs", {})
            server_configs = config.get("server_configs", {})
            trainer_kwargs = algorithm.get("trainer_kwargs", {})
            scheduler_kwargs = OmegaConf.create(algorithm.get("scheduler_kwargs", {}))
            resolved_num_clients = (
                num_clients
                if num_clients is not None
                else scheduler_kwargs.get("num_clients")
            )
            if resolved_num_clients is None:
                raise ValueError(
                    "num_clients must be specified either as an argument or in "
                    "algorithm_configs.scheduler_kwargs.num_clients in the config."
                )
            resolved_num_clients = int(resolved_num_clients)
            config = OmegaConf.create(
                {
                    "client_configs": {
                        "model_configs": config.get("model_configs", {}),
                        "train_configs": OmegaConf.merge(
                            trainer_kwargs,
                            {"trainer": algorithm.get("trainer", "VanillaTrainer")},
                        ),
                    },
                    "server_configs": OmegaConf.merge(
                        server_configs,
                        {
                            "logging_output_dirname": log_configs.get(
                                "logging_output_dirname", "./output"
                            ),
                            "logging_output_filename": log_configs.get(
                                "logging_output_filename", "result"
                            ),
                            "device": algorithm.get("device", "cpu"),
                            "num_clients": resolved_num_clients,
                            "num_global_epochs": int(
                                algorithm.get("num_global_epochs", 1)
                            ),
                            "num_clients_per_round": int(
                                scheduler_kwargs.get(
                                    "num_clients_per_round", resolved_num_clients
                                )
                            ),
                            "scheduler": algorithm.get("scheduler", "SyncScheduler"),
                            "scheduler_kwargs": scheduler_kwargs,
                            "aggregator": algorithm.get(
                                "aggregator", "FedAvgAggregator"
                            ),
                            "aggregator_kwargs": algorithm.get("aggregator_kwargs", {}),
                        },
                    ),
                }
            )
        elif num_clients is not None:
            config.server_configs.num_clients = int(num_clients)
        super().__init__(server_agent_config=config)

    def load_server_val_dataset(self, server_dataset):
        self._val_dataset = server_dataset
        self._val_dataloader = torch.utils.data.DataLoader(
            self._val_dataset, batch_size=1, shuffle=False, num_workers=0
        )

    def set_sample_size(
        self, client_id=None, sample_size=None, sample_sizes=None, **kwargs
    ):
        if sample_sizes is not None:
            for cid, sz in sample_sizes.items():
                self.aggregator.set_client_sample_size(cid, sz)
        else:
            super().set_sample_size(client_id, sample_size, **kwargs)
