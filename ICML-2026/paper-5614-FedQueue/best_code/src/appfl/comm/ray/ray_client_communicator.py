import ray
from omegaconf import OmegaConf
from appfl.comm.utils.executor import (
    get_sample_size_executor,
    data_readiness_report_executor,
    train_executor,
)


@ray.remote
class RayClientCommunicator:
    def __init__(self, server_agent_config, client_agent_config):
        self.client_config = OmegaConf.merge(
            server_agent_config.client_configs, client_agent_config
        )

    def get_sample_size(self):
        return get_sample_size_executor(client_agent_config=self.client_config)

    def data_readiness_report(self):
        return data_readiness_report_executor(client_agent_config=self.client_config)

    def train(self, model_ref, metadata=None):
        if isinstance(model_ref, ray.ObjectRef):
            model = ray.get(model_ref)
        else:
            model = model_ref
        return train_executor(
            client_agent_config=self.client_config,
            model=model,
            meta_data=metadata,
        )
