from appfl.agent import ClientAgent
from appfl.comm.utils.client_utils import (
    load_global_model,
    send_local_model,
)


def get_sample_size_executor(
    client_agent_config=None,
    **kwargs,
):
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    if (
        hasattr(client_agent_config, "data_readiness_configs")
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        )
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_agent_config)
    return None, {"sample_size": client_agent.get_sample_size()}


def data_readiness_report_executor(
    client_agent_config=None,
    **kwargs,
):
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    if (
        hasattr(client_agent_config, "data_readiness_configs")
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        )
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_agent_config)
    return None, {
        "data_readiness": client_agent.generate_readiness_report(client_agent_config)
    }


def train_executor(
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    if meta_data is not None and "local_steps" in meta_data:
        client_agent.trainer.train_configs.num_local_steps = meta_data["local_steps"]
    if (
        hasattr(client_agent_config, "data_readiness_configs")
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics, "cadremodule_configs"
        )
        and hasattr(
            client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs,
            "remedy_action",
        )
        and client_agent_config.data_readiness_configs.dr_metrics.cadremodule_configs.remedy_action
    ):
        client_agent.adapt_data(client_config=client_agent_config)
    if model is not None:
        model = load_global_model(client_agent.client_agent_config, model)
        client_agent.load_parameters(model)
    client_agent.train(**meta_data)
    local_model = client_agent.get_parameters()
    if isinstance(local_model, tuple):
        local_model, meta_data_local = local_model
    else:
        meta_data_local = {}
    local_model = send_local_model(
        client_agent.client_agent_config,
        local_model,
        meta_data["local_model_key"] if "local_model_key" in meta_data else None,
        meta_data["local_model_url"] if "local_model_url" in meta_data else None,
    )
    return local_model, meta_data_local
