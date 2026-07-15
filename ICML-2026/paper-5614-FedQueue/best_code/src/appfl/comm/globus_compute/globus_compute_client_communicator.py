def globus_compute_client_entry_point(
    task_name="N/A",
    client_agent_config=None,
    model=None,
    meta_data=None,
):
    """
    Entry point for the Globus Compute client endpoint for federated learning.
    :param `task_name`: The name of the task to be executed.
    :param `client_agent_config`: The configuration for the client agent.
    :param `model`: [Optional] The model to be used for the task.
    :param `meta_data`: [Optional] The metadata for the task.
    :return `model_local`: The local model after the task is executed. [Return `None` if the task does not return a model.]
    :return `meta_data_local`: The local metadata after the task is executed. [Return `{}` if the task does not return metadata.]
    """
    # The new way to execute task for globus compute is to import the executor functions
    # from client's local appfl instead of sending the function logics within the function.
    try:
        from appfl.comm.utils.executor import (
            get_sample_size_executor,
            data_readiness_report_executor,
            train_executor,
        )

        if task_name == "get_sample_size":
            return get_sample_size_executor(client_agent_config=client_agent_config)
        elif task_name == "data_readiness_report":
            return data_readiness_report_executor(
                client_agent_config=client_agent_config
            )
        elif task_name == "train":
            return train_executor(
                client_agent_config=client_agent_config,
                model=model,
                meta_data=meta_data,
            )
        else:
            raise NotImplementedError(
                f"Task {task_name} is not implemented in the client endpoint."
            )
    # Continue to support the old client appfl version until version 2.0.0
    except (ModuleNotFoundError, ImportError):
        from appfl.agent import ClientAgent
        from appfl.comm.utils.client_utils import (
            load_global_model,
            send_local_model,
        )

        client_agent = ClientAgent(client_agent_config=client_agent_config)

        if (
            hasattr(client_agent_config, "data_readiness_configs")
            and hasattr(
                client_agent_config.data_readiness_configs.dr_metrics,
                "cadremodule_configs",
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

        if task_name == "get_sample_size":
            return None, {
                "sample_size": client_agent.get_sample_size(),
                "_deprecated": True,
            }

        elif task_name == "data_readiness_report":
            return None, {
                "data_readiness": client_agent.generate_readiness_report(
                    client_agent_config
                ),
                "_deprecated": True,
            }

        elif task_name == "train":
            if meta_data is not None and "local_steps" in meta_data:
                client_agent.trainer.train_configs.num_local_steps = meta_data[
                    "local_steps"
                ]
            client_agent.train(**meta_data)
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, meta_data_local = local_model
            else:
                meta_data_local = {}
            local_model = send_local_model(
                client_agent.client_agent_config,
                local_model,
                meta_data["local_model_key"]
                if "local_model_key" in meta_data
                else None,
                meta_data["local_model_url"]
                if "local_model_url" in meta_data
                else None,
            )
            meta_data_local["_deprecated"] = True
            return local_model, meta_data_local
        else:
            raise NotImplementedError(
                f"Task {task_name} is not implemented in the client endpoint."
            )
