APPFL Server Agent
==================

Functionalities
---------------

APPFL server agent acts on behalf of the federated learning server to fulfill various tasks that the clients request to do, including

- Get the current model parameters (useful for initialization and getting the final trained model)
- Get the configurations for local training and other tasks that are shared among all clients (e.g., type of trainers, compression configurations, model architecture, etc.)
- Update the global model using the model parameters/gradients trained locally from the clients
- Other tasks that the server agent needs to do to manage the federated learning process (e.g., record the sample size of each client)

Specifically, the current server agent has the following functionalities.

.. note::

    User can also define their functionalities by either inheriting the ``ServerAgent`` class or directly adding new methods to the current server agent. Additionally, if you think your added functionalities are useful for other users, please consider contributing to the APPFL package by submitting a pull request.

.. code-block:: python

    class ServerAgent:
        def __init__(
            self,
            server_agent_config: ServerAgentConfig = ServerAgentConfig()
        ) -> None:
            """
            Initialize the server agent with the configurations.
            """

        def get_client_configs(self, **kwargs) -> DictConfig:
            """
            Return the FL configurations that are shared among all clients.
            """

        def global_update(
            self,
            client_id: Union[int, str],
            local_model: Union[Dict, OrderedDict, bytes],
            blocking: bool = False,
            **kwargs
        ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Update the global model using the local model from a client and return the updated global model.
            :param: `client_id`: A unique client id for server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
            :param: `local_model`: The local model from a client, can be serialized bytes.
            :param: `blocking`: The global model may not be immediately available for certain aggregation methods (e.g. any synchronous method).
                Setting `blocking` to `True` will block the client until the global model is available.
                Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
            :return: The updated global model (as a Dict or OrderedDict), and optional metadata (as a Dict) if `blocking` is `True`.
                Otherwise, return the `Future` object of the updated global model and optional metadata.
            """

        def get_parameters(
            self,
            blocking: bool = False,
            **kwargs
        ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Return the global model to the clients.
            :param: `blocking`: The global model may not be immediately available (e.g. if the server wants to wait for all client
                to send the `get_parameters` request before returning the global model for same model initialization).
                Setting `blocking` to `True` will block the client until the global model is available.
            :param: `kwargs`: Additional arguments for the method. Specifically,
                - `init_model`: whether getting the initial model (which should be same among all clients, thus blocking)
                - `serial_run`: set `True` if for serial simulation run, thus no blocking is needed.
                - `globus_compute_run`: set `True` if for globus compute run, thus no blocking is needed.
            """

        def set_sample_size(
                self,
                client_id: Union[int, str],
                sample_size: int,
                sync: bool = False,
                blocking: bool = False,
            ) -> Optional[Union[Dict, Future]]:
            """
            Set the size of the local dataset of a client.
            :param: client_id: A unique client id for server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
            :param: sample_size: The size of the local dataset of a client.
            :param: sync: Whether to synchronize the sample size among all clients. If `True`, the method can return the relative weight of the client.
            :param: blocking: Whether to block the client until the sample size of all clients is synchronized.
                If `True`, the method will return the relative weight of the client.
                Otherwise, the method may return a `Future` object of the relative weight, which will be resolved
                when the sample size of all clients is synchronized.
            """

        def training_finished(self, internal_check: bool = False) -> bool:
            """
            Indicate whether the training is finished.
            :param: Whether this is an internal check (e.g., for the server to check if the training is finished) or it is requested by the client.
            """

        def server_terminated(self):
            """
            Indicate whether the server can be terminated from listening to the clients.
            """

Configurations
--------------

As shown above, to create a server agent, you need to provide the configurations for the server agent. The configurations for the server agent are defined in the ``appfl.config.ServerAgentConfig`` class, which can be directly loaded from a YAML file. The following file is an example configuration YAML file for the server agent.

The configuration files is composed of two main parts:

- ``client_configs``: Containing the configurations that are shared among all clients. This part is used to define the configurations for the federated learning process, including the type of trainers, the model architecture, the compression configurations, etc.
- ``server_configs``: Containing the configurations for the server agent, including the configurations for the aggregation method, the scheduling method, etc.


.. literalinclude:: ../_static/server_fedavg.yaml
    :language: yaml
    :caption: Server agent configuration YAML file for FedAvg algorithm on the MNIST dataset with a simple CNN model.

Client Configurations
~~~~~~~~~~~~~~~~~~~~~

For client configurations that are shared among all clients, it is composed of three main components:

- ``train_configs``: This component contains all training-related configurations, which can be further classified into the following sub-components:

    - *Trainer configurations*: It should be noted that the required trainer configurations depend on the trainer you use. You can also define your own trainer with any additional configurations you need, and then provide those configurations under ``client_config.train_configs`` in the server configuration yaml file.

        - ``trainer``: The class name of the trainer you would like to use for client local training. The trainer name should be defined in ``src/appfl/trainer``. For example, ``VanillaTrainer`` simply updates the model for a certain number of epochs or batches.
        - ``mode``: For ``VanillaTrainer``, mode is a required configuration to with allowable values ``epoch`` or ``step`` to specify whether you want to train for a certain number of epochs or only a certain number of steps/batches.
        - ``num_local_steps`` / ``num_local_epochs``: Number of steps (if ``mode=step``) or epochs (if ``mode=epoch``) for an FL client in each local training round.
        - ``optim``: Name of the optimizer to use from the ``torch.optim`` module.
        - ``optim_args``: Keyword arguments for the selected optimizer.
        - ``do_validation``: Whether to perform client-side validation in each training round.
        - ``do_pre_validation``: Whether to perform client-side validation prior to local training.
        - ``use_dp``: Whether to use differential privacy.
        - ``epsilon``, ``clip_grad``, ``clip_value``, ``clip_norm``: Parameters used if differential privacy is enabled.
    - *Loss function*: To specify the loss function to use during local training, we provide two options:

        - Loss function from ``torch``: By providing the name of the loss function available in `torch.nn` (e.g., ``CrossEntropyLoss``) in ``loss_fn`` and corresponding arguments in ``loss_fn_kwargs``, user can employ loss function available in PyTorch.
        - Loss function defined in local file: User can define their own loss function by inheriting ``nn.Module`` and defining its ``forward()`` function. Then the user needs to provide the path to the defined loss function file in ``loss_fn_path``, and the class name of the defined loss function in ``loss_fn_name``.
    - *Metric function*: To specify the metric function used during validation, user need to provide path to the file containing the metric function in ``metric_path`` and the name of the metric function in ``metric_name``.
    - *Dataloader settings*: While the server-side configuration does not contain any information about each client's local dataset, it can specify the configurations when converting the dataset to dataloader, such as the batch size and whether to shuffle.
- ``model_configs``: This component contains the definition of the machine learning model used in the FL experiment. The model architecture should be defined as a ``torch.nn.Module`` in a local file on the server-side and then provides the following information:

    - ``model_path``: Path to the model definition file.
    - ``model_name``: Class name of the defined model.
    - ``model_kwargs``: Keyword arguments for initiating a model.
- ``comm_configs``: This component contains the settings for the communication between the FL server and clients, such as the ``compression_configs``.

Server Configurations
~~~~~~~~~~~~~~~~~~~~~

Specifically, it contains the following key components:

- *Scheduler configurations*: User can specify the name of the scheduler (``scheduler``), and the corresponding keyword arguments (``scheduler_kwargs``). All supported schedulers are available at ``src/appfl/scheduler``.
- *Aggregator configurations*: User can specify the name of the aggregator (``aggregator``), and the corresponding keyword arguments (``aggregator_kwargs``). All supported aggregators are available at ``src/appfl/aggregator``.
- *Communicator configurations*: Containing the configurations for the communication between the server and clients, such as the ``grpc_configs``.
- *Logging configurations and others*: Containing the configurations for logging  such as the ``logging_output_dirname`` and ``logging_output_filename``, as well as ```num_global_epochs```.

.. note::

    You may notices that both ``server_configs`` and ``client_configs`` have a ``comm_configs`` fields. Actually, when creating the server agent, its communication configurations will be the merging of ``server_configs.comm_configs`` and ``client_configs.comm_configs``. However, ``client_configs.comm_configs`` will also be shared with clients, while ``server_configs.comm_configs`` will not. As we want the clients to be aware of the compressor configurations, we put ``compressor_configs`` under ``client_configs.comm_configs`` to share with the clients during the FL experiment.
