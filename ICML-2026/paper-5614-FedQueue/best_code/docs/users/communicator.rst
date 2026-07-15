APPFL Communicator
==================

The APPFL communicator is used for exchanging

- various types of model data (e.g. parameters, gradients, compressed bytes, etc.)
- metadata such as configurations, control signals, etc.

for different tasks on the server/client agent side to run.

In APPFL, we support the following types of communication protocols:

- :ref:`gRPC: Google Remote Procedure Call`
- :ref:`MPI: Message Passing Interface`
- :ref:`Globus Compute`

gRPC: Google Remote Procedure Call
----------------------------------

gRPC can be used either for simulating federated learning on a single machine or cluster, or for running federated learning on real-world distributed machines. It is composed of two parts:

- gRPC Server Communicator (``appfl.comm.grpc.GRPCServerCommunicator``) which creates a server for listening to incoming requests from clients for various tasks.
- gRPC Client Communicator (``appfl.comm.grpc.GRPCClientCommunicator``) which sends requests to the server for various tasks.

gRPC Server Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

For the server side, the server only needs to create an instance of ``GRPCServerCommunicator`` and call the ``serve`` method (available in ``appfl.comm.grpc``) to start the server. The server will listen to incoming requests from clients for various tasks.

The server can handle the following tasks:

- Get configurations that are shared among all clients via the ``GetConfiguration`` method.
- Get the global model via the ``GetGlobalModel`` method.
- Update the global model with the local model from the client via the ``UpdateGlobalModel`` method.
- Invoke custom action on the server via the ``InvokeCustomAction`` method.

.. note::

    You can add any custom tasks by implementing the corresponding methods in the ``InvokeCustomAction`` class.

.. code:: python

    class GRPCServerCommunicator(GRPCCommunicatorServicer):
        def __init__(
            self,
            server_agent: ServerAgent,
            max_message_size: int = 2 * 1024 * 1024,
            logger: Optional[ServerAgentFileLogger] = None,
        ) -> None:
            """
            Creates a gRPC server communicator.
            :param `server_agent`: `ServerAgent` object
            :param `max_message_size`: Maximum message size in bytes to be sent/received.
                Object size larger than this will be split into multiple messages.
            :param `logger`: A logger object for logging messages
            """

        def GetConfiguration(self, request, context):
            """
            Client requests the FL configurations that are shared among all clients from the server.
            :param: `request.header.client_id`: A unique client ID
            :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
            :return `response.header.status`: Server status
            :return `response.configuration`: YAML serialized FL configurations
            """

        def GetGlobalModel(self, request, context):
            """
            Return the global model to clients. This method is supposed to be called by
            clients to get the initial and final global model. Returns are sent back as a
            stream of messages.
            :param: `request.header.client_id`: A unique client ID
            :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
            :return `response.header.status`: Server status
            :return `response.global_model`: Serialized global model
            """

        def UpdateGlobalModel(self, request_iterator, context):
            """
            Update the global model with the local model from a client. This method will
            return the updated global model to the client as a stream of messages.
            :param: request_iterator: A stream of `DataBuffer` messages - which contains
                serialized request in `UpdateGlobalModelRequest` type.

            If concatenating all messages in `request_iterator` to get a `request`, then
            :param: request.header.client_id: A unique client ID
            :param: request.local_model: Serialized local model
            :param: request.meta_data: YAML serialized metadata dictionary (if needed)
            """

        def InvokeCustomAction(self, request, context):
            """
            This function is the entry point for any custom action that the server agent
            can perform. The server agent should implement the custom action and call this
            function to perform the action.
            :param: `request.header.client_id`: A unique client ID
            :param: `request.action`: A string tag representing the custom action
            :param: `request.meta_data`: YAML serialized metadata dictionary for the custom action (if needed)
            :return `response.header.status`: Server status
            :return `response.meta_data`: YAML serialized metadata dictionary for return values (if needed)
            """

gRPC Client Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

During the federated learning process, the client can communicate to the server by invoking corresponding methods in the ``GRPCClientCommunicator`` class. For example, after a client finish a local training round, it can send the local model to the server for global aggregation by calling the ``update_global_model`` method.

.. note::

    You can add any custom tasks by implementing the corresponding methods in the ``invoke_custom_action`` class. Also make sure that the server has the corresponding handler codes implemented in the ``InvokeCustomAction`` method.

.. code:: python

    class GRPCClientCommunicator:
        def __init__(
            self,
            client_id: Union[str, int],
            *,
            server_uri: str,
            use_ssl: bool = False,
            use_authenticator: bool = False,
            root_certificate: Optional[Union[str, bytes]] = None,
            authenticator: Optional[str] = None,
            authenticator_args: Dict[str, Any] = {},
            max_message_size: int = 2 * 1024 * 1024,
            **kwargs,
        ):
            """
            Create a channel to the server and initialize the gRPC client stub.

            :param client_id: A unique client ID.
            :param server_uri: The URI of the server to connect to.
            :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
            :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
            :param root_certificate: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
            :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
            :param authenticator_args: The arguments to pass to the authenticator.
            :param max_message_size: The maximum message size in bytes.
            """

        def get_configuration(self, **kwargs) -> DictConfig:
            """
            Get the federated learning configurations from the server.
            :param kwargs: additional metadata to be sent to the server
            :return: the federated learning configurations
            """

        def get_global_model(self, **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Get the global model from the server.
            :param kwargs: additional metadata to be sent to the server
            :return: the global model with additional metadata (if any)
            """

        def update_global_model(self, local_model: Union[Dict, OrderedDict, bytes], **kwargs) -> Tuple[Union[Dict, OrderedDict], Dict]:
            """
            Send local model to FL server for global update, and return the new global model.
            :param local_model: the local model to be sent to the server for global aggregation
            :param kwargs: additional metadata to be sent to the server
            :return: the updated global model with additional metadata. Specifically, `meta_data["status"]` is either "RUNNING" or "DONE".
            """

        def invoke_custom_action(self, action: str, **kwargs) -> Dict:
            """
            Invoke a custom action on the server.
            :param action: the action to be invoked
            :param kwargs: additional metadata to be sent to the server
            :return: the response from the server
            """

Example Usage (gRPC)
~~~~~~~~~~~~~~~~~~~~

Below shows an example on how to start a server using `GRPCServerCommunicator`, which waits for incoming requests from clients.

.. literalinclude:: ../../examples/grpc/run_server.py
    :language: python
    :caption: Running Federated Learning with gRPC Server Communicator.

To interact with the server and start an FL experiment, you can start a client using `GRPCClientCommunicator` as shown below.

.. literalinclude:: ../../examples/grpc/run_client.py
    :language: python
    :caption: Running Federated Learning with gRPC Client Communicator.


MPI: Message Passing Interface
------------------------------

MPI can be used for simulating federated learning on a single machine or a cluster of machines. It is composed of two parts:

- MPI Server Communicator (``appfl.comm.mpi.MPIServerCommunicator``) which starts a server to listen to incoming requests from clients for various tasks.
- MPI Client Communicator (``appfl.comm.mpi.MPIClientCommunicator``) which sends requests to the server for various tasks.

MPI Server Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

For the server side, the server only needs to create an instance of ``MPIServerCommunicator`` and call the ``serve`` method to start the server. The server will listen to incoming requests from clients for various tasks.

The server can handle the following tasks:

- Get configurations that are shared among all clients via the ``_get_configuration`` method.
- Get the global model via the ``_get_global_model`` method.
- Update the global model with the local model from the client via the ``_update_global_model`` method.
- Invoke custom action on the server via the ``_invoke_custom_action`` method.

.. note::

    The server will automatically stop itself after reaching the specified ``num_global_epochs``.

.. note::

    You can add any custom tasks by implementing the corresponding methods in the ``_invoke_custom_action`` class.

.. code:: python

    class MPIServerCommunicator:
        def __init__(
            self,
            comm,
            server_agent: ServerAgent,
            logger: Optional[ServerAgentFileLogger] = None,
        ) -> None:
            """
            Create an MPI server communicator.
            :param `comm`: MPI communicator object
            :param `server_agent`: `ServerAgent` object
            :param `logger`: A logger object for logging messages
            """

        def serve(self):
            """
            Start the server to serve the clients.
            """

        def _get_configuration(
            self, client_rank: int, request: MPITaskRequest
        ) -> MPITaskResponse:
            """
            Client requests the FL configurations that are shared among all clients from the server.

            :param: `client_rank`: The rank of the client in MPI
            :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.meta_data`: YAML serialized FL configurations
            """

        def _get_global_model(
            self, client_rank: int, request: MPITaskRequest
        ) -> Optional[MPITaskResponse]:
            """
            Return the global model to clients. This method is supposed to provide clients with
            the initial and final global model.

            :param: `client_rank`: The rank of the client(s) in MPI
            :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)

                - `meta_data['_client_ids']`: A list of client ids to get the global model for batched clients
                - `meta_data['init_model']`: Whether to get the initial global model or not
            :return `response.status`: Server status
            :return `response.payload`: Serialized global model
            :return `response.meta_data`: YAML serialized metadata dictionary (if needed)
            """

        def _update_global_model(
            self, client_rank: int, request: MPITaskRequest
        ) -> Optional[MPITaskResponse]:
            """
            Update the global model with the local model from the client,
            and return the updated global model to the client.

            :param: `client_rank`: The rank of the client in MPI
            :param: `request.payload`: Serialized local model
            :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.payload`: Serialized updated global model
            :return `response.meta_data`: YAML serialized metadata dictionary (if needed)
            """

        def _invoke_custom_action(
            self,
            client_rank: int,
            request: MPITaskRequest,
        ) -> Optional[MPITaskResponse]:
            """
            Invoke custom action on the server.
            :param: `client_rank`: The rank of the client in MPI
            :param: `request.meta_data`: YAML serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.meta_data`: YAML serialized metadata dictionary (if needed)
            """

MPI Client Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

During the federated learning process, the client can communicates to the server by invoking corresponding methods in the ``MPIClientCommunicator`` class. For example, after a client finish a local training round, it can send the local model to the server for global aggregation by calling the ``update_global_model`` method.

.. note::

    You can add any custom tasks by implementing the corresponding methods in the ``invoke_custom_action`` class. Also make sure that the server has the corresponding handler codes implemented in the ``_invoke_custom_action`` method.

.. code:: python

    class MPIClientCommunicator:
        """
        MPI client communicator for federated learning.

        :param comm: the MPI communicator from mpi4py
        :param server_rank: the rank of the server in the MPI communicator
        :param client_id: [optional] an optional client ID for one client for logging purposes, mutually exclusive with client_ids
        :param client_ids: [optional] a list of client IDs for a batched clients,
            this is only required when the MPI process represents multiple clients
        """

        def get_configuration(self, **kwargs) -> DictConfig:
            """
            Get the federated learning configurations from the server.

            :param kwargs: additional metadata to be sent to the server
            :return: the federated learning configurations
            """

        def get_global_model(
            self, **kwargs
        ) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Get the global model from the server.

            :param kwargs: additional metadata to be sent to the server
            :return: the global model with additional metadata (if any)
            """

        def update_global_model(
            self,
            local_model: Union[Dict, OrderedDict, bytes],
            client_id: Optional[Union[str, int]] = None,
            **kwargs,
        ) -> Tuple[Union[Dict, OrderedDict], Dict]:
            """
            Send local model(s) to the FL server for global update, and return the new global model.

            :param local_model: the local model to be sent to the server for global aggregation

                - `local_model` can be a single model if one MPI process has only one client or one MPI process
                    has multiple clients but the user wants to send one model at a time
                - `local_model` can be a dictionary of multiple models as well if one MPI process has multiple clients
                    and the user wants to send all models
            :param client_id (optional): the client ID for the local model. It is only required when the MPI process has multiple clients
                and the user only wants to send one model at a time.
            :param kwargs (optional): additional metadata to be sent to the server. When sending local models for multiple clients,
                use the client ID as the key and the metadata as the value, e.g.,

            ```
            update_global_model(
                local_model=...,
                kwargs = {
                    client_id1: {key1: value1, key2: value2},
                    client_id2: {key1: value1, key2: value2},
                }
            )
            ```
            :return model: the updated global model

                - Note: the global model is only one model even if multiple local models are sent, which means that
                the server should have synchronous aggregation. If asynchronous aggregation is needed, the user should
                pass the local models one by one.

            :return meta_data: additional metadata from the server. When updating local models for multiple clients, the response will
                be a dictionary with the client ID as the key and the response as the value, e.g.,
            ```
            {
                client_id1: {ret1: value1, ret2: value2},
                client_id2: {ret1: value1, ret2: value2},
            }
            ```
            """

        def invoke_custom_action(
            self, action: str, client_id: Optional[Union[str, int]] = None, **kwargs
        ) -> Dict:
            """
            Invoke a custom action on the server.

            :param action: the action to be invoked
            :param client_id (optional): the client ID for the action. It is only required when the MPI process has multiple clients
                and the action is specific to a client instead of all clients.
            :param kwargs (optional): additional metadata to be sent to the server. When invoking custom action for multiple clients,
                use the client ID as the key and the metadata as the value, e.g.,
            ```
            invoke_custom_action(
                action=...,
                kwargs = {
                    client_id1: {key1: value1, key2: value2},
                    client_id2: {key1: value1, key2: value2},
                }
            )
            ```
            :return: the response from the server (if any). When invoking custom action for multiple clients, the response will
                be a dictionary with the client ID as the key and the response as the value, e.g.,
            ```
            {
                client_id1: {ret1: value1, ret2: value2},
                client_id2: {ret1: value1, ret2: value2},
            }
            ```
            """

Example Usage (MPI)
~~~~~~~~~~~~~~~~~~~

Here is an example of how to use the MPI communicator in APPFL to start FL experiments.


.. literalinclude:: ../../examples/mpi/run_mpi.py
    :language: python
    :caption: Running Federated Learning with MPI Communicator.

Globus Compute
--------------

`Globus Compute <https://globus-compute.readthedocs.io/>`_ is a distributed function as a service platform, which can be used for running federated learning on real-world distributed machines. It can turn each client into an endpoint which can be patched with remote functions on the server side to run federated learning tasks. It is composed of two parts:

- Globus Compute Server Communicator (``appfl.comm.globus_compute.GlobusComputeServerCommunicator``), which can send task to the client endpoints and receive the results.
- Globus Compute Client Entry Point (``appfl.comm.globus_compute.GlobusComputeClientCommunicator.globus_compute_client_entry_point``), which is the entry point for the client to execute the task and send the results back to the server.

Globus Compute Server Communicator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Globus Compute is a server-driven communication protocol, where the server sends tasks to the clients and receives the results to control the FL process. It is composed of the following methods to send tasks to the clients and receive the results:

.. code:: python

    class GlobusComputeServerCommunicator:
        """
        Communicator used by the federated learning server which plans to use Globus Compute
        for orchestrating the federated learning experiments.

        Globus Compute is a distributed function-as-a-service platform that allows users to run
        functions on specified remote endpoints. For more details, check the Globus Compute SDK
        documentation at https://globus-compute.readthedocs.io/en/latest/endpoints.html.

        :param `gcc`: Globus Compute client object
        :param `server_agent_config`: The server agent configuration
        :param `client_agent_configs`: A list of client agent configurations.
        :param [Optional] `logger`: Optional logger object.
        """

        def send_task_to_all_clients(
            self,
            task_name: str,
            *,
            model: Optional[Union[Dict, OrderedDict, bytes]] = None,
            metadata: Union[Dict, List[Dict]] = {},
            need_model_response: bool = False,
        ):
            """
            Send a specific task to all clients.
            :param `task_name`: Name of the task to be executed on the clients
            :param [Optional] `model`: Model to be sent to the clients
            :param [Optional] `metadata`: Additional metadata to be sent to the clients
            :param `need_model_response`: Whether the task requires a model response from the clients
                If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
            """
            pass

        def send_task_to_one_client(
            self,
            client_endpoint_id: str,
            task_name: str,
            *,
            model: Optional[Union[Dict, OrderedDict, bytes]] = None,
            metadata: Optional[Dict] = {},
            need_model_response: bool = False,
        ):
            """
            Send a specific task to one specific client endpoint.
            :param `client_endpoint_id`: The client endpoint id to which the task is sent.
            :param `task_name`: Name of the task to be executed on the clients
            :param [Optional] `model`: Model to be sent to the clients
            :param [Optional] `metadata`: Additional metadata to be sent to the clients
            :param `need_model_response`: Whether the task requires a model response from the clients
                If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
            """
            pass

        def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
            """
            Receive task results from all clients that have running tasks.
            :return `client_results`: A dictionary containing the results from all clients - Dict[client_endpoint_id, client_model]
            :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_endpoint_id, client_metadata]
            """
            pass

        def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
            """
            Receive task results from the first client that finishes the task.
            :return `client_endpoint_id`: The client endpoint id from which the result is received.
            :return `client_model`: The model returned from the client
            :return `client_metadata`: The metadata returned from the client
            """
            pass

        def shutdown_all_clients(self):
            """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
            pass

        def cancel_all_tasks(self):
            """Cancel all on-the-fly client tasks."""
            pass

Globus Compute Client Entry Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all tasks the server sent to the client, the tasks should be implemented in the client entry point function, which is the entry point for the client to execute the task and send the results back to the server. Below shows the current client entry point function which only supports the ``get_sample_size`` and ``train`` tasks. User can freely add more tasks by implementing the corresponding functions in the client entry point.

.. code:: python

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
        from appfl.agent import ClientAgent
        from appfl.comm.globus_compute.utils.client_utils import load_global_model, send_local_model

        client_agent = ClientAgent(client_agent_config=client_agent_config)
        if model is not None:
            model = load_global_model(client_agent.client_agent_config, model)
            client_agent.load_parameters(model)

        if task_name == "get_sample_size":
            return None, {
                "sample_size": client_agent.get_sample_size()
            }

        elif task_name == "train":
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, meta_data_local = local_model
            else:
                meta_data_local = {}
            local_model = send_local_model(
                client_agent.client_agent_config,
                local_model,
                meta_data["local_model_key"],
                meta_data["local_model_url"],
            )
            return local_model, meta_data_local
