APPFL Client Agent
==================

Functionalities
---------------

APPFL client agent acts on behalf of the FL client to fulfill various tasks related to local training, including

- Load general configurations shared among all clients from the server.
- Load model parameters received from the server.
- Perform the local training job according to the configuration.
- Prepare local model parameters and other optional metadata for communication with the server.
- Other tasks that the client agent needs to perform for certain FL algorithms (e.g. get the size of the local dataset).

Specifically, the current client agent has the following functionalities.

.. note::

    User can also define their functionalities by either inheriting the ``ClientAgent`` class or directly adding new methods to the current client agent. Additionally, if you think your added functionalities are useful for other users, please consider contributing to the APPFL package by submitting a pull request.

.. code-block:: python

    class ClientAgent:
        def __init__(
            self,
            client_agent_config: ClientAgentConfig = ClientAgentConfig()
        ) -> None:
            """
            Initialize the client agent with configurations.
            """

        def load_config(self, config: DictConfig) -> None:
            """
            Load additional configurations provided by the server.
            """

        def get_id(self) -> str:
            """
            Return a unique client id for server to distinguish clients.
            """

        def get_sample_size(self) -> int:
            """
            Return the size of the local dataset.
            """

        def train(self) -> None:
            """
            Train the model locally.
            """

        def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
            """
            Return parameters for communication.
            :return parameters: The parameters to be sent to the server,
                can be of type Dict, OrderedDict, bytes (if compressed), or
                Tuple[Union[Dict, OrderedDict, bytes], Dict] with optional
                metadata in Dict type.
            """

        def load_parameters(self, params) -> None:
            """
            Load parameters from the server.
            """

Configurations
--------------

As shown above, to create a client agent, you need to provide the configurations for the client agent. The configurations for the client agent are defined in the ``appfl.config.ClientAgentConfig`` class, which can be loaded from a YAML file. The following shows an example configuration YAML file for the client agent,

.. note::

    It should be noted that the configurations in client YAML file only contain the configurations that are **specific** for one client. The general configurations shared among all clients are sent from the server.

.. literalinclude:: ../_static/client.yaml
    :language: yaml
    :caption: Client agent configuration YAML file for a specific client.


The above configuration file contains several client-**specific** cnofigurations, such as the device to use and the way to load the private local dataset:

- ``client_id``: This is a unique identifier (among all clients in the current FL experiment) for the client. The server uses this identifier to distinguish between different clients and for logging purposes.
- ``data_configs``: This is the most important component in the client configuration file. It contains the path in client's local machine to the file which defines how to load the local dataset (``dataset_path``), the name of the function in the file to load the dataset (``dataset_name``), and any keyword arguments if needed (``dataset_kwargs``).
- ``train_configs``: This contains the device to use in training and some logging configurations.
- ``comm_configs``: The client may also need to specify some communication settings in order to connect to the server. For example, if the experiment uses gRPC as the communication method, then the client needs to specify the ``server_uri``, ``max_message_size``, and ``use_ssl`` to establish the connection to the server.
