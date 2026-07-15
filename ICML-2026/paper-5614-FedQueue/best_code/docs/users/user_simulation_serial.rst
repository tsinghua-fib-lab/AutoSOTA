Simulating PPFL (Serial)
========================

In this section, we describe how to simulate PPFL on a single machine by having different FL clients running serially.

.. note::

    It should be noted that serial simulation can only be used for synchronous FL algorithms.

First, user needs to load configuration files for the client and server agents and specifies the number of clients to simulate.

.. code-block:: python

    from omegaconf import OmegaConf

    num_clients = 10
    server_agent_config = OmegaConf.load(
        '<your_example_path>/configs/mnist/server_fedavg.yaml'
    )
    client_agent_configs = [
        OmegaConf.load('<your_example_path>/configs/mnist/client_1.yaml')
        for _ in range(args.num_clients)
    ]

Then, user needs to change certain fields in the server agent configuration to make it compatible with ``num_clients``. User also needs to change the client agent configurations for different clients.

.. code-block:: python

    # Update server agent configuration
    server_agent_config.server_configs.num_clients = args.num_clients

    # Update client agent configurations
    for i in range(args.num_clients):
        client_agent_configs[i].client_id = f'Client{i+1}'
        client_agent_configs[i].data_configs.dataset_kwargs.num_clients = args.num_clients
        client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
        client_agent_configs[i].data_configs.dataset_kwargs.visualization = True if i == 0 else False

The user then creates server agent and client agents using the configurations.

.. code-block:: python

    from appfl.agent import ClientAgent, ServerAgent

    server_agent = ServerAgent(server_agent_config=server_agent_config)
    client_agents = [
        ClientAgent(client_agent_config=client_agent_configs[i])
        for i in range(args.num_clients)
    ]

After creating the agents, the user can start the FL process by first having the server to provide general client configurations to all the clients using ``server_agent.get_client_configs()`` function. The clients load the configurations using ``client_agent.load_config()`` function.

Then, the clients get the initial global model from the server using ``server_agent.get_parameters()`` function. It should be noted that we set ``serial_run=True`` in the ``get_parameters()`` function to tell the server that the clients will run serially and it should not wait for all clients to the call this function before sending the global model, avoiding blocking the FL process. The clients load the global model using ``client_agent.load_parameters()`` function.

.. code-block:: python

    # Get additional client configurations from the server
    client_config_from_server = server_agent.get_client_configs()
    for client_agent in client_agents:
        client_agent.load_config(client_config_from_server)

    # Load initial global model from the server
    init_global_model = server_agent.get_parameters(serial_run=True)
    for client_agent in client_agents:
        client_agent.load_parameters(init_global_model)

.. note::

    Does it look a bit confusing that the server sends client configurations to all the clients? This is because, in FL, we usually want certain client-side configurations to be the same among all the clients, for example, the local trainer and the ML model architecture. So it becomes more convenient to first specify all those configurations on the server side to ensure uniformity, and then send those configurations to all clients at the beginning of the FL experiment.


Optionally, the clients can send their number of local training data to the server (which could be useful is the server needs to do weighted averaging), and the server can set the number of local data for each client using ``server_agent.set_sample_size()`` function.

.. code-block:: python

    # [Optional] Set number of local data to the server
    for i in range(args.num_clients):
        sample_size = client_agents[i].get_sample_size()
        server_agent.set_sample_size(
            client_id=client_agents[i].get_id(),
            sample_size=sample_size
        )

After the above initializations, the user can start the FL training loop.

- ``server_agent.training_finished()`` function returns ``True`` if the training is finished, i.e., meeting the stopping criteria.
- ``client_agent.train()`` function is used to perform local training on the client side.
- ``client_agent.get_parameters()`` function is used to get the local model parameters from the client side, which can be model state dictionary, model gradients, compressed model, etc, depending on the training and compressor settings.
- ``server_agent.global_update()`` is used to take the local model from one client, and return the updated global model whenever it is ready. However, for synchronous FL, the server has to receive local models one by one from all clients before updating the global model. Therefore, to avoid blocking the FL process, the ``blocking`` argument is set to ``False``, and the function returns a ``Future`` object that will be resolved when the server receives local models from all clients.
- When ``server_agent.global_update()`` gets called ``num_clients`` times, all the ``Future`` objects will be resolved, and the global model will be updated. The clients can then load the new global model using ``client_agent.load_parameters()`` function.

.. code-block:: python

    while not server_agent.training_finished():
        new_global_models = []
        for client_agent in client_agents:
            # Client local training
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            # "Send" local model to server and get a Future object for the new global model
            # The Future object will be resolved when the server receives local models from all clients
            new_global_model_future = server_agent.global_update(
                client_id=client_agent.get_id(),
                local_model=local_model,
                blocking=False,
                **metadata
            )
            new_global_models.append(new_global_model_future)
        # Load the new global model from the server
        for client_agent, new_global_model_future in zip(client_agents, new_global_models):
            client_agent.load_parameters(new_global_model_future.result())
