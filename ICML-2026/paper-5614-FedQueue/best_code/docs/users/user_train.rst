Training PPFL
=============

To run PPFL with decentralized data on multiple machines, we use `gRPC <https://grpc.io/docs/>`_ that  allows clients from different platforms to seamlessly connect to the server for federated learning. This contrasts with MPI where all clients and servers should reside in the same cluster.

Launch a gRPC server
--------------------

The server needs to load configuration from a configuration file, then we can create a gRPC FL serverto listen to request from clients by calling the ``serve()`` function from ``appfl.comm.grpc`` module.

.. code-block:: python

    from omegaconf import OmegaConf
    from appfl.agent import ServerAgent
    from appfl.comm.grpc import GRPCServerCommunicator, serve

    server_agent_config = OmegaConf.load('<path_to_config_file>.yaml')
    server_agent = ServerAgent(server_agent_config=server_agent_config)

    communicator = GRPCServerCommunicator(
        server_agent,
        logger=server_agent.logger,
        **server_agent_config.comm_configs.grpc_configs,
    )

    serve(
        communicator,
        **server_agent_config.server_configs.comm_configs.grpc_configs,
    )

Launch a gRPC client
--------------------

The client also loads configuration from a configuration file, and then it starts the FL process by sending different requests to the server.

.. code-block:: python

    from omegaconf import OmegaConf
    from appfl.agent import ClientAgent
    from appfl.comm.grpc import GRPCClientCommunicator

    client_agent_config = OmegaConf.load('<path_to_config_file>.yaml')

    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = GRPCClientCommunicator(
        client_id = client_agent.get_id(),
        **client_agent_config.comm_configs.grpc_configs,
    )

    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    # Send the number of local data to the server
    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)

    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        new_global_model, metadata = client_communicator.update_global_model(local_model, **metadata)
        if metadata['status'] == 'DONE':
            break
        if 'local_steps' in metadata:
            client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
        client_agent.load_parameters(new_global_model)
