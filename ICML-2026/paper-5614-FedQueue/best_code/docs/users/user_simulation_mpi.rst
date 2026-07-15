Simulating PPFL (MPI)
=====================

In this section, we describe how to simulate PPFL on a single machine or cluster by having the server and each client run on different MPI processes. It can be used for simulating both synchronous and asynchronous FL algorithms.

.. note::

    To run the MPI simulation, you need to use several MPI processes by using ``mpiexec`` command. For example, to run 4 MPI processes, you can use the following command:

    .. code-block:: bash

        mpiexec -n 4 python mpi_code.py

First, user needs to load configuration files for the client and server agents. The total number of clients is equal to the total number of MPI processes minus one (as one process is used for the server), and then make necessary changes to make the configurations compatible with ``num_clients``. With the configuration, we can create the server and client agents.

.. code-block:: python

    from mpi4py import MPI
    from omegaconf import OmegaConf
    from appfl.agent import ClientAgent, ServerAgent

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_clients = size - 1

    if rank == 0:
        # Load and update server configuration
        server_agent_config = OmegaConf.load("<path_to_server_config>.yaml")
        server_agent_config.server_configs.num_clients = num_clients
        # Create the server agent
        server_agent = ServerAgent(server_agent_config=server_agent_config)
    else:
        # Load and set client configuration
        client_agent_config = OmegaConf.load("<path_to_client_config>.yaml")
        client_agent_config.client_id = f'Client{rank}'
        client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
        client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
        client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 1 else False
        # Create the client agent
        client_agent = ClientAgent(client_agent_config=client_agent_config)

Then for the FL server, we can create an MPI communicator to serve the requests from the clients using the ``serve`` method.

.. code-block:: python

    from appfl.comm.mpi import MPIServerCommunicator
    if rank == 0:
        server_communicator = MPIServerCommunicator(
            comm,
            server_agent,
            logger=server_agent.logger
        )
        server_communicator.serve()

For the clients, we can start the FL training process by doing the following process:

- Create an MPI communicator for the client.
- Get and load the shared client configurations from the server (such as trainer and model architecture).
- Get and load the initial global model.
- Start the training process by calling the ``client_agent.train()`` method, and then send the updated model (``client_agent.get_parameters``) to the server until the end of the training process.

.. code-block:: python

    from appfl.comm.mpi import MPIClientCommunicator
    if rank != 0:
        client_communicator = MPIClientCommunicator(comm, server_rank=0)
        # Load the configurations and initial global model
        client_config = client_communicator.get_configuration()
        client_agent.load_config(client_config)
        init_global_model = client_communicator.get_global_model(init_model=True)
        client_agent.load_parameters(init_global_model)
        # Send the sample size to the server
        sample_size = client_agent.get_sample_size()
        client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)
        # Local training and global model update iterations
        while True:
            client_agent.train()
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, meta_data_local = local_model[0], local_model[1]
            else:
                meta_data_local = {}
            new_global_model, metadata = client_communicator.update_global_model(local_model, **meta_data_local)
            if metadata['status'] == 'DONE':
                break
            if 'local_steps' in metadata:
                client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
            client_agent.load_parameters(new_global_model)
