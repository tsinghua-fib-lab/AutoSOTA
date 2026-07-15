Example: Running a Batched MPI Example
=======================================

In this example, we will show how to run FL training with batched MPI communication, i.e., each MPI process represents multiple clients. The example script is available at `examples/mpi/run_batched_mpi.py <https://github.com/APPFL/APPFL/blob/main/examples/mpi/run_batched_mpi.py>`_.

Difference between Non-Batched and Batched MPI
----------------------------------------------

Below shows the difference between the non-batched and batched MPI examples.

.. note::

    The batched MPI example only supports synchronous FL training, i.e., ``scheduler="SyncScheduler"``.

.. code-block:: diff
    :linenos:

    import argparse
    + import numpy as np
    from mpi4py import MPI
    from omegaconf import OmegaConf
    from appfl.agent import ClientAgent, ServerAgent
    from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator

    argparse = argparse.ArgumentParser()
    argparse.add_argument("--server_config", type=str, default="./resources/configs/mnist/server_fedavg.yaml")
    argparse.add_argument("--client_config", type=str, default="./resources/configs/mnist/client_1.yaml")
    + argparse.add_argument("--num_clients", type=int, default=10)
    args = argparse.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    - num_clients = size - 1
    + num_clients = max(args.num_clients, size - 1)
    + # Split the clients into batches for each rank
    + client_batch = [[int(num) for num in array] for array in np.array_split(np.arange(num_clients), size - 1)]

    if rank == 0:
        # Load and set the server configurations
        server_agent_config = OmegaConf.load(args.server_config)
        server_agent_config.server_configs.num_clients = num_clients
        # Create the server agent and communicator
        server_agent = ServerAgent(server_agent_config=server_agent_config)
        server_communicator = MPIServerCommunicator(comm, server_agent, logger=server_agent.logger)
        # Start the server to serve the clients
        server_communicator.serve()
    else:
        # Set client configurations and create client agent
    -   client_agent_config = OmegaConf.load(args.client_config)
    -   client_agent_config.client_id = f'Client{rank}'
    -   client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    -   client_agent_config.data_configs.dataset_kwargs.client_id = rank - 1
    -   client_agent_config.data_configs.dataset_kwargs.visualization = True if rank == 1 else False
    -   client_agent = ClientAgent(client_agent_config=client_agent_config)
    +   client_agents = []
    +   for client_id in client_batch[rank - 1]:
    +       client_agent_config.client_id = f'Client{client_id}'
    +       client_agent_config.data_configs.dataset_kwargs.num_clients = num_clients
    +       client_agent_config.data_configs.dataset_kwargs.client_id = client_id
    +       client_agent_config.data_configs.dataset_kwargs.visualization = True if client_id == 0 else False
    +       client_agents.append(ClientAgent(client_agent_config=client_agent_config))
        # Create the client communicator
    -   client_communicator = MPIClientCommunicator(comm, server_rank=0, client_id=client_agent_config.client_id)
    +   client_communicator = MPIClientCommunicator(comm, server_rank=0, client_ids=[f"Client{client_id}" for client_id in client_batch[rank - 1]])
        # Get and load the general client configurations
        client_config = client_communicator.get_configuration()
    -   client_agent.load_config(client_config)
    +   for client_agent in client_agents:
    +       client_agent.load_config(client_config)
        # Get and load the initial global model
        init_global_model = client_communicator.get_global_model(init_model=True)
    -   client_agent.load_parameters(init_global_model)
    +   for client_agent in client_agents:
    +       client_agent.load_parameters(init_global_model)
        # [Optional] Send the sample size to the server
    -   sample_size = client_agent.get_sample_size()
    -   client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)
    +   client_sample_sizes = {
    +       client_id: {'sample_size': client_agent.get_sample_size()}
    +       for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents)
    +   }
    +   client_communicator.invoke_custom_action(action='set_sample_size', kwargs=client_sample_sizes)
        # Generate data readiness report
        if hasattr(client_config.data_readiness_configs, 'generate_dr_report') and client_config.data_readiness_configs.generate_dr_report:
    -       data_readiness = client_agent.generate_readiness_report(client_config)
    -       client_communicator.invoke_custom_action(action='get_data_readiness_report', **data_readiness)
    +       data_readiness = {
    +           client_id: client_agent.generate_readiness_report(client_config)
    +           for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents)
    +       }
    +       client_communicator.invoke_custom_action(action='get_data_readiness_report', kwargs=data_readiness)
        # Local training and global model update iterations
        while True:
    -       client_agent.train()
    -       local_model = client_agent.get_parameters()
    -       if isinstance(local_model, tuple):
    -           local_model, metadata = local_model[0], local_model[1]
    -       else:
    -           metadata = {}
    -       new_global_model, metadata = client_communicator.update_global_model(local_model, **metadata)
    +       client_local_models = {}
    +       client_metadata = {}
    +       for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents):
    +           client_agent.train()
    +           local_model = client_agent.get_parameters()
    +           if isinstance(local_model, tuple):
    +               local_model, metadata = local_model[0], local_model[1]
    +               client_metadata[client_id] = metadata
    +           client_local_models[client_id] = local_model
    +       new_global_model, metadata = client_communicator.update_global_model(client_local_models, kwargs=client_metadata)
    -       if metadata['status'] == 'DONE':
    +       if all(metadata[client_id]['status'] == 'DONE' for client_id in metadata):
                break
    -       client_agent.load_parameters(new_global_model)
    +       for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents):
    +           client_agent.load_parameters(new_global_model)
        client_communicator.invoke_custom_action(action='close_connection')

Below summarizes the main changes made to the script:

- The script evenly splits the clients into batches for each rank (lines 18-20), and initializes the client agents for each batch (lines 41-47).
- When creating the client MPI communicator for batched MPI, the script passes the client IDs of the batch to the communicator (line 50).
- For the invoked custom actions, the keyword arguments are passed as a dictionary ``kwargs`` with client IDs as keys (lines 64-68, 73-77).
- For updating the global model, the script passes a dictionary of trained local models with client IDs as keys (lines 83-87).
- For the metadata returned from the server, it is a dictionary with client IDs as keys, and a dictionary of metadata as values (line 89).

Running Batched MPI Example
---------------------------

You can run the batched MPI example with the following command to simulate 10 clients with 6 MPI processes, where one process is the server and the rest are clients, so each MPI client process represents two clients.

.. code-block:: bash

    mpiexec -n 6 python ./mpi/run_batched_mpi.py --num_clients 10

You can also run the batched MPI example with the following command to simulate 10 clients with 11 MPI processes, where one process is the server and the rest are clients, so each MPI client process only represents one client.

.. code-block:: bash

    mpiexec -n 11 python ./mpi/run_batched_mpi.py --num_clients 10
    # Note: this is equivalent to running the non-batched MPI example below
    mpiexec -n 11 python ./mpi/run_mpi.py

Extra: Running the Batched MPI Example for Asynchronous FL
----------------------------------------------------------

Though it is not very logical to run batched MPI communication with asynchronous FL training, you can still have each MPI process represent multiple clients running serially and sending updates asynchronously.

Below shows the changes needed in local training part to run the batched MPI example with asynchronous FL training. The example script is available at `examples/mpi/run_batched_mpi.py <https://github.com/APPFL/APPFL/blob/main/examples/mpi/run_batched_mpi_async.py>`_.

.. code-block:: diff
    :linenos:

    # Local training and global model update iterations
    + finish_flag = False
    while True:
    -   client_local_models = {}
    -   client_metadata = {}
    -   for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents):
    -       client_agent.train()
    -       local_model = client_agent.get_parameters()
    -       if isinstance(local_model, tuple):
    -          local_model, metadata = local_model[0], local_model[1]
    -          client_metadata[client_id] = metadata
    -       client_local_models[client_id] = local_model
    -   new_global_model, metadata = client_communicator.update_global_model(client_local_models, kwargs=client_metadata)
    -   if all(metadata[client_id]['status'] == 'DONE' for client_id in metadata):
    -       break
    -   for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents):
    -       client_agent.load_parameters(new_global_model)
    +   for client_id, client_agent in zip([f"Client{client_id}" for client_id in client_batch[rank - 1]], client_agents):
    +       client_agent.train()
    +       local_model = client_agent.get_parameters()
    +       if isinstance(local_model, tuple):
    +           local_model, metadata = local_model
    +       else:
    +           metadata = {}
    +       new_global_model, metadata = client_communicator.update_global_model(local_model, client_id=client_id, **metadata)
    +       if metadata['status'] == 'DONE':
    +           finish_flag = True
    +           break
    +       client_agent.load_parameters(new_global_model)
    +   if finish_flag:
    +       break
    client_communicator.invoke_custom_action(action='close_connection')

The main change made to the script is that: the client MPI process sends ``update_global_model`` request serially for each client in the batch and specify its client ID (line 16).

You can run the batched MPI example with the following command to simulate 10 clients with 6 MPI processes, where one process is the server and the rest are clients, so each MPI client process represents two clients.

.. code-block:: bash

    mpiexec -n 6 python ./mpi/run_batched_mpi_async.py --num_clients 10
