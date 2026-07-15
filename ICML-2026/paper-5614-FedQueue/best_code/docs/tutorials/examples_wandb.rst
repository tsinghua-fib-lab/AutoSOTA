Example: Using Weights & Biases
===============================

.. raw:: html

    <div style="display: flex; justify-content: center; width: 90%; margin: auto; margin-top: 35px; margin-bottom: 35px;">
        <div style="display: inline-block; ;">
            <img src="../_static/appfl-wandb.png" alt="wandb">
        </div>
    </div>

`Weights & Biases (wandb) <https://wandb.ai>`_ is a tool that helps you track your experiments. It's a great tool to use with APPFL. In this example, we'll show you how to use Weights & Biases with APPFL.

wandb Login
-----------

First, make sure you have a wandb account and login using the following command. For more information on how to login to wandb, please refer to the `wandb documentation <https://docs.wandb.ai/quickstart/>`_.

.. code-block:: bash

    wandb login

Client Configurations for wandb
-------------------------------

Each client and specify configurations for wandb in their client configuration yaml files. Here is an example of a client configuration file with wandb configurations.

.. code-block:: yaml

    client_id: Client1

    data_configs:
        ...
    ...

    wandb_configs:
      enable_wandb: True
      entity: <your_wandb_entity>
      project: appfl
      exp_name: appfl-mnist

The explanation of each configuration is as follows:

- ``enable_wandb``: A boolean value to enable or disable wandb.
- ``entity``: The entity name of the wandb project. If not specified, the default entity will be used.
- ``project``: The project name of the wandb project. If not specified, the default project will be used.
- ``exp_name``: The experiment name of the wandb project. If not specified, the default experiment name will be ``appfl``.

Example Run
-----------

To run a serial example with wandb, you can use the following command. Make sure to change the corresponding ``wandb_configs`` in the client configuration yaml files for your wandb account.s

.. code-block:: bash

    python ./serial/run_serial.py --num_clients 3 \
        --server_config ./resources/configs/mnist/server_fedavg.yaml \
        --client_config ./resources/configs/mnist/client_1_wandb.yaml

To run an MPI example with wandb, you can use the following command. Make sure to change the corresponding ``wandb_configs`` in the client configuration yaml files for your wandb account.

.. code-block:: bash

    mpiexec -n 3 python ./mpi/run_mpi.py --server_config ./resources/configs/mnist/server_fedcompass.yaml \
        --client_config ./resources/configs/mnist/client_1_wandb.yaml
