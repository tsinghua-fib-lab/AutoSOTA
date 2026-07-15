Example: Finetune a Vision Transformer model
============================================

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/m4rdOub2Y_o?si=HvMnsPyTfQGyx7TQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

``APPFL`` aims to make the transition from centralized to federated learning (FL) as seamless as possible, and this tutorial will demonstrate how to finetune a Vision Transformer (ViT) model in federated settings using the ``APPFL`` package.

Centralized learning
--------------------

In centralized learning, to train a machine learning model, we need a "trainer" that trains the model on a training dataset and evaluates it using an evaluation dataset. The key components of this process are the following:

- Model: A machine learning model that we want to train.
- Datasets: Datasets that contain the training and evaluation data.
- Trainer: Algorithm that trains the model on the training dataset and evaluates it on the evaluation dataset, more specifically, its key components are:

    (1) Loss function for updating the model parameters.
    (2) Optimizer and its hyperparameters (e.g., learning rate, momentum, etc.)
    (3) Metric function that measures the performance of the model.
    (4) Other hyperparameters (e.g., batch size, number of epochs/steps, etc.)

From centralized to federated learning
--------------------------------------

To move from centralized learning to federated learning, the following additional components are needed

- Exchanged parameters: What parameters are exchanged between the server and clients for aggregation purposes.
- Aggregation algorithms: How the parameters are aggregated.
- Other hyperparameters (e.g., number of clients, number of communication rounds, etc.)

In addition, we need to consider how to efficiently configure the distributed training process. In ``APPFL``, we choose to use a server configuration YAML file to specify necessary server-specific configurations (e.g., aggregation algorithm, number of communication rounds, number of clients, etc.) as well as general client configurations that should be the same for all clients (e.g., model architecture, trainer, loss function, metric function, optimizer and its hyperparameters, batch size, number of local epochs/steps, etc.). All these general client configurations will be shared with all clients at the beginning of the training process.

As for clients, in addition to the configurations shared from the server, each client should have its own configuration YAML file to specify client-specific configurations (e.g., functions for loading local private datasets, device, logging settings, etc.).

FL server configurations
------------------------

Server directory structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Below shows the directory structure for the FL server.

.. code-block:: text

    appfl_vit_finetuning_server
    ├── resources
    │   ├── vit.py                  # Model architecture
    │   ├── metric.py               # Metric function
    │   └── vit_ft_trainer.py       # Trainer
    ├── config.yaml                 # Server configuration file
    └── run_server.py               # Server launching script

Now let's take a look at each file.

Model architecture, metric function, and trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``resources/vit.py`` file contains a function that defines the ViT model architecture and freezes all layers except the last heads layer.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_server/resources/vit.py
    :language: python
    :caption: resources/vit.py - ViT model architecture

The ``resources/metric.py`` file contains the metric function that computes the accuracy of the model outputs.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_server/resources/metric.py
    :language: python
    :caption: resources/metric.py - Metric function

The ``resources/vit_ft_trainer.py`` file defines a trainer class for fine-tuning the ViT model. Specifically, it inherits the ``VanillaTrainer`` class from the ``appfl.algorithm.trainer`` module and overrides the ``get_parameters`` and ``load_parametes`` methods for only exchanging the heads layer parameters of the ViT model between the server and clients.

.. note::

    The ``VanillaTrainer`` is a trainer class that trains a model using the specified optimizer and loss function for several epochs or steps (i.e. batches), evaluates it using the given metric function, and finally returns the whole set of model parameters for aggregation. It is a good starting point for building your own trainer class. For this fine-tuning example, we only need to override the ``get_parameters`` and ``load_parameters`` methods to exchange only the heads layer parameters of the ViT model.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_server/resources/vit_ft_trainer.py
    :language: python
    :caption: resources/vit_ft_trainer.py - Trainer

Server configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``config.yaml`` is the server YAML configuration file, which contains both general client configurations (``client_configs``) and server-specific configurations (``server_configs``).

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_server/config.yaml
    :language: yaml
    :caption: config.yaml - Server configuration file

Here comes the line-by-line explanation of ``client_configs`` part:

- ``train_configs``: It contains all necessary configurations related to the local trainer.
- ``train_configs.trainer``: The name of the trainer class you want to use.
- ``train_configs.trainer_path``: The path to the file that defines the trainer class [Note: there is no need to specify this if you are using the ``VanillaTrainer`` class provided by the ``APPFL`` package].
- ``train_configs.mode``: The mode of training, either ``epoch`` or ``step``, where ``epoch`` means, in each local training round, training the model for a fixed number of epochs, and ``step`` means training the model for a fixed number of steps.
- ``train_configs.num_local_steps``: The number of local steps for each client if ``mode="step"``. [Note: Use ``num_local_epochs`` if you set ``mode="epoch"``].
- ``train_configs.optim``: The optimizer name available in the ``torch.optim`` module to use for training.
- ``train_configs.optim_args``: The hyperparameters of the optimizer.
- ``train_configs.loss_fn``: The loss function name available in the ``torch.nn`` module to use for training. [Note: You can also use a custom loss function, refer `here <../users/user_loss.html>`_ for instructions].
- ``train_configs.do_validation``: Whether the clients should perform validation after each local training round.
- ``train_configs.do_pre_validation``: Whether the clients should perform validation before each local training round (i.e., evaluate the received model parameters from the server).
- ``train_configs.metric_path``: The path to the file that defines the metric function [Note: see `here <../users/user_metric.html>`_ for instructions on defining metric functions].
- ``train_configs.metric_name``: The name of the metric function you want to use.
- ``train_configs.train_batch_size``: The batch size for training.
- ``train_configs.val_batch_size``: The batch size for validation.
- ``train_configs.train_data_shuffle``: Whether to shuffle the training data.
- ``train_configs.val_data_shuffle``: Whether to shuffle the validation data.
- ``model_configs``: It contains necessary information to load the model from the definition file - ``model_path`` is the absolute/relative path to the model definition file, ``model_name`` is the name of the model definition function. [Note: You can also load the model from a class definition. For more information, refer `here <../users/user_model.html>`_].

Here comes the line-by-line explanation of ``server_configs`` part:

- ``server_configs.aggregator``: The name of the aggregation algorithm provided by ``APPFL`` you want to use. Please refer to `here <../users/aggregator.html#available-aggregators>`_ for the list of provided aggregators. This can also be a custom aggregation algorithm, in which case you need to provide the path to the file that defines the custom aggregation algorithm in the ``aggregator_path`` field.
- ``server_configs.aggregator_kwargs``: The hyperparameters of the aggregation algorithm. In this example, ``client_weights_mode='equal'`` means that all clients have equal weights in the aggregation process, while ``client_weights_mode='sample_size'`` means that the weights of the clients are proportional to the number of samples they have.
- ``server_configs.scheduler``: The name of the scheduling algorithm provided by ``APPFL`` you want to use. As FedAvg is a synchronous algorithm we set it to be ``SyncScheduler`` here [Please refer to `here <../users/scheduler.html>`_ for the list of provided schedulers].
- ``server_configs.scheduler_kwargs``: The hyperparameters of the scheduling algorithm.  ``num_clients`` tells the scheduler the total number of clients in the training process, and ``same_init_model=True`` ensures that all clients start with the same initial model parameters.
- ``server_configs.device``: The device on which the server should run.
- ``server_configs.num_global_epochs``: The number of FL global communication epochs.
- ``server_configs.logging_output_dirname``: The directory name where the server logs will be saved.
- ``server_configs.logging_output_filename``: The filename where the server logs will be saved.
- ``comm_configs.grpc_configs``: It contains necessary configurations for the gRPC communication process - ``server_uri`` is the URI and port number where the server will be running,  ``max_message_size`` is the maximum size of each message, if the message size exceeds this value, the message will be automatically split into smaller messages, ``use_ssl`` is a boolean value that determines whether to use SSL for communication.

.. note::

    To enable SSL communication, please check this `tutorial <https://appfl.ai/en/latest/tutorials/examples_ssl.html>`_ for more details on how to generate SSL certificates for securing the gRPC connections.

Server launch script
~~~~~~~~~~~~~~~~~~~~

Below is the server launch script that reads the server configuration file, initializes the server agent, creates a gRPC server communicator, and finally starts serving.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_server/run_server.py
    :language: python
    :caption: run_server.py - Server launch script

User can run the server by executing the following command in a terminal:

.. code-block:: bash

    python run_server.py

FL client configurations
------------------------

Client directory structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Below shows the directory structure for the FL client.

.. code-block:: text

    appfl_vit_finetuning_client
    ├── resources
    │   └── vit_fake_dataset.py      # Dataset loader
    ├── config.yaml                  # Client configuration file
    └── run_client.py                # Client launching script

Now let's take a look at each file.

Dataset loader
~~~~~~~~~~~~~~

The ``resources/vit_fake_dataset.py`` file contains a function that generates a fake dataset for the client. In this example, we generate a fake dataset using the ``torch.utils.data.Dataset`` class which randomly returns a 3x224x224 tensor input image and a binary label for each data sample.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_client/resources/vit_fake_dataset.py
    :language: python
    :caption: resources/vit_fake_dataset.py - Dataset loader

Client configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``config.yaml`` is the client YAML configuration file, which contains client-specific configurations, such as the information of the dataset loader function, device, logging settings, etc.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_client/config.yaml
    :language: yaml
    :caption: config.yaml - Client configuration file

Client launch script
~~~~~~~~~~~~~~~~~~~~

Below is the client launch script. It reads the client configuration file to initialize a client agent and a gRPC client communicator. It then employs the client communicator sends various types of requests to the launched server to do federated learning:

(1) The client first uses the communicator to request general client configurations from the server and loads them.
(2) It then gets the initial global model parameters from the server.
(3) It then starts the (local training + global aggregation) loop until receiving a "DONE" status flag from the server.
(4) Finally, it sends a `close_connection` action to the server to close the connection.

.. literalinclude:: ./examples/vit_ft/appfl_vit_finetuning_client/run_client.py
    :language: python
    :caption: run_client.py - Client launch script

User can run the client by executing the following command:

.. code-block:: bash

    python run_client.py

.. note::

    As in the provided server configuration, the ``num_clients`` is set to 2, you need to run the client script twice in two separate terminals.

Result Logs
-----------

After running ``run_server.py`` in one terminal, and ``run_client.py`` in two separate terminals, you should see the following output in the server terminal:

.. code-block:: text

    [2024-09-01 14:58:20,081 INFO server]: Logging to ./output/result_Server_2024-09-01-14:58:20.txt
    [2024-09-01 14:58:20,081 INFO server]: Setting seed value to 42
    [2024-09-01 14:58:23,892 INFO server]: Received GetConfiguration request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:24,671 INFO server]: Received GetGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:27,148 INFO server]: Received GetConfiguration request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:27,899 INFO server]: Received GetGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:41,964 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:42,087 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:44,349 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:44,421 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:46,745 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:46,796 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:49,228 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:49,269 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:51,687 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:51,714 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:54,159 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:54,209 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:56,651 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:58:56,731 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:59,286 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:58:59,309 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:59:01,832 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:59:01,943 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:59:04,503 INFO server]: Received UpdateGlobalModel request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    [2024-09-01 14:59:04,583 INFO server]: Received UpdateGlobalModel request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:59:04,585 INFO server]: Received InvokeCustomAction close_connection request from client 12e7104d-eeb9-4f22-a421-d4b4f8cdaa91
    [2024-09-01 14:59:04,585 INFO server]: Received InvokeCustomAction close_connection request from client 0b5f9d48-10d3-4398-9e7b-485886399191
    Terminating the server ...

And the following output in the client terminals:

.. code-block:: text

    [2024-09-01 14:58:27,005 INFO Client1]: Logging to ./output/result_Client1_2024-09-01-14:58:27.txt
    [2024-09-01 14:58:39,355 INFO Client1]:      Round   Pre Val?       Time Train Loss Train Accuracy   Val Loss Val Accuracy
    [2024-09-01 14:58:40,338 INFO Client1]:          0          Y                                          0.7158      30.0000
    [2024-09-01 14:58:41,871 INFO Client1]:          0          N     1.5323     0.0862        50.0000     0.6990      50.0000
    [2024-09-01 14:58:42,826 INFO Client1]:          1          Y                                          0.6316      70.0000
    [2024-09-01 14:58:44,299 INFO Client1]:          1          N     1.4725     0.0704        60.0000     1.4802      30.0000
    [2024-09-01 14:58:45,161 INFO Client1]:          2          Y                                          0.9973      30.0000
    [2024-09-01 14:58:46,721 INFO Client1]:          2          N     1.5584     0.0725        70.0000     1.0347      30.0000
    [2024-09-01 14:58:47,600 INFO Client1]:          3          Y                                          1.3194      30.0000
    [2024-09-01 14:58:49,204 INFO Client1]:          3          N     1.6029     0.1010        30.0000     0.6233      70.0000
    [2024-09-01 14:58:50,091 INFO Client1]:          4          Y                                          0.6168      70.0000
    [2024-09-01 14:58:51,694 INFO Client1]:          4          N     1.6034     0.0727        60.0000     0.9816      30.0000
    [2024-09-01 14:58:52,598 INFO Client1]:          5          Y                                          0.8828      30.0000
    [2024-09-01 14:58:54,189 INFO Client1]:          5          N     1.5906     0.0759        30.0000     0.7135      30.0000
    [2024-09-01 14:58:55,048 INFO Client1]:          6          Y                                          0.6543      70.0000
    [2024-09-01 14:58:56,708 INFO Client1]:          6          N     1.6591     0.0833        40.0000     0.7175      30.0000
    [2024-09-01 14:58:57,596 INFO Client1]:          7          Y                                          0.7486      30.0000
    [2024-09-01 14:58:59,262 INFO Client1]:          7          N     1.6647     0.0673        60.0000     0.6146      70.0000
    [2024-09-01 14:59:00,216 INFO Client1]:          8          Y                                          0.6415      70.0000
    [2024-09-01 14:59:01,924 INFO Client1]:          8          N     1.7078     0.0775        40.0000     0.6073      70.0000
    [2024-09-01 14:59:02,819 INFO Client1]:          9          Y                                          0.6165      70.0000
    [2024-09-01 14:59:04,483 INFO Client1]:          9          N     1.6634     0.0493        90.0000     1.0570      70.0000
