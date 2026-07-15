Example: Run FL Experiment using Ray
===============================================

This tutorial describes how to run federated learning experiments with APPFL using Ray on cloud platforms such as AWS, GCP, and Azure. All the code snippets needed for this tutorial is available at the ``examples`` directory of the APPFL repository at `here <https://github.com/APPFL/APPFL/tree/main/examples>`_.

.. note::

    For more detailed information about Ray, please refer to the `Ray documentation <https://docs.ray.io/en/latest/index.html>`_.

Installation
------------

First, we should install the APPFL package on the local machines. Below shows how to install the APPFL package from its source code. For more information, please refer to the `APPFL installation guide <https://appfl.ai/en/latest/install/index.html>`_.

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    conda create -n appfl python=3.10 --y
    conda activate appfl
    pip install -e ".[examples]"

Client Configurations
---------------------

The server needs to collect certain information from the client to run the federated learning experiment. Below is an example of a client configuration file. It is available at ``examples/resources/config_ray/mnist/clients.yaml`` at the APPFL repository at `here <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_ray/mnist/clients.yaml>`_.

.. code-block:: yaml

    clients:
      - client_id: "Client1"
        train_configs:
          # Device [Optional]: default is "cpu"
          device: "cpu"
          # Logging [Optional]
          logging_output_dirname: "./output"
          logging_output_filename: "result"

        # Local dataset
        data_configs:
          dataset_path: "./resources/dataset/mnist_dataset.py"
          dataset_name: "get_mnist"
          dataset_kwargs:
            num_clients: 2
            client_id: 0
            partition_strategy: "class_noniid"
            visualization: True

      - client_id: "Client2"
        train_configs:
          # Device [Optional]: default is "cpu"
          device: "cpu"
          # Logging [Optional]
          logging_output_dirname: "./output"
          logging_output_filename: "result"

        # Local dataset
        data_configs:
          dataset_path: "./resources/dataset/mnist_dataset.py"
          dataset_name: "get_mnist"
          dataset_kwargs:
            num_clients: 2
            client_id: 1
            partition_strategy: "class_noniid"
            visualization: False


It should be noted that the client configuration file actually resides on the server machine, and the contents of the file are shared by the clients. Specifically, there are three main parts in the client configuration file:

- ``client_id``: It is the unique ID defined for the client machine.
- ``train_configs``: It contains the training configurations for the client, including the device to run the training, logging configurations, etc.
- ``data_configs``: It contains the information of a dataloader python file defined and shared by the clients to the server (located at ``dataset_path`` on the server machine). The dataloader file should contain a function (specified by ``dataset_name``) which can load the client's local private dataset when it is executing on the client's machine.


Server Configurations
---------------------

We have provide three sample server configuration files available at ``examples/resources/config_ray`` at the APPFL repository at `here <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_ray/>`_. The detailed description of the server configuration file can be found in the `APPFL documentation <https://appfl.ai/en/latest/users/server_agent.html#configurations>`_.

It should be noted that ``client_configs.comm_configs.ray_configs`` is optional and only needed be set if the user wants to assign a job to randomly, instead to a particular AWS instance, by setting ``assign_random`` as ``True`` (by default it is ``False``). You need to configure the same in ``ray_cluster_config.yaml`` as well.

To use AWS S3 for model parameter transmission, add a configuration under ``comm_configs`` as ``s3_configs``. Set ``enable_s3`` to True, and specify the ``s3_bucket`` field with the name of the S3 bucket that you want to use. Additionally, set ``s3_creds_file`` to the path of a CSV file containing AWS credentials.

.. code-block:: csv

    <region>,<access_key_id>,<secret_access_key>

.. note::

    The server can also set these information before running the experiment via the ``aws configure`` command.

Ray Cluster Configurations
--------------------------

Below is the cluster configuration file for running the experiment on AWS cloud environment.

.. code-block:: yaml

    # An unique identifier for the head node and workers of this cluster.
    cluster_name: appfl-ray

    # Cloud-provider specific configuration.
    provider:
        type: aws
        region: us-east-1
        cache_stopped_nodes: False # if set False terminates the instance when ray down is executed, True: instance stopped not terminated
        security_group:
            GroupName: ray_client_security_group
            IpPermissions:
                - FromPort: 8265
                  ToPort: 8265
                  IpProtocol: TCP
                  IpRanges:
                      # Allow traffic only from your local IP address.
                      - CidrIp: 0.0.0.0/0

    # The maximum number of workers nodes to launch in addition to the head node.
    max_workers: 2

    available_node_types:
        ray.head.default:
            resources: { }
            # Provider-specific config for this node type, e.g., instance type.
            # By default Ray auto-configures unspecified fields such as SubnetId and KeyName.
            # For more documentation on available fields, see
            # http://boto3.readthedocs.io/en/latest/reference/services/ec2.html#EC2.ServiceResource.create_instances
            node_config:
                InstanceType: t3.medium
                ImageId: 'ami-0dd6adfad4ad37eec' # Deep Learning Base Neuron AMI (Ubuntu 20.04) 20240216
        ray.worker.worker_1:
            # The minimum number of worker nodes of this type to launch.
            # This number should be >= 0. For FL experiment 1 is sufficient.
            min_workers: 1
            # The maximum number of worker nodes of this type to launch.
            # This parameter takes precedence over min_workers. For FL experiment 1 is sufficient.
            max_workers: 1
            # Set this to {${client_id} : 1}, client_id from examples/resources/config_ray/mnist/clients.yaml config file
            # Set it to empty if client task can be assigned randomly to any worker node
            resources: {Client1: 1}
            node_config:
                InstanceType: t3.medium
                ImageId: 'ami-0dd6adfad4ad37eec' # Deep Learning Base Neuron AMI (Ubuntu 20.04) 20240216
                InstanceMarketOptions:
                    MarketType: spot  # Configure worker nodes to use Spot Instances
                    SpotOptions:
                        MaxPrice: '0.05'
        ray.worker.worker_2:
            min_workers: 1
            max_workers: 1
            resources: {Client2: 1}
            node_config:
                InstanceType: t3.medium
                ImageId: 'ami-0dd6adfad4ad37eec' # Deep Learning Base Neuron AMI (Ubuntu 20.04) 20240216
                InstanceMarketOptions:
                    MarketType: spot  # Configure worker nodes to use Spot Instances
                    SpotOptions:
                        MaxPrice: '0.05'

    file_mounts: {
        "/home/ubuntu/APPFL": "../../../APPFL",
        "/home/ubuntu/resources": "../resources",
        "/home/ubuntu/run.py": "run.py"
    }

    setup_commands:
        ["conda config --remove channels intel",
         "conda create -n APPFL python=3.10 -y ",
         'conda activate APPFL && pip install ray["default"] && pip install confluent-kafka --prefer-binary && cd APPFL && pip install -e ".[examples]"',
         "(stat $HOME/anaconda3/envs/APPFL/ &> /dev/null && echo 'export PATH=\"$HOME/anaconda3/envs/APPFL/bin:$PATH\"' >> ~/.bashrc) || true"]

You can set the desired aws region under ``provider.region``

All the EC2 instance related configuration for head node or worker nodes goes in ``node_config`` which has ``InstanceType``, ``ImageId`` (AMI image id), spot vs on demand etc. For more documentation on available fields, `see <http://boto3.readthedocs.io/en/latest/reference/services/ec2.html#EC2.ServiceResource.create_instances>`_.

For other field description you can follow inline comments in ``examples/ray/ray_cluster_config.yaml``. Further you can check it out `here <https://docs.ray.io/en/latest/cluster/vms/getting-started.html#launch-a-cluster-on-a-cloud-provider>`_.


Running Experiment
------------------

Environment setup
~~~~~~~~~~~~~~~~~

Configure AWS credentials - IAM having ``AmazonEC2FullAccess`` and ``AmazonEC2RoleforSSM``

Cluster Creation
~~~~~~~~~~~~~~~~

Go inside ray example

.. code-block:: bash

    cd examples/ray/

Run below command, which brings up whole cluster that is described in ``examples/ray/ray_cluster_config.yaml``.

.. code-block:: bash

    ray up ray_cluster_config.yaml

.. note::

    For lower cluster spin up time create a custom AMI image by running setup command on given image id in ray_cluster_config.yaml. After creating custom AMI you can provide it in ray_cluster_config.yaml under ImageId attribute of each node

Checking cluster status
~~~~~~~~~~~~~~~~~~~~~~~

**From Local machine**

1. You can check cluster status by running

.. code-block:: bash

    ray exec ray_cluster_config.yaml 'ray status'

**From Head Node**

1. Go into head node using

.. code-block:: bash

    ray attach ray_cluster_config.yaml

2. Check cluster status after attaching to head node using

.. code-block:: bash

    ray status

Output of ray status would look like below

.. code-block:: bash

    ======== Autoscaler status: 2025-02-25 20:18:02.106153 ========
    Node status
    ---------------------------------------------------------------
    Active:
     1 ray.worker.worker_2
     1 ray.head.default
     1 ray.worker.worker_1
    Pending:
     (no pending nodes)
    Recent failures:
     (no failures)

    Resources
    ---------------------------------------------------------------
    Usage:
     0.0/6.0 CPU
     0.0/1.0 Client1
     0.0/1.0 Client2
     0B/7.64GiB memory
     0B/3.16GiB object_store_memory

    Demands:
     (no resource demands)


Job Submission
~~~~~~~~~~~~~~

**From Local machine**

1. Do port forwarding using

.. code-block:: bash

    ray dashboard ray_cluster_config.yaml

2. Now on another terminal you can submit job request using:

.. code-block:: bash

    ray job submit --address http://localhost:8265  -- python APPFL/examples/ray/run.py

**From Head Node**

1. Connect to head node

.. code-block:: bash

    ray attach ray_cluster_config.yaml

2. Run job using:

.. code-block:: bash

    python run.py

Stopping Cluster
~~~~~~~~~~~~~~~~
1. To stop cluster run

.. code-block:: bash

    ray down ray_cluster_config.yaml
