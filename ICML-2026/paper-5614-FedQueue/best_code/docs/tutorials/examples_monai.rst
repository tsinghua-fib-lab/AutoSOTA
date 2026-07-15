Example: Running APPFL using MONAI Bundle
=========================================

.. raw:: html

    <div style="display: flex; justify-content: center; width: 80%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/appfl-monai.png" alt="monai">
        </div>
    </div>

This tutorial describes how to run federated learning experiments via APPFL using MONAI Bundles to leverage a collection of medical imaging models available in `MONAI model zoo <https://monai.io/model-zoo.html>`_. This examples shows how to use MONAI Bundle to do 3D spleen CT segmentation using gRPC with two clients.

.. note::

    **Acknowledgement**: We extend our gratitude to the MONAI and NVFlare teams for their invaluable support and information throughout this tutorial. Specifically, this tutorial refers to the `NVFlare-MONAI integration tutorial <https://github.com/NVIDIA/NVFlare/tree/dev/integration/monai/examples/spleen_ct_segmentation_local>`_.

.. note::

    This tutorial is the beta version of the integration of MONAI Bundle with APPFL. The integration is still under active development.


Installation
------------

User can install ``appfl`` and ``monai`` packages from ``appfl``'s source code by running the following commands:

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[monai,examples]"

Download MONAI Bundle
---------------------

We suggest user to download the bundle into the ``examples/resources/monai`` directory.

.. code-block:: bash

    mkdir -p examples/resources/monai
    cd examples/resources/monai

Then, download the spleen CT segmentation bundle from MONAI Bundle repository:

.. code-block:: bash

    JOB_NAME=job
    python3 -m monai.bundle download --name "spleen_ct_segmentation" --version "0.4.6" --bundle_dir ./${JOB_NAME}/app/config

Download Data
-------------

User can create a ``download_spleen_data.py`` script within the ``examples/resources/monai`` directory with the following content to download spleen CT data:

.. code-block:: python

    import argparse

    from monai.apps.utils import download_and_extract


    def download_spleen_dataset(filepath, output_dir):
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        download_and_extract(url=url, filepath=filepath, output_dir=output_dir)


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--filepath",
            "-f",
            type=str,
            help="the file path of the downloaded compressed file.",
            default="./data/Task09_Spleen.tar",
        )
        parser.add_argument(
            "--output_dir", "-o", type=str, help="target directory to save extracted files.", default="./data"
        )
        args = parser.parse_args()
        download_spleen_dataset(args.filepath, args.output_dir)

Then, user can run the script to download the spleen CT data:

.. code-block:: bash

    JOB_NAME=job
    python download_spleen_dataset.py
    sed -i "s|/workspace/data/Task09_Spleen|${PWD}/data/Task09_Spleen|g" ${JOB_NAME}/app/config/spleen_ct_segmentation/configs/train.json

Launch the server
-----------------

User can launch the server in one terminal using the FedAvg algorithm by the following command:

.. code-block:: bash

    cd examples
    python grpc/run_server.py --config ./resources/configs/monai/server_fedavg.yaml


.. note::

    User can change to other federated learning algorithms by using other server configuration files in the ``./resources/configs/monai`` directory.

Launch the clients
------------------

In the server configuration file, we specify that the number of clients is two, so we open two separate terminals to launch the clients:

.. code-block:: bash

    cd examples
    python grpc/run_client_monai.py --config ./resources/configs/monai/client_1.yaml # terminal 1
    python grpc/run_client_monai.py --config ./resources/configs/monai/client_2.yaml # terminal 2

Experiment Results
------------------

A sample server log for the experiment is shown below:

.. code-block:: bash

    appfl: ✅[2025-01-19 04:04:05,174 server]: Logging to ./output/result_Server_2025-01-19-04-04-05.txt
    appfl: ✅[2025-01-19 04:07:00,973 server]: Received GetConfiguration request from client Client1
    appfl: ✅[2025-01-19 04:07:39,732 server]: Received UpdateGlobalModel request from client Client1
    appfl: ✅[2025-01-19 04:07:39,741 server]: Received the following meta data from Client1:
    {'round': 1,
    'val_accuracy': 0.9534343488656791,
    'val_accuracy_before_train': 0.7170387863353559,
    'val_mean_dice': 0.06496836245059967,
    'val_mean_dice_before_train': 0.03413229435682297}
    appfl: ✅[2025-01-19 04:08:02,911 server]: Received GetConfiguration request from client Client2
    appfl: ✅[2025-01-19 04:08:44,316 server]: Received UpdateGlobalModel request from client Client2
    appfl: ✅[2025-01-19 04:08:44,319 server]: Received the following meta data from Client2:
    {'round': 1,
    'val_accuracy': 0.9544978111412874,
    'val_accuracy_before_train': 0.7170388106327907,
    'val_mean_dice': 0.06501330435276031,
    'val_mean_dice_before_train': 0.034132301807403564}
    appfl: ✅[2025-01-19 04:09:01,715 server]: Received UpdateGlobalModel request from client Client2
    appfl: ✅[2025-01-19 04:09:01,717 server]: Received the following meta data from Client2:
    {'round': 2,
    'val_accuracy': 0.9604373494530939,
    'val_accuracy_before_train': 0.9539599266781169,
    'val_mean_dice': 0.06739335507154465,
    'val_mean_dice_before_train': 0.06500281393527985}
    appfl: ✅[2025-01-19 04:09:02,419 server]: Received UpdateGlobalModel request from client Client1
    appfl: ✅[2025-01-19 04:09:02,421 server]: Received the following meta data from Client1:
    {'round': 2,
    'val_accuracy': 0.9590474329176982,
    'val_accuracy_before_train': 0.9539599327524756,
    'val_mean_dice': 0.06810087710618973,
    'val_mean_dice_before_train': 0.06500281393527985}
    appfl: ✅[2025-01-19 04:09:18,915 server]: Received UpdateGlobalModel request from client Client2
    appfl: ✅[2025-01-19 04:09:18,916 server]: Received the following meta data from Client2:
    {'round': 3,
    'val_accuracy': 0.9941919290336074,
    'val_accuracy_before_train': 0.9597333312793902,
    'val_mean_dice': 0.005374908447265625,
    'val_mean_dice_before_train': 0.06778988242149353}
    appfl: ✅[2025-01-19 04:09:20,032 server]: Received UpdateGlobalModel request from client Client1
    appfl: ✅[2025-01-19 04:09:20,034 server]: Received the following meta data from Client1:
    {'round': 3,
    'val_accuracy': 0.9942095143020533,
    'val_accuracy_before_train': 0.9597333312793902,
    'val_mean_dice': 0.005186689551919699,
    'val_mean_dice_before_train': 0.06778988242149353}
    appfl: ✅[2025-01-19 04:09:36,668 server]: Received UpdateGlobalModel request from client Client2
    appfl: ✅[2025-01-19 04:09:36,670 server]: Received the following meta data from Client2:
    {'round': 4,
    'val_accuracy': 0.9946757274068845,
    'val_accuracy_before_train': 0.9941966548846786,
    'val_mean_dice': 0.0,
    'val_mean_dice_before_train': 0.00541022839024663}
    appfl: ✅[2025-01-19 04:09:37,791 server]: Received UpdateGlobalModel request from client Client1
    appfl: ✅[2025-01-19 04:09:37,793 server]: Received the following meta data from Client1:
    {'round': 4,
    'val_accuracy': 0.9946757091838083,
    'val_accuracy_before_train': 0.9941966548846786,
    'val_mean_dice': 0.0,
    'val_mean_dice_before_train': 0.00541022839024663}
    appfl: ✅[2025-01-19 04:09:53,942 server]: Received UpdateGlobalModel request from client Client1
    appfl: ✅[2025-01-19 04:09:53,944 server]: Received the following meta data from Client1:
    {'round': 5,
    'val_accuracy': 0.9708507136934122,
    'val_accuracy_before_train': 0.9946757274068845,
    'val_mean_dice': 0.18941788375377655,
    'val_mean_dice_before_train': 0.0}
    appfl: ✅[2025-01-19 04:09:56,546 server]: Received UpdateGlobalModel request from client Client2
    appfl: ✅[2025-01-19 04:09:56,548 server]: Received the following meta data from Client2:
    {'round': 5,
    'val_accuracy': 0.9699774821093128,
    'val_accuracy_before_train': 0.9946757274068845,
    'val_mean_dice': 0.19541805982589722,
    'val_mean_dice_before_train': 0.0}
    appfl: ✅[2025-01-19 04:09:56,649 server]: Received InvokeCustomAction close_connection request from client Client1
    appfl: ✅[2025-01-19 04:09:56,651 server]: Received InvokeCustomAction close_connection request from client Client2
    appfl: ✅[2025-01-19 04:09:57,537 server]: Terminating the server ...
