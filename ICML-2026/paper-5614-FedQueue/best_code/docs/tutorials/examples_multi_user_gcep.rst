Example: Setting Up a Multi-User Globus Compute Endpoint for APPFL
===================================================================

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/1PbpzkRb7_M?si=VuBQkfdtfCudDbr6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

This tutorial describes how to set up a multi-user Globus Compute Endpoint (GCEP) on a remote server (e.g., an AWS EC2 instance, an Azure VM, or any Linux machine) and to run APPFL federated learning experiments on it. A multi-user endpoint allows multiple users to share a single endpoint installation with proper identity mapping.

.. note::

    For more detailed information about Globus Compute, please refer to the `Globus Compute documentation <https://globus-compute.readthedocs.io/en/stable/index.html>`_.

Prerequisites
-------------

- A Linux server (e.g., AWS EC2, an Azure VM, or any remote machine) with ``sudo`` access
- A Globus account
- SSH access to the remote server

Step 1: Install Conda
---------------------

On the remote server, install Miniconda system-wide so that it is accessible to all users:

.. code-block:: bash

    sudo -i
    sudo mkdir -p /opt
    cd /opt

    sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sudo bash Miniconda3-latest-Linux-x86_64.sh

Then add the conda initialization script to the system-wide profile so that it is sourced for all users. Create or edit ``/etc/profile.d/conda.sh`` and add the following line:

.. code-block:: bash

    source /opt/conda/etc/profile.d/conda.sh

.. note::

    The Miniconda installer will ask for a target directory. Using ``/opt/conda`` (or another system-wide path) is recommended so all users can access the same Conda installation. Make sure to set the correct permissions for non-root users if needed.

Step 2: Create a Shared Conda Environment
------------------------------------------

Create a dedicated Conda environment for APPFL:

.. code-block:: bash

    sudo /opt/conda/bin/conda create -n appfl python=3.10
    conda activate appfl

Step 3: Transfer and Install APPFL
------------------------------------

Transfer the APPFL source code to the server. Download ``APPFL.zip`` from `Google Drive <https://drive.google.com/file/d/19Ujjs2pmrYVRaMwXxS2_ZhMVsgDx8lm9/view?usp=sharing>`_ to your local machine, then transfer it to the server:

.. code-block:: bash

    # Run this on your LOCAL machine to transfer the archive
    # Replace <key.pem>, <user>, and <host> with your own values
    scp -i <key.pem> /path/to/APPFL.zip <user>@<host>:/home/<user>/

    # Then on the REMOTE server
    sudo mkdir -p /opt/src
    cp /home/<user>/APPFL.zip /opt/src/
    cd /opt/src/
    unzip APPFL.zip

.. note::

    Replace ``<key.pem>``, ``<user>``, and ``<host>`` with your own SSH key file, remote username, and remote hostname/IP address.

Install the APPFL package:

.. code-block:: bash

    cd /opt/src/APPFL-main
    pip install -e ".[dev,mpi,examples]"

Step 4: Install FLamby
-----------------------

This tutorial uses `FLamby <https://github.com/owkin/FLamby>`_ for the Heart Disease federated dataset. The pre-packaged FLamby zip archive is available for download from `Google Drive <https://drive.google.com/file/d/10nDNQlsnk7LvA16krvLRkiMznOt3Xk9K/view?usp=sharing>`_.

Download ``FLamby.zip`` from the `Google Drive link <https://drive.google.com/file/d/10nDNQlsnk7LvA16krvLRkiMznOt3Xk9K/view?usp=sharing>`_, then transfer and install it on the server:

.. code-block:: bash

    # Run this on your LOCAL machine to transfer the archive
    # Replace <key.pem>, <user>, and <host> with your own values
    scp -i <key.pem> /path/to/FLamby.zip <user>@<host>:/home/<user>/

    # Then on the REMOTE server
    cp /home/<user>/FLamby.zip /opt/src/
    cd /opt/src/
    unzip FLamby.zip

Install the FLamby package and download the Heart Disease dataset:

.. code-block:: bash

    cd /opt/src/FLamby-main
    pip install monai==1.2.0
    pip install -e ".[heart]"
    cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
    python download.py --output-folder ./heart_disease_dataset

Step 5: Configure the Multi-User Globus Compute Endpoint
---------------------------------------------------------

On the server, configure a new Globus Compute endpoint in multi-user mode. Replace ``<endpoint-name>`` with your desired endpoint name:

.. code-block:: bash

    globus-compute-endpoint configure <endpoint-name>

This creates a default configuration directory at ``~/.globus_compute/<endpoint-name>/``. A multi-user endpoint requires an *identity mapping file* so that Globus can map incoming Globus identities to local system user accounts.

.. note::

    This step should be run as the system user who will own and manage the endpoint (often ``root`` or a dedicated service account).

Step 5a: Create the Identity Mapping File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The identity mapping file tells the endpoint which Globus identity (e.g., a Globus ID or institutional email) maps to which local Linux user. Remove the default example file and create your own:

.. code-block:: bash

    rm ~/.globus_compute/<endpoint-name>/example_identity_mapping_config.json
    vim ~/.globus_compute/<endpoint-name>/example_identity_mapping_config.json

Populate the file with the following structure. **Replace the placeholder values** with the actual Globus identities of your users and the corresponding local Linux usernames:

.. code-block:: json

    [
      {
        "DATA_TYPE": "expression_identity_mapping#1.0.0",
        "mappings": [
          {
            "source": "{username}",
            "match": "<globus-identity-1>",
            "output": "<local-linux-username>"
          },
          {
            "source": "{username}",
            "match": "<globus-identity-2>",
            "output": "<local-linux-username>"
          }
        ]
      }
    ]

Where:

- ``<globus-identity-N>`` is the Globus identity of the user (e.g., ``user@globusid.org`` or ``user@institution.edu``). You can find your Globus identity by logging into `app.globus.org <https://app.globus.org>`_ and visiting your account settings.
- ``<local-linux-username>`` is the Linux user account on the server that the Globus identity should be mapped to. All users can be mapped to the same shared account, or each user can have their own system account.

.. note::

    You can add as many ``mappings`` entries as needed — one per Globus user who should have access to the endpoint.

.. warning::

    Make sure the local Linux username(s) listed in the mapping actually exist on the server. Create them with ``useradd`` if they do not exist.

Step 5b: Start the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After saving the identity mapping file, start the endpoint:

.. code-block:: bash

    globus-compute-endpoint start <endpoint-name>

The endpoint UUID will be printed to the console after it starts. Save this UUID — it is needed by the APPFL server to submit tasks to this endpoint.
