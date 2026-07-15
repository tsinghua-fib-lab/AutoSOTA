Join the APPFL-Hosted International Federation
==============================================

This tutorial demonstrates how to join the international federated hosted by the APPFL team.

.. note::
    This tutorial requires the user to have a valid `Globus <https://www.globus.org/>`_ account. If you do not have an account, please go to `Globus <https://www.globus.org/>`_ and click the "LOG IN" button in the top right corner. Then, you can first look up your orgnization to see if it is already registered. If not, you can create a Globus ID to sign in or sign in with Github, Google, or ORCID.

Register with APPFL
-------------------

Once you have a Globus account, please join `APPFL's Discord <https://discord.com/invite/bBW56EYGUS>`_ and provide your Globus ID and email in the #international-federation channel. The APPFL team will then add you to the APPFL international federation and let you know when you are added.

.. note::
    The Globus ID can be found when you login to Globus, and click on "SETTINGS" in the left menu bar, then you can find your ID within the Account tab.

Generate Your Credentials
-------------------------

Once you are noticed by the APPFL team that you are added to the APPFL international federation, you can go to `Globus Developer Console <https://app.globus.org/settings/developers>`_ and click the "APPFL International" projects on the right. Then, click "APPFL International Federation" and you will see a console like the following. Please first note down the "Client UUID" shown in the console.

.. raw:: html

    <div style="display: flex; justify-content: center; width: 100%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/globus-appfl-international-console.png" alt="appfl_international_console">
        </div>
    </div>

Then, click "Add Client Secret" and you will see a pop-up windown asking you to enter a name for the credential. You may enter any name you like. After you click "Generate Secret", you will see a generated secret for you that will only be shown once. Please copy the secret and save it in a safe place. You will need this secret to run the APPFL international federation.

.. warning::
    The secret will only be shown once. If you lose it, you will need to delete the credential and generate a new one. Also, please do not share your secret with anyone else. The APPFL team will never ask you for your secret.

APPFL Installation
------------------

After obtaining the client UUID and secret, it is time to install the APPFL package on the machine you want to run the APPFL international federation. You can install the APPFL package by running the following steps:

1. Create a new conda environment (optional, but recommended):

   .. code-block:: bash

      conda create -n appfl python=3.10
      conda activate appfl

2. To install the APPFL package, you can either choose to install it directly from PyPI or install from source

**Install Directly from PyPI**

.. code-block:: bash

   pip install "appfl[examples]"

**Install from Source**


.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[examples]"



Create a Globus Compute Endpoint
--------------------------------

Once you have installed the APPFL package, you can create a Globus Compute endpoint on your machine to run the APPFL international federation. You can do this by running the following commands, where you will be using the client UUID and secret you obtained earlier.

.. code-block:: bash

    export GLOBUS_COMPUTE_CLIENT_ID=<Client UUID>
    export GLOBUS_COMPUTE_CLIENT_SECRET=<Client Secret>
    globus-compute-endpoint configure appfl-endpoint

.. note::

    You can replace "appfl-endpoint" with any name you like.

Configure the Globus Compute Endpoint
-------------------------------------

After creating the endpoint, you will be asked to configure the endpoint. If you are using a local compute or a cloud compute instance, you usually can use the default configuration. However, if you are using a HPC cluster or a Kubernetes cluster, you may need to modify the configuration file at ``~/.globus-compute/<your_endpoint_name>/config.yaml``.  You can find `example configuration files <https://globus-compute.readthedocs.io/en/latest/endpoints/endpoint_examples.html>`_ for different types of clusters and the `configuration reference <https://globus-compute.readthedocs.io/en/latest/endpoints/config_reference.html>`_ in the Globus Compute documentation.

After you have configured the endpoint, you can start the endpoint by running the following command:

.. code-block:: bash

    globus-compute-endpoint start appfl-endpoint

Then, please share the endpoint ID with the APPFL team in the #international-federation channel on Discord.
