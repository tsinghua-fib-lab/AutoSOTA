Endpoint Installation
=====================

This page describes how to install a `Globus Compute <https://globus-compute.readthedocs.io/en/stable/>`_ compute endpoint for federated learning (FL) clients on their local computing machines.

.. note::

	Gloubus Compute endpoint is only supported on linux machine. Please check this `documentation <./gce_install_win.rst>`_ for installing the endpoint on Windows machine using WSL.

.. note::

	Globus Compute uses port 443 for communication, so you need to make sure your firewall allows outbound traffic on port 443.

Conda environment
-----------------

We highly recommend to create a new conda virtual environment to install all the required packages for setting Globus Compute endpoint.

.. code-block:: bash

	conda create -n appfl python=3.10 --y
	conda activate appfl

Package Installation
--------------------

You just need to install the ``appfl`` package to your conda environment. You can either install it from PyPI or from the source code.

.. code-block:: bash

	pip install "appfl[examples]"
	# or
	git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
	cd APPFL
	pip install -e ".[examples]"


Create Globus Account
---------------------

If you do not have a globus account yet, please create a Globus account `here <https://app.globus.org/>`_.

.. note::

	If you can find your organization in Globus, it is highly recommended to use your organization account to log in to Globus as that makes it easier for your collaborators to verify your identity. Otherwise, you can register a Globus account using your commonly-used email address.

Setup a Globus Compute Endpoint
----------------------

Setup a Globus Compute endpoint using the following command. Please replace ``<ENDPOINT_NAME>`` with your own name such as ``anl-aws-gpu-01``.

.. note::

	You might be required to login with Globus when configuring the endpoint. Please follow the prompt instructions and finish the authentication steps.

.. code-block:: bash

	globus-compute-endpoint configure <ENDPOINT_NAME>

Configure the Endpoint
----------------------

The command above will create a configuration file ``$HOME/.globus_compute/<ENDPOINT_NAME>/config.yaml``. You should update this file with appropriate configurations for the computing resource you are using before starting the endpoint. We provide few suggestions on setting this configuration file.

	- If you are using your own linux machine or some virtual machines provided by cloud service provider such as AWS EC2 or Azure virtual machine, you probably do not need change most part of the config.py file. You just need to specify the number of blocks you want to allocate to the endpoint.

	- If you are using any supercomputer as your computing resources which uses some scheduler such as Slurm to allocate resources, you can find some example configurations for various supercomputers `here <https://globus-compute.readthedocs.io/en/stable/endpoints/endpoint_examples.html#>`_.

.. note::

	- If you have further questions about setting up Globus Compute endpoints, please join the `Globus Compute Slack <https://join.slack.com/t/funcx/shared_invite/zt-gfeclqkz-RuKjkZkvj1t~eWvlnZV0KA>`_ for help.

Start the Endpoint
------------------

Now you can start the endpoint using the following command by replacing ``<ENDPOINT_NAME>`` with your endpoint name. It should be noted that the working directory of the endpoint is ``$HOME/.globus_compute/<ENDPOINT_NAME>/tasks_working_dir``.

.. code-block:: bash

	$ globus-compute-endpoint start <ENDPOINT_NAME>

.. note::

	Whenever you start your endpoint, you should start it in the **created conda environment**.


Get your Endpoint Id
--------------------

The following command will print the id of your created endpoint.

.. code-block:: bash

	$ globus-compute-endpoint list


Run A Simple Test
-----------------

You can create a python script (e.g. ``test.py``) by copying the following codes to test if you have successfully set up a Globus Compute endpoint. You need to put your own endpoint id into the script, and you should see the printed result computed by your endpoint.

.. literalinclude:: ./endpoint.py
    :language: python
    :caption: Test script for testing Globus Compute endpoint setup.

Provide Your Endpoint to APPFLx Developer
-----------------------------------------

Dear APPFLx collaborators, currently, we have to ask the Globus Compute developers to manually add your endpoint id to a certain globus group for usage in experiments, and this process cannot be automated now for some safety reasons. Please provide your endpoint id to us via Slack, and we will register the endpoint for you.
