Endpoint Installation Using WSL
===============================

This page describes how to install a `Globus Compute <https://globus-compute.readthedocs.io/en/stable/>`_ compute endpoint for federated learning (FL) clients on Windows machine using WSL.


Install WSL
-----------

Follow `Install WSL | Microsoft Learn <https://learn.microsoft.com/en-us/windows/wsl/install>`_.


Install conda in WSL
--------------------

1. Update and upgrade WSL Linux distribution

.. code-block:: bash

	sudo apt-get update
	sudo apt-get upgrade

2. Download the Anaconda bash script

.. code-block:: bash

	wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

.. note::

	You can find other versions of Anaconda on https://www.anaconda.com/products/distribution.

3. Make the script executable

.. code-block:: bash

	chmod +x Anaconda3-2022.05-Linux-x86_64.sh

4. Run the script to install Anaconda

.. code-block:: bash

	bash Anaconda3-2022.05-Linux-x86_64.sh

5. Open a new terminal and verify the installation

.. code-block:: bash

	conda -version

Endpoint Installation
---------------------

Follow `Endpoint Installation <./gce_install.rst>`_.
