Installation
============

This page describes how to install APPFL on a machine independent of operating systems.

Conda environment
-----------------

We highly recommend to create new conda environment and install the required packages for APPFL.

.. code-block:: bash

    conda create -n APPFL python=3.10
    conda activate APPFL

User installation
-----------------

For most users, including data scientists, this simple installation is sufficient for running the package.

.. code-block:: bash

    pip install pip --upgrade
    pip install "appfl[mpi,examples]"

.. note::

    If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install "appfl[examples]""``.

If you want to even minimize the installation of package dependencies, you can use:

.. code-block:: bash

    pip install appfl

.. note::

    ``torch`` may need to be updated manually to support CUDA. Please check GPU support in `PyTorch <pytorch.org>`_.

Developer installation
----------------------

Code developers and contributors may want to work on the local repository.
To set up the development environment,

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[dev,mpi,examples]"

.. note::

    If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install -e ".[dev,examples]"``.


On Ubuntu, if the installation process failed, you can try:

.. code-block:: bash

    sudo apt install libopenmpi-dev,libopenmpi-bin,libopenmpi-doc
