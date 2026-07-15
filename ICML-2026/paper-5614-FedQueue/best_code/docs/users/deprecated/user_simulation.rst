Simulating PPFL
===============

This package provides users with the capability of simulating PPFL on either a single machine or an HPC cluster.

.. note::

    Running PPFL on multiple heterogeneous machines is described in :ref:`Training PPFL`.


We describe how to simulate PPFL with a given model and datasets. For simulation, we assume that ``test_data`` is available to validate the trained model.

Serial run
----------

Serial runs begin simply by calling the following API function.

.. autofunction:: appfl.run_serial.run_serial


Some remarks are made as follows:

- Parameter ``cfg: DictConfig`` reads the configuration of runs. See :ref:`How to set configuration` for details about configuration.
- Parameters ``model``, ``train_data``, and ``test_data`` should be given by users; see :ref:`User-defined model` and :ref:`User-defined dataset`.


Parallel run with MPI
---------------------

We can parallelize the PPFL simulation by usinig MPI through ``mpi4py`` package.
The following two API functions need to be called for parallelization.

.. autofunction:: appfl.run_mpi.run_server

.. autofunction:: appfl.run_mpi.run_client


The server and the clients begin by ``run_server`` and ``run_client``, respectively, where MPI communicator (e.g., ``MPI.COMM_WORLD`` in this example) is given as an argument.

.. note::

    We assume that MPI process 0 runs the server, and the other processes run clients.

.. note::

    ``mpiexec`` may need to specify additional argument to use CUDA: ``--mca opal_cuda_support 1``
