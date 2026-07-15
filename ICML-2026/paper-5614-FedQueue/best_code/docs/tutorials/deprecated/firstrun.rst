First run: MNIST
================

.. note::

    This tutorial is for the deprecated version 0 of APPFL, and you may see many deprecation warnings when running the code. We highly recommend using the latest version of APPFL.

We present how to run an example federated learning script on the MNIST dataset using APPFL.
We first make sure that the APPFL package and its dependencies are installed and change the directory to ``examples/examples_legacy`` directory.

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[examples]"
    cd examples/examples_legacy

The first example can be simply run using the following command, which launches a federated learning experiment with five clients. The federated learning server and five federated learning clients run serially on one machine for simulation.

.. code-block:: bash

    python ./mnist_serial.py --num_clients 5

If we want to run **synchronous** FL experiment in parallel using MPI, we can run the example using the following command.

.. code-block:: bash

    mpiexec -np 6 python ./mnist_mpi_sync.py

.. note::

    `-np 6` in the above command means that we are launching 6 MPI processes. If you do not explicitly specify `--num_clients`, then the number of clients is equal to the number of processes minus one (one MPI process is used as an FL server), which is 5 in this case. You can also explicitly specify the number of clients, e.g., `--num_clients 10` will launch ten FL clients with two clients running serially on one MPI process (also, one MPI process is used as an FL server).

.. note::

    ``mpiexec`` may need to specify additional argument to use CUDA: ``--mca opal_cuda_support 1``

If we want to run **asynchronous** FL experiment in parallel using MPI, we can run the example using the following command.

.. code-block:: bash

    mpiexec -np 6 python ./mnist_mpi_async.py

.. note::

    For asynchronous communication, the number of clients is exactly equal to the number of MPI processes minus one, so we are launching 5 clients running FL asynchronously for the above command. We do not allow more than one client to run serially on one MPI process for asynchronous cases as it does not make sense.

As the package name suggests, in addition to general differential privacy techniques, APPFL also provides two special synchronous privacy-preserving algorithms, `IIADMM <https://arxiv.org/pdf/2202.03672.pdf>`_ and `ICEADMM <https://arxiv.org/pdf/2110.15318.pdf>`_. We can run the privacy-preserving federated learning (PPFL) algorithms by running the following command.

.. code-block:: bash

    mpiexec -np 6 python ./mnist_mpi_privacy.py --server IIADMM
    mpiexec -np 6 python ./mnist_mpi_privacy.py --server ICEADMM


We can also simulate PPFL using the gRPC communication protocol. The following command launches five MPI processes, with one running the gRPC-based FL server and four running the gRPC-based FL clients.

.. code-block:: bash

    mpiexec -np 5 python ./mnist_grpc.py
