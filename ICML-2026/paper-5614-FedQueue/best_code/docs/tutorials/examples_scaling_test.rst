Example: Scaling Test of APPFL on GPU cluster
=============================================

In this tutorial, we describe how to run federated learning (FL) experiments using APPFL on GPU clusters to simulate large-scale FL scenarios with hundreds or even thousands of clients. This is particularly useful for testing the scalability and performance of FL algorithms in a distributed environment.

MPI Simulation Scripts
----------------------

We provide an MPI-simulation launching script for running large-scale FL experiments at ``examples/mpi/run_mpi_scaling.py``, which takes two important configuration parameters:

- ``gpu_per_node``: Number of GPUs available on each compute node (default is 4).
- ``clients_per_gpu``: Number of clients sharing a single GPU (default is 1).

Example Script to Launch MPI Simulation
---------------------------------------

Below shows an example PBS script to launch a scaling test with 512 clients, where each GPU is shared by 8 clients. The script assumes that you have a GPU cluster with 17 nodes, each having 4 GPUs, and you want to run the simulation using MPI. Specifically,

- In line 5, we allocates 1 nodes with 1 MPI slots for FL server, and 16 nodes with 32 MPI slots for FL clients.
- The nodes for FL clients allocates 32 MPI slots as each node has 4 GPUs and each GPU is shared by 8 clients, resulting in 32 clients per node.
- In line 17, we launch the MPI simulation with 513 processes (1 server + 512 clients) using the `mpiexec` command, and the ``$PBS_NODEFILE`` environment variable to specify the nodes allocated for the job.


.. code-block:: bash
	:linenos:

	#!/bin/bash
	#PBS -A <project_name>
	#PBS -q <queue_name>
	#PBS -l walltime=01:00:00
	#PBS -l select=1:ncpus=64:mpiprocs=1+16:ncpus=64:mpiprocs=32
	#PBS -l filesystems=home:eagle:grand

	module load conda
	conda activate /eagle/tpc/zilinghan/conda_envs/appfl
	cd /eagle/tpc/zilinghan/appfl/APPFL/examples

	export OMP_NUM_THREADS=1
	export OPENBLAS_NUM_THREADS=1
	export MKL_NUM_THREADS=1
	export NUMEXPR_NUM_THREADS=1

	mpiexec -np 513 --hostfile $PBS_NODEFILE python ./mpi/run_mpi_scaling.py --clients_per_gpu 8
