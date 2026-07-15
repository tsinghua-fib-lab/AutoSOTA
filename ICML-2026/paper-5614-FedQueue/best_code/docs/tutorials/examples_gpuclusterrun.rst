Example: Simulation on GPU cluster
==================================

This describes how to set up the environment to run APPFL using either gRPC or MPI in a GPU cluster for simulation, which is useful for benchmarking the performance of different FL algorithms on various datasets. In this example, we partition the CIFAR10 in an non-independent and identically distributed (non-IID) manner and train a Resnet-18 model using the federated learning.


gRPC Simulation on Polaris Cluster
----------------------------------

.. note::

	This section is generated based on the `Polaris supercomputer <https://docs.alcf.anl.gov/polaris/getting-started/>`_ at the Argonne Leadership Computing Facility (ALCF), which uses Portable Batch System (PBS) as it job scheduler.

Loading Modules
~~~~~~~~~~~~~~~

Most HPC clusters use `modules <https://hpc-wiki.info/hpc/Modules>`_ to manage the environment, and the module configuration may vary depending on the clusters you use. On the Polaris supercomputer, load necessary module via the following commands:

.. code-block:: bash

	module use /soft/modulefiles
	module load conda
	module save

Creating Conda Environment and Installing APPFL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, we can create a conda environment and install APPFL.

.. code-block:: bash

    conda create -n appfl python=3.10 # or conda create -p /path/to/env python=3.10
    conda activate appfl
    git clone https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[examples]"
    cd examples


Creating Batch Script
~~~~~~~~~~~~~~~~~~~~~
The Polaris supercomputer uses PBS workload manager for job management. Below is an example of a batch script to run the gRPC simulation on the Polaris cluster which launch one server and two clients. Please replace ``<your_project>`` with your project name, ``<your_env>`` with the name of the conda environment, and ``<your_appfl_path>`` with the path to the APPFL repository.

.. code-block:: bash
	:caption: submit.sh

	#!/bin/bash
	#PBS -A <your_project>
	#PBS -q debug
	#PBS -l walltime=00:15:00
	#PBS -l nodes=1:ppn=64
	#PBS -l filesystems=home:eagle:grand
	#PBS -m bae

	# Set proxy
	export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
	export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
	export http_proxy="http://proxy.alcf.anl.gov:3128"
	export https_proxy="http://proxy.alcf.anl.gov:3128"
	export ftp_proxy="http://proxy.alcf.anl.gov:3128"
	export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

	# Load modules and activate conda environment
	module use /soft/modulefiles
	module load conda
	conda activate <your_env>
	cd <your_appfl_path>/APPFL/examples

	# Launch the server
	python ./grpc/run_server.py --config ./resources/configs/cifar10/server_fedavg.yaml &
	sleep 20
	echo "Server is ready"

	# Launch the clients
	python ./grpc/run_client.py --config ./resources/configs/cifar10/client_1.yaml &
	python ./grpc/run_client.py --config ./resources/configs/cifar10/client_2.yaml &
	wait


.. note::

	On Polaris, it is important to set the proxy environment variables to access the internet from the cluster.

You can submit the script to run via the following command.

.. code-block:: bash

	qsub submit.sh

Two output files, ``submit.sh.o{job_id}`` and ``submit.sh.e{job_id}``, are generated when the script starts to run. You can check the output in real-time by running the following command.

.. code-block:: bash

	tail -f -n 10 submit.sh.o{job_id}
	# or
	tail -f -n 10 submit.sh.e{job_id}


MPI Simulation on Delta Cluster
-------------------------------

.. note::

	This tutorial is generated based on the `Delta supercomputer <https://docs.ncsa.illinois.edu/systems/delta/en/latest>`_ at the National Center for Supercomputing Applications (NCSA), which uses Slurm as it job scheduler.

Loading Modules
~~~~~~~~~~~~~~~

Most HPC clusters use `modules <https://hpc-wiki.info/hpc/Modules>`_ to manage the environment, and the module configuration may vary depending on the clusters you use. On the Delta supercomputer, the following modules are loaded.

.. code-block:: bash

	1) gcc/11.4.0   2) openmpi/4.1.6   3) cuda/11.8.0   4) cue-login-env/1.0   5) slurm-env/0.1   6) default-s11   7) anaconda3_gpu/23.9.0

You need to run ``module save`` to save the current module configuration.

.. code-block:: bash

	module save

Creating Conda Environment and Installing APPFL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, we can create a conda environment and install APPFL.

.. code-block:: bash

    conda create -n appfl python=3.10 # or conda create -p /path/to/env python=3.10
    conda activate appfl
    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[mpi,examples]"
    cd examples


Creating Batch Script
~~~~~~~~~~~~~~~~~~~~~
The Delta supercomputer uses Slurm workload manager for job management.

.. code-block:: bash
	:caption: submit.sh

	#!/bin/bash
	#SBATCH --mem=150g                              # required number of memory
	#SBATCH --nodes=1                               # number of required nodes
	#SBATCH --ntasks-per-node=6                    	# number of tasks per node [SHOULD BE EQUAL TO THE NUMBER OF CLIENTS+1]
	#SBATCH --cpus-per-task=1                       # <- match to OMP_NUM_THREADS
	#SBATCH --partition=gpuA40x4                    # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
	#SBATCH --account=<xxxx-delta-gpu>              # <- one of: replace xxxx with your project name
	#SBATCH --job-name=APPFL-test                   # job name
	#SBATCH --time=00:15:00                         # dd-hh:mm:ss for the job
	#SBATCH --gpus-per-node=1
	#SBATCH --gpu-bind=none

	source ~/.bashrc
	conda activate appfl
	cd <your_path_to_appfl>/examples
	mpiexec -np 6 python ./mpi/run_mpi.py --server_config ./resources/configs/cifar10/server_fedcompass.yaml \
		--client_config ./resources/configs/cifar10/client_1.yaml

The script can be submitted to the cluster using the following command.

.. code-block:: bash

	sbatch submit.sh

You may see the output.

.. code-block:: bash

	Submitted batch job {job_id}

The output file ``slurm-{job_id}.out`` is generated when the script starts to run, and you can check the output in real-time by running the following command.

.. code-block:: bash

	tail -f -n 10 slurm-{job_id}.out


Multi-GPU Training
------------------

APPFL supports distributed data parallelism (DDP) for multi-GPU training. To enable DDP, users only need to specify the device as a list of cuda devices in the client configuration file, for example (``examples/resources/configs/cifar10/client_1_multigpu.yaml``):

.. code-block:: yaml
	client_id: "Client1"
	train_configs:
		# Device
		device: "cuda:0,cuda:1,cuda:2,cuda:3"
		...

.. note::

	When you are using multi-GPU training, please make sure the training and validation batch size are divisible by the number of GPUs.

Below provides the batch script to run the multi-GPU training on Delta cluster using MPI.

.. code-block:: bash
	:caption: submit.sh

	#!/bin/bash
	#SBATCH --mem=150g                              # required number of memory
	#SBATCH --nodes=1                               # number of required nodes
	#SBATCH --ntasks-per-node=6                     # number of tasks per node [SHOULD BE EQUAL TO THE NUMBER OF CLIENTS+1]
	#SBATCH --cpus-per-task=1                       # <- match to OMP_NUM_THREADS
	#SBATCH --partition=gpuA40x4                    # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
	#SBATCH --account=<xxxx-delta-gpu>              # <- one of: replace xxxx with your project name
	#SBATCH --job-name=APPFL-test                   # job name
	#SBATCH --time=00:15:00                         # dd-hh:mm:ss for the job
	#SBATCH --gpus-per-node=4
	#SBATCH --gpu-bind=none

	# Activate conda environment
	source ~/.bashrc
	conda activate appfl
	cd <your_path_to_appfl>/examples

	# Launch the experiment
	mpiexec -np 6 python ./mpi/run_mpi.py --server_config ./resources/configs/cifar10/server_fedcompass.yaml \
			--client_config ./resources/configs/cifar10/client_1_multigpus.yaml

Below provides the batch script to run the multi-GPU training on Polaris cluster using MPI.

.. code-block:: bash
	:caption: submit.sh

	#!/bin/bash
	#PBS -A <your_project>
	#PBS -q debug
	#PBS -l walltime=00:15:00
	#PBS -l nodes=1:ppn=64
	#PBS -l filesystems=home:eagle:grand
	#PBS -m bae

	# Load modules and activate conda environment
	module use /soft/modulefiles
	module load conda
	conda activate <your_env>
	cd <your_appfl_path>/APPFL/examples

	# Launch the experiment
	mpiexec -np 6 python ./mpi/run_mpi.py --server_config ./resources/configs/cifar10/server_fedcompass.yaml \
			--client_config ./resources/configs/cifar10/client_1_multigpus.yaml
