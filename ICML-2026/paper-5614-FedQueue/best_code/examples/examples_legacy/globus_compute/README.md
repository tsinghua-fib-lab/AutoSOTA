# <img src="../../../docs/_static/globus_compute.jpg" alt="Globus Compute" width="40"> Instructions: Using Globus Compute for Federated Learning

## FL Client Setup
To setup a **real-world** federated learning client on a **distributed** computing machine, you need to (1) install the APPFL package locally, and (2) create and start a Globus Compute endpoint. [**Note**: Globus Compute endpoint is only supported on linux machine. If you use windows, please use WSL.]

1. Create a virtual environment using `conda`
    ```
    conda create -n APPFL python=3.8
    conda activate APPFL
    ```

2. Install APPFL package.
    ```
    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git appfl && cd appfl
    pip install -e ".[dev,examples]"
    ```

3. Creat a Globus account at https://app.globus.org. If you can find your organization in Globus, it is highly recommended to use your organization account to log in to Globus as that makes it easier for your collaborators to verify your identity (which is very import for building trust in FL). Otherwise, you can register a Globus account using your commonly-used email address.

4. Set up a Globus Compute endpoint using the following command, where `<YOUR_ENDPOINT_NAME>` is a name of your choice. You might be required to login with Globus for the first time, follow the url and complete all steps to get credential token.
    ```
    globus-compute-endpoint configure <YOUR_ENDPOINT_NAME>
    ```

5. Configure the endpoint. The command above will create a configuration file at `$HOME/.globus_compute/<YOUR_ENDPOINT_NAME>/config.yaml`. This file should be updated with the appropriate configurations for the computational system you are targeting before you start the endpoint. Globus Compute document shows some of the example setups [here](https://funcx.readthedocs.io/en/latest/endpoints.html#example-configurations). We also provide two example configuration yaml files here for [CPU](./endpoint_config/delta_ncsa_cpu.yaml) or [GPU](./endpoint_config/delta_ncsa_gpu.yaml) usage on the [Delta supercomputer @ NCSA](https://ncsa-delta-doc.readthedocs-hosted.com/en/latest/). Here comes the detailed explanation for the GPU configuration yaml file:
    - partition: The partition is a logical grouping of computing nodes for different usage, and you can obtain all partition information of your HPC by running `sinfo -s`
    - account: The account name for the SLURM scheduler to charge CPU/GPU hours. In Delta, if your group name is bbvf, then the account should be bbvf-delta-cpu or bbvf-delta-gpu for CPU/GPU usage.
    - exclusive: Whether to request nodes which are not shared with other running jobs. **(Note: In most cases, set it to False, otherwise, it is very very hard to get GPU resources.)**
    - worker_init: You need to specify the commands to be run before starting a worker, such as loading a module `module load` (or module collection `module restore`) and activating a conda environment. Separate commands by semicolon (`;`).
    - scheduler_option: Here you need to provide some #SBATCH (or other scheduler related) directives, such as setting up constraints, requesting GPUs, and disable GPU binding policy.
    - init_blocks: In terms of SLURM scheduling, this refers to the number of slurm batch jobs submitted when you start the endpoint. **(For our FL usecase, we just need 1.)**
    - min_blocks: Minimum number of slurm batch jobs when there are no/few tasks. If you choose 0, then the slurm job will be canceled when there are no tasks. **(For our FL usecase, we want to set this to 1 instead of 0.)**
    - max_blocks: Maximum number of slurm batch jobs when there are more tasks and available resources. **(For our FL usecase, we do not need auto scaling and just need to set this to 1.)**
        ```
        display_name: NCSA Delta GPU
        engine:
            type: HighThroughputEngine
            max_workers_per_node: 2
            worker_debug: False

            address:
                type: address_by_interface
                ifname: eth6.560

            provider:
                type: SlurmProvider
                partition: gpuA40x4
                account: bbvf-delta-gpu

                launcher:
                    type: SrunLauncher

                # Command to be run before starting a worker
                # e.g., "module load anaconda3; source activate gce_env"
                worker_init: "conda activate globus-compute"
                exclusive: False

                scheduler_options: "#SBATCH --constraint='projects'\n#SBATCH --gpus-per-node=1\n#SBATCH --gpu-bind=none"

                init_blocks: 1
                min_blocks: 1
                max_blocks: 1

                walltime: 00:30:00
        ```

6. Start the endpoint. Before starting the funcX endpoint, you need to first change to a certain directory you want, which will be the root directory for funcX when accessing your file system or writing output files. Please select that root directory carefully. When you are in your desired directory, run the following command by replacing `<YOUR_ENDPOINT_NAME>` with your endpoint name to start the funcX endpoint. [**Note**: Whenever you start your endpoint, you should start it in the created conda environment.]
    ```
    globus-compute-endpoint start <YOUR_ENDPOINT_NAME>
    ```

7. The client need to get and note down the endpoint id by running the following command.
    ```
    globus-compute-endpoint list
    ```

8. Finally, if the client wants to run the provided MNIST example using globus compute, it is highly recommended to first download the MNIST dataset into the computing machine by running the [`examples/globus_compute/datasets/download_mnist.py`](./datasets/download_mnist.py) and providing the `data_dir` argument, especially if all the clients are running on the same machine to avoid race condition during the run-time downloading.

## FL Server Setup
To set up a federated learning server using Globus Compute for real-world FL experiments on heterogeneous computing facilities, you need to do the following steps.

1. Creat a Globus account at https://app.globus.org. Same as above, try to find your institution first, or create an account using your commonly-used email address.

2. As Globus Compute has maximum size for the transferred data (10 MB for arguments and 5 MB for results), APPFL employs AWS S3 bucket to transfer (the relatively large) models. Therefore, you need to have an AWS account and create an S3 bucket. Please refer to the official AWS S3 documentation [here](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) and follow the steps under **Using the S3 console** to create a bucket. Note down the **bucket name** for your bucket, and set it as the bucket name (`s3_bucket`) in the [server configuration file](./configs_server/mnist_fedavg.yaml).

3. After creating the bucket, you also need to provide a credential file so that the FL server as well as the FL clients can access the created bucket through the python scripts to upload/download models. Here are the detailed steps to do so:
    - Sign in to the AWS Console and navigate to the **IAM** console.
    - In the left navigation pane, choose **Policies**, and then click the **Create policy** button.
    - Choose **JSON** as the policy editor, and replace the default content with the following JSON policy (and modify `your-bucket-name` accordingly).
        ```
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:*",
                    "Resource": [
                        "arn:aws:s3:::your-bucket-name",
                        "arn:aws:s3:::your-bucket-name/*"
                    ]
                }
            ]
        }
        ```
    - Name your policy (e.g., **`FullAccessToMyBucket`**) and click **Create policy**.
    - Go back to the IAM console navigation pane, choose **Users**, and click **Create user** button.
    - Give the user a name, then in the **Set permissions** section, select **Attach policies directly** as the permissions options, and then search for the policy you just created (e.g., **`FullAccessToMyBucket`**). Then click **Create user**.
    - After creating the user, go back to the **User** panel and click the name of the newly created user. Click the **Security credentials** tab, and in the **Access Key** subsection, click **Create access key**.
    - Select **Local code** as your use case and check the bottom confirmation box, then **Create access key**.
    - Download the access key credential `.csv` file and keep in a secure place on your local computer. The file should contains similar information as follows:
        ```
        Access key ID,Secret access key
        ABCDEFGHIJKLMN,OPQRSTUVWXYZ123
        ```

4. Now you already have the Access key ID and the Secret access key for the S3 bucket access, you have the following two ways to provide them to the scripts to use them.
    - **[Recommended]** Install `awscli` by running `pip install awscli`. Then run `aws configure` and enter the following information accordingly. The Default region name is the aws region where you create your bucket. You can just hit enter for Default output format. If you use this method, you do not need to set the `s3_creds` field in the [server configuration file](./configs_server/mnist_fedavg.yaml)
        ```
        $ aws configure
        AWS Access Key ID [****************ABCD]:
        AWS Secret Access Key [****************EFGH]:
        Default region name [us-east-1]:
        Default output format [None]:
        ```
    - Create a credential `.csv` file, and enter the region name, access key ID, and secret access key in a row. For example:
        ```
        us-east-1,ABCDEFGHIJKLMN,OPQRSTUVWXYZ123
        ```
        Then provide the path to the credential file (`s3_creds`) accordingly in the [server configuration file](./configs_server/mnist_fedavg.yaml). **It is important to keep this credential file secure and DO NOT share it publicly, e.g., push to your GitHub!!!!**

5. Setup other configurations for the server in the [server configuration file](./configs_server/mnist_fedavg.yaml), such as:
- `get_model`: function to get the trained model architecture
- `model`: arguments for the model if needed
- `get_loss`: function to get the training loss function
- `val_metric`: function to validate the trained model

6. The server also needs to collect the following information and files from the clients, and put them in the [client configuration file](./configs_client/mnist_iid.yaml).
- `endpoint_id`: Globus Compute Endpoint ID
- `device`: computing device for the client, `cpu` or `cuda`
- `get_data`: data loader function to load the client local dataset. [**Note**: This function will be run on the client computing machine, on the directory where the client starts the Globus Compute endpoint, so the client has to make sure that the data path in the data loader file is correct.]
- `data_dir`: [Optional] For the MNIST example, it should be the path to the client-downloaded MNIST datasets.
- `output_dir`: [Optional] Root directory for storing the outputs, default is `./output` if not provided. All the output files are stored in the nested directory: `output_dir/endpoint_id`.

7. Finally, the server can start the experiment by running the following script in the `examples` folder.
    ```
    python mnist_gc.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
    ```

## Appendix
Explanations for the provided server configuration files in `configs_server`:
- [`mnist_fedavg.yaml`](configs_server/mnist_fedavg.yaml): Server configuration for [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedavgm.yaml`](configs_server/mnist_fedavgm.yaml): Server configuration for [FedAvgMomentum](https://arxiv.org/abs/1909.06335) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedadam.yaml`](configs_server/mnist_fedadam.yaml): Server configuration for [FedAdam](https://arxiv.org/abs/2003.00295) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedadagrad.yaml`](configs_server/mnist_fedadagrad.yaml): Server configuration for [FedAdagrad](https://arxiv.org/abs/2003.00295) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedyogi.yaml`](configs_server/mnist_fedyogi.yaml): Server configuration for [FedYogi](https://arxiv.org/abs/2003.00295) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedasync.yaml`](configs_server/mnist_fedasync.yaml): Server configuration for [FedAsync](http://arxiv.org/abs/1903.03934) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedbuffer.yaml`](configs_server/mnist_fedbuffer.yaml): Server configuration for [FedBuffer](https://arxiv.org/abs/2106.06639) algorithm, and each client updates for one epoch for each local training round.
- [`mnist_fedavg_step_optim.yaml`](configs_server/mnist_fedavg_step_optim.yaml): Server configuration for [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedavgm_step_optim.yaml`](configs_server/mnist_fedavgm_step_optim.yaml): Server configuration for [FedAvgMomentum](https://arxiv.org/abs/1909.06335) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedadam_step_optim.yaml`](configs_server/mnist_fedadam_step_optim.yaml): Server configuration for [FedAdam](https://arxiv.org/abs/2003.00295) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedadagrad_step_optim.yaml`](configs_server/mnist_fedadagrad_step_optim.yaml): Server configuration for [FedAdagrad](https://arxiv.org/abs/2003.00295) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedyogi_step_optim.yaml`](configs_server/mnist_fedyogi_step_optim.yaml): Server configuration for [FedYogi](https://arxiv.org/abs/2003.00295) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedasync_step_optim.yaml`](configs_server/mnist_fedasync_step_optim.yaml): Server configuration for [FedAsync](http://arxiv.org/abs/1903.03934) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedbuffer_step_optim.yaml`](configs_server/mnist_fedbuffer_step_optim.yaml): Server configuration for [FedBuffer](https://arxiv.org/abs/2106.06639) algorithm, and each client updates for 100 steps (batches) for each local training round.
- [`mnist_fedcompass_step_optim.yaml`](configs_server/mnist_fedcompass_step_optim.yaml): Server configuration for [FedCompass](https://arxiv.org/abs/2309.14675) algorithm, and each client updates for different number of local steps according to their computing power for each local training round.

Explanations for the provided client configuration files in `configs_client`:
- [`mnist_iid.yaml`](configs_client/mnist_iid.yaml): The client local datasets are equally and randomly partitioned MNIST dataset, which is identically and independently distributed (IID).
- [`mnist_class_noiid.yaml`](configs_client/mnist_class_noiid.yaml): The client local datasets are non-IID partitioned MNIST dataset using the *Class Partition* strategy, check **Appendix D.1.1** of our [FedCompass](https://arxiv.org/pdf/2309.14675.pdf) paper for details.
- [`mnist_dual_dirichlet_noiid.yaml`](configs_client/mnist_dual_dirichlet_noiid.yaml): The client local datasets are non-IID partitioned MNIST dataset using the *Dual Dirichlet Partition* strategy, check **Appendix D.1.2** of our [FedCompass](https://arxiv.org/pdf/2309.14675.pdf) paper for details.
