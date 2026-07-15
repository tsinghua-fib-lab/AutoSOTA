import os
import time
import boto3
import pathlib
import requests
from omegaconf import OmegaConf
from appfl.misc.utils import get_last_function_name
from globus_sdk import AccessTokenAuthorizer
from globus_sdk.scopes import AuthScopes, ComputeScopes
from globus_compute_sdk import Client
from globus_compute_sdk.errors import TaskPending
from globus_compute_sdk.serialize import CombinedCode
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager


def endpoint_test():
    """Endpoint health status test function."""
    import torch

    return torch.cuda.is_available()


class APPFLxDataExchanger:
    def __init__(
        self,
        base_dir,
    ):
        self.s3 = boto3.client(
            service_name="s3",
            region_name="us-east-1",
            # aws_access_key_id=aws_access_key_id,
            # aws_secret_access_key=aws_secret_access_key
        )
        # self.EXP_DIR = 'experiment'
        self.STATUS_CHECK_TIMES = 30
        self.S3_BUCKET_NAME = "appflx-bucket"
        self.S3_MODEL_BUCKET_NAME = "appflx-models"
        self.task_id = self._get_ecs_taskid()
        self.base_dir = base_dir

    def _get_gcc(self, compute_token, openid_token):
        """Get the Globus Compute Client objects."""
        compute_login_manager = AuthorizerLoginManager(
            authorizers={
                ComputeScopes.resource_server: AccessTokenAuthorizer(compute_token),
                AuthScopes.resource_server: AccessTokenAuthorizer(openid_token),
            }
        )
        compute_login_manager.ensure_logged_in()
        gcc = Client(
            login_manager=compute_login_manager,
            code_serialization_strategy=CombinedCode(),
        )
        return gcc

    def _s3_download(self, bucket_name, key_name, file_folder, file_name):
        """
        Download file with `key_name` from S3 bucket `bucket_name`, and store it locally to `file_name`.
        Return true if the file exists and gets downloaded successfully, return false otherwise.
        """
        try:
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)
            self.s3.download_file(
                Bucket=bucket_name,
                Key=key_name,
                Filename=os.path.join(file_folder, file_name),
            )
            return True
        except Exception:
            # print(f'Error in downloading {key_name} from {bucket_name}: {e}')
            return False

    def _s3_upload(self, bucket_name, key_name, file_name, delete_local=True):
        """
        Upload the local file with name `file_name` to the S3 bucket `bucket_name` and save it as `key_name`.
        User can choose whether to delete the uploaded local file by specifying `delete_local`
        """
        try:
            self.s3.upload_file(Filename=file_name, Bucket=bucket_name, Key=key_name)
            if delete_local:
                os.remove(file_name)
            return True
        except Exception:
            # print(f'Error in uploading {key_name} from {bucket_name}: {e}')
            return False

    def _get_ecs_taskid(self):
        """Obtain the ECS task id within the ECS container"""
        try:
            metadata_url = "http://169.254.170.2/v2/metadata"
            metadata_response = requests.get(
                metadata_url, headers={"Metadata": "true"}, timeout=1
            )
            metadata = metadata_response.json()
            task_arn = metadata["TaskARN"]
            task_id = task_arn.split("/")[-1]
        except:  # noqa: E722
            task_id = "randomid"  # Only used for dev testing purpose
        return task_id

    def _prepare_data_dir(self):
        """Prepare the data directory for the current task."""
        _home = pathlib.Path.home()
        self.data_dir = os.path.join(_home, ".appfl", "appflx", self.task_id)
        if not os.path.exists(self.data_dir):
            pathlib.Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def _download_aidr_configurations(self):
        """Download the configuration file for the AI data readiness inspection."""
        configuration_s3_key = (
            f"{self.base_dir}/{self.task_id}/data_readiness_config.yaml"
        )
        self._prepare_data_dir()
        if not self._s3_download(
            bucket_name=self.S3_BUCKET_NAME,
            key_name=configuration_s3_key,
            file_folder=self.data_dir,
            file_name="data_readiness_config.yaml",
        ):
            raise Exception(
                f"Failed to download configuration file from {configuration_s3_key}"
            )
        server_config = OmegaConf.load(
            os.path.join(self.data_dir, "data_readiness_config.yaml")
        )
        # Download and process the client configuration files
        group_id = server_config.appflx_configs.group_id
        group_members = server_config.appflx_configs.group_members.split(",")
        compute_token = server_config.appflx_configs.compute_token
        openid_token = server_config.appflx_configs.openid_token
        gcc = self._get_gcc(compute_token, openid_token)
        test_function_id = gcc.register_function(endpoint_test)

        client_configs = []
        for member_id in group_members:
            client_config_key = f"{group_id}/{member_id}/client.yaml"
            dataloader_key = f"{group_id}/{member_id}/dataloader.py"
            if self._s3_download(
                bucket_name=self.S3_BUCKET_NAME,
                key_name=client_config_key,
                file_folder=os.path.join(self.data_dir, member_id),
                file_name="client.yaml",
            ) and self._s3_download(
                bucket_name=self.S3_BUCKET_NAME,
                key_name=dataloader_key,
                file_folder=os.path.join(self.data_dir, member_id),
                file_name="dataloader.py",
            ):
                client_config = OmegaConf.load(
                    os.path.join(self.data_dir, member_id, "client.yaml")
                )
                client_config.data_configs.dataset_path = os.path.join(
                    self.data_dir, member_id, "dataloader.py"
                )
                client_config.data_configs.dataset_name = get_last_function_name(
                    client_config.data_configs.dataset_path
                )
                client_endpoint_id = client_config.endpoint_id
                task_id = gcc.run(
                    endpoint_id=client_endpoint_id, function_id=test_function_id
                )
                for _ in range(self.STATUS_CHECK_TIMES):
                    try:
                        time.sleep(1)
                        gcc.get_result(task_id)
                        # print(f"Client {client_endpoint_id} is started")
                        client_configs.append(client_config)
                        break
                    except TaskPending:
                        continue
                    except Exception:
                        # print(e)
                        break
        if len(client_configs) == 0:
            raise Exception("All client endpoints are not started")

        server_config.server_configs.scheduler_kwargs = {
            "num_clients": len(client_configs)
        }
        return server_config, client_configs

    def upload_results(self, files):
        for file_name, file_path in files.items():
            file_key = f"{self.base_dir}/{self.task_id}/{file_name}"
            self._s3_upload(
                bucket_name=self.S3_BUCKET_NAME,
                key_name=file_key,
                file_name=file_path,
                delete_local=False,
            )

    def download_configurations(self, run_aidr_only=False):
        """Download the configuration file from S3 bucket."""
        if run_aidr_only:
            return self._download_aidr_configurations()
        # Download and process the server configuration file
        configuration_s3_key = f"{self.base_dir}/{self.task_id}/appfl_config.yaml"
        model_s3_key = f"{self.base_dir}/{self.task_id}/model.py"
        loss_s3_key = f"{self.base_dir}/{self.task_id}/loss.py"
        metric_s3_key = f"{self.base_dir}/{self.task_id}/metric.py"
        self._prepare_data_dir()
        if not self._s3_download(
            bucket_name=self.S3_BUCKET_NAME,
            key_name=configuration_s3_key,
            file_folder=self.data_dir,
            file_name="appfl_config.yaml",
        ):
            raise Exception(
                f"Failed to download configuration file from {configuration_s3_key}"
            )

        if not self._s3_download(
            bucket_name=self.S3_BUCKET_NAME,
            key_name=model_s3_key,
            file_folder=self.data_dir,
            file_name="model.py",
        ):
            raise Exception(f"Failed to download model file from {model_s3_key}")

        if not self._s3_download(
            bucket_name=self.S3_BUCKET_NAME,
            key_name=loss_s3_key,
            file_folder=self.data_dir,
            file_name="loss.py",
        ):
            raise Exception(f"Failed to download loss file from {loss_s3_key}")

        if not self._s3_download(
            bucket_name=self.S3_BUCKET_NAME,
            key_name=metric_s3_key,
            file_folder=self.data_dir,
            file_name="metric.py",
        ):
            raise Exception(f"Failed to download metric file from {metric_s3_key}")

        server_config = OmegaConf.load(os.path.join(self.data_dir, "appfl_config.yaml"))
        server_config.client_configs.model_configs.model_path = os.path.join(
            self.data_dir, "model.py"
        )
        server_config.client_configs.train_configs.loss_fn_path = os.path.join(
            self.data_dir, "loss.py"
        )
        server_config.client_configs.train_configs.metric_path = os.path.join(
            self.data_dir, "metric.py"
        )

        # Download and process the client configuration files
        group_id = server_config.appflx_configs.group_id
        group_members = server_config.appflx_configs.group_members.split(",")
        compute_token = server_config.appflx_configs.compute_token
        openid_token = server_config.appflx_configs.openid_token
        gcc = self._get_gcc(compute_token, openid_token)
        test_function_id = gcc.register_function(endpoint_test)

        client_configs = []
        for member_id in group_members:
            client_config_key = f"{group_id}/{member_id}/client.yaml"
            dataloader_key = f"{group_id}/{member_id}/dataloader.py"
            if self._s3_download(
                bucket_name=self.S3_BUCKET_NAME,
                key_name=client_config_key,
                file_folder=os.path.join(self.data_dir, member_id),
                file_name="client.yaml",
            ) and self._s3_download(
                bucket_name=self.S3_BUCKET_NAME,
                key_name=dataloader_key,
                file_folder=os.path.join(self.data_dir, member_id),
                file_name="dataloader.py",
            ):
                client_config = OmegaConf.load(
                    os.path.join(self.data_dir, member_id, "client.yaml")
                )
                client_config.data_configs.dataset_path = os.path.join(
                    self.data_dir, member_id, "dataloader.py"
                )
                client_config.data_configs.dataset_name = get_last_function_name(
                    client_config.data_configs.dataset_path
                )
                client_endpoint_id = client_config.endpoint_id
                task_id = gcc.run(
                    endpoint_id=client_endpoint_id, function_id=test_function_id
                )
                for _ in range(self.STATUS_CHECK_TIMES):
                    try:
                        time.sleep(1)
                        gcc.get_result(task_id)
                        # print(f"Client {client_endpoint_id} is started")
                        client_configs.append(client_config)
                        break
                    except TaskPending:
                        continue
                    except Exception:
                        # print(e)
                        break
        if len(client_configs) == 0:
            raise Exception("All client endpoints are not started")
        server_config.server_configs.num_clients = len(client_configs)
        # [Deprecated] The following code is used for the old version of the server configuration
        # and should no longer needed.
        if hasattr(server_config.server_configs, "aggregator_kwargs"):
            server_config.server_configs.aggregator_kwargs.num_clients = len(
                client_configs
            )
        else:
            server_config.server_configs.aggregator_kwargs = OmegaConf.create(
                {"num_clients": len(client_configs)}
            )
        if hasattr(server_config.server_configs, "scheduler_kwargs"):
            server_config.server_configs.scheduler_kwargs.num_clients = len(
                client_configs
            )
        else:
            server_config.server_configs.scheduler_kwargs = OmegaConf.create(
                {"num_clients": len(client_configs)}
            )
        return server_config, client_configs
