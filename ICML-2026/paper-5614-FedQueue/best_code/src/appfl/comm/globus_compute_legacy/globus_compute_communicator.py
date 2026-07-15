import time
import uuid
import os.path as osp
import concurrent.futures
from collections import OrderedDict
from appfl.config import ClientTask
from omegaconf import DictConfig, OmegaConf
from globus_compute_sdk import Executor, Client
from .utils.endpoint import GlobusComputeClientEndpoint
from appfl.comm.utils.s3_storage import CloudStorage, LargeObjectWrapper


class GlobusComputeCommunicator:
    def __init__(self, cfg: DictConfig, gcc: Client, logger):
        self.cfg = cfg
        self.logger = logger
        self.executing_tasks = {}
        self.executing_task_futs = {}
        self.clients = {
            client_idx: GlobusComputeClientEndpoint(client_idx, client_cfg)
            for client_idx, client_cfg in enumerate(self.cfg.clients)
        }
        self.gcx = Executor(funcx_client=gcc)
        self.use_s3bucket = cfg.server.s3_bucket is not None
        if self.use_s3bucket:
            self.logger.info(
                f"Using S3 bucket {cfg.server.s3_bucket} for model transfer."
            )
            CloudStorage.init(
                cfg,
                s3_tmp_dir=osp.join(cfg.server.output_dir, "tmp"),
                logger=self.logger,
            )
            cfg.server.s3_creds = ""

    def __register_task(self, task_id, task_fut, client_id, task_name):
        """Register new client task to the list of executing tasks - call after task submission"""
        self.executing_tasks[task_id] = OmegaConf.structured(
            ClientTask(
                task_id=task_id,
                task_name=task_name,
                client_idx=client_id,
                start_time=time.time(),
            )
        )
        self.executing_task_futs[task_fut] = task_id

    def __set_task_success_status(self, task_id, completion_time, client_log=None):
        """Change the status status of the given task to finished."""
        self.executing_tasks[task_id].end_time = float(completion_time)
        self.executing_tasks[task_id].success = True
        self.executing_tasks[task_id].log = client_log

    def __finalize_task(self, task_id):
        """Finalize the given task."""
        # Save task to log file
        self.cfg.logging_tasks.append(self.executing_tasks[task_id])
        # Reset client
        self.clients[self.executing_tasks[task_id].client_idx].status
        # Remove from executing tasks list
        self.executing_tasks.pop(task_id)

    def __process_client_results(self, results):
        """Process results from clients, download file from S3 if necessary"""
        if not CloudStorage.is_cloud_storage_object(results) or not self.use_s3bucket:
            return results
        else:
            return CloudStorage.download_object(
                results,
                to_device=self.cfg.server.device,
                delete_cloud=True,
                delete_local=True,
            )

    def __handle_params(self, args, kwargs):
        """Parse function's parameters and upload large parameters to S3"""
        # Handling args
        _args = list(args)
        for i, obj in enumerate(_args):
            if type(obj) is LargeObjectWrapper:
                if (not obj.can_send_directly) and self.use_s3bucket:
                    _args[i] = CloudStorage.upload_object(obj, register_for_clean=True)
                else:
                    _args[i] = obj.data
        args = tuple(_args)
        # Handling kwargs
        for k in kwargs:
            obj = kwargs[k]
            if type(obj) is LargeObjectWrapper:
                if (not obj.can_send_directly) and self.use_s3bucket:
                    kwargs[k] = CloudStorage.upload_object(obj, register_for_clean=True)
                else:
                    kwargs[k] = obj.data
        return args, kwargs

    def __handle_globus_compute_result(self, res, task_id, do_finalize=True):
        """Handle returned results from Globus Compute endpoint."""
        # Obtain the client logs
        client_log = {}
        if type(res) is tuple:
            res, client_log = res
        else:
            client_log = None
        # Download client results from S3 bucker if necessary
        res = self.__process_client_results(res)
        # Change the status of the finished task
        self.__set_task_success_status(task_id, time.time(), client_log)
        # Log the receipt of the task results
        client_name = self.clients[
            self.executing_tasks[task_id].client_idx
        ].client_cfg.name
        client_task_name = self.executing_tasks[task_id].task_name
        self.logger.info(
            "Received results of task '{}' from {}.".format(
                client_task_name, client_name
            )
        )
        # Finalize the experiment if necessary
        if do_finalize:
            self.__finalize_task(task_id)
        return res, client_log

    def decay_learning_rate(self):
        """Perform learning rate decay."""
        self.cfg.fed.args.optim_args.lr *= self.cfg.fed.args.server_lr_decay_exp_gamma
        self.logger.info(
            "Learning rate is set to %.06f." % (self.cfg.fed.args.optim_args.lr)
        )

    def set_learning_rate(self, lr, client_idx=None):
        """Set learning rate."""
        self.cfg.fed.args.optim_args.lr = lr
        if client_idx is None:
            self.logger.info(
                "Learning rate is set to %.06f." % (self.cfg.fed.args.optim_args.lr)
            )
        else:
            client_name = self.clients[client_idx].client_cfg.name
            self.logger.info(
                "Learning rate is set to %.06f at %s."
                % (self.cfg.fed.args.optim_args.lr, client_name)
            )

    def set_local_steps(self, num_local_steps, client_idx=None):
        """Set client local training steps."""
        self.cfg.fed.args.num_local_steps = num_local_steps
        if client_idx is None:
            self.logger.info(
                "Local training steps is set to %d."
                % (self.cfg.fed.args.num_local_steps)
            )
        else:
            client_name = self.clients[client_idx].client_cfg.name
            self.logger.info(
                "Local training steps is set to %d at %s."
                % (self.cfg.fed.args.num_local_steps, client_name)
            )

    def send_task_to_all_clients(self, exct_func, *args, silent=False, **kwargs):
        """Broadcast an executable task with all arguments to all federated learning clients."""
        # Prepare args, kwargs before sending to clients
        args, kwargs = self.__handle_params(args, kwargs)

        # Execute training tasks at clients
        for client_idx, _ in enumerate(self.cfg.clients):
            if self.use_s3bucket and exct_func.__name__ == "client_training":
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_idx}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                kwargs["local_model_key"] = local_model_key
                kwargs["local_model_url"] = local_model_url
            task_id, task_fut = self.clients[client_idx].submit_task(
                self.gcx, exct_func, *(self.cfg, client_idx, *args), **kwargs
            )
            self.__register_task(task_id, task_fut, client_idx, exct_func.__name__)

        # Logging
        if not silent:
            for task_id in self.executing_tasks:
                self.logger.info(
                    "Task '%s' is assigned to %s."
                    % (
                        exct_func.__name__,
                        self.cfg.clients[self.executing_tasks[task_id].client_idx].name,
                    )
                )

    def send_task_to_one_client(
        self, client_idx, exct_func, *args, silent=False, **kwargs
    ):
        """Send an executable task with all arguments to one federated learning client."""
        # Prepare args, kwargs before sending to clients
        args, kwargs = self.__handle_params(args, kwargs)

        if self.use_s3bucket and exct_func.__name__ == "client_training":
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_idx}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            kwargs["local_model_key"] = local_model_key
            kwargs["local_model_url"] = local_model_url
        task_id, task_fut = self.clients[client_idx].submit_task(
            self.gcx, exct_func, *(self.cfg, client_idx, *args), **kwargs
        )
        self.__register_task(task_id, task_fut, client_idx, exct_func.__name__)

        # Logging
        if not silent:
            self.logger.info(
                f"Task {exct_func.__name__} is assigned to {self.cfg.clients[client_idx].name}."
            )

    def receive_sync_endpoints_updates(self):
        """Receive synchronous updates from all client endpoints."""
        client_results = []
        client_logs = OrderedDict()
        while len(self.executing_task_futs):
            fut = next(concurrent.futures.as_completed(list(self.executing_task_futs)))
            task_id = self.executing_task_futs[fut]
            try:
                result = fut.result()
                client_idx = self.executing_tasks[task_id].client_idx
                client_local_result, client_logs[client_idx] = (
                    self.__handle_globus_compute_result(result, task_id)
                )
                client_results.append(client_local_result)
                del self.executing_task_futs[fut]
            except Exception as exc:
                self.logger.error(
                    "Task %s on %s is failed with an error."
                    % (
                        self.executing_tasks[task_id].task_name,
                        self.cfg.clients[self.executing_tasks[task_id].client_idx].name,
                    )
                )
                raise exc
        return client_results, client_logs

    def receive_async_endpoint_update(self):
        """Receive asynchronous update from only one client endpoint."""
        assert len(self.executing_task_futs), (
            "There is no active client endpoint running tasks."
        )
        client_log = OrderedDict()
        try:
            fut = next(concurrent.futures.as_completed(list(self.executing_task_futs)))
            result = fut.result()
            task_id = self.executing_task_futs[fut]
            client_idx = self.executing_tasks[task_id].client_idx
            client_local_result, client_log[client_idx] = (
                self.__handle_globus_compute_result(result, task_id)
            )
            del self.executing_task_futs[fut]
        except Exception as exc:
            self.logger.error(
                "Task %s on %s is failed with an error."
                % (
                    self.executing_tasks[task_id].task_name,
                    self.cfg.clients[self.executing_tasks[task_id].client_idx].name,
                )
            )
            raise exc
        return client_idx, client_local_result, client_log

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        for task_fut in self.executing_task_futs:
            task_fut.cancel()
            task_id = self.executing_task_futs[task_fut]
            client_idx = self.executing_tasks[task_id].client_idx
            self.clients[client_idx].cancel_task()
        self.executing_task_futs = {}
        self.executing_tasks = {}

    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
        self.logger.info("Shutting down all clients......")
        self.gcx.shutdown(wait=False, cancel_futures=True)
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )
