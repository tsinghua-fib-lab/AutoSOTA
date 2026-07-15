import uuid
from enum import Enum
from collections import OrderedDict


class ClientEndpointStatus(Enum):
    UNAVAILABLE = 0
    AVAILABLE = 1
    RUNNING = 2


class GlobusComputeClientEndpoint:
    def __init__(self, client_idx: int, client_cfg: OrderedDict):
        self.client_idx = client_idx
        self.client_cfg = client_cfg
        self._set_no_runing_task()

    @property
    def status(self):
        """Get the status of the globus compute client enpdoint, and update the status if the client task is finished."""
        if self._status == ClientEndpointStatus.RUNNING:
            if self.future.done():
                self._set_no_runing_task()
        return self._status

    def _set_no_runing_task(self):
        """Clear running task for the federated learning client."""
        self.future = None
        self._status = ClientEndpointStatus.AVAILABLE
        self.task_name = "N/A"
        self.executing_task_id = None

    def cancel_task(self):
        """Cancel the currently running task."""
        self._set_no_runing_task()

    def submit_task(self, gcx, exct_func, *args, callback=None, **kwargs):
        """
        Submit a task to the client globus comput endpoint
        Args:
            gcx: Globus Compute executor for submitting tasks to the Globus Compute client endpoint
            exct_func: Executale function to be run on the client endpoint.
            callback: Callback function called after the submitted task finishes.
            args: arguments for the executable function.
            kwargs: keyword arguments for the executable function.
        """
        if self.status != ClientEndpointStatus.AVAILABLE:
            return "0", None
        gcx.endpoint_id = self.client_cfg.endpoint_id
        self.future = gcx.submit(exct_func, *args, **kwargs)
        if callback is not None:
            self.future.add_done_callback(callback)
        self._status = ClientEndpointStatus.RUNNING
        self.task_name = exct_func.__name__
        self.executing_task_id = str(uuid.uuid4())
        return self.executing_task_id, self.future
