import copy
from typing import Any
from appfl.misc.deprecation import deprecated


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.scheduler instead."
)
class SchedulerDummy:
    def __init__(
        self, communicator, server: Any, num_clients: int, num_global_epochs: int
    ):
        self.iter = 0
        self.server = server
        self.communicator = communicator
        self.num_clients = num_clients
        self.num_global_epochs = num_global_epochs
        self.client_info = {i: {"step": 0} for i in range(num_clients)}

    def update(self):
        """Schedule update when receive information from one client."""
        client_idx, local_model = self.communicator.recv_local_model_from_client(
            copy.deepcopy(self.server.model)
        )
        self._update(local_model, client_idx)

    def _update(self, local_model: dict, client_idx: int):
        """Update the global model using the local model itself."""
        self.iter += 1
        self.validation_flag = True
        # Update the global model
        self.server.model.to("cpu")
        self.server.update(
            local_model, self.client_info[client_idx]["step"], client_idx
        )
        self.validation_flag = True
        self.client_info[client_idx]["step"] = self.server.global_step
        if self.iter < self.num_global_epochs:
            self._send_model(client_idx)

    def _send_model(self, client_idx: int):
        self.communicator.send_global_model_to_client(
            self.server.model.state_dict(), {"done": False}, client_idx
        )
