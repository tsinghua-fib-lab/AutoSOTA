class Communicator:
    def __init__(self):
        self.metrics = {
            "uplink_activation": 0,  # Client -> Server: Hidden States (Embeddings)
            "uplink_scalars": 0,  # Client -> Server: v vector (Projected gradients)
            "downlink_gradient": 0,  # Server -> Client: Activation Gradients
            "downlink_seeds": 0,  # Server -> Client: Random seeds for ZO
            "downlink_scalars": 0,  # Server -> Client: Aggregated v vector
            "uplink_model": 0,  # Client -> Server: Model upload
            "downlink_model": 0,  # Server -> Client: Model download
            "total_uplink": 0,
            "total_downlink": 0,
            "proceeded_samples": 0,
        }

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = 0

    def log_proceeded_samples(self, num_samples):
        self.metrics["proceeded_samples"] += num_samples

    def _get_tensor_size(self, tensor):
        if tensor is None:
            return 0
        return tensor.nelement() * tensor.element_size()

    def log_activation(self, hidden_states):
        size = self._get_tensor_size(hidden_states)
        self.metrics["uplink_activation"] += size
        self.metrics["total_uplink"] += size

    def log_activation_gradient(self, grad_tensor):

        size = self._get_tensor_size(grad_tensor)
        self.metrics["downlink_gradient"] += size
        self.metrics["total_downlink"] += size

    def log_seeds(self, num_seeds, num_clients):
        size = num_seeds * 4 * num_clients
        self.metrics["downlink_seeds"] += size
        self.metrics["total_downlink"] += size

    def log_scalars_uplink(self, num_scalars, num_clients):

        size = num_scalars * 4 * num_clients
        self.metrics["uplink_scalars"] += size
        self.metrics["total_uplink"] += size

    def log_scalars_downlink(self, num_scalars, num_clients):

        size = num_scalars * 4 * num_clients
        self.metrics["downlink_scalars"] += size
        self.metrics["total_downlink"] += size

    def log_model_upload(self, state_dict):
        size = 0
        for param in state_dict.values():
            size += self._get_tensor_size(param)
        self.metrics["uplink_model"] += size
        self.metrics["total_uplink"] += size

    def log_model_download(self, state_dict):
        size = 0
        for param in state_dict.values():
            size += self._get_tensor_size(param)
        self.metrics["downlink_model"] += size
        self.metrics["total_downlink"] += size

    def get_stats(self):
        return self.metrics
