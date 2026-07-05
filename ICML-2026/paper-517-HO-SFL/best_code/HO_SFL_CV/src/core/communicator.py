class Communicator:
    def __init__(self):
        self.metrics = {
            "uplink_activation": 0,
            "proceeded_samples": 0,
            "uplink_scalars": 0,
            "uplink_model": 0,
            "downlink_gradient": 0,
            "downlink_seeds": 0,
            "downlink_scalars": 0,
            "downlink_model": 0,
            "total_uplink": 0,
            "total_downlink": 0,
        }

    def log_proceeded_samples(self, tensor):
        batch_size = tensor.size(0)
        self.metrics["proceeded_samples"] += batch_size

    def log_activation(self, tensor):
        size = tensor.nelement() * tensor.element_size()
        self.metrics["uplink_activation"] += size
        self.metrics["total_uplink"] += size

    def log_activation_gradient(self, tensor):
        size = tensor.nelement() * tensor.element_size()
        self.metrics["downlink_gradient"] += size
        self.metrics["total_downlink"] += size

    def log_seeds(self, num_seeds):
        size = num_seeds * 4
        self.metrics["downlink_seeds"] += size
        self.metrics["total_downlink"] += size

    def log_scalars_uplink(self, num_scalars):
        size = num_scalars * 4
        self.metrics["uplink_scalars"] += size
        self.metrics["total_uplink"] += size

    def log_scalars_downlink(self, num_scalars):
        size = num_scalars * 4
        self.metrics["downlink_scalars"] += size
        self.metrics["total_downlink"] += size

    def log_model_upload(self, model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size = total_params * 4
        self.metrics["uplink_model"] += size
        self.metrics["total_uplink"] += size

    def log_model_download(self, model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size = total_params * 4
        self.metrics["downlink_model"] += size
        self.metrics["total_downlink"] += size

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = 0

    def get_stats(self):
        return self.metrics
