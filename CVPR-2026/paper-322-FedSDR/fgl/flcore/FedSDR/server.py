import torch
from fgl.flcore.base import BaseServer
from function import *
from fgl.model.gcn import *


class FedSDRServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedSDRServer, self).__init__(
            args, global_data, data_dir, message_pool, device)

    def execute(self):
        N_list = [self.message_pool[f"client_{client_id}"]["num_samples"]
                  for client_id in self.message_pool[f"sampled_clients"]]
        weights = S_client_weights_s(self.args.S_noi_list, N_list)
        with torch.no_grad():
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = weights[it]
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                       self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

        print("weights:", weights)

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }
