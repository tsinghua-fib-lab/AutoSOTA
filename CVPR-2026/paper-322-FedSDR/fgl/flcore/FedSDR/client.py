import torch
from fgl.flcore.base import BaseClient
from fgl.model.gcn import *
from function import *


class FedSDRClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedSDRClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device)

    def execute(self, round_id):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        if (round_id + 1) % 10 == 0:
            if self.client_id == 0:
                self.args.S_noi_list = []
                print("computing S_noi")
            self.args.S_noi_list.append(
                compute_S_noi(self.task.data, self.device))

        if self.args.graph_repair and (round_id + 2) % 10 == 0:
            print(f"client {self.client_id} is repairing...")
            input_dim = self.task.data.x.size(1)
            hid_dim = 64
            output_dim = self.args.classes
            gcn_model = GCN(input_dim, hid_dim, output_dim)
            gcn_model = gcn_model.to(self.device)

            with torch.no_grad():
                for (param, global_param) in zip(gcn_model.parameters(), self.message_pool["server"]["weight"]):
                    param.data.copy_(global_param)

            gcn_model.eval()
            with torch.no_grad():
                self.task.data = self.task.data.to(self.device)
                node_features, logits = gcn_model(self.task.data)

            similarity_matrix = compute_client_similarity_matrix(node_features)

            # Confidence-guided edge scoring (IDEA-008)
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
            n = similarity_matrix.size(0)
            entropy_diff = (entropy.unsqueeze(0) - entropy.unsqueeze(1)).abs()
            confidence_boost = 1.0 + 0.3 * entropy_diff
            similarity_matrix = similarity_matrix * confidence_boost

            # Adaptive alpha based on S_noi fidelity
            if hasattr(self.args, 'S_noi_list') and len(self.args.S_noi_list) > 0:
                import torch as _torch
                s_noi_tensor = _torch.tensor(self.args.S_noi_list)
                s_min, s_max = s_noi_tensor.min(), s_noi_tensor.max()
                if s_max - s_min > 1e-8:
                    s_normalized = (s_noi_tensor[self.client_id] - s_min) / (s_max - s_min)
                    adaptive_alpha = float(self.args.alpha + 0.2 * (1.0 - s_normalized.item()))
                    adaptive_alpha = max(0.05, min(0.9, adaptive_alpha))
                else:
                    adaptive_alpha = self.args.alpha
            else:
                adaptive_alpha = self.args.alpha

            self.task.data.edge_index = modify_edges(
                similarity_matrix, self.task.data.edge_index, self.task.data.num_nodes, device=self.device, alpha=adaptive_alpha)

        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
