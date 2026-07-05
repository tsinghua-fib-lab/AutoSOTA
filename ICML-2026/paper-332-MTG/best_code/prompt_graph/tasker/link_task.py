import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from .task import BaseTask


class LinkTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_data()
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.005, weight_decay=5e-4)
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_data(self):
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(self.device),
            T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                              add_negative_train_samples=False),
        ])
        self.dataset = Planetoid(root='data/Planetoid', name='Cora', transform=transform)
        # After applying the `RandomLinkSplit` transform, the data is transformed from
        # a data object to a list of tuples (train_data, val_data, test_data), with
        # each element representing the corresponding split.

    def train(self, train_data):
        self.gnn.train()
        self.optimizer.zero_grad()
        node_emb = self.gnn(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = self.gnn.decode(node_emb, edge_label_index).view(-1)
        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        self.gnn.eval()
        z = self.gnn(data.x, data.edge_index)
        out = self.gnn.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    def run(self):

        train_data, val_data, test_data = self.dataset[0]

        best_val_auc = final_test_auc = 0
        for epoch in range(1, 101):
            loss = self.train(train_data)
            val_auc = self.test(val_data)
            test_auc = self.test(test_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                  f'Test: {test_auc:.4f}')

        print(f'Final Test: {final_test_auc:.4f}')

        # z = self.gnn(test_data.x, test_data.edge_index)
        # final_edge_index = self.gnn.decode_all(z)
