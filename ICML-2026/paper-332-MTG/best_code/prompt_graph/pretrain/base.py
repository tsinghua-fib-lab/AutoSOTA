import torch
from torch.optim import Adam
from ..model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer


class PreTrain(torch.nn.Module):
    def __init__(self, gnn_type='TransformerConv', dataset_name='Cora', input_dim=128, hid_dim=128, gln=2,
                 num_epoch=1000, device: int = 5, graph_list=None, num_workers=0):
        super().__init__()
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.graph_list = graph_list
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim = hid_dim
        self.learning_rate = 0.001
        self.weight_decay = 0.00005
        self.num_workers = num_workers

    def initialize_gnn(self, input_dim, hid_dim):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
            self.gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
            self.gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
            self.gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

#     def load_node_data(self):
#         self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
#         self.data.to(self.device)
#         self.input_dim = self.dataset.num_features
#         self.output_dim = self.dataset.num_classes
