import torch


class Gprompt(torch.nn.Module):
    def __init__(self, input_dim):
        super(Gprompt, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        self.max_n_num = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, node_embeddings):
        node_embeddings = node_embeddings * self.weight
        # graph_embedding = graph_embedding.sum(dim=1)
        return node_embeddings
