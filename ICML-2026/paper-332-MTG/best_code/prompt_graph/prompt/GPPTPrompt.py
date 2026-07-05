import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SimpleMeanConv(MessagePassing):
    def __init__(self):
        super(SimpleMeanConv, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

# def perform_kmeans(h, num_clusters, device):
#     # gpu kmeans
#     cluster_ids, cluster_centers = kmeans(X=h, num_clusters=num_clusters, distance='euclidean', device=device)
#     return cluster_centers

class GPPTPrompt(torch.nn.Module):
    def __init__(self, n_hidden, center_num, n_classes, device):
        super(GPPTPrompt, self).__init__()
        self.center_num = center_num
        self.n_classes = n_classes
        self.device = device
        self.StructureToken = torch.nn.Linear(n_hidden, center_num, bias=False)
        self.StructureToken = self.StructureToken.to(device)  # structure token
        self.TaskToken = torch.nn.ModuleList()
        for i in range(center_num):
            self.TaskToken.append(torch.nn.Linear(2 * n_hidden, n_classes, bias=False))  # task token
        self.TaskToken = self.TaskToken.to(device)

    def _initialize_weights(self, layer):
        import torch.nn.init as init
        if isinstance(layer, nn.Linear):
            # You can choose any initialization method. Here, we use Xavier initialization.
            init.xavier_uniform_(layer.weight)
            # If you have bias, you can initialize it as well, but in this case, bias is False.
            # if layer.bias is not None:
            # init.constant_(layer.bias, 0)

    def weigth_init(self, h, edge_index, label, index):
        conv = SimpleMeanConv()
        h = conv(h, edge_index)
        features = h[index]
        labels = label[index.long()]

        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(features.detach().cpu())
        temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.StructureToken.weight.data = temp.clone().detach()

        p = []
        for i in range(self.n_classes):
            if (labels == i).sum() == 0:  # Bug in ProG
                print(f"Class {i} has no samples")
                p.append(torch.randn(1, features.size(1)).to(self.device))
            else:
                p.append(features[labels == i].mean(dim=0).view(1, -1))
        temp = torch.cat(p, dim=0).to(self.device)
        for i in range(self.center_num):
            self.TaskToken[i].weight.data = temp.clone().detach()
            # enzymes 600 images. 1-shot, 6 types of nodes, screen out 6 images. do..  while() --> Batch bg, bg.y of 6 graphs

    def update_StructureToken_weight(self, h):

        if h.size(0) > 20000:
            cluster_ids_x, cluster_centers = kmeans(X=h, num_clusters=self.center_num, distance='euclidean',
                                                    device=self.device)
            self.StructureToken.weight.data = cluster_centers.clone()
        else:
            cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(h.detach().cpu())
            temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
            self.StructureToken.weight.data = temp.clone()

    def get_TaskToken(self):
        pros = []
        for name, param in self.named_parameters():
            if name.startswith('TaskToken.'):
                pros.append(param)
        return pros

    def get_StructureToken(self):
        for name, param in self.named_parameters():
            if name.startswith('StructureToken.weight'):
                pro = param
        return pro

    def get_mid_h(self):
        return self.fea

    def forward(self, h, edge_index):
        device = h.device
        conv = SimpleMeanConv()
        h = conv(h, edge_index)
        self.fea = h
        out = self.StructureToken(h)
        index = torch.argmax(out, dim=1)
        out = torch.zeros(h.shape[0], self.n_classes).to(device)
        for i in range(self.center_num):
            out[index == i] = self.TaskToken[i](h[index == i])
        return out


def kmeans(X, num_clusters, distance='euclidean', device='cuda', max_iter=100, tol=1e-4):
    """
    Perform KMeans clustering on the input data X.

    Parameters:
    X : torch.Tensor
        Input data, shape [n_samples, n_features]
    num_clusters : int
        Number of clusters
    distance : str
        Distance metric ('euclidean' is currently supported)
    device : str
        Device to use ('cuda' or 'cpu')
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence

    Returns:
    cluster_ids_x : torch.Tensor
        Cluster assignment for each sample
    cluster_centers : torch.Tensor
        Cluster centers
    """

    if distance != 'euclidean':
        raise NotImplementedError("Currently only 'euclidean' distance is supported.")

    X = X.to(device)
    n_samples, n_features = X.shape

    # Randomly initialize cluster centers
    random_indices = torch.randperm(n_samples)[:num_clusters]
    cluster_centers = X[random_indices]

    for i in range(max_iter):
        # Compute distances and assign clusters
        distances = torch.cdist(X, cluster_centers)
        cluster_ids_x = torch.argmin(distances, dim=1)

        # Compute new cluster centers
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for k in range(num_clusters):
            cluster_k = X[cluster_ids_x == k]
            if len(cluster_k) > 0:
                new_cluster_centers[k] = cluster_k.mean(dim=0)

        # Check for convergence
        if torch.norm(new_cluster_centers - cluster_centers) < tol:
            break

        cluster_centers = new_cluster_centers

    return cluster_ids_x, cluster_centers

# # Example usage
# h = torch.randn(160000, 128).to('cuda')
# cluster_ids_x, cluster_centers = kmeans(X=h, num_clusters=10, distance='euclidean', device='cuda')
