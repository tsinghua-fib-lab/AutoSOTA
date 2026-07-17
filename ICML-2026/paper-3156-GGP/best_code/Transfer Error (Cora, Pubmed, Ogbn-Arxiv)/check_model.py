import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import sys
sys.path.insert(0, "/repo/Transfer Error (Cora, Pubmed, Ogbn-Arxiv)")
from Stretched_models import load_model, get_standard

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data', name='Cora')
data = dataset[0].to(device)
data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
data.x = data.x.float()

model = load_model(2, 'Cora_model_2_32.pt', data.num_features, 32, 7, device)
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    std_all_nodes = get_standard(data, model, device)
    print(f"Full-graph test accuracy: {acc:.4f}")
    print(f"Full-graph train accuracy: {train_acc:.4f}")
    print(f"Repr mean: {std_all_nodes.mean().item():.4f}, std: {std_all_nodes.std().item():.4f}")
    print(f"Repr min: {std_all_nodes.min().item():.4f}, max: {std_all_nodes.max().item():.4f}")
