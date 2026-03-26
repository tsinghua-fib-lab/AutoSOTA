
import torch
from torch_geometric.data import Data
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from sklearn.metrics import precision_score, recall_score, f1_score
from model import GCN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(0)

dataset_name = 'twitter'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('dataset/' + dataset_name + '/dataforGCN_train.csv')
test_data = pd.read_csv('dataset/' + dataset_name + '/dataforGCN_test.csv')

tweet_embeds = torch.load('dataset/' +dataset_name+ '/TweetEmbeds.pt', map_location=device)
tweet_graph = torch.load('dataset/' + dataset_name + '/TweetGraph.pt', map_location=device)

psesudo_data = pd.read_csv(f'dataset/{dataset_name}/twitter_analysis_results.csv')
psesudo_labels = torch.tensor(psesudo_data["analysis"].tolist(), dtype=torch.long).to(device)
psesudo_probs = torch.tensor(psesudo_data["prob"].tolist(), dtype=torch.float).to(device)

label_list_train = train_data["label"].tolist()
label_list_test = test_data["label"].tolist()

labels = []
for label_list in [label_list_train, label_list_test]:
    labels_i = torch.tensor(label_list, dtype=torch.long)
    labels.append(labels_i)

labels = torch.cat(labels, 0)

data = Data(
    x=tweet_embeds.float(),
    edge_index=tweet_graph.coalesce().indices(),
    edge_attr=None,
    train_mask=torch.tensor([True]*len(label_list_train) + [False]*(len(labels)-len(label_list_train))).bool(),
    test_mask=torch.tensor([False]*len(label_list_train) + [True]*(len(labels)-len(label_list_train))).bool(),
    y=labels
).to(device)
num_features = tweet_embeds.shape[1]
num_classes = 2

data.x = torch.cat([data.x, torch.zeros((data.num_nodes, num_classes), device=device)], dim=1)

class UniMP(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels, num_layers,
                heads, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_channels = hidden_channels // heads
                concat = True
            else:
                out_channels = num_classes
                concat = False
            conv = TransformerConv(in_channels, out_channels, heads,
                                concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index)).relu()
        x = self.convs[-1](x, edge_index)
        return x

data.y = data.y.view(-1)
model = UniMP(num_features + num_classes, num_classes, hidden_channels=64,
            num_layers=3, heads=2, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

train_mask = data.train_mask
test_mask = data.test_mask
test_mask_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

def train(label_rate=0.95):
    model.train()

    data.x[:, -num_classes:] = 0

    train_mask_idx = train_mask.nonzero(as_tuple=False).view(-1)
    mask = torch.rand(train_mask_idx.shape[0]) < label_rate
    train_labels_idx = train_mask_idx[mask]  
    train_unlabeled_idx = train_mask_idx[~mask] 

    # Select top 5% of test samples based on pre-computed probabilities
    num_pseudo = int(len(test_mask_idx) * 0.05)
    topk_indices = torch.topk(psesudo_probs, num_pseudo).indices

    test_psesudo_idx = test_mask_idx[topk_indices]
    selected_psesudo_labels = psesudo_labels[topk_indices] 
    data.x[
        torch.cat([train_labels_idx, test_psesudo_idx]), 
        -num_classes:
    ] = F.one_hot(
        torch.cat([data.y[train_labels_idx], selected_psesudo_labels]), 
        num_classes
    ).float()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_unlabeled_idx], data.y[train_unlabeled_idx])
    loss.backward()
    optimizer.step()

    use_labels = True
    n_label_iters = 3

    if use_labels and n_label_iters > 0:
        unlabel_idx = torch.cat([train_unlabeled_idx, data.test_mask.nonzero(as_tuple=False).view(-1)])
        with torch.no_grad():
            for _ in range(n_label_iters):
                torch.cuda.empty_cache()
                out = out.detach()
                data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)
                out = model(data.x, data.edge_index)

    return loss.item()

max_test_acc = 0
max_precision = 0
max_recall = 0
max_f1 = 0

best_epoch = 0

@torch.no_grad()
def test():
    model.eval()

    data.x[:, -num_classes:] = 0

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    
    data.x[train_idx, -num_classes:] = F.one_hot(data.y[train_idx], num_classes).float()

    unlabel_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    n_label_iters = 3
    for _ in range(n_label_iters):
        out = model(data.x, data.edge_index)
        data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)

    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=-1)
    
    y_true = data.y[test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()

    test_acc = (pred == data.y[test_mask]).sum().item() / pred.size(0)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    val_acc = 0

    return val_acc, test_acc, precision, recall, f1

best_precision = 0
best_recall = 0
best_f1 = 0

for epoch in range(1, 3001):
    loss = train()
    val_acc, test_acc, precision, recall, f1 = test()
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
    #     f'Test Acc: {test_acc:.4f}, Precision: {precision:.4f}, '
    #     f'Recall: {recall:.4f}, F1: {f1:.4f}')
    
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_epoch = epoch
    if epoch % 500 == 0:
        import sys
        print(f'Epoch {epoch}: acc={test_acc:.4f}, prec={precision:.4f}, f1={f1:.4f}', flush=True)
        sys.stdout.flush()

print(f'Best Epoch: {best_epoch}, Max Test Acc: {max_test_acc:.4f}, '
    f'Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}')