import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from datasets.loader import load_data, DummyDataset
from models.wrappers import MILClassifier as ModelWrapper
from utils.device import select_device


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs.extend(torch.sigmoid(output).cpu().numpy())
            labels.extend(target.cpu().numpy())

    return roc_auc_score(labels, probs)


def main():
    parser = argparse.ArgumentParser(description="Core MIL")
    parser.add_argument('--model', type=str, default='kf_pooling', 
                        choices=['kf_attention', 'kf_layer', 'kf_pooling', 
                                 'hf_attention', 'hf_layer', 'hf_pooling',
                                 'ein_attention', 'ein_layer', 'ein_pooling'])
    parser.add_argument('--dataset', type=str, default='tiger',
                        choices=['tiger', 'fox', 'elephant'])
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.96, metavar='M',
                        help='Learning rate step gamma (default: 0.96)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='hidden dimension d (default: 128)')
    parser.add_argument('--beta', type=float, default=None,
                        help='scaling factor beta; default derives from d')
    parser.add_argument('--num-states', type=int, default=1,
                        help='num pooling states (default: 1)')
    parser.add_argument('--num-memories', type=int, default=64,
                        help='num static memories (default: 64)')
    parser.add_argument('--bag-dropout', type=float, default=0.5,
                        help='bag dropout (default: 0.5)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='device to use (default: auto)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    parser.add_argument('--multiply', action='store_true')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    device = select_device(args.device, no_accel=args.no_accel)

    features, labels = load_data(args)
    features = np.array(features, dtype=object)
    labels = np.array(labels)
    input_dim = features[0].shape[-1]

    all_results = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        train_set = DummyDataset(features[train_idx], labels[train_idx])
        test_set = DummyDataset(features[test_idx], labels[test_idx])

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_set.collate
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            collate_fn=test_set.collate
        )

        model = ModelWrapper(
            mode=args.model,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=1,
            beta=args.beta,
            num_states=args.num_states,
            num_memories=args.num_memories,
            bag_dropout=args.bag_dropout,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            scheduler.step()

        auc = test(model, device, test_loader)
        fold_scores.append(auc)
        print(f"Fold {fold + 1} AUC: {auc:.4f}")

    mean_auc = np.mean(fold_scores)
    all_results.append(mean_auc)
    print(f"Repetition Mean AUC: {mean_auc:.4f}")

    print("\nFinal Results")
    print("Overall Mean AUC:", np.mean(all_results))

    return np.mean(all_results)


if __name__ == "__main__":
    main()
