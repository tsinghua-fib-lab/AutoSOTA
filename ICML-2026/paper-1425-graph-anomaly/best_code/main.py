import argparse
from utils import *
import warnings
from train_test import Detector
import numpy as np

def arg_parse(train_datasets, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--k', type=int, default=10, help='Few-shot k value')
    parser.add_argument('--json_dir', type=str, default='./params')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--h_feats', type=int, default=1024)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=128)

    parser.set_defaults(
        trials=5,
        k=50,
        json_dir='./params',
        epoch=100,
        lr=1e-4,
        drop_rate=0,
        h_feats=1024,
        num_hops=2,
        weight_decay=5e-5,
        num_layers=4,
        batch_size=512,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        device=device,
    )
    return parser.parse_args()

torch.cuda.set_device(0)

datasets_train = ["Amazon", "citeseer", "weibo", "ACM", "BlogCatalog", "cs", "photo"]
datasets_test = ["YelpChi", "Reddit", "questions", "Facebook", "Flickr", "cora", "pubmed"]

args = arg_parse(datasets_train, datasets_test)

print('Training on Source Datasets:', datasets_train)
print('Testing on Target Datasets:', datasets_test)

dims = 64
data_train_objs = []
data_test_objs = []

print("\nLoading Train Data...")
for name in datasets_train:
    d = Dataset(dims, name)
    d.sim_conv(args.num_hops)
    d.propagated(args.num_hops)
    data_train_objs.append(d)

print("\nLoading Test Data...")
for name in datasets_test:
    d = Dataset(dims, name)
    d.sim_conv(args.num_hops)
    d.propagated(args.num_hops)
    data_test_objs.append(d)

auc_dict = {}
pre_dict = {}

for t in range(args.trials):
    seed = t+1
    set_seed(seed)
    print(f"\n========== Trial {seed} ==========")

    detector = Detector(args)
    detector.train_mixed(data_train_objs)

    for d_obj in data_test_objs:
        if d_obj.name == 'Facebook':
            args.k = 10
        else:
            args.k = 50
        set_seed(seed)
        test_score = detector.test_one_dataset(d_obj)

        if d_obj.name not in auc_dict:
            auc_dict[d_obj.name] = []
            pre_dict[d_obj.name] = []

        auc_dict[d_obj.name].append(test_score['AUROC'])
        pre_dict[d_obj.name].append(test_score['AUPRC'])

auc_mean_dict, auc_std_dict, pre_mean_dict, pre_std_dict = {}, {}, {}, {}

print("\n========== Final Cross-Domain Results ==========")
for name in datasets_test:
    auc_mean_dict[name] = np.mean(auc_dict[name])
    auc_std_dict[name] = np.std(auc_dict[name])
    pre_mean_dict[name] = np.mean(pre_dict[name])
    pre_std_dict[name] = np.std(pre_dict[name])

    str_result = 'Target: {:<15} | AUROC: {:.4f}+-{:.4f} | AUPRC: {:.4f}+-{:.4f}'.format(
        name,
        auc_mean_dict[name],
        auc_std_dict[name],
        pre_mean_dict[name],
        pre_std_dict[name])
    print(str_result)