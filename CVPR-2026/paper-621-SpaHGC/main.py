import argparse
import os
from model import HeteroGNN, SpaMHCL
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train import TrainerModel
from pytorch_lightning.strategies import DDPStrategy
import glob
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import anndata
from sklearn.metrics import mean_squared_error

cudnn.benchmark = True
import torch_geometric
import sys
from dataset import HEST_Lymph_Node, HEST_Pancreas_xenium, HEST_Pancreas_Visium, Alex_Nat_Slice, Visium_BC_Slice, \
    SquamousCellCarcinoma, HER2Add_patients
from sklearn.model_selection import LeaveOneOut
sys.path.insert(0, "../")
# from v1.main import KFOLD

XFOLD = [i for i in range(4)]
loo = LeaveOneOut()
# skf = KFold(n_splits=3, shuffle=True, random_state=12345)
KFOLD = []
for x in loo.split(XFOLD):
    KFOLD.append(x)


def get_R(data1, data2, dim=1, func=pearsonr):
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):

        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


def load_dataset(pts, args):
    all_files = glob.glob(f"{args.graph_path}/{args.fold}/graphs_{args.encoder}/*.pt")

    selected_files = []

    for i in all_files:
        filename = os.path.basename(i)
        file_stem = os.path.splitext(filename)[0]

        if file_stem.isdigit() and int(file_stem) in pts:
            graph = torch.load(i)
            selected_files.append(graph)

    return selected_files


def main(args):
    cwd = os.getcwd()

    def write(director, name, *string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director, name), "a") as f:
            f.write(string + "\n")

    store_dir = os.path.join(args.output, str(args.fold))
    print = partial(write, cwd, args.output + "/" + "log_f" + str(args.fold))

    os.makedirs(store_dir, exist_ok=True)

    print(args)

    train_patient, test_patient = KFOLD[args.fold]

    train_dataset = load_dataset(train_patient, args)
    test_dataset = load_dataset(test_patient, args)

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=1,
    )

    test_loader = torch_geometric.loader.DataLoader(
        test_dataset,
        batch_size=1,
    )

    model = SpaMHCL(args.num_layers, args.mdim, args.gene_num)
    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir'])
    config = CONFIG(args.lr, print, args.verbose_step, args.weight_decay, store_dir)

    model = TrainerModel(config, model)

    plt = pl.Trainer(max_epochs=args.epoch, accelerator='gpu', val_check_interval=args.val_interval, logger=False,
                     num_sanity_val_steps=0)
    plt.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    model = SpaMHCL(args.num_layers, args.mdim, args.gene_num)
    model.load_state_dict(torch.load(args.output + f'/{args.fold}/best.pt'))
    model.to("cuda").eval()

    with torch.no_grad():
        for data in tqdm(test_loader):
            x_dict = {k: v.to("cuda") for k, v in data.x_dict.items()}
            edge_index_dict = {k: v.to("cuda") for k, v in data.edge_index_dict.items()}

            _, _, pred_count = model(x_dict, edge_index_dict)
            preds = pred_count.cpu().numpy()
            gt = data["target"]["y"].cpu().numpy()
            adata_pred = anndata.AnnData(preds)
            adata_gt = anndata.AnnData(gt)

    pcc = np.nanmean(get_R(adata_pred, adata_gt)[0])
    rmse = np.sqrt(mean_squared_error(adata_pred.X, adata_gt.X))

    adata_path = rf'./adata_pred_result/{args.dataset}/'
    os.makedirs(adata_path, exist_ok=True)
    adata_pred.write(adata_path + f"{HER2Add_patients[args.fold]}.h5ad")
    print('slice:', HER2Add_patients[args.fold], 'pcc:', pcc, "rmse:", rmse)
    log = f'./logs/training_{args.dataset}_log.txt'
    with open(log, 'a') as f:
        f.write(f'Fold {args.fold} - Test sample: {HER2Add_patients[args.fold]}\n')
        f.write(f'Pcc: {pcc:.4f}\n')
        f.write(f'rmse: {rmse:.4f}\n')
        f.write('--------------------------------\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--fold", default=0, type=int)  # 8
    parser.add_argument("--traget", default=0.9, type=float)
    parser.add_argument("--reference", default=0.9, type=float)
    parser.add_argument("--acce", default="ddp", type=str)
    parser.add_argument("--val_interval", default=0.8, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--verbose_step", default=10, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--mdim", default=1024, type=int)
    parser.add_argument("--output", default="results/her2+/uni", # resnet18 densenet121 vit_base deit_base uni
                        type=str)  # HEST_Lymph_Node HEST_Pancreas_xenium HEST_Pancreas_Visium
    parser.add_argument("--numk", default=6, type=int)
    parser.add_argument("--encoder", default="uni", type=str)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--gene_num", default=785, type=int)  # xenium : 159  Pancreas_Visium : 217
    parser.add_argument("--dataset", default="her2+", type=str)
    parser.add_argument("--graph_path", default=r'./HER2+/',
                        type=str)  # CSCC HEST_Lymph_Node

    args = parser.parse_args()
    main(args)
