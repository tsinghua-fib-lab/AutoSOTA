import numpy as np
import torch
import os
from collections import namedtuple
import torch_geometric
import sys

sys.path.insert(0, "../")
from dataset import Visium_BC_Dataset

import argparse
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import LeaveOneOut
XFOLD = [i for i in range(3)]
loo = LeaveOneOut()
KFOLD = []
for x in loo.split(XFOLD):
    KFOLD.append(x)


parser = argparse.ArgumentParser()
parser.add_argument("--savename", default="./Visium_BC", type=str)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--numk", default=7, type=int)
parser.add_argument("--encoder", default='uni', type=str)
parser.add_argument("--mdim", default=1024, type=str)
parser.add_argument("--index_path", default="Visium_BC", type=str)
parser.add_argument("--emb_path", default=r"./refer/Visium_BC/uni", type=str)
parser.add_argument("--data", default=r"D:\dataset\Alex_NatGen/",
                    type=str)  # D:\dataset\CSCC_data\GSE144240_RAW/

args = parser.parse_args()


def get_edge(x):
    edge_index = torch_geometric.nn.radius_graph(
        x,
        np.sqrt(2),
        None,
        False,
        max_num_neighbors=5,
        flow="source_to_target",
        num_workers=1,
    )

    return edge_index


def get_cross_edge(x):
    l = len(x)
    source = torch.LongTensor(range(l))

    pos = torch.cat([i[3] for i in x]).clone()
    posy = torch.cat([i[4] for i in x]).clone()

    b, n, c = pos.shape
    source = torch.repeat_interleave(source, n)

    ops = torch.cat((pos, posy), -1).view(b * n, -1)
    ops, inverse = torch.unique(ops, dim=0, return_inverse=True)
    unique_pos = ops[:, :c]
    unique_posy = ops[:, c:]

    pos_edge = torch.stack((source, inverse))

    return unique_pos, unique_posy, pos_edge


for fold in range(3):
    savename = args.savename + "/" + str(fold)
    os.makedirs(savename, exist_ok=True)

    temp_arg = namedtuple("arg", ["size", "numk", "mdim", "index_path", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.numk, args.mdim, args.emb_path + f"/{fold}/" + args.index_path, args.emb_path,
                        args.data)
    train_dataset = Visium_BC_Dataset(KFOLD[fold][0], None, None, temp_arg, train=True)

    temp_arg = namedtuple("arg", ["size", "numk", "mdim", "index_path", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.numk, args.mdim, args.emb_path + f"/{fold}/" + args.index_path, args.emb_path,
                        args.data)
    foldername = f"{savename}/graphs_{args.encoder}"
    os.makedirs(foldername, exist_ok=True)

    for iid in range(len(KFOLD[fold][0]) + len(KFOLD[fold][1])):

        dataset = Visium_BC_Dataset([iid], None, None, temp_arg, train=False)
        dataset.min = train_dataset.min.clone()
        dataset.max = train_dataset.max.clone()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1
        )
        img_data = []
        for x in loader:
            pos, p, py, pos_emb, pos_count = x["pos"], x["p_feature"], x["count"], x["pos_feature"], \
                x["pos_count"]
            img_data.append([pos, p, py, pos_emb, pos_count])

        target_edge = get_edge(torch.cat(([i[0] for i in img_data])).clone())

        unique_pos, unique_posy, cross_pos_edge = get_cross_edge(img_data)

        print(target_edge.size(), unique_pos.size(), unique_posy.size(), cross_pos_edge.size())

        data = torch_geometric.data.HeteroData()

        data["target"].pos = torch.cat(([i[0] for i in img_data])).clone()
        data["target"].x = torch.cat(([i[1] for i in img_data])).clone()
        data["target"].x = data["target"].x.squeeze()
        data["target"].y = torch.cat(([i[2] for i in img_data])).clone()

        assert len(data["target"]["pos"]) == len(data["target"]["x"]) == len(data["target"]["y"])

        data["reference"].x = torch.cat((unique_pos, unique_posy), -1)

        data['target', 'TS', 'target'].edge_index = target_edge
        data["reference", "CS", "target"].edge_index = cross_pos_edge[[1, 0]]

        edge_index = torch_geometric.nn.knn_graph(data["reference"]["x"], k=3, loop=False)
        data["reference", "Rs", "reference"].edge_index = edge_index

        print(data)

        torch.save(data, f"{foldername}/{iid}.pt")
