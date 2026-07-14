import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Dataset


class NBodyGeometricDataset(Dataset):
    def __init__(
        self, partition="train", max_samples=1e8, dataset_name="nbody_small", model_id="GNN"
    ):
        super().__init__()
        self.partition = partition
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.model_id = model_id

        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition

        if dataset_name == "nbody":
            self.suffix += "_charged5_initvel1"
        elif dataset_name == "nbody_small":
            self.suffix += "_charged5_initvel1small"
        elif dataset_name == "nbody_ood":
            self.suffix += "_charged5_initvel1ood"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.data, self.edges = self.load()

    def load(self):
        loc = np.load(f"data/nbody/loc_{self.suffix}.npy")
        vel = np.load(f"data/nbody/vel_{self.suffix}.npy")
        edges = np.load(f"data/nbody/edges_{self.suffix}.npy")
        charges = np.load(f"data/nbody/charges_{self.suffix}.npy")

        loc, vel = torch.tensor(loc, dtype=torch.float).transpose(2, 3), torch.tensor(
            vel, dtype=torch.float
        ).transpose(2, 3)

        edge_attr_list = []
        rows, cols = [], []
        n_nodes = loc.size(2)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr_list.append(
                        charges.squeeze(-1)[:, i] * charges.squeeze(-1)[:, j]
                    )
                    rows.append(i)
                    cols.append(j)
        charges = torch.tensor(charges, dtype=torch.float)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = (
            torch.tensor(edge_attr_list, dtype=torch.float)
            .transpose(0, 1)
            .unsqueeze(-1)
        )
        loc = loc[: self.max_samples]
        vel = vel[: self.max_samples]
        edge_attr = edge_attr[: self.max_samples]
        charges = charges[: self.max_samples]

        return (loc, vel, edge_attr, charges), edge_index

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        loc, vel, edge_attr, charges = self.data
        edge_index = self.edges

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_ood":
            frame_0, frame_T = 80, 90
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        pos = loc[idx, frame_0]  # [n_nodes, 3]
        vel = vel[idx, frame_0]  # [n_nodes, 3]
        y = loc[idx, frame_T]  # Target position
        x = torch.cat([pos, vel], dim=-1)  # [n_nodes, 6]

        if self.model_id == "GNN":
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr[idx], y=y)

        if self.model_id == "EGNN":
            charges_attr=edge_attr[idx]
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edge_index
            r_ij = torch.sum((pos[rows] - pos[cols])**2, 1).unsqueeze(1)
            r_ij = torch.sqrt(r_ij)
            edge_attr = torch.cat([edge_attr[idx], r_ij], 1).detach()
            data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, vel=vel, charges_attr=charges_attr)
        return data