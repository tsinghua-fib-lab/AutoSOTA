import torch
import pickle
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import coalesce

from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected


DATASET_LIST = [
    'roman_empire', 'minesweeper', 'questions', 'tolokers',
]
class Chame_Squir_Actor(InMemoryDataset):
    def __init__(self, root='data/', name=None, p2raw=None, transform=None, pre_transform=None):
        if name =='actor':
            name = 'film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(Chame_Squir_Actor, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data')
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def load_custom_data(data_path, to_undirected: bool = True):
    npz_data = np.load(data_path)
    # convert graph to bidirectional
    if to_undirected:
        edges = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
    else:
        edges = npz_data['edges']

    data = Data(
        x=torch.from_numpy(npz_data['node_features']),
        y=torch.from_numpy(npz_data['node_labels']),
        edge_index=torch.from_numpy(edges).T,
        train_mask=torch.from_numpy(npz_data['train_masks']).T,
        val_mask=torch.from_numpy(npz_data['val_masks']).T,
        test_mask=torch.from_numpy(npz_data['test_masks']).T,
    )
    return data


class SingleGraphDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.num_classes = len(torch.unique(data.y))
        self.num_features = data.x.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx: int) -> Data:
        if idx != 0:
            raise ValueError("Invalid index")
        else:
            return self.data


def DataLoader(name):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(osp.join('../data', name), name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        data = load_custom_data(f'../data/{name}_filtered_directed.npz', to_undirected= 'directed' not in name)
        dataset = SingleGraphDataset(data)
    elif name in ['actor']:
        dataset = Chame_Squir_Actor(root='../data/', name=name, transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='../data/',name=name, transform=T.NormalizeFeatures())
    elif name in DATASET_LIST:
        data = load_custom_data(f'../data/{name}.npz', to_undirected= 'directed' not in name)
        dataset = SingleGraphDataset(data)
        return dataset
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset
