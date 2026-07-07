from typing import Callable

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset, Actor, WebKB, CitationFull)
from torch_geometric.loader import (ClusterLoader, DataLoader,
                                    GraphSAINTEdgeSampler,
                                    GraphSAINTNodeSampler,
                                    GraphSAINTRandomWalkSampler,
                                    NeighborLoader, RandomNodeLoader,
                                    LinkNeighborLoader)
from torch_geometric.utils import (index_to_mask, negative_sampling,
                                   to_undirected, remove_self_loops,
                                   degree)
#from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import BaseTransform, RandomNodeSplit, RandomLinkSplit
from torch_geometric.data import Data

import graphgym.register as register
from graphgym.config import cfg
from graphgym.models.transform import create_link_label, neg_sampling_transform
from graphgym.contrib.loader.add_positional_encoding import add_node_attr_nc


def planetoid_dataset(name: str) -> Callable:
    return lambda root: Planetoid(root, name)

register.register_dataset('Cora', planetoid_dataset('Cora'))
register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
register.register_dataset('PubMed', planetoid_dataset('PubMed'))
register.register_dataset('PPI', PPI)

class Add_Node_Feature(BaseTransform):
    def forward(self, data: Data) -> Data:
        return self.__call__(data)

    def __init__(self, dataset_dir=None, transform_type='add'):
        self.dataset_dir = dataset_dir
        self.transform_type = transform_type

    def __call__(self, data: Data) -> Data:
        if self.transform_type == 'remove':
            if cfg.gnn.norm_mode != 'rel_rwpe':
                data.random_walk_pe = None
            if cfg.gnn.norm_mode != 'rel_lepe':
                data.laplacian_eigenvector_pe = None

            if cfg.dataset.task not in ['link_pred']:
                for key in ['edge_index_2hop', 'edge_index_knn', 'edge_index_knn_rwpe', 'edge_index_knn_lepe']:
                    if key == cfg.gnn.neigh:
                        data.edge_index = data[key]
                    data[key] = None
            else:
                data['edge_index_2hop'] = None
                data['edge_index_knn'] = None
                data['edge_index_knn_rwpe'] = None
                data['edge_index_knn_lepe'] = None
            return data
        else:
            attr_dict = add_node_attr_nc(data, self.dataset_dir)

        if self.transform_type == 'add':
            if cfg.gnn.norm_mode == 'rel_rwpe':
                data.random_walk_pe = attr_dict['random_walk_pe']
            if cfg.gnn.norm_mode == 'rel_lepe':
                data.laplacian_eigenvector_pe = attr_dict['laplacian_eigenvector_pe']
            data.avg_degree = attr_dict['avg_degree']

            if cfg.dataset.task not in ['link_pred']:
                for key, value in attr_dict.items():
                    if key == cfg.gnn.neigh:
                        data.edge_index = value
        elif self.transform_type == 'full':
            for key, value in attr_dict.items():
                data[key] = value
        else:
            raise ValueError('Unknown transform type: {}'.format(self.transform_type))

        return data

# Taken from: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/datasets.py
# Found in: https://github.com/pyg-team/pytorch_geometric/discussions/3334
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

# Taken from: https://github.com/lcicek/imdb-binary-gcn/blob/master/utility.py#L24
def initializeNodes(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

def load_pyg(name, dataset_dir, seed=42):
    """load_pyg
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
        dataset._data = Add_Node_Feature(dataset_dir)(dataset[0])
        if cfg.dataset.task == 'node':
            transform = RandomNodeSplit(num_val=0.2,
                                        num_test=0.2)
            dataset._data = transform(dataset[0])
        elif cfg.dataset.task in ['link_pred']:
            transform = RandomLinkSplit(
                num_val=0.2,
                num_test=0.2,
                is_undirected=True,
                neg_sampling_ratio=1.0,
                add_negative_train_samples=False
            )
            train_data, val_data, test_data = transform(dataset[0])
            dataset = [train_data, val_data, test_data]
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    elif 'Coauthor' in name:
        dataset = Coauthor(dataset_dir, name=name[8:])
        dataset._data = Add_Node_Feature(dataset_dir)(dataset[0])
        if cfg.dataset.task == 'node':
            transform = RandomNodeSplit(num_val=0.2,
                                        num_test=0.2)
            dataset._data = transform(dataset[0])
        elif cfg.dataset.task in ['link_pred']:
            transform = RandomLinkSplit(
                num_val=0.2,
                num_test=0.2,
                is_undirected=True,
                neg_sampling_ratio=1.0,
                add_negative_train_samples=False
            )
            train_data, val_data, test_data = transform(dataset[0])
            dataset = [train_data, val_data, test_data]
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    elif 'Amazon' in name:
        dataset = Amazon(dataset_dir, name=name[6:])
        dataset._data = Add_Node_Feature(dataset_dir)(dataset[0])
        if cfg.dataset.task == 'node':
            transform = RandomNodeSplit(num_val=0.2,
                                        num_test=0.2)
            dataset._data = transform(dataset[0])
        elif cfg.dataset.task in ['link_pred']:
            transform = RandomLinkSplit(
                num_val=0.2,
                num_test=0.2,
                is_undirected=True,
                neg_sampling_ratio=1.0,
                add_negative_train_samples=False
            )
            train_data, val_data, test_data = transform(dataset[0])
            dataset = [train_data, val_data, test_data]
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(dataset_dir, name)
        dataset._data = Add_Node_Feature(dataset_dir)(dataset[0])
        if cfg.dataset.task == 'node':
            transform = RandomNodeSplit(num_val=0.2,
                                        num_test=0.2)
            dataset._data = transform(dataset[0])
        elif cfg.dataset.task in ['link_pred']:
            transform = RandomLinkSplit(
                num_val=0.2,
                num_test=0.2,
                is_undirected=True,
                neg_sampling_ratio=1.0,
                add_negative_train_samples=False
            )
            train_data, val_data, test_data = transform(dataset[0])
            dataset = [train_data, val_data, test_data]
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    elif name == 'Actor':
        dataset = Actor(dataset_dir)
        dataset._data = Add_Node_Feature(dataset_dir)(dataset[0])
        if cfg.dataset.task == 'node':
            transform = RandomNodeSplit(num_val=0.2,
                                        num_test=0.2)
            dataset._data = transform(dataset[0])
        elif cfg.dataset.task in ['link_pred']:
            transform = RandomLinkSplit(
                num_val=0.2,
                num_test=0.2,
                is_undirected=True,
                neg_sampling_ratio=1.0,
                add_negative_train_samples=False
            )
            train_data, val_data, test_data = transform(dataset[0])
            dataset = [train_data, val_data, test_data]
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    elif name == 'DBLP':
        dataset = CitationFull(dataset_dir, name)
        dataset._data = Add_Node_Feature(dataset_dir)(dataset[0])
        if cfg.dataset.task == 'node':
            transform = RandomNodeSplit(num_val=0.2,
                                        num_test=0.2)
            dataset._data = transform(dataset[0])
        elif cfg.dataset.task in ['link_pred']:
            transform = RandomLinkSplit(
                num_val=0.2,
                num_test=0.2,
                is_undirected=True,
                neg_sampling_ratio=1.0,
                add_negative_train_samples=False
            )
            train_data, val_data, test_data = transform(dataset[0])
            dataset = [train_data, val_data, test_data]
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    elif name[:3] == 'TU_':
        if cfg.dataset.task in ['graph']:
            # TU_IMDB doesn't have node features
            if name[3:] in ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'REDDIT-BINARY']:
                dataset = TUDataset(dataset_dir, name[3:])
                initializeNodes(dataset)
                data_list = [Add_Node_Feature()(data) for data in dataset]
                dataset.data, dataset.slices = dataset.collate(data_list)
            else:
                dataset = TUDataset(dataset_dir, name[3:], pre_transform=Add_Node_Feature(transform_type='full'))
                data_list = [Add_Node_Feature(transform_type='remove')(data) for data in dataset]
                dataset.data, dataset.slices = dataset.collate(data_list)
        else:
            raise ValueError('Task {} not supported'.format(cfg.dataset.task))
    # elif name == 'Karate':
    #     dataset = KarateClub()
    # elif name == 'MNIST':
    #     dataset = MNISTSuperpixels(dataset_dir)
    # elif name == 'PPI':
    #     dataset = PPI(dataset_dir)
    # elif name == 'QM7b':
    #     dataset = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))

    if cfg.dataset.task == 'graph':
        num_graphs = len(dataset)
        torch.manual_seed(seed)
        indices = torch.randperm(num_graphs)
        val_size = int(0.2 * num_graphs)
        test_size = int(0.2 * num_graphs)
        train_size = num_graphs - val_size - test_size
        train_graph_index = indices[:train_size]
        val_graph_index = indices[train_size:train_size + val_size]
        test_graph_index = indices[train_size + val_size:]
        dataset.data.train_graph_index = train_graph_index
        dataset.data.val_graph_index = val_graph_index
        dataset.data.test_graph_index = test_graph_index

    return dataset

def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)

def load_ogb(name, dataset_dir):
    r"""

    Load OGB dataset objects.


    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if name[:4] == 'ogbn':
        dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(splits.keys()):
            mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        edge_index = to_undirected(dataset.data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])

    elif name[:4] == 'ogbg':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            dataset.transform = neg_sampling_transform
        else:
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes,
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset

def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    # Load from OGB formatted data
    # elif format == 'OGB':
    #     dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(format))
    return dataset

def set_dataset_info(dataset):
    r"""
    Set global dataset information

    Args:
        dataset: PyG dataset object

    """
    if cfg.dataset.task in ['link_pred']:
        # get dim_in and dim_out
        try:
            cfg.share.dim_in = dataset[0].x.shape[1]
        except Exception:
            cfg.share.dim_in = 1
        try:
            if cfg.dataset.task_type == 'classification':
                cfg.share.dim_out = torch.unique(dataset[0].edge_label).shape[0]
            else:
                cfg.share.dim_out = dataset[0].edge_label.shape[1]
        except Exception:
            cfg.share.dim_out = 1

        # count number of dataset splits
        cfg.share.num_splits = len(dataset)
    else:
        # get dim_in and dim_out
        try:
            #cfg.share.dim_in = dataset.data.x.shape[1]
            cfg.share.dim_in = dataset[0].x.shape[1]
        except Exception:
            cfg.share.dim_in = 1
        try:
            if cfg.dataset.task_type == 'classification':
                cfg.share.dim_out = torch.unique(dataset.data.y).shape[0]
            else:
                cfg.share.dim_out = dataset.data.y.shape[1]
        except Exception:
            cfg.share.dim_out = 1

        # count number of dataset splits
        cfg.share.num_splits = 1
        for key in dataset.data.keys():
            if 'val' in key:
                cfg.share.num_splits += 1
                break
        for key in dataset.data.keys():
            if 'test' in key:
                cfg.share.num_splits += 1
                break

def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset

def get_loader(dataset, type, sampler, batch_size, shuffle=True):
    if cfg.dataset.task in ['edge', 'link_pred']:
        if type == 'train':
            dataset = dataset[0]
        elif type == 'val':
            dataset = dataset[1]
        elif type == 'test':
            dataset = dataset[2]
        else:
            raise NotImplementedError("split is not support")
        if sampler == "full_batch":
            loader = DataLoader([dataset],
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=cfg.num_workers,
                                pin_memory=True)
        elif sampler == "neighbor":
            neg_sampling_ratio = 1.0 if type == 'train' else None
            loader = LinkNeighborLoader(dataset,
                                        num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        edge_label_index=dataset.edge_label_index,
                                        edge_label=dataset.edge_label,
                                        neg_sampling_ratio=neg_sampling_ratio,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
        else:
            raise NotImplementedError("%s sampler is not implemented!" % sampler)
    else:
        if sampler == "full_batch" or len(dataset) > 1:
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=cfg.num_workers,
                                pin_memory=True)
        elif sampler == "neighbor":
            mask = '{}_mask'.format(type)
            loader = NeighborLoader(dataset[0],
                                    num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    input_nodes=dataset[0][mask],
                                    num_workers=cfg.num_workers,
                                    pin_memory=True)
        else:
            raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader


# def get_loader_old(dataset, sampler, batch_size, shuffle=True):
#     if cfg.dataset.task in ['edge', 'link_pred']:
#         data = dataset[0]

#         # Prepare the edge label index and labels based on the split
#         split = 'train' if shuffle else 'val'  # Or 'test' depending on the loader
#         edge_index = getattr(data, f'{split}_edge_index')
#         edge_label = getattr(data, f'{split}_edge_label')

#         # Remove the edge labels from the data to prevent information leakage
#         data.edge_label = None
#         data.edge_label_index = None

#         # Use LinkNeighborLoader for link prediction tasks
#         loader_train = LinkNeighborLoader(
#             data,
#             num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
#             edge_label_index=edge_index,
#             edge_label=edge_label,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=cfg.num_workers,
#             pin_memory=True,
#         )
#         for batch in loader_train:
#             print()
#             print(batch)
#             exit()
#         exit()
#     else:
#         if sampler == "full_batch" or len(dataset) > 1:
#             loader_train = DataLoader(dataset,
#                                       batch_size=batch_size,
#                                       shuffle=shuffle,
#                                       num_workers=cfg.num_workers,
#                                       pin_memory=True)
#         elif sampler == "neighbor":
#             loader_train = NeighborLoader(dataset[0],
#                                           num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
#                                           batch_size=batch_size,
#                                           shuffle=shuffle,
#                                           num_workers=cfg.num_workers,
#                                           pin_memory=True)
#             for batch in loader_train:
#                 print()
#                 print(batch)
#                 exit()
#         elif sampler == "random_node":
#             loader_train = RandomNodeLoader(dataset[0],
#                                             num_parts=cfg.train.train_parts,
#                                             shuffle=shuffle,
#                                             num_workers=cfg.num_workers,
#                                             pin_memory=True)
#         elif sampler == "saint_rw":
#             loader_train = \
#                 GraphSAINTRandomWalkSampler(dataset[0],
#                                             batch_size=batch_size,
#                                             walk_length=cfg.train.walk_length,
#                                             num_steps=cfg.train.iter_per_epoch,
#                                             sample_coverage=0,
#                                             shuffle=shuffle,
#                                             num_workers=cfg.num_workers,
#                                             pin_memory=True)
#         elif sampler == "saint_node":
#             loader_train = \
#                 GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
#                                     num_steps=cfg.train.iter_per_epoch,
#                                     sample_coverage=0, shuffle=shuffle,
#                                     num_workers=cfg.num_workers,
#                                     pin_memory=True)
#         elif sampler == "saint_edge":
#             loader_train = \
#                 GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
#                                     num_steps=cfg.train.iter_per_epoch,
#                                     sample_coverage=0, shuffle=shuffle,
#                                     num_workers=cfg.num_workers,
#                                     pin_memory=True)
#         elif sampler == "cluster":
#             loader_train = \
#                 ClusterLoader(dataset[0],
#                             num_parts=cfg.train.train_parts,
#                             save_dir="{}/{}".format(cfg.dataset.dir,
#                                                     cfg.dataset.name.replace(
#                                                         "-", "_")),
#                             batch_size=batch_size, shuffle=shuffle,
#                             num_workers=cfg.num_workers,
#                             pin_memory=True)

#         else:
#             raise NotImplementedError("%s sampler is not implemented!" % sampler)
#     return loader_train

def get_multi_loader(dataset, batch_size, rank, world_size, shuffle=True):
    train_index = dataset[0].train_mask.nonzero().view(-1)
    train_index = train_index.split(train_index.size(0) // world_size)[rank]

    loader_train = NeighborLoader(
        dataset[0],
        input_nodes=train_index,
        num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    return loader_train

def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id],
                       None,
                       cfg.train.sampler,
                       cfg.train.batch_size,
                       shuffle=True)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset,
                       'train',
                       cfg.train.sampler,
                       cfg.train.batch_size,
                       shuffle=True)
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id],
                           None,
                           cfg.val.sampler,
                           cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            split_names = ['val', 'test']
            loaders.append(
                get_loader(dataset,
                           split_names[i],
                           cfg.val.sampler,
                           cfg.train.batch_size,
                           shuffle=False))
    # print('loaders:', loaders)
    # for batch in loaders[0]:
    #     print()
    #     print(batch)

    # exit()

    return loaders

# class RandomLinkMask(BaseTransform):
#     def __init__(self, num_val=0.1, num_test=0.1, is_undirected=True, neg_sampling_ratio=1.0):
#         self.num_val = num_val
#         self.num_test = num_test
#         self.is_undirected = is_undirected
#         self.neg_sampling_ratio = neg_sampling_ratio

#     def __call__(self, data):
#         import torch

#         # Remove self-loops and ensure undirected edges
#         edge_index = remove_self_loops(data.edge_index)[0]
#         if self.is_undirected:
#             edge_index = to_undirected(edge_index)

#         num_nodes = data.num_nodes
#         num_edges = edge_index.size(1)

#         # Split edges into train/val/test
#         num_val = int(num_edges * self.num_val)
#         num_test = int(num_edges * self.num_test)
#         num_train = num_edges - num_val - num_test

#         perm = torch.randperm(num_edges)
#         train_edge_index = edge_index[:, perm[:num_train]]
#         val_edge_index = edge_index[:, perm[num_train:num_train + num_val]]
#         test_edge_index = edge_index[:, perm[num_train + num_val:]]

#         # Negative sampling for training set
#         # num_neg_train = int(self.neg_sampling_ratio * num_train)
#         # train_neg_edge_index = negative_sampling(
#         #     edge_index=train_edge_index,
#         #     num_nodes=num_nodes,
#         #     num_neg_samples=num_neg_train
#         # )

#         # Negative sampling for validation set
#         num_neg_val = int(self.neg_sampling_ratio * num_val)
#         val_neg_edge_index = negative_sampling(
#             edge_index=edge_index,
#             num_nodes=num_nodes,
#             num_neg_samples=num_neg_val
#         )
#         # Remove any overlap with training edges
#         val_neg_edge_index = val_neg_edge_index[:, ~torch.isin(val_neg_edge_index, train_edge_index).all(dim=0)]

#         # Negative sampling for test set
#         num_neg_test = int(self.neg_sampling_ratio * num_test)
#         test_neg_edge_index = negative_sampling(
#             edge_index=edge_index,
#             num_nodes=num_nodes,
#             num_neg_samples=num_neg_test
#         )
#         # Remove any overlap with training and validation edges
#         test_neg_edge_index = test_neg_edge_index[:, ~torch.isin(test_neg_edge_index, torch.cat([train_edge_index, val_edge_index], dim=1)).all(dim=0)]

#         # **Set data.edge_index to train_edge_index for message passing**
#         data.edge_index = train_edge_index
#         data.y = None
#         data.train_mask = None
#         data.val_mask = None
#         data.test_mask = None

#         # Training edges and labels
#         #data.train_edge_index = torch.cat([train_edge_index, train_neg_edge_index], dim=1)
#         #data.train_edge_label = torch.cat([torch.ones(train_edge_index.size(1)), torch.zeros(train_neg_edge_index.size(1))], dim=0).long()
#         data.train_edge_index = train_edge_index
#         data.train_edge_label = torch.ones(train_edge_index.size(1), dtype=torch.long)

#         # Validation edges and labels
#         data.val_edge_index = torch.cat([val_edge_index, val_neg_edge_index], dim=1)
#         data.val_edge_label = torch.cat([torch.ones(val_edge_index.size(1)), torch.zeros(val_neg_edge_index.size(1))], dim=0).long()

#         # Test edges and labels
#         data.test_edge_index = torch.cat([test_edge_index, test_neg_edge_index], dim=1)
#         data.test_edge_label = torch.cat([torch.ones(test_edge_index.size(1)), torch.zeros(test_neg_edge_index.size(1))], dim=0).long()

#         return data
