from . import *

from torch_geometric.datasets import ZINC

class Alchemy(data.InMemoryDataset):

    def __init__(self, root: str, split: str, **kwargs):
        super().__init__(root, **kwargs)
        self.data, self.slices = torch.load(f"{self.processed_dir}/{split}.pt", weights_only=False)

    @property
    def raw_file_names(self):
        return ["./index.npy"]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):
        dataset = datasets.TUDataset(self.root, name="alchemy_full")

        index = f"{self.root}/{self.raw_file_names[0]}"
        index = np.load(index, allow_pickle=True)[()]

        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std

        for split in ["train", "val", "test"]:
            graph_split, ix = self.collate(dataset[index[split]])
            torch.save((graph_split, ix), f"{self.processed_dir}/{split}.pt")


class GraphCount(data.InMemoryDataset):

    NAME = [
        "boat", "chordal6", "chordal4_1", "chordal4_4", "chordal5_13", "chordal5_31", "chordal5_24",          # HOM
        "cycle3", "cycle4", "cycle5", "cycle6", "chordal4", "chordal5",   # ISO
    ]

    def __init__(self, root: str, split: str, name: str, task: str, **kwargs):
        super().__init__(root, **kwargs)

        assert name in self.NAME
        assert task in ["g", "v", "e"]

        graph, i = torch.load(f"{self.processed_dir}/{split}.pt", weights_only=False)
        self.data = data.Data(graph.x, graph.edge_index, y=graph[f"{name}:{task}"])
        self.slices = dict(x=i["x"], edge_index=i["edge_index"], y=i[f"{name}:{task}"])

    @property
    def raw_file_names(self):
        return ["graph.npy", "hom.npy", "iso.npy"]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):

        def to_pyg(A, hom, iso):
            to_pt = lambda count: { f"{name}:{task}": torch.from_numpy(count)
              for name, value in count.items() for task, count in zip(["g", "v", "e"], value) }
            return data.Data(
                x=torch.ones(len(A), 1, dtype=torch.int64), **to_pt(hom), **to_pt(iso),
                edge_index=torch.Tensor(np.vstack(np.where(A != 0))).type(torch.int64),
            )

        load = lambda f: np.load(f"{self.root}/{f}", allow_pickle=True)
        (graph, index), *count = map(load, self.raw_file_names)

        graph = map(to_pyg, graph, *count)

        if self.pre_filter is not None:
            graph = filter(self.pre_filter, graph)

        if self.pre_transform is not None:
            graph = map(self.pre_transform, graph)

        graph = list(graph) # run processing

        normalize = { name: torch.std(torch.cat([G[f"{name}:g"]
                      for G in graph])) for name in self.NAME }

        for split in ["train", "val", "test"]:

            from operator import itemgetter as get
            graph_split = get(*index[()][split])(graph)
            graph_split, ix = self.collate(graph_split)

            for key, std in normalize.items():
                graph_split[f"{key}:g"] /= std
                graph_split[f"{key}:v"] /= std
                graph_split[f"{key}:e"] /= std

            torch.save((graph_split, ix), f"{self.processed_dir}/{split}.pt")
