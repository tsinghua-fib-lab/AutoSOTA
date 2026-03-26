import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.itertools import Cyclic
from torch.utils.data import IterableDataset
from gluonts.dataset.common import FileDataset


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class CustomPretrainDataset(IterableDataset, ShuffleMixin):
    def __init__(
        self,
        dataset_path,
        retriever,
        mode="training",
        drop_prob=0.2,
        context_length=512,
        prediction_length=64,
        retrieve_lookback_length=64,
        top_k=5,
    ):
        super().__init__()

        assert mode in ("training", "validation", "test")

        self.drop_prob = drop_prob
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.retriever = retriever
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.retrieve_lookback_length = retrieve_lookback_length
        self.top_k = top_k

        # Ensure the dataset path exists
        if not self.dataset_path.is_dir():
            raise ValueError(f"Provided dataset_path {dataset_path} is not a directory.")
        
        # check files, all should be parquet
        if not all([f.suffix == ".parquet" for f in self.dataset_path.iterdir()]):
            raise ValueError("All files in the dataset_path should be parquet files.")

        # lazy loading
        self.dataset = FileDataset(self.dataset_path, freq="1H")

        if self.mode == "training":
            self.dataset = Cyclic(self.dataset)


    def __iter__(self):
        iterable = iter(self.dataset)
        if self.mode == "training":
            while True:
                entry = next(iterable)
                entry = {f: entry[f] for f in ['target', 'distances', 'indices']}
                entry['x'] = entry['target'][:self.context_length]
                entry['y'] = entry['target'][self.context_length:]
                entry['distances'] = entry['distances'][:self.top_k]
                entry['indices'] = entry['indices'][:self.top_k]

                if self.drop_prob > 0:
                    target = entry['target'].copy()
                    drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
                    mask = np.random.choice(
                        [True, False], size=len(target), p=[drop_p, 1 - drop_p]
                    )
                    target[mask] = np.nan
                    entry['target'] = target
                yield entry

        else:
            for entry in iterable:
                entry = {f: entry[f] for f in ['target', 'distances', 'indices']}
                entry['x'] = entry['target'][:self.context_length]
                entry['y'] = entry['target'][self.context_length:]
                yield entry


class Retriever_for_pretrain():
    def __init__(self, retrieval_database_path, dimension, embedding_model):
        self.retrieval_database_path = retrieval_database_path
        self.d = dimension #768
        self.index = None
        self.Y = None
        self.embedding_model = embedding_model

    def build_index(self):
        self.index = faiss.IndexFlatL2(self.d)  # euclidean distance

        database = pd.read_parquet(self.retrieval_database_path)
        embeddings = np.vstack(database["embedding"].to_numpy())
        self.x = database['x'].values
        self.y = database['y'].values
        self.whole_seq = np.concatenate([self.x.tolist(), self.y.tolist()], axis=-1)
        self.index.add(embeddings)

    def embedding(self, x_tensor):
        embeddings, _ = self.embedding_model.embed(x_tensor)
        return embeddings[:, -1, :].float().numpy()

    def search(self, query_vector, top_k, params=None):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # drop first or last
        if params is None:
            distances, indices = self.index.search(query_vector, top_k + 1)
        else:
            distances, indices = self.index.search(query_vector, top_k + 1, params=params)
        # drop first if first distance is 0
        mask = distances[:, 0] == 0
        distances = np.where(
            mask[:, None],
            distances[:, 1:], 
            distances[:, :-1]
        )
        indices = np.where(
            mask[:, None],
            indices[:, 1:], 
            indices[:, :-1]
        )
        
        return indices, distances


