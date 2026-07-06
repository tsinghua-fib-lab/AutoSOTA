import os
from typing import List, Optional, Tuple

import joblib

import torch
from torch.utils.data import DataLoader, Dataset
from utils.generate_edits import ReactionData


class RetroEditDataset(Dataset):
    def __init__(self, data_dir: str, **kwargs):
        self.data_dir = data_dir
        self.data_files = [
            os.path.join(self.data_dir, file)
            for file in os.listdir(self.data_dir)
            if "batch-" in file
        ]
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Retrieves a particular batch of tensors.

        Parameters
        ----------
        idx: int,
            Batch index
        """
        batch_tensors = torch.load(self.data_files[idx], map_location='cpu')

        return batch_tensors

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        return len(self.data_files)

    def collate(self, attributes: List[Tuple[torch.tensor]]) -> Tuple[torch.Tensor]:
        """Processes the batch of tensors to yield corresponding inputs."""
        assert isinstance(attributes, list)
        assert len(attributes) == 1

        attributes = attributes[0]
        graph_seq_tensors, edit_seq_labels, seq_mask = attributes
        return graph_seq_tensors, edit_seq_labels, seq_mask

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate)


class RetroEvalDataset(Dataset):
    def __init__(self, data_dir: str, data_file: str, use_rxn_class: bool = False):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, data_file)
        self.use_rxn_class = use_rxn_class
        self.dataset = joblib.load(self.data_file)

    def __getitem__(self, idx: int) -> ReactionData:
        """Retrieves the corresponding ReactionData

        Parameters
        ----------
        idx: int,
        Index of particular element
        """
        return self.dataset[idx]

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        return len(self.dataset)

    def collate(self, attributes: List[ReactionData]) -> Tuple[str, List[Tuple], List[List], Optional[List[int]]]:
        """Processes the batch of tensors to yield corresponding inputs."""
        rxns_batch = attributes
        prod_smi = [rxn_data.rxn_smi.split(">>")[-1]
                    for rxn_data in rxns_batch]
        edits = [rxn_data.edits for rxn_data in rxns_batch]
        edits_atom = [rxn_data.edits_atom for rxn_data in rxns_batch]

        if self.use_rxn_class:
            rxn_classes = [rxn_data.rxn_class for rxn_data in rxns_batch]
            return prod_smi, edits, edits_atom, rxn_classes
        else:
            return prod_smi, edits, edits_atom, None

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> DataLoader:
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate)
