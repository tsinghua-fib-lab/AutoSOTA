import os
import torch
import numpy as np
import torchvision
import math
import time
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Iterator, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import pickle

# import os
# import pickle
# from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
# import numpy as np
# from tqdm import tqdm
# import torch
# from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder


def load_all_graph_files(graph_dir: str) -> List[Dict[str, Any]]:
    """
    Load all *.graph.pkl files under the specified directory into a graph-data list.

    Returns: list whose entries contain file_path and graph_data
    """
    graph_dir = Path(graph_dir)
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_dir}")

    graph_files = sorted(graph_dir.rglob("*.graph.pkl"))
    if not graph_files:
        raise FileNotFoundError(f"No .graph.pkl files found in {graph_dir}")

    print(f"Found {len(graph_files)} graph files")
    graph_data_list = []

    for file in tqdm(graph_files, desc="Loading graph files"):
        with open(file, 'rb') as f:
            graph_data = pickle.load(f)
            graph_data_list.append({
                'file_path': str(file),
                'graph_data': graph_data
            })

    return graph_data_list


def compute_average_edge_weight_scores(nodes: List[Dict], adj_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute each node score as the average weight of its incident edges.

    Args:
        nodes: node list; each node contains 'image_name'
        adj_matrix: adjacency matrix (N, N)

    Returns:
        Dict[image_name] -> score (float)
    """
    N = len(nodes)
    avg_scores = {}

    W = adj_matrix if not hasattr(adj_matrix, 'toarray') else adj_matrix.toarray()

    for i in range(N):
        # Collect all nonzero incident edge weights for this node.
        weights = []
        for j in range(N):
            if i != j and W[i, j] > 0:
                weights.append(W[i, j])

        if len(weights) == 0:
            avg_score = 0.0  # isolated node
        else:
            avg_score = float(np.mean(weights))

        image_name = nodes[i]['image_name']
        avg_scores[image_name] = avg_score

    return avg_scores


def collect_all_scores(graph_data_list: List[Dict]) -> Dict[str, float]:
    """
    Iterate over all graph files and extract the average edge-weight score for each sample.

    Returns:
        Dict[image_name: str -> score: float]
    """
    all_scores = {}

    for item in tqdm(graph_data_list, desc="Computing average edge-weight scores for each graph"):
        graph_data = item['graph_data']
        nodes = graph_data['nodes']
        adj_matrix = graph_data['adj_matrix']

        scores_dict = compute_average_edge_weight_scores(nodes, adj_matrix)

        # Merge into the global dictionary; image_name should be unique in principle.
        for img_name, score in scores_dict.items():
            if img_name in all_scores:
                print(f"Warning: Duplicate image_name: {img_name}, overwriting the previous value")
            all_scores[img_name] = score

    print(f"Computed scores for {len(all_scores)} samples")
    return all_scores


def create_image_to_index_mapping(dataset: Dataset) -> Dict[str, int]:
    """
    Build a mapping from image_path (or basename) to index for an ImageFolder dataset.

    Note: ImageFolder.dataset.imgs is a list of (image_path, class_idx).
    Use the basename of image_path as the key.

    Returns:
        Dict[basename -> index]
    """
    mapping = {}
    for idx, (img_path, _) in tqdm(enumerate(dataset.imgs)):
        img_name = Path(img_path).name
        mapping[img_name] = idx

    print(f"dataset contains {len(mapping)} samples")
    return mapping


def match_scores_to_dataset(
    scores: Dict[str, float],
    dataset: Dataset
) -> List[Tuple[int, float]]:
    """
    Match graph-derived scores to dataset indices via image_name.

    Returns:
        List[(dataset_index, score)], usable for downstream sorting or sampling
    """
    img_to_idx = create_image_to_index_mapping(dataset)
    matched = []
    # matched = {}

    not_found = []

    for img_name, score in scores.items():
        base_name = Path(img_name).name  # keep only the filename
        if base_name in img_to_idx:
            dataset_idx = img_to_idx[base_name]
            matched.append((dataset_idx, score))
            # matched[dataset_idx] = score
        else:
            not_found.append(img_name)

    if not_found:
        print(f"Warning: {len(not_found)} samples were not found in the dataset: {not_found[:5]}...")

    print(f"Matched {len(matched)} samples to the dataset")
    return matched

def create_score_array_by_dataset_index(
    matched_results: List[Tuple[int, float]],
    dataset: Dataset,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Create a score array aligned with dataset indices from matched_results.

    Args:
        matched_results: List[(dataset_idx, score)], matched results
        dataset: Dataset, such as ImageFolder, used to determine the total length
        fill_value: fill value for unmatched samples; np.nan or 0.0 is recommended

    Returns:
        scores: np.ndarray, shape=(len(dataset),), scores[i] is the score for dataset[i]
    """
    n = len(dataset)
    scores = np.full(n, fill_value, dtype=np.float32)

    for idx, score in matched_results:
        if idx < 0 or idx >= n:
            print(f"Warning: Warning: index {idx} is outside the dataset range [0, {n-1}], skipping")
            continue
        scores[idx] = float(score)

    # Count valid entries.
    if np.isnan(fill_value):
        valid_count = np.sum(~np.isnan(scores))
    else:
        valid_count = np.sum(scores != fill_value)

    print(f"Created score array with shape: {scores.shape}")
    print(f"   - matched valid samples: {valid_count}")
    print(f"   - unmatched samples: {n - valid_count} (filled with {fill_value})")

    return scores

class GraphProbPrune(Dataset):
    def __init__(self, dataset, input_graph_dir, ratio = 0.5, num_epoch=None, delta = None):
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.node_scores = np.ones([len(self.dataset)])
        self.final_scores = np.ones([len(self.dataset)])
        self.transform = dataset.transform
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

        # Step 1: load all graphs
        graph_data_list = load_all_graph_files(input_graph_dir)

        # Step 2: compute the average edge-weight score for each sample
        scores = collect_all_scores(graph_data_list)  # image_name -> score

        # Step 3: match scores to the dataset
        matched_results = match_scores_to_dataset(scores, dataset)
        self.edge_scores = create_score_array_by_dataset_index(
                          matched_results=matched_results,
                          dataset=dataset,
                          fill_value=0.0  # or fill_value=0.0
                      )
        print(f"Successfully computed edge scores")


    def __setscore__(self, indices, values):
        self.node_scores[indices] = values
        self.final_scores[indices] = self.node_scores[indices] + self.edge_scores[indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight


    def prune(self, ratio):
        
        b = self.final_scores<=np.max(self.final_scores)
        well_learned_samples = np.where(b)[0]
        pruned_samples=np.where(np.invert(b))[0]
        print('{} now ratio is '.format(ratio))

        if well_learned_samples.shape==(0,) or self.final_scores.min()==self.final_scores.max():
            print('Cut {} samples for next iteration'.format(len(self.dataset)-len(well_learned_samples)))
            self.save_num += len(self.dataset)-len(well_learned_samples)
            np.random.shuffle(well_learned_samples)
            return well_learned_samples
        else:
            prob = torch.softmax(torch.from_numpy(self.final_scores[well_learned_samples]), dim=0).numpy()
            selected = np.random.choice(well_learned_samples, int(ratio * len(well_learned_samples)), p=prob)
            print('{} All samples '.format(len(self.dataset)))
            print('{} final_scores '.format(len(self.final_scores)))
            print('{} well_learned_samples '.format(len(well_learned_samples)))
            print('{} selecetd samples '.format(len(selected)))


            self.reset_weights()
            if len(selected)>0:
                self.weights[selected]=1/ratio
                # pruned_samples=np.append(pruned_samples,selected)
                pruned_samples=selected
            print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
            self.save_num += len(self.dataset)-len(pruned_samples)
            np.random.shuffle(pruned_samples)
            return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.final_scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.start_prune = 2
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        # elif self.seed<self.start_prune:
        #     self.seq = self.infobatch_dataset.no_prune()
        else:
            xiuzheng_ratio = min(1.0, self.infobatch_dataset.ratio + (1-self.infobatch_dataset.ratio) * (1-(self.seed-self.start_prune)/(self.stop_prune-self.start_prune)))
            self.seq = self.infobatch_dataset.prune(xiuzheng_ratio)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
#         self.sampler.reset()
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output

def is_master():
    if not torch.distributed.is_available():
        return True

    if not torch.distributed.is_initialized():
        return True

    if torch.distributed.get_rank()==0:
        return True

    return False

def split_index(t):
    low_mask = 0b111111111111111
    low = torch.tensor([x&low_mask for x in t])
    high = torch.tensor([(x>>15)&low_mask for x in t])
    return low,high

def recombine_index(low,high):
    original_tensor = torch.tensor([(high[i]<<15)+low[i] for i in range(len(low))])
    return original_tensor