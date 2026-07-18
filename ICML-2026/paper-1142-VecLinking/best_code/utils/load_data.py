import numpy as np
from scipy.spatial.distance import cdist
import torch
import os
from loguru import logger
from typing import List, Tuple, Union, Dict
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import umap

# Try to import BEIR for la2m partition
try:
    from beir.datasets.data_loader import GenericDataLoader
    from beir import util as beirUtil
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False


class SimpleBeirDataset:
    """Simple dataset class for la2m partition that provides required interface."""
    def __init__(self, corpus_ids: list):
        self.corpus_ids = corpus_ids
        self.id_to_internal = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}

    def batch_original_ids_to_internal_indices(self, doc_ids):
        """Convert original document IDs to internal indices."""
        return np.array([self.id_to_internal[doc_id] for doc_id in doc_ids if doc_id in self.id_to_internal])


def load_beir_qrels(dataset_name: str, data_path: str = "datasets") -> Tuple[Dict, list, "SimpleBeirDataset"]:
    """Load BEIR dataset qrels for la2m partition.

    Args:
        dataset_name: Name of the BEIR dataset (e.g., 'scifact', 'nfcorpus')
        data_path: Base path for dataset storage

    Returns:
        qrels: Query relevance judgments
        dataset_index: List of corpus document IDs
        dataset_obj: SimpleBeirDataset object for index conversion
    """
    if not BEIR_AVAILABLE:
        raise ImportError("BEIR is required for la2m partition. Install with: pip install beir")

    logger.debug(f"Loading BEIR qrels for la2m partition: {dataset_name}")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(data_path, "datasets")
    beir_data_path = beirUtil.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=beir_data_path).load(split="test")

    dataset_index = list(corpus.keys())
    dataset_obj = SimpleBeirDataset(dataset_index)

    logger.debug(f"Loaded {len(corpus)} corpus items, {len(queries)} queries, {len(qrels)} qrels")

    return qrels, dataset_index, dataset_obj

def save_npy(dir: str, key: str, value: np.ndarray):
    # save value as npy file
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, f"{key}.npy"), value)
    logger.debug(f"save cache {key} to {dir}")

def load_npy(dir: str, key: str, mmap_mode: str = None) -> np.ndarray:
    # load value from npy file
    if key.endswith(".npy"):
        key = key[:-4]
    try:
        result = np.load(os.path.join(dir, f"{key}.npy"), allow_pickle=True, mmap_mode=mmap_mode)
        logger.debug(f"load cache {key} from {dir}")
        return result
    except FileNotFoundError:
        logger.warning(f"cache {key} not found in {dir}")
        return None


class DataPartitioner:
    def __init__(self, labels: np.ndarray = None, total_ind: np.ndarray = None, partition_type: str = "random", overlap_ratio: float = 0.1, nonref_clu_choices: List[int] = None, emb1: np.ndarray = None, emb2: np.ndarray = None, knn_k: int = 50, qrels: dict = None, dataset_index: List[str] = None, dataset_obj = None, select_top_1: bool = True, remove_dup_answer: bool = True, **kwargs):
        self.labels = labels
        self.total_ind = total_ind if total_ind is not None else np.arange(len(labels))
        self.partition_type = partition_type
        self.overlap_ratio = overlap_ratio
        self.nonref_clu_choices = nonref_clu_choices
        self.emb1 = emb1
        self.emb2 = emb2
        self.knn_k = knn_k
        self.qrels = qrels
        self.dataset_index = dataset_index
        self.dataset_obj = dataset_obj
        self.select_top_1 = select_top_1
        self.remove_dup_answer = remove_dup_answer
        self.kwargs = kwargs
        self._compute_partitions()

    @classmethod
    def from_indices(cls, ind_emb1_unique: np.ndarray, ind_emb2_unique: np.ndarray, ind_nonref: np.ndarray = None):
        """Create DataPartitioner from pre-loaded indices.

        Args:
            ind_emb1_unique: Indices for first embedding partition
            ind_emb2_unique: Indices for second embedding partition
            ind_nonref: Non-reference indices (optional)
        """
        instance = cls.__new__(cls)
        instance.ind_emb1_unique = ind_emb1_unique
        instance.ind_emb2_unique = ind_emb2_unique
        instance.ind_emb1_nonref = ind_nonref if ind_nonref is not None else np.array([])
        instance.ind_emb2_nonref = ind_nonref if ind_nonref is not None else np.array([])
        instance.nonref_cluser_size = len(ind_nonref) if ind_nonref is not None else 0

        # Set other attributes to None since they're not available
        instance.labels = None
        instance.total_ind = None
        instance.partition_type = "preloaded"
        instance.overlap_ratio = None
        instance.nonref_clu_choices = None
        instance.kwargs = {}

        return instance

    def _compute_partitions(self):
        """Compute the partition indices based on the specified type"""
        if self.partition_type == "random":
            if self.total_ind is None:
                raise ValueError("random partition requires total_ind or labels to be provided")

            all_indices = np.random.permutation(self.total_ind)
            # overlap_ratio = len(self.nonref_clu_choices) / len(unique_labels)
            overlap_size = int(len(self.total_ind) * self.overlap_ratio)
            remaining_size = len(self.total_ind) - overlap_size
            emb1_unique_size = remaining_size // 2
            emb2_unique_size = remaining_size - emb1_unique_size

            self.ind_emb1_unique = all_indices[:emb1_unique_size]
            self.ind_emb2_unique = all_indices[emb1_unique_size:emb1_unique_size + emb2_unique_size]
            ind_overlap = all_indices[emb1_unique_size + emb2_unique_size:]
            self.ind_emb1_nonref = ind_overlap
            self.ind_emb2_nonref = ind_overlap
            self.ind_emb1_unique = np.concatenate([self.ind_emb1_nonref, self.ind_emb1_unique])
            self.ind_emb2_unique = np.concatenate([self.ind_emb2_nonref, self.ind_emb2_unique])
            self.nonref_cluser_size = len(ind_overlap)
            # Random partitioning does not need cluster labels.
            self.emb1_nclu = self.emb2_nclu = 1
            return

        else:
            unique_labels = np.unique(self.labels)
        if self.partition_type == "knn":
            # Randomly pick one point and get its top k neighbors from both embeddings
            if self.emb1 is None or self.emb2 is None:
                raise ValueError("knn partition requires emb1 and emb2 to be provided")

            # Randomly select a seed point
            seed_idx = np.random.choice(self.total_ind)
            logger.debug(f"knn partition: selected seed point {seed_idx}")

            # Compute distances from seed point to all points in emb1
            seed_emb1 = self.emb1[seed_idx:seed_idx+1]  # Shape: (1, d)
            distances_emb1 = cdist(seed_emb1, self.emb1, metric='euclidean').flatten()

            # Compute distances from seed point to all points in emb2
            seed_emb2 = self.emb2[seed_idx:seed_idx+1]  # Shape: (1, d)
            distances_emb2 = cdist(seed_emb2, self.emb2, metric='euclidean').flatten()

            # Get top k nearest neighbors from both embeddings
            k = min(self.knn_k, len(self.total_ind))
            knn_indices_emb1 = np.argsort(distances_emb1)[:k]
            knn_indices_emb2 = np.argsort(distances_emb2)[:k]

            # Take union of both neighbor sets
            union_indices = np.unique(np.concatenate([knn_indices_emb1, knn_indices_emb2]))
            logger.debug(f"knn partition: emb1 neighbors={len(knn_indices_emb1)}, emb2 neighbors={len(knn_indices_emb2)}, union={len(union_indices)}")

            # Sort union indices by distance in emb1 to identify overlap
            union_distances_emb1 = distances_emb1[union_indices]
            sorted_union_idx = np.argsort(union_distances_emb1)
            sorted_union_indices = union_indices[sorted_union_idx]

            # The nearest (k * overlap_ratio) points in emb1 become the overlap
            overlap_size = int(k * self.overlap_ratio)
            overlap_indices = sorted_union_indices[:overlap_size]
            
            remaining_indices = np.setdiff1d(self.total_ind, overlap_indices)[:len(sorted_union_indices) - overlap_size]
            
            remaining_emb1 = remaining_indices[:len(remaining_indices)//2]
            remaining_emb2 = remaining_indices[len(remaining_indices)//2:]

            # Construct final index sets
            self.ind_emb1_nonref = overlap_indices
            self.ind_emb2_nonref = overlap_indices
            self.ind_emb1_unique = np.concatenate([overlap_indices, remaining_emb1])
            self.ind_emb2_unique = np.concatenate([overlap_indices, remaining_emb2])
            self.nonref_cluser_size = len(overlap_indices)
            self.emb1_nclu = self.emb2_nclu = len(unique_labels)

            logger.debug(f"knn partition: overlap={len(overlap_indices)}, emb1_unique={len(self.ind_emb1_unique)}, emb2_unique={len(self.ind_emb2_unique)}")

        elif self.partition_type == "knn_aligned":
            # KNN with aligned seed selection: pick seed point with most similar neighbor ordering
            if self.emb1 is None or self.emb2 is None:
                raise ValueError("knn_aligned partition requires emb1 and emb2 to be provided")

            logger.debug("knn_aligned partition: finding seed point with most similar neighbor ordering")

            # Pre-compute distance matrices once (OPTIMIZATION)
            logger.debug(f"knn_aligned partition: computing distance matrices for {len(self.total_ind)} points")
            dist_matrix_emb1 = cdist(self.emb1, self.emb1, metric='euclidean')
            dist_matrix_emb2 = cdist(self.emb2, self.emb2, metric='euclidean')
            logger.debug("knn_aligned partition: distance matrices computed")

            # Step 1: Find the seed point with most common neighbors
            k_correlation = int(self.knn_k * self.overlap_ratio)
            k_correlation = min(k_correlation, len(self.total_ind) - 1)
            if k_correlation < 1:
                k_correlation = 1

            best_seed_idx = None
            best_overlap_count = -1

            # Check all points to find the one with most common neighbors
            for candidate_idx in self.total_ind:
                # Extract distances from pre-computed matrices (OPTIMIZATION)
                distances_emb1 = dist_matrix_emb1[candidate_idx]
                distances_emb2 = dist_matrix_emb2[candidate_idx]

                # Get top k nearest neighbors in both embeddings
                neighbors_emb1 = np.argsort(distances_emb1)[:k_correlation]
                neighbors_emb2 = np.argsort(distances_emb2)[:k_correlation]

                # Count common neighbors
                common_neighbors = np.intersect1d(neighbors_emb1, neighbors_emb2)
                overlap_count = len(common_neighbors)

                # Update best seed if this has more common neighbors
                if overlap_count > best_overlap_count:
                    best_overlap_count = overlap_count
                    best_seed_idx = candidate_idx

            seed_idx = best_seed_idx
            logger.debug(f"knn_aligned partition: selected seed point {seed_idx} with {best_overlap_count} common neighbors out of {k_correlation}")

            # Step 2: Use the selected seed with same logic as knn partition
            # Extract distances from pre-computed matrices (OPTIMIZATION)
            distances_emb1 = dist_matrix_emb1[seed_idx]
            distances_emb2 = dist_matrix_emb2[seed_idx]

            # Get top k nearest neighbors from both embeddings
            k = min(self.knn_k, len(self.total_ind))
            knn_indices_emb1 = np.argsort(distances_emb1)[:k]
            knn_indices_emb2 = np.argsort(distances_emb2)[:k]

            # Take union of both neighbor sets
            union_indices = np.unique(np.concatenate([knn_indices_emb1, knn_indices_emb2]))
            logger.debug(f"knn_aligned partition: emb1 neighbors={len(knn_indices_emb1)}, emb2 neighbors={len(knn_indices_emb2)}, union={len(union_indices)}")

            # Sort union indices by distance in emb1 to identify overlap
            union_distances_emb1 = distances_emb1[union_indices]
            sorted_union_idx = np.argsort(union_distances_emb1)
            sorted_union_indices = union_indices[sorted_union_idx]

            # The nearest (k * overlap_ratio) points in emb1 become the overlap
            overlap_size = int(k * self.overlap_ratio)
            overlap_indices = sorted_union_indices[:overlap_size]

            # Remaining points in union are split in half
            remaining_indices = sorted_union_indices[overlap_size:]
            half_size = len(remaining_indices) // 2
            remaining_emb1 = remaining_indices[:half_size]
            remaining_emb2 = remaining_indices[half_size:]

            # Construct final index sets
            self.ind_emb1_nonref = overlap_indices
            self.ind_emb2_nonref = overlap_indices
            self.ind_emb1_unique = np.concatenate([overlap_indices, remaining_emb1])
            self.ind_emb2_unique = np.concatenate([overlap_indices, remaining_emb2])
            self.nonref_cluser_size = len(overlap_indices)
            self.emb1_nclu = self.emb2_nclu = len(unique_labels)

            logger.debug(f"knn_aligned partition: overlap={len(overlap_indices)}, emb1_unique={len(self.ind_emb1_unique)}, emb2_unique={len(self.ind_emb2_unique)}")

        elif self.partition_type.startswith("cluster"):
            # self.overlap_clu = [4]
            clu_size = []
            for label in unique_labels:
                clu_size.append(np.sum(self.labels == label))
            clu_size = np.array(clu_size)
            clu_size = clu_size[clu_size.argsort()]
            logger.debug(f"cluster size: {clu_size}")

            if self.partition_type == "cluster":
                emb1_clu = unique_labels[:len(unique_labels)//2]
                emb2_clu = unique_labels[len(unique_labels)//2:]
                self.emb1_nclu = len(emb1_clu)
                self.emb2_nclu = len(emb2_clu)
                self.ind_emb1_unique = np.where(np.isin(self.labels, emb1_clu))[0]
                self.ind_emb2_unique = np.where(np.isin(self.labels, emb2_clu))[0]

            elif self.partition_type == "cluster_partial":
                nonref_nclu = len(self.nonref_clu_choices)
                nonref_clu_ind = np.where(np.isin(self.labels, self.nonref_clu_choices))[0]
                self.nonref_cluser_size = len(nonref_clu_ind)
                logger.debug(f"nonref cluster size: {self.nonref_cluser_size}")
                logger.debug(f"nonref cluster choices: {self.nonref_clu_choices}")
                emb1_clu = unique_labels[~np.isin(unique_labels, self.nonref_clu_choices)]
                self.emb1_clu_unique = emb1_clu[:len(emb1_clu)//2]
                self.emb2_clu_unique = emb1_clu[len(emb1_clu)//2:]
                self.emb1_nclu = len(self.emb1_clu_unique) + nonref_nclu
                self.emb2_nclu = len(self.emb2_clu_unique) + nonref_nclu
                self.ind_emb1_unique = np.where(np.isin(self.labels, self.emb1_clu_unique))[0]
                self.ind_emb2_unique = np.where(np.isin(self.labels, self.emb2_clu_unique))[0]
                self.ind_emb1_nonref = nonref_clu_ind
                self.ind_emb2_nonref = nonref_clu_ind
                self.ind_emb1_unique = np.concatenate([self.ind_emb1_nonref, self.ind_emb1_unique])
                self.ind_emb2_unique = np.concatenate([self.ind_emb2_nonref, self.ind_emb2_unique])

        elif self.partition_type == "la2m":
            # LA2M split: reference (overlap) contains no answers, unique parts have evenly distributed answers
            if self.qrels is None:
                raise ValueError("la2m partition requires qrels to be provided")
            if self.dataset_index is None:
                raise ValueError("la2m partition requires dataset_index to be provided")
            if self.dataset_obj is None:
                raise ValueError("la2m partition requires dataset_obj to be provided")

            logger.debug("LA2M partition: extracting answer indices from qrels")

            # Extract answer indices from qrels
            answer_doc_ids = self._extract_answer_indices_from_qrels(
                self.qrels,
                self.dataset_index,
                self.select_top_1
            )

            # Convert to internal indices
            answer_indices = self.dataset_obj.batch_original_ids_to_internal_indices(answer_doc_ids)

            # Process duplicates if needed
            if self.remove_dup_answer:
                unique_answer_indices = np.unique(answer_indices)
                removed_count = len(answer_indices) - len(unique_answer_indices)
                if removed_count > 0:
                    logger.debug(f"LA2M partition: removed {removed_count} duplicate answers")
                answer_indices = unique_answer_indices
            else:
                logger.debug(f"LA2M partition: keeping all {len(answer_indices)} answer documents including duplicates")

            # Categorize documents into answer and non-answer sets
            all_docs = set(self.total_ind)
            answer_docs = set(answer_indices)
            answer_docs_in_dataset = answer_docs.intersection(all_docs)
            non_answer_docs = all_docs - answer_docs_in_dataset

            logger.debug(f"LA2M partition: total={len(all_docs)}, answers={len(answer_docs_in_dataset)}, non-answers={len(non_answer_docs)}")

            # Allocate D0 (reference/overlap) from non-answer documents only
            d0_target_size = int(len(self.total_ind) * self.overlap_ratio)
            available_non_answer = len(non_answer_docs)

            if available_non_answer < d0_target_size:
                logger.warning(f"LA2M partition: insufficient non-answer docs ({available_non_answer}) for target D0 size ({d0_target_size})")
                d0_size = available_non_answer
            else:
                d0_size = d0_target_size

            # Randomly select D0 from non-answer documents
            if d0_size > 0:
                d0_docs = set(np.random.choice(list(non_answer_docs), size=d0_size, replace=False))
            else:
                d0_docs = set()
                logger.warning("LA2M partition: D0 is empty - no non-answer documents available")

            # Remaining non-answer documents
            remaining_non_answer_docs = non_answer_docs - d0_docs

            # Evenly split answer documents between D1 and D2
            answer_docs_list = list(answer_docs_in_dataset)
            np.random.shuffle(answer_docs_list)

            mid_answers = len(answer_docs_list) // 2
            d1_answer_docs = set(answer_docs_list[:mid_answers])
            d2_answer_docs = set(answer_docs_list[mid_answers:])

            logger.debug(f"LA2M partition: answer distribution - D1={len(d1_answer_docs)}, D2={len(d2_answer_docs)}")

            # Evenly split remaining non-answer documents between D1 and D2
            remaining_non_answer_list = list(remaining_non_answer_docs)
            np.random.shuffle(remaining_non_answer_list)

            mid_non_answer = len(remaining_non_answer_list) // 2
            d1_non_answer_docs = set(remaining_non_answer_list[:mid_non_answer])
            d2_non_answer_docs = set(remaining_non_answer_list[mid_non_answer:])

            logger.debug(f"LA2M partition: non-answer distribution - D1={len(d1_non_answer_docs)}, D2={len(d2_non_answer_docs)}")

            # Combine answer and non-answer for each set
            d1_docs = d1_answer_docs | d1_non_answer_docs
            d2_docs = d2_answer_docs | d2_non_answer_docs

            # Convert to indices (maintaining original order from total_ind)
            d0_index = np.array([doc for doc in self.total_ind if doc in d0_docs])
            d1_index = np.array([doc for doc in self.total_ind if doc in d1_docs])
            d2_index = np.array([doc for doc in self.total_ind if doc in d2_docs])

            # Set the partition attributes
            # Important: ind_emb1_nonref and ind_emb2_nonref are the OVERLAP (D0) which contains NO answers
            self.ind_emb1_nonref = d0_index
            self.ind_emb2_nonref = d0_index

            # ind_emb1_unique and ind_emb2_unique include both overlap and their unique parts
            self.ind_emb1_unique = np.concatenate([d0_index, d1_index])
            self.ind_emb2_unique = np.concatenate([d0_index, d2_index])

            self.nonref_cluser_size = len(d0_index)
            self.emb1_nclu = self.emb2_nclu = len(unique_labels)

            # Validate LA2M constraints
            answers_in_d0 = set(d0_index).intersection(answer_docs_in_dataset)
            if answers_in_d0:
                raise ValueError(f"LA2M constraint violated: {len(answers_in_d0)} answer documents found in D0 (reference)")

            logger.debug(f"LA2M partition results:")
            logger.debug(f"  D0 (overlap/reference): {len(d0_index)} docs (0 answers) - used as ind_emb1_nonref and ind_emb2_nonref")
            logger.debug(f"  D1 (emb1 unique): {len(d1_index)} docs ({len(d1_answer_docs)} answers)")
            logger.debug(f"  D2 (emb2 unique): {len(d2_index)} docs ({len(d2_answer_docs)} answers)")
            logger.debug(f"  Total ind_emb1_unique: {len(self.ind_emb1_unique)} (D0 + D1)")
            logger.debug(f"  Total ind_emb2_unique: {len(self.ind_emb2_unique)} (D0 + D2)")

        else:
            raise ValueError(f"Invalid partition type: {self.partition_type}")

    @staticmethod
    def _extract_answer_indices_from_qrels(qrels: dict, dataset_index: List[str], select_top_1: bool = True) -> List[str]:
        """
        Extract answer document indices from qrels.

        Args:
            qrels: query relevance judgments {query_id: {doc_id: relevance_score}}
            dataset_index: indices of all documents
            select_top_1: whether to select the top 1 relevant answer for each query

        Returns:
            answer_doc_ids: list of answer document IDs
        """
        answer_doc_ids = set()

        logger.debug(f"Extracting answer documents (select_top_1={select_top_1})")

        for query_id, doc_relevances in qrels.items():
            # Filter to only relevant documents (relevance > 0)
            relevant_docs = {doc_id: score for doc_id, score in doc_relevances.items() if score > 0}

            if not relevant_docs:
                logger.debug(f"No relevant documents found for query {query_id}")
                continue

            if select_top_1:
                # Select only the document with highest relevance score
                top_doc_id = max(relevant_docs.keys(), key=lambda x: relevant_docs[x])
                selected_docs = [top_doc_id]
            else:
                # Select all relevant documents
                selected_docs = list(relevant_docs.keys())

            # Add to answer set
            for doc_id in selected_docs:
                answer_doc_ids.add(doc_id)

        # Filter to only include docs that are in dataset_index
        dataset_set = set(dataset_index)
        valid_answer_indices = answer_doc_ids.intersection(dataset_set)

        selection_mode = "top-1" if select_top_1 else "all relevant"
        logger.debug(f"Extracted {len(valid_answer_indices)} valid answer documents using {selection_mode} strategy")
        logger.debug(f"  Total queries processed: {len(qrels)}")
        logger.debug(f"  Raw answer docs found: {len(answer_doc_ids)}")
        logger.debug(f"  Valid answer docs (in dataset): {len(valid_answer_indices)}")

        if len(answer_doc_ids) != len(valid_answer_indices):
            excluded = len(answer_doc_ids) - len(valid_answer_indices)
            logger.warning(f"Excluded {excluded} answer docs not found in dataset")

        return list(valid_answer_indices)


def align_dim(data1: Union[np.ndarray, torch.Tensor], data2: Union[np.ndarray, torch.Tensor] = None, method: str = 'pca', n_dim: int = 2) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    def pad_tensor(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError(f"Input tensor must be 2D, got shape {tensor.shape}")
        n, k = tensor.shape
        if k >= target_dim:
            return tensor[:, :target_dim]
        # Pad columns with zeros
        pad_size = target_dim - k
        padding = (0, pad_size)  # (padding_left, padding_right) for last dim
        return torch.nn.functional.pad(tensor, padding, mode='constant', value=0)

    def pad_numpy(array: np.ndarray, target_dim: int) -> np.ndarray:
        if array.ndim != 2:
            raise ValueError(f"Input array must be 2D, got shape {array.shape}")
        n, k = array.shape
        if k >= target_dim:
            return array[:, :target_dim]
        # Pad columns with zeros
        pad_width = ((0, 0), (0, target_dim - k))
        return np.pad(array, pad_width, mode='constant')

    is_tensor1 = isinstance(data1, torch.Tensor)
    is_tensor2 = isinstance(data2, torch.Tensor)

    if is_tensor1 and not is_tensor2:
        data1 = data1.cpu()
        data2 = torch.tensor(data2, dtype=data1.dtype, device=data1.device)
    elif not is_tensor1 and is_tensor2:
        data2 = data2.cpu()
        data1 = torch.tensor(data1, dtype=data2.dtype, device=data2.device)

    if method.lower() == 'zero':
        max_dim = max(data1.shape[1], data2.shape[1]) if data2 is not None else n_dim
        if is_tensor1:
            data1 = pad_tensor(data1, max_dim)
        else:
            data1 = pad_numpy(data1, max_dim)
        if data2 is not None:
            if is_tensor2:
                data2 = pad_tensor(data2, max_dim)
            else:
                data2 = pad_numpy(data2, max_dim)
        return data1, data2
    else:
        if is_tensor1:
            data1 = data1.cpu().numpy()
        if is_tensor2:
            data2 = data2.cpu().numpy()
        n_dim = min(data1.shape[1], data2.shape[1]) if data2 is not None else n_dim
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_dim)
        elif method.lower() == 'mds':
            reducer = MDS(n_components=n_dim)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_dim)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'pca', 'mds', or 'umap'")

        if data2 is None:
            data1 = reducer.fit_transform(data1)
            return data1
        else:
            if data1.shape[1] == n_dim:
                data2 = reducer.fit_transform(data2)
            else:
                data1 = reducer.fit_transform(data1)
            return data1, data2



def convert_global_to_local_indices(
    ref_indices_global: np.ndarray,
    ind_unique: np.ndarray
) -> np.ndarray:
    """
    Convert global indices to local indices within the unique embedding set.

    Args:
        ref_indices_global: Global indices in the full dataset
        ind_unique: Global indices of the unique embedding set

    Returns:
        Local indices within ind_unique
    """
    # Create a mapping from global to local indices
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(ind_unique)}

    # Convert reference indices
    ref_indices_local = np.array([
        global_to_local[global_idx]
        for global_idx in ref_indices_global
        if global_idx in global_to_local
    ], dtype=np.int32)

    return ref_indices_local
