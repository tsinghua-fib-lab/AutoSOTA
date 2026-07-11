import numpy as np
from itertools import combinations
import random
import copy
import torch

class SClineage_DataLoader:
    def __init__(
        self,
        train_count_matrix,
        train_lineages,
        test_count_matrix,
        test_lineages,
        batch_size=20,
        size_factor=0.5,
        unlabeled_per_batch=5,
        seed=None
    ):
        """
        Args:
            train_count_matrix (csr_matrix or np.ndarray): n_train_cells × n_genes
            train_lineages      (np.ndarray of shape [n_train_cells,1])
            test_count_matrix  (csr_matrix or np.ndarray): n_test_cells  × n_genes
            test_lineages       (np.ndarray of shape [n_test_cells, 1])
            batch_size          (int): number of positive pairs per batch
            size_factor         (float): fraction of all possible pairs per lineage
            unlabeled_per_batch (int): how many unlabeled cells/batch (drawn from test set)
            seed                (int | None): random seed
        """
        # --- store train vs test separately
        self.train_matrix  = train_count_matrix.toarray() if hasattr(train_count_matrix, "toarray") else train_count_matrix
        self.train_lineages = train_lineages.flatten()
        self.test_matrix   = test_count_matrix .toarray() if hasattr(test_count_matrix,  "toarray") else test_count_matrix
        self.test_lineages  = test_lineages.flatten()

        self.batch_size = batch_size
        self.size_factor = size_factor
        self.unlabeled_per_batch = unlabeled_per_batch

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # build all the positive‐pair machinery from TRAINING set
        self.lineage_info          = self._generate_lineage_info(self.train_lineages)
        self.avail_lineage_pairs   = self._generate_avail_lineage_pairs()
        self.batch_all_index       = self._generate_batch_all_index()
        self.batch_all_label       = self._generate_batch_all()
        self.lineage_array_label   = self._generate_lineage_array(self.batch_all_index,
                                                                 self.train_lineages)

        # build all the unlabeled machinery from TESTING set
        self.batch_all_unlabel, self.unlabeled_indices = self._generate_batch_all_unlabel()
        self.lineage_array_unlabel = self._generate_lineage_array(self.unlabeled_indices,
                                                                 self.test_lineages)

    def _generate_lineage_info(self, lineage_array):
        info = {}
        for lab in np.unique(lineage_array):
            info[lab] = np.where(lineage_array == lab)[0].tolist()
        return info

    def _generate_avail_lineage_pairs(self):
        avail = {}
        for lab, inds in self.lineage_info.items():
            all_pairs = list(combinations(inds, 2))
            k = int(self.size_factor * len(all_pairs))
            # ensure coverage
            used, essential = set(), []
            for i, j in all_pairs:
                if i not in used or j not in used:
                    essential.append((i, j))
                    used.update([i, j])
            # fill up to k
            remaining = [p for p in all_pairs if p not in essential]
            random.shuffle(remaining)
            extra = remaining[: max(0, k - len(essential))]
            avail[lab] = essential + extra
        return avail

    def _generate_batch_all_index(self):
        pool = copy.deepcopy(self.avail_lineage_pairs)
        batches, idx = {}, 0
        while pool:
            labs = list(pool.keys())
            selected = random.sample(labs, min(self.batch_size, len(labs)))
            batches[idx] = []
            for lab in selected:
                pair = random.choice(pool[lab])
                batches[idx].append(pair)
                pool[lab].remove(pair)
                if not pool[lab]:
                    del pool[lab]
            if len(selected) < self.batch_size:
                # fill extra from any lineage
                complement = list(self.avail_lineage_pairs.keys())
                extra = random.sample(complement, self.batch_size - len(selected))
                for lab in extra:
                    batches[idx].append(random.choice(self.avail_lineage_pairs[lab]))
            idx += 1
        return batches

    def _generate_batch_all(self):
        batch_all = {}
        for bidx, pairs in self.batch_all_index.items():
            batch_all[bidx] = [
                (
                    torch.tensor(self.train_matrix[i], dtype=torch.float),
                    torch.tensor(self.train_matrix[j], dtype=torch.float)
                )
                for (i, j) in pairs
            ]
        return batch_all

    def _generate_lineage_array(self, index_dict, lineage_array):
        """
        index_dict: dict batch_idx -> list of indices or list of pairs
        lineage_array: 1d array of the matching set
        returns a dict if index_dict values are lists of ints, or a flat array if pairs.
        """
        # if values are (i,j) pairs -> flatten to labels of both ends
        sample0 = next(iter(index_dict.values()))
        if isinstance(sample0[0], tuple):
            # labeled case
            labs = []
            for pairs in index_dict.values():
                for i, j in pairs:
                    labs.extend([lineage_array[i], lineage_array[j]])
            return np.array(labs, dtype=int).reshape(-1, 1)

        else:
            # unlabeled case: index_dict[b] is a list of individual cell‐indices
            out = {}
            for bidx, inds in index_dict.items():
                out[bidx] = np.array([lineage_array[idx] for idx in inds],
                                     dtype=int).reshape(-1, 1)
            return out

    def _generate_batch_all_unlabel(self):
        """
        Now sample unlabeled_per_batch cells *exclusively* from the TEST set.
        """
        batch_unlab, indices_unlab = {}, {}
        n_test = self.test_matrix.shape[0]

        for bidx in self.batch_all_index:
            # pick any test‐indices
            sample_n = min(self.unlabeled_per_batch, n_test)
            chosen = random.sample(range(n_test), sample_n)
            # extract their tensors
            batch_unlab[bidx] = [
                torch.tensor(self.test_matrix[idx], dtype=torch.float)
                for idx in chosen
            ]
            indices_unlab[bidx] = chosen
        return batch_unlab, indices_unlab

    def batch_generator(self):
        """
        Returns:
          batch_all_label, batch_all_unlabel,
          num_batches,
          lineage_array_label, lineage_array_unlabel
        """
        return (
            self.batch_all_label,
            self.batch_all_unlabel,
            len(self.batch_all_label),
            self.lineage_array_label,
            self.lineage_array_unlabel
        )

# example instantiation:
if __name__ == "__main__":
    import scanpy as sc
    ad_train = sc.read_h5ad("/Users/apple/Desktop/KB/data/LarryData/train_test/Larry_200_train.h5ad")
    ad_test  = sc.read_h5ad("/Users/apple/Desktop/KB/data/LarryData/train_test/Larry_200_test.h5ad")

    loader = SClineage_DataLoader(
        train_count_matrix = ad_train.X,
        train_lineages     = ad_train.obs["clone_id"].values.reshape(-1,1),
        test_count_matrix  = ad_test.X,
        test_lineages      = ad_test.obs["clone_id"].values.reshape(-1,1),
        batch_size         = 30,
        size_factor        = 0.5,
        unlabeled_per_batch= 10,
        seed               = 42
    )

    batches_label, batches_unlabel, n_batches, labels_label, labels_unlabel = loader.batch_generator()
    print(f"Batches: {n_batches}")
    for b in range(min(1, n_batches)):
        print(f"Batch {b}: {len(batches_label[b])} pairs, {len(batches_unlabel[b])} unlabeled")
        print("Label shapes:", labels_label[b].shape, labels_unlabel[b].shape)
        print("batches_label[b]: ", batches_label[b])
