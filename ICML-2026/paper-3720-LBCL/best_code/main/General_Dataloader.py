import scanpy as sc
import numpy as np
import DataLoader_combination_final as dl

def General_DataLoader(
    train_file_path: str,
    test_file_path:  str,
    batch_size:      int,
    size_factor:     float,
    unlabeled_per_batch: int,
    batch_seed:      int
):
    """
    Wrapper to generate contrastive‐learning batches with labeled (from train set)
    and unlabeled (from test set) subsets.

    Args:
        train_file_path (str): Path to .h5ad with training cells & 'clone_id'.
        test_file_path  (str): Path to .h5ad with test    cells & 'clone_id'.
        batch_size      (int): Number of positive lineage‐pairs per batch.
        size_factor     (float): Fraction of available lineage‐pairs to sample.
        unlabeled_per_batch (int): Number of unlabeled cells per batch (drawn from test set).
        batch_seed      (int): Random seed for reproducibility.

    Returns:
        batch_all_label       dict[int, list[ (tensor_i, tensor_j) ] ]
        batch_all_unlabel     dict[int, list[ tensor_k ] ]
        num_batches           int
        lineage_array_label   np.ndarray of shape (2 * num_batches * batch_size, 1)
        lineage_array_unlabel dict[int, np.ndarray of shape (n_unlab,1)]
    """
    # load train AnnData
    adata_train = sc.read_h5ad(train_file_path)
    train_matrix = adata_train.X
    train_lineages = adata_train.obs['clone_id'].values.reshape(-1, 1)

    # load test AnnData
    adata_test = sc.read_h5ad(test_file_path)
    test_matrix = adata_test.X
    test_lineages = adata_test.obs['clone_id'].values.reshape(-1, 1)

    # initialize our custom loader
    DLoader = dl.SClineage_DataLoader(
        train_count_matrix    = train_matrix,
        train_lineages        = train_lineages,
        test_count_matrix     = test_matrix,
        test_lineages         = test_lineages,
        batch_size            = batch_size,
        size_factor           = size_factor,
        unlabeled_per_batch   = unlabeled_per_batch,
        seed                  = batch_seed
    )

    # unpack everything
    batch_all_label, batch_all_unlabel, num_batches, \
      lineage_array_label, lineage_array_unlabel = DLoader.batch_generator()

    # some logging
    print(f"Number of batches: {num_batches}")
    print(f"Total labeled pairs: {num_batches} × {batch_size} = {num_batches * batch_size}")
    n_unlab = len(batch_all_unlabel[next(iter(batch_all_unlabel))])
    print(f"Unlabeled per batch (from test set): {n_unlab}")

    return (
        batch_all_label,
        batch_all_unlabel,
        num_batches,
        lineage_array_label,
        lineage_array_unlabel
    )


if __name__ == "__main__":
    # Example usage
    train_fp = "/Users/apple/Desktop/KB/data/LarryData/train_test/Larry_200_train.h5ad"
    test_fp  = "/Users/apple/Desktop/KB/data/LarryData/train_test/Larry_200_test.h5ad"

    bs       = 30
    sf       = 0.5
    unlab    = 10
    seed     = 42

    batches_label, batches_unlabel, n_batches, labels_label, labels_unlabel = \
      General_DataLoader(train_fp, test_fp, bs, sf, unlab, seed)

    print(f"labels_label shape: {labels_label.shape}")
    print(f"Batch 0: {len(batches_label[0])} pairs, {len(batches_unlabel[0])} unlabeled")