import math
import functools
import numpy as np


def clients_indices(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance(idxs, num_split: int):

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch
