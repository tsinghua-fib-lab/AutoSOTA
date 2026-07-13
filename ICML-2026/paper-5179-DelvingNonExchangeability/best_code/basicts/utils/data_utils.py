import pandas as pd
import numpy as np
import warnings

from tsl.data import SpatioTemporalDataset


def create_residuals_frame(residuals, index, channels_index=None, horizon=1, idx_type='datetime'):
    # Flatten index
    if idx_type == 'datetime':
        index = pd.DatetimeIndex(index.reshape(-1))
    elif idx_type == 'scalar':
        index = pd.Index(index.reshape(-1))
    else:
        raise ValueError('idx_type must be "datetime" or "scalar"')

    # residuals = np.repeat(np.repeat(np.arange(residuals.shape[0]).reshape(-1, 1, 1), 12, 1), 207, -1)
    residuals = residuals.reshape(-1, *residuals.shape[2:])
    df = pd.DataFrame(data=residuals, index=index,
                      columns=channels_index)

    lagged_residuals = {k: g.reset_index(drop=True) for k, g in df.groupby(level=0) if len(g) == horizon}
    lagged_residuals = pd.concat(lagged_residuals, axis=0).unstack()

    new_cols = pd.MultiIndex.from_tuples(
        [(x[0], f"{x[1]}_{x[2]}") for x in lagged_residuals.columns],
    )

    lagged_residuals.columns = new_cols
    return lagged_residuals

def filter_indices(
        dataset : SpatioTemporalDataset,
        indices,
        valid_indices,
        filter_by='window'
):
    """
    Remove any sample that does not completely overlap with indices.

    :param dataset: Ref dataset
    :param indices: Indices to filter
    :param valid_indices: Valid indices
    :return: Filtered indices
    """

    expanded_indices = dataset.expand_indices(indices)[filter_by].numpy()
    def is_in_idx(sample):
        return np.all(np.in1d(sample, valid_indices))
    mask = np.apply_along_axis(is_in_idx, 1, expanded_indices)
    return indices[mask]

def parse_and_filter_indices(target_dataset, indices):
    calib_indices = indices['calib_indices']
    test_indices = indices['test_indices']

    # filter indices incompatible with  new window lenght:
    calib_indices = calib_indices[np.in1d(calib_indices, target_dataset.indices)]
    test_indices = test_indices[np.in1d(test_indices, target_dataset.indices)]

    valid_input_indices = indices['valid_input_indices']
    valid_target_indices = indices['valid_target_indices']

    def filter_indices_(indices_):
        indices_ = filter_indices(target_dataset,
                             indices_,
                             valid_input_indices,
                             filter_by='window')
        indices_ = filter_indices(target_dataset,
                                  indices_,
                                  valid_target_indices,
                                  filter_by='horizon')
        return indices_

    calib_filtered = filter_indices_(calib_indices)
    test_filtered = filter_indices_(test_indices)
    overlapping_indices, _ = target_dataset.overlapping_indices(calib_filtered, test_filtered, as_mask=True)

    calib_filtered = calib_filtered[~overlapping_indices]
    if len(calib_filtered) <= target_dataset.samples_offset:
        warnings.warn(
            "Filtered calib indices are too small; falling back to unfiltered splits.",
            RuntimeWarning,
        )
        calib_fallback = calib_indices[np.isin(calib_indices, target_dataset.indices)]
        test_fallback = test_indices[np.isin(test_indices, target_dataset.indices)]
        overlapping_indices, _ = target_dataset.overlapping_indices(calib_fallback, test_fallback, as_mask=True)
        return calib_fallback[~overlapping_indices], test_fallback

    return calib_filtered, test_filtered


def find_close(el, seq):
    # find closest in seq and check is close
    idx = np.argmin(np.abs(np.array(seq) - el))
    assert np.isclose(seq[idx], el)

    return idx
