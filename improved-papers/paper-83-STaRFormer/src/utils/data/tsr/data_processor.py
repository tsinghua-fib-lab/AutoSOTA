def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data


def to_fd(series, sample_points=None):
    """
    Convert a set of time series to functional data
    :param series: a set of time series with the shape of N x L
    :param sample_points: sample point of the series
    :return: functional data representation of the time series
    """
    from skfda import FDataGrid

    if sample_points is None:
        sample_points = [x for x in range(series.shape[1])]

    return FDataGrid(series, sample_points)