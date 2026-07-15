import lifelines


def metric(y_true, pred):
    """Calculates the concordance index (c-index) between a series of event
    times and a predicted score.
    The c-index is the average of how often a model says X is greater than Y
    when, in the observed data, X is indeed greater than Y.
    The c-index also handles how to handle censored values.
    Parameters
    ----------
    y_true : numpy array of floats of dimension (n_samples, 2), real
            survival times from the observational data
    pred : numpy array of floats of dimension (n_samples, 1), predicted
            scores from a model
    Returns
    -------
    c-index: float, calculating using the lifelines library
    """

    c_index = lifelines.utils.concordance_index(y_true[:, 1], -pred, y_true[:, 0])

    return c_index
