from time import time

name = "TransformerTools"

transformers = ["pca", "fpca", "fpca_bspline"]


def fit_transformer(transformer_name, X_train, flatten=False, **kwargs):
    """
    Fit a transformer for a set of time series
    :param transformer_name:
    :param X_train:
    :param flatten:
    :param kwargs:
    :return:
    """
    print("[{}] Fitting transformer".format(name))
    start_time = time()

    if flatten:
        # if flatten, do not transform per dimension
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2], 1)
    transformer = create_transformer(transformer_name, **kwargs)

    X_train_transformed = transformer.fit_transform(X_train)

    elapsed_time = time() - start_time
    print("[{}] Transformer fitted, took {}s".format(name, elapsed_time))
    return X_train_transformed, transformer


def create_transformer(transformer_name, **kwargs):
    """
    Create a transformer
    :param transformer_name:
    :param kwargs:
    :return:
    """
    print("[{}] Creating transformer".format(name))
    if transformer_name == "pca":
        from transform.transformers import PCATransformer
        return PCATransformer(**kwargs)
    if transformer_name == "fpca":
        from transform.transformers import FPCATransformer
        return FPCATransformer(**kwargs)
    if transformer_name == "fpca_bspline":
        from transform.transformers import FPCATransformer
        return FPCATransformer(**kwargs)

    from transform.transformers import TimeSeriesTransformer
    return TimeSeriesTransformer(**kwargs)
