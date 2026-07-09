import pytest
import torch

import numpy as np
import torch.nn.functional as F

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


@pytest.mark.parametrize("adapter",  [None, 'pca', 'var', 'svd', 'rand'])
@pytest.mark.parametrize("device", ['cpu'])
def test_multichannel_extract_feats(device, adapter):
    # ==== read data ====
    # your X_train and X_test should be of the shape (n_samples, 1, seq_len=512)
    data = [np.load(f'data/HandMovementDirection/{variable}_{set_name}.npy')
            for variable in ['X', 'y'] for set_name in ['train', 'test']]
    X_train, X_test, y_train, y_test = data

    # if original sequence length is different, resize it, for example, using the following function:
    def resize(X):
        X_scaled = F.interpolate(torch.tensor(
            X, dtype=torch.float), size=512, mode='linear', align_corners=False)
        return X_scaled.numpy()
    X_train, X_test = resize(X_train), resize(X_test)

    assert ((X_train.shape == (160, 10, 512)) and (X_test.shape == (74, 10, 512))
            ), f"After applying resize function, the shape of X_train and X_test should be (160, 10, 512) and (74, 10, 512) but got instead {X_train.shape} and {X_test.shape}, respectively."

    # ==== apply standalone adapter ====
    standalone_adapters = ['pca', 'var', 'svd', 'rand']
    if adapter in standalone_adapters:
        if adapter == 'var':
            from mantis.adapters import VarianceBasedSelector
            adapter = VarianceBasedSelector(new_num_channels=5)
            adapter.fit(X_train)
            X_train, X_test = adapter.transform(
                X_train), adapter.transform(X_test)
        else:
            from mantis.adapters import MultichannelProjector
            adapter = MultichannelProjector(
                new_num_channels=5, patch_window_size=1, base_projector=adapter)
            adapter.fit(X_train)
            X_train, X_test = adapter.transform(
                X_train), adapter.transform(X_test)
        adapter = None

    # ==== init the model, load the weights ====
    network = Mantis8M(device=device)
    network = network.from_pretrained("paris-noah/Mantis-8M")
    model = MantisTrainer(device=device, network=network)

    # ==== extract deep features, then random forest ====:
    # get deep features
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)
    assert ((Z_train.shape[1] == 256 * X_train.shape[1]) and (Z_test.shape[1] == 256 * X_train.shape[1])
            ), f"The number of deep features should be {256 * X_train.shape[1]} but got instead {Z_train.shape[1]} and {Z_test.shape[1]} for Z_train and Z_test, respectively."

    # train a classifier
    from sklearn.ensemble import RandomForestClassifier
    predictor = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=0)
    predictor.fit(Z_train, y_train)

    # evaluate the performance
    y_pred = predictor.predict(Z_test)
    print(f'Accuracy on the test set is {np.mean(y_test == y_pred)}')

    print(
        f"The test of feature extraction for multi-channel classification with device={device} has been passed succesfully."
    )
