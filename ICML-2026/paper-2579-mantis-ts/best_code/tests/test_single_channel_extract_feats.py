import pytest
import torch

import numpy as np
import torch.nn.functional as F

from mantis.architecture import Mantis8M, MantisV1, MantisV2
from mantis.trainer import MantisTrainer


@pytest.mark.parametrize("device", ['cpu'])
def test_single_channel_extract_feats(device):
    # ==== read data ====
    # your X_train and X_test should be of the shape (n_samples, 1, seq_len=512)
    data = [np.load(f'data/GestureMidAirD1/{variable}_{set_name}.npy')
            for variable in ['X', 'y'] for set_name in ['train', 'test']]
    X_train, X_test, y_train, y_test = data

    # if original sequence length is different, resize it, for example, using the following function:
    def resize(X):
        X_scaled = F.interpolate(torch.tensor(
            X, dtype=torch.float), size=512, mode='linear', align_corners=False)
        return X_scaled.numpy()
    X_train, X_test = resize(X_train), resize(X_test)

    assert ((X_train.shape[2] == 512) and (X_test.shape[2] == 512)
            ), f"After applying resize function, the sequence length should be 512 but got instead {X_train.shape[2]} and {X_test.shape[2]} for X_train and X_test, respectively."

    for model_name in ["Mantis8M", "MantisV1", "MantisV2"]:
        # ==== init the model, load the weights ====
        if model_name == "Mantis8M":
            network = Mantis8M(device=device)
            network = network.from_pretrained("paris-noah/Mantis-8M")
        elif model_name == "MantisV1":
            network = MantisV1(device=device)
            network = network.from_pretrained("paris-noah/MantisPlus")
        elif model_name == "MantisV2":
            network = MantisV2(device=device)
            network = network.from_pretrained("paris-noah/MantisV2")
        model = MantisTrainer(device=device, network=network)
        # ==== extract deep features, then random forest ====:
        # get deep features
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
        assert ((Z_train.shape[1] == 256) and (Z_test.shape[1] == 256)
                ), f"The number of deep features should be 256 but got instead {Z_train.shape[1]} and {Z_test.shape[1]} for Z_train and Z_test, respectively."

        # train a classifier
        from sklearn.ensemble import RandomForestClassifier
        predictor = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=0)
        predictor.fit(Z_train, y_train)

        # evaluate the performance
        y_pred = predictor.predict(Z_test)
        print(f'{model_name}: Accuracy on the test set is {np.mean(y_test == y_pred)}')

    print(
        f"The test of feature extraction for single channel classification with device={device} has been passed succesfully."
    )
