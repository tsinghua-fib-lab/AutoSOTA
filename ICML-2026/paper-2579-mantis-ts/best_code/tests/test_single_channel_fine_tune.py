import pytest
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


@pytest.mark.parametrize("fine_tuning_type", ['head', 'full', 'scratch'])
@pytest.mark.parametrize("device", ['cpu'])
def test_single_channel_fine_tune(fine_tuning_type, device):
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

    # ==== init the model, load the weights ====
    network = Mantis8M(device=device)
    if fine_tuning_type != 'scratch':
        network = network.from_pretrained("paris-noah/Mantis-8M")
    model = MantisTrainer(device=device, network=network)

    # ==== using fit function of MantisTrainer ====

    # initialize some training parameters
    def init_optimizer(params): return torch.optim.AdamW(
        params, lr=2e-3, betas=(0.9, 0.999), weight_decay=0.05)
    num_epochs = 2
    head = nn.Sequential(
        nn.LayerNorm(network.hidden_dim * X_train.shape[1]),
        nn.Linear(network.hidden_dim * X_train.shape[1], 100),
        nn.LayerNorm(100),
        nn.Linear(100, np.unique(y_train).shape[0])
    )
    batch_size = 257
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    learning_rate_adjusting = False

    # fine-tune the model
    model.fit(X_train, y_train, fine_tuning_type=fine_tuning_type, init_optimizer=init_optimizer, num_epochs=num_epochs,
              batch_size=batch_size, criterion=criterion, head=head, learning_rate_adjusting=learning_rate_adjusting)
    # evaluate the performance
    y_pred = model.predict(X_test)
    print(f'Accuracy on the test set is {np.mean(y_test == y_pred)}')

    print(
        f"The test for single channel classification with fine_tuning_type={fine_tuning_type} and device={device} has been passed succesfully."
    )
