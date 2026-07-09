import pytest
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


@pytest.mark.parametrize("fine_tuning_type", ['head', 'full'])
@pytest.mark.parametrize("adapter",  [None, 'pca', 'lcomb'])
@pytest.mark.parametrize("device", ['cpu'])
def test_multichannel_fine_tune(fine_tuning_type, adapter, device):
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
    new_num_channels = 5
    standalone_adapters = ['pca', 'var', 'svd', 'rand']
    if adapter in standalone_adapters:
        if adapter == 'var':
            from mantis.adapters import VarianceBasedSelector
            adapter = VarianceBasedSelector(new_num_channels=new_num_channels)
            adapter.fit(X_train)
            X_train, X_test = adapter.transform(
                X_train), adapter.transform(X_test)
        else:
            from mantis.adapters import MultichannelProjector
            adapter = MultichannelProjector(
                new_num_channels=new_num_channels, patch_window_size=1, base_projector=adapter)
            adapter.fit(X_train)
            X_train, X_test = adapter.transform(
                X_train), adapter.transform(X_test)
        adapter = None
    elif adapter is not None:
        if adapter == 'lcomb':
            from mantis.adapters import LinearChannelCombiner
            adapter = LinearChannelCombiner(num_channels=X_train.shape[1], new_num_channels=new_num_channels)
        # change fine_tuning_type to learn adapter as well
        if fine_tuning_type == 'head':
            fine_tuning_type = 'adapter_head'

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
    if adapter is None:
        head = nn.Sequential(
            nn.LayerNorm(network.hidden_dim * X_train.shape[1]),
            nn.Linear(network.hidden_dim * X_train.shape[1], 100),
            nn.LayerNorm(100),
            nn.Linear(100, np.unique(y_train).shape[0])
        )
    else:
        head = nn.Sequential(
            nn.LayerNorm(network.hidden_dim * new_num_channels),
            nn.Linear(network.hidden_dim * new_num_channels, 100),
            nn.LayerNorm(100),
            nn.Linear(100, np.unique(y_train).shape[0])
        )
    batch_size = 257
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    learning_rate_adjusting = False

    # fine-tune the model
    model.fit(X_train, y_train, fine_tuning_type=fine_tuning_type, init_optimizer=init_optimizer, num_epochs=num_epochs,
              batch_size=batch_size, criterion=criterion, head=head, adapter=adapter, learning_rate_adjusting=learning_rate_adjusting)

    # evaluate the performance
    y_pred = model.predict(X_test)
    print(f'Accuracy on the test set is {np.mean(y_test == y_pred)}')

    print(
        f"The test for multichannel classification with fine_tuning_type={fine_tuning_type}, adapter={adapter} and device={device} has been passed succesfully."
    )
