import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    def __init__(self, n_features, n_lookback, n_lstm_layers, n_hidden_size):
        super().__init__()
        self.n_features = n_features
        self.lookback = n_lookback
        self.n_lstm_layers = n_lstm_layers
        self.n_hidden_size = n_hidden_size
        self.fcnn_in_size = self.n_hidden_size * self.lookback

        # LSTM
        self.lstm_model = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        # FCNN
        self.FCLayer1 = nn.Linear(self.fcnn_in_size, self.fcnn_in_size // 2)
        self.FCLayer2 = nn.Linear(self.fcnn_in_size // 2, self.fcnn_in_size // 4)
        self.FCLayer3 = nn.Linear(self.fcnn_in_size // 4, 1)
        self.prelu1 = nn.PReLU(self.fcnn_in_size // 2)
        self.prelu2 = nn.PReLU(self.fcnn_in_size // 4)

        # Dummy variable for inferring the model device
        self.device_detector_parameter = nn.Parameter(torch.empty(0))

    def forward(self, x):
        # allow only for batched inputs
        if self.n_features > 1:
            if len(x.shape) != 3:
                raise ValueError("Only accepting batched input!")
        else:
            if len(x.squeeze) != 2:
                raise ValueError("Only accepting batched input!")

        # initialize output and cell state
        ht, ct = self._init_hidden(x.shape)

        # get output of LSTM
        x, (ht, ct) = self.lstm_model(x, (ht.detach(), ct.detach()))

        # pass through fully connected layers
        return self._forward_fcnn(torch.cat(x.unbind(dim=1), dim=1))

    def _init_hidden(self, shape):
        # define initial h0 and c0 as zeros

        ht = torch.zeros((self.n_lstm_layers, shape[0], self.n_hidden_size)).to(
            self._get_device()
        )
        ct = torch.zeros((self.n_lstm_layers, shape[0], self.n_hidden_size)).to(
            self._get_device()
        )

        return ht, ct

    def _get_device(self):
        return self.device_detector_parameter.device

    def _forward_fcnn(self, x):
        # pass LSTM outputs through a FCNN
        x = self.FCLayer1(x)
        x = self.prelu1(x)
        x = self.FCLayer2(x)
        x = self.prelu2(x)
        x = self.FCLayer3(x)

        # rake absolute value of output to ensure non-negativity
        return x.abs()
