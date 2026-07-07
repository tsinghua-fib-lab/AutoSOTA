import torch
from torch import nn
"""
TungKieu, BinYang, andChristian S. Jensen. Outlier detection for multidimensional time series
 using deep neural networks. In IEEE International Conference on Mobile Data Management,
 pages 125–134, 2018.
"""

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)

    def forward(self, X):
        outputs, (hidden, cell) = self.lstm(X)
        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, X, hidden, cell):
        output, (hidden, cell) = self.lstm(X, (hidden, cell))
        output = self.relu(output)
        output = self.fc(output)
        return output, hidden, cell


class Model(nn.Module):
    def __init__(self, configs):#input_size=8, hidden_size=4, num_layers=2, dropout=0.1
        super(Model, self).__init__()
        self.task_name = configs.task_name
        input_size = configs.c_out
        hidden_size = configs.d_model
        num_layers = configs.e_layers
        dropout = configs.dropout
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc.sub(means)
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc.div(stdev)

            batch_size, sequence_length, feature_length = x_enc.size()
            hidden, cell = self.encoder(x_enc)

            output = []
            temp_input = torch.zeros((batch_size, 1, feature_length), dtype=torch.float).to(x_enc.device)
            for t in range(sequence_length):
                temp_input, hidden, cell = self.decoder(temp_input, hidden, cell)
                output.append(temp_input)

            inv_idx = torch.arange(sequence_length - 1, -1, -1).long()  # 翻转
            dec_out = torch.cat(output, dim=1)[:, inv_idx, :]

            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out.mul(
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1,  sequence_length, 1)))
            dec_out = dec_out.add(
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, sequence_length, 1)))
            return dec_out
        return None