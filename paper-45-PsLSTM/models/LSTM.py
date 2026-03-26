import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.embedding_dim = configs.embedding_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.channel, hidden_size=self.embedding_dim, 
                            num_layers=configs.num_layers, batch_first=True)

        self.projection = nn.Linear(self.embedding_dim, self.pred_len * self.channel)

    def forward(self, x):

        # x: [Batch, Input length, Channel]
        
        x, (h_n, c_n) = self.lstm(x)  # x: [Batch, Input length, Hidden_dim]
        
        x = h_n[-1]  # [Batch, Hidden_dim]

        x = self.projection(x)  # [Batch, Pred_len * Channel]

        x = x.view(-1, self.pred_len, self.channel)  # [Batch, Pred_len, Channel]

        return x
