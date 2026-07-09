import torch
import math
import torch.nn as nn



class LSTM(nn.Module):

    def __init__(self, embedding_dim, vocab_size, num_layers, dropout_rate=0.65):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.n_embd = embedding_dim


        self.word_embeddings = nn.Embedding(vocab_size, self.n_embd)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=self.n_embd,
                            hidden_size=self.n_embd,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=dropout_rate)

        # The linear layer that maps from hidden state space to tag space
        self.head = nn.Linear(self.n_embd, vocab_size)
        
        self.init_weights()

    def forward(self, sentence, prev_state):

        embeds = self.word_embeddings(sentence)

        lstm_out, state = self.lstm(embeds, prev_state)

        logits = self.head(lstm_out)

        return logits, state

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.n_embd)
        self.word_embeddings.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.head.weight.data.uniform_(-init_range_other, init_range_other)
        self.head.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.n_embd,
                    self.n_embd).uniform_(-init_range_other, init_range_other) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.n_embd, 
                    self.n_embd).uniform_(-init_range_other, init_range_other) 

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.n_embd).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.n_embd).to(device)
        return hidden, cell 


