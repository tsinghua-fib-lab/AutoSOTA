import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = gcn(seq1, adj, sparse)

        h_3 = h_1 * self.prompt

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = gcn(seq2, adj, sparse)

        h_4 = h_2 * self.prompt

        ret = self.disc(c, h_3, h_4
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
