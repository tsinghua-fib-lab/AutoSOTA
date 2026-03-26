import torch.nn as nn


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #x_enc is of shape B L N, where B is the batch size, L is the sequence length, and N is the number of variates
        #This is the seasonal naive model. Thus, dec_out is of len pred_len, and dec_out[:7] = x_enc[-7:]. This is repeated for the next 7 days, and so on.
        dec_out = x_enc[:, -7:, :].repeat(1, self.pred_len//7 + 1, 1)[:,:self.pred_len,:]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]