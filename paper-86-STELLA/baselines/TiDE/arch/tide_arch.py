import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import data_transformation_4_xformer

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


# TiDE
class TiDE(nn.Module):
    """
    Paper: Long-term Forecasting with TiDE: Time-series Dense Encoder
    Official Code: https://github.com/lich99/TiDE
    Link: https://arxiv.org/abs/2304.08424
    Venue: TMLR 2023
    Task: Long-term Time Series Forecasting
    """

    def __init__(self,  **model_args):
        super(TiDE, self).__init__()

        self.seq_len = model_args['seq_len']  # L
        self.label_len = int(model_args['label_len'])
        self.pred_len = model_args['pred_len']  # H
        self.hidden_dim = model_args['d_model']
        self.res_hidden = model_args['d_model']
        self.encoder_num = model_args['e_layers']
        self.decoder_num =  model_args['d_layers']
        # self.freq = model_args["freq"]
        self.bias = model_args["bias"]
        self.feature_encode_dim = model_args["feature_encode_dim"]
        self.decode_dim = model_args['c_out']
        self.temporalDecoderHidden = model_args['d_ff']
        dropout = model_args["dropout"]

        # self.feature_dim = freq_map[self.freq]
        self.feature_dim = model_args["feature_dim"]

        if self.feature_dim > 0:
            flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim
            self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, self.bias)
        else:
            flatten_dim = self.seq_len

        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, self.bias), *(
                    [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (self.encoder_num - 1)))

        self.decoders = nn.Sequential(*(
                        [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (
                            self.decoder_num - 1)),
                                          ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len,
                                                   dropout, self.bias))
        if self.feature_dim > 0:
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1,
                                            dropout, self.bias)
        else:
            self.temporalDecoder = ResBlock(self.decode_dim, self.temporalDecoderHidden, 1, dropout, self.bias)
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=self.bias)


    def forward_xformer(self, x_enc, x_mark_enc, x_dec, batch_y_mark) -> torch.Tensor:
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        if self.feature_dim > 0:
            feature = self.feature_encoder(batch_y_mark)
            hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        else:
            hidden = self.encoders(x_enc)
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
        
        if self.feature_dim > 0:
            dec_out = self.temporalDecoder(torch.cat([feature[:, self.seq_len:], decoded], dim=-1)).squeeze(
                -1) + self.residual_proj(x_enc)
        else:
            dec_out = self.temporalDecoder(decoded).squeeze(-1) + self.residual_proj(x_enc)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out



    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        x_enc, x_mark_enc, x_dec, batch_y_mark = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=0)

        batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)

        batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)
        dec_out = torch.stack([self.forward_xformer(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) for feature in
                               range(x_enc.shape[-1])], dim=-1)
        return dec_out.unsqueeze(-1)  # [B, L, D]