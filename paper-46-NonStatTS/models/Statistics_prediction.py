import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_


class Statistics_prediction(nn.Module):
    def __init__(self, configs):
        super(Statistics_prediction, self).__init__()
        self.configs = configs
        self.seq_len = configs.DATA.SEQ_LEN
        self.pred_len = configs.DATA.PRED_LEN
        self.period_len = configs.DATA.PERIOD_LEN
        self.channels = configs.DATA.N_VAR if configs.DATA.FEATURES == 'M' else 1
        self.station_type = configs.DATA.STATION_TYPE

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        configs_new = copy.deepcopy(self.configs)
        configs_new.DATA.SEQ_LEN = self.configs.DATA.SEQ_LEN // self.period_len
        configs_new.DATA.LABEL_LEN = self.configs.DATA.LABEL_LEN // self.period_len
        configs_new.DATA.PRED_LEN = self.pred_len_new
        self.model = MLP(configs_new, mode='mean').float()
        self.model_std = MLP(configs_new, mode='std').float()

    def normalize(self, input):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            std = torch.std(input, dim=-2, keepdim=True)
            norm_input = (input - mean) / (std + self.epsilon)
            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)
            outputs_mean = self.model(mean.squeeze(2) - mean_all, input - mean_all) * self.weight[0] + mean_all * \
                           self.weight[1]
            outputs_std = self.model_std(std.squeeze(2), input)

            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)

            return norm_input.reshape(bs, len, dim), outputs[:, -self.pred_len_new:, :]

        else:
            return input, None

    def de_normalize(self, input, station_pred):
        if self.station_type == 'adaptive':
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = station_pred[:, :, :self.channels].unsqueeze(2)
            std = station_pred[:, :, self.channels:].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, len, dim)


        else:
            return input


class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.DATA.SEQ_LEN
        self.pred_len = configs.DATA.PRED_LEN
        self.channels = configs.DATA.N_VAR if configs.DATA.FEATURES == 'M' else 1
        self.period_len = configs.DATA.PERIOD_LEN
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
