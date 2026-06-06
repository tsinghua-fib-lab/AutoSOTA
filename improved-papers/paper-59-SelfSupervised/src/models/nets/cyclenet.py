import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.snrlinear import SNLinear

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index].transpose(-1, -2)


class Model(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 num_channels: int, 
                 dim: int,
                 model_type: str,
                 cycle_len: int, 
                 with_revin: bool, 
                 if_snr: bool,
                 loss_names: list[str], 
                 **kwargs):
        super(Model, self).__init__()

        self.seq_len = input_size
        self.pred_len = output_size
        self.enc_in = num_channels
        self.cycle_len = cycle_len
        self.model_type = model_type
        self.d_model = dim
        self.use_revin = with_revin

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            if if_snr:
                self.model = nn.Sequential(
                    SNLinear(self.seq_len, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.pred_len)
                )
            else:
                self.model = nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.pred_len)
                )

    def predict(self, x, x_mark=None, *args, **kwargs):
        cycle_index = x_mark['cycle_index']
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=-1, keepdim=True)
            seq_var = torch.var(x, dim=-1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len).repeat(x.shape[0] // x_mark['cycle_index'].shape[0], 1, 1)

        # forecasting with channel independence (parameters-sharing)
        y_hat = self.model(x)

        # add back the cycle of the output data
        y_hat = y_hat + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len).repeat(y_hat.shape[0] // x_mark['cycle_index'].shape[0], 1, 1)

        # instance denorm
        if self.use_revin:
            y_hat = y_hat * torch.sqrt(seq_var) + seq_mean

        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None, mode='train', **kwargs):
        # x: (batch_size, enc, seq_len), cycle_index: (batch_size,)
        result = {}
        y_hat = self.predict(x, x_mark)

        result['y_hat'] = y_hat
        result['loss_tar'] = F.mse_loss(y_hat, y)
        result['loss_total'] = result['loss_tar']
        return result
