# ---------------------------------------------------------------------------------
# Portions of this file are derived from Large-Time-Series-Model
# - Source: https://github.com/thuml/Large-Time-Series-Model
# - Paper: Timer: Generative Pre-trained Transformers Are Large Time Series Models
# - License: MIT License

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
from einops import rearrange, repeat
from torch import nn

from probts.model.forecaster import Forecaster


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2402.02368.pdf
    """

    def __init__(self, ckpt_path):
        super().__init__()
        if ckpt_path and ckpt_path != "":
            if ckpt_path.endswith('.pt'):
                # print(f"Loading Timer model from {ckpt_path}")
                self.timer = torch.jit.load(ckpt_path)
        else:
            raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.timer(x_enc, x_mark_enc, x_dec, x_mark_dec)


class Timer(Forecaster):
    def __init__(
        self,
        label_len: int = 576,
        ckpt_path: str = None,
        ckpt_path_finetune: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.no_training = True
        
        self.output_patch_len = 96 # fixed by the pre-trained model
        self.label_len = label_len

        # Load Timer
        self.model = Model(ckpt_path)
        if ckpt_path_finetune:
            print(f"Loading Timer finetune model from {ckpt_path_finetune}")
            self.model.load_state_dict(torch.load(ckpt_path_finetune))
       

    def forecast(self, batch_data, num_samples=None):        
        # for now, we only support batch_size=1
        B, _, K = batch_data.past_target_cdf.shape
        inputs = batch_data.past_target_cdf[:, -self.context_length:, ...]
        x_mark_enc = batch_data.past_time_feat[:, -self.context_length:, ...]
        x_mark_dec = batch_data.future_time_feat
        x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec], dim=1)

        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        x_mark_enc = repeat(x_mark_enc, 'b l f -> (b k) l f', k=K)
        x_mark_dec = repeat(x_mark_dec, 'b l f -> (b k) l f', k=K)

        dec_inp = torch.zeros_like(inputs[:, -self.prediction_length:, :]).float()
        dec_inp = torch.cat((inputs[:, -self.label_len:, ...], dec_inp), dim=1).float()

        inference_steps = self.prediction_length // self.output_patch_len
        dis = self.prediction_length - inference_steps * self.output_patch_len
        if dis != 0:
            inference_steps += 1

        pred_y = []

        for j in range(inference_steps):
            if len(pred_y) != 0:
                inputs = torch.cat([inputs[:, self.output_patch_len:, :], pred_y[-1]], dim=1)
                tmp = x_mark_dec[:, j - 1:j, :]
                x_mark_enc = torch.cat([x_mark_enc[:, 1:, :], tmp], dim=1)

            outputs = self.model(inputs, x_mark_enc, dec_inp, x_mark_dec)
            pred_y.append(outputs[:, -self.output_patch_len:, :])

        pred_y = torch.cat(pred_y, dim=1)
        if dis != 0:
            pred_y = pred_y[:, :-dis, :]
        pred_y = rearrange(pred_y, '(b k) l 1 -> b l k', b=B, k=K)
        pred_y = pred_y[:, :self.prediction_length, :]
        return pred_y.unsqueeze(1)
