import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from chronos import BaseChronosPipeline


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        model_path = getattr(configs, 'pretrained_model_path', 'amazon/chronos-bolt-base')
        local_files_only = getattr(configs, 'local_files_only', False)
        device_map = getattr(configs, 'device_map', None)
        if device_map is None or (str(device_map).startswith("cuda") and not torch.cuda.is_available()):
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device_map,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = torch.reshape(x_enc, (B*C, L))
        output = self.model.predict(x_enc, prediction_length=self.pred_len)
        output = output.mean(dim=1)
        dec_out = torch.reshape(output, (B, output.shape[-1], C)).to(x_enc.device)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
