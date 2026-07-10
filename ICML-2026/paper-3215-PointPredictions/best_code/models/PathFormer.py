import torch
import torch.nn as nn
from layers.AMS import AMS
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin


        self.aux_loss = 0

        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.batch_norm = configs.batch_norm

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_feature=False):

        balance_loss = 0

        # Norm
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')


        out = self.start_fc(x_enc.unsqueeze(-1))

        batch_size = x_enc.shape[0]

        # Pass through AMS layers (Adaptive Multi-Scale)
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss


        self.aux_loss = balance_loss

        # --- Capture Latent Feature for KD (Knowledge Distillation) ---
        # out shape: [Batch, Seq_Len, Num_Nodes, D_Model]
        # Permute to [Batch, Num_Nodes, Seq_Len, D_Model] for alignment
        latent_feature = out.permute(0, 2, 1, 3)
        # ------------------------------------------------------------

        # Flatten for the projection head: [B, N, S, D] -> [B, N, S*D]
        out = latent_feature.reshape(batch_size, self.num_nodes, -1)

        # Project to prediction length: [B, N, S*D] -> [B, N, Pred_Len] -> [B, Pred_Len, N]
        out = self.projections(out).transpose(2, 1)

        # De-norm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        if return_feature:
            return out, latent_feature

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_feature=False):

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':

            output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_feature)


            if return_feature:

                dec_out, feature = output
                return dec_out[:, -self.pre_len:, :], feature
            else:

                dec_out = output
                return dec_out[:, -self.pre_len:, :]


        return None