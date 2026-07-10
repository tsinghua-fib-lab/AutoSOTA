import torch
import torch.nn as nn
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import numpy as np

class ReVIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ReVIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x, return_stats=False):
        # x: [B, L, C]
        mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, C]
        var = torch.var(x, dim=1, keepdim=True)  # [B, 1, C]

        self.mean = mean
        self.var = var

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        x_out = norm_x * self.weight + self.bias

        if return_stats:
            return x_out, (mean, var)
        return x_out

    def reverse(self, x_out, mean=None, var=None):
        if mean is None:
            mean = self.mean
        if var is None:
            var = self.var

        x_normalized = (x_out - self.bias) / self.weight
        x_reversed = x_normalized * torch.sqrt(var + self.eps) + mean
        return x_reversed


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.inv_len = configs.inv_len
        self.patch_size =configs.patch_size
        self.time_dim=configs.time_dim
        self.c_out=configs.enc_in
        self.dsa=configs.dsa
        self.dsb=configs.dsb
        self.ksize=configs.ksize

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(self.seq_len, 512),
            nn.GELU(),
            nn.Linear(512, self.pred_len),
        )
        self.router_sender = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
            self.d_model, 1
        )
        self.router_receiver = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
            self.d_model, 1
        )
        self.projection_1 = nn.Linear(self.seq_len // self.patch_size, self.d_model)
        self.projection_2 = nn.Linear(self.pred_len // self.patch_size, self.d_model)

        self.projection_3 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.GELU(),
            nn.Linear(self.d_model*2, self.pred_len // self.patch_size)
        )

        self.revin = ReVIN(self.enc_in)

        self.codebook = nn.Parameter(torch.randn(self.inv_len, self.enc_in))
        nn.init.xavier_normal_(self.codebook)
        
        self.time_enc_x = nn.Sequential(
                                      nn.Linear(self.time_dim, self.c_out//self.dsa),
                                      nn.LayerNorm(self.c_out//self.dsa),
                                      nn.GELU(),
                                      nn.Linear(self.c_out//self.dsa, self.c_out//self.dsb),
                                      nn.LayerNorm(self.c_out//self.dsb),
                                      nn.GELU(),
                                      nn.Conv1d(in_channels=self.seq_len,out_channels=self.seq_len,kernel_size=self.ksize,padding='same'),
                                      nn.Linear(self.c_out//self.dsb, self.c_out),
                                      )
        self.time_enc_y = nn.Sequential(
                                      nn.Linear(self.time_dim, self.c_out//self.dsa),
                                      nn.LayerNorm(self.c_out//self.dsa),
                                      nn.GELU(),
                                      nn.Linear(self.c_out//self.dsa, self.c_out//self.dsb),
                                      nn.LayerNorm(self.c_out//self.dsb),
                                      nn.GELU(),
                                      nn.Conv1d(in_channels=self.pred_len,out_channels=self.pred_len,kernel_size=self.ksize,padding='same'),
                                      nn.Linear(self.c_out//self.dsb, self.c_out),
                                      )


    def norm(self, x, cycle_index, timestamp_x):

        gather_index = (cycle_index.view(-1, 1) - torch.arange(self.seq_len, device=x.device).view(1, -1)) % self.inv_len
        phase_anchor_x = self.codebook[gather_index]+timestamp_x

        residual_x = x - phase_anchor_x

        residual_x_norm, (mean, var) = self.revin(residual_x, return_stats=True)

        x_in = residual_x_norm + phase_anchor_x

        return x_in,phase_anchor_x, mean, var

    def phase_router(self, y_backbone, phase_anchor_x):


        b, t, c = y_backbone.size()

        feat_x = phase_anchor_x.permute(0, 2, 1) 
        feat_y = y_backbone.permute(0, 2, 1)  


        feat_x = feat_x.reshape(b * c, -1, self.patch_size) 
        feat_y = feat_y.reshape(b * c, -1, self.patch_size)


        feat_x = self.projection_1(feat_x.permute(0, 2, 1))
        feat_y = self.projection_2(feat_y.permute(0, 2, 1))


        router_buffer, _ = self.router_sender(feat_x, feat_y, feat_y, attn_mask=None)
        router_receive, _ = self.router_receiver(feat_y, router_buffer, router_buffer, attn_mask=None)


        E = self.projection_3(router_receive).permute(0, 2, 1).reshape(b, c, -1).permute(0, 2, 1)

        E = E[:, -self.pred_len:, :]

        return E

    def denorm(self, y_backbone, phase_anchor_x, mean, var,timestamp_y):

        phase_anchor_y = self.phase_router(y_backbone, phase_anchor_x)+timestamp_y


        residual_y = y_backbone - phase_anchor_y

        residual_y_denorm = self.revin.reverse(residual_y, mean=mean, var=var)


        y_final = residual_y_denorm + phase_anchor_y

        return y_final

    def forward(self, x, cycle_index,batch_x_mark,batch_y_mark,do_mixup=True):
        timestamp_x = self.time_enc_x(batch_x_mark)
        timestamp_y = self.time_enc_y(batch_y_mark)

        x_in, phase_anchor_x, mean, var = self.norm(x, cycle_index,timestamp_x)

        mixup_params = None
        if self.training and do_mixup:
            lam = np.random.beta(0.15, 0.15)
            batch_size = x.shape[0]
            index = torch.randperm(batch_size).to(x.device)

            x_in_mixed = lam * x_in + (1.0 - lam) * x_in[index, :]
            mean_mixed = lam * mean + (1.0 - lam) * mean[index, :]
            sigma = torch.sqrt(var + 1e-5)
            sigma_mixed = lam * sigma + (1.0 - lam) * sigma[index, :]
            var_mixed = sigma_mixed * sigma_mixed - 1e-5
            mixup_params = (lam, index)
        else:
            x_in_mixed = x_in
            mean_mixed = mean
            var_mixed = var

        y_backbone = self.backbone(x_in_mixed.permute(0, 2, 1)).permute(0, 2, 1)

        y_final = self.denorm(y_backbone,phase_anchor_x,mean_mixed,var_mixed,timestamp_y)
        
        if self.training:
            return y_final, mixup_params
        else:
            return y_final
