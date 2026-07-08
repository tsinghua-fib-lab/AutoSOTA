import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GlobalBatchWeighting(nn.Module):
    def __init__(self, d_model):
        super(GlobalBatchWeighting, self).__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats_norm, feats_ano, feats_query):
        ref_features = torch.cat([feats_norm, feats_query], dim=0)

        mu_global = ref_features.mean(dim=0)
        var_global = ref_features.var(dim=0, unbiased=False) + 1e-6
        mu_ano = feats_ano.mean(dim=0)

        snr_score = (mu_ano - mu_global).pow(2) / var_global
        weights = self.sigmoid(snr_score * self.scale + self.bias)

        return weights

class PromptGADModel(nn.Module):
    def __init__(self, input_feature_dim=5, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1,
                 k_shot=10):
        super(PromptGADModel, self).__init__()
        self.d_model = d_model
        self.k_shot = k_shot

        self.feature_proj = nn.Linear(input_feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.batch_weighting = GlobalBatchWeighting(d_model)
        self.metric_proj = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def _generate_inductive_mask(self, k, batch_size, device):
        total_len = 2 * k + batch_size
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        mask[:2 * k, :2 * k] = 0.0
        mask[2 * k:, :2 * k] = 0.0
        query_self_mask = torch.eye(batch_size, device=device)
        query_self_mask = query_self_mask.masked_fill(query_self_mask == 0, float('-inf'))
        query_self_mask = query_self_mask.masked_fill(query_self_mask == 1, 0.0)
        mask[2 * k:, 2 * k:] = query_self_mask
        return mask

    def forward(self, support_normal, support_anomaly, query_samples):
        k = support_normal.size(0)
        batch_size = query_samples.size(0)

        emb_sup_norm = self.feature_proj(support_normal)
        emb_sup_ano = self.feature_proj(support_anomaly)
        emb_query = self.feature_proj(query_samples)

        full_sequence = torch.cat([emb_sup_norm, emb_sup_ano, emb_query], dim=0).unsqueeze(0)
        src_mask = self._generate_inductive_mask(k, batch_size, full_sequence.device)
        enc_output = self.transformer(full_sequence, mask=src_mask).squeeze(0)

        feats_norm = enc_output[0: k]
        feats_ano = enc_output[k: 2 * k]
        feat_query = enc_output[2 * k:]

        weights = self.batch_weighting(feats_norm, feats_ano, feat_query)

        feats_norm_w = feats_norm * weights
        feats_ano_w = feats_ano * weights
        feat_query_w = feat_query * weights

        feats_norm_p = self.metric_proj(feats_norm_w)
        feats_ano_p = self.metric_proj(feats_ano_w)
        feat_query_p = self.metric_proj(feat_query_w)

        feats_norm_p = F.normalize(feats_norm_p, dim=-1)
        feats_ano_p = F.normalize(feats_ano_p, dim=-1)
        feat_query_p = F.normalize(feat_query_p, dim=-1)

        sim_n = torch.matmul(feat_query_p, feats_norm_p.T)
        sim_a = torch.matmul(feat_query_p, feats_ano_p.T)

        all_sims = torch.cat([sim_n, sim_a], dim=1)
        attn_weights = F.softmax(all_sims * self.logit_scale.exp(), dim=1)

        indicator = torch.cat([
            torch.ones(1, k, device=all_sims.device) * -1.0,
            torch.ones(1, k, device=all_sims.device) * 1.0
        ], dim=1)

        raw_score = torch.sum(attn_weights * indicator, dim=1, keepdim=True)
        final_prob = (1 + raw_score) / 2

        return final_prob