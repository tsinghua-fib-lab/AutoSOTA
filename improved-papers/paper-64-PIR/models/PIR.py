import torch
import numpy as np
import torch.nn as nn
from . import PatchTST, iTransformer, TimeMixer, SparseTSF
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import copy
import time


class QualityEstimator(nn.Module):
    def __init__(self, configs):
        super(QualityEstimator, self).__init__()
        self.seq_proj = nn.Linear(configs.seq_len, configs.refine_d_model)
        self.pred_proj = nn.Linear(configs.pred_len, configs.refine_d_model)
        self.activation = nn.Sigmoid()
        self.loss_estimation = nn.Sequential(
            nn.Linear(3 * configs.refine_d_model, configs.refine_d_model),
            nn.GELU(),
            nn.Linear(configs.refine_d_model, 1),
            nn.ReLU()
        )
        self.quality_estimation = nn.Sequential(
            nn.Linear(1, 1),
            self.activation
        )
        self.retrieval_weight = nn.Sequential(
            nn.Linear(configs.retrieval_num + 1, configs.refine_d_model),
            nn.GELU(),
            nn.Linear(configs.refine_d_model, 1),
            self.activation
        )

        self.quality_estimation[0].weight = nn.Parameter(torch.ones(1, 1))
        self.quality_estimation[0].bias = nn.Parameter(torch.zeros(1, ))

    def forward(self, x_enc, x_pred, sims, channel_indicator):
        x_enc = self.seq_proj(x_enc.permute(0, 2, 1))
        pred_enc = self.pred_proj(x_pred.permute(0, 2, 1))
        loss_estimated = self.loss_estimation(torch.cat([x_enc, pred_enc, channel_indicator], dim=-1))
        alpha = self.quality_estimation(loss_estimated).permute(0, 2, 1)
        beta = self.retrieval_weight(torch.cat([loss_estimated, sims], dim=-1)).permute(0, 2, 1)
        return loss_estimated, alpha, beta


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_c = configs.enc_in
        self.including_time_features = configs.including_time_features
        self.retrieval_stride = configs.retrieval_stride
        self.use_norm = configs.use_norm

        self._build_model()
        self.refine_embedding = nn.Linear(configs.pred_len, configs.refine_d_model)
        self.refiner = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.refine_d_model,
                        configs.n_heads),
                    configs.refine_d_model,
                    configs.refine_d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.refine_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.refine_d_model)
        )
        self.refine_projection = nn.Linear(configs.refine_d_model, configs.pred_len)
        self.quality_estimator = QualityEstimator(configs)
        self.channel_indicator = nn.Parameter(
            torch.randn(configs.enc_in, configs.refine_d_model) + torch.ones(configs.enc_in, configs.refine_d_model))
        self.retrieval_mode = 'series'
        self.retrieval_num = configs.retrieval_num
        self.time_projection_point = nn.Linear(5, 1)
        self.time_projection_temporal = nn.Linear(configs.pred_len, configs.refine_d_model)
        self.time_backbone, self.time_retrieval, self.time_revision = 0, 0, 0
        # Learnable temperature for retrieval softmax
        self.retrieval_temperature = nn.Parameter(torch.ones(1) * 0.5)

    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimeMixer': TimeMixer,
            'SparseTSF': SparseTSF,
        }
        config_model = copy.deepcopy(self.configs)
        self.model = model_dict[self.configs.backbone].Model(config_model).float()

    def construct_index(self, num):
        key_len = self.seq_len if self.retrieval_mode == 'series' else self.configs.d_model
        self.keys = torch.zeros(num, key_len, self.in_c, device=self.channel_indicator.device)
        self.values = torch.zeros(num, self.pred_len, self.in_c, device=self.channel_indicator.device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc, y, index):
        bs = x_enc.shape[0]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            y = (y - means) / stdev
        if self.retrieval_mode == 'series':
            x_key = x_enc
        elif self.retrieval_mode == 'embedding':
            x_key = self.model.encode(x_enc)
        else:
            raise NotImplementedError
        self.keys[index, :, :] = x_key
        self.values[index, :, :] = y
        self.index += bs
        torch.cuda.empty_cache()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, index=None, mode='pretrain', timing=False):
        bs = x_dec.shape[0]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if mode == 'pretrain':
            dec_out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.use_norm:
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            return dec_out

        else:
            t0 = time.time()
            intermediate_results = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            t1 = time.time()
            retrieval_results, sims, t = self.retrieval(x_enc, index)
            t2 = time.time()

            refine_enc = self.refine_embedding(intermediate_results.permute(0, 2, 1))
            if self.including_time_features:
                time_embedding_point = self.time_projection_point(x_mark_dec).permute(0, 2, 1)
                time_embedding = self.time_projection_temporal(time_embedding_point)
                refine_enc = torch.cat([time_embedding, refine_enc], dim=1)
                refine_out, _ = self.refiner(refine_enc)
                refine_out = self.refine_projection(refine_out)[:, 1:, :].permute(0, 2, 1)
            else:
                refine_out, _ = self.refiner(refine_enc)
                refine_out = self.refine_projection(refine_out).permute(0, 2, 1)
            loss_estimated, alpha, beta = self.quality_estimator(x_enc, intermediate_results, sims,
                                                                 self.channel_indicator.unsqueeze(0).repeat(bs, 1, 1))
            dec_out = intermediate_results + alpha * refine_out + beta * retrieval_results
            t3 = time.time()
            if timing:
                self.time_backbone += t1 - t0
                self.time_retrieval += t2 - t1
                self.time_revision += t3 - t2
            if self.use_norm:
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

                intermediate_results = intermediate_results * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                intermediate_results = intermediate_results + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

            return dec_out, (intermediate_results, loss_estimated)

    def retrieval(self, x, index):
        bs = x.shape[0]
        k = self.retrieval_num
        if self.retrieval_mode == 'series':
            queries = x
        else:
            queries = self.model.encode(x)
        keys = self.keys
        # keys = self.keys.transpose(2, 1).reshape(-1, self.seq_len)
        t0 = time.time()
        dis = self.cosine_similarity(queries, keys)
        if self.training:
            # offline
            self_range = torch.arange(-self.configs.seq_len, self.configs.seq_len + 1, device=x.device).unsqueeze(0)
            invalid_index = index.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.retrieval_stride
            invalid_index[torch.where(invalid_index < 0)] = 0
            invalid_index[torch.where(invalid_index >= self.index)] = self.index - 1
            row_idx = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, 2 * self.configs.seq_len + 1)
            dis[:, row_idx, invalid_index] = -100

            # online
            # invalid_index = torch.arange(self.index).unsqueeze(0).repeat(bs,1) #bs*len
            # index = index // self.retrieval_stride
            # for i in range(bs):
            #     mask_index = min(max(k, index[i]),self.index - 1)
            #     invalid_index[i, :mask_index] = mask_index
            # row_idx = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, self.index)
            # dis[:, row_idx, invalid_index] = 0
        dis_topk, indices_topk = torch.topk(dis, dim=2, k=k)
        sims = dis_topk.permute(1, 0, 2) # bs*c*k
        # Temperature-scaled softmax: temperature controls concentration of retrieval weights
        temp = torch.clamp(self.retrieval_temperature.abs(), min=0.1, max=2.0)
        probs_topk = torch.softmax(dis_topk / temp, dim=2).unsqueeze(-1)  # c*bs*k*1
        t = time.time()-t0

        # values = self.values.permute(2, 0, 1)[torch.arange(self.in_c).unsqueeze(1).repeat(1, bs * k),
        #          indices_topk.view(self.in_c, -1), :]
        # values = values.reshape(self.in_c, bs, -1, self.pred_len)
        values = self.value_permute  # [in_c, N, pred_len]

        # reshape 为 [1, in_c, N, pred_len]，为 batch gather 做准备
        values = values.unsqueeze(0)  # [1, in_c, N, pred_len]

        # indices_topk.shape = [bs, in_c, k]
        # 需要扩展为 [bs, in_c, k, 1] 以便 gather
        indices = indices_topk.permute(1, 0, 2).unsqueeze(-1)  # [in_c, bs, k, 1]

        # 转换 values 为 [in_c, 1, N, pred_len] 以与 indices 对齐
        values = values.expand(bs, -1, -1, -1)  # [in_c, 1, N, pred_len]

        # gather
        values = torch.gather(values, 2, indices.expand(-1, -1, -1, values.size(-1))).permute(1,0,2,3)  # [in_c, bs, k, pred_len]

        output = torch.sum(probs_topk * values, dim=2).permute(1, 2, 0)  # weighted-sum ver
        return output, sims, 0

    def cosine_similarity(self, queries, keys):
        # equals to person_similarity when revin=True, since std=1, mean=0
        if len(queries.shape) == 2:  # B*L
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=-1)
            k_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        elif len(queries.shape) == 3:  # B*L*C
            queries = queries.permute(2, 0, 1)
            keys = keys.permute(2, 0, 1)
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=-1)
            k_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.permute(0, 2, 1))
