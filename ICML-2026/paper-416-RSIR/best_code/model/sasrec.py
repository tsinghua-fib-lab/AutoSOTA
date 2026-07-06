import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer
from module import data_augmentation
from data import dataset
from copy import deepcopy

class SASRecQueryEncoder(torch.nn.Module):
    def __init__(
            self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional=False, training_pooling_type='origin', eval_pooling_type='last') -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.training_pooling_layer = SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type=self.eval_pooling_type)

    def forward(self, batch, need_pooling=True):
        seq_len = batch['seqlen']
        if batch.get('seq_emb', None) is None:
            user_hist = batch['in_'+self.fiid]
            positions = torch.arange(user_hist.size(1), dtype=torch.long, device=seq_len.device)
            positions = positions.unsqueeze(0).expand_as(user_hist)
            position_embs = self.position_emb(positions)
            seq_embs = self.item_encoder(user_hist)

            mask4padding = user_hist == 0  # BxL
        else:
            seq_embs = batch['seq_emb']
            positions = torch.arange(seq_embs.size(1), dtype=torch.long, device=seq_len.device)
            positions = positions.unsqueeze(0)
            position_embs = self.position_emb(positions)
            mask4padding = None

        L = seq_embs.size(1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=seq_len.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=seq_len.device)
        try:
            transformer_input = batch['input_weight'] * (seq_embs + position_embs)
        except:
            transformer_input = seq_embs + position_embs
        transformer_out = self.transformer_layer(
            src=self.dropout(transformer_input),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD
        if not need_pooling:
            return transformer_out
        else:
            if self.training:
                return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                return self.eval_pooling_layer(transformer_out, batch['seqlen'])

class SASRec(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.query_encoder = SASRecQueryEncoder(
            self.fiid,
            self.embed_dim,
            self.max_seq_len,
            config['model']['head_num'],
            config['model']['hidden_size'],
            config['model']['dropout_rate'],
            config['model']['activation'],
            config['model']['layer_norm_eps'],
            config.get('num_layer'),
            self.item_embedding,
        )

    def current_epoch_trainloaders(self, nepoch):
        return super().current_epoch_trainloaders(nepoch)

    def forward(self, batch, need_pooling=True):
        return self.query_encoder(batch, need_pooling)

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def training_step(self, batch, nepoch, reduce=True, return_query=False, align=False):
        if align:
            query = self.query_encoder(batch, need_pooling=True)
            alignment = self.alignment(query, self.item_embedding.weight[batch[self.fiid]])
            uniformity = 1 * (
                self.uniformity(query) +
                self.uniformity(self.item_embedding.weight[batch[self.fiid]])
            )
            loss_value = alignment + uniformity
            return loss_value
        else:
            return super().training_step(batch, nepoch, reduce, return_query)