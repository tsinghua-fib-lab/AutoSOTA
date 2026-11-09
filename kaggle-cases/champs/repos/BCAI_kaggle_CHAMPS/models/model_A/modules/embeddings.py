## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.hierarchical_embedding import HierarchicalEmbedding

__author__ = "shaojieb"


class TokenFeatureEmbedding(nn.Module):
    def __init__(self, n_hierarchy, n_tokens, d_embeds):
        super().__init__()
        self.d_embeds = d_embeds if type(d_embeds) == list else [d_embeds]*n_hierarchy
        assert len(n_tokens) == n_hierarchy, "n_tokens must agree with n_hierarchy"
        assert len(self.d_embeds) == n_hierarchy, "dims must agree with n_hierarchy"
        self.embedding = HierarchicalEmbedding(n_hierarchy, n_tokens, d_embeds)
    
    def forward(self, tokens, features):
        raise NotImplementedError

    
class LearnableEmbedding(TokenFeatureEmbedding):
    def __init__(self, n_hierarchy, ntokens, d_embeds, d_feature, n_feature):
        super(LearnableEmbedding, self).__init__(n_hierarchy, ntokens, d_embeds)
        self.n_feature = n_feature
        self.d_feature = d_feature
        if n_feature > 0:
            self.proj = nn.Linear(n_feature, d_feature)
    
    def forward(self, tokens, features):
        # tokens is of dimension (bsz x seq_len x [# of hierarchies])
        # features is of dimension (bsz x seq_len x n_feature)
        n_feature = self.n_feature
        if len(features.shape) <= 2 or n_feature > 0:
            features = features[:,:,None]
        assert (n_feature == 0) or (features.size(2) == self.n_feature), "Number of features do not match"
        token_embedding = self.embedding(tokens)
        if n_feature > 0:
            encoding = self.proj(features) if self.d_feature > self.n_feature else features
            return torch.cat([token_embedding, encoding], dim=2) 
        return token_embedding


class SineEmbedding(TokenFeatureEmbedding):
    def __init__(self, n_hierarchy, ntokens, d_embeds, n_feature=1):
        super(SineEmbedding, self).__init__(n_hierarchy, ntokens, d_embeds)
        self.n_feature = n_feature
        self.projs = nn.ModuleList([nn.Linear(max(self.d_embeds), max(self.d_embeds)) for _ in range(n_feature)])
    
    def forward(self, tokens, features):
        # tokens is of dimension (bsz x seq_len x [# of hierarchies])
        # features is of dimension (bsz x seq_len x n_feature)
        n_feature = self.n_feature
        if len(features.shape) <= 2 and n_feature > 0:
            features = features[:,:,None]
        assert (n_feature == 0) or (features.size(2) == self.n_feature), "Number of features do not match"
        dev = tokens.device
        d_embed = max(self.d_embeds)
        token_embedding = self.embedding(tokens)
        inv_freq = 1 / (1000 ** (torch.arange(0.0, d_embed, 2.0) / d_embed))[None,None,:].to(dev).type(token_embedding.dtype)
        for i in range(self.n_feature):
            encoding = torch.zeros(features.shape[0], features.shape[1], d_embed).to(dev).type(token_embedding.dtype)
            encoding[:,:,0::2] = torch.cos(features[:,:,i].unsqueeze(2)*inv_freq)
            encoding[:,:,1::2] = torch.sin(features[:,:,i].unsqueeze(2)*inv_freq)
            token_embedding += self.projs[i](encoding).type(token_embedding.dtype)
        return token_embedding               
