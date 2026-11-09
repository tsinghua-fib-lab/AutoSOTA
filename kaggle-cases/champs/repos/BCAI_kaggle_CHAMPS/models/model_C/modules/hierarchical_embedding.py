## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

__author__ = "shaojieb"


class HierarchicalEmbedding(nn.Module):
    def __init__(self, n_hierarchy, n_tokens, d_embeds):
        super().__init__()
        self.n_hierarchy = n_hierarchy
        d_embeds = d_embeds if type(d_embeds) == list else [d_embeds]*n_hierarchy
        self.d_embeds = d_embeds
        assert len(n_tokens) == n_hierarchy, "n_tokens must agree with n_hierarchy"
        assert len(self.d_embeds) == n_hierarchy, "d_embeds must agree with n_hierarchy"
        
        self.embeds = nn.ModuleList([])
        for i in range(n_hierarchy):
            # Project embeddings to higher dimensions
            if d_embeds[i] != max(d_embeds):
                layer = nn.Sequential(nn.Embedding(n_tokens[i], d_embeds[i]), nn.Linear(d_embeds[i], max(d_embeds)))
            else:
                layer = nn.Embedding(n_tokens[i], d_embeds[i])
            self.embeds.append(layer)
        
    def forward(self, x):
        # x has dimension (bsz x seq_len x [# of hierarchies])
        if len(x.shape) == 2: 
            x = x[:,:,None]
        assert x.size(2) == self.n_hierarchy
        embed_res = 0
        for type_index_level in range(self.n_hierarchy):
            embed_res += self.embeds[type_index_level](x[:,:,type_index_level])   # Embed this subtype, and project back to full dimension
        return embed_res

    
if __name__ == "__main__":
    bsz = 5
    seq_len = 12
    n_hierarchy = 3
    x = torch.cat([torch.LongTensor(bsz, seq_len, 1).random_(1, 15),
                   torch.LongTensor(bsz, seq_len, 1).random_(1, 50),
                   torch.LongTensor(bsz, seq_len, 1).random_(1, 120)], dim=2)
    embedding = HierarchicalEmbedding(n_hierarchy, [300,200,100], [16, 51, 121])
    embed_x = embedding(x)
    print(embed_x.shape)    # Should be (bsz x seq_len x 300)
