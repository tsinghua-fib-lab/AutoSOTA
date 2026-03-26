# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import generate_binomial_mask, generate_continuous_mask

class SimConv4(torch.nn.Module):
    def __init__(self, input_dims, output_dims,hidden_dims=64, mask_mode='binomial'):
        super(SimConv4, self).__init__()
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_size = output_dims
        self.name = "conv4"
        self.mask_mode = mask_mode

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(hidden_dims, hidden_dims, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(hidden_dims),
          torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(hidden_dims, hidden_dims, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(hidden_dims),
          torch.nn.ReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(hidden_dims, hidden_dims, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(hidden_dims),
          torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(hidden_dims, output_dims, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(output_dims),
          torch.nn.ReLU(),
          torch.nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,mask=None):
        # x_ = x.view(x.shape[0], 1, -1) #(B, T, Ch)
        ## B x Ch x T

        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0


        x_t = torch.permute(x,[0,2,1])
        h = self.layer1(x_t)  # (B, T, H)
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        h = self.flatten(h)
        h = F.normalize(h, dim=1)
        h = torch.unsqueeze(h,1)
        return h
