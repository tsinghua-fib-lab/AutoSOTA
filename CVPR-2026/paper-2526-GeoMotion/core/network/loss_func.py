# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import torch

def BCELoss(pred, gt, weight):
    return torch.nn.functional.binary_cross_entropy(pred, gt, weight)

def BCELoss_trivial(pred, gt):
    return torch.nn.functional.binary_cross_entropy(pred, gt)

def BCEWithLogitsLoss(pred, gt, weight):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, gt, weight)