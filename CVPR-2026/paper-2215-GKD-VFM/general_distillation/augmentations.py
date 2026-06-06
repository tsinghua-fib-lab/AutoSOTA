# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
from torchvision import transforms
import torch

logger = logging.getLogger("dinov2")



def collate_data_and_cast_aug(
    samples_list,
    mask_ratio,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_first_n=False,
    mask_generator=None,
    clone_batch=1,
    patch_size = 14,
    img_size = 224
):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = 1

    assert n_global_crops > 0, "global crops number should be > 0"
    collated_global_crops = torch.stack([s[i] for i in range(n_global_crops) for s in samples_list])

    labels = [s[1] for s in samples_list]
    labels = torch.LongTensor(labels)
    collated_global_labels = labels.repeat(n_global_crops)

    B = len(collated_global_crops)
    N = n_tokens
    h_patches = w_patches = img_size // patch_size
    n_samples_masked = int(B * mask_probability)

    masks_list = []
    upperbound = 0

    masks_enc = torch.full((1,), 0, dtype=torch.int32)
    masks_pred = torch.full((1,), 0, dtype=torch.int32)
    # specify the number of masks to append
    number_masks = n_samples_masked * clone_batch
    # do per-sample masking
    if isinstance(mask_ratio, (tuple, list)) and len(mask_ratio) == 2:
        probs = torch.linspace(*mask_ratio, number_masks + 1)
        for i in range(0, number_masks):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
    else:
        mask_ratio = mask_ratio[0]
        # apply the same mask ratio to all images
        for i in range(0, number_masks):
            masks_list.append(torch.BoolTensor(mask_generator(int(N * mask_ratio))))
            upperbound += int(N * mask_ratio)

    # append masks for unmasked samples
    for i in range(n_samples_masked, B):
        # masks_list.append(torch.BoolTensor(mask_generator(0)))
        masks_list.append(torch.BoolTensor(mask_generator.get_none_mask()))

    if not mask_first_n and mask_probability > 0.0:  # shuffle masking -- not shuffling for mae-style
        random.shuffle(masks_list)

    collated_img_masks = torch.stack(masks_list)
    collated_masks = collated_img_masks.flatten(1)
    # collated_img_masks = collated_img_masks.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
    # rgb_mean = collated_global_crops.mean(dim=[2,3], keepdim=True)
    # collated_global_masked_crops = torch.where(collated_img_masks.unsqueeze(1).repeat_interleave(4, dim=2).repeat_interleave(4, dim=3), rgb_mean, collated_global_crops)

    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        # "collated_global_masked_crops": collated_global_masked_crops.to(dtype),
        "collated_global_labels": collated_global_labels,
        "collated_img_masks": collated_img_masks,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "masks_enc": masks_enc,
        "masks_pred": masks_pred,
    }