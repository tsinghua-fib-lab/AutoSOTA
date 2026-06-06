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

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed.nn
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import numpy as np
from torch import Tensor
import models_dinov2


def interpolate_pos_embed_(
    weight: dict, key="pos_embed", crop_size=(224, 224), kernel_conv=14, index=0
):
    pos_cls, pos_tokens = weight[key][:, :1, :], weight["pos_embed"][:, 1+index:, :]
    embed_dim = pos_tokens.shape[-1]
    orig_size = int(pos_tokens.shape[-2] ** 0.5)
    orig_shape = (-1, orig_size, orig_size, embed_dim)
    crop_size = tuple(L // kernel_conv for L in crop_size)
    resized_pos_tokens = F.interpolate(
        pos_tokens.reshape(*orig_shape).permute(0, 3, 1, 2),
        size=crop_size,
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = resized_pos_tokens.shape
    resized_pos_tokens = resized_pos_tokens.permute(0, 2, 3, 1).reshape(
        -1, np.prod(crop_size), embed_dim
    )
    weight[key] = torch.cat((pos_cls, resized_pos_tokens), dim=1)
    print(
        f"Convert pos embedding: {pos_tokens.shape} -> {orig_shape} -> {dst_shape} -> {resized_pos_tokens.shape}"
    )


def interpolate_patch_embed_(weight, key="patch_embed.proj.weight", kernel_conv=16):
    assert key in weight, f"{key} must in {weight.keys()}"
    ori_shape = weight[key].shape
    if ori_shape[-1] != kernel_conv:
        weight[key] = F.interpolate(
            weight[key].float(),
            size=(kernel_conv, kernel_conv),
            mode="bicubic",
            align_corners=False,
        )


class QSD(nn.Module):
    def __init__(
        self,
        stu_dim: int,
        tea_dim: int,
        num_heads: int = 1
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.proj_q = nn.Sequential(
            nn.LayerNorm(stu_dim),
            nn.Linear(stu_dim, tea_dim))
        self.proj_v = nn.Sequential(
            nn.LayerNorm(stu_dim),
            nn.Linear(stu_dim, tea_dim))

    def forward(self, patch_stu: Tensor, patch_tea: Tensor) -> Tensor:
        B, N, C = patch_tea.shape
        q = self.proj_q(patch_stu).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = patch_tea.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.proj_v(patch_stu).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return x


class MetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()

        import_student = getattr(models_dinov2, cfg.target_model)
        student = import_student(img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            init_values=None,
            ffn_layer='mlp',
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1)

        embed_dim = student.embed_dim

        if cfg.domain_distillation: 
            import_teacher = getattr(models_dinov2, cfg.teacher_model)
            teacher = import_teacher(img_size=cfg.input_size,
                patch_size=cfg.patch_size,
                init_values=1.0e-05,
                ffn_layer='mlp',
                block_chunks=0,
                num_register_tokens=0,
                interpolate_antialias=False,
                interpolate_offset=0.1)
            checkpoints = torch.load(cfg.teacher_path, map_location='cpu')
            interpolate_patch_embed_(checkpoints, kernel_conv=cfg.patch_size)
            interpolate_pos_embed_(checkpoints, crop_size=(cfg.input_size, cfg.input_size), kernel_conv=cfg.patch_size)
            missing_keys, unexpected_keys = teacher.load_state_dict(checkpoints, False)
            print('teacher missing_keys: ', missing_keys)
            print('teacher unexpected_keys: ', unexpected_keys)
        else:
            if cfg.teacher_model == 'vit_base':
                teacher = torch.hub.load(cfg.teacher_path, 'dinov2_vitb14_lc', source='local').backbone
            elif cfg.teacher_model == 'vit_small':
                teacher = torch.hub.load(cfg.teacher_path, 'dinov2_vits14_lc', source='local').backbone
            elif cfg.teacher_model == 'vit_large':
                teacher = torch.hub.load(cfg.teacher_path, 'dinov2_vitl14_lc', source='local').backbone
            elif cfg.teacher_model == 'vit_giant':
                teacher = torch.hub.load(cfg.teacher_path, 'dinov2_vitg14_lc', source='local').backbone

        teacher.eval()
        student_model_dict['backbone'] = student
        teacher_model_dict['backbone'] = teacher
        
        self.embed_dim = embed_dim

        # initialize parameters and checks
        self.total_n_global_crops = cfg.batch_size

        self.student = nn.ModuleDict(student_model_dict)
        # inherit
        state_dict = torch.load(cfg.student_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        if cfg.domain_distillation:
            checkpoints = OrderedDict()
            for k,v in state_dict.items():
                if 'student' in k:
                    checkpoints[k.replace("student.backbone.", "")] = v

            distillation_token = 0
        else:
             checkpoints = state_dict
             distillation_token = 1
        interpolate_patch_embed_(checkpoints, kernel_conv=cfg.patch_size)
        interpolate_pos_embed_(checkpoints, crop_size=(cfg.input_size, cfg.input_size), kernel_conv=cfg.patch_size, index=distillation_token)
        missing_keys, unexpected_keys = self.student.backbone.load_state_dict(checkpoints, False)
        print('student missing_keys: ', missing_keys)
        print('student unexpected_keys: ', unexpected_keys)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        teacher_embed_dim = teacher.embed_dim

        self.patch_head = QSD(embed_dim, teacher_embed_dim)
        self.fea_head = QSD(embed_dim, teacher_embed_dim)
        self.token_head = QSD(embed_dim, teacher_embed_dim)

        self.soft_criterion = torch.nn.MSELoss()

        for param in self.teacher.backbone.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        global_crops = inputs["collated_global_crops"]
        
        masks = inputs["collated_masks"]
        mask_indices_list = inputs["mask_indices_list"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = inputs["upperbound"]

        # compute teacher output
        # @torch.no_grad()
        def compute_teacher_output():
            with torch.no_grad():
                teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            return teacher_cls_tokens, teacher_patch_tokens

        # get the teacher outputs
        (
            teacher_cls_tokens,
            teacher_patch_tokens
        ) = compute_teacher_output()
        
        cur_masks = masks if self.cfg.mask_probability > 0 else None

        student_backbone_output_dict, student_backbone_output_dict_unmask = self.student.backbone(
            [global_crops, global_crops], masks=[cur_masks, None], is_training=True
        )

        student_cls_token_unmask = student_backbone_output_dict_unmask["x_norm_clstoken"]
        student_patch_tokens_unmask = student_backbone_output_dict_unmask["x_norm_patchtokens"]
        student_patch_tokens = student_backbone_output_dict["x_norm_patchtokens"]

        ## projection head
        student_patch_tokens_unmask = self.fea_head(student_patch_tokens_unmask, teacher_patch_tokens)
        
        student_cls_token_unmask = self.token_head(student_cls_token_unmask.unsqueeze(1), teacher_cls_tokens.unsqueeze(1))
        
        student_patch_tokens_pro = self.patch_head(student_patch_tokens, teacher_patch_tokens)

        ## token objective
        distillation_loss_token = self.soft_criterion(student_cls_token_unmask, teacher_cls_tokens.unsqueeze(1))

        ## fea objective
        student_whole_fea = torch.cat((student_cls_token_unmask,student_patch_tokens_unmask),dim=1)
        teacher_whole_fea = torch.cat((teacher_cls_tokens.unsqueeze(1),teacher_patch_tokens),dim=1)
        distillation_loss_fea = self.soft_criterion(student_whole_fea, teacher_whole_fea)

        ## patch objective
        patch_loss = self.soft_criterion(student_patch_tokens_pro, teacher_patch_tokens)
        
        # coefficient
        token_loss = self.cfg.lambda_token * distillation_loss_token
        fea_loss = self.cfg.lambda_fea * distillation_loss_fea
        patch_loss = self.cfg.lambda_patch * patch_loss

        # compute the total loss
        total_loss = token_loss + fea_loss + patch_loss

        # return the final loss dict
        loss_dict = {"patch_loss": patch_loss, "fea_loss": fea_loss, "token_loss": token_loss, "loss": total_loss}
        
        return loss_dict