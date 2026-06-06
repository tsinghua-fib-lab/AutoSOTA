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
import models_eva02


def interpolate_pos_embed(checkpoint_model, patch_size=16, new_size=32):
    patch_embed = checkpoint_model["patch_embed.proj.weight"]
    C_o, C_in, H, W = patch_embed.shape
    patch_embed = torch.nn.functional.interpolate(
        patch_embed.float(), size=(patch_size, patch_size), mode="bicubic", align_corners=False
    )
    checkpoint_model["patch_embed.proj.weight"] = patch_embed
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens.float(),
            size=(new_size, new_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed
    if "positional_embedding" in checkpoint_model:
        positional_embedding_checkpoint = checkpoint_model["positional_embedding"]
        embedding_size = positional_embedding_checkpoint.shape[-1]
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int(
            (positional_embedding_checkpoint.shape[-2] - num_extra_tokens) ** 0.5
        )
        # height (== width) for the new position embedding
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
        extra_tokens = positional_embedding_checkpoint[:num_extra_tokens, :]
        # only the position tokens are interpolated
        pos_tokens = positional_embedding_checkpoint[num_extra_tokens:, :]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens.float(),
            size=(new_size, new_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
        new_positional_embedding = torch.cat((extra_tokens, pos_tokens), dim=0)
        checkpoint_model["positional_embedding"] = new_positional_embedding


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

        import_student = getattr(models_eva02, cfg.target_model)
        student = import_student(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size)
        
        embed_dim = student.embed_dim
        
        import_teacher = getattr(models_eva02, cfg.teacher_model)
        teacher_backbone = import_teacher(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size)
        
        checkpoint = torch.load(cfg.teacher_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'module' in checkpoint:
            pretrained_dict = checkpoint['module']
        else:
            pretrained_dict = checkpoint
        
        for k in list(pretrained_dict.keys()):
            if 'rope' in k:
                pretrained_dict.pop(k)

        interpolate_pos_embed(pretrained_dict, patch_size=cfg.patch_size, new_size= cfg.input_size // cfg.patch_size)
        missing_keys, unexpected_keys = teacher_backbone.load_state_dict(pretrained_dict, False)
        print('teacher missing_keys: ', missing_keys)
        print('teacher unexpected_keys: ', unexpected_keys)
        teacher_backbone.eval()

        student_model_dict['backbone'] = student
        teacher_model_dict['backbone'] = teacher_backbone
        
        self.embed_dim = embed_dim

        # initialize parameters and checks
        self.total_n_global_crops = cfg.batch_size

        self.student = nn.ModuleDict(student_model_dict)
        # inherit

        checkpoint = torch.load(cfg.student_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'module' in checkpoint:
            pretrained_dict = checkpoint['module']
        else:
            pretrained_dict = checkpoint
        

        if cfg.domain_distillation: 
            checkpoints = OrderedDict()
            for k, v in pretrained_dict.items():
                if 'student' in k and 'rope' not in k:
                    checkpoints[k.replace("student.backbone.", "")] = v
        else:
            for k in list(pretrained_dict.keys()):
                if 'rope' in k:
                    pretrained_dict.pop(k)
            checkpoints = pretrained_dict

        interpolate_pos_embed(checkpoints, patch_size=cfg.patch_size, new_size= cfg.input_size // cfg.patch_size)
        missing_keys, unexpected_keys = self.student.backbone.load_state_dict(checkpoints, False)
        print('student missing_keys: ', missing_keys)
        print('student unexpected_keys: ', unexpected_keys)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        teacher_embed_dim = teacher_backbone.embed_dim
        
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
                teacher_backbone_output_dict = self.teacher.backbone(global_crops)
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
            [global_crops, global_crops], masks=[cur_masks, None])

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
        total_loss = patch_loss + fea_loss + token_loss

        # return the final loss dict
        loss_dict = {"patch_loss": patch_loss, "fea_loss": fea_loss, "token_loss": token_loss, "loss": total_loss}
        
        return loss_dict