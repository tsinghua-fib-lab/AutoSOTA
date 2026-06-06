import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.network.oanet import OANBlock
from core.network.cus_transformer import Transformer
from core.network.embeddings import FreqEmbedder, get_1d_sincos_pos_embed_from_grid, get_2d_embedding
from core.network.decoder import TransformerPointNet
class pt_transformer(nn.Module):
    # point trajectory transformer
    def __init__(self, in_out_channels=16, stride=2, extra_info=False, nhead=4, in_dim=12, time_att=True):
        super(pt_transformer, self).__init__()
        self.stride = stride
        self.in_out_channels = in_out_channels
        self.input_fc1 = nn.Conv2d(in_dim, 16, (1,1))
        self.fc2 = nn.Conv2d(16, in_out_channels, (1,1))
        self.transformer_model = Transformer(d_model=in_out_channels, nhead=nhead, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=64, dropout=0.1, activation='relu', time_att=time_att)
        # Initialize hook here after defining the transformer model
        self._register_hooks()
        
    def _register_hooks(self):
        # Define the hook function
        def forward_hook(module, input, output):
            # 检查输出是否为 tuple
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if out is not None:  # 检查 out 是否为 None
                        if torch.isnan(out).any() or torch.isinf(out).any():
                            print(f"NaN or Inf detected in module {module} at tuple index {i}")
            else:
                if output is not None:  # 检查 output 是否为 None
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"NaN or Inf detected in module {module}")

        # Register hook for each module inside the transformer model
        for module in self.transformer_model.modules():
            module.register_forward_hook(forward_hook)

    def project(self, traj_pad):
        x = F.relu(self.input_fc1(traj_pad))
        x = F.relu(self.fc2(x))
        return x
    
    def extract_feature(self, traj_pad, pad_mask):
        # input traj data: [B, C, N, L], normalized (x,y) coord, pad_mask: [B, 1, N, L]
        # output features: [B, N, C]
        self.L = traj_pad.shape[-1]
        input_traj = traj_pad.permute(3,0,2,1).reshape(self.L, -1, self.in_out_channels) # [T, N, E]
        input_pad_mask = (pad_mask.reshape(-1, self.L) > 0.5) # [N, T]
        # 检查 input_traj 和 input_pad_mask 是否有 NaN 或 Inf
        if torch.isnan(input_traj).any() or torch.isinf(input_traj).any():
            print("NaN or Inf detected in input_traj")
        if torch.isnan(input_pad_mask).any() or torch.isinf(input_pad_mask).any():
            print("NaN or Inf detected in input_pad_mask")

        # 检查每一行是否全部为 True
        rows_all_true = torch.all(input_pad_mask, dim=1)  # 输出形状为 [256] 的布尔张量
        # 找到哪些行是全 True 的
        all_true_indices = torch.where(rows_all_true)[0]  # 获取所有行全为 True 的行索引
        if len(all_true_indices) > 0:
            for idx in all_true_indices:
                input_pad_mask[idx, 0] = False  # 将该行的第一个元素设置为 False
            print(f"以下行在 input_pad_mask 中全部为 True: {all_true_indices.tolist()}")

        output_feat = self.transformer_model(input_traj, input_traj, \
                src_key_padding_mask=input_pad_mask, tgt_key_padding_mask=input_pad_mask) # [T, N, E]
        if torch.isnan(output_feat).any() or torch.isinf(output_feat).any():
            print("NaN or Inf detected in output_feat")
        output = output_feat.reshape(self.L, traj_pad.shape[0], traj_pad.shape[2], -1) # [L, B, N, E]
        global_output = torch.max(output, dim=0, keepdim=False)[0] # [B, N, E]
        return global_output

    def forward(self, traj_pad, pad_mask):
        # image: [B, 3, H, W, L], depth: [B, 1, H, W, L]
        # input traj data: [B, 2, N, L], normalized (x,y) coord, pad_mask: [B, 1, N, L]
        # Return: geo_feature: [B, C, N]
        traj_pad_proj = self.project(traj_pad)
        traj_feature = self.extract_feature(traj_pad_proj, pad_mask)
        return traj_feature.permute(0,2,1)

class traj_oa_depth(nn.Module):
    # Trajectory classification model with transformer encoder and OANet decoder
    def __init__(self, extra_info=True, oanet=True, enable_pos_embed=True, load_dino=True, target_feature_dim=None, dino_later=True, dino_woatt=True,time_att=True,tracks=True,depth=True):
        super(traj_oa_depth, self).__init__()
        self.extra_info = extra_info
        self.oanet = oanet
        self.enable_pos_embed = enable_pos_embed
        self.in_dim = 4
        self.load_dino = load_dino
        self.dino_later = dino_later
        self.dino_woatt = dino_woatt
        self.time_att = time_att
        self.load_tracks = tracks
        self.load_depths = depth
        if self.load_depths:
            self.in_dim += 2
        if not self.load_tracks:
            self.in_dim -= 4
            self.extra_info = False
            self.enable_pos_embed = False
        if self.load_dino and not dino_later:
            if target_feature_dim is None:
                self.in_dim += 768
            else:
                self.in_dim += target_feature_dim
        if self.extra_info:
            self.in_dim += 2
        
        out_dim = 16
        nhead = 4
        in_dim_decoder = out_dim
        if self.load_dino and dino_later and not dino_woatt:
            in_dim_decoder = out_dim + 768
        
        if self.oanet:
            self.decoder = OANBlock(net_channels=128, input_channel=in_dim_decoder, depth=8, clusters=100)
        else:
            enable_dino = self.load_dino and self.dino_later and self.dino_woatt
            self.decoder = TransformerPointNet(out_dim=in_dim_decoder, enable_dino=enable_dino)
            
        if self.enable_pos_embed:
            self.pe_model = FreqEmbedder(2, 10, include_input=True)
            self.in_dim = self.in_dim - 4 + self.pe_model.out_dim * 2

        self.joint_encoder = pt_transformer(in_out_channels=out_dim, extra_info=extra_info, nhead=nhead, in_dim=self.in_dim, time_att=time_att)
    
    def gather_point(self, traj, points):
        b, c, h, w, l = points.shape
        # traj: [B, 2, N, L], points: [B, 3, H, W, L]
        points_src = points.permute(0,4,1,2,3).reshape(b*l,c,h*w)
        traj = traj.permute(0,3,1,2).reshape(b*l, 2, -1)
        traj_1dim = (traj[:,1,:] * h).to(torch.int) * w + (traj[:,0,:] * w).to(torch.int)
        traj_1dim = traj_1dim.unsqueeze(1).repeat(1,c,1).to(torch.int64).clamp(0, h*w-1)
        traj_points = torch.gather(points_src, dim=-1, index=traj_1dim) # [b*l, 3, N]
        traj_3d = traj_points.reshape(b,l,c,-1).permute(0,2,3,1)
        return traj_3d

    def augment_traj(self, depth, traj, mask, visible, confi, dinos):
        # point_3d: [B, 3, H, W, L]
        # return: traj_aug: [B, 10, N, L]
        L = traj.shape[-1]
        # disparity only
        if self.load_depths:
            traj_3d = self.gather_point(traj, depth)
            motion_3d = torch.zeros_like(traj_3d).to(traj.device)
            motion_3d[:,:,:,:-1] = (traj_3d[:,:,:,1:] - traj_3d[:,:,:,:-1]) * (1.0 - mask[:,:,:,1:])
        motion_2d = torch.zeros_like(traj).to(traj.device)
        motion_2d[:,:,:,:-1] = (traj[:,:,:,1:] - traj[:,:,:,:-1]) * (1.0 - mask[:,:,:,1:])
        if self.enable_pos_embed:
            # NeRF-like position embeddings, return [N,L,E] 
            traj = self.pe_model(traj.squeeze(0).permute(1,2,0)).permute(2,0,1).unsqueeze(0)
            motion_2d = self.pe_model(motion_2d.squeeze(0).permute(1,2,0)).permute(2,0,1).unsqueeze(0)
        
        traj_aug = []
        if self.load_tracks:
            traj_aug += [traj, motion_2d]
        if self.load_depths:
            traj_aug += [traj_3d, motion_3d]
        if self.extra_info:
            traj_aug += [visible, confi]
        if self.load_dino and not self.dino_later:
            traj_aug += [dinos]
        traj_aug = torch.cat(traj_aug, dim=1)

        return traj_aug
    
    def forward(self, batch):
        # First extract the image features
        # img: [B, 3, H, W, L], traj: [B, 2, N, L], depth: [B, 1, H, W, L]
        # mask: [B, 1, N, L], visible: [B, 1, N, L], confidence: [B, 1, N, L]
        depths, trajs, masks = batch["depth"], batch["traj"], batch["mask"]
        visibles, confis, dinos = None, None, None
        if self.extra_info:
            visibles, confis = batch["visib_value"], batch["confi_value"]
        if self.load_dino:
            dinos = batch["dino"]
        aug_trajs = self.augment_traj(depths, trajs, masks, visibles, confis, dinos)
        mask = self.joint_encoder(aug_trajs, masks) # [B=1, C=16, N]
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("NaN or Inf detected in feat")

        if dinos is not None and self.dino_later:
            dinos = dinos.squeeze(-1)  # [B, 768, N]
            mask = torch.cat([mask, dinos], dim=1)  # 拼接后形状为 [B, 16 + 768, N]

        mask = self.decoder(mask.unsqueeze(-1))
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("NaN or Inf detected in mask")
        mask = torch.sigmoid(mask)
        return mask