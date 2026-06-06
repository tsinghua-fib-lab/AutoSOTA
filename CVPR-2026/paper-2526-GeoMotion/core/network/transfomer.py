import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from core.network.embeddings import FreqEmbedder, get_1d_sincos_pos_embed_from_grid, get_2d_embedding
from core.network.blocks import (
    Mlp,
    BasicEncoder,
    AttnBlock,
    CorrBlock,
    Attention,
)
from core.network.oanet import OANBlock

class pt_att_transformer(nn.Module):
    # point trajectory transformer
    def __init__(self, window_size, in_out_channels=16,in_dim=12, efficient=True):
        super(pt_att_transformer, self).__init__()
        self.L = window_size
        self.in_out_channels = in_out_channels
        self.input_fc1 = nn.Conv2d(in_dim, 16, (1,1))
        self.fc2 = nn.Conv2d(16, in_out_channels, (1,1))
        if efficient:
            self.transformer_model = EfficientUpdateFormer(
                input_dim=in_out_channels,
                output_dim= in_out_channels,
                add_space_attn=True,
                )
        else:
            self.transformer_model = UpdateFormer(
                input_dim=in_out_channels,
                output_dim=in_out_channels,
                add_space_attn=True,
            )
        # Initialize hook here after defining the transformer model
        # self._register_hooks()
        
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
        if torch.isnan(input_traj).any() or torch.isinf(input_traj).any():
            print("NaN or Inf detected in input_traj")
        if torch.isnan(input_pad_mask).any() or torch.isinf(input_pad_mask).any():
            print("NaN or Inf detected in input_pad_mask")
        # [T, N, E], [N,T] -> [B,N,T,E], [B*T, N] -> [B, N, T, E] -> [T, B, N, E]
        output_feat = self.transformer_model(input_traj.permute(1,0,2).unsqueeze(0), input_pad_mask.permute(1,0))
        if torch.isnan(output_feat).any() or torch.isinf(output_feat).any():
            print("NaN or Inf detected in output_feat")
        output = output_feat.permute(2, 0, 1, 3) # [L, B, N, E]
        global_output = torch.max(output, dim=0, keepdim=False)[0] # [B, N, E]
        return global_output

    def forward(self, traj_pad, pad_mask):
        # image: [B, 3, H, W, L], depth: [B, 1, H, W, L]
        # input traj data: [B, 2, N, L], normalized (x,y) coord, pad_mask: [B, 1, N, L]
        # Return: geo_feature: [B, C, N]
        traj_pad_proj = self.project(traj_pad)
        traj_feature = self.extract_feature(traj_pad_proj, pad_mask)
        return traj_feature.permute(0,2,1)

class traj_seg(nn.Module):
    # Trajectory classification model with transformer encoder
    def __init__(self, window_size, input_hw, extra_info=False, out_dim=1, enable_time_emb=False, enable_pos_embed=False,efficient=True):
        super(traj_seg, self).__init__()
        self.window_size = window_size
        self.input_hw = input_hw
        self.image_grid()
        self.extra_info = extra_info
        self.enable_time_emb = enable_time_emb
        self.enable_pos_embed = enable_pos_embed
        self.in_dim = 10
        if extra_info:
            self.in_dim += 2
        if enable_pos_embed:
            self.pe_model = FreqEmbedder(2, 10, include_input=True)
            self.in_dim = self.in_dim - 4 + self.pe_model.out_dim * 2
            
        if enable_time_emb:
            time_grid = torch.linspace(0, window_size - 1, window_size).reshape(1, window_size, 1)
            self.register_buffer(
                "time_emb", get_1d_sincos_pos_embed_from_grid(self.in_dim, time_grid[0])
            )
        self.joint_encoder = pt_att_transformer(window_size, in_out_channels=out_dim, in_dim=self.in_dim, efficient=efficient)
        self.decoder = OANBlock(net_channels=128, input_channel=self.joint_encoder.in_out_channels, depth=8, clusters=100)
    
    def image_grid(self):
        h, w = self.input_hw[0], self.input_hw[1]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h,w))
        self.xy = np.stack([xx,yy,ones], axis=-1)
        self.xy_t = torch.from_numpy(self.xy).reshape(-1,3).permute(1,0).unsqueeze(0).float().cuda()
        fx, fy = (h + w) / 2.0, (h + w) / 2.0
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
        self.K_inv = np.linalg.inv(K)
        self.K_inv_t = torch.from_numpy(self.K_inv).unsqueeze(0).float().cuda()
    
    def depth_project(self, depth):
        # depth: [B, 1, H, W, L]
        b, _, h, w, l = depth.shape
        depth_b = depth.permute(0,4,1,2,3).reshape(b*l, 1, h*w)
        point_3d = depth_b * self.K_inv_t.bmm(self.xy_t)
        point_3d_img = point_3d.reshape(b,l,3,h,w).permute(0,2,3,4,1)
        return point_3d_img
    
    def gather_point(self, traj, points):
        b, _, h, w, l = points.shape
        # traj: [B, 2, N, L], points: [B, 3, H, W, L]
        points_src = points.permute(0,4,1,2,3).reshape(b*l,3,h*w)
        traj = traj.permute(0,3,1,2).reshape(b*l, 2, -1)
        traj_1dim = (traj[:,1,:] * h).to(torch.int) * w + (traj[:,0,:] * w).to(torch.int)
        traj_1dim = traj_1dim.unsqueeze(1).repeat(1,3,1).to(torch.int64).clamp(0, h*w-1)
        traj_points = torch.gather(points_src, dim=-1, index=traj_1dim) # [b*l, 3, N]
        traj_3d = traj_points.reshape(b,l,3,-1).permute(0,2,3,1)
        return traj_3d

    def augment_traj(self, depth, traj, mask, visible, confi):
        # point_3d: [B, 3, H, W, L]
        # return: traj_aug: [B, 10, N, L]
        L = traj.shape[-1]
        point_3d = self.depth_project(depth)
        traj_3d = self.gather_point(traj, point_3d)
        motion_2d = torch.zeros_like(traj).to(traj.device)
        motion_2d[:,:,:,:-1] = (traj[:,:,:,1:] - traj[:,:,:,:-1]) * (1.0 - mask[:,:,:,1:])
        motion_3d = torch.zeros_like(traj_3d).to(traj.device)
        motion_3d[:,:,:,:-1] = (traj_3d[:,:,:,1:] - traj_3d[:,:,:,:-1]) * (1.0 - mask[:,:,:,1:])
        
        if self.enable_pos_embed:
            # NeRF-like position embeddings, return [N,L,E] 
            traj = self.pe_model(traj.squeeze(0).permute(1,2,0)).permute(2,0,1).unsqueeze(0)
            motion_2d = self.pe_model(motion_2d.squeeze(0).permute(1,2,0)).permute(2,0,1).unsqueeze(0)
            # traj should be [B * N, L, 2] -> [1,N,2,L] -> [N,L,E]
            # traj = traj.squeeze(0).permute(1,2,0)
            # traj = get_2d_embedding(traj, self.in_dim)
            # motion_2d = motion_2d.squeeze(0).permute(1,2,0)
            # motion_2d = get_2d_embedding(motion_2d, 64, cat_coords=True)
        if self.extra_info:
            traj_aug = torch.cat([traj, motion_2d, traj_3d, motion_3d, visible, confi], dim=1)
        else:
            traj_aug = torch.cat([traj, motion_2d, traj_3d, motion_3d], dim=1)
        if self.enable_time_emb:
            traj_aug = traj_aug + self.time_emb.permute(0,2,1).unsqueeze(2).expand_as(traj_aug)
        return traj_aug
    
    def forward(self, batch):
        # First extract the image features
        # img: [B, 3, H, W, L], traj: [B, 2, N, L], depth: [B, 1, H, W, L]
        # mask: [B, 1, N, L], visible: [B, 1, N, L], confidence: [B, 1, N, L]
        depths, trajs, masks = batch["depth"], batch["traj"], batch["mask"]
        visibles, confis = None, None
        if self.extra_info:
            visibles, confis = batch["visib_value"], batch["confi_value"]
        aug_trajs = self.augment_traj(depths, trajs, masks, visibles, confis)
        feat = self.joint_encoder(aug_trajs, masks)
        mask = self.decoder(feat.unsqueeze(-1))
        mask = torch.sigmoid(mask)
        return mask

class UpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=12,
        time_depth=12,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        x = self.input_transform(input_tensor)

        j = 0
        for i in range(len(self.time_blocks)):
            B, N, T, _ = x.shape
            x_time = rearrange(x, "b n t c -> (b n) t c", b=B, t=T, n=N)
            x_time = self.time_blocks[i](x_time)

            x = rearrange(x_time, "(b n) t c -> b n t c ", b=B, t=T, n=N)
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_blocks)) == 0
            ):
                x_space = rearrange(x, "b n t c -> (b t) n c ", b=B, t=T, n=N)
                x_space = self.space_blocks[j](x_space)
                x = rearrange(x_space, "(b t) n c -> b n t c  ", b=B, t=T, n=N)
                j += 1

        flow = self.flow_head(x)
        return flow

class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        tokens = self.input_transform(input_tensor)
        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)
        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )
                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]
        flow = self.flow_head(tokens)
        return flow

class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x