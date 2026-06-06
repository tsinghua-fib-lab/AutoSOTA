import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 bias=True):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1,
            bias=bias)
        self.w_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(1, kernel_size), 
            stride=1, 
            padding=(0, kernel_size//2),
            groups=mid_channels)
        
        self.h_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(kernel_size, 1), 
            stride=1, 
            padding=(kernel_size//2, 0),
            groups=mid_channels
        )
        
        self.out_proj = nn.Conv2d(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x



class HoGEdgeGateConv(nn.Module):
    def __init__(self,
                 in_dim,
                 nbins,
                 cell_size = (8, 8)):
        super().__init__()

        self.nbins = nbins
        self.cell_size = cell_size

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.ReLU(inplace=True),  
            nn.AdaptiveAvgPool2d((1, 1))   
        )


        self.weight = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim//2, out_channels=in_dim),
            nn.GroupNorm(in_dim//8, in_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim//8, in_dim)
        )

        self.fuse_block = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim//2, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(in_dim//8, in_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        x = image2patches(x)

        x_hog = self.get_hog_feature(x)
        x_hog = self.hog_feat(x_hog)

        x1 = self.sigmoid(self.weight(x + x_hog))
        x2 = self.conv(x)
        x = x1 * x2

        x = patches2image(x)

        x = x + residual
        x = self.fuse_block(x)

        return x



    def get_hog_feature(self, x):
        
        x_mean = x.mean(dim=1, keepdim=True)
        B, _, H, W = x_mean.shape
        device = x_mean.device

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32).view(1, 1, 3, 3).to(device)
        dx = F.conv2d(x_mean.float(), sobel_x, padding=1)  # b, 1, h, w
        dy = F.conv2d(x_mean.float(), sobel_y, padding=1)
        
        # direction
        gradient_dir = torch.atan2(dy, dx)       # [-π，π] 
        gradient_dir = torch.abs(gradient_dir)   # [0，π] 
        
        # cells - use adaptive cell size for small feature maps
        cell_h, cell_w = self.cell_size
        # Ensure at least 1 cell in each dimension
        if H < cell_h:
            cell_h = max(1, H)
        if W < cell_w:
            cell_w = max(1, W)
        H_cells = int(H / cell_h)
        W_cells = int(W / cell_w)

        # 
        dirs_crop = gradient_dir[:, :, :H_cells*cell_h, :W_cells*cell_w]

        # [B, H_cells, W_cells, cell_h*cell_w]
        dirs = dirs_crop.reshape(B, H_cells, W_cells, -1)
        
        bin_with = torch.pi / self.nbins
        bin_indices = (dirs / bin_with).floor().long() 
        bin_indices = torch.clamp(bin_indices, 0, self.nbins-1) 

        bin_indices_flat = bin_indices.reshape(B*H_cells*W_cells, dirs.shape[-1]) 
        weight = []
        for i in range(bin_indices_flat.shape[0]):
            bins = bin_indices_flat[i]
            count = torch.bincount(bins, minlength=self.nbins)
            weight.append(count)

        weight = torch.stack(weight, dim=0).reshape(B, H_cells , W_cells, -1) / 64 # B, H_cells , W_cells, self.bins 

        start = torch.pi / (2*self.nbins)
        hog_feature = torch.linspace(start, torch.pi - start, self.nbins).to(device).repeat(B, H_cells, W_cells,1) * weight

        return hog_feature.permute(0, 3, 1, 2)
    




def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = einops.rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = einops.rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x
