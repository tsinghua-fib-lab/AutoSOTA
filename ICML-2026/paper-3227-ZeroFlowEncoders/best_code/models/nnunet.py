import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Improved Building Blocks
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # FIXED: Logic to handle channel sizes after concatenation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # If bilinear, we do NOT reduce channels during upsampling.
            # Concatenation Input = (in_channels) + (in_channels // 2 from skip)
            conv_in_channels = in_channels + (in_channels // 2)
            self.conv = DoubleConv(conv_in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # If ConvTranspose, we reduce channels by half during upsampling.
            # Concatenation Input = (in_channels // 2) + (in_channels // 2 from skip) = in_channels
            conv_in_channels = in_channels
            self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding for odd shapes
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection (x2) with upsampled input (x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 2. Universal Dynamic Networks (Fixed)
# ==========================================

class DynamicEncoder(nn.Module):
    def __init__(self, latent_dim=64, img_size=64):
        super(DynamicEncoder, self).__init__()
        self.img_size = img_size
        
        # Calculate layers to reach 4x4 bottleneck
        self.num_layers = int(math.log2(img_size) - 2)
        if self.num_layers < 2:
            raise ValueError(f"Image size {img_size} is too small. Minimum 16x16 required.")

        layers = []
        in_channels = 3
        current_channels = 32 if img_size <= 32 else 64
        
        for i in range(self.num_layers):
            layers.append(nn.Conv2d(in_channels, current_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(current_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_channels = current_channels
            current_channels *= 2
            
        self.net = nn.Sequential(*layers)
        
        final_channels = current_channels // 2
        self.flat_size = final_channels * 4 * 4
        self.fc = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x):
        x = x.view(-1, 3, self.img_size, self.img_size)
        x = self.net(x)
        x = x.view(-1, self.flat_size)
        x = self.fc(x)
        return x

class DynamicDecoder(nn.Module):
    def __init__(self, latent_dim=64, img_size=64):
        super(DynamicDecoder, self).__init__()
        self.img_size = img_size
        
        self.num_layers = int(math.log2(img_size) - 2)
        self.start_dim = 4 
        
        base_channels = 32 if img_size <= 32 else 64
        self.start_channels = base_channels * (2**(self.num_layers - 1))
        
        self.fc_map = nn.Linear(latent_dim, self.start_channels * self.start_dim * self.start_dim)
        
        layers = []
        curr_channels = self.start_channels
        
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            out_channels = 3 if is_last else curr_channels // 2
            
            layers.append(nn.ConvTranspose2d(curr_channels, out_channels, kernel_size=4, stride=2, padding=1))
            
            if not is_last:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(True))
                curr_channels = out_channels
            else:
                layers.append(nn.Sigmoid())
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_map(x)
        x = x.view(-1, self.start_channels, self.start_dim, self.start_dim)
        img = self.net(x)
        return img

class DynamicUNetVectorField(nn.Module):
    def __init__(self, encx_dim, img_size=64, bilinear=True):
        super(DynamicUNetVectorField, self).__init__()
        self.img_size = img_size
        self.bilinear = bilinear
        
        self.decoder = DynamicDecoder(latent_dim=encx_dim, img_size=img_size)

        self.depth = int(math.log2(img_size) - 2)
        start_ch = 32 if img_size <= 32 else 64
        
        # --- Down Path ---
        self.inc = DoubleConv(10, start_ch)
        self.downs = nn.ModuleList()
        
        curr_ch = start_ch
        for i in range(self.depth):
            self.downs.append(Down(curr_ch, curr_ch * 2))
            curr_ch *= 2
            
        # --- Up Path ---
        self.ups = nn.ModuleList()
        # curr_ch is currently at bottleneck (e.g. 1024 or 2048)
        
        for i in range(self.depth):
            in_ch = curr_ch
            
            # The last block outputs 'start_ch' to match the initial input size
            if i == self.depth - 1:
                out_ch = start_ch
            else:
                # Standard U-Net step: cut channels in half
                out_ch = curr_ch // 2 
            
            self.ups.append(Up(in_ch, out_ch, bilinear))
            
            # Update curr_ch for next iteration
            # If bilinear, we output 'out_ch' from the DoubleConv.
            # If we are doing standard U-net symmetry, the next input will be this output.
            curr_ch = out_ch
            
        self.outc = OutConv(start_ch, 3)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, y, encx, x, t):
        batch_size = y.size(0)
        
        y = y.view(batch_size, 3, self.img_size, self.img_size)
        x = x.view(batch_size, 3, self.img_size, self.img_size)
        encx_img = self.decoder(encx) 
        t_map = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, self.img_size, self.img_size)
        
        x_in = torch.cat([y, encx_img, x, t_map], dim=1) 
        
        # Store skip connections
        x1 = self.inc(x_in)
        skips = [x1]
        
        feat = x1
        for down_layer in self.downs:
            feat = down_layer(feat)
            skips.append(feat)
            
        # The last element in skips is the bottleneck result
        curr = skips.pop() 
        
        # Up path
        for up_layer in self.ups:
            # We pop the matching skip connection (from deep to shallow)
            skip = skips.pop()
            curr = up_layer(curr, skip)
            
        logits = self.outc(curr)
        return logits.view(batch_size, -1)