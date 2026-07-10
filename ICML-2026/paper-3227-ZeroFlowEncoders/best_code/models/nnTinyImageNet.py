import torch
import torch.nn as nn
from torchvision.models import resnet18

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, img_size=64):
        super(Decoder, self).__init__()
        self.img_size = img_size
        
        # --- 1. Dynamic Upsampling Strategy ---
        # We assume we always upsample by factor of 2 per layer.
        # We work backwards from img_size until we reach a small "start dimension" (e.g., between 4 and 8)
        
        current_dim = img_size
        num_upsamples = 0
        while current_dim > 6: # Stop when we get to 4, 5, or 6
            if current_dim % 2 != 0:
                raise ValueError(f"img_size {img_size} must be divisible by powers of 2 (e.g. 32, 64, 96, 128)")
            current_dim //= 2
            num_upsamples += 1
            
        self.start_dim = current_dim # This will be 4 for 32/64, and 6 for 96
        
        # --- 2. Project Latent Vector -> Spatial Feature Map ---
        # We map latent_dim -> (64 channels * start_dim * start_dim)
        self.fc_map = nn.Linear(latent_dim, 64 * self.start_dim * self.start_dim)
        
        # --- 3. Build Upsampling Layers ---
        layers = []
        in_channels = 64
        
        for i in range(num_upsamples):
            is_last = (i == num_upsamples - 1)
            out_channels = 3 if is_last else in_channels // 2
            
            # Standard ConvTranspose2d: doubles spatial dimension
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 
                                             kernel_size=4, stride=2, padding=1))
            
            if not is_last:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(True))
                in_channels = out_channels
            else:
                # Final activation for image output
                layers.append(nn.Sigmoid())
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (Batch_Size, latent_dim)
        x = self.fc_map(x)
        
        # Reshape to starting grid
        x = x.view(-1, 64, self.start_dim, self.start_dim) 
        
        img = self.net(x)
        return img # Output shape: (Batch_Size, 3, img_size, img_size)

class Encoder(nn.Module):
    def __init__(self, latent_dim, img_size=64):
        super(Encoder, self).__init__()
        self.img_size = img_size
        
        # 1. Load standard ResNet-18
        # We don't use 'pretrained=True' typically for SSL, as we want to learn from scratch.
        self.net = resnet18(weights=None) 
        
        # 2. Modify the first layer for small images (Tiny ImageNet/CIFAR)
        # Standard ResNet uses 7x7 conv stride 2 + MaxPool, which is too aggressive for 64x64.
        # We replace it with 3x3 conv stride 1 and remove MaxPool.
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        
        # 3. Replace the final Fully Connected layer
        # ResNet-18 output before FC is always 512 dimensions.
        self.net.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        # 1. Keep the interface's reshape logic 
        # (Handling flattened inputs if they come in that way)
        if x.dim() == 2:
            x = x.view(-1, 3, self.img_size, self.img_size)
            
        # 2. Pass through ResNet-18
        # The resnet18 forward pass automatically handles:
        # Conv layers -> Global Average Pooling -> Flatten -> FC
        x = self.net(x)
        
        return x

class VectorField(nn.Module):
    def __init__(self, encx_dim, img_size=64):
        super(VectorField, self).__init__()
        self.img_size = img_size
        self.encx_dim = encx_dim # Store dimension for reshaping later
        
        # --- 1. The Backbone (Downsampling) ---
        self.net = resnet18(weights=None)
        
        # MODIFY INPUT: Now takes 7 channels (y=3 + x=3 + t=1)
        # encx is REMOVED from the beginning.
        self.net.conv1 = nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity() 
        
        del self.net.avgpool
        del self.net.fc
        
        # --- 2. The Head (Upsampling) ---
        # ResNet Layer4 output is 512 channels at 8x8 resolution.
        # We will concatenate encx (size: encx_dim) at this layer.
        # So, the input to the first upsampling layer is 512 + encx_dim.
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512 + encx_dim, 256, kernel_size=4, stride=2, padding=1), # 8->16
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16->32
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, y, encx, x, t):
        batch_size = y.size(0)
        H, W = self.img_size, self.img_size
        
        # 1. Prepare Pixel Inputs (y, x, t only)
        y_reshaped = y.view(batch_size, 3, H, W)
        x_reshaped = x.view(batch_size, 3, H, W)
        t_expanded = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, H, W)
        
        # Concatenate Input -> (B, 7, H, W)
        input_tensor = torch.cat([y_reshaped, x_reshaped, t_expanded], dim=1)
        
        # 2. Downsampling (ResNet Backbone)
        h = self.net.conv1(input_tensor)
        h = self.net.bn1(h)
        h = self.net.relu(h)
        
        h1 = self.net.layer1(h)  # 64x64
        h2 = self.net.layer2(h1) # 32x32
        h3 = self.net.layer3(h2) # 16x16
        h4 = self.net.layer4(h3) # 8x8, 512 channels
        
        # 3. Inject encx at Latent Layer
        # h4 shape is (B, 512, 8, 8)
        # encx shape is (B, encx_dim)
        
        # We expand encx to match spatial dimensions (B, encx_dim, 8, 8)
        latent_h, latent_w = h4.shape[2], h4.shape[3]
        encx_expanded = encx.view(batch_size, self.encx_dim, 1, 1).expand(batch_size, self.encx_dim, latent_h, latent_w)
        
        # Concatenate along channel dimension
        # Result shape: (B, 512 + encx_dim, 8, 8)
        h4_cat = torch.cat([h4, encx_expanded], dim=1)
        
        # 4. Upsampling
        u1 = self.up1(h4_cat) # -> 16x16
        u2 = self.up2(u1)     # -> 32x32
        u3 = self.up3(u2)     # -> 64x64
        
        # 5. Final Prediction
        out = self.final_conv(u3) # (B, 3, 64, 64)
        
        return out.view(batch_size, -1)