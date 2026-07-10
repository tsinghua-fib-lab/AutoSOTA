import torch 
from torch import nn
import math

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

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim, img_size=64):
        super(Encoder, self).__init__()
        self.img_size = img_size
        
        # Input channels 3 (RGB)
        # Note: Conv layers with padding=1 preserve spatial dimensions (32->32, 64->64)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # Compresses 16 channels to 1 channel before flattening
        self.conv3 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1) 

        # CHANGED: Dynamic flattening size based on img_size
        self.fc = torch.nn.Linear(img_size * img_size, latent_dim)

    def forward(self, x):
        # Reshape to (Batch, 3, img_size, img_size)
        x = x.view(-1, 3, self.img_size, self.img_size) 
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        
        # Flatten: (B, 1, H, W) -> (B, H*W)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

class VectorField(torch.nn.Module):
    def __init__(self, encx_dim, img_size=64):
        super(VectorField, self).__init__()
        self.img_size = img_size
        
        # Input channels: y(3) + encx(3) + x(3) + t(1) = 10 channels
        self.conv1 = torch.nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.conv3 = torch.nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1) 
        self.relu = torch.nn.ReLU()

        # Pass img_size to decoder so it knows target output size
        self.decoder = Decoder(latent_dim=encx_dim, img_size=img_size)
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, y, encx, x, t):
        batch_size = y.size(0)
        H, W = self.img_size, self.img_size
        
        # Reshape y and x to (B, 3, H, W)
        y_reshaped = y.view(batch_size, 3, H, W)
        x_reshaped = x.view(batch_size, 3, H, W)

        # encx_reshaped is (B, 3, H, W)
        encx_reshaped = self.decoder(encx)
        
        # Expand t to (B, 1, H, W)
        t_expanded = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, H, W)
        
        # Concatenate
        input_conv = torch.cat([y_reshaped, encx_reshaped, x_reshaped, t_expanded], dim=1) 
        
        dy_dt = self.relu(self.conv1(input_conv))
        dy_dt = self.relu(self.conv2(dy_dt))
        dy_dt = self.conv3(dy_dt) # Output is (B, 3, H, W)
        
        # Flatten to (B, 3 * H * W)
        dy_dt = dy_dt.view(batch_size, -1) 
        return dy_dt