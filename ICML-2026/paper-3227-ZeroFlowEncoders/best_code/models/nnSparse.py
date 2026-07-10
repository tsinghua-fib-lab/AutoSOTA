import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Decoder(nn.Module):
    def __init__(self, encx_dim=64):
        super(Decoder, self).__init__()
        
        # 1. Project Latent Vector -> Spatial Feature Map
        # We start with a small 4x4 spatial grid. 
        # 64 channels * 4 * 4 = 1024 neurons needed.
        self.fc_map = nn.Linear(encx_dim, 64 * 4 * 4)
        
        # 2. Upsampling Stack
        self.net = nn.Sequential(
            # Layer 1: 4x4 -> 8x8
            # ConvTranspose2d params: (in, out, kernel, stride, padding)
            # Formula: Output = (Input-1)*Stride - 2*Padding + Kernel
            # (4-1)*2 - 2*1 + 4 = 8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), # Optional: Helps training stability
            nn.ReLU(True),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Layer 3: 16x16 -> 32x32 (Final Output)
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            
            # Output Activation:
            # Use Sigmoid if your images are [0, 1]
            # Use Tanh if your images are [-1, 1]
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Input shape: (Batch_Size, 64)
        
        # 1. Project and Reshape
        x = self.fc_map(x)
        x = x.view(-1, 64, 4, 4) # Unflatten to (Batch, 64, 4, 4)
        
        # 2. Decode
        img = self.net(x)
        return img # Output shape: (Batch_Size, 1, 32, 32)

# the encoder neural network (MLP)
class Encoder(torch.nn.Module):
    def __init__(self, encx_dim):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.conv3 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32

        self.fc = torch.nn.Linear(32*32, encx_dim)
    def forward(self, x):
        # Assuming x is (batch_size, input_dim) where input_dim is 32*32
        x = x.view(-1, 1, 32, 32) # Reshape to (batch_size, channels, height, width)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x
    
class LinearEncoder(torch.nn.Module):
    def __init__(self, encx_dim):
        super(LinearEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(32*32, encx_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
    

class LinearDecoder(torch.nn.Module):
    def __init__(self, encx_dim):
        super(LinearDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(encx_dim, 32*32)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1, 32, 32) # Reshape to (batch_size, channels, height, width)
        return x

class VectorField(nn.Module):
    def __init__(self, encx_dim, num_classes=10):
        super(VectorField, self).__init__()
        
        # 1. Embeddings
        # We process 'x' and 't' separately to give them more semantic weight
        self.x_embed = nn.Embedding(num_classes, 32) # Map class to 32-dim vector
        self.t_embed = nn.Sequential(                # Map time to 32-dim vector
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        # 2. Decoder for encx (Kept from your original design)
        # Assuming LinearDecoder is defined elsewhere or is a simple linear layer
        self.decoder = Decoder(encx_dim)
        # self.decoder = LinearDecoder(encx_dim) 

        # 3. The Backbone (ResNet with Dilation)
        # Input channels: 
        #   y (1) + encx (1) + x_embed (1) + t_embed (32) = 35 channels
        self.in_conv = nn.Conv2d(35, 64, kernel_size=3, padding=1)
        
        # Residual Blocks with increasing dilation to see the whole 32x32 image
        self.res1 = ResidualBlock(64, 64, dilation=1)
        self.res2 = ResidualBlock(64, 64, dilation=2)
        self.res3 = ResidualBlock(64, 64, dilation=4)
        self.res4 = ResidualBlock(64, 64, dilation=1)

        # 4. Final Output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, y, encx, x, t):
        """
        y:    (B, 32*32) or (B, 1, 32, 32) - Current state
        encx: (B, encx_dim)                - Encoded condition
        x:    (B, )                        - Class labels (int)
        t:    (B, 1)                       - Time
        """
        B = y.shape[0]

        # --- A. Preprocessing Inputs ---
        # 1. Process Image y
        y = y.view(B, 1, 32, 32)
        
        # 2. Process Condition encx
        # Output of decoder should be (B, 1, 32, 32) based on your logic
        encx_img = self.decoder(encx) 
        
        # 3. Process Label x (Learnable Embedding)
        # x_vec = self.x_embed(x.long())       # (B, 32)
        # x_map = x_vec.view(B, 32, 1, 1).expand(B, 32, 32, 32)
        x = x.view(B, 1, 32, 32)
        
        # 4. Process Time t (MLP Embedding)
        # We assume t is (B, 1) or (B)
        if t.dim() == 1: t = t.unsqueeze(1)
        t_vec = self.t_embed(t)              # (B, 32)
        t_map = t_vec.view(B, 32, 1, 1).expand(B, 32, 32, 32)

        # --- B. Feature Fusion ---
        # Concatenate: 1 + 1 + 1 + 32 = 35 channels
        out = torch.cat([y, encx_img, x, t_map], dim=1)

        # --- C. Network Body (ResNet) ---
        out = self.in_conv(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        
        # --- D. Final Projection ---
        out = self.final_conv(out) # (B, 1, 32, 32)
        
        # Flatten to match your output requirement
        return out.view(B, -1)

# --- Helper Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation):
        super().__init__()
        # Valid padding for dilation: padding = dilation
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.GroupNorm(8, out_c) # GroupNorm is better for small batches
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1) # Standard conv
        self.bn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU() # Smoother than ReLU

        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        return self.act(out + residual)