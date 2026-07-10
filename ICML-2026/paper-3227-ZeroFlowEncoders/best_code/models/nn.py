import torch 
from torch import nn

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
    
class LinearDecoder(nn.Module):
    def __init__(self, encx_dim=64):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(encx_dim, 32*32)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x.view(-1, 1, 32, 32)

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

# vector field neural network over time steps (CNN)
class VectorField(torch.nn.Module):
    def __init__(self, encx_dim):
        super(VectorField, self).__init__()
        # Input to the first conv layer will be (B, 4, 32, 32)
        # where the 4 channels come from y, encx (expanded), x, and t (expanded)
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.conv3 = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.relu = torch.nn.ReLU()

        # self.decoder = torch.nn.Linear(encx_dim, 32*32)
        self.decoder = Decoder(encx_dim)
        # self.decoder = LinearDecoder(encx_dim)
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, y, encx, x, t):
        batch_size = y.size(0)
        # Reshape y and x to (B, 1, 32, 32)
        y_reshaped = y.view(batch_size, 1, 32, 32)
        x_reshaped = x.view(batch_size, 1, 32, 32)

        encx_reshaped = self.decoder(encx)
        # encx_reshaped = encx_decoded.view(batch_size, 1, 32, 32)
        # Expand t to (B, 1, 32, 32)
        t_expanded = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, 32, 32)
        # Concatenate along the channel dimension
        input_conv = torch.cat([y_reshaped, encx_reshaped, x_reshaped, t_expanded], dim=1) # (B, 4, 32, 32)
        dy_dt = self.relu(self.conv1(input_conv))
        dy_dt = self.relu(self.conv2(dy_dt))
        dy_dt = self.conv3(dy_dt) # Output is (B, 1, 32, 32)
        dy_dt = dy_dt.view(batch_size, -1) # Flatten to (B, 32*32)
        return dy_dt
