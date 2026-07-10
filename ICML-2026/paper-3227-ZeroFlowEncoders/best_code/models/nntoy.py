import torch 
from torch import nn
import math

class CNNEncoder(torch.nn.Module):
    def __init__(self, inputdim=64, hidden_dim=32):
        """
        Args:
            inputdim: Total number of features (W*W). Must be a perfect square.
            hidden_dim: Number of channels in the hidden convolutional layers.
        """
        super(CNNEncoder, self).__init__()
        self.inputdim = inputdim
        
        # 1. Determine spatial dimension W from inputdim
        self.W = int(math.sqrt(inputdim))
        if self.W * self.W != inputdim:
            raise ValueError(f"inputdim {inputdim} must be a perfect square (e.g., 64, 784) for a 2D grid.")

        # 2. Define CNN Architecture
        # We use padding=1 with kernel_size=3 to maintain spatial resolution (W x W)
        self.cnn = nn.Sequential(
            # Input: (Batch, 1, W, W) -> Mask is treated as 1-channel image
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Output: (Batch, 1, W, W) -> Map back to 1 channel for gating
            nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        )

    def get_gates(self, mask):
        # mask shape: (batch_size, inputdim)
        
        # 1. Reshape Flattened -> 2D Grid: (B, 1, W, W)
        x = mask.view(-1, 1, self.W, self.W)
        
        # 2. Pass through CNN
        x = self.cnn(x)
        
        # 3. Flatten 2D Grid -> Flattened: (B, inputdim)
        w = x.view(-1, self.inputdim)
        
        return torch.sigmoid(w)
    
    def get_gates_sum(self, mask):
        return torch.sum(self.get_gates(mask))
    
    def forward(self, y, mask):
        # y: (batch_size, inputdim)
        # mask: (batch_size, inputdim)
        
        gates = self.get_gates(mask)  # Returns (batch_size, inputdim)
        
        # Element-wise multiplication (broadcasting not needed as shapes match)
        out = y * gates
        return out

class LSTMEncoder(torch.nn.Module):
    def __init__(self, inputdim=64, hiddn_dim=128, bidirectional=True, sensitivity=1.0):
        """
        Args:
            sensitivity (float): Controls the LSTM's long-range memory.
                                 > 1.0 : High sensitivity (remembers distant history).
                                 0.0   : Standard/Neutral.
                                 < 0.0 : Low sensitivity (focuses on immediate context).
        """
        super(LSTMEncoder, self).__init__()
        self.inputdim = inputdim
        
        # 1. Embedding Layer
        self.embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=16)
        
        # 2. Bidirectional LSTM
        lstm_input_size = 16 
        self.lstm = torch.nn.LSTM(input_size=lstm_input_size, 
                                  hidden_size=hiddn_dim, 
                                  batch_first=True,
                                  bidirectional=bidirectional)
        
        # 3. Projection
        # Input dim is hiddn_dim * 2 because of bidirectionality
        self.fc_out = torch.nn.Linear(hiddn_dim * 2, 1)
        
        self.sparse_weights = torch.nn.Parameter(torch.randn(1, inputdim))

        # --- SENSITIVITY CONTROL ---
        # Apply the bias initialization immediately after creation
        self._init_forget_gate(sensitivity)
    
    def _init_forget_gate(self, value):
        """
        Manually sets the forget gate bias to 'value' for all LSTM layers.
        PyTorch default is 0.0. Setting this to 1.0+ enables long-term memory.
        """
        for name, param in self.lstm.named_parameters():
            if 'bias_hh' in name:
                # The bias tensor in PyTorch is concatenated: [Input, Forget, Cell, Output]
                # We want to slice out the 2nd quarter (The Forget Gate)
                hidden_size = self.lstm.hidden_size
                
                # Apply the fill value to the forget gate slice
                param.data[hidden_size : 2*hidden_size].fill_(value)

    def get_gates(self, mask):
        # mask shape: (batch_size, inputdim) containing 0s and 1s
        mask_indices = mask.long() 
        
        # Embed binary spikes -> (batch, seq_len, 16)
        x = self.embedding(mask_indices)
        
        # Run Bidirectional LSTM -> (batch, seq_len, hiddn_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Project to scalar gate -> (batch, seq_len, 1)
        w = self.fc_out(lstm_out)
        
        # Squeeze -> (batch, seq_len)
        w = w.squeeze(-1)
        
        return torch.sigmoid(w)
    
    def get_gates_sum(self, mask):
        return torch.sum(self.get_gates(mask))
    
    def forward(self, y, mask):
        gates = self.get_gates(mask)
        out = y * gates
        return out

class Encoder(torch.nn.Module):
    def __init__(self, inputdim = 64, hiddn_dim=128):
        super(Encoder, self).__init__()
        self.inputdim = inputdim
        
        self.fc1 = torch.nn.Linear(inputdim, hiddn_dim)
        self.fc2 = torch.nn.Linear(hiddn_dim, hiddn_dim)
        self.fc3 = torch.nn.Linear(hiddn_dim, inputdim)
        
        self.sparse_weights = torch.nn.Parameter(torch.randn(1, inputdim))
    
    def get_gates(self, mask):
        # sparse selector on y
        w = torch.relu(self.fc1(mask))
        w = torch.relu(self.fc2(w))
        w = self.fc3(w)
        # w = self.sparse_weights
        return torch.sigmoid(w)
    
    def get_gates_sum(self, mask):
        return torch.sum(self.get_gates(mask))
    
    def forward(self, y, mask):
        # y: (batch_size, inputdim)
        gates = self.get_gates(mask)  # (batch_size, inputdim )
        out = y * gates
        return out

# vector field neural network over time steps (CNN)
class Conv1dEncoder(torch.nn.Module):
    """1D Convolutional Encoder for Markov chain structure recovery.
    
    Processes the 50-dim mask as a 1D signal with kernel_size=7,
    naturally capturing the 3rd-order Markov chain adjacency 
    (each dimension's gate depends on neighbors within ±3).
    """
    def __init__(self, inputdim=64, hidden_dim=32, kernel_size=7):
        super(Conv1dEncoder, self).__init__()
        self.inputdim = inputdim
        padding = kernel_size // 2
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, hidden_dim, kernel_size, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
    
    def get_gates(self, mask):
        # mask: (batch_size, inputdim)
        x = mask.unsqueeze(1)  # (B, 1, inputdim)
        x = self.conv(x)       # (B, 1, inputdim)
        return torch.sigmoid(x.squeeze(1))
    
    def get_gates_sum(self, mask):
        return torch.sum(self.get_gates(mask))
    
    def forward(self, y, mask):
        # y: (batch_size, inputdim)
        gates = self.get_gates(mask)
        out = y * gates
        return out

class VectorField(torch.nn.Module):
    def __init__(self, inputdim = 64, hiddn_dim=1024):
        super(VectorField, self).__init__()
        self.inputdim = inputdim
        
        self.fc1 = torch.nn.Linear(inputdim + inputdim + inputdim + inputdim + 1, hiddn_dim)
        self.fc2 = torch.nn.Linear(hiddn_dim, hiddn_dim)
        self.fc3 = torch.nn.Linear(hiddn_dim, inputdim)
        
    def forward(self, xt, enc_y, y, idx, t):
        # xt: (batch_size, inputdim)
        # ency: (batch_size, inputdim)
        # y: (batch_size, inputdim)
        # t: (batch_size, 1)
        
        inp = torch.cat([xt, enc_y, y, idx, t], dim=1) # Concatenate imputed data and time
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out