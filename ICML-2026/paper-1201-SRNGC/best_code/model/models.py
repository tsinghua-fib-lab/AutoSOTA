import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPModel(nn.Module):
    def __init__(
        self,
        dim: int,
        lag: int,
        hidden_dim: int,
        num_layers: int,
        componentwise: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.lag = lag
        self.componentwise = componentwise

        input_dim = dim * lag
        if componentwise:
            # dim separate MLPs, each seeing the full input and outputting 1 scalar
            self.blocks = nn.ModuleList([
                MLPBlock(input_dim, hidden_dim, 1, num_layers, dropout)
                for _ in range(dim)
            ])
        else:
            # one full MLP, outputting all dims at once
            self.block = MLPBlock(input_dim, hidden_dim, dim, num_layers, dropout)

    def forward(self, x):
        """
        x: [B, lag * dim]
        returns: [B, dim]
        """
        if self.componentwise:
            # run each scalar head separately on the same input
            outs = [block(x) for block in self.blocks]   # list of [B, 1]
            y = torch.cat(outs, dim=1)                   # [B, dim]
        else:
            y = self.block(x)                            # [B, dim]
        return y

class LSTMBlock(nn.Module):
    def __init__(self, dim, lag, hidden_dim, num_layers, output_dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.lag = lag

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, lag * dim]
        B = x.size(0)
        x_seq = x.view(B, self.lag, self.dim)        # [B, lag, dim]
        _, (h_n, _) = self.lstm(x_seq)               # h_n: [num_layers, B, hidden_dim]
        h_last = h_n[-1]                             # [B, hidden_dim]
        y = self.fc(h_last)                          # [B, output_dim]
        return y

class LSTMModel(nn.Module):
    def __init__(
        self,
        dim: int,
        lag: int,
        hidden_dim: int,
        num_layers: int,
        componentwise: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.lag = lag
        self.componentwise = componentwise

        if componentwise:
            # dim separate LSTM blocks, each seeing full input and predicting one scalar
            self.blocks = nn.ModuleList([
                LSTMBlock(dim, lag, hidden_dim, num_layers, output_dim=1, dropout=dropout)
                for _ in range(dim)
            ])
        else:
            # one full LSTM block predicting all dims
            self.block = LSTMBlock(dim, lag, hidden_dim, num_layers, output_dim=dim, dropout=dropout)

    def forward(self, x):
        """
        x: [B, lag * dim]
        returns: [B, dim]
        """
        if self.componentwise:
            outs = [block(x) for block in self.blocks]   # list of [B, 1]
            y = torch.cat(outs, dim=1)                   # [B, dim]
        else:
            y = self.block(x)                            # [B, dim]
        return y

class SRUBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        lag: int, 
        dim_stats: int,              # Replaces both dim_iid_stats and dim_rec_stats
        dim_rec_stats_feedback: int, #  original parameter
        dim_final_stats: int,        #  original parameter
        A: list,                     # Scales
        output_dim: int,
        dropout: float = 0.0         # Kept for signature compatibility
    ):
        super().__init__()
        self.dim = dim
        self.lag = lag
        self.numScales = len(A)
        self.dim_stats = dim_stats

        # Register A_mask as a buffer so it automatically lives on the correct device (CPU/GPU)
        A_list = [scale for scale in A for _ in range(dim_stats)]
        self.register_buffer('A_mask', torch.tensor(A_list, dtype=torch.float32).view(1, -1))

        #  original MLPs
        self.lin_xr2phi = nn.Linear(dim + dim_rec_stats_feedback, dim_stats, bias=True)
        self.lin_r = nn.Linear(self.numScales * dim_stats, dim_rec_stats_feedback, bias=True)
        self.lin_o = nn.Linear(self.numScales * dim_stats, dim_final_stats, bias=True) 
        self.lin_y = nn.Linear(dim_final_stats, output_dim, bias=True)

    def forward(self, x):
        # x shape: [B, lag * dim]
        B = x.size(0)
        
        # Reshape to sequence: [B, lag, dim]
        x_seq = x.view(B, self.lag, self.dim)  
        
        # Initialize recurrent state for the batch
        # Replaces self.u_t_prev.fill_(0)
        u_t = torch.zeros(B, self.dim_stats * self.numScales, device=x.device, dtype=x.dtype)
        
        #  SEQUENTIAL LOOP
        for t in range(self.lag):
            x_t = x_seq[:, t, :]  # [B, dim]

            # 1. Update r_t based on previous state
            r_t = F.elu(self.lin_r(u_t))

            # 2. Generate iid statistics: phi_t
            # Note: torch.cat with dim=1 replaces torch.flatten() to safely handle batch sizes > 1
            xr_t = torch.cat([x_t, r_t], dim=1)
            phi_t = F.elu(self.lin_xr2phi(xr_t))
            
            # 3. Generate multiscale recurrent statistics: u_t
            phi_tile = phi_t.repeat(1, self.numScales)
            u_t = self.A_mask * u_t + (1 - self.A_mask) * phi_tile

        # 4. Generate final statistics and predicted output from the LAST state
        o_t = F.elu(self.lin_o(u_t))
        y_last = self.lin_y(o_t)  # [B, output_dim]

        return y_last


class SRUModel(nn.Module):
    def __init__(
        self,
        dim: int,
        lag: int,
        dim_stats: int,
        dim_rec_stats_feedback: int,
        dim_final_stats: int,
        A: list,
        componentwise: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.lag = lag
        self.componentwise = componentwise

        if componentwise:
            # dim separate SRU blocks, each predicting one scalar
            self.blocks = nn.ModuleList([
                SRUBlock(
                    dim=dim, lag=lag, dim_stats=dim_stats, 
                    dim_rec_stats_feedback=dim_rec_stats_feedback, 
                    dim_final_stats=dim_final_stats, 
                    A=A, output_dim=1, dropout=dropout
                )
                for _ in range(dim)
            ])
        else:
            # one full SRU block predicting all dims
            self.block = SRUBlock(
                dim=dim, lag=lag, dim_stats=dim_stats, 
                dim_rec_stats_feedback=dim_rec_stats_feedback, 
                dim_final_stats=dim_final_stats, 
                A=A, output_dim=dim, dropout=dropout
            )

    def forward(self, x):
        """
        x: [B, lag * dim]
        returns: [B, dim]
        """
        if self.componentwise:
            outs = [block(x) for block in self.blocks]   # list of [B, 1]
            y = torch.cat(outs, dim=1)                   # [B, dim]
        else:
            y = self.block(x)                            # [B, dim]
        return y

class ResidualBlock(nn.Module):
    def __init__(self, input, hidden, output, dropout):
        super(ResidualBlock, self).__init__()
        self.linear_1 = torch.nn.utils.parametrizations.weight_norm(nn.Linear(input, hidden))
        self.linear_2 = nn.Linear(hidden, output)
        self.linear_res = nn.Linear(input, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output)

    def forward(self, x):
        """
        x: [Batch, hidden]
        """
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        res = self.linear_res(x)
        out = h + res
        out = self.layernorm(out)
        return out
    
    def struct_loss(self):
        return torch.sum(self.linear_res.weight ** 2)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, hidden_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.inputgate = nn.Linear(input_dim, hidden_dim)
        self.outputgate = nn.Linear(hidden_dim, output_dim)
        self.inputgate = torch.nn.utils.parametrizations.weight_norm(self.inputgate)

        modules = [ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout) for _ in range(layers)]
        self.encoders = nn.ModuleList(modules)
        
    def forward(self, x):
        x = self.inputgate(x)
        for net in self.encoders:
            x = net(x)
        x = self.outputgate(x)
        return x

class TSMixerBlock(nn.Module):
    """
    Causally-Safe TSMixer Block.
    No LayerNorms (prevents false causal entanglement).
    Uses weight_norm to match ResidualMLP.
    """
    def __init__(self, lag, dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        # 1. Time mixing (operates over 'lag')
        self.time_linear1 = parametrizations.weight_norm(nn.Linear(lag, lag))
        self.time_linear2 = nn.Linear(lag, lag)
        
        # 2. Feature mixing (operates over 'dim', expanding to hidden_dim)
        self.feature_linear1 = parametrizations.weight_norm(nn.Linear(dim, hidden_dim))
        self.feature_linear2 = nn.Linear(hidden_dim, dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, lag, dim]
        
        # --- Time Mixing ---
        h = x.transpose(1, 2)            # Swap to [B, dim, lag]
        h = self.time_linear1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.time_linear2(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)            # Swap back to [B, lag, dim]
        x = x + h                        # Residual connection
        
        # --- Feature Mixing ---
        h = self.feature_linear1(x)      
        h = self.relu(h)
        h = self.dropout(h)
        h = self.feature_linear2(h)      
        h = self.dropout(h)
        x = x + h                        # Residual connection
        
        return x

class TSMixer(nn.Module):
    """
    TSMixer architecture adapted for 1-step multivariate prediction.
    Now includes input/output gates to perfectly match ResidualMLP's structure.
    Supports 0 layers.
    """
    def __init__(self, dim, lag, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.lag = lag
        
        self.blocks = nn.ModuleList([
            TSMixerBlock(lag, dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # --- Replaced single projection with ResidualMLP's Gate Logic ---
        self.inputgate = nn.Linear(lag * dim, hidden_dim)
        self.inputgate = parametrizations.weight_norm(self.inputgate)
        self.outputgate = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # x is natively [B, lag * dim]. Reshape for mixing blocks.
        B = x.size(0)
        x_seq = x.view(B, self.lag, self.dim)
        
        # Sequence extraction (skipped if num_layers == 0)
        for block in self.blocks:
            x_seq = block(x_seq)
            
        # Flatten back out
        x_flat = x_seq.reshape(B, -1)
        
        # --- Match ResidualMLP's 0-block forwarding ---
        h = self.inputgate(x_flat)
        out = self.outputgate(h)
        
        return out

class CausalSSMBlock(nn.Module):
    """
    A single State Space block.
    Strictly channel-independent to preserve causal sparsity.
    """
    def __init__(self, dim, state_dim, dropout=0.1):
        super(CausalSSMBlock, self).__init__()
        
        # Isolated state space parameters per variable
        self.A = nn.Parameter(torch.empty(dim, state_dim))
        self.B = nn.Parameter(torch.empty(dim, state_dim))
        self.C = nn.Parameter(torch.empty(dim, state_dim))
        
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.C, mean=0.0, std=0.1)
        
        # Mamba traditionally uses SiLU (Swish) activation
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, lag, dim]
        B, lag, dim = x.shape
        h = torch.zeros(B, dim, self.A.shape[1], device=x.device)
        
        ssm_outputs = []
        for t in range(lag):
            x_t = x[:, t, :] # [B, dim]
            
            # Element-wise math: Variable A never touches Variable B
            h = self.A * h + self.B * x_t.unsqueeze(-1)
            y_t = torch.sum(self.C * h, dim=-1) 
            
            ssm_outputs.append(y_t)
            
        out_seq = torch.stack(ssm_outputs, dim=1) # [B, lag, dim]
        
        out_seq = self.act(out_seq)
        out_seq = self.dropout(out_seq)
        
        # Residual connection (causally safe because it's pure addition)
        return out_seq + x

class CausalSimpleMamba(nn.Module):
    """
    Multi-layer, pure-PyTorch SSM baseline.
    Now includes input/output gates to perfectly match ResidualMLP's structure.
    Supports 0 layers.
    """
    def __init__(self, dim, lag, state_dim, num_layers, dropout=0.1):
        super(CausalSimpleMamba, self).__init__()
        self.dim = dim
        self.lag = lag
        
        self.blocks = nn.ModuleList([
            CausalSSMBlock(dim, state_dim, dropout) for _ in range(num_layers)
        ])
        
        # --- Replaced single projection with ResidualMLP's Gate Logic ---
        self.inputgate = nn.Linear(dim * lag, state_dim)
        self.inputgate = parametrizations.weight_norm(self.inputgate)
        self.outputgate = nn.Linear(state_dim, dim)

    def forward(self, x):
        # x natively [B, lag * dim]
        B = x.size(0)
        x_seq = x.view(B, self.lag, self.dim)
        
        # Sequence extraction (skipped if num_layers == 0)
        for block in self.blocks:
            x_seq = block(x_seq)
            
        # Flatten for causal mapping
        x_flat = x_seq.view(B, -1)
        
        # --- Match ResidualMLP's 0-block forwarding ---
        h = self.inputgate(x_flat)
        out = self.outputgate(h)
        
        return out
