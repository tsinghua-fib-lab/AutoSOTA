import torch
from torch import nn
from layers.RevIN import RevIN
from layers.SEMPO_EncDec import TSTEncoder, TowerEncoder, PretrainHead
from layers.pos_encoding import positional_encoding


class Model(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, configs):

        super().__init__()

        assert configs.head_type in ['pretrain', 'prediction'], 'head type should be either pretrain or prediction'
        head_dropout:float = 0.2
        n_heads:int = 16
        d_ff:int = 256
        norm:str = 'RMSNorm'
        attn_dropout:float = 0.
        dropout:float = 0.
        act:str = "silu"
        res_attention:bool = True
        pre_norm:bool = True
        store_attn:bool = False
        pe:str = 'zeros'
        learn_pe:bool = True
        self.freq_num:int = 4
        self.n_vars = configs.c_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
       
        # Norm
        self.revin_layer = RevIN(self.n_vars, affine=True)

        # Patching
        self.patch_num = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride * (self.patch_num - 1)
        self.s_begin = self.seq_len - tgt_len
       
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num * self.n_vars, self.d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
       
        # Input encoding
        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
           
        # Encoder
        self.encoder = TSTEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                  n_layers=self.e_layers, store_attn=store_attn)
        self.decoder = TowerEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                n_layers=self.d_layers, store_attn=store_attn)
        self.head = PretrainHead(self.d_model, self.patch_len, head_dropout)
       
        self.freq_len = self.seq_len // 2 + 1
        self.theta = nn.Parameter(torch.rand(1))
        self.tau_main = nn.Parameter(torch.rand(self.freq_num) * (self.seq_len // 2 + 1))
        self.mu_main = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))
        self.tau_res = nn.Parameter(torch.rand(self.freq_num) * (self.seq_len // 2 + 1))
        self.mu_res = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))
         

    def adaptive_energy_mask(self, z):
        bs, _, _ = z.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(z).pow(2).sum(dim=-1)

        # Flatten energy across freq_len and n_vars dimensions and then compute median
        # Flattening freq_len and n_vars into a single dimension
        flat_energy = energy.view(bs, -1)
        # Compute median
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  
        # Reshape to match the original dimensions
        median_energy = median_energy.view(bs, 1)  
        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        energy_mask = ((normalized_energy > self.theta).float() - self.theta).detach() + self.theta
        energy_mask = energy_mask.unsqueeze(-1)
        return energy_mask
   
    def adaptive_frequency_mask(self, z, tau, mu):
        freq_num, bs, freq_len, n_vars = z.shape
        freq_indices = torch.arange(freq_len, device=z.device).unsqueeze(0).unsqueeze(2).expand(bs, -1, n_vars)
        tau = tau.view(freq_num, 1, 1, 1).expand(-1, bs, freq_len, n_vars)
        mu = mu.view(freq_num, 1, freq_len, 1).expand(-1, bs, -1, n_vars)
        freq_mask = torch.where(
            mu == 1,
            (freq_indices < tau).float(),
            (freq_indices >= tau).float()
        )
        return freq_mask
   
    def decomposed_frequency_learning(self, x):
        bs, seq_len, n_vars = x.shape

        # apply FFT along the time dimension
        z = torch.fft.rfft(x, dim=1, norm='ortho')     # [bs x freq_len x n_vars]

        # dominant energy mask
        energy_mask = self.adaptive_energy_mask(z)
        # main energy part
        z_main = z * energy_mask
        # residual energy part
        z_res = z - z_main
        z_res = z_res.unsqueeze(0).expand(self.freq_num, -1, -1, -1)       # [freq_num x bs x freq_len x n_vars]
        z_main = z_main.unsqueeze(0).expand(self.freq_num, -1, -1, -1)     # [freq_num x bs x freq_len x n_vars]

        # frequency mask in main energy part
        main_freq_mask = self.adaptive_frequency_mask(z_main, self.tau_main, self.mu_main)
        # frequency mask in residual energy part
        res_freq_mask = self.adaptive_frequency_mask(z_res, self.tau_res, self.mu_res)
        z = z_main * main_freq_mask + z_res * res_freq_mask    # [freq_num x bs x freq_len x n_vars]

        # apply inverse FFT
        x = torch.fft.irfft(z, n=seq_len, dim=2, norm='ortho')   # [freq_num x bs x seq_len x n_vars]
        return x

   
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # norm  
        x = self.revin_layer(x_enc, 'norm')     # [bs x seq_len x n_vars]

        # decomposed frequency learning
        x = self.decomposed_frequency_learning(x)    # [freq_num x bs x seq_len x n_vars]

        # do patching    
        x = x[:, :, self.s_begin:, :]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)   # [freq_num x bs x patch_num x n_vars x patch_len]
        freq_num, bs, patch_num, n_vars, patch_len = x.shape
        x = x.reshape(-1, patch_num, n_vars, patch_len)

        # patch embedding
        x = self.patch_embed(x)    # [bs * freq_num x patch_num x n_vars x d_model] 

        # pos embedding
        x = x.transpose(1, 2)      # [bs * freq_num x n_vars x patch_num x d_model]        
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model))     # [bs' x n_vars * patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                 # [bs' x n_vars * patch_num x d_model]

        x = self.encoder(u)        # [bs' x d_model x n_vars * patch_num]
        x = torch.reshape(x, (-1, n_vars, patch_num, self.d_model))      # [bs' x n_vars x patch_num x d_model]
        x = x.permute(0, 1, 3, 2)                                        # [bs' x n_vars x d_model x patch_num]

        # multi-scale frequency aggregation
        x = x.view(-1, bs, n_vars, self.d_model, patch_num).mean(dim=0)  # [bs x n_vars x d_model x patch_num] 
        x_vec = x

        x = self.decoder(x)
        x = self.head(x)
        x = x.reshape(bs, patch_num * patch_len, n_vars)

        # denorm
        x = self.revin_layer(x, 'denorm')
        return x, x_vec
