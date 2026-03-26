import re, os
from typing import Optional
import torch
from torch import nn
from layers.RevIN import RevIN
from layers.SEMPO_EncDec import TSTEncoder, TowerEncoder, MixtrueExpertsLayer, PredictionHead, PretrainHead
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
        n_heads:int = 16
        head_dropout:float = 0.2
        individual:bool = False
        d_ff:int = 256
        norm:str = 'RMSNorm'
        attn_dropout:float = 0.
        dropout1:float = 0.
        dropout2:float = 0.2
        act:str = "silu"       # or silu
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
        self.head_type = configs.head_type
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.domain_len = configs.domain_len
        self.horizon_lengths = configs.horizon_lengths
       
        # Norm
        self.revin_layer_x = RevIN(self.n_vars, affine=True)
       
        # Projection
        self.projection_x = nn.Linear(self.seq_len, self.seq_len)

        # Patching
        self.patch_num = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride * (self.patch_num - 1)
        self.s_begin = self.seq_len - tgt_len
       
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num * self.n_vars, self.d_model)

        # Residual dropout
        self.dropout1 = nn.Dropout(dropout1)
       
        # Input encoding
        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
       
        # Encoder
        self.encoder = TSTEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout1, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                  n_layers=self.e_layers, store_attn=store_attn)
        self.decoder = TowerEncoder(self.d_model, n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout1, pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                n_layers=self.d_layers, store_attn=store_attn)
        self.head = PretrainHead(self.d_model, self.patch_len, head_dropout)

       
        self.freq_len = self.seq_len // 2 + 1
        self.theta = nn.Parameter(torch.rand(1))
        self.tau_main = nn.Parameter(torch.rand(self.freq_num) * (self.seq_len // 2 + 1))
        self.mu_main = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))
        self.tau_res = nn.Parameter(torch.rand(self.freq_num) * (self.seq_len // 2 + 1))
        self.mu_res = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))
           
        # load weight
        if configs.data == 'UTSD':
            setting = re.sub(r'_SEMPO_', '_SEMPO_CL_', configs.setting)
            print('loading pretrained encoder-decoder')
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
           
            modules_to_load = {
                "module.W_pos": self.W_pos,
                "module.theta": self.theta,
                "module.tau_main": self.tau_main,
                "module.mu_main": self.mu_main,
                "module.tau_res": self.tau_res,
                "module.mu_res": self.mu_res,
                "module.patch_embed": self.patch_embed,
                "module.encoder": self.encoder,
                "module.decoder": self.decoder,
                "module.head": self.head,
            }
            for prefix, target in modules_to_load.items():
                if isinstance(target, torch.nn.Module):
                    module_state_dict = {
                        k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k.startswith(prefix)
                    }
                    target.load_state_dict(module_state_dict, strict=True)
                else:
                    target.data.copy_(state_dict[prefix])
           
        # Frozen
        for param in [self.W_pos, self.theta, self.tau_main, self.mu_main, self.tau_res, self.mu_res]:
            param.requires_grad = False
        for module in [self.patch_embed, self.encoder, self.decoder, self.head]:    
            for param in module.parameters():
                param.requires_grad = False

        # Prefix
        self.domain_EnMoE = MixtrueExpertsLayer(prefix_projection=True, num_virtual_tokens=self.domain_len,
                            token_dim=self.d_model, encoder_hidden_size=self.d_model, n_layers=self.e_layers)
        self.domain_DeMoE = MixtrueExpertsLayer(prefix_projection=True, num_virtual_tokens=self.domain_len,
                            token_dim=self.d_model, encoder_hidden_size=self.d_model, n_layers=self.d_layers)
        self.domain_tokens = torch.arange(self.domain_len).long()
        # prefix dropout
        self.dropout2 = nn.Dropout(dropout2)
       
        # Head    
        pretrain_head_list = []
        for horizon_length in self.horizon_lengths:
            pretrain_head_list.append(PredictionHead(individual, self.n_vars, self.d_model, self.patch_num, horizon_length, head_dropout))
        self.pretrain_heads = nn.ModuleList(pretrain_head_list)
   
 
    def mixture_of_experts(self, x, bs, encode=True):
        if encode:
            past_key_values = self.domain_EnMoE(x, self.domain_tokens.to(x.device))      # [bs' x patch_num * n_vars x 2 * e_layers * d_model]
            past_key_values = past_key_values.view(self.e_layers, 2, bs * self.freq_num, -1, self.d_model)   # [e_layers x 2 x bs' x patch_num * n_vars x d_model]           
        else:
            x = x.reshape(bs, -1, self.d_model)
            past_key_values = self.domain_DeMoE(x, self.domain_tokens.to(x.device))   # [bs x patch_num * n_vars x 2 * d_layers * d_model]
            past_key_values = past_key_values.view(self.d_layers, 2, bs, -1, self.d_model)    # [d_layers x 2 x bs x patch_num * n_vars x d_model]
        past_key_values = self.dropout2(past_key_values)
        return past_key_values
   
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
        freq_num, bs, freq_len, patch_num = z.shape
        freq_indices = torch.arange(freq_len, device=z.device).unsqueeze(0).unsqueeze(2).expand(bs, -1, patch_num)
        tau = tau.view(freq_num, 1, 1, 1).expand(-1, bs, freq_len, patch_num)
        mu = mu.view(freq_num, 1, freq_len, 1).expand(-1, bs, -1, patch_num)
        freq_mask = torch.where(
            mu == 1,
            (freq_indices < tau).float(),
            (freq_indices >= tau).float()
        )
        return freq_mask
   
    def decomposed_frequency_learning(self, x):
        bs, seq_len, n_vars = x.shape                     
        
        # apply FFT along the time dimension
        z = torch.fft.rfft(x, dim=1, norm='ortho')  # [bs x freq_len x n_vars]
        
        # dominant energy mask
        energy_mask = self.adaptive_energy_mask(z)
        # main energy part
        z_main = z * energy_mask
        # residual energy part
        z_res = z - z_main
        z_res = z_res.unsqueeze(0).expand(self.freq_num, -1, -1, -1)  # [freq_num x bs x freq_len x n_vars]
        z_main = z_main.unsqueeze(0).expand(self.freq_num, -1, -1, -1)  # [freq_num x bs x freq_len x n_vars]
        
        # frequency mask in main energy part
        main_freq_mask = self.adaptive_frequency_mask(z_main, self.tau_main, self.mu_main)
        # frequency mask in residual energy part
        res_freq_mask = self.adaptive_frequency_mask(z_res, self.tau_res, self.mu_res)
        z = z_main * main_freq_mask + z_res * res_freq_mask  # [freq_num x bs x freq_len x n_vars]
        
        # apply inverse FFT
        x = torch.fft.irfft(z, n=seq_len, dim=2, norm='ortho')  # [freq_num x bs x seq_len x n_vars]
        return x
   
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # norm  
        x = self.revin_layer_x(x_enc, 'norm')        # [bs x seq_len x n_vars]

        # projection 
        x = self.projection_x(x.permute(0, 2, 1)).permute(0, 2, 1)

        # decomposed frequency learning
        x = self.decomposed_frequency_learning(x)    # [freq_num x bs x seq_len x n_vars]    

        # do patching    
        x = x[:, :, self.s_begin:, :] 
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [freq_num x bs x patch_num x n_vars x patch_len]
        freq_num, bs, patch_num, n_vars, patch_len = x.shape
        x = x.reshape(-1, patch_num, n_vars, patch_len)

        # patch embedding
        x = self.patch_embed(x)    # [bs * freq_num x patch_num x n_vars x d_model]                                                   

        # pos embedding
        x = x.transpose(1, 2)  # [bs * freq_num x n_vars x patch_num x d_model]        
        u = torch.reshape(x, (-1, n_vars * patch_num, self.d_model))  # [bs' x n_vars * patch_num x d_model]
        u = self.dropout1(u + self.W_pos)  # [bs' x n_vars * patch_num x d_model] 

        # domain prefix, concat in K and V
        u_d = self.mixture_of_experts(u, bs=bs, encode=True)  # [e_layers x 2 x bs' x patch_num * n_vars x d_model]
        
        # encoder
        x = self.encoder(u, u_d)  # [bs' x d_model x n_vars * patch_num]
        x = x.reshape(-1, n_vars, patch_num, self.d_model)  # [bs' x n_vars x patch_num x d_model]
        x = x.permute(0, 1, 3, 2)  # [bs' x n_vars x d_model x patch_num]                                                      
        
        # multi-scale frequency aggregation
        x = x.view(-1, bs, n_vars, self.d_model, patch_num).mean(dim=0)  # [bs x n_vars x d_model x patch_num]
          
        # domain prefix, concat in K and V
        x_d = self.mixture_of_experts(x, bs=bs, encode=False)  # [d_layers x 2 x bs x patch_num * n_vars x d_model]
            
        # decoder
        x = self.decoder(x, x_d)  # [bs x n_vars x d_model x patch_num]                                                 
        
        # head
        y = [self.revin_layer_x(head(x), 'denorm') for head in self.pretrain_heads]
            
        x = self.head(x)
        x = x.reshape(bs, patch_num * patch_len, n_vars)
        x = self.revin_layer_x(x, 'denorm')
        return y,  x

       
       