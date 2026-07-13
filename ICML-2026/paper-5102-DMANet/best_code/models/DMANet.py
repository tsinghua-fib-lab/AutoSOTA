import torch
import torch.nn as nn
from layers.RevIN import RevIN

class CombinedAntiAlias(nn.Module):
    def __init__(self, configs, in_channels, out_channels, freq_len,act=True,bn=None,drop=True):
        super().__init__()
        self.configs = configs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freq_len = freq_len
        self.kernel_size = configs.kernel_size
        self.stride = configs.down_sampling_window
        self.ICR = min(torch.tensor(self.out_channels / self.in_channels),
                       self.kernel_size) / (self.stride)
        
        self.bn1 = nn.BatchNorm1d(self.in_channels) if bn else None
        self.bn2 = nn.BatchNorm1d(self.in_channels) if bn else None
        self.act = nn.GELU() if act else None
        self.DWConv = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding=(self.kernel_size - 1) // 2 ,
                groups=in_channels,  
                bias=False
            )
        self.PWConv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.drop = nn.Dropout(configs.dropout) if drop else None

    def apply_antialias_mask(self, x_freq):
        nyquist_idx = int(x_freq.shape[-1] * (self.ICR / 2))
        mask = torch.zeros_like(x_freq)
        mask[..., :nyquist_idx] = 1.0
        return x_freq * mask

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=-1, norm="ortho")
        x_freq = self.apply_antialias_mask(x_freq)
        x = torch.fft.irfft(x_freq, n=x.size(-1), dim=-1, norm="ortho")

        x_time = self.DWConv(x)

        if self.bn1:
            x_time = self.bn1(x_time)

        x_out = self.PWConv(x_time)

        if self.act:
            x_out =  self.act(x_out)
        
        if self.bn2:
            x_out = self.bn2(x_out)
        
        if self.drop:
            x_out = self.drop(x_out)
            
        return x_out

class MultiScaleEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_layers = configs.down_sampling_layers
        self.layernorm = nn.LayerNorm(configs.d_model)
        self.enc_in=configs.enc_in
        self.aa_layers = nn.ModuleList()
        self.kernel_size = configs.kernel_size
        self.stride = configs.down_sampling_window
        current_channels = configs.enc_in

        for i in range(self.num_layers):
            # 动态计算输出通道数（确保整数）
            layer = CombinedAntiAlias(
                configs,
                in_channels=current_channels,
                out_channels=max(1, int(current_channels * configs.down_sampling_c)),
                freq_len=configs.d_model//(configs.down_sampling_window**i)
            )
            self.aa_layers.append(layer)
            current_channels = layer.out_channels
            
        # 新增自适应权重参数
        self.fusion_weights = nn.Parameter(
            torch.ones(configs.down_sampling_layers, 1, 1, 1)
        )
        self.scale=0.02
        self.weight = nn.ParameterList([
                nn.Parameter(torch.randn(configs.enc_in, 
                max(1, int((configs.enc_in)* (configs.down_sampling_c ** i))), 
                configs.d_model//(configs.down_sampling_window**i)) * 0.02)
                for i in range(1, self.num_layers+1)
                    ])

    def joint_interpolation(self, i, x, target_len,len_ratio=None):

        x_fft = torch.fft.rfft(x, dim=-1)
        w = torch.fft.rfft(self.weight[i], dim=-1, norm='ortho')
        expanded_fft = torch.einsum('bcf,ocf->bof', x_fft, w)

        out_fft = torch.zeros(*expanded_fft.shape[:2], 
                            target_len//2+1, 
                            device=x.device)
        out_fft[..., :expanded_fft.size(2)] = expanded_fft

        out = torch.fft.irfft(out_fft, n=target_len)

        if len_ratio:
            out = out * (1 / (self.aa_layers[self.num_layers-i-1].ICR/2)) 

        return out 
    
    def forward(self, x):
        x = self.layernorm(x)
        current = x
        pyramid = []
        fused = 0
        for i, layer in enumerate(self.aa_layers):
            down_feat = layer(current)  
            fused = self.joint_interpolation(
                i,
                x = down_feat,
                target_len=x.size(-1)
            ) 
            pyramid.append(fused)
            current = down_feat 
        stacked = torch.stack(pyramid, dim=0)  
        weighted = stacked * self.fusion_weights.softmax(dim=0)
        output = weighted.sum(dim=0)
        return output

class Model(nn.Module):
    """最终模型"""
    def __init__(self, configs):
        super().__init__()
        self.revin_layer = RevIN(configs.enc_in)
        self.encoders = nn.ModuleList([
            MultiScaleEncoder(configs) for _ in range(configs.e_layers)
        ])
        self.layernorm1 = nn.LayerNorm(configs.d_model)
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model*configs.d_ff),
            nn.LeakyReLU(),
            nn.Linear(configs.d_model*configs.d_ff, configs.pred_len)
        )
        self.embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.W_pos = nn.Parameter(torch.empty(1, 1, configs.d_model)) 
        nn.init.uniform_(self.W_pos, -0.02, 0.02) 
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_norm = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)
        x = self.embedding(x_norm) + self.W_pos  
        for encoder in self.encoders:
            x = encoder(x) + x 
        features = self.layernorm1(x)
        output = self.decoder(features)
        output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        return output