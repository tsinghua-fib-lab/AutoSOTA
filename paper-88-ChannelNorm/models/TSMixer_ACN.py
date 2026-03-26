import torch.nn as nn
import torch
from layers.Embed import DataEmbedding_inverted

class CosineSimilarityWeightedAverage(nn.Module):
    def __init__(self, C, D,temperature):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C, D))
        self.bias = nn.Parameter(torch.zeros(C, D))
        self.temperature = temperature
        
        self.weight_global = nn.Parameter(torch.ones(C, D))
        self.bias_global = nn.Parameter(torch.ones(C, D))

    def forward(self, input):
        # Step 1: Normalize input along dimension D
        input_norm = input / input.norm(dim=-1, keepdim=True)  # Shape: (N, C, D)
        
        # Step 2: Compute cosine similarity between channels
        cosine_similarity = torch.matmul(input_norm, input_norm.transpose(1, 2))  # Shape: (N, C, C)
        
        # Step 3: Apply softmax to get attention weights
        attention_weights = torch.softmax(cosine_similarity/self.temperature, dim=-1)  # Shape: (N, C, C)
        
        # Step 4: Expand self.weight to match the batch size
        weight_expanded = self.weight.unsqueeze(0).expand(input.size(0), -1, -1)  # Shape: (N, C, D)
        bias_expanded = self.bias.unsqueeze(0).expand(input.size(0), -1, -1)  # Shape: (N, C, D)
        # Step 5: Compute the weighted average of self.weight using attention weights
        avg_weight = torch.matmul(attention_weights, weight_expanded)  # Shape: (N, C, D)
        avg_bias = torch.matmul(attention_weights, bias_expanded)  # Shape: (N, C, D)
        avg_weight = avg_weight * self.weight_global.unsqueeze(0)
        avg_bias = avg_bias * self.bias_global.unsqueeze(0)
        return (avg_weight*input) +  avg_bias
    
    
class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()
        self.individual = False
        self.channels = configs.enc_in

        self.temporal = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )
        
        self.norm = CosineSimilarityWeightedAverage(configs.enc_in, configs.d_model, configs.temperature)

    def forward(self, x):
        # x: [B, L, D]
        if self.individual:
            output = torch.zeros([x.size(0), x.size(1), x.size(2)],
                                          dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.temporal[i](
                    x[:, :, i])
        else:
            x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        # self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.projection = nn.Linear(configs.d_model, configs.pred_len)

        ##
        # Embedding
        embed_dim = configs.d_model 
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, embed_dim, configs.embed, configs.freq,
                                                    configs.dropout)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, _, N = x_enc.shape
        # Embedding
        x_enc = self.enc_embedding(x_enc, None)  # [b n d]
        x_enc = x_enc.transpose(1, 2)  # [b n D] -> [b D n]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        dec_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')

