import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPointEmbedding(nn.Module):
    def __init__(self, input_channel, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(input_channel, embed_dim)

    def forward(self, x):
        # x shape: [batch_size, input_channel, num_points, 1]
        x = x.squeeze(3).transpose(1, 2)  # [batch_size, num_points, input_channel]
        x = self.embedding(x)  # [batch_size, num_points, embed_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1, enable_dino=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        if enable_dino:
            linear_layer = nn.Linear(embed_dim+768, feedforward_dim)
        else:
            linear_layer = nn.Linear(embed_dim, feedforward_dim)
        self.ffn = nn.Sequential(
            linear_layer,
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, addition):
        # x shape: [num_points, batch_size, embed_dim]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        if addition is not None:
            x_combined = torch.cat([x, addition], dim=-1)
        else:
            x_combined = x
        
        ffn_output = self.ffn(x_combined)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # x shape: [num_points, batch_size, embed_dim]
        # memory shape: [num_downsampled_points, batch_size, embed_dim]
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output, _ = self.cross_attention(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

class TransformerPointNet(nn.Module):
    def __init__(self, out_dim=16, net_channels=128, depth=8, num_heads=8, feedforward_dim=512, dropout=0.1, enable_dino=True):
        super(TransformerPointNet, self).__init__()
        self.enable_dino = enable_dino
        
        self.embedding = TransformerPointEmbedding(input_channel=out_dim, embed_dim=net_channels)

        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=net_channels, num_heads=num_heads, 
                                    feedforward_dim=feedforward_dim, dropout=dropout,
                                    enable_dino=enable_dino)
            for _ in range(depth // 2)
        ])
        
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(embed_dim=net_channels, num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout)
            for _ in range(depth // 2)
        ])
        
        self.output_layer = nn.Linear(net_channels, 1)

    def forward(self, data):
        # data: [batch_size, input_channel, num_points, 1]
        x_attention = data
        if self.enable_dino:
            x_attention = data[:,:16,:,:]
        x_attention = self.embedding(x_attention)
        x_attention = x_attention.transpose(0, 1)  # Transformer expects [num_points, batch_size, embed_dim]
        
        addition = None
        if self.enable_dino:
            x_no_attention = data[:,16:,:,:]
            x_no_attention = x_no_attention.squeeze(-1).permute(2, 0, 1)
            addition = x_no_attention
        
        # Encoder Pass
        memory = x_attention
        for encoder in self.encoders:
            memory = encoder(memory, addition)

        # Decoder Pass
        output = x_attention
        for decoder in self.decoders:
            output = decoder(output, memory)

        output = output.transpose(0, 1)  # [batch_size, num_points, embed_dim]
        logits = self.output_layer(output).transpose(1, 2)
        return logits
