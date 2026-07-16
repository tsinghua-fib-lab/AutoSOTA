import math
import torch.nn as nn
import torch.nn.functional as F
from layers.BioFormer_EncDec import (
    Encoder,
    EncoderLayer,
    FreqBandAlign,
    TimeSegAlign,
    Sample_con_ln,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PyramidConvEmbedding

class Model(nn.Module):
    """
    BioFormer model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len_merge = 0
        for i in range(3):
            self.seq_len_merge += math.ceil(configs.seq_len / pow(2, i+1))
        augmentations = configs.augmentations.split(",")

        # Embedding
        self.enc_embedding = PyramidConvEmbedding(
            c_in = configs.enc_in,
            d_model = configs.d_model,
            augmentation = augmentations,
            )
        
        # FreqBandAlign 
        fbd_layers = (
            [FreqBandAlign(configs.num_class, configs.mag_learning, configs.phase_learning,num_bands=6, d_token=configs.d_model // 2)
            for _ in range(configs.e_layers)]
            if configs.use_FBD else None
        )

        # TimeSegAlign 
        tbd_layers = (
            [TimeSegAlign(configs.num_class,num_bands = 6)
            for _ in range(configs.e_layers)]
            if configs.use_FBD else None
        )

        # norm 
        norm_layer = (
            Sample_con_ln(normalized_shape=configs.d_model,
                                 proj_hidden=configs.d_model // 2,
                                 hidden_dim=configs.d_model,
                                 alpha=configs.use_ASSLN)
            if configs.use_ASSLN else
            nn.LayerNorm(configs.d_model)
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            fbd_layers,  
            norm_layer,
        )
   
        # Decoder
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.seq_len_merge * configs.d_model, configs.num_class
            )
            

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            # Embedding
            enc_out = self.enc_embedding(x_enc)
            
            # Encoder
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            
            # Output
            output = self.act(enc_out) 
            output = self.dropout(output)
            
            output = output.reshape(output.shape[0], -1)    # (batch_size, seq_length * d_model)
            output = self.projection(output)                # (batch_size, num_classes)
            return output
