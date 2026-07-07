from typing import Optional

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .query import Query_Gen_transformer
from .gauss_mild import GaussConv
from .util import mutual_information
import torch.nn.functional as F
import numpy as np
#from .query import Query_Gen

class infonet(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        query_gen: Query_Gen_transformer,
        decoder_query_dim: int
    ):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.query_gen = query_gen
        self.mild = GaussConv(size=15, nsig=3, channels=1)
        
        #self.query = nn.Parameter(torch.randn(1, decoder_query_dim, decoder_query_dim))

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ):
    
        latents = self.encoder(inputs, input_mask)
        query = self.query_gen(inputs)
        
        outputs = self.decoder(
            x_q=query,
            latents=latents,
            query_mask=query_mask
        )
        #print(outputs.shape)
        #torch.save(outputs.cpu().numpy(), "lookuptable.pth")
        outputs = outputs.unsqueeze(1)
        outputs = self.mild(outputs)
        outputs = outputs.squeeze(1)
        #torch.save(outputs.cpu().numpy(), "lookuptable_smoothed.pth")
        #print("saved!!!!!!!!!!!!!!!1")
        mi_lb = mutual_information(inputs, outputs)

        return mi_lb