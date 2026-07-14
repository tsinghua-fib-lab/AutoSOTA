import torch
import numpy as np
import torch.nn.functional as F
from models.MGN import GnBlock, Encoder
from models.Common import MLP
from tqdm import tqdm
import os
import torch.nn as nn
from torch.utils.checkpoint import checkpoint



class MGN_shared(torch.nn.Module):
    def __init__(self,  depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e):
        super().__init__()
        
        self.encoder1=Encoder(edge_input_size, y_input_size+pos_input_size, hidden_size)
        self.encoder2=MLP(edge_input_size,hidden_size, hidden_size)
        processor_list1 = []
        for _ in range(depth):
            processor_list1.append(GnBlock(hidden_size, ib_n, ib_e))
        self.processor_list1=nn.ModuleList(processor_list1)
        processor_list2 = []
        for _ in range(depth):
            processor_list2.append(GnBlock(hidden_size, ib_n, ib_e))
        self.processor_list2=nn.ModuleList(processor_list2)
        
        self.ib_d=ib_e
        self.ib_n=ib_n
        self.edge_input_size=edge_input_size
    
    def _run_block(self, x, e, edge_attr, block):
       
        return block((x, e, edge_attr))
        

    def forward(self, l_pos, l_y, l_e,  h_pos, h_e, edge_attr=None):
        
        if edge_attr==None:
            edge_attr_l=torch.concatenate([l_pos[l_e[0]],l_pos[l_e[1]]],-1)
            

        graph1=(torch.concatenate([l_y,l_pos],-1), l_e, edge_attr_l)
        graph1= self.encoder1(graph1)  

        x, e, edge_attr=graph1
        for model in self.processor_list1:    
            
            x, e, edge_attr = checkpoint(
                    self._run_block,        # 함수 f
                    x, e, edge_attr, model, # 인자들
                    use_reentrant=False
                )

        graph1 = x, e, edge_attr    
        x1, _, _ =graph1
        
    
        x2=F.interpolate((x1).reshape(1, 32,32,-1).permute(0, 3, 1, 2), scale_factor=32,mode='bilinear').permute(0, 2,3,1).reshape(1024**2,-1)
        edge_attr_h=torch.concatenate([h_pos[h_e[0]],h_pos[h_e[1]]],-1)
        edge_attr_h=self.encoder2(edge_attr_h)

       
        graph2=(x2, h_e, edge_attr_h)
        x, e, edge_attr=graph2
        for model in self.processor_list2:
            
           
            
            x, e, edge_attr = checkpoint(
                    self._run_block,        # 함수 f
                    x, e, edge_attr, model, # 인자들
                    use_reentrant=False
                )

        graph2 = x, e, edge_attr  
           
        
        x, _, _ =graph2
        return x


class Decoder_F(torch.nn.Module):
    def __init__(self, hidden_size, output_size, residual):
        super().__init__()  
        self.decoder=MLP(hidden_size, hidden_size,output_size)
        self.residual=residual
    def forward(self, emb, l_y, l_pos, h_pos):
        x=emb
        x=self.decoder(x)
        if self.residual:
            return x+F.interpolate((l_y).reshape(1, 32,32,-1).permute(0, 3, 1, 2), scale_factor=32,mode='bilinear').permute(0, 2,3,1).reshape(1024**2,-1)
        else:
            return x


class Decoder_G(torch.nn.Module):
    def __init__(self, hidden_size, output_size, residual):
        super().__init__()  
        self.decoder=MLP(hidden_size, hidden_size,output_size)
        self.residual=residual
    def forward(self, emb1, l_y1, l_pos1, h_pos1, emb2, l_y2, l_pos2, h_pos2):
        x1=emb1
        x2=emb2
        x=self.decoder(x1-x2)
        if self.residual:       
            return x+F.interpolate((l_y1-l_y2).reshape(1, 32,32,-1).permute(0, 3, 1, 2), scale_factor=32,mode='bilinear').permute(0, 2,3,1).reshape(1024**2,-1)
        else:
            return x


