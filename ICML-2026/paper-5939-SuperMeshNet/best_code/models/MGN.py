import torch.nn.functional as F
import torch.nn as nn
from utils.knn_interpolate import knn_interpolate
import torch
from torch_scatter import scatter_add
from models.Common import MLP


 
class EdgeBlock(nn.Module):

    def __init__(self, hidden_size):
        
        super(EdgeBlock, self).__init__()
        self.net1=MLP(hidden_size*3,hidden_size,hidden_size)
        
                  
    def forward(self, graph):
        
        node_attr, edge_index,  edge_attr = graph
        senders_idx, receivers_idx = edge_index
        edges_to_collect1 = []
        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]
       
        
        edges_to_collect1.append(senders_attr)
        edges_to_collect1.append(receivers_attr)
        edges_to_collect1.append(edge_attr)               
        collected_edges1 = torch.cat(edges_to_collect1, dim=1)
        edge_attr = self.net1(collected_edges1) 
        x=node_attr
        
        return x, edge_index, edge_attr


    
class NodeBlock(nn.Module):

    def __init__(self, hidden_size, ib_e):

        super(NodeBlock, self).__init__()        
        self.net = MLP(hidden_size*2,hidden_size,hidden_size)  
        self.ib_e=ib_e     
              
    def forward(self, graph):      
        node_attr, edge_index, edge_attr = graph
        nodes_to_collect = []  
        
        senders_idx1, receivers_idx1 = edge_index          
        agg_received_edges1 = scatter_add(edge_attr, receivers_idx1, dim=0, dim_size=len(node_attr))  
        if self.ib_e:
            agg_received_edges1=agg_received_edges1-torch.mean(agg_received_edges1,0)      

        nodes_to_collect.append(node_attr)
        nodes_to_collect.append(agg_received_edges1)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)
        
        return x,  edge_index, edge_attr


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size,
                input_size,                
                hidden_size):
        
        super(Encoder, self).__init__()
        self.nb_encoder = MLP(input_size, hidden_size, hidden_size)
        self.eb_encoder = MLP(edge_input_size, hidden_size, hidden_size)
           
    def forward(self, graph):
       
        node_attr,  edge_index, edge_attr=graph
        x = self.nb_encoder(node_attr)     
        edge_attr= self.eb_encoder(edge_attr)
        
        return x, edge_index, edge_attr


class GnBlock(nn.Module):

    def __init__(self, hidden_size, ib_n, ib_e):

        super(GnBlock, self).__init__()     
        self.eb_module = EdgeBlock(hidden_size)
        self.nb_module = NodeBlock(hidden_size, ib_e)
        self.ib_n = ib_n
        
       
        
    def forward(self, graph):
        x_last,  edge_index, edge_attr_last =graph
      
        graph = self.eb_module((x_last,  edge_index, edge_attr_last))
        graph = self.nb_module(graph)
        x, _, edge_attr = graph
        
        edge_attr = edge_attr+edge_attr_last
        x = x+x_last
        
        if self.ib_n:
            x=x-torch.mean(x,0)
        
                
                
        return x, edge_index,  edge_attr


class MGN_shared(torch.nn.Module):
    def __init__(self,  depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e):
        super().__init__()
        
        self.encoder1=Encoder(edge_input_size, y_input_size+pos_input_size, hidden_size)
        self.encoder2=MLP(pos_input_size*2,hidden_size, hidden_size)
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
        

    def forward(self, l_pos, l_y, l_e,  h_pos, h_e, edge_attr=None):
        
        if edge_attr==None:
            edge_attr_l=torch.concatenate([l_pos[l_e[0]],l_pos[l_e[1]]],-1)
        else:
            edge_attr_l=edge_attr
            
        
        graph1=(torch.concatenate([l_y,l_pos],-1), l_e, edge_attr_l)
        graph1= self.encoder1(graph1)  

        
        for model in self.processor_list1:    
            graph1 = model(graph1) 
        
        x1, _, _ =graph1
        
    
        x2=knn_interpolate(x1,l_pos,h_pos) 
        edge_attr_h=torch.concatenate([h_pos[h_e[0]],h_pos[h_e[1]]],-1)
        edge_attr_h=self.encoder2(edge_attr_h)
        graph2=(x2, h_e, edge_attr_h)
        
        for model in self.processor_list2:
            graph2 = model(graph2)      
        
        x, _, _ =graph2
        return x
    

class MGN_shared2(torch.nn.Module):
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
        

    def forward(self, l_pos, l_y, l_e,  h_pos, h_e, t, edge_attr=None):
        
        if edge_attr==None:
            edge_attr_l=torch.concatenate([l_pos[l_e[0]],l_pos[l_e[1]]],-1)
            

        graph1=(torch.concatenate([l_y,l_pos,t],-1), l_e, edge_attr_l)
        graph1= self.encoder1(graph1)  

        
        for model in self.processor_list1:    
            graph1 = model(graph1) 
        
        x1, _, _ =graph1
        
    
        x2=knn_interpolate(x1,l_pos,h_pos) 
        edge_attr_h=torch.concatenate([h_pos[h_e[0]],h_pos[h_e[1]]],-1)
        edge_attr_h=self.encoder2(edge_attr_h)
        graph2=(x2, h_e, edge_attr_h)
        
        for model in self.processor_list2:
            graph2 = model(graph2)      
        
        x, _, _ =graph2
        return x
        
        

class GnBlock_wo_residual(nn.Module):

    def __init__(self, hidden_size, ib_n, ib_e):

        super(GnBlock_wo_residual, self).__init__()     
        self.eb_module = EdgeBlock(hidden_size)
        self.nb_module = NodeBlock(hidden_size, ib_e)
        self.ib_n = ib_n
        
       
        
    def forward(self, graph):
        x_last,  edge_index, edge_attr_last =graph
      
        graph = self.eb_module((x_last,  edge_index, edge_attr_last))
        graph = self.nb_module(graph)
        x, _, edge_attr = graph
        
        edge_attr = edge_attr
        x = x
        
        if self.ib_n:
            x=x-torch.mean(x,0)
        
                
                
        return x, edge_index,  edge_attr


class MGN_shared_wo_residual(torch.nn.Module):
    def __init__(self,  depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e):
        super().__init__()
        
        self.encoder1=Encoder(edge_input_size, y_input_size+pos_input_size, hidden_size)
        self.encoder2=MLP(pos_input_size*2,hidden_size, hidden_size)
        processor_list1 = []
        for _ in range(depth):
            processor_list1.append(GnBlock_wo_residual(hidden_size, ib_n, ib_e))
        self.processor_list1=nn.ModuleList(processor_list1)
        processor_list2 = []
        for _ in range(depth):
            processor_list2.append(GnBlock_wo_residual(hidden_size, ib_n, ib_e))
        self.processor_list2=nn.ModuleList(processor_list2)
        
        self.ib_d=ib_e
        self.ib_n=ib_n
        self.edge_input_size=edge_input_size
        

    def forward(self, l_pos, l_y, l_e,  h_pos, h_e, edge_attr=None):
        
        if edge_attr==None:
            edge_attr_l=torch.concatenate([l_pos[l_e[0]],l_pos[l_e[1]]],-1)
        else:
            edge_attr_l=edge_attr
            

        graph1=(torch.concatenate([l_y,l_pos],-1), l_e, edge_attr_l)
        graph1= self.encoder1(graph1)  

        
        for model in self.processor_list1:    
            graph1 = model(graph1) 
        
        x1, _, _ =graph1
        
    
        x2=knn_interpolate(x1,l_pos,h_pos) 
        edge_attr_h=torch.concatenate([h_pos[h_e[0]],h_pos[h_e[1]]],-1)
        edge_attr_h=self.encoder2(edge_attr_h)
        graph2=(x2, h_e, edge_attr_h)
        
        for model in self.processor_list2:
            graph2 = model(graph2)      
        
        x, _, _ =graph2
        return x
    

class MGN_shared_visual(torch.nn.Module):
    def __init__(self,  depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e):
        super().__init__()
        
        self.encoder1=Encoder(edge_input_size, y_input_size+pos_input_size, hidden_size)
        self.encoder2=MLP(pos_input_size*2,hidden_size, hidden_size)
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
        

    def forward(self, l_pos, l_y, l_e,  h_pos, h_e, edge_attr=None):
        
        if edge_attr==None:
            edge_attr_l=torch.concatenate([l_pos[l_e[0]],l_pos[l_e[1]]],-1)
        else:
            edge_attr_l=edge_attr
            
        x_list=[]
        graph1=(torch.concatenate([l_y,l_pos],-1), l_e, edge_attr_l)
        graph1= self.encoder1(graph1)  

        x_list.append(graph1[0].detach().cpu())
        for model in self.processor_list1:    
            graph1 = model(graph1) 
            x_list.append(graph1[0].detach().cpu())
        
        x1, _, _ =graph1
        
    
        x2=knn_interpolate(x1,l_pos,h_pos) 
        edge_attr_h=torch.concatenate([h_pos[h_e[0]],h_pos[h_e[1]]],-1)
        edge_attr_h=self.encoder2(edge_attr_h)
        graph2=(x2, h_e, edge_attr_h)
        
        x_list.append(graph2[0].detach().cpu())
        for model in self.processor_list2:
            graph2 = model(graph2)
            x_list.append(graph2[0].detach().cpu())      
        
        x, _, _ =graph2
        return x, x_list