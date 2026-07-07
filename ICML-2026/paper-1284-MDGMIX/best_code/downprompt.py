import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
from torch_geometric.utils import normalize_edge_index
import copy

class downstreamprompt(nn.Module):
    def __init__(self, x, num_domains, feature_dim):
        super(downstreamprompt, self).__init__()
        self.fusion_token = composedtoken(num_domains)

    def forward(self,features, sp_adj, data, domain_token, gcn):
        target_data = copy.deepcopy(data)
        target_data.x, p = self.fusion_token(data.x, domain_token)
        embed = gcn(target_data).squeeze(0)
    
        return embed 



class downprompt(nn.Module):
    def __init__(self, x, num_domains, ft_in, nb_classes, feature_dim):
        super(downprompt, self).__init__()

        
        self.downstreamPrompt = downstreamprompt(x, num_domains, feature_dim)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.ema_ave = None
        self.ema_momentum = 0.9



    def forward(self,features, sp_adj, data, domain_token, gcn, idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, sp_adj,data, domain_token, gcn) 

        rawret = embeds[idx]

        num =  rawret.shape[0]
     
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)
            if self.ema_ave is None:
                self.ema_ave = self.ave.clone().detach()
            else:
                self.ema_ave = self.ema_momentum * self.ema_ave + (1 - self.ema_momentum) * self.ave.detach()

        prototypes = self.ema_ave if self.ema_ave is not None else self.ave
        ret = F.cosine_similarity(rawret.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)

        ret = F.softmax(ret, dim=1)
 
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downprompt_graph(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  fea_pretext_weights, str_pretext_weights,
                  combines, type_='mul', ablation = 'all'):
        super(downprompt_graph, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreamprompt(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.ema_ave = None
        self.ema_momentum = 0.9


    def forward(self,features,adj,sparse,gcn,idx,batch,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)
        
        ret = F.cosine_similarity(rawret.unsqueeze(1), self.ave.unsqueeze(0), dim=-1)

        ret = F.softmax(ret, dim=1)



        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

def averageemb(labels, rawret):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel



class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    def forward(self, graph_embedding):
    
        assert len(graph_embedding) == self.weight.shape[1], 'length must equal'
        ans = torch.zeros_like(graph_embedding[0])
        for i in range(len(graph_embedding)):
            ans += self.weight[0][i] * graph_embedding[i]
        return ans

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding1, graph_embedding2):

        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class composedtoken(nn.Module):
    def __init__(self, num_domains, type_='mul'):
        super(composedtoken, self).__init__()
      
        self.prompt = weighted_prompt( num_domains )
        self.type = type_

        

    def forward(self, seq, texttoken):        
        texttoken = self.prompt(texttoken)

      
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'cat':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = torch.cat((texttoken , seq), dim=1)     
        if self.type == 'mul':
            rets = texttoken * seq
        return rets, texttoken


class textprompt(nn.Module):
    def __init__(self, hid_units, type_='mul'):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype = type_
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding=self.weight * graph_embedding

        return graph_embedding

