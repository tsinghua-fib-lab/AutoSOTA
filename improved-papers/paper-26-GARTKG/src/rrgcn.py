import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from diffusion.difffu_21 import MultiHeadedAttention
from rgcn.reservoir_sampler import *
from rgcn.CorruptTriplesGlobal import *

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    # def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, attn_heads, opn, initial_h,sequence_len, num_bases=-1, num_basis=-1,
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, attn_heads, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False,data_list=None,all_known_entities=None,reservoirSampler=None,args=None):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.temperature_object=0.5
        self.args=args
        # self.initial_h=initial_h

        self.w1 = torch.nn.Parameter(torch.Tensor(
            self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(
            self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(
            self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(
            num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.linear = nn.Linear(2*self.h_dim, self.h_dim)
        self.loss_ce = nn.CrossEntropyLoss()
        
        # self.corrupt = CorruptTriplesGlobal(args, data_list, all_known_entities)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(
                self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        
        #TIE
        # self.reservoirSampler=
        # self.corrupt = CorruptTriplesGlobal(args,data_list,all_known_entities)
        self.loss_del=torch.nn.BCELoss()
        self.loss_mse=torch.nn.MSELoss()
        
        
        self.loss_reply=torch.nn.MSELoss()
        
        
        self.linear_map=nn.Linear(self.h_dim, self.h_dim)
        self.linear_trans=nn.Linear(self.h_dim, self.h_dim)

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.rgcn_history = RGCNCell(num_ents,
                                     h_dim,
                                     h_dim,
                                     num_rels * 2,
                                     num_bases,
                                     num_basis,
                                     num_hidden_layers,
                                     dropout,
                                     self_loop,
                                     skip_connect,
                                     encoder_name,
                                     self.opn,
                                     self.emb_rel,
                                     use_cuda,
                                     analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight,
                                gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.attention = MultiHeadedAttention(
            heads=attn_heads, hidden_size=self.h_dim, dropout=dropout)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5)) 

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(
                num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(
                num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def select_entity(self, tag, embeding, use_cuda):
        entity_all = torch.cat([tag[:, 0], tag[:, 2]])
        s_o_embeding = torch.cat([embeding[:, 0, :], embeding[:, 2, :]], dim=0)
        # s_o_embeding = embeding[:, 0, :]
        embeding_dict = dict()
        # for i, ent in enumerate(entity_all):
        #     if str(ent) in embeding_dict.keys():
        #         embeding_dict[str(ent)] = embeding_dict[str(ent)]+s_o_embeding[i]
        #     else:
        #         embeding_dict[str(ent)]=s_o_embeding[i]

        entity_all, indices = torch.unique(entity_all, return_inverse=True)
        ent_embeding = torch.zeros((self.num_ents, self.h_dim)).to(
            self.gpu) if use_cuda else torch.zeros(1)
        # for i,ent in enumerate(entity_all):
        #     ent_embeding[int(ent)]=embeding_dict[str(ent)]
        for i, ent in enumerate(entity_all):
            ent_embeding[int(ent)] = s_o_embeding[indices == i].mean(dim=0)

        return entity_all, ent_embeding

    def forward(self, g_list, static_graph, use_cuda, model_output=None,model_output_x_t=None, tag=None, diffu_rep="add", output_reply_triple=None, current_query_history_graph=None):
        gate_list = []
        degree_list = []
        # self.h,static_emb =self.initial_sh(static_graph)
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat(
                (self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(
                static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(
                self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        if model_output is not None:        
            model_output = torch.concat([model_output[:,0:2,:], model_output_x_t.unsqueeze(dim=1)], dim=1)
    
            entity_all, ent_embeding = self.select_entity(
                tag, model_output, use_cuda)

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda(
            ) if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(
                    x_input, self.emb_rel)    # 第1层输入
                self.h_0 = F.normalize(
                    self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(
                    x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(
                    self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(
                current_h) if self.layer_norm else current_h

            if model_output is not None:
                if diffu_rep == "add":
                    current_h[entity_all[:], :] = self.alpha*current_h[entity_all[:],
                                                            :]+(1-self.alpha)*ent_embeding[entity_all[:], :]
                if diffu_rep == "Linear":
                    current_h[entity_all[:], :] = current_h[entity_all[:],:]+ent_embeding[entity_all[:], :]

            time_weight = F.sigmoid(
                torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)

            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list

    def predict(self, test_graph, num_rels, static_graph, test_triplets, model_output,model_output_o, tag, use_cuda, current_query_history_triples=None):
        
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            mask = inverse_test_triplets[:, 1] < self.num_rels
            inverse_test_triplets[mask, 1] = inverse_test_triplets[mask, 1] + self.num_rels
            inverse_test_triplets[~mask, 1] = inverse_test_triplets[~mask, 1] - self.num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))


            evolve_embs, _, r_emb, _, _ = self.forward(
                test_graph, static_graph, use_cuda, model_output=model_output,model_output_x_t=model_output_o, tag=tag, current_query_history_graph=current_query_history_triples)
            embedding = F.normalize(
                evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(
                embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(
                embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel
    def loss_of_reply(self, e_embedding, r_embedding, reply,mode='e'):
        if mode is 'e':
            scores_ob = self.decoder_ob.forward(
                    e_embedding, r_embedding, reply).view(-1, self.num_ents)
            loss = self.loss_e(scores_ob, reply[:, 2])
        elif mode is 'r':
            score_rel = self.rdecoder.forward(
                e_embedding, r_embedding, reply, mode="train").view(-1, 2 * self.num_rels)
            loss = self.loss_r(score_rel, reply[:, 1])
        return loss

    def loss_diffu_ce(self, rep_diffu,r_emb,labels, true_triples=None, mask_seq=None, emb_ent=None):

        loss = 0
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        # """
        scores_ob = self.decoder_ob.forward(rep_diffu, r_emb, labels).view(-1, self.num_ents)
        loss_ent = self.loss_e(scores_ob, labels[:, 2])
        
        return loss_ent
    def get_all_triple(self,triples):
        inverse_triples = triples[:, [2, 1, 0]]
        mask = inverse_triples[:, 1] < self.num_rels
        inverse_triples[mask, 1] = inverse_triples[mask, 1] + self.num_rels
        inverse_triples[~mask, 1] = inverse_triples[~mask, 1] - self.num_rels
        
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        return all_triples
    
    
    def get_loss(self, glist, triples, static_graph, use_cuda,known_entities=None,negative_rate=None, model_output=None,model_output_x_t=None,tag=None, diffuc=None, output_reply_triple=None, current_query_history_graph=None,mode=None,inc_model=None):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(
            self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(
            self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(
            self.gpu) if use_cuda else torch.zeros(1)
        loss_Reply = torch.zeros(1).cuda().to(
            self.gpu) if use_cuda else torch.zeros(1)
        
        if model_output is not None:
            entity_all, ent_embeding = self.select_entity(
                tag, model_output, use_cuda)
            output_reply_triple = torch.tensor(output_reply_triple).cuda()
            all_output_reply_triple=self.get_all_triple(output_reply_triple)

        all_triples=self.get_all_triple(triples)
        
        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda, model_output,model_output_x_t, tag, diffu_rep=diffuc,
                                                            output_reply_triple=output_reply_triple, current_query_history_graph=current_query_history_graph)
        pre_emb = F.normalize(
            evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        if model_output is not None:
            loss_Reply=self.loss_reply(pre_emb[entity_all,:],ent_embeding[entity_all,:])
        
        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(
                pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
            
        if model_output is not None:
            loss_ent += self.loss_of_reply(pre_emb,r_emb,output_reply_triple,mode='e')

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(
                pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        
        if model_output is not None:
            loss_ent += self.args.mu*self.loss_of_reply(pre_emb,r_emb,all_output_reply_triple,mode='e')
            loss_rel += self.args.mu_r*self.loss_of_reply(pre_emb,r_emb,all_output_reply_triple,mode='r')

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(
                            static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * \
                            torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * \
                        torch.sum(torch.masked_select(
                            math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(
                            static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * \
                            torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * \
                        torch.sum(torch.masked_select(
                            math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static, evolve_embs[-1],loss_Reply


class Initial_h(nn.Module):
    def __init__(self, args, num_words, num_ents, num_static_rels, layer_norm):
        super(Initial_h, self).__init__()
        
        self.num_words = num_words
        self.num_ents = num_ents
        self.num_static_rels = num_static_rels
        self.layer_norm = layer_norm
        self.gpu=args.gpu
        self.h_dim=args.n_hidden

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(
            num_ents, self.h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.words_emb = torch.nn.Parameter(torch.Tensor(
            self.num_words, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.words_emb)

        self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, args.n_bases,
                                                activation=F.rrelu, dropout=args.dropout, self_loop=False, skip_connect=False)

    def forward(self,static_graph):
        # if self.use_static:
        static_graph = static_graph.to(self.gpu)
        static_graph.ndata['h'] = torch.cat(
            (self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
        self.statci_rgcn_layer(static_graph, [])
        static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
        static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
        self.h = static_emb
        # else:
        #     self.h = F.normalize(
        #         self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        #     static_emb = None

        return self.h, static_emb
