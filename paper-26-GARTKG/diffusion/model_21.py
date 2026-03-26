import torch.nn as nn
import torch
import math
from diffusion.difffu_21 import DiffuRec
import torch.nn.functional as F
import numpy as np
from diffusion.regcn import RGCNCell, RGCNBlockLayer
import torch as th
import random
import pickle
# from decoder import ConvTransE


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.weight_mlp = nn.Linear(hidden_size, hidden_size)
        self.bias_mlp = nn.Linear(hidden_size, hidden_size)
        self.variance_epsilon = eps

    def forward(self, x, weight=None):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if weight != None:
            return self.weight_mlp(weight) * x + self.bias_mlp(weight)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args, encoder_name, num_ents, num_rels,
                #  initial_h,
                 num_bases=-1,
                 num_hidden_layers=1, 
                 dropout=0, 
                 self_loop=False,
                 use_cuda=False, max_time=-1,
                 num_words=None,
                 num_static_rels=None):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size

        # nn.Linear(hidden_size)
        self.embed_dropout = nn.Dropout(args.dropout)
        self.time_max_len = max_time
        # 1 for padding object and 1 for condition subject and 1 for cls
        self.time_embeddings = nn.Embedding(
            self.time_max_len+1+1+1, self.emb_dim)
        # self.pos_embeddings = nn.Embedding(self.max_len+2, self.emb_dim)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.LayerNorm_static = LayerNorm(args.hidden_size, eps=1e-12)

        self.condition_linear = nn.Linear(3*args.hidden_size, args.hidden_size)

        self.seen_label_embedding = nn.Embedding(2, self.emb_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.max_len = args.max_len

        self.use_static = args.add_static_graph
        # self.layer_norm_gcn = args.layer_norm_gcn
        self.temperature_object = args.temperature_object
        self.pattern_noise_radio = args.pattern_noise_radio
        self.gpu = args.gpu
        self.num_rels = num_rels
        self.num_ents = num_ents
        # self.concat_con = args.concat_con
        # self.refinements_radio = args.refinements_radio
        # self.add_memory = args.add_memory
        # self.add_ood_loss_energe = args.add_ood_loss_energe
        # self.add_info_nce_loss = args.add_info_nce_loss
        self.loss = nn.MSELoss()
        
        self.linear_map=nn.Linear(self.emb_dim, self.emb_dim)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

        # self.seen_addition = args.seen_addition
        # self.kl_interst = args.kl_interst
        # self.add_frequence = args.add_frequence
        
        # self.initial_h=initial_h

        self.emb_rel = torch.nn.Parameter(torch.Tensor(num_rels*2, self.emb_dim),
                                          requires_grad=True).float()
        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, self.emb_dim),
                                          requires_grad=True).float()

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(
                num_words, self.emb_dim), requires_grad=True).float()
            torch.nn.init.trunc_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.emb_dim, self.emb_dim, num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)

        self.gate_weight = nn.Parameter(torch.Tensor(1, self.emb_dim))
        nn.init.constant_(self.gate_weight, 1)
        self.relation_cell_1 = nn.GRUCell(self.emb_dim*2, self.emb_dim)

        self.logistic_regression = nn.Linear(1, 2)
        self.frequence_linear = nn.Linear(1, self.emb_dim, bias=False)

        self.mlp_model = nn.Sequential(nn.Linear(
            self.emb_dim, self.emb_dim*2), nn.GELU(), nn.Linear(self.emb_dim*2, 2))

        self.weight_energy = torch.nn.Parameter(torch.Tensor(1, self.num_ents),
                                                requires_grad=True).float()
        torch.nn.init.uniform_(self.weight_energy)
        torch.nn.init.trunc_normal_(self.emb_rel)
        torch.nn.init.trunc_normal_(self.emb_ent)
        torch.nn.init.trunc_normal_(self.time_embeddings.weight)

        torch.nn.init.uniform_(self.frequence_linear.weight, 0, 1)

    def diffu_pre(self, item_rep, tag_emb, sr_embs, mask_seq, t, c, query_sub3=None):
        # seq_rep_diffu, item_rep_out = self.diffu(
        #     item_rep, tag_emb, sr_embs, mask_seq, t, c, query_sub3)
        # return seq_rep_diffu, item_rep_out
        seq_rep_diffu = self.diffu(
            item_rep, tag_emb, sr_embs, mask_seq, t, c, query_sub3)
        return seq_rep_diffu

    def reverse(self, model, tag, item_rep, noise_x_t, sr_embs, c, mask_seq, history_glist, triples, static_graph, use_cuda, args, mask=None, query_sub3=None):
        reverse_pre = self.diffu.reverse_p_sample(
            model, tag, item_rep, noise_x_t, c, sr_embs, mask_seq, history_glist, triples, static_graph, args, use_cuda, mask, query_sub3)
        return reverse_pre

    # def loss_diffu_ce(self, rep_diffu, labels,query_object3,history_tail_seq=None, one_hot_tail_seq=None,true_triples=None,mask_seq=None,emb_ent=None):
    def loss_diffu_ce(self, rep_diffu, labels, query_object3=None, true_triples=None, mask_seq=None, emb_ent=None):

        loss = 0
        scores = (rep_diffu) @ emb_ent[:].t() / \
            (math.sqrt(self.emb_dim)*self.temperature_object)
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        # """

        # return self.loss_ce(scores, labels.squeeze(-1)) + loss+self.loss(rep_diffu,query_object3)
        return self.loss_ce(scores, labels[:, 2].squeeze(-1)) + loss
        # return self.loss(noise,rep_diffu)
        # return self.loss(rep_diffu,noise)

    def regularization_memory(self):
        cos_mat = torch.matmul(
            self.emb_ent[:], self.emb_ent[:-2].transpose(1, 0))
        cos_sim = torch.norm((cos_mat - torch.eye(self.num_ents, self.num_ents).to(
            cos_mat.device))/math.sqrt(self.emb_dim))  # not real mean
        return cos_sim
    def gereration_feature(self,args,train_sample_num,model_diffpre,model_repre,last_history,last_output,static_graph_diffu,output_re,output,use_cuda,train_list,mode='c'):

        if train_sample_num > 1:
            if args.delete_his_prompt:
                if train_sample_num-1<args.reply_batch:
                    reply_time_index=list(range(train_sample_num))
                else:
                    reply_time_index=random.sample(list(range(train_sample_num)), args.reply_batch)
                current_query_history_triples=set()
                for i in reply_time_index:
                    current_query_history_triples.update(tuple(arr) for arr in train_list[i])
                if len(current_query_history_triples)> args.reply_batch_num:
                    current_query_history_triples=random.sample(current_query_history_triples, args.reply_batch_num)
            else:
                current_query_history_triples = set()
                if train_sample_num-2<args.reply_batch:
                    reply_time_index=list(range(train_sample_num-1))
                else:
                    reply_time_index=random.sample(list(range(train_sample_num-1)), args.reply_batch)
                
                for i in reply_time_index:
                    # if mode =='c':
                    related_history=pickle.load(open("../data/{}/history_snap_v2/snap_{}.pkl".format(args.dataset,i),"rb"))
                    if len(related_history) > 0:
                        for element in output_re:
                            if str(element[0]) in related_history.keys():
                                current_query_history_triples.update(
                                    related_history[str(element[0])])

        diffu_rep = None
        targets = None
        output_reply_triple = None
        sequence = None
        if args.diffuc:
            
            if  train_sample_num >= 2:
                output_reply_triple = torch.tensor(list(current_query_history_triples)).cuda(
                ) if use_cuda else torch.tensor(list(current_query_history_triples))
                
                if len(output_reply_triple)>0 and args.delete_feature_reply is False:
                    scores, diffu_rep, weights, t, _, ent_emb, noise, _, targets = model_diffpre(
                        sequence, output_reply_triple, args, False, use_cuda, static_graph=static_graph_diffu, ct=train_sample_num, model=model_repre, history_glist=last_history, triples=last_output)
                    diffu_rep = diffu_rep.detach()
                else:
                    diffu_rep=None
                    output_reply_triple=None
                    
        return output,diffu_rep,output_reply_triple

    # def diffu_rep_pre(self, rep_diffu,e_embs,history_tail_seq=None, one_hot_tail_seq=None):
    def diffu_rep_pre(self, rep_diffu, e_embs):

        scores = (rep_diffu[0]) @ e_embs[:-1].t() / \
            (math.sqrt(self.emb_dim)*self.temperature_object)
        return scores

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(
            torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  # not real mean
        return cos_sim
    
    def select_entity(self, tag, embeding, use_cuda):
        entity_all = torch.cat([tag[:, 0], tag[:, 2]])
        
        s_o_embeding = torch.cat([embeding[:, 0, :], embeding[:, 2, :]], dim=0)
        embeding_dict = dict()
       
        entity_all, indices = torch.unique(entity_all, return_inverse=True)
        ent_embeding = torch.zeros((self.num_ents, self.emb_dim)).to(
            self.gpu) if use_cuda else torch.zeros(1)
        
        for i, ent in enumerate(entity_all):
            ent_embeding[int(ent)] = s_o_embeding[indices == i].mean(dim=0)

        return entity_all, ent_embeding


    # def forward(self, sequence, tag, train_flag=True,use_cuda=True,history_tail_seq = None,seen_before_label=None,static_graph=None,ct=None):
    def forward(self, sequence, tag, args, train_flag=True, use_cuda=True, static_graph=None, ct=None, model=None, history_glist=None, triples=None,model_output=None, targets=None):
        # initial_h, _=self.initial_h(static_graph)
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat(
                (self.emb_ent, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = self.LayerNorm_static(static_emb)
            padding = torch.zeros_like(static_emb[[-1]]).to(self.gpu)
            initial_h = torch.concat([static_emb, padding], dim=0)
        else:
            initial_h = self.emb_ent
            padding = torch.zeros_like(self.emb_ent[[-1]]).to(self.gpu)
            initial_h = torch.concat([initial_h, padding], dim=0)
            static_emb = None
            
        if model_output is not None:
            entity_all, ent_embeding = self.select_entity(
                targets, model_output, use_cuda)
            initial_h[entity_all[:], :] = self.alpha*initial_h[entity_all[:],
                                                            :]+(1-self.alpha)*self.linear_map(ent_embeding[entity_all[:], :])
            # initial_h[entity_all[:], :] = self.alpha*initial_h[entity_all[:],
            #                                                 :]+(1-self.alpha)*ent_embeding[entity_all[:], :]
        mask_seq=None
        noise=None
        tagets = None
        
        if train_flag:  # 训练过程
            # B x H
            query_object3 = initial_h[tag[:, 2]].unsqueeze(1)  # 目标实体
            query_subject = initial_h[tag[:, 0]].unsqueeze(1)  # 目标实体
            query_relation = initial_h[tag[:, 1]].unsqueeze(1)  # 目标实体

            t, weights = self.diffu.schedule_sampler.sample(
                query_object3.shape[0], query_object3.device)  # t is sampled from schedule_sampler

            query_object_noise = self.diffu.q_sample(
                query_object3, t)

            inter_embeddings = torch.concat(
                [query_subject, query_relation, query_object_noise], dim=1)

            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(
                inter_embeddings))  # dropout first than layernorm

            c = query_subject+query_relation

            rep_diffu = self.diffu_pre(
                inter_embeddings_drop, inter_embeddings_drop[:, -1, :], initial_h, mask_seq, t, c)
            rep_diffu = rep_diffu[:,-1,:]
            
            rep_item, object_protype_gt = None, None
        else:

            query_subject = initial_h[tag[:, 0]].unsqueeze(1)  # 目标实体
            query_relation = initial_h[tag[:, 1]].unsqueeze(1)  # 目标实体
            query_object3 = th.randn_like(initial_h[tag[:, 0]].unsqueeze(1))

            inter_embeddings = torch.concat(
                [query_subject, query_relation, query_object3], dim=1)

            inter_embeddings_drop = self.LayerNorm(self.embed_dropout(
                inter_embeddings))  # dropout first than layernorm

            t, weights = self.diffu.schedule_sampler.sample(
                query_subject.shape[0], query_subject.device)
            c = query_subject+query_relation

            rep_diffu = self.reverse(model, tag, inter_embeddings_drop.detach().requires_grad_(True), inter_embeddings_drop[:, -1, :].detach(
            ).requires_grad_(True), initial_h.detach().requires_grad_(True), c, mask_seq, history_glist, triples, static_graph, use_cuda, args, [])

            rep_item, t, object_protype_gt = None, None, None

        scores = None

        # return scores, rep_diffu[0], rep_item, t, mask_seq,initial_h
        return scores, rep_diffu, rep_item, t, mask_seq, initial_h, noise, query_object3, tagets


def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre
