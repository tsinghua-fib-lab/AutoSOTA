import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import sys


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class SASRec(nn.Module):
    def __init__(self, opt, num_node):
        super(SASRec, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node 
        self.dim = opt.hiddenSize
        self.n_layers = getattr(opt, 'n_layers', 2)
        self.n_heads = getattr(opt, 'n_heads', 2)
        self.dropout = getattr(opt, 'dropout_local', 0.2)
        self.max_len = getattr(opt, 'max_len', 200)
        
        self.item_embedding = nn.Embedding(self.num_node + 1, self.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.dim)
        self.emb_dropout = nn.Dropout(self.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.n_heads,
            dim_feedforward=self.dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.layer_norm = nn.LayerNorm(self.dim)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=opt.lr_dc_step, 
            gamma=opt.lr_dc
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, inputs, attention_mask):
        seq_length = inputs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs)
        
        item_emb = self.item_embedding(inputs)
        pos_emb = self.position_embedding(position_ids)
        
        x = item_emb + pos_emb
        x = self.emb_dropout(x)
        
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=inputs.device), diagonal=1).bool()
        padding_mask = (inputs == 0)
        
        output = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        return output

    def compute_scores(self, hidden):
        test_item_emb = self.item_embedding.weight 
        scores = torch.matmul(hidden, test_item_emb.transpose(0, 1)) 
        scores[:, 0] = -np.inf 
        return scores

def forward(model, data, mode='train'):
    if mode == 'train':
        inputs, mask, targets = data[0], data[1], data[2]
        inputs = trans_to_cuda(inputs).long()
        targets = trans_to_cuda(targets).long()
        
        hidden = model(inputs, attention_mask=None) 
        scores = model.compute_scores(hidden)
        
        scores = scores.view(-1, scores.size(-1))
        targets = targets.view(-1)
        return targets, scores
    else:
        inputs = data[0]
        inputs = trans_to_cuda(inputs).long()
        hidden = model(inputs, attention_mask=None) 
        
        seq_lens = (inputs != 0).sum(dim=1) - 1 
        batch_size = hidden.size(0)
        last_hidden = hidden[torch.arange(batch_size, device=hidden.device), seq_lens] 
        
        scores = model.compute_scores(last_hidden) 
        return scores


def train_test(model, train_data, test_data, topk=[20], mode='train', coral_agent=None):
    if mode == 'train':
        # [Training logic remains identical]
        print('start training: ', datetime.datetime.now())
        model.train()
        total_loss = 0.0
        train_loader = torch.utils.data.DataLoader(
            train_data, num_workers=4, batch_size=model.batch_size, shuffle=True, pin_memory=True
        )
        
        for data in tqdm(train_loader):
            model.optimizer.zero_grad()
            targets, scores = forward(model, data, mode='train')
            loss = model.loss_function(scores, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            model.optimizer.step()
            total_loss += loss.item()
        
        print('\tLoss:\t%.3f' % total_loss)
        model.scheduler.step()
    
        print('start predicting: ', datetime.datetime.now())
        model.eval()
        test_loader = torch.utils.data.DataLoader(
            test_data, num_workers=4, batch_size=model.batch_size, shuffle=False, pin_memory=True
        )
        
        k_list = topk
        hit_all = [[] for _ in k_list]
        mrr_all = [[] for _ in k_list]
        
        with torch.no_grad():
            for data in test_loader:
                # Unpack all data
                inputs = data[0]
                mask = data[1]
                targets = data[2]
                cat_inputs = data[3]
                rate_inputs = data[4]
                user_ids = data[5]

                # 1. Base SASRec Scores (Run Forward Pass)
                scores = forward(model, data, mode='test') 
                scores = trans_to_cpu(scores).detach().numpy()
                scores[:, 0] = -np.inf 
                
                targets = targets.cpu().numpy()
                
                # Default Logic: Get Top K purely by score
                # We fetch more candidates (rerank_k) to allow re-shuffling
                rerank_k = 200 
                
                # argpartition puts largest k elements at the end
                ind = np.argpartition(scores, -rerank_k, axis=1)[:, -rerank_k:]
                
                # Sort these top candidates by SASRec score descending
                row_indices = np.arange(scores.shape[0])[:, None]
                top_scores = scores[row_indices, ind]
                sort_indices = np.argsort(-top_scores, axis=1)
                sorted_ind = ind[row_indices, sort_indices] # [B, rerank_k]
                
                # Default final ranks is just SASRec order
                final_ranks = sorted_ind


                if coral_agent is not None:
                    batch_b = inputs.shape[0]
                    reranked_batch = []
                    
                    for b in range(batch_b):
                        uid = user_ids[b].item()
                        u_input = inputs[b].numpy()
                        u_cat = cat_inputs[b].numpy()
                        u_rate = rate_inputs[b].numpy()
                        
                        # 1. Rebuild History 
                        coral_agent.load_history_from_tensors(uid, u_input, u_cat, u_rate)
                        valid_len = np.count_nonzero(u_input)
                        current_t = valid_len 
                        
                        # 2. Get Category Policy Scores (Eq. 11)
                        # Calculates: Utility + Exploration - Lambda * Risk
                        cat_policy_scores, D_t = coral_agent.get_category_policy_scores(uid, current_t)
                        
                        # 3. Sort Categories by Safety/Policy Score
                        sorted_cats = np.argsort(-cat_policy_scores)
                        
                        # 4. Re-bucket items based on Category Priority
                        candidates = sorted_ind[b] 
                        cat_buckets = {}
                        
                        for item in candidates:
                            c = coral_agent.item_to_cat.get(item, 0)
                            if c not in cat_buckets:
                                cat_buckets[c] = []
                            cat_buckets[c].append(item)
                        
                        new_order = []
                        # Fill list starting from best categories
                        for c in sorted_cats:
                            if c in cat_buckets:
                                new_order.extend(cat_buckets[c])
                                del cat_buckets[c]
                        
                        # Append remaining items (if any, e.g. unknown cat)
                        for remaining in cat_buckets.values():
                            new_order.extend(remaining)
                        
                        reranked_batch.append(np.array(new_order))
                    
                    # Overwrite final_ranks with CORAL's order
                    final_ranks = np.array(reranked_batch)

                for i in range(len(targets)):
                    target = targets[i]
                    pred = final_ranks[i]
                    
                    for k_idx, k in enumerate(k_list):
                        if k > len(pred):
                            pred_k = pred
                        else:
                            pred_k = pred[:k]
                            
                        if target in pred_k:
                            hit_all[k_idx].append(1)
                            rank_idx = np.where(pred_k == target)[0][0]
                            mrr_all[k_idx].append(1.0 / (rank_idx + 1))
                        else:
                            hit_all[k_idx].append(0)
                            mrr_all[k_idx].append(0.0)
        
        hit_avg = [np.mean(h) * 100 for h in hit_all]
        mrr_avg = [np.mean(m) * 100 for m in mrr_all]
        
        return hit_avg, mrr_avg, None