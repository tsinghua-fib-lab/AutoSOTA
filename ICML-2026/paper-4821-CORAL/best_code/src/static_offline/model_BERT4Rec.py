import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

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

class BERT4Rec(nn.Module):
    def __init__(self, opt, num_node):
        super(BERT4Rec, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node 
        
        # Token IDs: 0 (Pad), 1..N (Items), N+1 (Mask)
        self.mask_token = num_node + 1
        
        # Model config
        self.dim = getattr(opt, 'hiddenSize', 64)
        self.n_layers = getattr(opt, 'n_layers', 2)
        self.n_heads = getattr(opt, 'n_heads', 2)
        self.dropout = getattr(opt, 'dropout_local', 0.2)
        self.max_len = getattr(opt, 'max_len', 50)
        self.intermediate_size = getattr(opt, 'intermediate_size', 256)
        
        # Embedding
        self.item_embedding = nn.Embedding(self.num_node + 2, self.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len + 2, self.dim)
        self.emb_dropout = nn.Dropout(self.dropout)
        
        # Norm-First (Pre-LN) is generally more stable
        self.layer_norm = nn.LayerNorm(self.dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.n_heads,
            dim_feedforward=self.intermediate_size,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Output Head
        self.out = nn.Linear(self.dim, self.num_node + 1) 
        
        # Loss
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, inputs, mask=None):
        # inputs: [Batch, Seq_Len]
        seq_length = inputs.size(1)
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs)
        
        item_emb = self.item_embedding(inputs)
        pos_emb = self.position_embedding(position_ids)
        
        x = item_emb + pos_emb
        x = self.emb_dropout(x)
        
        # Padding Mask
        if mask is None:
            padding_mask = (inputs == 0)
        else:
            padding_mask = (mask == 0)
        
        # Transformer Forward
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Apply final norm (needed for Pre-LN architecture)
        output = self.layer_norm(output)
        
        return output

    def compute_scores(self, hidden):
        # 1. Weight Sharing: Use item embedding weights
        test_item_emb = self.item_embedding.weight 
        
        # 2. Dot Product
        logits = torch.matmul(hidden, test_item_emb.transpose(0, 1))
        return logits


def train_test(model, train_loader, test_loader, topk=[10, 100], mode='train'):
    
    # --- TRAINING PHASE ---
    if mode == 'train':
        print(f'Start training: {datetime.datetime.now()}')
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc='Training'):
            inputs = trans_to_cuda(batch['input_ids'])
            labels = trans_to_cuda(batch['labels'])
            
            model.optimizer.zero_grad()
            
            hidden = model(inputs)
            scores = model.compute_scores(hidden)
            scores[:, 0] = -np.inf # Mask padding
            
            scores_flat = scores.view(-1, scores.size(-1))
            labels_flat = labels.view(-1)
            
            loss = model.loss_function(scores_flat, labels_flat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            model.optimizer.step()
            total_loss += loss.item()
            
        print('\tLoss:\t%.3f' % total_loss)
        return [], []

    # --- EVALUATION PHASE ---
    print(f'Start predicting: {datetime.datetime.now()}')
    model.eval()
    
    k_list = topk
    hit_all = [[] for _ in k_list]
    mrr_all = [[] for _ in k_list]
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            inputs = trans_to_cuda(batch['input_ids'])
            targets = batch['target'].numpy()
            
            # Standard Inference
            hidden = model(inputs)
            
            mask_token_id = model.mask_token
            mask_positions = (inputs == mask_token_id)
            
            # Locate the mask token to predict
            if not mask_positions.any():
                 # Fallback: if no mask, use last non-zero token
                 seq_lens = (inputs != 0).sum(dim=1) - 1
                 batch_indices = torch.arange(inputs.size(0), device=inputs.device)
                 last_hidden = hidden[batch_indices, seq_lens]
            else:
                 rows = torch.arange(inputs.size(0), device=inputs.device)
                 cols = mask_positions.float().argmax(dim=1)
                 last_hidden = hidden[rows, cols]

            final_scores = model.compute_scores(last_hidden)
            
            # Metrics Calculation
            scores = trans_to_cpu(final_scores).detach().numpy()
            
            # Mask out padding and mask_token from prediction candidates
            scores[:, 0] = -np.inf
            if scores.shape[1] > model.mask_token:
                scores[:, model.mask_token] = -np.inf
            
            max_k = max(k_list)
            scores_tensor = torch.tensor(scores)
            top_scores = torch.topk(scores_tensor, max_k, dim=1)
            pred_indices = top_scores.indices.numpy()
            
            for i in range(len(targets)):
                target = targets[i]
                pred = pred_indices[i]
                
                for k_idx, k in enumerate(k_list):
                    sub_pred = pred[:k]
                    if target in sub_pred:
                        hit_all[k_idx].append(1)
                        rank = np.where(sub_pred == target)[0][0]
                        mrr_all[k_idx].append(1.0 / (rank + 1))
                    else:
                        hit_all[k_idx].append(0)
                        mrr_all[k_idx].append(0.0)
    
    hit_avg = [np.mean(h) * 100 for h in hit_all]
    mrr_avg = [np.mean(m) * 100 for m in mrr_all]
    
    return hit_avg, mrr_avg