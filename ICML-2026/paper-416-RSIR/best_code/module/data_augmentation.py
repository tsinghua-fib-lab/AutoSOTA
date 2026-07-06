import math
import os
import torch
import random
import faiss
import copy
import pickle
import scipy.sparse as sp
import numpy as np
from typing import Optional, Union
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from module import functional as recfn
from module import layers
from data import dataset
from tqdm import tqdm

class Item_Crop(torch.nn.Module):

    def __init__(self, tao=0.2):
        super().__init__()
        self.tao = tao

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)

        croped_sequences = []
        croped_seq_lens = torch.zeros(batch_size, dtype=seq_lens.dtype, device=seq_lens.device)
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            sub_len = max(1, int(self.tao * seq_len))
            croped_seq_lens[i] = sub_len
            start_index = torch.randint(low = 0, high = seq_len - sub_len + 1, size=(1,)).item()
            croped_sequence = sequences[i][start_index : start_index + sub_len]
            croped_sequences.append(croped_sequence)

        return pad_sequence(croped_sequences, batch_first=True), croped_seq_lens


class Item_Mask(torch.nn.Module):

    def __init__(self, mask_id, gamma=0.7):
        super().__init__()
        self.gamma = gamma
        self.mask_id = mask_id

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        copied_sequence = copy.deepcopy(sequences)

        for i in range(batch_size):
            seq_len = seq_lens[i]
            sub_len = int(self.gamma * seq_len)
            mask_idx = np.random.choice(seq_len.item(), size=sub_len, replace=False).astype(np.int64)
            copied_sequence[i][mask_idx] = self.mask_id

        return copied_sequence, seq_lens


class Item_Reorder(torch.nn.Module):

    def __init__(self, beta=0.2) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)

        reordered_sequences = []
        for i in range(batch_size):
            seq = sequences[i]
            seq_len = seq_lens[i]
            sub_len = int(self.beta * seq_len)
            start_index = random.randint(a = 0, b = seq_len - sub_len) # [a, b]
            reordered_index = list(range(sub_len))
            random.shuffle(reordered_index)
            sub_seq = seq[start_index : start_index + sub_len][reordered_index]
            reordered_sequences.append(torch.cat([seq[ : start_index], sub_seq, seq[start_index + sub_len : ]]))

        return torch.stack(reordered_sequences, dim=0), seq_lens

class Item_Random(torch.nn.Module):

    def __init__(self, mask_id, tao=0.2, gamma=0.7, beta=0.2) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.augmentation_methods = [Item_Crop(tao=tao), Item_Mask(mask_id, gamma=gamma), Item_Reorder(beta=beta)]

    def forward(self, sequences, seq_lens):
        return self.augmentation_methods[random.randint(0, len(self.augmentation_methods) - 1)](sequences, seq_lens)

class Item_Substitute(torch.nn.Module):

    def __init__(self, item_similarity_model, substitute_rate=0.1) -> None:
        super().__init__()
        if isinstance(item_similarity_model, list):
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate

    def _ensemble_sim_models(self, top_1_one, top_1_two):
        indicator = top_1_one[1] >= top_1_two[1]
        top_1 = torch.zeros_like(top_1_one[0])
        top_1[indicator] = top_1_one[0][indicator]
        top_1[~indicator] = top_1_two[0][~indicator]
        return top_1

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)

        substituted_sequences = []
        for i in range(batch_size):
            seq = sequences[i]
            seq_len = seq_lens[i]
            sub_len = max(1, int(self.substitute_rate * seq_len))
            substitute_idx = np.random.choice(seq_len.item(), size=sub_len, replace=False)
            substituted_sequence = copy.deepcopy(seq)
            selected_items = substituted_sequence[substitute_idx]

            if self.ensemble:
                top_1_one = self.item_sim_model_1(selected_items, with_score=True)
                top_1_two = self.item_sim_model_2(selected_items, with_score=True)
                substitute_items = self._ensemble_sim_models(top_1_one, top_1_two)
                substituted_sequence[substitute_idx] = substitute_items
            else:
                substitute_items = self.item_similarity_model(selected_items, with_score=False)
                substituted_sequence[substitute_idx] = substitute_items

            substituted_sequences.append(substituted_sequence)

        return torch.stack(substituted_sequences, dim=0), seq_lens


class Item_Insert(torch.nn.Module):

    def __init__(self, item_similarity_model, insert_rate=0.4) -> None:
        super().__init__()
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate

    def _ensemble_sim_models(self, top_1_one, top_1_two):
        if top_1_one[1] >= top_1_two[1]:
            return top_1_one[0]
        else:
            return top_1_two[0]

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)

        inserted_sequences = []
        new_seq_lens = torch.zeros_like(seq_lens, device=seq_lens.device)
        for i in range(batch_size):
            seq = sequences[i]
            seq_len = seq_lens[i]
            sub_len = max(1, int(self.insert_rate * seq_len))
            insert_idx = np.random.choice(seq_len.item(), size=sub_len, replace=False)
            inserted_sequence = []

            new_seq_lens[i] = seq_len + sub_len
            for j in range(seq_len):
                if j in insert_idx:
                    # only support top 1.
                    if self.ensemble:
                        top_1_one = self.item_sim_model_1(seq[j].item(), with_score=True)
                        top_1_two = self.item_sim_model_2(seq[j].item(), with_score=True)
                        insert_item = self._ensemble_sim_models(top_1_one, top_1_two)
                        inserted_sequence.append(insert_item)
                    else:
                        insert_item = self.item_similarity_model(seq[j].item(), with_score=False)
                        inserted_sequence.append(insert_item)
                inserted_sequence.append(seq[j].item())

            inserted_sequences.append(torch.tensor(inserted_sequence, device=seq.device))

        return pad_sequence(inserted_sequences, batch_first=True), new_seq_lens


class Random_Augmentation(torch.nn.Module):

    def __init__(self, augment_threshold, short_seq_aug_methods:list, long_seq_aug_methods:list) -> None:
        super().__init__()
        self.augment_threshold = augment_threshold
        self.short_seq_aug_methods = short_seq_aug_methods
        self.long_seq_aug_methods = long_seq_aug_methods

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)

        new_seqs = []
        new_seq_lens = []
        for i in range(batch_size):
            seq = sequences[i].unsqueeze(0) # [L] -> [B(1), L]
            seq_len = seq_lens[i].unsqueeze(0) #[] -> [B(1)]

            if seq_len > self.augment_threshold:
                aug_method = random.choice(self.long_seq_aug_methods)
            else:
                aug_method = random.choice(self.short_seq_aug_methods)

            seq_, seq_len_ = aug_method(seq, seq_len)
            new_seqs.append(seq_.squeeze(0)) # [B(1), L] -> [L]
            new_seq_lens.append(seq_len_) # [B(1)]

        new_seqs = pad_sequence(new_seqs, batch_first=True)
        new_seq_lens = torch.cat(new_seq_lens, dim=0)

        return new_seqs, new_seq_lens


class EdgeDropout(torch.nn.Module):
    """
    Drop some edges in the graph in sparse COO or dgl format. It is used in GNN-based models.
    It is a out-place operation.
    Parameters:
        dropout_prob(float): probability of a edge to be droped.
    """
    def __init__(self, dropout_prob: float, num_users: int, num_items: int) -> None:
        super().__init__()
        self.keep_prob = 1.0 - dropout_prob
        self.edge_dropout_dgl = None

    def forward(self, X):
        """
        Returns:
            (torch.Tensor or dgl.DGLGraph): the graph after dropout in sparse COO or dgl.DGLGraph format.
        """
        if not self.training:
            return X
        if isinstance(X, torch.Tensor) and X.is_sparse and (not X.is_sparse_csr):
            X = X.coalesce()
            random_tensor = torch.rand(X._nnz(), device=X.device) + self.keep_prob
            random_tensor = torch.floor(random_tensor).type(torch.bool)
            indices = X.indices()[:, random_tensor]
            values = X.values()[random_tensor] * (1.0 / self.keep_prob)
            return torch.sparse_coo_tensor(indices, values, X.shape, dtype=X.dtype)
        elif 'dgl' in str(type(X)):
            import dgl
            if self.edge_dropout_dgl == None:
                self.edge_dropout_dgl = dgl.DropEdge(p=1.0 - self.keep_prob)
            new_X = copy.deepcopy(X)
            new_X = self.edge_dropout_dgl(new_X)
            return new_X
        else:
            raise ValueError(f"EdgeDropout doesn't support graph with type {type(X)}")

class NodeDropout(torch.nn.Module):
    """
    Drop some nodes and the edges connected to them in the graph in sparse COO or dgl format.
    It is a out-place operation.
    Parameters:
        dropout_prob(float): probability of a node to be droped.
    """
    def __init__(self, dropout_prob: float, num_users: int, num_items: int) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dropout_prob = dropout_prob

    def forward(self, X):
        if not self.training:
            return X
        nodes_flag = torch.tensor([False] * (self.num_users + self.num_items), device=X.device)
        user_drop_indices = torch.randperm(self.num_users)[:int(self.num_users * self.dropout_prob)]
        item_drop_indices = torch.randperm(self.num_items)[:int(self.num_items * self.dropout_prob)]
        nodes_flag[user_drop_indices] = True # if True, the node needs to be dropped.
        nodes_flag[item_drop_indices + self.num_users] = True
        if isinstance(X, torch.Tensor) and X.is_sparse and (not X.is_sparse_csr):
            nodes_flag = nodes_flag.to(X.device)
            nodes_indices = torch.arange(self.num_users + self.num_items, device=X.device, dtype=torch.long)[~nodes_flag]
            diag = torch.sparse_coo_tensor(torch.stack([nodes_indices, nodes_indices]), \
                torch.ones(len(nodes_indices), device=X.device, dtype=X.dtype), \
                size=(self.num_users+self.num_items, self.num_users+self.num_items))
            new_X = torch.sparse.mm(X, diag)
            new_X = torch.sparse.mm(diag, new_X)
            return new_X
        elif 'dgl' in str(type(X)):
            def edges_with_droped_nodes(edges):
                src, dst, _ = edges.edges()
                return torch.logical_or(nodes_flag[src], nodes_flag[dst])
            droped_edges = X.filter_edges(edges_with_droped_nodes)
            new_X = copy.deepcopy(X)
            new_X.remove_edges(droped_edges)
            return new_X
        else:
            raise ValueError(f"NodeDropout doesn't support graph with type {type(X)}")


class InfoNCELoss(torch.nn.Module):
    '''
    Parameters:
    neg_type(str): 'batch_both', 'batch_single', 'all'
    '''

    def __init__(self, temperature:float=1.0, sim_method:str='inner_product', neg_type:str='batch_both') -> None:
        super().__init__()
        self.temperature = temperature
        self.sim_method = sim_method
        self.neg_type = neg_type

    def forward(self, augmented_rep_i: torch.Tensor, augmented_rep_j: torch.Tensor, \
        instance_labels=None, all_reps: Optional[torch.Tensor]=None, reduce=True):
        # augmented_rep_i, augmented_rep_j : [B, D], all_reps : [N, D]
        # labels: [B]
        # negative items: 2*N - 1
        if self.neg_type == 'batch_both':
            assert all_reps == None, "all_reps is used when the negative strategy is 'all'."
            batch_size = augmented_rep_i.size(0)
            if self.sim_method == 'inner_product':
                sim_ii = torch.matmul(augmented_rep_i, augmented_rep_i.T) / self.temperature # [B, B]
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature
            elif self.sim_method == 'cosine':
                augmented_rep_i = F.normalize(augmented_rep_i, p=2, dim=-1)
                augmented_rep_j = F.normalize(augmented_rep_j, p=2, dim=-1)
                sim_ii = torch.matmul(augmented_rep_i, augmented_rep_i.T) / self.temperature
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature

            if instance_labels is not None:
                '''
                do de-noise as ICLRec, if data_1 and data_2 have the same label,
                then (data_1_i, data_2_i) and (data_1_i, data_2_j) won't be treated as negative samples.
                '''
                # instance_labels: [batch_size]
                mask = torch.eq(instance_labels.unsqueeze(-1), instance_labels) # [B, B]
                sim_ii[mask == 1] = float('-inf')
                mask = mask.fill_diagonal_(False)
                sim_ij[mask == 1] = float('-inf')
            else:
                mask = torch.eye(batch_size, dtype=torch.long).to(augmented_rep_i.device)
                sim_ii[mask == 1] = float('-inf')

            logits = torch.cat([sim_ij, sim_ii], dim=-1) # [B, 2 * B]
            labels = torch.arange(batch_size, dtype=torch.long, device=augmented_rep_i.device) # [B]
            if reduce:
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels, reduction='none') / batch_size
            return loss

        elif self.neg_type == 'batch_single':
            assert all_reps == None, "all_reps is used when the negative strategy is 'all'."
            batch_size = augmented_rep_i.size(0)
            if self.sim_method == 'inner_product':
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature # [B, B]
            elif self.sim_method == 'cosine':
                augmented_rep_i = F.normalize(augmented_rep_i, p=2, dim=-1)
                augmented_rep_j = F.normalize(augmented_rep_j, p=2, dim=-1)
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature # [B, B]
            if instance_labels is not None:
                '''
                do de-noise as ICLRec, if data_1 and data_2 have the same label,
                then (data_1_i, data_2_i) and (data_1_i, data_2_j) won't be treated as negative samples.
                '''
                # instance_labels: [batch_size]
                mask = torch.eq(instance_labels.unsqueeze(-1), instance_labels) # [B, B]
                mask = mask.fill_diagonal_(False)
                sim_ij[mask == 1] = float('-inf')

            labels = torch.arange(batch_size, dtype=torch.long, device=augmented_rep_i.device) # [B]
            if reduce:
                loss = F.cross_entropy(sim_ij, labels)
            else:
                loss = F.cross_entropy(sim_ij, labels, reduction='none') / batch_size
            return loss

        elif self.neg_type == 'all':
            # In graph models, the negative items usually are all items in the dataset.
            # augmented_rep_i : [B, D], augmented_rep_j : [B, D], all_reps: [N, D]
            # negative items : N
            assert all_reps != None, "all_reps shouldn't be None."
            assert instance_labels == None, "instance_labels is used when the negative strategy is 'batch'."

            batch_size = augmented_rep_i.size(0)
            if self.sim_method == 'inner_product':
                sim_ij = torch.matmul(augmented_rep_i, all_reps.T) / self.temperature # [B, N]
                sim_ii = (augmented_rep_i * augmented_rep_j).sum(dim=-1) / self.temperature # [B]
            elif self.sim_method == 'cosine':
                augmented_rep_i = F.normalize(augmented_rep_i, p=2, dim=-1)
                augmented_rep_j = F.normalize(augmented_rep_j, p=2, dim=-1)
                all_reps = F.normalize(all_reps, p=2, dim=-1)
                sim_ij = torch.matmul(augmented_rep_i, all_reps.T) / self.temperature # [B, N]
                sim_ii = (augmented_rep_i * augmented_rep_j).sum(dim=-1) / self.temperature # [B]

            loss = torch.mean(torch.logsumexp(sim_ij, dim=-1) - sim_ii) # [B, N] -> [B] -> []

            return loss
        else:
            raise ValueError(f'{self.neg_type} is not supported, neg_type should be "batch" or "all".')

# Graph augmentation models
class SGLAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        if self.config['aug_type'] == 'ED':
            self.augmentation = EdgeDropout(self.config['ssl_ratio'], self.num_users, self.num_items)
        elif self.config['aug_type'] == 'ND':
            self.augmentation = NodeDropout(self.config['ssl_ratio'], self.num_users, self.num_items)
        elif self.config['aug_type'] == 'RW':
            self.augmentation = EdgeDropout(self.config['ssl_ratio'], self.num_users, self.num_items)
        self.InfoNCELoss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='cosine', neg_type='all')

    def get_gnn_embeddings(self, user_emb, item_emb, adj_mat, gnn_net):
        adj_mat = adj_mat.to(user_emb.weight.device)
        if self.config['aug_type'] in ['ED', 'ND']:
            adj_mat_aug = self.augmentation(adj_mat)
        elif self.config['aug_type'] == 'RW':
            adj_mat_aug = []
            for i in range(len(gnn_net.combiners)):
                adj_mat_aug.append(self.augmentation(adj_mat))
        # [num_users + num_items, dim]
        embeddings = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        # {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]}
        all_embeddings = gnn_net(adj_mat_aug, embeddings)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        # [num_users + num_items, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        return torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

    def forward(self, batch, user_emb:torch.nn.Embedding, item_emb:torch.nn.Embedding, adj_mat, gnn_net:torch.nn.Module):
        output_dict = {}
        user_all_vec1, item_all_vec1 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)
        user_all_vec2, item_all_vec2 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)

        user_cl_loss = \
            self.InfoNCELoss_fn(user_all_vec1[batch[self.fuid]], user_all_vec2[batch[self.fuid]], all_reps=user_all_vec2[1:])
        item_cl_loss = \
            self.InfoNCELoss_fn(item_all_vec1[batch[self.fiid]], item_all_vec2[batch[self.fiid]], all_reps=item_all_vec2[1:])

        output_dict['cl_loss'] = user_cl_loss + item_cl_loss

        return output_dict


class NCLAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.hyper_layers = self.config['hyper_layers']
        self.InfoNCELoss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='cosine', neg_type='all')

        self.cluster = faiss.Kmeans(d=self.config['embed_dim'], k=self.config['num_clusters'], gpu=False)

        self.user_centroids, self.user_2cluster = None, None
        self.item_centroids, self.item_2cluster = None, None

    def forward(self, batch, all_embeddings_list: list):
        # all_embeddings_list : {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]}
        output_dict = {}
        center_embeddings = all_embeddings_list[0]
        context_embeddings = all_embeddings_list[self.hyper_layers * 2]
        user_center_embeddings, item_center_embeddings = torch.split(center_embeddings, [self.num_users, self.num_items], dim=0)
        user_context_embeddings, item_context_embeddings = torch.split(context_embeddings, [self.num_users, self.num_items], dim=0)

        # structure
        user_structure_loss = \
            self.InfoNCELoss_fn(user_context_embeddings[batch[self.fuid]], user_center_embeddings[batch[self.fuid]], all_reps=user_center_embeddings[1:])
        item_structure_loss = \
            self.InfoNCELoss_fn(item_context_embeddings[batch[self.fiid]], item_center_embeddings[batch[self.fiid]], all_reps=item_center_embeddings[1:])
        output_dict['structure_cl_loss'] = user_structure_loss + self.config['alpha'] * item_structure_loss

        # semantic
        user2cluster = self.user_2cluster[batch[self.fuid]]
        user_proto_loss = \
            self.InfoNCELoss_fn(user_center_embeddings[batch[self.fuid]], self.user_centroids[user2cluster], all_reps=self.user_centroids)
        item2cluster = self.item_2cluster[batch[self.fiid]]
        item_proto_loss = \
            self.InfoNCELoss_fn(item_center_embeddings[batch[self.fiid]], self.item_centroids[item2cluster], all_reps=self.item_centroids)
        output_dict['semantic_cl_loss'] = user_proto_loss + self.config['alpha'] * item_proto_loss

        return output_dict

    @torch.no_grad()
    def e_step(self, user_emb: torch.nn.Embedding, item_emb: torch.nn.Embedding):
        user_embeddings = user_emb.weight.detach().cpu().numpy()
        item_embeddings = item_emb.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)
        self.user_centroids = self.user_centroids.to(user_emb.weight.device)
        self.user_2cluster = self.user_2cluster.to(user_emb.weight.device)
        self.item_centroids = self.item_centroids.to(item_emb.weight.device)
        self.item_2cluster = self.item_2cluster.to(item_emb.weight.device)

    @torch.no_grad()
    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        # x : [num_users, dim]
        self.cluster.train(x[1:])
        cluster_cents = self.cluster.centroids

        _, I = self.cluster.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.from_numpy(cluster_cents)

        node2cluster = torch.LongTensor(I).squeeze(dim=-1) # [num_users] or [num_items]
        return centroids, node2cluster


class SimGCLAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.fuid = train_data.fuid
        self.fiid = train_data.fiid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.InfoNCELoss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='cosine', neg_type=self.config['cl_neg_type'])

    def get_gnn_embeddings(self, user_emb, item_emb, adj_mat, gnn_net):
        adj_mat = adj_mat.to(user_emb.weight.device)
        # [num_users + num_items, dim]
        embeddings = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        # {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]}
        all_embeddings = gnn_net(adj_mat, embeddings, perturbed=True)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        # [num_users + num_items, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        return torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

    def forward(self, batch, user_emb:torch.nn.Embedding, item_emb:torch.nn.Embedding, adj_mat, gnn_net:torch.nn.Module):
        output_dict = {}
        device = user_emb.weight.device
        u_idx = torch.unique(batch[self.fuid]).to(device)
        i_idx = torch.unique(batch[self.fiid]).to(device)
        user_all_vec1, item_all_vec1 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)
        user_all_vec2, item_all_vec2 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)

        if self.config['cl_neg_type'] == 'all':
            user_cl_loss = \
                self.InfoNCELoss_fn(user_all_vec1[u_idx], user_all_vec2[u_idx], all_reps=user_all_vec2[1:])
        else:
            user_cl_loss = \
                self.InfoNCELoss_fn(user_all_vec1[u_idx], user_all_vec2[u_idx])
        if self.config['cl_neg_type'] == 'all':
            item_cl_loss = \
                self.InfoNCELoss_fn(item_all_vec1[i_idx], item_all_vec2[i_idx], all_reps=item_all_vec2[1:])
        else:
            item_cl_loss = \
                self.InfoNCELoss_fn(item_all_vec1[i_idx], item_all_vec2[i_idx])

        output_dict['cl_loss'] = user_cl_loss + item_cl_loss

        return output_dict

# Sequence augmentation models
class CL4SRecAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        if self.config['augment_type'] == 'item_crop':
            self.augmentation = Item_Crop(self.config['tau'])
        elif self.config['augment_type'] == 'item_mask':
            self.augmentation = Item_Mask(mask_id=train_data.num_items, gamma=self.config['gamma'])
        elif self.config['augment_type'] == 'item_reorder':
            self.augmentation = Item_Reorder(beta=self.config['beta'])
        elif self.config['augment_type'] == 'item_random':
            self.augmentation = Item_Random(
                mask_id=train_data.num_items,
                tao=self.config['tau'],
                gamma=self.config['gamma'],
                beta=self.config['beta'],
            )
        else:
            raise ValueError(f"augmentation type: '{self.config['augment_type']}' is invalided")
        self.InfoNCE_loss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='inner_product', neg_type='batch_both')

    def forward(self, batch, query_encoder:torch.nn.Module, reduce=True):
        output_dict = {}
        seq_augmented_i, seq_augmented_i_len = self.augmentation(batch['in_' + self.fiid], batch['seqlen']) # seq: [B, L] seq_len : [B]
        seq_augmented_j, seq_augmented_j_len = self.augmentation(batch['in_' + self.fiid], batch['seqlen'])

        seq_augmented_i_out = query_encoder({"in_" + self.fiid: seq_augmented_i, "seqlen": seq_augmented_i_len},\
            need_pooling=False) # [B, L, D]
        seq_augmented_i_out = recfn.seq_pooling_function(seq_augmented_i_out, seq_augmented_i_len, pooling_type='mean') # [B, D]

        seq_augmented_j_out = query_encoder({"in_" + self.fiid: seq_augmented_j, "seqlen" : seq_augmented_j_len},\
            need_pooling=False) # [B, L, D]
        seq_augmented_j_out = recfn.seq_pooling_function(seq_augmented_j_out, seq_augmented_j_len, pooling_type='mean') # [B, D]

        mask_idx = batch['seqlen'] == 1
        seq_augmented_i_out = seq_augmented_i_out[~mask_idx]
        seq_augmented_j_out = seq_augmented_j_out[~mask_idx]

        cl_loss = self.InfoNCE_loss_fn(seq_augmented_i_out, seq_augmented_j_out, reduce=reduce)
        output_dict['cl_loss'] = cl_loss
        return output_dict

class Cluster(torch.nn.Module):
    
    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        self.cluster = faiss.Kmeans(d=self.config['embed_dim'], k=256, gpu=False)

        self.centroids = None

    @torch.no_grad()
    def train_kmeans(self, query_encoder, trainloader, device):
        
        # clustering
        training_data = []
        for batch_idx, batch in enumerate(trainloader):
            seq_out = query_encoder(batch, need_pooling=False)
            seq_out = recfn.seq_pooling_function(seq_out, batch['seqlen'], pooling_type='mean')
            training_data.append(seq_out) # [B, D]
        training_data = torch.cat(training_data, dim=0).cpu().numpy()

        res = faiss.StandardGpuResources()
        d = 64  # 输入向量维度
        M = 8   # 每个子空间的聚类中心数量
        nbits = 8  # 每个子向量的比特数
        index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引移动到GPU设备
        # 使用训练数据训练PQ索引
        index.train(training_data)

        # 添加向量到索引
        index.add(training_data)
        # 搜索相似向量
        k = 5   # 返回的最相似向量数量
        D, I = index.search(training_data, k)
        return D, I
        # self.cluster.train(training_data.cpu().numpy())
        # self.centroids = torch.from_numpy(self.cluster.centroids).to(device)
        # cluster_assignment = torch.tensor(self.cluster.assign(training_data.cpu().numpy())[1])
        # _, cluster_count = torch.unique(cluster_assignment, return_counts=True)
        # weight = 1 / cluster_count
        # N = len(trainloader.dataset)
        # from torch.distributions import Bernoulli
        # weight = 1 / cluster_count
        # sample_weight = torch.clamp_max(weight[cluster_assignment] * (N / 128), max=1)
        # mask = Bernoulli(sample_weight).sample().bool()
        
        # return trainloader.dataset.data_index[mask]


class ICLRecAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        if self.config['augment_type'] == 'item_crop':
            self.augmentation = Item_Crop()
        elif self.config['augment_type'] == 'item_mask':
            self.augmentation = Item_Mask(mask_id=train_data.num_items)
        elif self.config['augment_type'] == 'item_reorder':
            self.augmentation = Item_Reorder()
        elif self.config['augment_type'] == 'item_random':
            self.augmentation = Item_Random(mask_id=train_data.num_items)
        else:
            raise ValueError(f"augmentation type: '{self.config['augment_type']}' is invalided")

        self.InfoNCE_loss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='inner_product', neg_type='batch_both')

        if self.config['intent_seq_representation_type'] == 'concat':
            self.cluster = faiss.Kmeans(d=self.config['embed_dim'] * self.config['max_seq_len'], k=self.config['num_intent_clusters'], gpu=False)
        else:
            self.cluster = faiss.Kmeans(d=self.config['embed_dim'], k=self.config['num_intent_clusters'], gpu=False)
        self.centroids = None

    @torch.no_grad()
    def train_kmeans(self, query_encoder, trainloader, device):
        # intentions clustering
        kmeans_training_data = []
        for batch_idx, batch in enumerate(trainloader):
            seq_out = query_encoder(batch, need_pooling=False)
            seq_out = recfn.seq_pooling_function(seq_out, batch['seqlen'], pooling_type=self.config['intent_seq_representation_type'])
            kmeans_training_data.append(seq_out) # [B, D]

        kmeans_training_data = torch.cat(kmeans_training_data, dim=0)
        self.cluster.train(kmeans_training_data.cpu().numpy())
        self.centroids = torch.from_numpy(self.cluster.centroids).to(device)

    def forward(self, batch, seq_out:torch.Tensor, query_encoder:torch.nn.Module):
        output_dict = {}
        # augmented sequence representation without pooling.
        # Because instance CL and Intent CL may use different pooling operations, we perform pooling operations in specific CL tasks.
        seq_augmented_i, seq_augmented_i_len = self.augmentation(batch['in_'+self.fiid], batch['seqlen']) # seq: [B, L] seq_len : [B]
        seq_augmented_j, seq_augmented_j_len = self.augmentation(batch['in_'+self.fiid], batch['seqlen'])
        seq_augmented_i_out = query_encoder({"in_"+self.fiid : seq_augmented_i, "seqlen" : seq_augmented_i_len}, \
            need_pooling=False) # [B, L, D]
        seq_augmented_j_out = query_encoder({"in_"+self.fiid : seq_augmented_j, "seqlen" : seq_augmented_j_len}, \
            need_pooling=False) # [B, L, D]

        # Instance CL
        instance_seq_i_out = recfn.seq_pooling_function(seq_augmented_i_out, seq_augmented_i_len, \
            pooling_type=self.config['instance_seq_representation_type']) # [B, L * D] or [B, D]
        instance_seq_j_out = recfn.seq_pooling_function(seq_augmented_j_out, seq_augmented_j_len, \
            pooling_type=self.config['instance_seq_representation_type']) # [B, L * D]
        instance_loss = self.InfoNCE_loss_fn(instance_seq_i_out, instance_seq_j_out) # [B, 2B]
        instance_loss_rev = self.InfoNCE_loss_fn(instance_seq_j_out, instance_seq_i_out)

        # Intent CL
        seq_out = recfn.seq_pooling_function(seq_out, batch['seqlen'], pooling_type=self.config['intent_seq_representation_type'])
        _, intent_ids = self.cluster.index.search(seq_out.cpu().detach().numpy(), 1)
        seq2intents = self.centroids[intent_ids.squeeze(-1)]
        intent_seq_i_out = recfn.seq_pooling_function(seq_augmented_i_out, seq_augmented_i_len, \
            pooling_type=self.config['intent_seq_representation_type']) # [B, D]
        intent_seq_j_out = recfn.seq_pooling_function(seq_augmented_j_out, seq_augmented_j_len, \
            pooling_type=self.config['intent_seq_representation_type']) # [B, D]
        intent_ids = torch.from_numpy(intent_ids.squeeze(-1)).to(seq_out.device)
        intent_loss_i = self.InfoNCE_loss_fn(intent_seq_i_out, seq2intents, instance_labels=intent_ids)
        intent_loss_j = self.InfoNCE_loss_fn(intent_seq_j_out, seq2intents, instance_labels=intent_ids)

        output_dict['instance_cl_loss'] = 0.5 * (instance_loss + instance_loss_rev)
        output_dict['intent_cl_loss'] = 0.5 * (intent_loss_i + intent_loss_j)

        return output_dict


class OnlineItemSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = None

    def update_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings.weight[1:].detach().clone()
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -100.0, 100.0
        for item_idx in range(self.item_embeddings.size(0)):
            item_vector = self.item_embeddings[torch.tensor(item_idx).to(self.item_embeddings.device)].view(-1, 1)
            item_similarity = torch.mm(self.item_embeddings, item_vector).view(-1)
            max_score = max(torch.max(item_similarity), max_score)
            min_score = min(torch.min(item_similarity), min_score)
        return max_score, min_score

    def forward(self, item_idx:Union[torch.Tensor, int], top_k=1, with_score=False):
        '''
        Args:
        item_idx (torch.Tensor or int)
        top_k (int)
        with_score (bool)
        Return:
        indices (torch.Tensor or int): [batch_size] or int
        values (torch.Tensor or float): [batch_size] or float
        '''
        # item_idx : [batch_size] or int
        item_idx = item_idx - 1
        assert top_k == 1, 'only support top 1'
        if type(item_idx) == int:
            item_vector = self.item_embeddings[item_idx] # [dim]
            item_similarity = torch.mv(self.item_embeddings, item_vector) # [num_items]
            # normalize
            # Different from the min-max normalization in source code.
            item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
            item_similarity[item_idx] = -float('inf') # assume that item_idx is not 0.

            scores, indices = item_similarity.topk(top_k)
            indices = indices + 1
            if with_score:
                return indices[0].item(), scores[0].item()
            else:
                return indices[0].item()
        else:
            item_vector = self.item_embeddings[item_idx] # [batch_size, dim]
            item_similarity = torch.mm(item_vector, self.item_embeddings.T) # [batch_size, num_items]
            # normalize
            item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
            item_similarity[torch.arange(item_idx.size(0), device=item_idx.device), item_idx] = -float('inf') # assume that item_idx is not 0.

            scores, indices = item_similarity.topk(top_k) # [B, 1]
            indices = indices + 1
            if with_score:
                return indices.squeeze(-1).to(item_idx.device), scores.squeeze(-1).to(item_idx.device)
            else:
                return indices.squeeze(-1).to(item_idx.device)



