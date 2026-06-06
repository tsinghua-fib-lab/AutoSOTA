from torch import nn, einsum
import torch.nn.functional as F
import random
import torch
from einops import rearrange, repeat, pack, unpack
from torch.cuda.amp import autocast
import torch.distributed as distributed


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, '... -> 1 ...')


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
        samples,
        num_clusters,
        num_iters=10,
        use_cosine_sim=False,
        sample_fn=batched_sample_vectors,
        all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d=dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b=batch)
    return embeds.gather(2, indices)


# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)


def average_neighbor_features(adj, features):
    # 4. 计算每个节点的邻居数量（度数）
    deg = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1) + 1e-10  # [N, 1]

    # 5. 邻居特征求和
    neighbor_sum = torch.sparse.mm(adj, features)  # [N, D]

    # 6. 求平均
    neighbor_avg = neighbor_sum / deg  # [N, D]

    return neighbor_avg


class GCN(nn.Module):
    def __init__(self, args, in_feats, h_feats=32, num_layers=2, dropout_rate=0, activation='ReLU', num_hops=4,
                 **kwargs):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        if num_layers == 0:
            return
        self.layers.append(nn.Linear(in_feats, h_feats))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.cross_attn = CrossAttn(h_feats * num_hops)

        input_dim = h_feats
        codebook_size = args.code_size
        # codebook_size = in_feats.shape[0]
        self.vq = VectorQuantize(dim=input_dim, codebook_size=codebook_size, codebook_dim=input_dim, decay=0.95,
                                 commitment_weight=4.0,
                                 use_cosine_sim=True)
        self.node_decoder = nn.Linear(input_dim, input_dim)
        # LayerNorm to normalize embeddings before VQ for consistent quantization
        self.vq_norm = nn.LayerNorm(input_dim)
        self.edge_decoder = nn.Linear(input_dim, input_dim)
        self.top_k = args.topk

    def forward(self, h, data):
        x_list = h.x_list
        # Z^{[l]} = MLP(X^{[l]}
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]

        residual_list = []
        # Z^{[0]}
        first_element = x_list[0]
        quantized, _, commit_loss, dist, codebook = self.vq(self.vq_norm(x_list[-1]))

        quantized_node = self.node_decoder(quantized)

        feature_rec_loss = F.mse_loss(x_list[-1], quantized_node)

        codebook = codebook.squeeze()

        # H 是 [N, D] 的节点特征
        # h_neighbor = average_neighbor_features(h.adj_without_loop, quantized)
        # h_neighbor = average_neighbor_features(h.adj_without_loop, quantized)

        # H_neighbor = torch.sparse.mm(h.adj_without_loop, quantized)

        # print("weibo")
        # else:
        # print("other")

        x_list = self.get_concat_h(x_list, codebook, data)

        for h_i in x_list[1:]:
            # R^{[l]} = Z^{[l]}-Z^{[0]}
            dif = h_i - first_element
            residual_list.append(dif)
        # H = [R^{[1]} || ... || R^{[L]}]
        residual_embed = torch.hstack(residual_list)

        loss = feature_rec_loss + commit_loss

        # residual_embed = quantized - h_neighbor
        codebook = torch.cat([codebook, codebook], dim=1)
        # quantized = torch.cat([quantized, quantized], dim=1)

        # dif_2 = quantized - first_element
        # quantized = torch.cat([dif_2, dif_2], dim=1)

        # return h_neighbor, loss, x_list[-1], codebook
        # return h_neighbor, loss, quantized, codebook

        # return residual_embed, loss, quantized, codebook
        return residual_embed, loss, x_list[-1], codebook

    def get_concat_h(self, x_list, codebook, data):
        aggregation = "mean"
        aggregated_features = torch.zeros(x_list[0].shape[0], x_list[0].shape[1], dtype=torch.float32)
        if data.name in ["weibo", "BlogCatalog"]:
            # print(self.top_k)
            for i, h_i in enumerate(x_list[1:]):
                # 计算 X 和 codebook 之间的余弦相似度
                # cos_sim = F.cosine_similarity(h_i.unsqueeze(1), codebook.unsqueeze(0), dim=2)
                h_i_norm = self.vq_norm(h_i)
                cos_sim = torch.matmul(h_i_norm, codebook.T)
                top_k_similar_idx = cos_sim.topk(k=self.top_k, dim=1).indices
                top_k_codes = codebook[top_k_similar_idx]
                # 聚合操作
                if aggregation == 'mean':
                    aggregated_features = top_k_codes.mean(dim=1)
                elif aggregation == 'sum':
                    aggregated_features = top_k_codes.sum(dim=1)
                elif aggregation == 'max':
                    aggregated_features = top_k_codes.max(dim=1).values
                elif aggregation == 'min':
                    aggregated_features = top_k_codes.min(dim=1).values

                h_i = 0.5 * h_i + 0.5 * aggregated_features
                x_list[i + 1] = h_i
        return x_list


# 特征聚合
def get_code_emb(feats, mode="mean"):
    if mode == 'mean':
        return feats.mean(dim=1)
    elif mode == 'sum':
        return feats.sum(dim=1)
    elif mode == 'max':
        return feats.max(dim=1).values
    elif mode == 'min':
        return feats.min(dim=1).values
    else:
        raise NotImplementedError(f"Unknown aggregation mode: {mode}")


class CrossAttn(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttn, self).__init__()
        self.embedding_dim = embedding_dim
        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    # def cross_attention(self, query_X, support_X):
    #     Q = self.Wq(query_X)  # query
    #     K = self.Wk(support_X)  # key
    #     attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
    #         torch.tensor(self.embedding_dim, dtype=torch.float32))
    #     attention_weights = F.softmax(attention_scores, dim=1)
    #     # print(attention_weights)
    #     weighted_query_embeddings = torch.matmul(attention_weights, support_X)
    #     return weighted_query_embeddings

    def get_train_loss(self, X, node_emb, X_code, y, num_prompt):
        positive_indices = torch.nonzero((y == 1)).squeeze(1).tolist()

        all_negative_indices = torch.nonzero((y == 0)).squeeze(1).tolist()

        negative_indices = random.sample(all_negative_indices, len(positive_indices))
        # H_q_i, y_i == 1
        query_p_embed = X[positive_indices]
        # H_q_i, y_i == 0
        query_n_embed = X[negative_indices]
        # H_q

        query_embed = torch.vstack([query_p_embed, query_n_embed])

        remaining_negative_indices = list(set(all_negative_indices) - set(negative_indices))

        if len(remaining_negative_indices) < num_prompt:
            raise ValueError(f"Not enough remaining negative indices to select {num_prompt} support nodes.")

        support_indices = random.sample(remaining_negative_indices, num_prompt)
        support_indices = torch.tensor(support_indices).to(y.device)
        # H_k
        support_embed = X[support_indices]

        # the updated query node embeddings
        # \tilde{H_q}
        # query_tilde_embeds = self.cross_attention(query_embed, support_embed)
        # tilde_p_embeds: \tilde{H_q_i}, y_i == 1; tilde_n_embeds: \tilde{H_q_i}, y_i == 0;

        query_tilde_embeds = torch.mean(support_embed, dim=0).unsqueeze(0).repeat(2 * len(positive_indices), 1)

        tilde_p_embeds, tilde_n_embeds = query_tilde_embeds[:len(positive_indices)], query_tilde_embeds[
                                                                                     len(positive_indices):]

        yp = torch.ones([len(negative_indices)]).to(y.device)
        yn = -torch.ones([len(positive_indices)]).to(y.device)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, 1), if y_i == 0
        # X_code = X_code.squeeze()

        cos_sim_n = torch.matmul(query_n_embed, X_code.T)
        # # 找出每个query最相似的码本索引
        top_idx_n = cos_sim_n.argmax(dim=1)  # [600]
        code_n = X_code[top_idx_n]

        loss_qn = F.cosine_embedding_loss(query_n_embed, tilde_n_embeds, yp)
        loss_code_n = F.cosine_embedding_loss(code_n, tilde_n_embeds, yp)
        # cos_embed_loss(H_q_i, \tilde{H_q_i}, -1), if y_i == 1

        cos_sim_p = torch.matmul(query_p_embed, X_code.T)
        # 找出每个query最相似的码本索引
        # top_idx_p = cos_sim_p.argmax(dim=1)  # [600]
        top_idx_p = cos_sim_p.argmin(dim=1)  # [600]
        code_p = X_code[top_idx_p]
        loss_qp = F.cosine_embedding_loss(query_p_embed, tilde_p_embeds, yn)
        loss_code_p = F.cosine_embedding_loss(code_p, tilde_p_embeds, yn)
        loss = torch.mean(loss_qp + loss_qn + loss_code_n + loss_code_p)
        # loss = torch.mean(loss_qp + loss_qn)

        return loss

    def get_test_score(self, X, codebook_sum, prompt_mask, y):
        # prompt node indices
        negative_indices = torch.nonzero((prompt_mask == True)).squeeze(1).tolist()

        n_support_embed = X[negative_indices]

        cos_sim_n = torch.matmul(n_support_embed, codebook_sum.T)
        top_idx_n = cos_sim_n.argmax(dim=1)
        code_n = codebook_sum[top_idx_n]
        # query node indices
        query_indices = torch.nonzero(prompt_mask == False).squeeze(1).tolist()
        # H_q
        query_embed = X[query_indices]

        query_tilde_embed = torch.mean(n_support_embed, dim=0).unsqueeze(0).repeat(len(query_indices), 1)
        query_tilde_embed_code = torch.mean(code_n, dim=0).unsqueeze(0).repeat(len(query_indices), 1)

        diff = query_embed - query_tilde_embed
        diff_code = query_embed - query_tilde_embed_code
        # score
        query_score = torch.sqrt(torch.sum(diff ** 2, dim=1))
        query_score_code = torch.sqrt(torch.sum(diff_code ** 2, dim=1))

        test_score = (query_score + query_score_code) / 2
        # test_score = query_score
        # test_score = query_score_code

        return test_score

    def get_test_score_2(self, node_neighbor, node_emb, codebook_sum, prompt_mask, y, data):
        negative_indices = torch.nonzero((prompt_mask == True) & (y == 0)).squeeze(1).tolist()
        n_support_embed = node_emb[negative_indices]

        query_tilde_embed = torch.mean(n_support_embed, dim=0)
        query_tilde_embed = query_tilde_embed.unsqueeze(0)  # 形状变为 [1, 1024]
        query_tilde_embed = query_tilde_embed.expand(node_emb.shape[0], node_emb.shape[1])

        diff = node_emb - query_tilde_embed

        n_support_embed = node_neighbor[negative_indices]
        query_tilde_embed = torch.mean(n_support_embed, dim=0)
        query_tilde_embed = query_tilde_embed.unsqueeze(0)

        query_tilde_embed = query_tilde_embed.expand(node_neighbor.shape[0], node_neighbor.shape[1])
        diff2 = node_neighbor - query_tilde_embed
        query_score_code = torch.sqrt(torch.sum(diff2 ** 2, dim=1))
        query_score = torch.sqrt(torch.sum(diff ** 2, dim=1))

        return (query_score + query_score_code) / 2

        # return query_score

        # return query_score_code

    def get_test_score_1(self, node_neighbor, node_emb, codebook_sum, prompt_mask, y, data):
        node_emb = node_emb / torch.norm(node_emb, dim=-1, keepdim=True)
        node_neighbor = node_neighbor / torch.norm(node_neighbor, dim=-1, keepdim=True)

        sim_matrix = torch.mm(node_emb, node_neighbor.T)
        sim_matrix = sim_matrix * data.adj_norm.to_dense().to(sim_matrix.device)

        sim_matrix[torch.isinf(sim_matrix)] = 0
        sim_matrix[torch.isnan(sim_matrix)] = 0
        row_sum = torch.sum(data.adj_norm.to_dense(), 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.

        message = torch.sum(sim_matrix, 1)
        r_inv = r_inv.to(sim_matrix.device)
        query_score = message * r_inv

        return query_score


class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            codebook_dim=None,
            heads=1,
            separate_codebook_per_head=False,
            decay=0.8,
            eps=1e-5,
            kmeans_init=False,
            kmeans_iters=10,
            sync_kmeans=True,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            commitment_weight=1.,
            orthogonal_reg_weight=0.,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads
        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        self._codebook = codebook_class(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss,
            sample_codebook_temp=sample_codebook_temp
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack([indices], 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d=codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes, = unpack(codes, ps, 'b * d')
        return codes

    def forward(self, x, mask=None):
        only_one = x.ndim == 2

        if only_one:
            x = rearrange(x, 'b d -> b 1 d')
        # print(x.shape)
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)
        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h=heads)

        quantize, embed_ind, dist, embed = self._codebook(x)
        # print(embed)
        # print(quantize.shape, embed.shape)
        # print(self.training)
        codes = self.get_codes_from_indices(embed_ind)
        # print(codes.shape)
        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                detached_quantize = quantize.detach()

                if exists(mask):
                    # with variable lengthed sequences
                    commit_loss = F.mse_loss(detached_quantize, x, reduction='none')

                    if is_multiheaded:
                        mask = repeat(mask, 'b n -> c (b h) n', c=commit_loss.shape[0],
                                      h=commit_loss.shape[1] // mask.shape[0])

                    commit_loss = commit_loss[mask].mean()
                else:
                    commit_loss = F.mse_loss(detached_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self._codebook.embed

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h=heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h=heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h=heads)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h=height, w=width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')
            embed_ind = rearrange(embed_ind, 'b 1 -> b')
        # print(self._codebook.embed)
        return quantize, embed_ind, loss, dist, self._codebook.embed


class EuclideanCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            num_codebooks=1,
            kmeans_init=False,
            kmeans_iters=10,
            sync_kmeans=True,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        # print(dim, codebook_size, num_codebooks)
        embed = init_fn(num_codebooks, codebook_size, dim)
        # print(embed)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
                use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
        # print(self.initted)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4
        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')
        # print(flatten.shape)
        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        # print(embed)
        # print(flatten.shape, embed.shape)
        # flatten: 2110 * 3703 embed: 8192*3703
        # print(f"flatten: {flatten}")
        dist = -torch.cdist(flatten, embed, p=2)
        # print(dist.shape)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        # print(embed_ind.shape)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        # print(embed_onehot.shape)
        embed_ind = embed_ind.view(*shape[:-1])
        # print(embed_ind.shape)
        quantize = batched_embedding(embed_ind, self.embed)
        # print(embed_onehot.shape)
        if self.training:
            cluster_size = embed_onehot.sum(dim=1)
            # print(cluster_size.shape)
            self.all_reduce_fn(cluster_size)
            self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            # print(cluster_size.shape)
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        # print(quantize.shape)
        return quantize, embed_ind, dist, self.embed


class CosineSimCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            num_codebooks=1,
            kmeans_init=False,
            kmeans_iters=10,
            sync_kmeans=True,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook

        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)

        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        # print(embed_ind.shape)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        # print(embed_onehot.shape)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)

            self.cluster_size.data.lerp_(bins, 1 - self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            self.embed.data.lerp_(embed_normalized, 1 - self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind, dist, self.embed

    # main class
