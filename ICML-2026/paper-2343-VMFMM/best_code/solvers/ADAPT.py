import torch
import torch.nn.functional as F
import torch.nn as nn


def get_zero_shot_logits(query_features, clip_prototypes):
    clip_logits = 100 * query_features @ clip_prototypes
    return clip_logits.squeeze()


def calculate_batch_entropy(logits):
    # entropy = -sum(p * log p)
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def constructed_knowledge_banks_transductive(preds, image_features, losses, prob_maps, bank_size):
    """
        Update Knowledge Banks by selecting the top-'bank_size' samples for each class based on entropy.
        Return: dict[class_idx] = [(feat_i, loss_i, prob_i), ...]
    """
    cache = {}
    unique_preds = preds.unique(sorted=True)
    for pred in unique_preds:
        pred = int(pred.item())
        idxs = (preds == pred).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        if idxs.numel() <= bank_size:
            selected_items = [(image_features[i], float(losses[i].item()), prob_maps[i]) for i in idxs]
        else:
            # Select samples with lowest entropy (highest confidence)
            topk = losses[idxs].topk(min(len(idxs), bank_size), largest=False)[1]
            selected_idxs = idxs[topk]
            selected_items = [(image_features[i], float(losses[i].item()), prob_maps[i]) for i in selected_idxs]
        cache[pred] = selected_items
    return cache

def update_knowledge_banks_online(bank, pred, feature, loss, prob_map, bank_size):
    """update vecs, labels, cache_pro"""
    update = False
    pred = pred.item()
    if pred not in bank:
        bank[pred] = []

    # Prepare the new item
    item = (feature.squeeze(0), loss.item(), prob_map.squeeze(0))
    
    existing_count = len(bank[pred])
    if existing_count < bank_size:
        bank[pred].append(item)
        update = True
    else:
        # Find the item with the highest loss to potentially replace
        # If the new item's loss is lower, replace it
        max_idx = max(range(existing_count), key=lambda i: bank[pred][i][1])
        if item[1] < bank[pred][max_idx][1]:
            bank[pred][max_idx] = item
            update = True

    return update, [feature, pred, prob_map]


def param_estimation_transductive(image_features, banks, initial_mean, alpha, text_sample_prob):
    """
        Gaussian Discriminant Analysis (GDA) parameter estimation using constructed knowledge banks.
        M: total cached samples
    """
    K = initial_mean.shape[0]
    D = initial_mean.shape[1]

    sorted_classes = sorted(banks.keys())
    vecs = torch.cat([item[0].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0)  # [M, D]
    labels = torch.tensor([class_idx for class_idx in sorted_classes for _ in banks[class_idx]])  # [M]
    cache_pro = torch.cat([item[2].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0)  # [M, K]

    # update class mean (weighted by cache and current prob)
    # mus = torch.cat([(((cache_pro[labels == i][:, i].unsqueeze(1) * vecs[labels == i]).sum(dim=0) + (image_features * text_sample_prob[:,i].unsqueeze(1)).sum(dim=0)) / ((cache_pro[labels == i][:, i].sum()) + text_sample_prob[:,i].sum())).unsqueeze(0) if i in banks.keys() else initial_mean[i].unsqueeze(0) for i in range(initial_mean.shape[0])])
    mus_list = []
    for i in range(K):
        if i in banks:
            mask = (labels == i)
            # Weight from cache samples
            cache_w = cache_pro[mask][:, i].unsqueeze(1)  # [m_i, 1]
            cache_sum = cache_w.sum()
            cache_part = (cache_w * vecs[mask]).sum(dim=0) if cache_sum > 0 else torch.zeros(D).cuda()

            # Weight from current batch samples
            cur_w = text_sample_prob[:, i].unsqueeze(1)  # [N, 1]
            cur_sum = cur_w.sum()
            cur_part = (cur_w * image_features).sum(dim=0) if cur_sum > 0 else torch.zeros(D).cuda()

            denom = cache_sum + cur_sum
            mu_i = (cache_part + cur_part) / denom if denom > 0 else initial_mean[i]
            mus_list.append(mu_i.unsqueeze(0))
        else:
            mus_list.append(initial_mean[i].unsqueeze(0))
    mus = torch.cat(mus_list, dim=0)  # [K, D]
    # EMA toward initial mean
    mus = alpha * mus + (1 - alpha) * initial_mean  # [K, D]

    # KS Estimator (Bayes ridge-type estimator)
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in banks.keys()], dim=0)  # [M, D]
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda()) # [D, D]

    ps = torch.ones(K).cuda() * 1. / K  # [K]
    W = torch.einsum('nd,dc->cn', mus, cov_inv)  # [D, K]
    b = ps.log() - 0.5 * torch.einsum('nd,dc,nc->n', mus, cov_inv, mus)  # [K]

    # Precompute similarity matrix and cache logits
    similarity_matrix = image_features @ vecs.T  # [N, M]
    cache_values = (F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=K)).cuda().half()  # [M, K]
    cache_logits = cache_pro * cache_values  # [M, K]

    return W, b, mus, similarity_matrix, cache_logits

def param_estimation_online(added_sample, banks_tensors, initial_mean, prev_mus, alpha):
    """Online Gaussian distribution parameter estimation with Constructed Knowledge Banks."""
    K = initial_mean.shape[0]
    image_features, pred, img_pro = added_sample
    vecs, labels, cache_pro = banks_tensors
    cache_keys = torch.unique(labels)

    mus = prev_mus.clone()
    mask = labels==pred
    selected_vecs = vecs[mask]  # (M, D)
    selected_cache_pro = cache_pro[mask, pred].unsqueeze(1)  # (M, 1)

    # update mean with added_samples in Constructed Knowledge Banks
    new_mu = ((selected_cache_pro * selected_vecs).sum(dim=0) + img_pro[0][pred] * image_features[0])/ (selected_cache_pro.sum() +img_pro[0][pred]).unsqueeze(0)
    new_mu = alpha * new_mu + (1 - alpha) * initial_mean[pred]
    mus[pred] = new_mu

    # KS Estimator (Bayes ridge-type estimator)
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in cache_keys])### [num_samples, dim]
    n, d = center_vecs.shape
    if n == 1:
        Sigma = torch.eye(d).cuda()
    else:
        Sigma = center_vecs.T.cov()
    trace = Sigma.trace()
    cov_inv = d * torch.linalg.pinv((n - 1) * Sigma + trace * torch.eye(d).cuda())

    ps = torch.ones(initial_mean.shape[0]).cuda() * 1. / initial_mean.shape[0]

    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

    # Precompute similarity matrix and cache logits
    similarity_matrix = image_features @ vecs.T  # [N, M]
    cache_values = (F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=K)).cuda().half()  # [M, K]
    cache_logits = cache_pro * cache_values  # [M, K]
    
    return W, b, mus, similarity_matrix, cache_logits

def _banks_to_tensors(bank, K):
    if len(bank) == 0:
        vecs = torch.zeros(0, 0).cuda().float()
        labels = torch.zeros(0, dtype=torch.long).cuda()
        cache_pro = torch.zeros(0, K).cuda().float()
        return vecs, labels, cache_pro

    vecs_list, labels_list, probs_list = [], [], []
    for cls_idx, items in sorted(bank.items()):
        for feat, _, prob in items:
            vecs_list.append(feat.unsqueeze(0).cuda().float())
            labels_list.append(cls_idx)
            probs_list.append(prob.unsqueeze(0).cuda().float())
    vecs = torch.cat(vecs_list, dim=0) # [M, D], float32
    labels = torch.tensor(labels_list, dtype=torch.long).cuda() # [M]
    cache_pro = torch.cat(probs_list, dim=0) # [M, K]
    return vecs, labels, cache_pro

def compute_final_prediction(clip_logits, GDA_logits, similarity_matrix, cache_logits):
    """
        clip_logits: [N, K]
        GDA_logits: [N, K]
        similarity_matrix: [N, M]
        cache_logits: [M, K]
    """
    # log P(y|x) from GDA
    intermediate = torch.log_softmax(GDA_logits, dim=1) # [N, K]
    if cache_logits.numel() > 0:
        intermediate += (50.0 / (max(len(cache_logits), 1) * 2.0)) * (similarity_matrix @ cache_logits) # [N, K]
        
    # For numerical stability
    intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
    final_logits = clip_logits * torch.exp((1 / 50.0) * intermediate) # [N, K]
    final_logits = final_logits / torch.sum(final_logits, dim=1, keepdim=True)
    return final_logits


@torch.no_grad()
def ADAPT_transductive_solver(query_features, query_labels, clip_prototypes, alpha=0.9, bank_size=12):
    """
        ADAPT (transductive) solver
    """
    query_labels = query_labels.cuda().float() # [N]
    clip_prototypes = clip_prototypes.cuda().float() # Text features, [1, D, K]
    query_features = query_features.cuda().float() # Image features, [N, D]

    # Calculate zero-shot logits and soft labels
    clip_logits = get_zero_shot_logits(query_features, clip_prototypes) # [N, K]
    y_hat = F.softmax(clip_logits, dim=1) # [N, K]

    # Construct knowledge banks (by selecting low-entropy samples per class)
    losses = calculate_batch_entropy(clip_logits) # [N]
    preds = clip_logits.argmax(dim=-1) # [N]
    banks = constructed_knowledge_banks_transductive(preds, query_features, losses, y_hat, bank_size)

    # Parameter estimation for GDA
    initial_mean = clip_prototypes.permute(2,0,1).squeeze(1) # [K, D]
    W, b, mus, similarity_matrix, cache_logits = param_estimation_transductive(
        query_features, banks, initial_mean, alpha, text_sample_prob=y_hat
    )
    # Compute GDA logits
    GDA_logits = (query_features @ W) + b # [N, K]
    
    # Final predictions
    final_logits = compute_final_prediction(clip_logits, GDA_logits, similarity_matrix, cache_logits)

    return y_hat.cpu(), final_logits.cpu()

class ADAPT_online_solver(nn.Module):
    """
        ADAPT (online) solver
    """
    def __init__(self, K, d, alpha=0.9, bank_size=12):
        super().__init__()
        self.K = int(K)
        self.d = int(d)
        self.alpha = alpha
        self.bank_size = bank_size

        self.bank = {}             # dict[int] -> list[(feat[D], loss(float), prob[K])]
        self.mean = None           # [K, D]
        self.W = None              # [D, K]
        self.b = None              # [K]

    @torch.no_grad()
    def forward(self, query_features, query_labels, clip_prototypes):
        query_labels = query_labels.cuda().float()
        clip_prototypes = clip_prototypes.cuda().float()
        query_features = query_features.cuda().float()
    
        initial_mean = clip_prototypes.permute(2,0,1).squeeze(1) # [K, D]
        if self.mean is None:
            self.mean = initial_mean.clone()

        # Zero-shot predictions
        clip_logits = get_zero_shot_logits(query_features, clip_prototypes) # [N, K]
        y_hat = F.softmax(clip_logits, dim=1) # [N, K]
        losses = calculate_batch_entropy(clip_logits) # [N]
        preds = clip_logits.argmax(dim=-1) # [N]

        N, K = query_features.shape[0], self.K
        final_logits = torch.empty(N, K).cuda().float()
        # Sample-wise online update
        for j in range(N):
            feat_j = query_features[j:j+1]
            logit_j = clip_logits[j:j+1]
            loss_j = losses[j:j+1]
            prob_j = y_hat[j:j+1]
            pred_j = preds[j]
            # Construct Knowledge Banks
            update_sign, added_sample = update_knowledge_banks_online(
                self.bank, pred_j, feat_j, loss_j, prob_j, self.bank_size
            )

            # Parameter Estimation
            vecs, labels, cache_pro = _banks_to_tensors(self.bank, self.K)
            if update_sign:
                W, b, self.mean, sim_j, cache_logits_j = param_estimation_online(
                    added_sample, (vecs, labels, cache_pro), initial_mean, prev_mus=self.mean, alpha=self.alpha
                )
                self.W, self.b = W, b
            else:
                one_hot = F.one_hot(labels.to(torch.int64), num_classes=self.K).float().to(vecs.device)  # [M, K]
                cache_logits_j = cache_pro * one_hot # [M, K]
                sim_j = feat_j @ vecs.T # [1, M]
                
            # Compute GDA logits    
            GDA_logits_j = (feat_j @ self.W) + self.b # [1, K]
            
            # Final predictions
            final_logits_j = compute_final_prediction(logit_j, GDA_logits_j, sim_j, cache_logits_j) # [1, K]
            
            final_logits[j:j+1] = final_logits_j

        return y_hat.cpu(), final_logits.cpu()
