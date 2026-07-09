# -*- coding: utf-8 -*-
import os
import yaml
import torch
import math
import numpy as np
import clip
from PIL import Image
import torch.nn.functional as F
import operator

class TDA_solver(torch.nn.Module):
    def __init__(self, K, d,
                 pos_capacity = 3, 
                 neg_capacity = 2,
                 pos_beta = 5.0, 
                 neg_beta = 1.0, 
                 pos_alpha = 2.0, 
                 neg_alpha = 0.117,
                 upper_entropy_bound = 0.5, 
                 lower_entropy_bound = 0.2,
                 lower_probability_bound = 0.03):
        super(TDA_solver, self).__init__()
        self.K = K
        self.TDA_cfg = {'positive': {'enabled': True, 'shot_capacity': pos_capacity, 
                                     'alpha': pos_alpha, 'beta': pos_beta}, 
           'negative': {'enabled': True, 
                        'shot_capacity': neg_capacity, 
                        'alpha': neg_alpha, 'beta': neg_beta, 
                        'entropy_threshold': {'lower': lower_entropy_bound, 'upper': upper_entropy_bound}, 
                        'mask_threshold': {'lower': lower_probability_bound, 'upper': 1.0}}}
        self.pos_cache = {}
        self.neg_cache = {}

    def forward(self, query_features, query_labels, clip_prototypes, device = 'cuda'):
        with torch.no_grad():
            text_logits = 100. * query_features.cuda()@clip_prototypes.squeeze()
            acc, pos_cache, neg_cache = run_test_tda(self.TDA_cfg['positive'], 
                                                    self.TDA_cfg['negative'],
                                                    query_features.cuda(), 
                                                    query_labels.cuda(), 
                                                    clip_prototypes.squeeze(), 
                                                    pos_cache = self.pos_cache, 
                                                    neg_cache = self.neg_cache)

            self.pos_cache = pos_cache
            self.neg_cache = neg_cache
            if query_features.shape[0] > 1:
                # Run prediction on the batch using the cache updated with the complete batch.
                final_logits = torch.zeros((query_features.shape[0], self.K))
                for j_ in range(query_features.shape[0]):
                    fl,_,_ = compute_tda_logits(self.TDA_cfg['positive'], 
                                                self.TDA_cfg['negative'], 
                                                query_features[j_:j_+1,:].cuda(), 
                                                clip_prototypes.squeeze(),
                                                self.pos_cache, 
                                                self.neg_cache, 
                                                do_update_cache = False)
                    final_logits[j_,:] = fl.squeeze()
        return text_logits.softmax(-1).cpu(), final_logits.softmax(-1).cpu()
    
    
# Most of the code below is taken from https://github.com/kdiAAA/TDA.
def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights
def get_clip_logits(image_features, clip_weights):
    with torch.no_grad():
        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return clip_logits, loss, prob_map, pred
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = [] # Cached features
        cache_values = [] # Classes present in the cache
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

def compute_tda_logits(pos_cfg, neg_cfg, query_features, clip_weights,
                       pos_cache = {}, neg_cache = {}, do_update_cache = False):
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
    with torch.no_grad():
        clip_logits, loss, prob_map, pred = get_clip_logits(query_features, clip_weights)
        prop_entropy = get_entropy(loss, clip_weights)
        if do_update_cache:
            if pos_enabled:
                update_cache(pos_cache, pred, [query_features, loss], pos_params['shot_capacity'])
    
            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [query_features, loss, prob_map], neg_params['shot_capacity'], True)

        final_logits = clip_logits.clone()
        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(query_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(query_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
    return final_logits, pos_cache, neg_cache


def run_test_tda(pos_cfg, neg_cfg, query_features, query_labels, clip_weights,
                 pos_cache = {}, neg_cache = {}):
    indices = [[i] for i in range(query_features.shape[0])]
    with torch.no_grad():
        accuracies = []

        #Test-time adaptation
        for i in range(query_features.shape[0]):
            indexes = indices[i]
            images_features = query_features[indexes]
            targets = query_labels[indexes]
            final_logits, pos_cache, neg_cache = compute_tda_logits(pos_cfg, neg_cfg, 
                                                                    images_features, clip_weights,
                                   pos_cache, neg_cache, do_update_cache = True)
            acc = cls_acc(final_logits, targets)  
            accuracies.append(acc)

    avg_accuracy = sum(accuracies)/len(accuracies)
    return avg_accuracy, pos_cache, neg_cache
