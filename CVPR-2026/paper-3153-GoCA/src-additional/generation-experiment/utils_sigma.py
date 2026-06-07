import math
import torch
import random
import torchvision
import numpy as np
from output_mask import output_mask
from diffusers import DiffusionPipeline


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_scheduler(pipe:DiffusionPipeline, scheduler):
    if scheduler=='DDIM':
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, rescale_betas_zero_snr=False, timestep_spacing="leading") #https://arxiv.org/pdf/2305.08891.pdf

    
    elif scheduler=='DPMSlover':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,  solver_order=2, algorithm_type='dpmsolver++' ) 
        #solver_order=2 for conditional 3 for unconditional sampling 
    

    else:
        return 




import abc
from typing import Union, Tuple, List
from einops import rearrange

import torch.nn.functional as F
from gaussian_smoothing import GaussianSmoothing
def get_mask(feature_extractor, model_config, prompt, r: int=4):
    features = feature_extractor.feature_store.stored_feats
    # for k, v in features.items():
    #     print(k, v.shape)
    # exit()

    # ----- separate positive and negative pass of cfg ----- #
    features_pos = {}
    features_neg = {}
    for k, v in features.items():
        if v.shape[0] == 2:
            features_pos[k] = v[1]
            features_neg[k] = v[0]
        else:
            features_pos[k] = v
            features_neg[k] = v

    # ----- head-wise aggregation ----- #
    head_method = model_config.cross_attn_setting['head_method']
    for features in [features_pos, features_neg]:
        for k, v in features.items():
            if 'self' in k:
                continue
            if not 'map' in k:
                continue

            dtype = v.dtype

            # calculate natural weights for head aggregation
            # retrieve tensors
            post_value = features[k.replace('map', 'special-post-value')]  # space^2 x (dim x head)
            weight = features[k.replace('map', 'special-weight')]  # (dim x head) x (dim x head)
            value = features[k.replace('map', 'special-value')]  # head x n x dim
            weight = weight.T  # align to common mathematical formulation
            head_count = v.shape[0]
            head_dim = int(weight.shape[0] // head_count)
            # chunk linear projection into sum of multiple heads
            per_head_results = []
            per_head_measures = []
            for head in range(head_count):
                head_value = value[head,:,:]  # n x dim
                head_post_value = post_value[:,head*head_dim:head*head_dim+head_dim]  # space^2 x dim
                head_weight = weight[head*head_dim:head*head_dim+head_dim,:]  # dim x (dim x head)
                head_result = head_post_value @ head_weight  # space^2 x (dim x head)
                per_head_results.append(head_result)
                head_measure = head_value @ head_weight  # n x (dim x head)
                per_head_measures.append(head_measure)
            per_head_results = torch.stack(per_head_results)  # head x space^2 x (dim x head)
            # calculate weights
            per_head_results = per_head_results.permute(1, 0, 2)  # space^2 x head x (dim x head)
            # (for each latent pixel, each head yields a (dim x head) dimension vector)

            # method#0: average (vanilla)
            if head_method == 'average':
                natural_weight = torch.ones(
                    [per_head_results.shape[0], per_head_results.shape[1]],
                    dtype=dtype
                ).to(per_head_results.device)
                natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

            # method#4: dot-product w/o clamp
            elif head_method == 'dot-product w/o clamp':
                all_head_results = post_value @ weight  # space^2 x (dim x head)
                all_head_results = all_head_results.unsqueeze(-1)
                dot_product = torch.bmm(per_head_results, all_head_results).squeeze(-1)  # space^2 x head
                natural_weight = dot_product
                natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

            # use head-wise natural aggregation
            natural_aggregation = torch.zeros(v.mean(dim=0).shape, dtype=dtype).to(v.device)
            for head in range(head_count):
                this_head_map = v[head]  # space^2 x n
                this_head_weight = natural_weight[:,head].unsqueeze(-1)  # space^2 x 1
                increment = this_head_map * this_head_weight
                natural_aggregation = natural_aggregation + increment

            features[k] = natural_aggregation

    # ----- layer-wise aggregation ----- #
    layer_method = model_config.cross_attn_setting['layer_method']
    ref_layer = model_config.cross_attn_setting['ref_layer']
    size = model_config.feat_size
    for features in [features_pos, features_neg]:
        attn_weights = {}
        for k, v in features.items():
            if 'self' in k:
                continue
            if not 'map' in k:
                continue

            # retrieve features
            ref = features[ref_layer]  # dim x h x w
            # cross-q
            # tgt = features[k.replace('map', 'q')][0]  # dim x h x w
            tgt = features[k.replace('cross', 'self')].mean(dim=0)

            # get dense feature spatial affinity
            def get_attention_scores(
                query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
            ) -> torch.Tensor:
                r"""
                Compute the attention scores.

                Args:
                    query (`torch.Tensor`): The query tensor.
                    key (`torch.Tensor`): The key tensor.
                    attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

                Returns:
                    `torch.Tensor`: The attention probabilities/scores.
                """
                dtype = query.dtype

                if attention_mask is None:
                    baddbmm_input = torch.empty(
                        query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                    )
                    beta = 0
                else:
                    baddbmm_input = attention_mask
                    beta = 1

                attention_scores = torch.baddbmm(
                    baddbmm_input,
                    query,
                    key.transpose(-1, -2),
                    beta=beta,
                    alpha=query.shape[-1]**-0.5,
                )
                del baddbmm_input

                attention_probs = attention_scores.softmax(dim=-1)
                del attention_scores

                attention_probs = attention_probs.to(dtype)

                return attention_probs

            # ref to spatial affinity
            ref_hw = ref.shape[-1]
            ref = ref.reshape(-1, ref_hw * ref_hw)  # dim x (h x w)
            ref = ref.permute(1, 0).unsqueeze(0)  # mocking head x seq_len x dim
            ref = get_attention_scores(ref, ref)[0]  # (h x w) x (h x w)

            # resize tgt to ref's shape
            tgt_hw = int(math.sqrt(tgt.shape[0]))
            tgt = tgt.reshape(tgt_hw, tgt_hw, tgt_hw, tgt_hw)
            target_size = (ref_hw, ref_hw)
            # we resize the key part using bicubic (or bilinear)
            tgt = F.interpolate(tgt, target_size, mode='bilinear')
            tgt = tgt.permute(2, 3, 0, 1)
            # and resize the query part using nearest, i.e., repeating the maps
            tgt = F.interpolate(tgt, target_size, mode='nearest')
            tgt = tgt.permute(2, 3, 0, 1)

            # compute similarity
            # method#1 dot-product similarity
            if layer_method == 'dot-product similarity':
                ref = ref.reshape(-1)
                tgt = tgt.reshape(-1)
                natural_weight = (ref * tgt).sum()

            # vanilla: average
            elif layer_method == 'vanilla':
                natural_weight = torch.ones([ref_hw * ref_hw], dtype=ref.dtype).to(ref.device)

            attn_weights[k] = natural_weight

        # then normalize these weights
        normed_attn_weights = {}

        resize_target = size
        ids = []
        weights = []
        for k, v in attn_weights.items():
            if len(v.shape) == 1:
                hw = int(math.sqrt(v.shape[0]))
                v = v.reshape(hw, hw)
                v = torch.nn.functional.interpolate(
                    v.unsqueeze(0).unsqueeze(0), resize_target,
                    mode='nearest'
                ).squeeze(0).squeeze(0)
            ids.append(k)
            weights.append(v.clamp(min=0))
        weights = torch.stack(weights)
        weights = weights / weights.sum(dim=0, keepdim=True)
        for k, v in zip(ids, range(weights.shape[0])):
            normed_attn_weights[k] = weights[v]  # (h x w)

        # at last aggregate maps
        aggregated_attn = []
        for k, v in features.items():
            if 'self' in k:
                continue
            if not 'map' in k:
                continue

            attn_weight = normed_attn_weights[k]  # h x w
            if len(attn_weight.shape) == 2:
                attn_weight = attn_weight.unsqueeze(-1)

            # print(v.shape)  # space^2 x n

            hw = int(math.sqrt(v.shape[0]))
            v = v.reshape(hw, hw, -1).permute(2, 0, 1)
            v = torch.nn.functional.interpolate(
                v.unsqueeze(0), resize_target,
                mode='bilinear'
            ).squeeze(0)
            v = v.permute(1, 2, 0)  # space x space x n

            v = v * attn_weight

            aggregated_attn.append(v)

        aggregated_attn = torch.stack(aggregated_attn).sum(dim=0)  # space x space x n
        aggregated_attn = aggregated_attn.clamp(min=0)
        aggregated_attn = aggregated_attn / aggregated_attn.sum(dim=2, keepdim=True)
        features['aggregated_attn'] = aggregated_attn

    # ----- rescaling ----- #
    rescale_method = model_config.cross_attn_setting['rescale_method']

    # we do rescaling among non-sos and non-pad tokens
    # utilize tokenizer to know which tokens are what
    def raw_tokenization(s):
        out = feature_extractor.pipe.tokenizer(
            s,
            padding="max_length",
            max_length=feature_extractor.pipe.tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = out['input_ids']
        vocab = feature_extractor.pipe.tokenizer.get_vocab()
        vocab = {v:k for k, v in vocab.items()}
        input_tokens = [vocab[id] for id in input_ids]
        return input_tokens
    def sd15_tokenization(s):
        raw = raw_tokenization(s)
        refined = []
        for token in raw:
            if token == '<|startoftext|>':
                token = '<sos>'
            elif token == '<|endoftext|>':
                token = '<pad>'
            else:
                token = token.replace('</w>', '')
            refined.append(token)
        special_tokens = ['<sos>', '<pad>']
        global_tokens = ['<sos>']
        return refined, special_tokens, global_tokens
    def t5_tokenization(s):
        raw = raw_tokenization(s)
        refined = []
        for token in raw:
            if token == '▁':
                token = '<insert>'
            elif token == '</s>':
                token = '<eos>'
            elif token == '<pad>':
                token = '<pad>'
            else:
                token = token.replace('▁', '')
            refined.append(token)
        special_tokens = ['<insert>', '<eos>', '<pad>']
        global_tokens = ['<eos>']
        return refined, special_tokens, global_tokens
    version = model_config.version
    if version == '1-5':
        tokenizer = sd15_tokenization
    elif version == 'xl':
        tokenizer = sd15_tokenization
    elif version == 'pixart-sigma':
        tokenizer = t5_tokenization
    elif version == 'pixart-sigma-512':
        tokenizer = t5_tokenization
    else:
        raise NotImplementedError
    tokens, special_tokens, global_tokens = tokenizer(prompt[0])
    rescaling_token_id = []
    global_token_id = None
    for idx, token in enumerate(tokens):
        if token not in special_tokens:
            rescaling_token_id.append(idx)
        if token in global_tokens:
            global_token_id = idx

    raw_scores = features_pos['aggregated_attn']  # space x space x n
    features_neg['aggregated_attn'][:,:,0] = raw_scores[:,:,global_token_id]

    if rescale_method == 'raw':
        processed_scores = raw_scores
    elif rescale_method == 'per-token renorm+':
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            raw_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)
    elif rescale_method == 'sum-1 rescaling + per-token renorm+':
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            processed_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)

    # print(processed_scores.shape)  # h x w x n
    features_pos['aggregated_attn'] = processed_scores

    # ----- spatial affinity propagation ----- #
    # aggregate self-attention
    features = features_pos
    layer = model_config.space_attn_setting['layer']
    not_aggregated_feat = []
    resize_target = []
    for k, v in features.items():
        if k not in layer:
            continue
        self_attn = v
        # print(self_attn.shape)  # head x (h x w) x (h x w)
        self_attn = self_attn.mean(0)
        # print(self_attn.shape)  # (h x w) x (h x w)
        space_len = int(math.sqrt(self_attn.shape[0]))
        self_attn = self_attn.reshape(space_len, space_len, space_len, space_len)
        not_aggregated_feat.append(self_attn)
        resize_target.append(space_len)

    # then average over resolutions
    resize_target = size
    aggregated_feat = []
    for one_res_feat in not_aggregated_feat:
        # shape is in fact query x key
        # we resize the key part using bicubic (or bilinear)
        one_res_feat = F.interpolate(one_res_feat, resize_target, mode='bilinear')
        one_res_feat = one_res_feat.permute(2, 3, 0, 1)
        # and resize the query part using nearest, i.e., repeating the maps
        one_res_feat = F.interpolate(one_res_feat, resize_target, mode='nearest')
        one_res_feat = one_res_feat.permute(2, 3, 0, 1)
        aggregated_feat.append(one_res_feat)
    aggregated_feat = torch.stack(aggregated_feat).mean(dim=0)

    # affinity propagation
    order = model_config.postprocess_setting[0]['order']
    for features in [features_pos, features_neg]:
        cross_feat = features['aggregated_attn'].permute(2, 0, 1)
        space_feat = aggregated_feat

        seq_len = cross_feat.shape[0]
        space_len = space_feat.shape[0]

        cross_feat = cross_feat.reshape(seq_len, space_len * space_len).permute(1, 0)
        space_feat = space_feat.reshape(space_len * space_len, space_len * space_len)

        for _ in range(order):
            cross_feat = space_feat @ cross_feat

        cross_feat = cross_feat.permute(1, 0).reshape(seq_len, space_len, space_len)
        amin = cross_feat.amin(dim=[1, 2], keepdim=True)
        amax = cross_feat.amax(dim=[1, 2], keepdim=True)
        cross_feat = (cross_feat - amin) / (amax - amin)

        features['aggregated_attn'] = cross_feat
    
    # ----- argmax for mask ----- #
    # in positive map, make pads zero
    first_pad_in_prompt = rescaling_token_id[-1] + 1
    features_pos['aggregated_attn'][first_pad_in_prompt:,:,:] = 0

    # renormalize fore_ca
    raw_scores = features_neg['aggregated_attn']
    processed_scores = torchvision.transforms.functional.gaussian_blur(
        raw_scores, kernel_size=5
    )
    amin = processed_scores.amin(dim=[1, 2], keepdim=True)
    amax = processed_scores.amax(dim=[1, 2], keepdim=True)
    processed_scores = (processed_scores - amin) / (amax - amin)
    features_neg['aggregated_attn'] = processed_scores

    # align to original codes' behavior
    new_ca = features_pos['aggregated_attn']  # n x h x w
    new_ca = new_ca.unsqueeze(0)  # 1 x n x h x w

    fore_ca = features_neg['aggregated_attn']  # n x h x w
    fore_ca = torch.stack([fore_ca[0], 1-fore_ca[0]]).unsqueeze(0)  # 1 x 2 x h x w

    # debug
    # output_mask(new_ca[0].argmax(dim=0).cpu(), fore_ca[0].argmax(dim=0).cpu())

    max_ca, inds = torch.max(new_ca[:,:], dim=1) 
    max_ca = max_ca.unsqueeze(1) # 
    ca_mask = (new_ca==max_ca).float() # b 77/10 16 16 


    max_fore, inds = torch.max(fore_ca[:,:], dim=1) 
    max_fore = max_fore.unsqueeze(1) # 
    fore_mask = (fore_ca==max_fore).float() # b 77/10 16 16 
    fore_mask = 1.0-fore_mask[:,:1] # b 1 16 16


    return [ ca_mask, fore_mask]
