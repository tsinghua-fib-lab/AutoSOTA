import math
import json
import torch
import einops
import torchvision
import numpy as np
import torch.nn.functional as F

def cross_attn_prepare(task, attention, layer, data):
    if 'layer' in data:
        layer.update({l: True for l in data['layer']})
    if task == 'per_class':
        per_class_cross_prepare(attention, layer)
    elif task == 'all_together':
        all_together_cross_prepare(attention, layer)
    elif task == 'manual_prompt':
        manual_prompt_cross_prepare(attention, layer)
    elif task == 'goca':
        goca_cross_prepare(attention, layer, data['ref_layer'])
    else:
        raise NotImplementedError

def cross_attn(task, data):
    if task == 'per_class':
        return per_class_cross_run(
            data['feature_extractor'],
            data['img'],
            data['base_prompt'],
            data['base_len'],
            data['class_prompts'],
            data['size'],
            data['t'],
        )
    elif task == 'all_together':
        return all_together_cross_run(
            data['feature_extractor'],
            data['img'],
            data['base_prompt'],
            data['base_len'],
            data['class_prompts'],
            data['size'],
            data['t'],
        )
    elif task == 'manual_prompt':
        return manual_prompt_cross_run(
            data['feature_extractor'],
            data['img'],
            data['prompt'],
            data['prompt_len'],
            data['size'],
            data['t'],
        )
    elif task == 'goca':
        return goca_cross_run(
            data['feature_extractor'],
            data['img'],
            data['prompt'],
            data['rescaling_token_id'],
            data['background_objects'],
            data['objects'],
            data['size'],
            data['t'],
            data['head_method'],
            data['layer_method'],
            data['rescale_method'],
            data['ref_layer'],
            data['keep_all_objects'],
        )
    else:
        raise NotImplementedError

def min_max_norm(a):
    return (a - a.min()) / (a.max() - a.min())

def max_norm(a):
    return a / a.max()


def per_class_cross_prepare(attention, layer):
    attention.add('up_cross')

# scale_record = {}
# with open('./scale_record.json') as f:
#     scale_record = json.load(f)
def per_class_cross_run(feature_extractor, img, base_prompt, base_len, class_prompts, size, t):
    '''
    example:
    cross_attn_setting = {
        'task': 'per_class',
        'base_prompt': 'a cat stands on grass in the outdoor',
        'class_prompts': ['cat', 'grass', 'wall'],
        'base_len': 8,
        'res_index': [0, 1],
    }
    '''
    # global scale_record

    def generate_prompt(base_prompt, class_label):
        return f'{base_prompt}, {class_label}'

    all_class_feat = []
    for class_prompt in class_prompts:
        class_len = len(class_prompt.split(' '))
        prompt = generate_prompt(base_prompt, class_prompt)
        prompt = feature_extractor.encode_prompt(prompt)

        features = feature_extractor.extract(prompt, 1, [img], t=t)

        # first gather cross-attns from each resolution
        one_class_feat = []
        resize_target = []
        for k, v in features.items():
            if not ('cross' in k and 'map' in k):
                continue
            cross_attn = v[0]  # 1 (batch_size) x head x hw x seq_len -> head x hw x seq_len
            cross_attn = cross_attn.mean(dim=0)  # hw x seq_len
            h = int(math.sqrt(cross_attn.shape[0]))
            cross_attn = cross_attn.reshape(h, h, -1).permute(2, 0, 1)  # this contains the entire sequence
            # print(cross_attn.shape)  # seq_len x h x w
            # sum-1 norm
            # cross_attn = cross_attn / cross_attn[0,:,:].unsqueeze(0)
            cross_attn = cross_attn[base_len+2:base_len+2+class_len]  # base_len + <start_of_sentence> + comma
            cross_attn = cross_attn.mean(dim=0)
            # print(cross_attn.shape)  # h x w
            one_class_feat.append(cross_attn)
            resize_target.append(cross_attn.shape[0])

            # record min max
            # if class_prompt not in scale_record:
            #     scale_record[class_prompt] = {}
            # if k not in scale_record[class_prompt]:
            #     scale_record[class_prompt][k] = {'min': [], 'max': []}
            # scale_record[class_prompt][k]['min'].append(cross_attn.amin().item())
            # scale_record[class_prompt][k]['max'].append(cross_attn.amax().item())

            # use min max
            # scale = scale_record[class_prompt][k]
            # cross_attn = (cross_attn - scale['min']) / (scale['max'] - scale['min'])
            # cross_attn = cross_attn.clamp(min=0, max=1)

        # then average over resolutions
        resize_target = size
        one_class_feat_mean = []
        for one_res_feat in one_class_feat:
            one_res_feat = F.interpolate(one_res_feat.unsqueeze(0).unsqueeze(0), resize_target)
            one_res_feat = one_res_feat.squeeze(0).squeeze(0)
            one_class_feat_mean.append(one_res_feat)
        one_class_feat_mean = torch.stack(one_class_feat_mean).mean(dim=0)
        # print(one_class_feat_mean.shape)  # h x w
        one_class_feat_mean = min_max_norm(one_class_feat_mean)

        all_class_feat.append(one_class_feat_mean)
    all_class_feat = torch.stack(all_class_feat)
    # print(all_class_feat.shape)  # class x h x w

    return all_class_feat, features


def all_together_cross_prepare(attention, layer):
    attention.add('up_cross')

def all_together_cross_run(feature_extractor, img, base_prompt, base_len, class_prompts, size, t):
    '''
    example:
    cross_attn_setting = {
        'task': 'all_together',
        'base_prompt': 'a cat stands on grass in the outdoor',
        'class_prompts': ['cat', 'grass', 'wall'],
        'base_len': 8,
        'res_index': [0, 1],
    }
    '''

    prompt = ', '.join([base_prompt] + class_prompts)
    prompt = feature_extractor.encode_prompt(prompt)

    features = feature_extractor.extract(prompt, 1, [img], t=t)

    # first gather cross-attns from each resolution
    not_picked_feat = []
    resize_target = []
    for k, v in features.items():
        if not ('cross' in k and 'map' in k):
            continue
        cross_attn = v[0]  # 1 (batch_size) x head x hw x seq_len -> head x hw x seq_len
        cross_attn = cross_attn.mean(dim=0)  # hw x seq_len
        h = int(math.sqrt(cross_attn.shape[0]))
        cross_attn = cross_attn.reshape(h, h, -1).permute(2, 0, 1)  # seq_len x h x w
        not_picked_feat.append(cross_attn)
        resize_target.append(h)

    # then average over resolutions
    # resize_target = max(resize_target)
    resize_target = size
    not_picked_feat_mean = []
    for one_res_feat in not_picked_feat:
        one_res_feat = F.interpolate(one_res_feat.unsqueeze(0), resize_target)
        one_res_feat = one_res_feat.squeeze(0)
        not_picked_feat_mean.append(one_res_feat)
    not_picked_feat_mean = torch.stack(not_picked_feat_mean).mean(dim=0)
    # print(not_picked_feat_mean.shape)  # seq_len x h x w

    # get attns for each class
    all_class_feat = []
    index = base_len + 2  # base_len + <start_of_sentence> + comma
    for class_prompt in class_prompts:
        class_len = len(class_prompt.split(' '))
        one_class_feat = not_picked_feat_mean[index:index+class_len]
        # print(one_class_feat.shape)  # class_len x h x w
        one_class_feat = one_class_feat.mean(dim=0)
        # print(one_class_feat.shape)  # h x w
        one_class_feat = min_max_norm(one_class_feat)
        all_class_feat.append(one_class_feat)

        index += 1 + class_len  # comma + class_prompt
    
    all_class_feat = torch.stack(all_class_feat)
    return all_class_feat, features


def manual_prompt_cross_prepare(attention, layer):
    attention.add('up_cross')

def manual_prompt_cross_run(feature_extractor, img, prompt, prompt_len, size, t):
    '''
    example:
    cross_attn_setting = {
        'task': 'manual_prompt',
        'prompt': 'monkey with hat walking',
        'prompt_len': 4,
        'res_index': [0, 1],
    }
    '''

    prompt = feature_extractor.encode_prompt(prompt)

    features = feature_extractor.extract(prompt, 1, [img], t=t)

    # first gather cross-attns from each resolution
    not_picked_feat = []
    resize_target = []
    for k, v in features.items():
        if not ('cross' in k and 'map' in k):
            continue
        cross_attn = v[0]  # 1 (batch_size) x head x hw x seq_len -> head x hw x seq_len
        cross_attn = cross_attn.mean(dim=0)  # hw x seq_len
        h = int(math.sqrt(cross_attn.shape[0]))
        cross_attn = cross_attn.reshape(h, h, -1).permute(2, 0, 1)  # seq_len x h x w
        not_picked_feat.append(cross_attn)
        resize_target.append(h)

    # then average over resolutions
    # resize_target = max(resize_target)
    resize_target = size
    not_picked_feat_mean = []
    for one_res_feat in not_picked_feat:
        one_res_feat = F.interpolate(one_res_feat.unsqueeze(0), resize_target, mode='bicubic').clamp(min=0)
        one_res_feat = one_res_feat.squeeze(0)
        not_picked_feat_mean.append(one_res_feat)
    not_picked_feat_mean = torch.stack(not_picked_feat_mean).mean(dim=0)
    # print(not_picked_feat_mean.shape)  # seq_len x h x w

    # get attns for each class
    all_class_feat = []
    index = 1  # <start_of_sentence>
    while index <= prompt_len:
        one_class_feat = not_picked_feat_mean[index]
        # print(one_class_feat.shape)  # h x w
        one_class_feat = min_max_norm(one_class_feat)
        all_class_feat.append(one_class_feat)

        index += 1
    
    all_class_feat = torch.stack(all_class_feat)
    # all_class_feat = all_class_feat / (all_class_feat.sum(0, keepdim=True) + 1e-6)
    return all_class_feat, features


def store_scale_record():
    global scale_record
    for class_prompt, per_class_record in scale_record.items():
        for layer, per_layer_record in per_class_record.items():
            scale_record[class_prompt][layer]['min'] = np.mean(per_layer_record['min'])
            scale_record[class_prompt][layer]['max'] = np.mean(per_layer_record['max'])
    with open('scale_record.json', 'w') as f:
        f.write(json.dumps(scale_record))


def goca_cross_prepare(attention, layer, ref_layer):
    additional_layer = set()
    for l in layer:
        additional_layer.add(l.replace('map', 'special-value'))
        additional_layer.add(l.replace('map', 'special-post-value'))
        additional_layer.add(l.replace('map', 'special-weight'))

        # additional_layer.add(l.replace('cross-map', 'special-cross-before'))
        # additional_layer.add(l.replace('cross-map', 'special-cross-after'))

        # additional_layer.add(l[:l.find('block')] + 'special-before')
        # additional_layer.add(l[:l.find('block')] + 'out')
        # cross-q
        # additional_layer.add(l.replace('cross-map', 'cross-q'))
        additional_layer.add(l.replace('cross', 'self'))
    additional_layer.add(ref_layer)
    layer |= additional_layer

# multi-token
# multi_token_method = 'sum'
multi_token_method = 'mean'
def goca_cross_run(
    feature_extractor, img, prompt, rescaling_token_id, background_objects, objects, size, t,
    head_method, layer_method, rescale_method, ref_layer, keep_all_objects,
):
    if feature_extractor.version in ['hunyuan', 'flux']:
        prompt_embeds = prompt
    else:
        prompt_embeds = feature_extractor.encode_prompt(prompt)

    # ----- debug: print prompt structure ----- #
    # print(feature_extractor.pipe.tokenizer)
    # out = feature_extractor.pipe.tokenizer(
    #     prompt,
    #     padding="max_length",
    #     max_length=feature_extractor.pipe.tokenizer.model_max_length,
    #     truncation=True,
    # )
    # input_ids = out['input_ids']
    # attention_mask = out['attention_mask']
    # vocab = feature_extractor.pipe.tokenizer.get_vocab()
    # vocab = {v:k for k, v in vocab.items()}
    # input_tokens = [vocab[id] for id in input_ids]
    # print(input_tokens)
    # print(attention_mask)
    # input_tokens = [vocab[id] for id in input_ids]
    # print(rescaling_token_id)
    # exit()

    features = feature_extractor.extract(prompt_embeds, 1, [img], t=t)

    for k, v in features.items():
        if 'self' in k:
            continue
        if not 'weight' in k and not 'special-value' in k:
            features[k] = v[0]

    # ----- head-wise aggregation ----- #
    for k, v in features.items():
        if 'self' in k:
            continue
        if not 'map' in k:
            continue

        dtype = v.dtype

        # calculate natural weights for head aggregation
        head_count = v.shape[0]
        if k.replace('map', 'special-value') not in features:
            force_vanilla = True
        else:
            force_vanilla = False
        if not (head_method == 'average' or force_vanilla):
            # retrieve tensors
            post_value = features[k.replace('map', 'special-post-value')]  # space^2 x (dim x head)
            weight = features[k.replace('map', 'special-weight')]  # (dim x head) x (dim x head)
            value = features[k.replace('map', 'special-value')]  # head x n x dim
            weight = weight.T  # align to common mathematical formulation
            head_dim = int(weight.shape[0] // head_count)
            # chunk linear projection into sum of multiple heads
            per_head_results = []
            per_head_measures = []
            for head in range(head_count):
                # V_n
                head_value = value[head,:,:]  # n x dim
                # A_nV_n
                head_post_value = post_value[:,head*head_dim:head*head_dim+head_dim]  # space^2 x dim
                # W_n
                head_weight = weight[head*head_dim:head*head_dim+head_dim,:]  # dim x (dim x head)
                # A_nV_nW_n
                head_result = head_post_value @ head_weight  # space^2 x (dim x head)
                head_result = torch.nan_to_num(head_result)
                per_head_results.append(head_result)
                # V_nW_n
                head_measure = head_value @ head_weight  # n x (dim x head)
                head_measure = torch.nan_to_num(head_measure)
                per_head_measures.append(head_measure)
            per_head_results = torch.stack(per_head_results)  # head x space^2 x (dim x head)
            # calculate weights
            per_head_results = per_head_results.permute(1, 0, 2)  # space^2 x head x (dim x head)
            # (for each latent pixel, each head yields a (dim x head) dimension vector)

        # method#0: average (vanilla)
        if head_method == 'average' or force_vanilla:
            natural_weight = torch.ones(
                [v.shape[1], v.shape[0]],
                dtype=dtype
            ).to('cuda')
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

        # method#1: l2-norm
        elif head_method == 'l2-norm':
            natural_weight = torch.linalg.norm(per_head_results, dim=2)  # space^2 x head
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

        # method#2: cosine
        elif head_method == 'cosine':
            all_head_results = post_value @ weight  # space^2 x (dim x head)
            all_head_results = all_head_results.unsqueeze(1)  # space^2 x 1 x (dim x head)
            natural_weight = torch.nn.functional.cosine_similarity(per_head_results, all_head_results, dim=2)
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

        # method#3: dot-product
        elif head_method == 'dot-product':
            all_head_results = post_value @ weight  # space^2 x (dim x head)
            all_head_results = all_head_results.unsqueeze(-1)
            dot_product = torch.bmm(per_head_results, all_head_results).squeeze(-1)  # space^2 x head
            natural_weight = dot_product.clamp(min=0)
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)
            natural_weight = torch.nan_to_num(natural_weight, nan=1.)

        # method#4: dot-product w/o clamp
        elif head_method == 'dot-product w/o clamp':
            all_head_results = post_value @ weight  # space^2 x (dim x head)
            all_head_results = all_head_results.unsqueeze(-1)
            dot_product = torch.bmm(per_head_results, all_head_results).squeeze(-1)  # space^2 x head
            natural_weight = dot_product
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

        # method#5: cosine + V_i @ W_i measure
        elif head_method == 'cosine + VW':
            all_head_results = post_value @ weight  # space^2 x (dim x head)
            all_head_results = all_head_results.unsqueeze(1)  # space^2 x 1 x (dim x head)
            natural_weight = torch.nn.functional.cosine_similarity(per_head_results, all_head_results, dim=2)
            for head in range(len(per_head_measures)):
                measure = torch.linalg.norm(per_head_measures[head].float(), dim=(0,1), ord=2)  # scalar
                natural_weight[:,head] = natural_weight[:,head] * measure
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

        # method#6: dot-product + V_i @ W_i measure
        elif head_method == 'dot-product + VW':
            all_head_results = post_value @ weight  # space^2 x (dim x head)
            all_head_results = all_head_results.unsqueeze(-1)
            dot_product = torch.bmm(per_head_results, all_head_results).squeeze(-1)  # space^2 x head
            natural_weight = dot_product.clamp(min=0)
            for head in range(len(per_head_measures)):
                measure = torch.linalg.norm(per_head_measures[head].float(), dim=(0,1), ord=2)  # scalar
                natural_weight[:,head] = natural_weight[:,head] * measure
            natural_weight = natural_weight / natural_weight.sum(dim=1, keepdim=True)

        # use head-wise natural aggregation
        natural_aggregation = torch.zeros(v.mean(dim=0).shape, dtype=dtype).to(v.device)
        for head in range(head_count):
            this_head_map = v[head]  # space^2 x n
            this_head_weight = natural_weight[:,head].unsqueeze(-1)  # space^2 x 1
            increment = this_head_map * this_head_weight
            natural_aggregation = natural_aggregation + increment

        features[k] = natural_aggregation

    # result: now each map feature in features is in space^2 x n shape, with heads aggregated


    # ----- layer-wise aggregation ----- #
    attn_weights = {}
    for k, v in features.items():
        if 'self' in k:
            continue
        if not 'map' in k:
            continue

        # retrieve features
        ref = features[ref_layer][0]  # dim x h x w
        # cross-q
        # tgt = features[k.replace('map', 'q')][0]  # dim x h x w
        tgt = features[k.replace('cross', 'self')][0].mean(dim=0)

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

        # cross-q tgt to spatial affinity
        # tgt_hw = tgt.shape[-1]
        # tgt = tgt.reshape(-1, tgt_hw * tgt_hw)  # dim x (h x w)
        # tgt = tgt.permute(1, 0).unsqueeze(0)  # mocking head x seq_len x dim
        # tgt = get_attention_scores(tgt, tgt)[0]  # (h x w) x (h x w)

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

        # resize ref to tgt's shape
        # tgt_hw = int(math.sqrt(tgt.shape[0]))
        # ref = ref.reshape(ref_hw, ref_hw, ref_hw, ref_hw)
        # target_size = (tgt_hw, tgt_hw)
        # # we resize the key part using bicubic (or bilinear)
        # ref = F.interpolate(ref, target_size, mode='bilinear')
        # ref = ref.permute(2, 3, 0, 1)
        # # and resize the query part using nearest, i.e., repeating the maps
        # ref = F.interpolate(ref, target_size, mode='nearest')
        # ref = ref.permute(2, 3, 0, 1)

        # compute similarity
        # method#1 dot-product similarity
        if layer_method == 'dot-product similarity':
            ref = ref.reshape(-1)
            tgt = tgt.reshape(-1)
            natural_weight = (ref * tgt).sum()

        # method#2 mse
        elif layer_method == 'mse':
            tgt = tgt.reshape(ref_hw * ref_hw, ref_hw * ref_hw)
            natural_weight = 1 / torch.linalg.norm(ref - tgt, dim=1).sum()

        # method#3 iou-like
        elif layer_method == 'iou-like':
            tgt = tgt.reshape(ref_hw * ref_hw, ref_hw * ref_hw)
            concat = torch.stack([tgt, ref])  # 2 x (h x w) x (h x w)
            intersection = concat.amin(dim=0).sum()
            union = concat.amax(dim=0).sum()
            natural_weight = intersection / (union + 1e-4)

        # vanilla: average
        elif layer_method == 'vanilla':
            natural_weight = torch.ones([ref_hw * ref_hw], dtype=ref.dtype).to(ref.device)

        # print(k)
        # print(natural_weight)
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
    # for idx in range(resize_target):
    #     print(weights[:,idx])
    # print(weights)
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

    # ----- rescaling ----- #
    raw_scores = aggregated_attn  # space x space x n

    # aggregate multi-token objects before rescaling
    all_class_feat = []  # n x h x w
    for idxs in objects.values():
        if multi_token_method == 'sum':
            all_class_feat.append(raw_scores[:,:,idxs].sum(dim=-1))
        else:
            all_class_feat.append(raw_scores[:,:,idxs].mean(dim=-1))
    for idxs in background_objects:
        if multi_token_method == 'sum':
            all_class_feat.append(raw_scores[:,:,idxs].sum(dim=-1))
        else:
            all_class_feat.append(raw_scores[:,:,idxs].mean(dim=-1))
    rescaling_factor = raw_scores[:,:,rescaling_token_id].sum(dim=-1)  # also aggregate rescaling factor
    rescaling_token_id = [len(all_class_feat)]  # modify the rescaling_token_id to point to the new factor
    all_class_feat.append(rescaling_factor)
    all_class_feat = torch.stack(all_class_feat).permute(1, 2, 0)

    raw_scores = all_class_feat

    if rescale_method == 'raw':
        processed_scores = raw_scores
    elif rescale_method == 'per-token renorm':
        processed_scores = raw_scores / raw_scores.amax(dim=[0, 1], keepdim=True)
    elif rescale_method == 'per-token renorm+':
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            raw_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)
    elif rescale_method == 'sum-1 rescaling':
        # processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
    elif rescale_method == 'sum-1 rescaling + per-token renorm':
        # processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
        processed_scores = processed_scores / processed_scores.amax(dim=[0, 1], keepdim=True)
    elif rescale_method == 'sum-1 rescaling + per-token renorm+':
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            processed_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)
    elif rescale_method == 'sum-1 rescaling + per-token renorm x raw':
        # processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
        processed_scores = processed_scores / processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = processed_scores * raw_scores
    elif rescale_method == 'sum-1 rescaling + per-token renorm+ x raw':
        # processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            processed_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)
        processed_scores = processed_scores * raw_scores
    elif rescale_method == 'sum-1 rescaling + per-token renorm+ x raw + renorm':
        # processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
        factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
        processed_scores = raw_scores / factor
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            processed_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)
        processed_scores = processed_scores * raw_scores
        processed_scores = torchvision.transforms.functional.gaussian_blur(
            processed_scores.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        amin = processed_scores.amin(dim=[0, 1], keepdim=True)
        amax = processed_scores.amax(dim=[0, 1], keepdim=True)
        processed_scores = (processed_scores - amin) / (amax - amin)

    all_class_feat = processed_scores[:,:,:rescaling_token_id[0]]  # only keep object maps
    all_class_feat = all_class_feat.permute(2, 0, 1)
    # if len(all_class_feat) != 0:
    #     all_class_feat = torch.stack(all_class_feat)
    # else:
    #     all_class_feat = None

    if keep_all_objects is not None:
        return all_class_feat, features
    else:
        # postprocess for background object
        first_bg_id = len(objects)
        all_class_feat, all_background_feat = all_class_feat[:first_bg_id], all_class_feat[first_bg_id:]
        # special case: only one token
        if len(all_class_feat.shape) == 2:
            all_class_feat = all_class_feat.unsqueeze(0)
        if len(all_background_feat.shape) == 2:
            all_background_feat = all_background_feat.unsqueeze(0)
        # merge background
        if all_background_feat.shape[0] > 0:
            all_background_feat = all_background_feat.amax(dim=0, keepdim=True)
        else:
            all_background_feat = torch.zeros(
                (all_class_feat.shape[1], all_class_feat.shape[2]),
                dtype=all_background_feat.dtype
            )
            all_background_feat = all_background_feat.unsqueeze(0)
            all_background_feat = all_background_feat.to(all_class_feat.device)
        all_class_feat = torch.concat([all_class_feat, all_background_feat], dim=0)

    # structure: n target classes, 1 supporting object
    return all_class_feat, features
