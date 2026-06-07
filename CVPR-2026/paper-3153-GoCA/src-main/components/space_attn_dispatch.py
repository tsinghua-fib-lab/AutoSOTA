import math
import torch
import torch.nn.functional as F

def space_attn_prepare(task, attention, layer, data):
    if 'layer' in data:
        layer.update({l: True for l in data['layer']})
    if task == 'self_attn':
        self_attn_prepare(attention, layer)
    elif task == 'activation_similarity':
        pass
    else:
        raise NotImplementedError

def space_attn(task, data):
    if task == 'self_attn':
        return self_attn_run(
            data['features'], data['size'], data['layer']
        )
    elif task == 'activation_similarity':
        return activation_similarity_run(
            data['features'],
            data['layer'],
            data['size'],
        )
    else:
        raise NotImplementedError

def self_attn_prepare(attention, layer):
    # attention.add('up_self')
    pass

def self_attn_run(features, size, layer):
    '''
    example:
    space_attn_setting = {
        'task': 'self_attn',
        'layer': ['up-level3-repeat1-vit-block0-self-map']
    }
    '''
    # first gather self-attns from each resolution
    not_aggregated_feat = []
    resize_target = []
    for k, v in features.items():
        if k not in layer:
            continue
        self_attn = v[0]
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
    # print(aggregated_feat.shape)  # h x w x h x w

    # final normalization
    # we use x/x.sum() norm for each index in the query part
    # aggregated_feat = aggregated_feat.reshape(resize_target * resize_target, resize_target * resize_target)
    # aggregated_feat = aggregated_feat / aggregated_feat.sum(dim=1, keepdim=True)
    # aggregated_feat = aggregated_feat.reshape(resize_target, resize_target, resize_target, resize_target)

    return aggregated_feat


def activation_similarity_run(features, layers, size):
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

    space_attns = []
    for layer in layers:
        feat = features[layer][0]
        # print(feat.shape)  # dim x h x w
        space_len = feat.shape[-1]
        feat = feat.reshape(-1, space_len * space_len)
        # print(feat.shape)  # dim x (h x w)

        feat = feat.permute(1, 0).unsqueeze(0)  # mocking head x seq_len x dim
        space_attn = get_attention_scores(feat, feat)[0]
        # print(space_attn.shape)  # (h x w) x (h x w)

        space_attns.append(space_attn)
    space_attns = torch.stack(space_attns).mean(0)
    space_attns = space_attns.reshape(space_len, space_len, space_len, space_len)

    space_attns = F.interpolate(space_attns, size, mode='bicubic')
    space_attns = space_attns.permute(2, 3, 0, 1)
    space_attns = F.interpolate(space_attns, size, mode='bicubic')
    space_attns = space_attns.permute(2, 3, 0, 1)
    return space_attns
