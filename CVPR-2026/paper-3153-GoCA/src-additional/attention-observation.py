import os
import math
import torch
import numpy as np
import torchvision
import diffusion_feature
import torch.nn.functional as F

from PIL import Image

# ----- settings: you may want to config these -----#

# feature extractor
version = '1-5'
img_size = 512
layer = [
    'down-level0-repeat0-vit-block0-cross-map',
    'down-level0-repeat1-vit-block0-cross-map',
    'down-level1-repeat0-vit-block0-cross-map',
    'down-level1-repeat1-vit-block0-cross-map',
    'down-level2-repeat0-vit-block0-cross-map',
    'down-level2-repeat1-vit-block0-cross-map',
    'mid-vit-block0-cross-map',
    'up-level1-repeat0-vit-block0-cross-map',
    'up-level1-repeat1-vit-block0-cross-map',
    'up-level1-repeat2-vit-block0-cross-map',
    'up-level2-repeat0-vit-block0-cross-map',
    'up-level2-repeat1-vit-block0-cross-map',
    'up-level2-repeat2-vit-block0-cross-map',
    'up-level3-repeat0-vit-block0-cross-map',
    'up-level3-repeat1-vit-block0-cross-map',
    'up-level3-repeat2-vit-block0-cross-map',
]
# layer = [
#     'down-level1-repeat0-vit-block0-cross-map',
#     'down-level1-repeat0-vit-block1-cross-map',
#     'down-level1-repeat1-vit-block0-cross-map',
#     'down-level1-repeat1-vit-block1-cross-map',

#     'down-level2-repeat0-vit-block0-cross-map',
#     'down-level2-repeat0-vit-block1-cross-map',
#     'down-level2-repeat0-vit-block2-cross-map',
#     'down-level2-repeat0-vit-block3-cross-map',
#     'down-level2-repeat0-vit-block4-cross-map',
#     'down-level2-repeat0-vit-block5-cross-map',
#     'down-level2-repeat0-vit-block6-cross-map',
#     'down-level2-repeat0-vit-block7-cross-map',
#     'down-level2-repeat0-vit-block8-cross-map',
#     'down-level2-repeat0-vit-block9-cross-map',

#     'down-level2-repeat1-vit-block0-cross-map',
#     'down-level2-repeat1-vit-block1-cross-map',
#     'down-level2-repeat1-vit-block2-cross-map',
#     'down-level2-repeat1-vit-block3-cross-map',
#     'down-level2-repeat1-vit-block4-cross-map',
#     'down-level2-repeat1-vit-block5-cross-map',
#     'down-level2-repeat1-vit-block6-cross-map',
#     'down-level2-repeat1-vit-block7-cross-map',
#     'down-level2-repeat1-vit-block8-cross-map',
#     'down-level2-repeat1-vit-block9-cross-map',

#     'mid-vit-block0-cross-map',
#     'mid-vit-block1-cross-map',
#     'mid-vit-block2-cross-map',
#     'mid-vit-block3-cross-map',
#     'mid-vit-block4-cross-map',
#     'mid-vit-block5-cross-map',
#     'mid-vit-block6-cross-map',
#     'mid-vit-block7-cross-map',
#     'mid-vit-block8-cross-map',
#     'mid-vit-block9-cross-map',

#     'up-level0-repeat0-vit-block0-cross-map',
#     'up-level0-repeat0-vit-block1-cross-map',
#     'up-level0-repeat0-vit-block2-cross-map',
#     'up-level0-repeat0-vit-block3-cross-map',
#     'up-level0-repeat0-vit-block4-cross-map',
#     'up-level0-repeat0-vit-block5-cross-map',
#     'up-level0-repeat0-vit-block6-cross-map',
#     'up-level0-repeat0-vit-block7-cross-map',
#     'up-level0-repeat0-vit-block8-cross-map',
#     'up-level0-repeat0-vit-block9-cross-map',

#     'up-level0-repeat1-vit-block0-cross-map',
#     'up-level0-repeat1-vit-block1-cross-map',
#     'up-level0-repeat1-vit-block2-cross-map',
#     'up-level0-repeat1-vit-block3-cross-map',
#     'up-level0-repeat1-vit-block4-cross-map',
#     'up-level0-repeat1-vit-block5-cross-map',
#     'up-level0-repeat1-vit-block6-cross-map',
#     'up-level0-repeat1-vit-block7-cross-map',
#     'up-level0-repeat1-vit-block8-cross-map',
#     'up-level0-repeat1-vit-block9-cross-map',

#     'up-level0-repeat2-vit-block0-cross-map',
#     'up-level0-repeat2-vit-block1-cross-map',
#     'up-level0-repeat2-vit-block2-cross-map',
#     'up-level0-repeat2-vit-block3-cross-map',
#     'up-level0-repeat2-vit-block4-cross-map',
#     'up-level0-repeat2-vit-block5-cross-map',
#     'up-level0-repeat2-vit-block6-cross-map',
#     'up-level0-repeat2-vit-block7-cross-map',
#     'up-level0-repeat2-vit-block8-cross-map',
#     'up-level0-repeat2-vit-block9-cross-map',

#     'up-level1-repeat0-vit-block0-cross-map',
#     'up-level1-repeat0-vit-block1-cross-map',
#     'up-level1-repeat1-vit-block0-cross-map',
#     'up-level1-repeat1-vit-block1-cross-map',
#     'up-level1-repeat2-vit-block0-cross-map',
#     'up-level1-repeat2-vit-block1-cross-map',
# ]
# layer = [
#     'vit-block10-cross-map',
#     'vit-block11-cross-map',
#     'vit-block12-cross-map',
#     'vit-block13-cross-map',
#     'vit-block14-cross-map',
#     'vit-block15-cross-map',
#     'vit-block16-cross-map',
#     'vit-block17-cross-map',
# ]
# input
img = Image.open('./cat.png')
prompt = 'cat on grass before wall'
target_token_id = [[1], [3], [5]]  # check out the tokenizer output to know the correct idx

# output
output_dir = './vis-output'
os.makedirs(output_dir, exist_ok=True)

# type of visualization
visualization_type = 'objective'
# visualization_type = 'renormed'
# visualization_type = 'ranking'

# head aggregation
head_method = 'average'
# head_method = 'l2-norm'
# head_method = 'cosine'
# head_method = 'dot-product'
# head_method = 'dot-product w/o clamp'
# head_method = 'cosine + VW'
# head_method = 'dot-product + VW'
visualize_head_weight = False
visualize_head = False

# layer aggregation
layer_method = 'vanilla'
# layer_method = 'iou-like'
# layer_method = 'mse'
# layer_method = 'dot-product similarity'
ref_layer = 'up-level2-repeat1-vit-block0-cross-q'  # remember to change this when switching model
print_layer_weight = False

# rescaling
# rescale_method = 'raw'
# rescale_method = 'per-token renorm'
rescale_method = 'per-token renorm+'
# rescale_method = 'sum-1 rescaling'
# rescale_method = 'sum-1 rescaling + per-token renorm'
# rescale_method = 'sum-1 rescaling + per-token renorm+'
# rescale_method = 'sum-1 rescaling + per-token ranking renorm'
# rescale_method = 'sum-1 rescaling + per-token renorm x raw'
# rescale_method = 'sum-1 rescaling + per-token renorm+ x raw'
# rescale_method = 'sum-1 rescaling + per-token renorm+ x raw + renorm'
# rescale_method = 'compared with min'
# additional per-pixel rescaling setting
rescaling_token_id = [1, 3, 5]

# postprocess: spatial affinity propagation
do_postprocess = False
order = 2

# final display
# show attention scores at specified pixels
truncate_len = 8  # truncate the long prompt, showing only first n tokens configged here
pixel_location = [ # (index in img_size // 8)
    [24, 20],
    [30, 40],
    [50, 30],
    [12, 35],
    [ 3, 35],
    [42, 25],
]
# for segmentation mask
background_threshold = 0.5


# ----- annotate image with pixel location ----- #

img_resized = img.resize((img_size, img_size))
img_resized.save(os.path.join(output_dir, 'input.png'))
for pixel in pixel_location:
    img_annotated = np.array(img_resized)
    i_start = pixel[0] * 8
    j_start = pixel[1] * 8
    i_end = i_start + 8
    j_end = j_start + 8
    img_annotated[i_start:i_end,j_start:j_end] = np.array([255, 0, 0])
    img_annotated = Image.fromarray(img_annotated)
    img_annotated.save(os.path.join(output_dir, f'{pixel[0]}-{pixel[1]}.png'))


# ----- feature extraction ----- #

# add supporting layers to extract
additional_layer = set()
for l in layer:
    additional_layer.add(l.replace('map', 'special-value'))
    additional_layer.add(l.replace('map', 'special-post-value'))
    additional_layer.add(l.replace('map', 'special-weight'))

    additional_layer.add(l.replace('cross', 'self'))
additional_layer.add(ref_layer)
layer = layer + list(additional_layer)

# initialize a feature extractor
df = diffusion_feature.FeatureExtractor(
    layer={l: True for l in layer},
    version=version,
    img_size=img_size,
    device='cuda',
)

# prepare a prompt
# this prompt should be describing the image content
if df.version in ['hunyuan', 'flux']:
    prompt_embeds = prompt
else:
    prompt_embeds = df.encode_prompt(prompt)

# run the extraction
features = df.extract(prompt_embeds, batch_size=1, image=[img])

# check the results
for k, v in features.items():
    if 'self' in k:
        continue
    if not 'weight' in k and not 'special-value' in k:
        features[k] = v[0]
    print(k, features[k].shape)


# ----- print prompt structure ----- #

print(df.pipe.tokenizer)
out = df.pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=df.pipe.tokenizer.model_max_length,
    truncation=True,
)
input_ids = out['input_ids']
attention_mask = out['attention_mask']
vocab = df.pipe.tokenizer.get_vocab()
vocab = {v:k for k, v in vocab.items()}
input_tokens = [vocab[id] for id in input_ids]
print(input_tokens)
print(attention_mask)
input_tokens = [vocab[id] for id in input_ids][:truncate_len]


# ----- visualization tool ----- #

def visualize_map(token_map, save_name):
    '''shape: space x space'''
    # convert into RGB image
    image = token_map.clamp(min=0, max=1) * 255
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.float().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image).resize((512, 512))
    image.save(os.path.join(output_dir, f'{save_name}.png'))

def process_one_nhw_map_renormed(token_scores, names):
    # vanilla approach: per-token re-normalization
    token_scores = token_scores / token_scores.amax(dim=[1,2], keepdim=True)

    for idx, name in enumerate(names):
        visualize_map(token_scores[idx], name)

def process_one_nhw_map_objective(token_scores, names):
    for idx, name in enumerate(names):
        visualize_map(token_scores[idx], name)

def ranking_map(token_scores, names):
    second_largest = token_scores.topk(k=2, dim=0)[0][1]  # h x w
    token_scores = token_scores - second_largest.unsqueeze(0)
    token_scores = token_scores.clamp(min=0)  # 1st largest get positive values, others are 0.0
    token_scores = token_scores / token_scores.amax(dim=[1,2], keepdim=True)

    # make the winner region more obvious
    token_scores = token_scores * 0.7 + (token_scores > 0) * 0.3  # n x h x w

    for idx, name in enumerate(names):
        visualize_map(token_scores[idx], name)

if visualization_type == 'objective':
    process_one_nhw_map = process_one_nhw_map_objective
elif visualization_type == 'renormed':
    process_one_nhw_map = process_one_nhw_map_renormed
elif visualization_type == 'ranking':
    process_one_nhw_map = ranking_map


# ----- head-wise aggregation ----- #

for k, v in features.items():
    if 'self' in k:
        continue
    if not 'map' in k:
        continue

    dtype = v.dtype

    # calculate natural weights for head aggregation
    # retrieve tensors
    head_count = v.shape[0]
    if not head_method == 'average':
        post_value = features[k.replace('map', 'special-post-value')]  # space^2 x (dim x head)
        weight = features[k.replace('map', 'special-weight')]  # (dim x head) x (dim x head)
        value = features[k.replace('map', 'special-value')]  # head x n x dim
        weight = weight.T  # align to common mathematical formulation
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

    # visualize head weights
    if visualize_head_weight:
        hw = int(math.sqrt(natural_weight.shape[0]))
        visualize_weight = natural_weight.permute(1, 0).reshape(-1, hw, hw)
        process_one_nhw_map(visualize_weight, [f'{k}-head{head}-weight' for head in range(head_count)])

    # use head-wise natural aggregation
    natural_aggregation = torch.zeros(v.mean(dim=0).shape, dtype=v.dtype).to(v.device)
    for head in range(head_count):
        this_head_map = v[head]  # space^2 x n
        this_head_weight = natural_weight[:,head].unsqueeze(-1)  # space^2 x 1
        increment = this_head_map * this_head_weight
        natural_aggregation = natural_aggregation + increment

        if visualize_head:
            hw = int(math.sqrt(this_head_map.shape[0]))
            to_visualize = this_head_map.permute(1, 0).reshape(-1, hw, hw)
            process_one_nhw_map(to_visualize, [f'{n}-{k}-head{head}' for n in target_token_id])

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
    ref = features[ref_layer]  # dim x h x w
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

resize_target = int(512 // 8)
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
if print_layer_weight:
    for k, v in normed_attn_weights.items():
        print(k, v)

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
for idxs in target_token_id:
    all_class_feat.append(raw_scores[:,:,idxs].sum(dim=-1))
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
    for_min_max = torchvision.transforms.functional.gaussian_blur(
        raw_scores.permute(2, 0, 1), kernel_size=13
    ).permute(1, 2, 0)
    processed_scores = torchvision.transforms.functional.gaussian_blur(
        raw_scores.permute(2, 0, 1), kernel_size=7
    ).permute(1, 2, 0)
    amin = for_min_max.amin(dim=[0, 1], keepdim=True) * 1.1
    amax = for_min_max.amax(dim=[0, 1], keepdim=True)
    processed_scores = (processed_scores - amin) / (amax - amin)
    processed_scores = processed_scores.clamp(min=0, max=1)
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
    # processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
    factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
    processed_scores = raw_scores / factor
    for_min_max = torchvision.transforms.functional.gaussian_blur(
        processed_scores.permute(2, 0, 1), kernel_size=13
    ).permute(1, 2, 0)
    processed_scores = torchvision.transforms.functional.gaussian_blur(
        processed_scores.permute(2, 0, 1), kernel_size=7
    ).permute(1, 2, 0)
    amin = for_min_max.amin(dim=[0, 1], keepdim=True) * 1.1
    amax = for_min_max.amax(dim=[0, 1], keepdim=True)
    processed_scores = (processed_scores - amin) / (amax - amin)
    processed_scores = processed_scores.clamp(min=0, max=1)
elif rescale_method == 'sum-1 rescaling + per-token ranking renorm':
    factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
    processed_scores = raw_scores / factor
    processed_scores = torchvision.transforms.functional.gaussian_blur(
        processed_scores.permute(2, 0, 1), kernel_size=5
    ).permute(1, 2, 0)
    first_largest = processed_scores[:,:,:-1].amax(dim=2, keepdim=True)
    second_largest = processed_scores[:,:,:-1].topk(k=2, dim=2)[0][:,:,1].unsqueeze(-1)
    where_is_largest = processed_scores == first_largest
    # when is 1st, do x - 2nd, otherwise, do x - 1st
    processed_scores_1 = where_is_largest * (processed_scores - second_largest)
    processed_scores_2 = ~where_is_largest * (processed_scores - first_largest)
    processed_scores = processed_scores_1 + processed_scores_2
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
elif rescale_method == 'compared with min':
    processed_scores = torchvision.transforms.functional.gaussian_blur(
        raw_scores.permute(2, 0, 1), kernel_size=5
    ).permute(1, 2, 0)
    amin = processed_scores.amin(dim=[0, 1], keepdim=True)
    processed_scores = processed_scores / amin

all_class_feat = processed_scores[:,:,:rescaling_token_id[0]]  # only keep object maps

# ----- postprocess ----- #

if do_postprocess:
    all_class_feat = all_class_feat.permute(2, 0, 1)
    # first gather self-attns from each resolution
    not_aggregated_feat = []
    for k, v in features.items():
        if 'self-map' not in k:
            continue
        self_attn = v[0]
        # print(self_attn.shape)  # head x (h x w) x (h x w)
        self_attn = self_attn.mean(0)
        # print(self_attn.shape)  # (h x w) x (h x w)
        space_len = int(math.sqrt(self_attn.shape[0]))
        self_attn = self_attn.reshape(space_len, space_len, space_len, space_len)
        not_aggregated_feat.append(self_attn)

    # then average over resolutions
    aggregated_feat = []
    for one_res_feat in not_aggregated_feat:
        # shape is in fact query x key
        # we resize the key part using bicubic (or bilinear)
        one_res_feat = F.interpolate(one_res_feat, resize_target, mode='bilinear')
        one_res_feat = one_res_feat.permute(2, 3, 0, 1)
        # and resize the query part using nearest, i.e., repeating the maps
        # (changed to bilinear for better visualization)
        one_res_feat = F.interpolate(one_res_feat, resize_target, mode='bilinear')
        one_res_feat = one_res_feat.permute(2, 3, 0, 1)
        aggregated_feat.append(one_res_feat)
    aggregated_feat = torch.stack(aggregated_feat).mean(dim=0)

    cross_feat = all_class_feat
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

    all_class_feat = cross_feat.permute(1, 2, 0)


# ----- final display ----- #

def print_line(list_of_content, not_logit=False):
    if not_logit:
        to_print = '\t'.join([f'{c}' for c in list_of_content])
    else:
        to_print = f'{list_of_content[0]}\t'
        list_of_content = list_of_content[1:]
        to_print += '\t'.join([f'{c:.4f}' for c in list_of_content])
    print(to_print)

tab_tokens = [str(t) for t in target_token_id]
for pixel in pixel_location:
    print(pixel)
    scale_div = (img_size // 8) / hw
    pixel_of_interest = all_class_feat[int(pixel[0]/scale_div),int(pixel[1]/scale_div),:truncate_len]
    # print tabs
    print_line(['id'] + tab_tokens, not_logit=True)
    # print avg
    print_line(['scores'] + [pixel_of_interest[idx].item() for idx in range(pixel_of_interest.shape[0])])

all_class_feat = all_class_feat.permute(2, 0, 1)

to_show = all_class_feat
to_show = to_show / to_show.amax()
process_one_nhw_map(to_show, [f'final-{idx}' for idx in target_token_id])

# visualize segment map
background = torch.zeros(all_class_feat[0].shape, dtype=all_class_feat.dtype).to(all_class_feat.device)
background[:] = background_threshold
to_show = torch.concat([background.unsqueeze(0), all_class_feat], dim=0)

palette = [
    [0.4420,  0.5100 , 0.4234],
    [0.8562,  0.9537 , 0.3188],
    [0.2405,  0.4699 , 0.9918],
    [0.8434,  0.9329  ,0.7544],
    [0.3748,  0.7917 , 0.3256],
    [0.0190,  0.4943 , 0.3782],
    [0.7461 , 0.0137 , 0.5684],
    [0.1644,  0.2402 , 0.7324],
    [0.0200 , 0.4379 , 0.4100],
    [0.5853 , 0.8880 , 0.6137],
    [0.7991 , 0.9132 , 0.9720],
    [0.6816 , 0.6237  ,0.8562],
    [0.9981 , 0.4692 , 0.3849],
    [0.5351 , 0.8242 , 0.2731],
    [0.1747 , 0.3626 , 0.8345],
    [0.5323 , 0.6668 , 0.4922],
    [0.2122 , 0.3483 , 0.4707],
    [0.6844,  0.1238 , 0.1452],
    [0.3882 , 0.4664 , 0.1003],
    [0.2296,  0.0401 , 0.3030],
    [0.5751 , 0.5467 , 0.9835],
    [0.1308 , 0.9628,  0.0777],
    [0.2849  ,0.1846 , 0.2625],
    [0.9764 , 0.9420 , 0.6628],
    [0.3893 , 0.4456 , 0.6433],
    [0.8705 , 0.3957 , 0.0963],
    [0.6117 , 0.9702 , 0.0247],
    [0.3668 , 0.6694 , 0.3117],
    [0.6451 , 0.7302,  0.9542],
    [0.6171 , 0.1097,  0.9053],
    [0.3377 , 0.4950,  0.7284],
    [0.1655,  0.9254,  0.6557],
    [0.9450  ,0.6721,  0.6162]
]
palette = (np.array(palette)*255).astype(np.uint8)
palette_img = palette[:to_show.shape[0]]
palette_img = palette_img.reshape((1,)+palette_img.shape)
palette_img = Image.fromarray(palette_img).resize((512, 512), Image.NEAREST)
palette_img.save(os.path.join(output_dir, 'palette.png'))

segments = to_show.argmax(dim=0).cpu().numpy()
img = palette[segments]
img = Image.fromarray(img).resize((512, 512))
img.save(os.path.join(output_dir, 'segment.png'))
