import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from conversation import get_conv_template
import types
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import json
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from collections import defaultdict
import matplotlib.patches as patches

SELECT_LAYER = [15]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    block_indices = []  # <--- 新增：用于存储每个块的编号
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        block_indices.append([i % (target_width // image_size), i // (target_width // image_size)])
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        block_indices.append([-1,-1])
    return processed_images,block_indices

def load_image_from_path(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGBA')
    transform = build_transform(input_size=input_size)
    images,block_indices = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values,block_indices

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images,block_indices = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values,block_indices

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):

    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    generation_config['eos_token_id'] = eos_token_id
    return pixel_values,input_ids,attention_mask,generation_config

def get_attention(model,
            pixel_values = None,
            input_ids = None,
            attention_mask = None,
            visual_features = None,
            output_hidden_states = None,
            target_indices = None):
    with torch.no_grad():
        assert model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = model.extract_feature(pixel_values)
            input_embeds = model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = model.language_model.get_input_embeddings()(input_ids)
        outputs = model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                target_indices=target_indices,
            )
    return outputs

from transformers.models.qwen2.modeling_qwen2 import *
import torch.nn.functional as F
def layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    target_indices=None,
    **kwargs: Unpack[FlashAttentionKwargs],) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )
    if kwargs.get("output_attentions", True):
        def chunked_attention(query_states, key_states, head_dim, target_indices=None, chunk_size=512, attention_mask=None):
            """
            先提取 target_indices 对应的 query token，再在其上进行分块处理。
            
            参数:
                query_states: [B, H, Q_LEN, D]
                key_states:   [B, H, K_LEN, D]
                head_dim:     D
                target_indices: Optional[List[int] or Tensor] 需要计算 attention 的 query token 索引列表
                chunk_size:   每次处理多少个 target token
                attention_mask: Optional[Tensor] 形状 [B, 1, Q_LEN, K_LEN] 或类似
                
            返回:
                attn_weights: [B, H, Q_LEN, K_LEN]，未计算的位置为 0
            """
            # ✅ 调整输入张量的维度顺序：[B, Q_LEN, H, D] -> [B, H, Q_LEN, D]
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            # query_states = query_states.permute(0, 2, 1, 3).contiguous()
            # key_states = key_states.permute(0, 2, 1, 3).contiguous()
            # print("query_states shape: ", query_states.shape)
            # print("key_states shape: ", key_states.shape)
            B, H, Q_LEN, D = query_states.shape
            _, _, K_LEN, _ = key_states.shape
            device = query_states.device
            dtype = query_states.dtype

            # 统一类型
            key_states = key_states.to(dtype)

            # 初始化输出张量
            # print(B,H,Q_LEN,K_LEN)
            # attn_weights = torch.zeros(B, 1, Q_LEN,K_LEN,device=device, dtype=dtype)

            cpu_attn_weight = [[[None for _ in range(Q_LEN)]] for _ in range(B)]

            scale = 1.0 / math.sqrt(head_dim)

            # 如果没有指定 target_indices，则全部计算
            if target_indices is None:
                target_indices = torch.arange(Q_LEN, device=device)
            else:
                if not isinstance(target_indices, torch.Tensor):
                    target_indices = torch.tensor(target_indices, dtype=torch.long, device=device)
                else:
                    target_indices = target_indices.to(device).long()

            num_targets = len(target_indices)
            # print(num_targets)
            # ✅ 先提取所有需要计算的 query token
            selected_query_states = query_states.index_select(dim=2, index=target_indices)  # [B, H, num_targets, D]

            # ✅ 在这些选中的 token 上分块处理
            for i in range(0, num_targets, chunk_size):
                end_i = min(i + chunk_size, num_targets)
                current_indices = target_indices[i:end_i]  # 当前 chunk 的原始位置
                q_chunk = selected_query_states[:, :, i:end_i, :]  # [B, H, chunk_q, D]
                B,H,_,_ = q_chunk.shape
                attn_chunk = torch.zeros(B, 1, end_i - i, K_LEN, device=device, dtype=dtype)
                for h in range(H):
                    q_chunk_h = q_chunk[:, h, :, :][:,None]
                    key_states_h = key_states[:, h, :, :][:,None]
                    # 计算当前 chunk 的 attention
                    attn_chunk_h = torch.matmul(q_chunk_h, key_states_h.transpose(2, 3)) * scale  # [B, H, chunk_q, K_LEN]
                    # 应用 mask（如果存在）
                    if attention_mask is not None:
                        causal_mask = attention_mask.index_select(dim=2, index=current_indices)  # [B, 1, chunk_q, K_LEN]
                        attn_chunk_h += causal_mask

                    # Fix precision issues
                    if dtype in (torch.float16, torch.bfloat16):
                        attn_chunk_h = torch.where(torch.isinf(attn_chunk_h), torch.zeros_like(attn_chunk_h), attn_chunk_h)

                    # Softmax 升精度
                    attn_chunk_h = F.softmax(attn_chunk_h, dim=-1, dtype=torch.float32).to(dtype)
                    attn_chunk += attn_chunk_h
                    del attn_chunk_h
                # # ✅ 转到 CPU，在 H 维度求均值，得到 [B, 1, chunk_q, K_LEN]
                attn_chunk_cpu = (attn_chunk/H).detach().cpu().float().numpy()  # [B, 1, chunk_q, K_LEN]
                # print("attn_chunk_cpu shape: ",attn_chunk_cpu.shape)
                del attn_chunk
                
                # # # ✅ 写入结果到 attn_weights (list)
                for b in range(B):
                    # print(end_i,i)
                    for j in range(end_i - i):  # 当前 chunk 中的 index
                        q_idx = current_indices[j].item()  # 原始 query token index
                        cpu_attn_weight[b][0][q_idx] = attn_chunk_cpu[b, 0, j]  # 存为 numpy array

                torch.cuda.empty_cache()
            # cpu_attn_weight = attn_weights.cpu().float().numpy()
            # del attn_weights
            torch.cuda.empty_cache()
            return cpu_attn_weight
        # print(query_states.shape, key_states.shape)
        #B H Q_LEN K_LEN
        attn_weights = chunked_attention(query_states, key_states, self.head_dim, target_indices, chunk_size=128)
    else:
        attn_weights = None
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def qwen2_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_indices=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = ()
        compute_layer_index_sum = 0
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            compute_layer_index_sum += 1
            if compute_layer_index_sum in SELECT_LAYER and output_attentions:
                output_attentions_cp = True
            else:
                output_attentions_cp = False
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions_cp,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    target_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions_cp,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    target_indices=target_indices,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions_cp:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

def messages2out(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True):
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    for i in range(len(input_ids[0])):
        if tokenizer.decode(input_ids[0][i]) == "<|im_end|>":
            end_ques = i
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,history=None, return_history=True)
    return response,end_ques

def messages2att(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True):
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    img_start = []
    img_end = []
    idx2word_dicts = {}
    need_2_att_w = []
    split_nums = len(pixel_values)
    for i in range(len(input_ids[0])):
        words = tokenizer.decode(input_ids[0][i])
        idx2word_dicts[input_ids[0][i].cpu().item()] = words
        if input_ids[0][i].cpu().item() == 151665:
            all_img_start=i+1
        if input_ids[0][i].cpu().item() == 151666:
            all_img_end=i
    for i in range(len(input_ids[0])):
        if i>all_img_end:
            need_2_att_w.append(i)
    # print(all_img_start,all_img_end)
    img_start = list(range(all_img_start,all_img_end-256+1,256))
    img_end = list(range(all_img_start+256,all_img_end+1,256))
    out = get_attention(model, pixel_values, input_ids, attention_mask, target_indices=need_2_att_w)
    # print(out['attentions'])
    return out['attentions'],idx2word_dicts,img_start,img_end

from sklearn.cluster import DBSCAN
import base64
import io
import cv2
import PIL.Image as Image
from io import BytesIO

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def process_notsave(block_indices, start_k, end_k, attention, input_ids, img_url, img_start, img_end, sig):
    accept_att = {}
    noise_token_num = 10
    noise_mean = [[0 for k in range(noise_token_num-1)] for i in range(len(block_indices))]
    for k in range(start_k,end_k):
        if input_ids[0][k].cpu().item() >= 151643:
            continue
        max_att_mean = 0
        max_att_sum = 0
        for per in range(len(block_indices)):
            if len(block_indices[per]) == 1:
                #只有缩略图
                start = [img_start[max_att_sum]]
                end = [img_end[max_att_sum]]
            else:
                #去掉最后那个缩略图
                start = img_start[max_att_sum: max_att_sum + len(block_indices[per]) - 1]
                end = img_end[max_att_sum: max_att_sum + len(block_indices[per]) - 1]
            max_att_sum += len(block_indices[per])
            layer_sum = []
            #对每一层做
            for i in range(len(attention)):
                block_sum = []
                #计算每一块
                for block in range(len(start)):
                    k_att_map = []
                    for row in attention[i][0]:
                        k_att_map.append(row[k])
                    k_att_map = np.array(k_att_map)
                    attention_map = k_att_map[:,start[block]:end[block]].reshape(-1,16,16).mean(axis=0)
                    block_sum.append(attention_map)
                # noise_mean = noise_mean/len(start)
                #还原每一块放在原位的attention
                if len(block_indices[per]) == 1:
                    block_loc = [0,0]
                else:
                    block_loc = block_indices[per][-2]
                attention_map = np.zeros([(block_loc[1]+1)*16,(block_loc[0]+1)*16])
                for block in range(len(start)):
                    attention_map[block_indices[per][block][1]*16: (block_indices[per][block][1]+1)*16, block_indices[per][block][0]*16: (block_indices[per][block][0]+1)*16] = block_sum[block]
                layer_sum.append(attention_map)
            #对比找到最大attention的map
            mean_layer_sum = np.array(layer_sum).mean(axis=0,keepdims=True)
            sum_per_img_att = mean_layer_sum.max()
            # print(sum_per_img_att)
            if max_att_mean < sum_per_img_att:
                max_att_mean = sum_per_img_att
                img_idx = per
                accept_att_map = mean_layer_sum
            if sig>0: mean_layer_sum = gaussian_filter(mean_layer_sum, sigma=sig)
            mean_layer_sum = mean_layer_sum - mean_layer_sum.min()
            mean_layer_sum = mean_layer_sum / mean_layer_sum.max()
            # print(k,start_k,end_k)
            if k < start_k+noise_token_num:
                noise_mean[per][start_k-k] = mean_layer_sum
        if k >= start_k+noise_token_num:
            # accept_att_map = noise_mean[per]/9
            if sig>0: accept_att_map = gaussian_filter(accept_att_map, sigma=sig)
            accept_att_map = accept_att_map - accept_att_map.min()
            accept_att_map = accept_att_map / accept_att_map.max()
            # print(np.array(noise_mean[img_idx]))
            accept_att_map = accept_att_map - np.array(noise_mean[img_idx]).mean(axis=0)
            accept_att_map[accept_att_map<0] = 0
            if accept_att_map.max() == 0: continue
            accept_att_map = accept_att_map - accept_att_map.min()
            accept_att_map = accept_att_map / accept_att_map.max()
            # accept_att_map = accept_att_map + noise_mean[img_idx].mean()/9
        else:
            continue
        if not img_idx in accept_att:
            accept_att[img_idx] = {}
        accept_att[img_idx][k]=accept_att_map
    return accept_att

def extract_cluster_boxes_normalized(
    attention_map, 
    eps_normalized=0.1, 
    min_samples=2):
    """
    输入:
        attention_map: 二维 numpy array (H, W)，值为 [0~1]
        eps_normalized: 归一化的最大邻域距离，范围 [0, 1]，表示相对于图像对角线
        min_samples: 成为一个簇所需的最小样本数
    
    输出:
        cluster_boxes: 每个聚类对应的最小外接矩形列表 [(x_min, y_min, x_max, y_max), ...]
    """
    H, W = attention_map.shape

    # Step 1: 提取所有非零点的坐标
    coords = np.column_stack(np.where(attention_map > 0.5))  # 使用阈值提取前景点

    if len(coords) == 0:
        return []

    # Step 2: 计算图像对角线长度，用于归一化 eps
    diag_length = np.sqrt(H**2 + W**2)
    eps_actual = diag_length * eps_normalized  # 把归一化 eps 转换为实际像素距离

    # Step 3: 使用 DBSCAN 聚类
    clustering = DBSCAN(eps=eps_actual, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # 忽略噪声点

    # print(f"找到 {n_clusters} 个聚类")

    # Step 4: 对每个聚类提取最小外接矩形
    cluster_boxes = []
    for label in range(n_clusters):
        idxs = coords[labels == label]
        y_min, x_min = idxs.min(axis=0)
        y_max, x_max = idxs.max(axis=0)
        cluster_boxes.append((x_min, y_min, x_max+1, y_max+1))

    return cluster_boxes

def merge_duplicate_boxes_to_dict_avg(data, iou_threshold=0.5):
    """
    将所有重叠的 box 聚类合并，并归属到最后出现的 seq_id。
    合并方式为：对每个聚类内的 box 取平均值。
    
    参数:
        data: dict {seq_id: [box1, box2, ...]}
        iou_threshold: float
    
    返回:
        dict {final_seq_id: [merged_box1, merged_box2, ...]}
    """

    # Step 1: 所有 box 放入 flat list 并记录原始 seq_id
    all_boxes = []
    for seq_id, boxes in data.items():
        for box in boxes:
            all_boxes.append((seq_id, box))

    # Step 2: 初始化 Union-Find 数据结构（用于聚类）
    parent = list(range(len(all_boxes)))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Step 3: 根据 IOU 构建图连接关系（Union 所有重合的 box）
    n = len(all_boxes)
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(all_boxes[i][1], all_boxes[j][1])
            if iou > iou_threshold:
                union(i, j)

    # Step 4: 按照聚类分组，取平均 box，并记录最大 seq_id
    clusters = defaultdict(list)
    for idx, (seq_id, box) in enumerate(all_boxes):
        root = find(idx)
        clusters[root].append((seq_id, box))

    merged_results = []
    for cluster in clusters.values():
        final_seq_id = max(seq_id for seq_id, _ in cluster)
        boxes_in_cluster = [box for _, box in cluster]
        avg_box = average_boxes(boxes_in_cluster)
        merged_results.append((final_seq_id, avg_box))

    # Step 5: 构建输出 dict
    result = defaultdict(list)
    for seq_id, box in merged_results:
        result[seq_id].append(tuple(round(c, 6) for c in box))  # 四舍五入便于去重

    # 对每个 seq_id 下的 box 去重
    for seq_id in result:
        seen = set()
        unique_boxes = []
        for box in result[seq_id]:
            t = tuple(box)
            if t not in seen:
                seen.add(t)
                unique_boxes.append(box)
        result[seq_id] = unique_boxes

    # 按 seq_id 排序返回
    return dict(sorted(result.items()))

def Add_box_border(mbbox, radius=0.05):
    x0 = 0 if mbbox[0] - radius < 0 else mbbox[0] - radius
    y0 = 0 if mbbox[1] - radius < 0 else mbbox[1] - radius
    x1 = 1 if mbbox[2] + radius > 1 else mbbox[2] + radius
    y1 = 1 if mbbox[3] + radius > 1 else mbbox[3] + radius
    return (x0, y0, x1, y1)


from collections import defaultdict

def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 确保 x1 < x2, y1 < y2
    x1_1, x2_1 = sorted([x1_1, x2_1])
    y1_1, y2_1 = sorted([y1_1, y2_1])
    x1_2, x2_2 = sorted([x1_2, x2_2])
    y1_2, y2_2 = sorted([y1_2, y2_2])

    # 计算交集区域
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def average_boxes(boxes):
    """
    计算一组 bounding boxes 的平均 box
    """
    n = len(boxes)
    sum_x1 = sum(box[0] for box in boxes)
    sum_y1 = sum(box[1] for box in boxes)
    sum_x2 = sum(box[2] for box in boxes)
    sum_y2 = sum(box[3] for box in boxes)

    avg_x1 = sum_x1 / n
    avg_y1 = sum_y1 / n
    avg_x2 = sum_x2 / n
    avg_y2 = sum_y2 / n

    return (avg_x1, avg_y1, avg_x2, avg_y2)

def place_on_center(canvas_bgra, content_bgra):
    """一个辅助函数，将 content 图像(BGRA)居中放置在 canvas 画布(BGRA)上"""
    canvas_h, canvas_w, _ = canvas_bgra.shape
    content_h, content_w, _ = content_bgra.shape

    if content_h > canvas_h or content_w > canvas_w:
        scale = min(canvas_h / content_h, canvas_w / content_w)
        new_h, new_w = int(content_h * scale), int(content_w * scale)
        content_bgra = cv2.resize(content_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        content_h, content_w = new_h, new_w

    paste_x = (canvas_w - content_w) // 2
    paste_y = (canvas_h - content_h) // 2
    
    # 使用Alpha通道作为蒙版来粘贴
    alpha_mask = content_bgra[:, :, 3] / 255.0
    
    # 遍历每个颜色通道
    for c in range(0, 3):
        canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c] = \
            alpha_mask * content_bgra[:, :, c] + \
            (1 - alpha_mask) * canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c]
            
    # 更新画布的alpha通道
    canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3] = \
        np.maximum(canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3], content_bgra[:, :, 3])
        
    return canvas_bgra

def swap_and_rebuild_dict(nested_dict):
    """
    将两层嵌套字典的内外层 key 对调。
    
    输入:
        nested_dict: 形如 {outer_key: {inner_key: value}}
    输出:
        new_dict: 形如 {inner_key: {outer_key: value}}
    """
    new_dict = {}

    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in new_dict:
                new_dict[inner_key] = {}
            new_dict[inner_key][outer_key] = value
            
    return dict(sorted(new_dict.items()))

def pil_to_base64(pil_img, format="PNG"):
    buffered = BytesIO()
    # 如果 pil_img.format 不存在，使用指定的默认格式
    img_format = pil_img.format if pil_img.format else format
    pil_img.save(buffered, format=img_format)  # 使用指定格式保存图像到内存
    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"


def decompose_bbox_by_alpha(image_bgra, bbox, alpha_threshold=10):
    """
    将单个BBox根据其Alpha通道分解为多个不包含透明区域的子BBox。

    Args:
        image_bgra (np.array): 4通道的BGRA格式图像。
        bbox (list or tuple): 单个边界框 [x0, y0, x1, y1]。
        alpha_threshold (int): 用于判断像素是否透明的阈值。
                               高于此值的Alpha被认为是不透明的。

    Returns:
        list: 一个包含多个子BBox [x, y, w, h] 的列表。
              如果BBox内没有不透明区域，则返回空列表。
    """
    x0, y0, x1, y1 = bbox
    img_h, img_w, _ = image_bgra.shape

    # 确保BBox坐标在图像范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)

    if x0 >= x1 or y0 >= y1:
        return []

    # 1. 提取BBox内的区域，并获取其Alpha通道
    roi = image_bgra[y0:y1, x0:x1]
    alpha_channel = roi[:, :, 3]  # BGRA格式的Alpha通道在索引3

    # 2. 二值化Alpha通道
    # 使用cv2.THRESH_BINARY，高于阈值的像素变为255，否则为0
    _, mask = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)

    # 3. 寻找轮廓
    # cv2.RETR_EXTERNAL 只检测最外层的轮廓，这正是我们需要的
    # cv2.CHAIN_APPROX_SIMPLE 压缩轮廓，节省内存
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 将轮廓转换为BBox
    sub_bboxes = []
    for contour in contours:
        # 计算轮廓的边界框 (x, y, w, h)
        sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(contour)
        
        # 将子BBox的相对坐标转换回原图的绝对坐标
        abs_x0 = x0 + sub_x
        abs_y0 = y0 + sub_y
        abs_x1 = abs_x0 + sub_w
        abs_y1 = abs_y0 + sub_h
        
        sub_bboxes.append([abs_x0, abs_y0, abs_x1, abs_y1])
        
    return sub_bboxes

def merge_overlapping_bboxes(bboxes):
    """
    合并列表中所有重叠的BBox。

    Args:
        bboxes (list): 一个包含多个BBox [x0, y0, x1, y1] 的列表。

    Returns:
        list: 一个新的BBox列表，其中所有重叠的BBox已被合并。
    """
    if not bboxes:
        return []

    # 使用索引来操作，避免在迭代时修改列表
    bboxes = [list(b) for b in bboxes] # 确保是可修改的列表

    while True:
        merged_one = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                box1 = bboxes[i]
                box2 = bboxes[j]

                # 检查是否重叠
                # 如果一个box的右边在另一个的左边之外，或者上边在下边之外，则不重叠
                is_overlapping = not (box1[2] < box2[0] or  # box1在box2左侧
                                      box1[0] > box2[2] or  # box1在box2右侧
                                      box1[3] < box2[1] or  # box1在box2上方
                                      box1[1] > box2[3])   # box1在box2下方

                if is_overlapping:
                    # 合并两个BBox
                    new_x0 = min(box1[0], box2[0])
                    new_y0 = min(box1[1], box2[1])
                    new_x1 = max(box1[2], box2[2])
                    new_y1 = max(box1[3], box2[3])
                    
                    # 用合并后的大BBox替换第一个，并删除第二个
                    bboxes[i] = [new_x0, new_y0, new_x1, new_y1]
                    bboxes.pop(j)
                    
                    # 因为我们合并了，需要从头开始重新检查
                    merged_one = True
                    break # 跳出内层j循环
                else:
                    j += 1
            
            if merged_one:
                break # 跳出外层i循环，重新开始while True
            else:
                i += 1
        
        # 如果完整遍历一次后没有任何合并发生，则结束
        if not merged_one:
            break
            
    return bboxes

def compact_and_center_with_relative_pos(imgidx, img_nums, image, normalized_bboxes, n=1):
    """
    将 BBox 区域紧凑排列（保留相对位置），然后居中放置在透明背景上。
    
    新增功能:
    - n (int): 一个阈值。找出所有被至少 n 个 BBox 覆盖的区域，
              然后返回所有与这些区域有交集的原始 BBox 的并集。

    Args:
        image (str): 原始图像的路径或base64字符串。
        normalized_bboxes (list of lists): Bounding box 列表。
        n (int): 重叠阈值，默认为1。
    """
    # ✅ 1. 解码并统一转换为4通道BGRA格式
    # if not image.startswith('data:image;base64,'):
    #     image64 = image_to_base64(image).split(',')[1]
    # elif ',' in image:
    #     image64 = image.split(',')[1]
    # image_data = base64.b64decode(image64)
    # pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    pil_img = image
    img_cv_bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    img_h, img_w, _ = img_cv_bgra.shape

    # ✅ 2. 修复空BBox情况的返回值
    if not normalized_bboxes:
        return None, []
        
    # --- 坐标转换 ---
    initial_pixel_bboxes = []
    for n_box in normalized_bboxes:
        nx0, ny0, nx1, ny1 = n_box
        x0, y0 = int(nx0 * img_w), int(ny0 * img_h)
        x1, y1 = int(nx1 * img_w), int(ny1 * img_h)
        initial_pixel_bboxes.append([x0, y0, x1, y1])

    # --- 修正后的逻辑：基于重叠阈值 n 筛选贡献BBox的并集 ---
    
    # 步骤 1: 创建重叠计数图，识别高置信度像素区域
    overlap_map = np.zeros((img_h, img_w), dtype=np.uint16)
    for bbox in initial_pixel_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 < x1 and y0 < y1:
            overlap_map[y0:y1, x0:x1] += 1
    
    # 根据阈值 n 创建高置信度区域的二元掩码
    threshold_mask = (overlap_map >= n)
    
    # 如果没有任何区域满足阈值，则返回空
    if not np.any(threshold_mask):
         return None, []

    # 步骤 2: 查找所有与高置信度区域有交集的原始BBox
    contributing_bboxes = []
    for bbox in initial_pixel_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 < x1 and y0 < y1 and np.any(threshold_mask[y0:y1, x0:x1]):
            contributing_bboxes.append(bbox)
            
    if not contributing_bboxes:
        return None, []

    # 步骤 3: 计算所有贡献BBox的并集
    final_merged_bboxes = merge_overlapping_bboxes(contributing_bboxes)
    
    # 使用这个最终合并后的BBox列表进行后续操作
    bboxes = np.array(final_merged_bboxes, dtype=int)
    # --- 筛选逻辑结束 ---

    # --- 新增的分解步骤 (可以保留，作用于最终的并集区域) ---
    decomposed_bboxes = []
    for bbox in bboxes:
        # 对每个初始BBox进行分解
        sub_bboxes = decompose_bbox_by_alpha(img_cv_bgra, bbox)
        decomposed_bboxes.extend(sub_bboxes)
    
    if not decomposed_bboxes:
        return None,[]

    # 使用分解后的BBox列表进行后续操作
    bboxes = np.array(decomposed_bboxes, dtype=int)
    # --- 分解结束 ---

    # ✅ 3. 您的 "bbox_only_region.png" 逻辑，现在作用于最终筛选出的区域
    masked_img_bgra = np.zeros_like(img_cv_bgra) 
    for x0, y0, x1, y1 in bboxes:
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(img_w, x1), min(img_h, y1)
        if x0_c < x1_c and y0_c < y1_c:
            masked_img_bgra[y0_c:y1_c, x0_c:x1_c] = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
            
    masked_img_rgba = cv2.cvtColor(masked_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_masked = Image.fromarray(masked_img_rgba)
    pil_result_masked.save(f"{img_nums}_{imgidx}_result_transparent_bg.png")

    # --- 紧凑排列逻辑 (逻辑不变) ---
    x_coords = sorted(list(set(bboxes[:, [0, 2]].flatten())))
    y_coords = sorted(list(set(bboxes[:, [1, 3]].flatten())))

    x_map, new_x = {}, 0
    for i in range(len(x_coords) - 1):
        x_map[x_coords[i]] = new_x
        start_x, end_x = x_coords[i], x_coords[i+1]
        if any(b[0] < end_x and b[2] > start_x for b in bboxes):
            new_x += (end_x - start_x)
    x_map[x_coords[-1]] = new_x
    new_total_width = new_x

    y_map, new_y = {}, 0
    for i in range(len(y_coords) - 1):
        y_map[y_coords[i]] = new_y
        start_y, end_y = y_coords[i], y_coords[i+1]
        if any(b[1] < end_y and b[3] > start_y for b in bboxes):
            new_y += (end_y - start_y)
    y_map[y_coords[-1]] = new_y
    new_total_height = new_y

    # ✅ 4. 创建4通道透明画布，并从4通道源图粘贴
    composite_image_bgra = np.zeros((new_total_height, new_total_width, 4), dtype=np.uint8)
    # ori_area = img_w*img_h
    # return_bboxes = []
    for x0, y0, x1, y1 in bboxes:
        y0_c, y1_c = max(0, y0), min(img_h, y1)
        x0_c, x1_c = max(0, x0), min(img_w, x1)
        if y0_c >= y1_c or x0_c >= x1_c: continue
        # area = (x1-x0)*(y1-y0)
        # if area/ori_area > 0.25 or area/ori_area < 0.001: continue
        # return_bboxes.append([x0, y0, x1, y1])
        roi = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
        paste_x, paste_y = x_map[x0], y_map[y0]
        h, w, _ = roi.shape
        composite_image_bgra[paste_y : paste_y + h, paste_x : paste_x + w] = roi

    # ✅ 5. 创建最终的4通道透明画布，并居中粘贴
    final_canvas_bgra = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    final_img_bgra = place_on_center(final_canvas_bgra, composite_image_bgra)
    
    final_img_rgba = cv2.cvtColor(final_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_centered = Image.fromarray(final_img_rgba)
    pil_result_centered.save(f"{img_nums}_{imgidx}_result_transparent_bg_center.png")

    # ✅ 6. 对紧凑图进行缩放和返回
    composite_image_rgba = cv2.cvtColor(composite_image_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result = Image.fromarray(composite_image_rgba)

    up_sclae = 1
    new_size = (round(pil_result.width * up_sclae), round(pil_result.height * up_sclae))
    pil_result = pil_result.resize(new_size, Image.Resampling.BILINEAR)
    pil_result.save(f"{img_nums}_{imgidx}_result.png")
    
    return_norm_bboxes = []
    for x0, y0, x1, y1 in bboxes:
        return_norm_bboxes.append([x0/img_w, y0/img_h, x1/img_w, y1/img_h])
    return pil_result, return_norm_bboxes

def hot_attention_map_show(image, attention_map, bounding_boxes=None, alpha=0.5, save=None):
    """
    在图像上叠加注意力热力图，并可选地绘制与注意力图等比缩放的边界框。
    新增功能：显示 colorbar 表示注意力值（0~1），使用发散色系突出中值 0.5。

    Args:
        image (np.array): 原始图像，形状为 (H, W, 3)，值范围 [0,1] 或 [0,255]
        attention_map (np.array): 注意力图，形状为 (h, w)，值范围建议 [0,1]
        bounding_boxes (list, optional): 
            边界框列表，每个为 [x0, y0, x1, y1]，基于 attention_map 尺寸。
            自动缩放到图像尺寸。默认 None。
        alpha (float, optional): 热力图透明度。默认 0.5。
        save (str, optional): 保存路径。若为 None，则显示图像。
    """
    img_height, img_width, _ = image.shape
    attn_height, attn_width = attention_map.shape

    # 计算缩放因子
    zoom_h = img_height / attn_height
    zoom_w = img_width / attn_width

    # 缩放注意力图到图像尺寸
    attention_resized = zoom(attention_map, (zoom_h, zoom_w))

    # 创建绘图
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    ax.imshow(image)

    # 使用发散色图：RdYlBu_r（红-黄-蓝反向），0.5 对应黄色/白色，非常清晰
    # 其他可选：'RdBu', 'PiYG', 'coolwarm'
    cmap = 'hot'

    # 叠加注意力热力图
    im = ax.imshow(attention_resized, cmap=cmap, alpha=alpha)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Attention Score', fontsize=16)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    cbar.ax.tick_params(labelsize=14)

    # 绘制边界框（如果提供）
    if bounding_boxes:
        for box in bounding_boxes:
            x0, y0, x1, y1 = box
            scaled_x0 = x0 * zoom_w
            scaled_y0 = y0 * zoom_h
            scaled_x1 = x1 * zoom_w
            scaled_y1 = y1 * zoom_h
            width = scaled_x1 - scaled_x0
            height = scaled_y1 - scaled_y0

            rect = patches.Rectangle(
                (scaled_x0, scaled_y0),
                width,
                height,
                linewidth=8,
                edgecolor='lime',    # 高对比度颜色：荧光绿
                facecolor='none',
                linestyle='-',
                alpha=0.9
            )
            ax.add_patch(rect)

    # 关闭坐标轴
    ax.axis('off')

    # 紧凑布局
    fig.tight_layout(pad=0)

    # 保存或显示
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
    else:
        plt.show()

def from_img_and_att_get_cropbox(model, tokenizer, pixel_values, block_indices, question, generation_config,attention, idx2word_dicts, img_url, img_start, img_end, end_ques,sig,thre, history=None, return_history=True):
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    start_k = img_end[-1]
    for i in range(len(input_ids[0])):
        if tokenizer.decode(input_ids[0][i]) == "<|im_end|>":
            end_ques = i
    end_k = end_ques
    accept_att = process_notsave(block_indices, start_k, end_k, attention, input_ids, img_url, img_start, img_end, sig)
    # print(accept_att)
    imgs_words_att_box = {}
    for img_idx in accept_att:
        # image = plt.imread(img_url[img_idx])
        accept_word_att = accept_att[img_idx]
        words_att_box = {}
        pil_img = np.array(img_url[img_idx])[:,:,-1]
        for word in accept_word_att:
            att_map = accept_word_att[word][0]
            # att_map = gaussian_filter(att_map, sigma=3)
            H,W = att_map.shape
            att_map = att_map * (cv2.resize(pil_img,(W,H))>127)
            # att_map = att_map-att_map.min()
            # att_map = att_map/att_map.max()
            # hot_attention_map_show(image,att_map,save=f'{img_idx}_{word}_{dicts[inputs["input_ids"][0][word].cpu().item()].replace(r"/",r"[]")}_{np.std(att_map):.06f}_{calculate_attention_entropy(att_map):.06f}.png')
            #去掉后50%的信号
            # threshold = np.percentile(att_map, 75)
            # att_map = np.where(att_map < threshold, 0, att_map)
            boxs, rigion_nums = find_top_n_attended_regions(att_map, 100, thre)
            # att_map[att_map<0.5] = 0
            #将这个保存下来为图片，用PIL
            # max_val_coords = np.unravel_index(np.argmax(att_map), att_map.shape)
            # hot_attention_map_show(image,att_map,save=f'{img_idx}_{word}_{dicts[inputs["input_ids"][0][word].cpu().item()].replace(r"/",r"[]")}_threshold_{np.std(att_map):.06f}_{calculate_attention_entropy(att_map):.06f}.png')
            # binarized_map = att_map >= 0.5
            # labeled_map = label(binarized_map, connectivity=2)
            # target_label = labeled_map[max_val_coords]
            # regions = regionprops(labeled_map)
            # for region in regions:
            #     if region.label == target_label:
                    # region.bbox 返回 (y0, x0, y1, x1)
                    # y0,x0,y1,x1 = region.bbox
                    # boxs = [[x0,y0,x1,y1]]
            # boxs = extract_cluster_boxes_normalized(att_map)
            # print(boxs)
            # total_attention = np.sum(att_map)
            img_height, img_width = att_map.shape
            # total_area = img_width * img_height
            # min_area = total_area*0.001  # 0.1%
            # max_area = total_area*0.25    # 1%
            save_boxs = []
            if boxs:
                H, W = att_map.shape
                words_att_box[word] = []
                for box in boxs:
                    x0,y0,x1,y1 = box
                    # if x0 == 0 and y0 == 0: continue
                    box_area = (x1 - x0) * (y1 - y0)
                    # if not (min_area <= box_area):
                    #     continue
                    # 计算 box 面积
                    bbox_norm = (x0 / W, y0 / H, x1 / W, y1 / H)
                    # Ambbox = Add_box_border(bbox_norm,radius=0.05)
                    Ambbox = bbox_norm
                    x0,y0,x1,y1 = Ambbox
                    # region = att_map[int(y0*H):int(y1*H)+1, int(x0*W):int(x1*W)+1]
                    # region_sum = np.sum(region)
                    # if region_sum > total_attention / 2:
                    words_att_box[word].append(Ambbox)
                    save_boxs.append(box)
            # image = cv2.resize(np.array(img_url[img_idx]), (np.array(img_url[img_idx]).shape[1] // 4, np.array(img_url[img_idx]).shape[0] // 4))
            # image = np.array(image)
            # hot_attention_map_show(image,att_map,save=f'{img_idx}_{word}_{tokenizer.decode(input_ids[0][word]).replace(r"/",r"[]")}.png',bounding_boxes=save_boxs)
        imgs_words_att_box[img_idx] = words_att_box

    for img_idx in imgs_words_att_box:
        max_word_idx = 0
        for words_idx in imgs_words_att_box[img_idx]:
            max_word_idx = max(max_word_idx,words_idx)

    # img_merged_boxes = {}
    # for img_idx in imgs_words_att_box:
    #     merged_boxes = imgs_words_att_box[img_idx]
    #     flag = True
    #     while flag:
    #         tmp_merged = merge_duplicate_boxes_to_dict_avg(merged_boxes)
    #         if tmp_merged == merged_boxes:
    #             flag = False
    #         merged_boxes = tmp_merged
    #     img_merged_boxes[img_idx] = merged_boxes
    # img_merged_boxes = swap_and_rebuild_dict(img_merged_boxes)
    img_merged_boxes = swap_and_rebuild_dict(imgs_words_att_box)

    words_lines = {}
    get_words = ""
    # print(start_k,end_k)
    for i in range(start_k,end_k):
        token_idx = input_ids[0][i].cpu().item()
        # print(i,dicts[token_idx],end="||")
        # if token_idx < 151643:
            # get_words+=dicts[token_idx]
        for word in img_merged_boxes:
            if i == word+1:
                words_lines[word] = get_words
                get_words = ''
    for word in img_merged_boxes:
        if i == word:
            words_lines[word] = get_words
            get_words = ''
    words_lines[-1] = get_words
    get_words = ''
    # print(img_merged_boxes)
    # print(words_lines)
    image_list = []
    for i in range(len(img_url)):
    #     if not img_url[i].startswith('data:image;base64,'):
    #         image64 = image_to_base64(img_url[i]).split(',')[1]
    #     elif ',' in img_url[i]:
    #         image64 = img_url[i].split(',')[1]
    #     image_data = base64.b64decode(image64)
        # image = np.array(Image.open(io.BytesIO(image_data)))
        image_list.append(img_url[i])
    crop_list = {}
    bounding_boxes = {}
    highlight_imgs = []
    for word in img_merged_boxes:
        if not word in crop_list:
            crop_list[word] = {}
        for imgidx in img_merged_boxes[word]:
            if not imgidx in bounding_boxes: bounding_boxes[imgidx] = []
            for boxid in range(len(img_merged_boxes[word][imgidx])):
                bounding_boxes[imgidx].append(img_merged_boxes[word][imgidx][boxid])
    for imgidx in bounding_boxes:
        # highlight_imgs.append(blur_non_roi_base64(img_url[imgidx],bounding_boxes[imgidx]))
        if not bounding_boxes[imgidx]: continue
        img,bboxs = compact_and_center_with_relative_pos(imgidx,len(img_url),img_url[imgidx],bounding_boxes[imgidx])
        bounding_boxes[imgidx] = bboxs
        highlight_imgs.append(img)
        # highlight_imgs.append(compact_and_center_with_relative_pos(imgidx,len(img_url),img_url[imgidx],bounding_boxes[imgidx]))
        # highlight_imgs.append(compact_and_center_with_relative_pos_in_ori(image_list[imgidx],bounding_boxes[imgidx]))
    return img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes

def find_top_n_attended_regions(att_map, n, threshold=0.5):
    """
    从注意力图中找到前n个最受关注的连通区域。

    这个函数通过以下步骤工作：
    1. 使用阈值对注意力图进行二值化，以识别高关注度区域。
    2. 对二值化图进行连通域分析，找到所有独立的区域。
    3. 为每个区域计算一个“关注度分数”（区域内所有注意力值的总和）。
    4. 根据分数对所有区域进行降序排序。
    5. 返回排名前n的区域的边界框。如果总区域数小于n，则返回所有区域。

    参数:
    att_map (np.ndarray): 二维的注意力图，值通常在0到1之间。
    n (int): 需要寻找的顶部区域的数量。
    threshold (float, optional): 用于二值化的阈值。默认为 0.5。

    返回:
    list: 一个包含边界框的列表。每个边界框格式为 [x_min, y_min, x_max, y_max]。
          列表按关注度分数降序排列。
    """
    # 1. 二值化处理和连通域分析（与原代码相同）
    map_area = att_map.shape[0] * att_map.shape[1]
    binarized_map = att_map >= threshold
    if not np.any(binarized_map):  # 如果阈值化后没有任何区域，直接返回空列表
        return [],0
        
    labeled_map = label(binarized_map, connectivity=2)
    regions = regionprops(labeled_map)

    # 2. 为每个区域计算分数并存储
    scored_regions = []
    for region in regions:
        # 创建一个与att_map同样大小的掩码，其中只有当前区域为True
        mask = (labeled_map == region.label)
        # 计算该区域内所有像素在原始att_map上的注意力值总和作为分数
        score = np.sum(att_map[mask])
        scored_regions.append({
            'score': score,
            'bbox': region.bbox  # bbox格式为 (y0, x0, y1, x1)
        })
        # if 0 == region.bbox[0] and 0 == region.bbox[1]: return [],0

    # 3. 根据分数对区域进行降序排序
    sorted_regions = sorted(scored_regions, key=lambda r: r['score'], reverse=True)

    # # 4. 选择前n个区域（如果不够n个，则全选）
    # if n > len(sorted_regions):
    #     n = len(sorted_regions)
    # top_n_regions = sorted_regions[:n]

    final_boxes = []
    # 5. 提取并格式化边界框
    get_num = 0
    for region in sorted_regions:
        y0, x0, y1, x1 = region['bbox']
        # 转换为 [x_min, y_min, x_max, y_max] 格式
        box_area = (y1-y0) * (x1-x0)
        # if box_area/map_area < 0.001:
        #     continue
        get_num += 1
        final_boxes.append([x0, y0, x1, y1])
        # if 0 == x0 and 0 == y0: return [],0

    # final_boxes = []
    # for region in sorted_regions:
    #     y0, x0, y1, x1 = region['bbox']
    #     final_boxes.append([x0, y0, x1, y1])

    return final_boxes, len(final_boxes)

def once_cot_infer(model,tokenizer,pixel_values,block_indices,question,generation_config, img_url,sig,thre):
    #得到att

    prompt_ques = """Your task is to extract entities from a user's question. You must follow a strict set of rules to deconstruct and reformat these entities into a canonical, attribute-based format. The output should be a single line of comma-separated values.

Extraction Rules:

Deconstruct Object Descriptions: For any object described with adjectives, first state the core noun, then list its properties using a with [property] format.

Example Transformation: "the large blue truck" becomes truck with large size with blue color.
Example Transformation: "the man in the green shirt" becomes man in a shirt with green color.
Standardize Possessives: Convert possessive forms (like X's Y) into an of structure (Y of X).

Examples:

Question: Which one is closer to the camera, the black vehicle or the silver vehicle?
Answer: vehicle with black color, vehicle with silver color

Question: What is the color of the woman's handbag? Blue or white?
Answer: handbag of woman

Question: What is the man in the green shirt holding next to the wooden table?
Answer: man in a shirt with green color, table with wooden material

Question: What is the color of the guard's glove?
Answer: glove of guard

Question: Is the dog on the left or right side of the scooter?
Answer: dog, scooter

Now, extract entities from the question: """

#     prompt_ques = """You are an AI assistant for advanced, structured entity extraction. Your task is to identify key entities from a text (question and options) based on a hierarchical logic, and then format them according to specific rules.

# Part 1: Core Extraction Logic
# You must first determine the type of question to decide what to extract.

# Specific Inquiry: If the question asks about a specific, named entity (e.g., "Is there a red bicycle?"), you must only extract that entity from the question. Do not extract anything from the options in this case.

# General Inquiry with Context: If the question asks about a general placeholder entity with descriptive context (like location or attributes, e.g., "What is the object on the left?"), you must extract both:

# The general entity and its context from the question.
# All specific candidate entities from the meaningful options (e.g., "A. cat, B. dog").
# Pure General Inquiry: If the question is purely general without a useful placeholder (e.g., "Which is correct?", "What do you see?"), you must only extract the specific entities from the meaningful options.

# Exclusion Rule: Always ignore generic options like "Yes", "No", "True", "False", "All of the above", or "None of the above".

# Part 2: Formatting Rules
# After extracting entities, you must reformat them as follows:

# Adjective Formatting: If an entity has a descriptive adjective (e.g., "white rabbit"), reformat it as [Noun] with [Adjective].
# Example: white rabbit -> rabbit with white.
# Location Formatting: If an entity has a locational description (e.g., "object in the upper right corner"), reformat it as [Noun] on the [Location].
# Example: object in the upper right corner -> object on the upper right.
# Combined Formatting: If an entity has both, apply both rules.
# Example: blue car on the left -> car with blue on the left.
# Part 3: Output Format
# The final output must be a single string containing all processed and formatted entities, separated by commas.

# Examples
# Example 1: Specific Inquiry (Logic #1)

# Input Text:
# "Can you see a red bicycle in the picture? A. Yes, B. No"

# Expected Output:
# bicycle with red

# Example 2: General Inquiry with Context (Logic #2) - Your New Example

# Input Text:
# "What is the object in the upper right corner? A. A cat, B. A dog"

# Expected Output:
# object on the upper right, cat, dog

# Example 3: Pure General Inquiry (Logic #3)

# Input Text:
# "Based on the picture, which option is correct? A. There is a cat. B. There is a dog. C. There is a giraffe."

# Expected Output:
# cat, dog, giraffe

# Example 4: Combined Formatting

# Input Text:
# "What do you see in the image? A. A blue car on the left, B. A large house"

# Expected Output:
# car with blue on the left, house with large

# Example 5:

# Input Text:
# "What is the number of persons in the image?\n(A) 17\n(B) 14\n(C) 24\n(D) 13\n(E) The image does not feature the related information."

# Expected Output:
# persons

# Example 6:

# Input Text:
# "How many characters are there in the picture?\n(A) 2.\n(B) 3.\n(C) 4.\n(D) 1.\n(E) The image does not feature the related information."

# Expected Output:
# characters

# Example 7:

# Input Text:
# "What color is the shed on the right window of the house with solar panels on the roof in the left area of the picture?\n(A) Red\n(B) White\n(C) Green\n(D) Blue\n(E) This image doesn't feature the color."

# Expected Output:
# shed on the right window of the house with solar panels on the roof in the left area

# Now, process the following text directly:

# Input Text: """
    # prompt_ques += '\"'+question.split("<image>\n")[-1].replace("\nAnswer with the option's letter from the given choices directly.","")+'\"'+"\nExpected Output: "
    prompt_ques += question.split("<image>\n")[-1].split("\n")[0]+"\nAnswer: "
    prompt_output_text,end_ques = messages2out(model, tokenizer, None, prompt_ques, generation_config, history=None, return_history=True)
    # print(prompt_ques)
    # print(prompt_output_text)
    attention,idx2word_dicts,img_start,img_end = messages2att(model, tokenizer, pixel_values, "Search the following entities in the images: "+prompt_output_text, generation_config, history=None, return_history=True)  # Retrieve attention from model outputs
    img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes = from_img_and_att_get_cropbox(model, tokenizer, pixel_values, block_indices, "Search the following entities in the images: "+prompt_output_text, generation_config,attention, idx2word_dicts, img_url, img_start, img_end, end_ques,sig,thre, history=None, return_history=True)
    #加上这次处理新出的图
    # print(highlight_imgs)
    for h_img in highlight_imgs:
        # print(h_img)
        pixel_values_tmp,block_indices_tmp = load_image(h_img, max_num=128)
        pixel_values = torch.cat([pixel_values,pixel_values_tmp.to(torch.bfloat16).to(model.device)],dim=0)
        block_indices = block_indices + [block_indices_tmp]
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    output_text,end_ques = messages2out(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    return output_text,crop_list,highlight_imgs,pixel_values,block_indices,words_lines,img_merged_boxes,bounding_boxes,prompt_output_text

def create_directory(path):
    """
    创建给定路径的目录，包括所有必要的父目录。

    :param path: 完整的目录路径字符串
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created successfully at {path}")
    except Exception as e:
        print(f"Failed to create directory at {path}: {e}")

import jsonlines
def load_dataset_Vstar_json(path):
    Vstar_list = []
    with open(path, 'r', encoding='utf-8') as f:
        Vstar_list = json.load(f)
    mmetype_Vstarbench = []
    for i in range(len(Vstar_list)):
        dict_i = {}
        dict_i["id"] = Vstar_list[i]["id"]
        dict_i["Text"] = Vstar_list[i]["question"]
        # dict_i["Choices"] = "\n".join(Vstar_list[i]["text"].split("\n")[1:-1])
        dict_i["Ground truth"] = Vstar_list[i]["labels"]
        dict_i["image"] = Vstar_list[i]["image_path"]
        if "box_json" in Vstar_list[i]:
            dict_i["box_json"] = Vstar_list[i]["box_json"]
        dict_i["category"] = Vstar_list[i]["category"]
        mmetype_Vstarbench.append(dict_i)
    return mmetype_Vstarbench

def serialize_dict(my_dict, file_path):

    """
    将一个字典序列化为一行 JSON，追加写入到 .jsonl 文件。
    
    每次调用写入一行，不换行嵌套，符合 JSONL 标准。
    
    参数:
        my_dict: 要写入的字典（可能包含 ndarray、np.int64 等）
        file_path: 输出的 .jsonl 文件路径
    """
    def serialize_obj(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: serialize_obj(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_obj(item) for item in obj]
        else:
            return obj

    # 序列化整个字典
    serialized_dict = serialize_obj(my_dict)

    # 追加写入一行 JSON
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serialized_dict, ensure_ascii=False, indent=4) + '\n')
