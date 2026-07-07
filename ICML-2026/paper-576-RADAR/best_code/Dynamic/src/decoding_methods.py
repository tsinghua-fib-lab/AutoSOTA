import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.utils import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
try:
    # 老版本 transformers
    from transformers.generation.utils import _crop_past_key_values  # type: ignore
except Exception:
    _crop_past_key_values = None

def crop_past_key_values(past_key_values, max_length: int):
    """
    兼容函数：尽量复用 transformers 的实现；如果没有，就做一个简化裁剪。
    注意：不同模型的 past_key_values 结构可能不同（tuple/list of layers）。
    """
    if _crop_past_key_values is not None:
        return _crop_past_key_values(past_key_values, max_length)

    # 简化版：对每一层的 key/value 在 seq_len 维度裁剪到最后 max_length
    if past_key_values is None:
        return None

    cropped = []
    for layer in past_key_values:
        # layer 通常是 (k, v) 或更复杂结构
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            k, v, *rest = layer
            # 常见形状: (batch, heads, seq, head_dim)
            try:
                k = k[:, :, -max_length:, :]
                v = v[:, :, -max_length:, :]
                cropped.append((k, v, *rest))
            except Exception:
                # 如果结构不符合，直接不裁剪
                cropped.append(layer)
        else:
            cropped.append(layer)
    return tuple(cropped)



@torch.no_grad()
def secure_decoding(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        temperature: Optional[float] = 0.01,
        tokenizer: Optional = None,
        eta: Optional[float] = 1.0,
        gamma: Optional[float] = 1.0,
        **model_kwargs,
):
    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

    max_len = stopping_criteria[0].max_length

    generated_tokens = []
    ii = 0 
    while True:
        ii+=1  
        with torch.no_grad():
            outputs = self(input_ids, **model_kwargs)

        new_logits_full = outputs.logits[:, -1, :]  # last token prediction
        no_retrieval_token = new_logits_full[-1,:].argmax()
        new_logits = new_logits_full[:-1,:] # remove no-retrieval prompt

        # get the softmax of the next_token_logits
        next_token_confidence = torch.nn.functional.softmax(new_logits/temperature, dim=-1)
        kk = next_token_confidence.shape[0]
        weights = gamma ** torch.arange(kk, device=next_token_confidence.device, dtype=next_token_confidence.dtype)
        weighted_confidence = next_token_confidence * weights.unsqueeze(1)
        next_token_agg = torch.sum(weighted_confidence, dim=0)
            
        if ii == 1:
            ## Here, we should reduce the probability of the "I" token
            ## We can do this by setting the probability of the "I" token to 0
            I_token = tokenizer("I", return_tensors="pt")["input_ids"][0][1].item()
            n_token = tokenizer("/n", return_tensors="pt")["input_ids"][0][1].item()
            Please_token = tokenizer("Please", return_tensors="pt")["input_ids"][0][1].item()
            next_token_agg[I_token] = 0  # set the probability of the "I" token to 0
            next_token_agg[n_token] = 0 # set the probability of the "/n" token to 0
            next_token_agg[Please_token] = 0 # set the probability of the "Please" token to 0

        top2_tokens = torch.topk(next_token_agg, 2)
        if top2_tokens.values[0] - top2_tokens.values[1]>eta:
            selected_tokens = top2_tokens.indices[0] # top 1 token
        else: # no retrieval token
            selected_tokens = no_retrieval_token

        input_ids = torch.cat((input_ids, selected_tokens.repeat(input_ids.shape[0], 1)), dim=-1)
        new_cur_len = input_ids.shape[-1]
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)
        model_kwargs["past_key_values"] = outputs.past_key_values
        model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones_like(model_kwargs["attention_mask"][:, :1])], dim=-1)
        generated_tokens.append(selected_tokens)

        # stop if we exceed the maximum length
        if (selected_tokens == eos_token_id_tensor.item()).any():
            break
        
        if all(stopping_criteria(input_ids, scores)):
            break

    return generated_tokens