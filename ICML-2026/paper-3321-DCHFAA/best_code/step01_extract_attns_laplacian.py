"""
Extract attention patterns using Laplacian operator for hallucination detection.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import transformers
from tqdm import tqdm
import argparse

from generation import LLM

transformers.logging.set_verbosity(40)

data_response_names = {
    'summary': 'Summary',
    'qa': 'Answer',
    'data2txt': 'Answer',
}


def load_ragtruth(file_path, debug=False, subsample=None):
    """Load RAGTruth dataset."""
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        if debug:
            data = data[:10]
        if subsample is not None:
            data = [data[i] for i in range(len(data)) if i % subsample == 0]

        for idx in range(len(data)):
            data_index = data[idx]['index']
            context = data[idx]['document']
            new_item = dict(
                context=context,
                response=data[idx]['response'],
                data_index=data_index
            )
            list_data_dict.append(new_item)

    return list_data_dict


def build_prompt(context, response, data_type='summary'):
    """Build prompt for generation."""
    prompt = context
    input_text_prompt = prompt + response
    return input_text_prompt


_LAPLACIAN_KERNEL_1D = None

def _get_laplacian_kernel(device, dtype):
    """Get or create Laplacian convolution kernel."""
    global _LAPLACIAN_KERNEL_1D
    if _LAPLACIAN_KERNEL_1D is None or _LAPLACIAN_KERNEL_1D.device != device or _LAPLACIAN_KERNEL_1D.dtype != dtype:
        _LAPLACIAN_KERNEL_1D = torch.tensor([1.0, -2.0, 1.0], device=device, dtype=dtype).view(1, 1, 3)
    return _LAPLACIAN_KERNEL_1D


def laplacian_1d_batch(x, return_score=True):
    """
    Batch compute Laplacian features for 1D data.
    
    Args:
        x (torch.Tensor): Input data, shape (Seq_Len,) or (Num_Heads, Seq_Len)
        return_score (bool): Whether to return aggregated scalar score
    
    Returns:
        torch.Tensor: Laplacian transformed vector (same shape as input)
        torch.Tensor: (optional) Laplacian score (L2 norm), shape (Num_Heads,) or scalar
    """
    original_shape = x.shape
    original_dim = x.dim()
    device = x.device
    dtype = x.dtype
    
    if original_dim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
        squeeze_first = True
        squeeze_second = True
    elif original_dim == 2:
        x = x.unsqueeze(1)
        squeeze_first = False
        squeeze_second = True
    else:
        raise ValueError(f"Unsupported tensor dimension: {original_dim}. Expected 1D or 2D tensor.")
    
    kernel = _get_laplacian_kernel(device, dtype)
    laplacian_out = F.conv1d(x, kernel, padding=1)
    
    if return_score:
        l2_score = torch.norm(laplacian_out, p=2, dim=-1).squeeze(-1)
        return laplacian_out, l2_score

    return laplacian_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num-gpus", type=str, default="auto")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="dataset_example/ragtruth/anno-Summary-7b.jsonl")
    parser.add_argument("--output-path", type=str, default="outputs/attn-features-summary-7b.pt")
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auth-token", type=str, default=None)
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--max-memory", type=int, default=45)
    parser.add_argument("--start-sample-idx", type=int, default=None)
    parser.add_argument("--end-sample-idx", type=int, default=None)
    
    args = parser.parse_args()
    
    start_idx = args.start_sample_idx
    end_idx = args.end_sample_idx
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.data_type is None:
        if "Summary" in args.data_path:
            args.data_type = "summary"
        elif "QA" in args.data_path:
            args.data_type = "qa"
        elif "Data2txt" in args.data_path:
            args.data_type = "data2txt"
        else:
            raise ValueError("Please specify the data type.")
    
    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    list_data_dict = load_ragtruth(fp, debug=args.debug, subsample=args.subsample)
    
    llm = LLM(
        args.model_name, args.device, args.num_gpus, 
        auth_token=args.auth_token, 
        max_memory=args.max_memory)
    stop_word_list = ["#Document#:", "#Question#:", "#Article#:", "Q:", "\end{code}"]
    llm.set_stop_words(stop_word_list)
    mode = "vanilla"

    teacher_forcing_dict = {}
    response_list = []
    for idx in tqdm(range(len(list_data_dict))):
        sample = list_data_dict[idx]
        response = sample['response']
        tokenized_response = llm.tokenizer(response, return_tensors='pt')['input_ids'].squeeze(0)
        teacher_forcing_dict[sample['data_index']] = tokenized_response
        response_list.append(response)
        
    to_save_list = []
    extra_prompt_length = len(llm.tokenizer(f"\n#{data_response_names[args.data_type]}#:")['input_ids']) - 1
    
    if start_idx is not None and end_idx is not None:
        list_data_dict = list_data_dict[start_idx:end_idx]
    elif start_idx is not None:
        list_data_dict = list_data_dict[start_idx:]

    for idx in tqdm(range(len(list_data_dict))):
        sample = list_data_dict[idx]

        teacher_forcing_ids = teacher_forcing_dict[sample['data_index']].to(args.device).unsqueeze(0)
        input_text = build_prompt(sample['context'], f"\n#{data_response_names[args.data_type]}#:", data_type=args.data_type)

        max_new_tokens_tf = teacher_forcing_ids.shape[-1]
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens_tf, 
            do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, 
            temperature=args.temperature, mode=mode, 
            return_attentions=True, teacher_forcing_seq=teacher_forcing_ids
        )
        model_completion, attentions, model_completion_ids = llm.generate(
            input_text, **generate_kwargs)

        context_length = attentions[0][0].shape[-1] - extra_prompt_length
        new_token_length = len(attentions)
        num_layers = len(attentions[0])
        num_heads = attentions[0][0].shape[1]
        
        context_laplacian_score = torch.zeros((num_layers, num_heads, new_token_length))
        new_tokens_laplacian_score = torch.zeros((num_layers, num_heads, new_token_length))
        all_previous_laplacian_score = torch.zeros((num_layers, num_heads, new_token_length))

        for i in range(len(attentions)):
            for l in range(num_layers):
                attn_on_context = attentions[i][l][0, :, -1, :context_length]
                attn_on_new_tokens = attentions[i][l][0, :, -1, context_length:]
                attn_all_previous = attentions[i][l][0, :, -1, :]
                
                _, context_lap = laplacian_1d_batch(attn_on_context, return_score=True)
                _, new_tokens_lap = laplacian_1d_batch(attn_on_new_tokens, return_score=True)
                _, all_previous_lap = laplacian_1d_batch(attn_all_previous, return_score=True)
                
                context_laplacian_score[l, :, i] = context_lap
                new_tokens_laplacian_score[l, :, i] = new_tokens_lap
                all_previous_laplacian_score[l, :, i] = all_previous_lap

        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]

        to_save = {
            'data_index': sample['data_index'],
            'response': response_list[idx],
            'model_completion': model_completion,
            'model_completion_ids': model_completion_ids,
            'full_input_text': input_text,
            'context_laplacian_l2_score': context_laplacian_score,
            'new_tokens_laplacian_l2_score': new_tokens_laplacian_score,
            'all_previous_laplacian_l2_score': all_previous_laplacian_score,
        }
        to_save_list.append(to_save)

        if len(to_save_list) % 5 == 0:
            torch.save(to_save_list, args.output_path)
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save(to_save_list, args.output_path)
