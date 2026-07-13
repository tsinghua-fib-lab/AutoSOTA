import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.text_generation_performance import metrics

from prototype_attn import ProtoBroadcastLM
from llama_baseline import create_llama31_from_args
from mamba import create_mamba_from_args
from deltanet import create_deltanet_from_args

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder

def load_tokenizer(path):
    try:
        if path.endswith('.json'):
            tokenizer_obj = Tokenizer.from_file(path)
            # Configure BPEDecoder to handle </w> suffix
            tokenizer_obj.decoder = BPEDecoder(suffix="</w>")
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
            # Set special tokens manually based on our inspection
            tokenizer.pad_token = "<pad>"
            tokenizer.bos_token = "<bos>"
            tokenizer.eos_token = "<eos>"
            tokenizer.unk_token = "<unk>"
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
        return tokenizer

    except Exception as e:
        print(f"Failed to load tokenizer from {path}: {e}")
        return AutoTokenizer.from_pretrained("gpt2")

class TokWrapper:
    def __init__(self, tok):
        self.specials = {
            '<pad>': tok.pad_token_id if tok.pad_token_id is not None else 0,
            '<sos>': tok.bos_token_id if tok.bos_token_id is not None else 1,
            '<bos>': tok.bos_token_id if tok.bos_token_id is not None else 1,
            '<eos>': tok.eos_token_id if tok.eos_token_id is not None else 2
        }

def load_checkpoint(path, device):
    if os.path.isdir(path):
        trial_path = os.path.join(path, 'trial000')
        if os.path.exists(trial_path):
            path = trial_path
            
        args_path = os.path.join(path, 'args.json')
        weights_path = os.path.join(path, 'model_state_dict.pth')
        
        if not os.path.exists(args_path) or not os.path.exists(weights_path):
            raise ValueError(f"Could not find args.json or model_state_dict.pth in {path}")
            
        print(f"Loading args from {args_path}")
        with open(args_path, 'r') as f:
            args = json.load(f)
            
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        
        return args, state_dict
    else:
        print(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=device)
        return ckpt['args'], ckpt['model_state_dict']

def load_protot_model(path, device):
    args, state_dict = load_checkpoint(path, device)
    
    if isinstance(args, dict):
        a = SimpleNamespace(**args)
    else:
        a = args
        
    vocab_size = getattr(a, 'VOCAB_SIZE', 16000)
    emb_dim = getattr(a, 'EMB_DIM', getattr(a, 'BOTTLENECK', 384))
    layers = getattr(a, 'LAYERS', 6)
    r = getattr(a, 'R', 32)
    seq_len = getattr(a, 'SEQ_LEN', 256)
    
    ffn_inner_size = int(2.7 * emb_dim)
    ffn_inner_size = (ffn_inner_size // 16) * 16
    
    model = ProtoBroadcastLM(
        vocab_size=vocab_size,
        dim=emb_dim,
        depth=layers,
        r=r,
        max_seq_len=seq_len,
        ffn_inner_size=ffn_inner_size,
        dropout=0.0,
        pad_id=0,
        tie_weights=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, seq_len

def load_llama_model(path, device, tokenizer):
    args, state_dict = load_checkpoint(path, device)
    
    if isinstance(args, dict):
        a = SimpleNamespace(
            VOCAB_SIZE=args.get('VOCAB_SIZE', 16000),
            BOTTLENECK=args.get('BOTTLENECK', args.get('EMB_DIM', 384)),
            LAYERS=args.get('LAYERS', 6),
            HEADS=args.get('HEADS', 6),
            SEQ_LEN=args.get('SEQ_LEN', 256),
            TIE_HEAD=args.get('TIE_HEAD', True),
            DEVICE=device
        )
    else:
        a = args
        a.DEVICE = device


    
    ffn_key = None
    for key in state_dict.keys():
        if 'mlp.gate_proj.weight' in key:
            ffn_key = key
            break
    
    if ffn_key:
        tf_ffn_size = state_dict[ffn_key].shape[0]
        print(f"Inferred FFN size from checkpoint: {tf_ffn_size}")
    else:
        tf_ffn_ratio = args.get('TF_FFN_RATIO', 2.7) if isinstance(args, dict) else getattr(args, 'TF_FFN_RATIO', 2.7)
        tf_ffn_size = int(tf_ffn_ratio * a.BOTTLENECK)
        print(f"Using FFN size from ratio: {tf_ffn_size}")
    
    model = create_llama31_from_args(a, TokWrapper(tokenizer), tf_ffn_size)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, a.SEQ_LEN

def load_mamba_model(path, device):
    args, state_dict = load_checkpoint(path, device)
    
    if isinstance(args, dict):
        a = SimpleNamespace(
            BOTTLENECK=args.get('BOTTLENECK', args.get('model_dim', 384)),
            LAYERS=args.get('LAYERS', args.get('num_layers', 6)),
            VOCAB_SIZE=args.get('VOCAB_SIZE', args.get('vocab_size', 16000)),
            DEVICE=device
        )
        seq_len = args.get('seq_length', 256)
    else:
        a = args
        a.DEVICE = device
        seq_len = getattr(args, 'SEQ_LEN', 256)

    model = create_mamba_from_args(a, pad_idx=0)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, seq_len

def load_deltanet_model(path, device, tokenizer):
    args, state_dict = load_checkpoint(path, device)
    
    if isinstance(args, dict):
        a = SimpleNamespace(
            VOCAB_SIZE=args.get('VOCAB_SIZE', 16000),
            BOTTLENECK=args.get('BOTTLENECK', args.get('EMB_DIM', 384)),
            LAYERS=args.get('LAYERS', 6),
            HEADS=args.get('HEADS', 4),
            SEQ_LEN=args.get('SEQ_LEN', 256),
            TIE_HEAD=args.get('TIE_HEAD', True),
            DEVICE=device
        )
    else:
        a = args
        a.DEVICE = device
        
    tf_ffn_ratio = getattr(a, 'TF_FFN_RATIO', 2.7)
    if isinstance(args, dict):
        tf_ffn_ratio = args.get('TF_FFN_RATIO', 2.7)
        
    tf_ffn_size = int(tf_ffn_ratio * a.BOTTLENECK)
    tf_ffn_size = (tf_ffn_size // 16) * 16
    
    model = create_deltanet_from_args(a, PAD_IDX=0, TF_FFN_SIZE=tf_ffn_size, tok=TokWrapper(tokenizer))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, a.SEQ_LEN

def apply_repetition_penalty(logits, generated, penalty):
    if penalty is None or penalty <= 1.0:
        return logits
    logits = logits.clone()
    token_ids = generated[0].unique()
    for token_id in token_ids:
        token_id = token_id.item()
        token_logits = logits[:, token_id]
        positive = token_logits > 0
        token_logits = torch.where(positive, token_logits / penalty, token_logits * penalty)
        logits[:, token_id] = token_logits
    return logits


def mask_repeated_ngrams(logits, generated, ngram_size):
    if ngram_size is None or ngram_size <= 1:
        return logits
    cur_len = generated.size(1)
    if cur_len + 1 < ngram_size:
        return logits
    generated_list = generated[0].tolist()
    ngram_dict = {}
    for i in range(cur_len - ngram_size + 1):
        prefix = tuple(generated_list[i : i + ngram_size - 1])
        next_token = generated_list[i + ngram_size - 1]
        ngram_dict.setdefault(prefix, set()).add(next_token)
    prefix = tuple(generated_list[-(ngram_size - 1) :])
    banned = ngram_dict.get(prefix, set())
    if not banned:
        return logits
    logits = logits.clone()
    for token_id in banned:
        logits[:, token_id] = float('-inf')
    return logits


def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = logits / temperature

    if top_k and top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        kth_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth_values, torch.full_like(logits, float('-inf')), logits)

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        cutoff_mask = cumulative_probs > top_p
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
        cutoff_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff_mask, float('-inf'))
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    device='cuda',
    seq_len=256,
    input_ids=None,
    pad_id=0,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.05,
    no_repeat_ngram_size=4,
):
    if input_ids is None:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
    else:
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.size(1) >= seq_len:
                break
            
            pad_mask = (generated == pad_id)
            outputs = model(generated, pad_mask=pad_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            next_token_logits = logits[:, -1, :]
            next_token_logits = apply_repetition_penalty(next_token_logits, generated, repetition_penalty)
            next_token_logits = mask_repeated_ngrams(next_token_logits, generated, no_repeat_ngram_size)
            next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    new_tokens = generated[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model checkpoints')
    parser.add_argument('--names', nargs='+', required=True, help='Names of models')
    parser.add_argument('--tokenizer', type=str, default='tok/fineweb_bpe_16000.json')
    parser.add_argument('--dataset', type=str, default='cnn_dailymail')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--output', type=str, default='generation_samples.json')
    parser.add_argument('--temperature', type=float, default=0.8, help='Softmax temperature for sampling')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k filtering (0 disables)')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p / nucleus sampling cutoff (1 disables)')
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help='Penalty >1.0 discourages reused tokens')
    parser.add_argument('--no_repeat_ngram', type=int, default=4, help='Block repeating n-grams (<=1 disables)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Dataset
    print(f"Loading dataset {args.dataset}...")
    try:
        if args.dataset.endswith('.npz'):
            data = np.load(args.dataset)
            tokens = data['tokens'] # numpy array
            
            chunk_size = 200
            prompt_len = 100
            
            samples = []
            num_needed = args.num_samples

            for i in range(0, len(tokens) - chunk_size, chunk_size):
                if len(samples) >= num_needed:
                    break
                    
                chunk = tokens[i : i + chunk_size]
                prompt_tokens = chunk[:prompt_len]
                ref_tokens = chunk[prompt_len:]
                
                prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                
                if len(prompt_text) < 10 or len(ref_text) < 10:
                    continue
                    
                samples.append({
                    'prompt': prompt_text, 
                    'reference': ref_text, 
                    'prompt_ids': torch.tensor(prompt_tokens, dtype=torch.long)
                })
        elif args.dataset.endswith('.json'):
            with open(args.dataset, 'r') as f:
                records = json.load(f)
            samples = []
            for item in records:
                if len(samples) >= args.num_samples:
                    break
                prompt = item.get('prompt', '').strip()
                reference = item.get('reference', '').strip()
                if not prompt:
                    continue
                samples.append({'prompt': prompt, 'reference': reference})
        elif args.dataset == 'cnn_dailymail':
            dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation', streaming=True)
            samples = []
            count = 0
            for item in dataset:
                if count >= args.num_samples:
                    break
                
                if 'article' in item:
                    text = item['article'][:500]
                    ref = item['highlights']
                elif 'text' in item:
                    text = item['text'][:200]
                    ref = item['text'][200:400]
                    if len(text) < 50: continue
                else:
                    continue
                    
                samples.append({'prompt': text, 'reference': ref})
                count += 1
        else:
            # Fallback to some text dataset
            dataset = load_dataset('wikitext', 'wikitext-2-v1', split='validation', streaming=True)
            samples = []
            count = 0
            for item in dataset:
                if count >= args.num_samples:
                    break
                
                if 'text' in item:
                    text = item['text'][:200]
                    ref = item['text'][200:400]
                    if len(text) < 50: continue
                else:
                    continue
                    
                samples.append({'prompt': text, 'reference': ref})
                count += 1
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    results = {'models': {}}
    
    for model_path, model_name in zip(args.models, args.names):
        print(f"Evaluating {model_name}...")
        
        if 'llama' in model_name.lower():
            model, max_len = load_llama_model(model_path, device, tokenizer)
        elif 'mamba' in model_name.lower():
            model, max_len = load_mamba_model(model_path, device)
        elif 'delta' in model_name.lower():
            model, max_len = load_deltanet_model(model_path, device, tokenizer)
        else:
            model, max_len = load_protot_model(model_path, device)
            
        model_results = []
        rouge_scores = []
        bleu_scores = []
        
        for sample in tqdm(samples):
            prompt = sample['prompt']
            reference = sample['reference']
            
            if 'prompt_ids' in sample:
                input_ids = sample['prompt_ids']
                if len(input_ids) > max_len - 50:
                    input_ids = input_ids[:max_len-50]
                    prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            else:
                prompt_tokens = tokenizer.encode(prompt)
                if len(prompt_tokens) > max_len - 50:
                    prompt = tokenizer.decode(prompt_tokens[:max_len-50]).replace('</w>', '')
                input_ids = None
            
            gen_only = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=50,
                device=device,
                seq_len=max_len,
                input_ids=input_ids,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram,
            )
            
            rouge = metrics.calculate_rouge_l(reference, gen_only)
            bleu = metrics.calculate_bleu(reference, gen_only)
            
            rouge_scores.append(rouge)
            bleu_scores.append(bleu)
            
            model_results.append({
                'prompt': prompt,
                'reference': reference,
                'generated': gen_only,
                'rouge': rouge,
                'bleu': bleu
            })
            
        avg_rouge = np.mean(rouge_scores)
        avg_bleu = np.mean(bleu_scores)
        
        print(f"{model_name} - ROUGE-L: {avg_rouge:.4f}, BLEU: {avg_bleu:.4f}")
        
        results['models'][model_name] = {
            'rouge_l': avg_rouge,
            'bleu': avg_bleu,
            'samples': model_results
        }
        
        del model
        torch.cuda.empty_cache()
        
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
