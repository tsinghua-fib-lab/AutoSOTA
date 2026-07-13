import os
import torch
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype_attn import ProtoBroadcastLM
from llama_baseline import create_llama31_from_args
from mamba import create_mamba_from_args
from deltanet import create_deltanet_from_args
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_tokenizer(path):
    try:
        if path.endswith('.json'):
            tokenizer_obj = Tokenizer.from_file(path)
            tokenizer_obj.decoder = BPEDecoder(suffix="</w>")
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
            tokenizer.pad_token = "<pad>"
            tokenizer.bos_token = "<bos>"
            tokenizer.eos_token = "<eos>"
            tokenizer.unk_token = "<unk>"
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
        return tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {path}: {e}")

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

def load_protot_model(path, device, max_len=None):
    args, state_dict = load_checkpoint(path, device)
    
    # Handle args being a dict or Namespace
    if isinstance(args, dict):
        a = SimpleNamespace(**args)
    else:
        a = args
        
    vocab_size = getattr(a, 'VOCAB_SIZE', 16000)
    emb_dim = getattr(a, 'EMB_DIM', getattr(a, 'BOTTLENECK', 384))
    layers = getattr(a, 'LAYERS', 6)
    r = getattr(a, 'R', 32)
    seq_len = getattr(a, 'SEQ_LEN', 256)
    
    if max_len is not None and max_len > seq_len:
        print(f"Overriding model SEQ_LEN {seq_len} with {max_len} for benchmarking")
        seq_len = max_len
    
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
    return model

def load_llama_model(path, device, tokenizer, max_len=None):
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

    if max_len is not None and max_len > a.SEQ_LEN:
        print(f"Overriding model SEQ_LEN {a.SEQ_LEN} with {max_len} for benchmarking")
        a.SEQ_LEN = max_len

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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"LLaMA checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
        
    model.to(device)
    
    model.eval()
    return model

def load_mamba_model(path, device):
    args, state_dict = load_checkpoint(path, device)
    
    if isinstance(args, dict):
        a = SimpleNamespace(
            BOTTLENECK=args.get('BOTTLENECK', args.get('model_dim', 384)),
            LAYERS=args.get('LAYERS', args.get('num_layers', 6)),
            VOCAB_SIZE=args.get('VOCAB_SIZE', args.get('vocab_size', 16000)),
            DEVICE=device
        )
    else:
        a = args
        a.DEVICE = device

    model = create_mamba_from_args(a, pad_idx=0)
    model.load_state_dict(state_dict)
    model.to(device)
    
    model.eval()
    return model

def load_deltanet_model(path, device, tokenizer, max_len=None):
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
        
    if max_len is not None and max_len > getattr(a, 'SEQ_LEN', 0):
        print(f"Overriding model SEQ_LEN {getattr(a, 'SEQ_LEN', 0)} with {max_len} for benchmarking")
        a.SEQ_LEN = max_len
        
    tf_ffn_ratio = getattr(a, 'TF_FFN_RATIO', 2.7)
    if isinstance(args, dict):
        tf_ffn_ratio = args.get('TF_FFN_RATIO', 2.7)
        
    tf_ffn_size = int(tf_ffn_ratio * a.BOTTLENECK)
    tf_ffn_size = (tf_ffn_size // 16) * 16
    
    model = create_deltanet_from_args(a, PAD_IDX=0, TF_FFN_SIZE=tf_ffn_size, tok=TokWrapper(tokenizer))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DeltaNet checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    
    model.eval()
    return model


def benchmark_throughput(model, device, context_len, num_steps=50, warmup=5):
    vocab_size = getattr(model, "vocab_size", 1000)
    input_ids = torch.randint(0, vocab_size, (1, context_len), device=device)
    pad_mask = torch.zeros((1, context_len), dtype=torch.bool, device=device)

    print(f"Warming up for {warmup} steps...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids, pad_mask=pad_mask)
    torch.cuda.synchronize()
            
    print(f"Benchmarking for {num_steps} steps...")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_steps):
            model(input_ids, pad_mask=pad_mask)
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    it_per_s = 1.0 / avg_time_per_step
    
    return it_per_s

def aggregate_results(files):
    all_results = []
    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            all_results.extend(data)
            
    all_results.sort(key=lambda x: (x['model'], x['context_length']))
    
    print("\nAggregated Results Summary:")
    print(f"{'Model':<30} {'Context':<10} {'Throughput (it/s)':<20}")
    print("-" * 60)
    for res in all_results:
        print(f"{res['model']:<30} {res['context_length']:<10} {res['throughput']:<20.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Paths to model checkpoints')
    parser.add_argument('--names', nargs='+', help='Names of models')
    parser.add_argument('--tokenizer', type=str, default='tok/fineweb_bpe_16000.json')
    parser.add_argument('--context_lengths', nargs='+', type=int, default=[2048, 4096, 8192, 16384], help='Context lengths to benchmark')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps to measure')
    parser.add_argument('--output', type=str, default='throughput_results.json', help='Output JSON file')
    parser.add_argument('--aggregate', nargs='+', help='List of JSON files to aggregate and print')
    args = parser.parse_args()
    
    if args.aggregate:
        aggregate_results(args.aggregate)
        return

    if not args.models or not args.names:
        print("Error: --models and --names are required unless --aggregate is used.")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = load_tokenizer(args.tokenizer)
    
    results = []
    
    max_benchmark_len = max(args.context_lengths) if args.context_lengths else 2048
    
    for model_path, model_name in zip(args.models, args.names):
        print(f"Loading {model_name}...")
        
        if 'llama' in model_name.lower():
            model = load_llama_model(model_path, device, tokenizer, max_len=max_benchmark_len)
        elif 'mamba' in model_name.lower():
            model = load_mamba_model(model_path, device)
        elif 'delta' in model_name.lower():
            model = load_deltanet_model(model_path, device, tokenizer, max_len=max_benchmark_len)
        else:
            model = load_protot_model(model_path, device, max_len=max_benchmark_len)
            
        print(f"Benchmarking {model_name}...")
        
        for seq_len in args.context_lengths:
            try:
                print(f"  Context Length: {seq_len}")
                it_s = benchmark_throughput(model, device, seq_len, num_steps=args.steps)
                print(f"  Result: {it_s:.2f} it/s")
                results.append({
                    'model': model_name,
                    'context_length': seq_len,
                    'throughput': it_s
                })
            except RuntimeError as e:
                print(f"  OOM or Error at length {seq_len}: {e}")
                results.append({
                    'model': model_name,
                    'context_length': seq_len,
                    'throughput': 0.0,
                    'error': str(e)
                })
                
        del model
        torch.cuda.empty_cache()
        
    print("\nResults Summary:")
    print(f"{'Model':<30} {'Context':<10} {'Throughput (it/s)':<20}")
    print("-" * 60)
    for res in results:
        print(f"{res['model']:<30} {res['context_length']:<10} {res['throughput']:<20.2f}")
        
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
