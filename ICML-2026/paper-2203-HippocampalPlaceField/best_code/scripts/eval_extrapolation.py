import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
import math
import argparse
import os
from torch.amp import autocast
from OLMo.olmo.config import TrainConfig
from OLMo.olmo.model import OLMo
from OLMo.olmo.tokenizer import Tokenizer
from transformers import AutoTokenizer

def evaluate_on_length(model, tokenizer, dataset, eval_len, device, num_samples=30):
    model.eval()
    
    def tokenize_and_chunk(examples):
        all_token_ids = []
        for text in examples.get("text", []):
            if text:
                all_token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        total_tokens = (len(all_token_ids) // eval_len) * eval_len
        if total_tokens == 0: return {"input_ids": []}
        return {"input_ids": [all_token_ids[i:i+eval_len] for i in range(0, total_tokens, eval_len)]}

    small_dataset = dataset.select(range(min(500, len(dataset))))
    processed_dataset = small_dataset.map(tokenize_and_chunk, batched=True, batch_size=100, remove_columns=dataset.column_names)
    
    if len(processed_dataset) == 0:
        print(f"Warning: No samples for len {eval_len}")
        return float('inf')

    final_dataset = processed_dataset.select(range(min(num_samples, len(processed_dataset))))
    
    def collate_fn(batch):
        return {"input_ids": torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)}

    loader = DataLoader(final_dataset, batch_size=1, collate_fn=collate_fn)

    total_loss = 0.0
    total_tokens = 0
    
    print(f"Running eval on {len(final_dataset)} samples of length {eval_len}...")
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:].contiguous()
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='sum')
            
            total_loss += loss.item()
            total_tokens += labels.numel()

    if total_tokens == 0: return float('inf')
    return math.exp(total_loss / total_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local tokenizer path")
    parser.add_argument("--lengths", nargs='+', type=int, default=[2048, 4096, 8192])

    parser.add_argument("--force_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=85.0)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = TrainConfig.load(args.config)
    

    if args.force_scaled_rope:
        print(f"!!! FORCING SCALED ROPE (Sigma={args.sigma}) !!!")
        cfg.model.use_scaled_rope1 = True
        cfg.model.scaled_rope_sigma = args.sigma
        cfg.model.decay_func = "exp"
    
    cfg.model.flash_attention = True 
    
    print(f"Loading from {args.checkpoint}")
    model = OLMo(cfg.model)
    

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if 'model' in state_dict: state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if args.tokenizer_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        except Exception:
            tokenizer = Tokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = Tokenizer.from_pretrained("allenai/olmo-1b")
    val_data = load_from_disk(args.data_path)

    print("-" * 40)
    for length in args.lengths:
        try:
            ppl = evaluate_on_length(model, tokenizer, val_data, length, device)
            print(f"Length: {length} | PPL: {ppl:.4f}")
        except Exception as e:
            print(f"Length: {length} | Error: {e}")
    print("-" * 40)

if __name__ == "__main__":
    main()
