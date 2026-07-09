import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import math
import random
import numpy as np
from torch.amp import autocast
from transformers import AutoTokenizer

# Imports from OLMo
from OLMo.olmo.config import ModelConfig
from OLMo.olmo.model import OLMo
from datasets import load_from_disk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--local_data_path", type=str, required=True)
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    parser.add_argument("--model_size", type=str, default="20M", choices=["20M", "60M", "300M"])


    parser.add_argument("--alibi", action="store_true")
    parser.add_argument("--fope", action="store_true")
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--nope", action="store_true")
    parser.add_argument("--xpos", action="store_true")
    parser.add_argument("--rope_scale", type=float, default=None)
    

    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=-1)
    parser.add_argument("--sigma_list", nargs='+', default=None)
    

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8) 
    parser.add_argument("--max_tokens", type=int, default=100_000_000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=20)


    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    final_sigmas = None
    if args.sigma_list is not None:
        final_sigmas = []
        for s in args.sigma_list:
            if s == "None": final_sigmas.append(None)
            else: final_sigmas.append(float(s))

    print(f"Loading Tokenizer from {args.local_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path)
    
    raw_vocab_size = tokenizer.vocab_size
    vocab_size = ((raw_vocab_size + 63) // 64) * 64

    dataset_dict = load_from_disk(args.local_data_path)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    tokenized_datasets = dataset_dict.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=4, desc="Tokenizing"
    )

    block_size = args.seq_len + 1
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4, desc="Grouping")

    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        data = torch.tensor(input_ids, dtype=torch.long)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()

    train_loader = DataLoader(lm_datasets['train'], batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(lm_datasets['validation'], batch_size=args.micro_batch_size, collate_fn=collate_fn)

    grad_accum_steps = args.global_batch_size // args.micro_batch_size

    if args.max_train_steps and args.max_train_steps > 0:
        total_steps = args.max_train_steps
    else:
        total_steps = args.max_tokens // (args.global_batch_size * args.seq_len)

    init_std_val = 0.02
    if args.model_size == "20M":
        cur_d, cur_h, cur_l, cur_mlp = 256, 8, 8, 8
    elif args.model_size == "60M":
        cur_d, cur_h, cur_l, cur_mlp = 512, 8, 8, 8
    elif args.model_size == "300M":
        cur_d, cur_h, cur_l, cur_mlp = 1024, 16, 16, 8
        print(">>> Using OLMo-300M Configuration (16 Layers, MLP Ratio 8)")
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # Baseline Configuration Logic
    use_alibi, use_fope, use_yarn, use_nope, use_xpos = False, False, False, False, False
    use_rope, use_flash_attention = True, True
    rope_scaling_config = None 

    if args.alibi:
        use_alibi, use_rope, use_flash_attention = True, False, False
    elif args.fope:
        use_fope = True
        scale = args.rope_scale if args.rope_scale else max(1.0, args.seq_len / 512.0)
        rope_scaling_config = {"type": "linear", "factor": scale}
    elif args.yarn:
        use_yarn = True
    elif args.nope:
        use_nope, use_rope, use_fope = True, False, False
    elif args.xpos:
        use_xpos, use_rope = True, False


    cfg = ModelConfig(
        d_model=cur_d, 
        n_heads=cur_h, 
        n_layers=cur_l, 
        mlp_ratio=cur_mlp,
        max_sequence_length=args.seq_len,
        vocab_size=vocab_size,
        embedding_size=vocab_size, 
        init_std=init_std_val,
        init_cutoff_factor=3,
        rope=use_rope,
        alibi=use_alibi,
        fope=use_fope, 
        yarn_enabled=use_yarn,
        yarn_target_max_position_embeddings=args.seq_len if use_yarn else None,
        yarn_max_position_embeddings=512, 
        use_scaled_rope1=args.use_scaled_rope,
        scaled_rope_sigma=args.sigma,
        scaled_rope_sigmas=final_sigmas,
        rope_scaling_threshold=args.rope_scaling_threshold,
        flash_attention=use_flash_attention
    )

    if rope_scaling_config:
        cfg.rope_scaling = rope_scaling_config
        if args.fope: cfg.scaled_rope_sigma = rope_scaling_config["factor"]

    if use_nope: cfg.nope = True
    if use_xpos: cfg.xpos = True
    
    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    model.train()
    step = 0
    total_loss = 0.0 
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    
    log_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    LOG_INTERVAL = 10 

    while step < total_steps:
        current_step_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=x)
                loss = nn.functional.cross_entropy(outputs.logits.view(-1, vocab_size), y.view(-1))
                loss = loss / grad_accum_steps
            
            loss.backward()
            current_step_loss += loss.item() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += current_step_loss
        step += 1
        
        if step % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
            log_file.write(f"{step},{avg_loss},{ppl}\n")
            log_file.flush()
            total_loss = 0.0

        if step % args.eval_interval == 0:
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids=vx)
                        loss = nn.functional.cross_entropy(
                            outputs.logits.view(-1, vocab_size), 
                            vy.view(-1), 
                            reduction='sum'
                        )
                    total_val_loss += loss.item()
                    total_val_tokens += vy.numel()
            
            if total_val_tokens > 0:
                avg_loss = total_val_loss / total_val_tokens
                print(f">>> VAL PPL: {math.exp(avg_loss):.2f}")
            model.train()

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    log_file.close()

if __name__ == "__main__":
    main()