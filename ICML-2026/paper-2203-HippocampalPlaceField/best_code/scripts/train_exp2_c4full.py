import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import math
import random
import numpy as np
import wandb
import subprocess
import sys
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


def get_git_info():
    git_info = {}
    try:
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        git_info["short_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        git_info["is_dirty"] = len(git_status) > 0
        git_info["dirty_files"] = git_status if git_info["is_dirty"] else "None"
        
        git_info["remote_url"] = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        
        if git_info["remote_url"].startswith("git@"):
            git_info["github_commit_url"] = git_info["remote_url"].replace(
                "git@github.com:", "https://github.com/"
            ).replace(".git", "") + f"/commit/{git_info['commit_hash']}"
        elif git_info["remote_url"].startswith("https"):
            git_info["github_commit_url"] = git_info["remote_url"].replace(
                ".git", ""
            ) + f"/commit/{git_info['commit_hash']}"
        else:
            git_info["github_commit_url"] = "Unknown"
            
    except subprocess.CalledProcessError as e:
        git_info["error"] = f"Git command failed: {e.output.decode('utf-8')}"
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    except Exception as e:
        git_info["error"] = f"Get git info failed: {str(e)}"
        git_info["commit_hash"] = "unknown"
        git_info["short_commit"] = "unknown"
    
    return git_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)

    parser.add_argument("--dataset_path", type=str, required=True, help="Path containing c4_30M_train/val")
    parser.add_argument("--train_size", type=int, default=5000000)
    parser.add_argument("--val_size", type=int, default=10000)
    
    parser.add_argument("--local_tokenizer_path", type=str, required=True)
    
    parser.add_argument("--model_size", type=str, default="20M", choices=["20M", "60M", "300M"])

    # Local/global attention parameters.
    parser.add_argument("--local_window_size", type=int, default=-1, help="Local attention window size")
    parser.add_argument("--num_local_layers", type=int, default=0, help="Number of layers using local attention")

    parser.add_argument("--alibi", action="store_true")
    parser.add_argument("--fope", action="store_true")
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--nope", action="store_true") # NoPE
    parser.add_argument("--xpos", action="store_true") # XPos
    parser.add_argument("--rope_scale", type=float, default=None)
    
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rope_scaling_threshold", type=int, default=-1)
    parser.add_argument("--sigma_list", nargs='+', default=None)
    parser.add_argument("--learnable_sigma", action="store_true", help="Enable learnable sigma for HIPE")
    
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=8) 
    parser.add_argument("--max_tokens", type=int, default=100_000_000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    
    parser.add_argument("--wandb_mode", type=str, default="offline", help="Wandb mode")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb offline tracking directory")
    parser.add_argument("--num_proc", type=int, default=8, help="Processes for dataset preprocessing")

    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    git_info = get_git_info()

    run_tags = [args.model_size, f"len_{args.seq_len}", f"seed_{args.seed}"]
    run_tags.append(f"commit_{git_info['short_commit']}")
    run_tags.append(f"dirty_{git_info['is_dirty']}" if "is_dirty" in git_info else "dirty_unknown")

    if args.alibi or args.xpos or args.fope or args.nope or not args.use_scaled_rope:
        run_group = "Exp2-C4-Baselines"
        run_tags.append("baseline")
    else:
        run_group = "Exp2-C4-HIPE"
        run_tags.append("hipe")
        run_tags.append(f"sigma_{args.sigma}")
        if args.learnable_sigma:
            run_tags.append("learnable_sigma")

    if args.wandb_dir is not None:
        os.makedirs(args.wandb_dir, exist_ok=True)

    wandb.init(
        project="Position Embedding",
        group=run_group,
        tags=run_tags,
        name=args.run_id,
        config=vars(args),
        dir=args.wandb_dir if args.wandb_dir else args.output_dir,
        mode=args.wandb_mode
    )

    wandb.config.update(git_info)

    final_sigmas = None
    if args.sigma_list is not None:
        final_sigmas = []
        for s in args.sigma_list:
            if s == "None": final_sigmas.append(None)
            else: final_sigmas.append(float(s))

    print(f"Loading Tokenizer from {args.local_tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, local_files_only=True)
    except Exception as e:
        print(f"Error loading AutoTokenizer: {e}")
        from OLMo.olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained(args.local_tokenizer_path, eos_token_id=50256, pad_token_id=50256)
    
    raw_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    vocab_size = ((raw_vocab_size + 63) // 64) * 64
    print(f">>> Resizing Vocab: {raw_vocab_size} -> {vocab_size} (Aligned to 64)")
    wandb.config.update({"actual_vocab_size": vocab_size})

    print(f"Loading C4 Data from: {args.dataset_path}")
    train_path = os.path.join(args.dataset_path, "c4_30M_train")
    val_path = os.path.join(args.dataset_path, "c4_30M_validation")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_full = load_from_disk(train_path)
    val_full = load_from_disk(val_path)

    real_train_size = min(args.train_size, len(train_full))
    real_val_size = min(args.val_size, len(val_full))
    print(f"Selecting subset: Train={real_train_size}, Val={real_val_size}")
    
    train_ds = train_full.select(range(real_train_size))
    val_ds = val_full.select(range(real_val_size))

    print(">>> Pre-tokenizing dataset (this improves speed significantly)...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    train_cols = train_ds.column_names
    val_cols = val_ds.column_names

    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=train_cols, num_proc=args.num_proc, desc="Tokenizing Train")
    tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=val_cols, num_proc=args.num_proc, desc="Tokenizing Val")

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

    lm_train = tokenized_train.map(group_texts, batched=True, num_proc=args.num_proc, desc="Grouping Train")
    lm_val = tokenized_val.map(group_texts, batched=True, num_proc=args.num_proc, desc="Grouping Val")

    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        data = torch.tensor(input_ids, dtype=torch.long)
        x = data[:, :-1].contiguous() 
        y = data[:, 1:].contiguous()  
        return x, y
    
    print(f">>> Processed Dataset: Train={len(lm_train)} chunks, Val={len(lm_val)} chunks")

    train_loader = DataLoader(lm_train, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(lm_val, batch_size=args.micro_batch_size, collate_fn=collate_fn)

    grad_accum_steps = args.global_batch_size // args.micro_batch_size

    if args.max_train_steps is not None and args.max_train_steps > 0:
        total_steps = args.max_train_steps
        print(f">>> [DEBUG MODE] Training for fixed steps: {total_steps}")
    else:
        total_steps = args.max_tokens // (args.global_batch_size * args.seq_len)
        print(f">>> [FULL MODE] Training for max tokens: {args.max_tokens} (~{total_steps} steps)")

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

    use_alibi = False
    use_fope = False
    use_yarn = False
    use_nope = False 
    use_xpos = False 
    use_rope = True 
    rope_scaling_config = None 
    
    use_flash_attention = True

    if args.local_window_size > 0:
        use_flash_attention = False
        print(f">>> [Config] Local Attention ENABLED | Window: {args.local_window_size} | FlashAttention DISABLED")

    if args.alibi:
        use_alibi = True
        use_rope = False 
        use_flash_attention = False
        print(">>> [Config] ALiBi ENABLED | RoPE DISABLED | FlashAttention DISABLED")
    elif args.fope:
        use_fope = True
        scale = args.rope_scale if args.rope_scale else max(1.0, args.seq_len / 512.0)
        rope_scaling_config = {"type": "linear", "factor": scale}
        print(f">>> [Config] FoPE (Linear Scaling) ENABLED | Scale: {scale}")
    elif args.yarn:
        use_yarn = True
        print(f">>> [Config] YaRN ENABLED | Target Len: {args.seq_len}")
    elif args.nope:
        use_nope = True
        use_rope = False
        use_fope = False
        print(">>> [Config] NoPE ENABLED | RoPE DISABLED")
    elif args.xpos:
        use_xpos = True
        use_rope = False 
        print(">>> [Config] XPos ENABLED | RoPE DISABLED")

    cfg = ModelConfig(
        d_model=cur_d, 
        n_heads=cur_h, 
        n_layers=cur_l, 
        mlp_ratio=cur_mlp,
        max_sequence_length=args.seq_len,
        vocab_size=vocab_size,
        embedding_size=vocab_size, 
        init_std=init_std_val,
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
        learnable_sigma=args.learnable_sigma,
        flash_attention=use_flash_attention,
        local_window_size=args.local_window_size,
        num_local_layers=args.num_local_layers
    )

    if rope_scaling_config is not None:
        cfg.rope_scaling = rope_scaling_config
        if args.fope:
             cfg.scaled_rope_sigma = rope_scaling_config["factor"]

    if use_nope: cfg.nope = True
    if use_xpos: cfg.xpos = True
    
    model = OLMo(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Experiment: {args.run_id} -> Saving to {args.output_dir}")
    print(f">>> Git Commit: {git_info['commit_hash']} (short: {git_info['short_commit']})")
    print(f">>> Git Branch: {git_info.get('branch', 'unknown')}")
    print(f">>> Code Dirty: {git_info.get('is_dirty', 'unknown')}")

    model.train()
    step = 0
    total_loss = 0.0 
    
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    
    log_path = os.path.join(args.output_dir, "log.txt")
    log_file = open(log_path, "w")
    log_file.write(f"Git Commit: {git_info['commit_hash']}\n")
    log_file.write(f"Git Short Commit: {git_info['short_commit']}\n")
    log_file.write(f"Git Branch: {git_info.get('branch', 'unknown')}\n")
    log_file.write(f"Code Dirty: {git_info.get('is_dirty', 'unknown')}\n")
    log_file.write(f"GitHub Commit URL: {git_info.get('github_commit_url', 'unknown')}\n")
    log_file.write("Step,Loss,PPL\n")
    log_file.flush()

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
        
        if step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_path = os.path.join(ckpt_dir, f"model_step_{step}.pt")
            
            torch.save(model.state_dict(), save_path)
            print(f">>> Checkpoint saved at step {step}: {save_path}")


        if step % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.4f} | LR: {lr:.2e}")

            if args.learnable_sigma and args.use_scaled_rope:
                sigma_values = []
                for name, param in model.named_parameters():
                    if "sigma" in name:
                        sigma_values.extend(param.detach().cpu().numpy().tolist())
                if sigma_values:
                    wandb.log({"train/avg_sigma": sum(sigma_values) / len(sigma_values), "step": step})

            wandb.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/lr": lr,
                "step": step
            })

            log_file.write(f"Step {step},Loss: {avg_loss:.4f},PPL: {ppl:.4f}\n")
            log_file.flush()
            total_loss = 0.0

        if step % args.eval_interval == 0:
            print(">>> Running Validation...")
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
                avg_val_loss = total_val_loss / total_val_tokens
                val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else 1e9 
                
                print(f">>> VAL PPL: {val_ppl:.4f}")

                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/ppl": val_ppl,
                    "step": step
                })

                log_file.write(f"VAL,{step},{avg_val_loss},{val_ppl}\n")
                log_file.flush() 
                
            model.train()

    print("Saving model checkpoint...")
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path, base_path=args.output_dir)
    print("Training Finished.")
    log_file.close()
    
    wandb.finish()

if __name__ == "__main__":
    main()
