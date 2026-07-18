import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

def entropy_from_logits(logits: torch.Tensor):
    logits = logits.to(torch.float32)
    log_pd = torch.nn.functional.log_softmax(logits, dim=-1)
    pd = torch.exp(log_pd)
    return -torch.sum(pd * log_pd, dim=-1)

def get_chunk_end_token_indices(answer_token_ids, tokenizer, encountered_dot_tokens):
    chunk_end_indices = []
    for i, tid in enumerate(answer_token_ids):
        decoded_token = tokenizer.decode([tid])
        if '.\n' in decoded_token:
            chunk_end_indices.append(i + 1)
            if tid not in encountered_dot_tokens:
                encountered_dot_tokens[int(tid)] = decoded_token
    return chunk_end_indices

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--path", type=str, required=True, help="Original results_xk.jsonl")
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    total_dot_tokens = {}
    for tid in range(len(tokenizer)):
        decoded_token = tokenizer.decode([tid], skip_special_tokens=False)
        
        if '.\n' in decoded_token:
            total_dot_tokens[int(tid)] = decoded_token

    total_dot_tokens = dict(sorted(total_dot_tokens.items()))

    encountered_dot_tokens = {}
    data = []
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    chunk_data = []
    all_chunk_entropies = []
    chunk_counts = []

    print_encode_sample = True
    for item in tqdm(data, desc="Processing chunks"):
        if "Qwen3" in args.model:
            prompt_text = tokenizer.apply_chat_template(
                item["prompt"], 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=args.enable_thinking
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                item["prompt"], 
                tokenize=False, 
                add_generation_prompt=True, 
            )
        full_text = prompt_text + item["model_pred"]
        
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        # print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
    
        input_ids = inputs["input_ids"][0]
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # print(tokenizer.decode(prompt_ids, skip_special_tokens=False))
        prompt_len = len(prompt_ids)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, :, :]
            entropies = entropy_from_logits(logits).cpu().numpy()
        
        prompt_entropies = entropies[:prompt_len]
        vllm_entropies = np.concatenate([prompt_entropies, np.array(item["entropies"][:-1])])

        answer_token_ids = input_ids[prompt_len:].tolist()
        chunk_end_offsets = get_chunk_end_token_indices(answer_token_ids, tokenizer, encountered_dot_tokens)
    
        if not chunk_end_offsets:
            chunk_end_offsets = [len(answer_token_ids)]
        
        prev_offset = 0
        chunks_info = []
        for offset in chunk_end_offsets:
            if offset <= prev_offset:
                continue
                
            start_ent_idx = prompt_len + prev_offset - 1
            end_ent_idx = prompt_len + offset - 1
            
            chunk_entropy_vals = entropies[start_ent_idx : end_ent_idx]
            chunk_entropy_vals_vllm = vllm_entropies[start_ent_idx : end_ent_idx]
            avg_entropy = np.mean(chunk_entropy_vals) if len(chunk_entropy_vals) > 0 else 0
            avg_entropy_vllm = np.mean(chunk_entropy_vals_vllm) if len(chunk_entropy_vals_vllm) > 0 else 0
            
            chunk_text = tokenizer.decode(answer_token_ids[prev_offset:offset])
            
            chunks_info.append({
                "chunk_idx": len(chunks_info),
                "start_token_idx": prompt_len + prev_offset,
                "end_token_idx": prompt_len + offset,
                "avg_entropy": float(avg_entropy),
                "avg_entropy_vllm": float(avg_entropy_vllm),
                "text": chunk_text
            })
            all_chunk_entropies.append(avg_entropy)
            prev_offset = offset
            
        if prev_offset < len(answer_token_ids):
            chunk_entropy_vals = entropies[prompt_len + prev_offset - 1 : len(input_ids) - 1]
            avg_entropy = np.mean(chunk_entropy_vals) if len(chunk_entropy_vals) > 0 else 0
            avg_entropy_vllm = np.mean(vllm_entropies[prompt_len + prev_offset - 1 : len(input_ids) - 1]) if len(vllm_entropies[prompt_len + prev_offset - 1 : len(input_ids) - 1]) > 0 else 0
            chunks_info.append({
                "chunk_idx": len(chunks_info),
                "start_token_idx": prompt_len + prev_offset,
                "end_token_idx": len(input_ids),
                "avg_entropy": float(avg_entropy),
                "avg_entropy_vllm": float(avg_entropy_vllm),
                "text": tokenizer.decode(answer_token_ids[prev_offset:], skip_special_tokens=False)
            })

        chunk_counts.append(len(chunks_info))
        chunk_data.append({
            "problem": item.get("problem", ""),
            "ground_truth": item.get("ground_truth", ""),
            "pre_text": prompt_text,
            "model_pred": item["model_pred"],
            "score": item.get("score", 0),
            "chunks": chunks_info
        })

    save_dir = os.path.join(os.path.dirname(os.path.dirname(args.path)), "chunk_analysis")
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    with open(os.path.join(save_dir, "encountered_dot_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(encountered_dot_tokens.items())), f, indent=4, ensure_ascii=False)
    
    with open(os.path.join(save_dir, "total_dot_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(total_dot_tokens, f, indent=4, ensure_ascii=False)

    with open(os.path.join(save_dir, "chunk_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4, ensure_ascii=False)

    if chunk_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(chunk_counts, bins=max(1, max(chunk_counts)), color='skyblue', edgecolor='black', alpha=0.8)
        mean_v = np.mean(chunk_counts)
        median_v = np.median(chunk_counts)
        plt.title(f"Chunk Count per Response (Mean: {mean_v:.2f}, Median: {median_v:.1f})")
        plt.xlabel("Number of Chunks")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "chunk_count.png"))
        plt.close()

    if all_chunk_entropies:
        plt.figure(figsize=(10, 6))
        plt.hist(all_chunk_entropies, bins=30, color='orange', edgecolor='black', alpha=0.8)
        
        p90 = np.percentile(all_chunk_entropies, 90)
        plt.axvline(p90, color='red', linestyle='--', linewidth=2, label=f'90th Percentile: {p90:.3f}')

        p80 = np.percentile(all_chunk_entropies, 80)
        p85 = np.percentile(all_chunk_entropies, 85)
        p95 = np.percentile(all_chunk_entropies, 95)
        
        plt.title(f"Chunk Entropy Distribution\n"
                  f"80th: {p80:.3f}, 85th: {p85:.3f}, 90th: {p90:.3f}, 95th: {p95:.3f}")
        plt.xlabel("Entropy")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "chunk_entropy.png"))
        plt.close()
    
    if all_chunk_entropies:
        all_chunk_entropies_vllm = [chunk["avg_entropy_vllm"] for item in chunk_data for chunk in item["chunks"]]
        plt.figure(figsize=(10, 6))
        plt.hist(all_chunk_entropies_vllm, bins=30, color='green', edgecolor='black', alpha=0.8)
        
        p90_vllm = np.percentile(all_chunk_entropies_vllm, 90)
        plt.axvline(p90_vllm, color='red', linestyle='--', linewidth=2, label=f'90th Percentile: {p90_vllm:.3f}')

        p80_vllm = np.percentile(all_chunk_entropies_vllm, 80)
        p85_vllm = np.percentile(all_chunk_entropies_vllm, 85)
        p95_vllm = np.percentile(all_chunk_entropies_vllm, 95)
        
        plt.title(f"Chunk Entropy vLLM Distribution\n"
                  f"80th: {p80_vllm:.3f}, 85th: {p85_vllm:.3f}, 90th: {p90_vllm:.3f}, 95th: {p95_vllm:.3f}")
        plt.xlabel("Entropy vLLM")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "chunk_entropy_vllm.png"))
        plt.close()

    print(f"Analysis complete. Results: {save_dir}")