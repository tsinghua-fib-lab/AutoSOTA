import os
import torch
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer
import pickle as pkl
from tqdm import tqdm

from baukit import TraceDict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True, help="Path to the collected_activations.json")
    parser.add_argument("--entropy_min", type=float, default=0.0)
    parser.add_argument("--entropy_max", type=float, default=10.0)
    parser.add_argument("--period_min", type=int, default=1)
    parser.add_argument("--period_max", type=int, default=10)
    parser.add_argument("--write_size", type=int, default=100)
    args = parser.parse_args()

    if "Qwen3" in args.model:
        from custom_hf.modeling_qwen3 import CustomQwen3ForCausalLM
        model = CustomQwen3ForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif "llama" in args.model.lower():
        from custom_hf.modeling_llama import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    ATTENTIONS = [f"model.layers.{i}.self_attn.attn_retain" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    LAYERS = [f"model.layers.{i}.layer_retain" for i in range(model.config.num_hidden_layers)]

    with open(args.json_path, "r") as f:
        all_data = json.load(f)
    
    buffer = []
    json_dir = os.path.dirname(args.json_path)
    output_dir = os.path.join("/".join(json_dir.split("/")[:-1]), "activations")
    output_path = os.path.join(output_dir, f"activations_entropy_{args.entropy_min}_{args.entropy_max}_period_{args.period_min}_{args.period_max}.pkl")
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in tqdm(enumerate(all_data), desc="Collecting activations", total=len(all_data)):
        for chunk in sample['chunks']:
            entropy = chunk['ori_chunk_entropy']
            period_idx = chunk['period_idx']
            input_texts = []
            if args.entropy_min <= entropy <= args.entropy_max and args.period_min <= period_idx <= args.period_max:
                scores = []
                for pos_text in chunk['positive_completions']:
                    input_texts.append(chunk['context'] + pos_text)
                    scores.append(1.0)
                for neg_text in chunk['negative_completions']:
                    input_texts.append(chunk['context'] + neg_text)
                    scores.append(0.0)
                for input_text, score in zip(input_texts, scores):
                    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
                    with TraceDict(model, ATTENTIONS + MLPS + LAYERS) as ret:
                        _ = model(input_ids=input_ids)
                    
                    attn_acts = torch.stack([ret[a].output[0, -1, :].detach().cpu().to(torch.float32) for a in ATTENTIONS]).numpy()
                    mlp_acts = torch.stack([ret[m].output[0, -1, :].detach().cpu().to(torch.float32) for m in MLPS]).numpy()
                    layer_acts = torch.stack([ret[l].output[0, -1, :].detach().cpu().to(torch.float32) for l in LAYERS]).numpy()

                    buffer.append({
                        "problem_id": sample['problem_id'],
                        "chunk_idx": chunk['chunk_idx'],
                        "score": score,
                        "entropy": entropy,
                        "period_idx": period_idx,
                        "attention_activations": attn_acts,
                        "mlp_activations": mlp_acts,
                        "layer_activations": layer_acts
                    })
        if len(buffer) >= args.write_size:
            with open(output_path, "ab") as f:
                pkl.dump(buffer, f)
            buffer.clear()
    if buffer:
        with open(output_path, "ab") as f:
            pkl.dump(buffer, f)
        buffer.clear()
        
    if os.path.exists(output_path):
        print(f"Consolidating data into a single object...")
        final_list = []
        with open(output_path, "rb") as f:
            while True:
                try:
                    final_list.extend(pkl.load(f))
                except EOFError:
                    break
        
        temp_output_path = output_path + ".tmp"
        with open(temp_output_path, "wb") as f:
            pkl.dump(final_list, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        os.replace(temp_output_path, output_path)
        print(f"Done! Final file size: {len(final_list)} samples.")
    

            
                
    

    
    

