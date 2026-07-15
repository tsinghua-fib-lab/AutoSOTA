import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_FILE = "data/induction_heads_phi35_mini.json"

def get_induction_score(model, tokenizer, seq_len=300, num_examples=10):
    """
    Computes induction score for all heads.
    """
    print(f"Loading model {MODEL_NAME} (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Generate random repeated sequences
    # "Random tokens A B C ... A"
    # To simplify, let's use a meaningful repeated string structure
    # "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
    vocab_size = model.config.vocab_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Accumulate scores: (Layer, Head) -> Score
    head_scores = torch.zeros(num_layers, num_heads)
    
    print("Computing Induction Scores...")
    
    for _ in tqdm(range(num_examples)):
        # Create a random sequence of length seq_len // 2
        # Then repeat it.
        half_len = seq_len // 2
        rand_tokens = torch.randint(100, vocab_size-100, (1, half_len))
        input_ids = torch.cat([rand_tokens, rand_tokens], dim=1).to(model.device)
        
        # Run forward pass, getting all attentions
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs.attentions # Tuple of (batch, heads, seq, seq)
            
        # Analyze attention
        # We focus on the second half of the sequence (the repetition)
        # For a token at `i` (in 2nd half), it matches token at `i - half_len`.
        # Induction head should attend to `(i - half_len) + 1`.
        
        # Indices in 2nd half: range(half_len, 2 * half_len - 1)
        # Why -1? predicting next token? No, attention at step `i` looks back.
        # If input is [A1, B1, ..., A2, B2]
        # At step corresponding to A2 (index `half_len`), we want to predict B2.
        # The head processes A2. It looks back. It sees A1 at index 0.
        # It should attend to index 0+1 = 1 (value B1).
        # So for query at `i` (where input_ids[i] == input_ids[i - half_len]),
        # target attention index is `i - half_len + 1`.
        
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: [1, num_heads, seq_len, seq_len]
            # Squeeze batch
            layer_attn = layer_attn[0] # [num_heads, seq, seq]
            
            # Collect score for this example
            # Iterate over positions in the repeated part
            # Let's verify start/end.
            # Sequence: T_0 ... T_{L-1} T_0 ... T_{L-1}
            # Length = 2L.
            # Query at index `L`. input[L] is T_0.
            # Previous occurrence of T_0 was at index 0.
            # Induction head pattern: Attend to `previous_token_pos + 1`.
            # If current is T_0 (at L), it looks for T_0 (at 0), attends to 0+1=1 (T_1).
            # This allows copying T_1 to be the output.
            
            # So for query idx `q` in [half_len, 2*half_len - 2]:
            # (Stop at -2 because if at very last token, 'previous+1' might be out of causal bounds if valid)
            # Actually, standard induction:
            # Query at `q` (content T). Prev T at `p`. Attend to `p+1`.
            
            scores = []
            valid_queries = range(half_len, 2 * half_len - 1)
            
            target_indices = torch.tensor([q - half_len + 1 for q in valid_queries]).to(model.device)
            query_indices = torch.tensor(list(valid_queries)).to(model.device)
            
            # Gather attention values at specific target indices
            # layer_attn: [heads, seq, seq]
            # select columns: target_indices
            # select rows: query_indices
            
            # Advanced indexing:
            # We want attn[:, q, t] for each q,t pair
            
            # Let's loop heads for clarity or use vectorization
            # batch_attn: [heads, num_queries, seq]
            # sliced = layer_attn[:, query_indices, :]
            # values = sliced[:, range(len(query_indices)), target_indices] ?
            
            # Simple loop to be safe with torch indexing
            for h in range(num_heads):
                head_attn = layer_attn[h] # [seq, seq]
                
                # Extract diagonal-ish entries
                # row `q`, col `q - half_len + 1`
                attn_vals = head_attn[query_indices, target_indices]
                
                # Mean attention to induction token
                head_scores[layer_idx, h] += attn_vals.mean().item()

    # Average over examples
    head_scores /= num_examples
    
    # Flatten and sort
    # list of (layer, head, score)
    all_heads = []
    for l in range(num_layers):
        for h in range(num_heads):
            all_heads.append({
                "layer": l,
                "head": h,
                "score": float(head_scores[l, h])
            })
            
    # Sort descending
    all_heads.sort(key=lambda x: x['score'], reverse=True)
    
    # Top 50
    top_50 = all_heads[:50]
    
    print("\nTop 10 Induction Heads:")
    for i in range(10):
        print(f"Rank {i+1}: L{top_50[i]['layer']} H{top_50[i]['head']} (Score: {top_50[i]['score']:.4f})")
        
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(top_50, f, indent=2)
    print(f"Saved top 50 heads to {OUTPUT_FILE}")

if __name__ == "__main__":
    import argparse
    # allow simple run
    main_func = get_induction_score
    main_func(None, None) 
