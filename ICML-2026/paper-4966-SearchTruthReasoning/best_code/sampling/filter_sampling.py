import os
from argparse import ArgumentParser
import json
import math
import random 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the sampled_results.jsonl")
    parser.add_argument("--n_periods", type=int, default=3)
    parser.add_argument("--only_false", action="store_true")
    args = parser.parse_args()

    sampled_data_path = args.path

    filtered_chunks = []
    problem_max_chunk_idx = {}
    with open(sampled_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            problem_max_chunk_idx[data['problem_id']] = max(problem_max_chunk_idx.get(data['problem_id'], 0), data['chunk_idx'])
            flag = True
            if args.only_false:
                if data['original_score'] == 1.0:
                    flag = False
            if 0.0 in data['sampled_scores'] and 1.0 in data['sampled_scores'] and flag:
                positive_chunks = []
                negative_chunks = []
                for i, sampled_score in enumerate(data['sampled_scores']):
                    if sampled_score==1.0:
                        positive_chunks.append(data['sampled_chunk_text'][i])
                    elif sampled_score==0.0:
                        negative_chunks.append(data['sampled_chunk_text'][i])
                filtered_chunks.append(
                    {
                        "problem_id": data['problem_id'],
                        "chunk_idx": data['chunk_idx'],
                        "problem": data['problem'],
                        "ground_truth": data['ground_truth'],
                        "context": data['context'],
                        "ori_chunk_text": data['ori_chunk_text'],
                        "ori_chunk_entropy": data['ori_chunk_entropy'],
                        "original_score": data['original_score'],
                        "positive_completions": positive_chunks,
                        "negative_completions": negative_chunks
                    }
                )
    
    if len(filtered_chunks) > 3000:
        random.seed(42)
        print(f"Ori: {len(filtered_chunks)} chunks, sampled to 3000 chunks.")
        filtered_chunks = random.sample(filtered_chunks, 3000)
    
    filtered_data = {}
    for chunk in filtered_chunks:
        p_id = chunk['problem_id']
        if p_id not in filtered_data:
            filtered_data[p_id] = {
                "problem_id": p_id,
                "problem": chunk['problem'],
                "ground_truth": chunk['ground_truth'],
                "chunks": []
            }
        filtered_data[p_id]["chunks"].append({
            "chunk_idx": chunk['chunk_idx'],
            "context": chunk['context'],
            "ori_chunk_text": chunk['ori_chunk_text'],
            "ori_chunk_entropy": chunk['ori_chunk_entropy'],
            "period_idx": math.ceil((chunk['chunk_idx'] + 1) / (problem_max_chunk_idx[p_id] + 1) * args.n_periods),
            "original_score": chunk['original_score'],
            "positive_completions": chunk['positive_completions'],
            "negative_completions": chunk['negative_completions']
        })
    
    for p_id in filtered_data:
        filtered_data[p_id]["chunks"] = sorted(filtered_data[p_id]["chunks"], key=lambda x: x['chunk_idx'])

    total_chunks = 0
    for p_id in filtered_data:
        total_chunks += len(filtered_data[p_id]["chunks"])
    print(f"Total chunks: {total_chunks}")
    total_positive = sum(len(chunk['positive_completions']) for p in filtered_data.values() for chunk in p['chunks'])
    total_negative = sum(len(chunk['negative_completions']) for p in filtered_data.values() for chunk in p['chunks'])
    print(f"Total positive completions: {total_positive}")
    print(f"Total negative completions: {total_negative}")
    
    save_path = os.path.join(os.path.dirname(sampled_data_path), "filtered_sampling_results.json")
    save_data = list(filtered_data.values())
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    print(f"Filtered sampling results saved to {save_path}")
