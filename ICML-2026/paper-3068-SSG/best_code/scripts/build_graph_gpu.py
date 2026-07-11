import pickle
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm

def format_duration(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    print(f'Using device: {device}')
    t_total = time.perf_counter()

    # Load embeddings
    print(f'Loading embeddings from {args.input}...')
    data = torch.load(args.input, map_location='cpu', weights_only=False)
    indices = data['indices'].numpy()
    labels = data['labels'].numpy()
    embeddings = data['embeddings'].numpy()
    N, D = embeddings.shape
    print(f'Loaded {N} samples, dim={D}')

    # Build node list
    print('Building node list...')
    nodes = [{'idx': int(indices[i]), 'label': int(labels[i])} for i in range(N)]

    # Normalize and move to GPU
    print('Computing edge scores on GPU...')
    t0 = time.perf_counter()
    X = torch.from_numpy(embeddings).float().to(device)
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)

    # Compute all pairwise cosine similarities: [N, N]
    # Use batched approach to avoid OOM
    batch_size = 5000
    edge_weight_sum = torch.zeros(N, dtype=torch.float64, device='cpu')
    edge_count = torch.zeros(N, dtype=torch.int64, device='cpu')

    num_batches = (N + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc='Computing similarities'):
        start = b * batch_size
        end = min(start + batch_size, N)
        # [batch, N] similarity
        sim = X[start:end] @ X.T  # GPU matmul
        sim = sim.clamp(-1, 1)
        # For each node in batch, sum similarities with all other nodes
        # Exclude self-similarity (diagonal)
        batch_sum = sim.sum(dim=1)  # [batch]
        # Subtract self-similarity (which is 1.0)
        for i in range(end - start):
            global_i = start + i
            batch_sum[i] -= 1.0  # subtract self
        
        # For each node in batch
        for i in range(end - start):
            global_i = start + i
            edge_weight_sum[global_i] += (N - 1) - batch_sum[i].item()  # sum of (1 - sim)
            edge_count[global_i] += (N - 1)
        
        # Also accumulate for nodes NOT in this batch
        # sim[batch, all] contains similarities for batch nodes with ALL nodes
        # For nodes NOT in batch, we accumulate from the transpose perspective
        for i in range(end - start):
            global_i = start + i
            # For each j not in current batch range that's < global_i, contribute
            # This is handled through symmetric accumulation
        
        del sim
        torch.cuda.empty_cache()

    # Actually, the double loop approach above is still O(N^2).
    # Let me use a smarter approach: for each batch, compute batch_sim @ X.T
    # and accumulate in a vectorized way
    
    t1 = time.perf_counter()
    print(f'First pass: {format_duration(t1 - t0)}')

    # REDO: vectorized approach using GPU matmul
    print('Redoing with vectorized approach...')
    t0 = time.perf_counter()
    
    # Compute row sums of similarity matrix in batches
    row_sums = torch.zeros(N, dtype=torch.float64)
    
    for b in tqdm(range(num_batches), desc='Row sums'):
        start = b * batch_size
        end = min(start + batch_size, N)
        sim = X[start:end] @ X.T  # [batch, N]
        sim = sim.clamp(-1, 1)
        row_sums[start:end] = sim.sum(dim=1).cpu().double()
        del sim
    
    # Edge score for node i = average distance to all other nodes
    # = average(1 - similarity) = 1 - average(similarity excluding self)
    # = 1 - (row_sum[i] - 1) / (N - 1)
    avg_sim = (row_sums - 1.0) / (N - 1)  # exclude self
    avg_dist = 1.0 - avg_sim
    
    edge_scores = {}
    for i in range(N):
        edge_scores[int(indices[i])] = float(avg_dist[i])
    
    t1 = time.perf_counter()
    print(f'Edge scores computed in {format_duration(t1 - t0)}')

    # Compute within-cluster feature variance
    X_cpu = X.cpu().numpy()
    feature_mean = X_cpu.mean(axis=0)
    within_var = float(((X_cpu - feature_mean) ** 2).mean())

    # Build graph
    graph_data = {
        'nodes': nodes,
        'num_nodes': N,
        'within_cluster_feature_variance': within_var,
        'edge_scores': edge_scores,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving to {output_path}...')
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = output_path.stat().st_size / 1024**2
    print(f'File size: {file_size:.2f} MB')
    
    t_end = time.perf_counter()
    print(f'Total time: {format_duration(t_end - t_total)}')
    print('Done!')

if __name__ == '__main__':
    main()
