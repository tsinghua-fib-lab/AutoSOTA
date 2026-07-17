import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse

def read_trajectory_json(file_path):
    """Read a trajectory JSON file and concatenate all conversation content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract conversation history
        history = data.get('history', [])
        question = data.get('question', '')
        
        # Concatenate all conversation content
        text_parts = [f"Question: {question}"] if question else []
        
        for entry in history:
            # Handle both handcrafted and non-handcrafted formats
            agent_name = entry.get('role', entry.get('name', 'Unknown'))
            content = entry.get('content', '')
            text_parts.append(f"{agent_name}: {content}")
        
        return '\n'.join(text_parts)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_trajectory_similarities(dataset_path, model_name='BAAI/bge-m3'):
    """
    Compute pairwise similarities between all trajectories in a dataset.
    
    Args:
        dataset_path: Path to the dataset's individual_trajectories directory
        model_name: Name of the transformer model to use
    
    Returns:
        dict: Mapping of trajectory indices to lists of similar trajectories (sorted by similarity)
    """
    print(f"\n=== Computing similarities for {dataset_path} ===")
    
    # Initialize transformer model
    print(f"Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Get all JSON files (excluding file_mapping.json)
    json_files = []
    trajectory_texts = []
    file_indices = []
    
    for f in sorted(os.listdir(dataset_path)):
        if f.endswith('.json') and f != 'file_mapping.json':
            file_path = os.path.join(dataset_path, f)
            try:
                # Extract file number
                file_num = int(f.replace('.json', ''))
                text = read_trajectory_json(file_path)
                if text:  # Only add if we successfully read the content
                    json_files.append(f)
                    trajectory_texts.append(text)
                    file_indices.append(file_num)
            except ValueError:
                print(f"Skipping file with non-numeric name: {f}")
                continue
    
    print(f"Found {len(trajectory_texts)} trajectory files")
    
    if not trajectory_texts:
        print("No valid trajectories found!")
        return {}
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = []
    
    # Process in batches for efficiency
    batch_size = 8
    for i in tqdm(range(0, len(trajectory_texts), batch_size), desc="Encoding trajectories"):
        batch_texts = trajectory_texts[i:i+batch_size]
        
        # Tokenize the batch
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=8192, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        # Move to CPU and convert to numpy
        embeddings.extend(batch_embeddings.cpu().numpy())
    
    embeddings = np.array(embeddings)
    
    # Compute pairwise similarities
    print("Computing pairwise similarities...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create similarity mappings
    similarity_mappings = {}
    
    for i, file_idx in enumerate(file_indices):
        # Get similarities for this trajectory
        similarities = similarity_matrix[i]
        
        # Create pairs of (index, similarity) for all other trajectories
        similarity_pairs = []
        for j, other_idx in enumerate(file_indices):
            if i != j:  # Exclude self-similarity
                similarity_pairs.append((other_idx, similarities[j]))
        
        # Sort by similarity (descending)
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the indices in order of similarity
        similar_indices = [idx for idx, _ in similarity_pairs]
        
        similarity_mappings[file_idx] = similar_indices
    
    print(f"Computed similarities for {len(similarity_mappings)} trajectories")
    
    return similarity_mappings

def main():
    parser = argparse.ArgumentParser(description="Generate trajectory similarity mappings for multiple datasets")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/correct_error",
        help="Base directory containing all datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["arc", "hotpot", "musique", "wikimqa", "math500", "mmlu_pro", "gaia"],
        help="List of datasets to process (default: 7 CORRECT-Error datasets)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Transformer model to use for embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/similarities",
        help="Output directory for similarity mappings"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset in args.datasets:
        dataset_path = os.path.join(args.results_dir, dataset, "individual_trajectories")
        
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path not found: {dataset_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"Path: {dataset_path}")
        
        # Compute similarities
        similarities = compute_trajectory_similarities(dataset_path, args.model)
        
        if similarities:
            # Save to JSON file
            output_file = os.path.join(args.output_dir, f"{dataset}_trajectory_similarities.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(similarities, f, indent=2)
            
            print(f"Saved similarity mappings to: {output_file}")
            
            # Print some statistics
            print(f"\nStatistics for {dataset}:")
            print(f"  Total trajectories: {len(similarities)}")
            print(f"  First 5 trajectory mappings:")
            for idx, (key, similar_list) in enumerate(sorted(similarities.items())[:5]):
                if similar_list:
                    top_3 = similar_list[:3]
                    print(f"    Trajectory {key}: Most similar = {top_3}")
        else:
            print(f"No similarities computed for {dataset}")
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"Similarity mappings saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()