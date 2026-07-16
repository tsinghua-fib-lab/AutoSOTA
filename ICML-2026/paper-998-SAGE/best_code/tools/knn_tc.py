"""
knn_tc.py

Unified text KNN computation script (for Text Classification tasks)

Supported embedding methods:
- LLaMA 3.1-8B: meta-llama/Llama-3.1-8B-Instruct (last hidden state)
- Ministral-8B: mistralai/Ministral-8B-Instruct-2410 (last hidden state)
- Qwen3-Embedding-8B: Qwen/Qwen3-Embedding-8B (dedicated embedding model)

Supported datasets:
- AG_News: fancyzhx/ag_news (first 10000 from train split, text field only)
- MMLU: cais/mmlu (all data, subject+question+choices combination)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import torch


# ==================== Embedding Methods ====================

class LlamaEmbedder:
    """LLaMA 3.1-8B Embedder (using last hidden state)"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading LLaMA: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.device = self.model.device
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Force right-side padding (to ensure correct last token position)
        self.tokenizer.padding_side = 'right'

        self.embed_dim = self.model.config.hidden_size
        print(f"✓ LLaMA loaded (hidden_size={self.embed_dim})")

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 8, pooling: str = "last") -> np.ndarray:
        """
        pooling: "last" (last token) or "mean" (mean pooling)
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="LLaMA embedding"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']
            
            if pooling == "last":
                # Last non-padding token
                batch_embeddings = []
                for j in range(len(batch_texts)):
                    seq_len = attention_mask[j].sum().item()
                    embedding = last_hidden[j, seq_len - 1, :]
                    batch_embeddings.append(embedding)
                embeddings = torch.stack(batch_embeddings)
            elif pooling == "mean":
                # mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.concatenate(all_embeddings, axis=0)


class MinistralEmbedder:
    """Ministral-8B Embedder (using last hidden state)"""
    
    def __init__(self, model_name: str = "mistralai/Ministral-8B-Instruct-2410"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading Ministral: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.device = self.model.device
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Force right-side padding (Mistral defaults to left padding, which causes incorrect last token position)
        self.tokenizer.padding_side = 'right'
        
        self.embed_dim = self.model.config.hidden_size
        print(f"✓ Ministral loaded (hidden_size={self.embed_dim})")
    
    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 8, pooling: str = "last") -> np.ndarray:
        """
        pooling: "last" or "mean"
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Ministral embedding"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']
            
            if pooling == "last":
                # Original method: last token
                batch_embeddings = []
                for j in range(len(batch_texts)):
                    seq_len = attention_mask[j].sum().item()
                    embedding = last_hidden[j, seq_len - 1, :]
                    batch_embeddings.append(embedding)
                embeddings = torch.stack(batch_embeddings)
            
            elif pooling == "mean":
                # New method: mean pooling (consistent with Qwen3-Embedding)
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.concatenate(all_embeddings, axis=0)


class Qwen3EmbeddingEmbedder:
    """Qwen3-Embedding-8B Embedder (dedicated embedding model)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading Qwen3-Embedding: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.device = self.model.device
        
        self.embed_dim = self.model.config.hidden_size
        print(f"✓ Qwen3-Embedding loaded (hidden_size={self.embed_dim})")
    
    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract embeddings from a list of texts"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Qwen3-Embedding"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Qwen3-Embedding outputs last_hidden_state
            # Use mean pooling or last token
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.concatenate(all_embeddings, axis=0)


# ==================== Dataset Loading ====================

def load_ag_news_texts(max_samples: int = 10000) -> Tuple[List[str], str]:
    """Load AG_News dataset text field (first N from train split)"""
    from datasets import load_dataset
    
    print(f"\nLoading AG_News (train, max={max_samples})...")
    dataset = load_dataset("fancyzhx/ag_news", split="train", trust_remote_code=True)
    
    total = len(dataset)
    n_samples = min(max_samples, total)
    
    print(f"Extracting {n_samples} texts (out of {total} total)...")
    texts = []
    for i in tqdm(range(n_samples), desc="Loading texts"):
        text = dataset[i]['text']
        texts.append(text)
    
    print(f"✓ Loaded {len(texts)} texts")
    return texts, "ag_news"


def load_mmlu_texts(max_samples: int = None) -> Tuple[List[str], str]:
    """Load MMLU dataset (test split only, combining subject+question+choices)"""
    from datasets import load_dataset
    
    print(f"\nLoading MMLU (test split only)...")
    dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    
    total = len(dataset)
    n_samples = min(max_samples, total) if max_samples else total
    
    print(f"Extracting {n_samples} texts (out of {total} total)...")
    all_texts = []
    
    for i in tqdm(range(n_samples), desc="Loading texts"):
        item = dataset[i]
        # Combine subject + question + choices
        subject = item.get('subject', '')
        question = item.get('question', '')
        choices = item.get('choices', [])
        
        # Format choices
        if isinstance(choices, list):
            choices_str = " | ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
        else:
            choices_str = str(choices)
        
        # Combine into full text
        combined_text = f"[{subject}] {question}\nChoices: {choices_str}"
        all_texts.append(combined_text)
    
    print(f"✓ Loaded {len(all_texts)} texts")
    return all_texts, "mmlu"


def load_dataset_texts(dataset_name: str, max_samples: int = None) -> Tuple[List[str], str]:
    """Load texts by dataset name"""
    if dataset_name.lower() == "ag_news":
        return load_ag_news_texts(max_samples=max_samples or 10000)
    elif dataset_name.lower() == "mmlu":
        return load_mmlu_texts(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ==================== KNN Computation ====================

def topk_neighbors_all(
    emb: np.ndarray,
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find neighbors within the dataset itself (excluding self)"""
    print(f"\nComputing KNN (k={k})...")
    N, D = emb.shape
    nn_idx = np.empty((N, k), dtype=np.int32)
    nn_score = np.empty((N, k), dtype=np.float32)
    
    for start in range(0, N, query_block):
        end = min(start + query_block, N)
        q = emb[start:end]
        scores = q @ emb.T
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        scores[rows, cols] = -np.inf
        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        top_sc = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_sc, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sc = np.take_along_axis(top_sc, order, axis=1)
        # Clamp to [-1, 1] (prevent floating point precision from slightly exceeding range)
        top_sc = np.clip(top_sc, -1.0, 1.0)
        nn_idx[start:end] = top_idx.astype(np.int32)
        nn_score[start:end] = top_sc.astype(np.float32)
        if (start // query_block) % 10 == 0:
            print(f"  Processed queries [{start}:{end}) / {N}")
    
    return nn_idx, nn_score


# ==================== Output Formatting ====================

def save_neighbors_jsonl(
    knn_indices: np.ndarray,
    knn_scores: np.ndarray,
    dataset_name: str,
    method_name: str,
    output_path: str
):
    """Save as compact JSONL format (keeping only id and neighbor info)"""
    print(f"\nSaving to {output_path}...")
    
    N, k = knn_indices.shape
    
    with open(output_path, 'w') as f:
        for i in range(N):
            item = {
                'index': i,
                'neighbors': [
                    {
                        'index': int(knn_indices[i, j]),
                        'cosine': float(knn_scores[i, j])
                    }
                    for j in range(k)
                ]
            }
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Saved {N} samples")


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Text Classification KNN (knn_tc)")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ag_news', 'mmlu'],
                       help='Dataset name')
    parser.add_argument('--method', type=str, required=True,
                       choices=['llama', 'ministral', 'qwen3emb'],
                       help='Embedding method')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples (default: 10000 for AG_News, all for MMLU)')
    parser.add_argument('--k', type=int, default=9,
                       help='Number of neighbors (default: 9)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: outputs/text_classification)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto if not specified)')
    parser.add_argument('--pooling', type=str, default='last',
                       choices=['last', 'mean'],
                       help='Pooling method: last (last token) or mean (mean pooling)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "outputs" / "text_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default batch size (auto-adjust based on dataset and model)
    batch_size = args.batch_size
    if batch_size is None:
        if args.dataset == 'mmlu':
            # MMLU texts are longer (question + choices), need smaller batch size
            batch_size = 2
        else:
            batch_size = 8
    
    # Print configuration
    print("=" * 80)
    print("📝 Text Classification KNN Calculator")
    print("=" * 80)
    print(f"Dataset:     {args.dataset}")
    print(f"Method:      {args.method}")
    print(f"Pooling:     {args.pooling}")
    print(f"Max samples: {args.max_samples or 'default'}")
    print(f"Neighbors:   {args.k}")
    print(f"Batch size:  {batch_size}")
    print(f"Output dir:  {output_dir}")
    print("=" * 80)
    
    # Load dataset
    texts, dataset_name = load_dataset_texts(args.dataset, args.max_samples)
    
    # Initialize embedder
    print(f"\nInitializing {args.method} embedder...")
    
    if args.method == "llama":
        embedder = LlamaEmbedder()
        method_name = "llama"
    elif args.method == "ministral":
        embedder = MinistralEmbedder()
        method_name = "ministral"
    elif args.method == "qwen3emb":
        embedder = Qwen3EmbeddingEmbedder()
        method_name = "qwen3emb"
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Extract embeddings
    print(f"\n📊 Extracting embeddings...")
    if args.method == "qwen3emb":
        # Qwen3-Embedding has built-in mean pooling, no pooling parameter needed
        embeddings = embedder.embed_texts(texts, batch_size=batch_size)
    else:
        embeddings = embedder.embed_texts(texts, batch_size=batch_size, pooling=args.pooling)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Compute KNN
    knn_indices, knn_scores = topk_neighbors_all(embeddings, k=args.k)
    
    # Save results (filename includes pooling method unless it's the default 'last')
    if args.method == "qwen3emb" or args.pooling == "last":
        output_file = output_dir / f"{dataset_name}_{method_name}_neighbors.jsonl"
    else:
        output_file = output_dir / f"{dataset_name}_{method_name}_{args.pooling}_neighbors.jsonl"
    save_neighbors_jsonl(
        knn_indices,
        knn_scores,
        dataset_name,
        method_name,
        str(output_file)
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("📈 Statistics")
    print("=" * 80)
    print(f"Total samples:       {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Mean cosine score:   {knn_scores.mean():.4f}")
    print(f"Min cosine score:    {knn_scores.min():.4f}")
    print(f"Max cosine score:    {knn_scores.max():.4f}")
    print("=" * 80)
    print(f"✅ Done! Output: {output_file}")


if __name__ == "__main__":
    main()
