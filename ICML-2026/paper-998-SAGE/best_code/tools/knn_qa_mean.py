"""
knn_qa_mean.py

QA task KNN computation script (using LLM hidden state with mean pooling)
Uses input + answer to find neighbors

Supported datasets:
- TruthfulQA: uses question + generated_answer to find neighbors
- HaluEval: uses knowledge + dialogue_history + generated_answer to find neighbors

Supported embedding methods:
- LLaMA 3.1-8B: meta-llama/Llama-3.1-8B-Instruct (last hidden state + mean pooling)
- Ministral-8B: mistralai/Ministral-8B-Instruct-2410 (last hidden state + mean pooling)
- Qwen3-Embedding-8B: Qwen/Qwen3-Embedding-8B (dedicated embedding model, built-in mean pooling)

Difference from knn_qa.py:
- knn_qa.py uses SentenceTransformer to load Qwen3-Embedding
- This script directly loads transformers models, supports LLaMA and Ministral mean/last pooling

Run:
  # TruthfulQA
  python3 knn_qa_mean.py \
    --input_glob "/path/to/jsons/*.json" \
    --out_jsonl "/path/to/out/neighbors.jsonl" \
    --method llama \
    --dataset truthfulqa

  # HaluEval
  python3 knn_qa_mean.py \
    --input_glob "/path/to/halueval/*.json" \
    --out_jsonl "/path/to/out/neighbors.jsonl" \
    --method llama \
    --dataset halueval
"""

import argparse
import glob
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

import numpy as np
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
            torch_dtype=torch.bfloat16,
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
    def embed_texts(self, texts: List[str], batch_size: int = 8, pooling: str = "mean") -> np.ndarray:
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
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
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
            torch_dtype=torch.bfloat16,
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
    def embed_texts(self, texts: List[str], batch_size: int = 8, pooling: str = "mean") -> np.ndarray:
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
                # Last token
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
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
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
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.device = self.model.device
        
        self.embed_dim = self.model.config.hidden_size
        print(f"✓ Qwen3-Embedding loaded (hidden_size={self.embed_dim})")
    
    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 8, pooling: str = "mean") -> np.ndarray:
        """
        Qwen3-Embedding has built-in mean pooling, pooling parameter is ignored
        """
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
            # Use mean pooling
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


# ==================== Data Loading ====================

def build_text_truthfulqa(obj: Dict[str, Any]) -> str:
    """TruthfulQA: use question + generated_answer"""
    q = (obj.get("question", "") or "").strip()
    a = (obj.get("generated_answer", "") or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def build_text_halueval(obj: Dict[str, Any]) -> str:
    """HaluEval: use knowledge + dialogue_history + generated_answer"""
    knowledge = (obj.get("knowledge", "") or "").strip()
    dialogue = (obj.get("dialogue_history", "") or obj.get("context", "") or "").strip()
    answer = (obj.get("generated_answer", "") or obj.get("response", "") or "").strip()
    
    # Combine into a single text
    parts = []
    if knowledge:
        parts.append(f"Knowledge: {knowledge}")
    if dialogue:
        parts.append(f"Dialogue: {dialogue}")
    if answer:
        parts.append(f"Answer: {answer}")
    
    return "\n".join(parts) if parts else "Empty"


def detect_dataset_type(obj: Dict[str, Any]) -> str:
    """Auto-detect dataset type"""
    if "knowledge" in obj or "dialogue_history" in obj:
        return "halueval"
    elif "question" in obj:
        return "truthfulqa"
    else:
        return "unknown"


def load_items(path: str) -> List[Dict[str, Any]]:
    """Load a single JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list JSON.")
    return data


# ==================== KNN Computation ====================

def topk_neighbors_all(
    emb: np.ndarray,
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find neighbors within the dataset itself (excluding self)
    emb: (N, D) float32, already normalized.
    Returns:
      nn_idx: (N, k) int32
      nn_score: (N, k) float32
    """
    print(f"\nComputing KNN (k={k})...")
    N, D = emb.shape
    nn_idx = np.empty((N, k), dtype=np.int32)
    nn_score = np.empty((N, k), dtype=np.float32)

    for start in range(0, N, query_block):
        end = min(start + query_block, N)
        q = emb[start:end]                 # (B, D)
        scores = q @ emb.T                 # (B, N)

        # Exclude self: set diagonal positions to -inf for this block
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        scores[rows, cols] = -np.inf

        # Get top-k indices per row (unsorted), then sort them by score desc
        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]  # (B, k)
        top_sc = np.take_along_axis(scores, top_idx, axis=1)        # (B, k)

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


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="QA KNN with Mean Pooling (knn_qa_mean)")
    parser.add_argument("--input_glob", type=str, required=True,
                       help="Glob pattern for input JSON files")
    parser.add_argument("--out_jsonl", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--method", type=str, required=True,
                       choices=['llama', 'ministral', 'qwen3emb'],
                       help='Embedding method')
    parser.add_argument("--dataset", type=str, default="auto",
                       choices=['auto', 'truthfulqa', 'halueval'],
                       help='Dataset type: auto (auto-detect), truthfulqa, halueval')
    parser.add_argument("--pooling", type=str, default="mean",
                       choices=['last', 'mean'],
                       help='Pooling method: last (last token) or mean (mean pooling)')
    parser.add_argument("--embed_batch", type=int, default=8,
                       help="Batch size for embedding")
    parser.add_argument("--query_block", type=int, default=256,
                       help="Block size for KNN query")
    parser.add_argument("--k", type=int, default=9,
                       help="Number of neighbors")
    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("📝 QA KNN Calculator (Mean Pooling)")
    print("=" * 80)
    print(f"Input glob:   {args.input_glob}")
    print(f"Output:       {args.out_jsonl}")
    print(f"Method:       {args.method}")
    print(f"Dataset:      {args.dataset}")
    print(f"Pooling:      {args.pooling}")
    print(f"Embed batch:  {args.embed_batch}")
    print(f"Query block:  {args.query_block}")
    print(f"K neighbors:  {args.k}")
    print("=" * 80)

    # 1) Load + build texts + keep meta mapping
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    dataset_type = args.dataset

    print(f"\n📥 Loading data from {len(paths)} files...")
    for p in paths:
        items = load_items(p)
        for row_in_file, obj in enumerate(items):
            # Auto-detect or use specified dataset type
            if dataset_type == "auto":
                detected_type = detect_dataset_type(obj)
                if row_in_file == 0:
                    print(f"  Auto-detected dataset type: {detected_type}")
                    dataset_type = detected_type
            
            # Build text based on dataset type
            if dataset_type == "halueval":
                txt = build_text_halueval(obj)
            else:  # truthfulqa or unknown
                txt = build_text_truthfulqa(obj)
            
            texts.append(txt)
            meta.append({
                "global_id": len(meta),                 # 0..N-1
                "source_file": os.path.basename(p),
                "row_in_file": row_in_file,
                "index": obj.get("index", None),
            })

    N = len(texts)
    print(f"✓ Loaded {N} samples from {len(paths)} files.")
    print(f"✓ Dataset type: {dataset_type}")
    
    if dataset_type == "halueval":
        print("  ➡️  Using: knowledge + dialogue_history + generated_answer")
    else:
        print("  ➡️  Using: question + generated_answer")
    
    # Print example
    if texts:
        print(f"\nSample text (first):")
        sample_text = texts[0][:300] + "..." if len(texts[0]) > 300 else texts[0]
        print(f"   {sample_text}")

    # 2) Initialize embedder
    print(f"\n🔧 Initializing {args.method} embedder...")
    
    if args.method == "llama":
        embedder = LlamaEmbedder()
    elif args.method == "ministral":
        embedder = MinistralEmbedder()
    elif args.method == "qwen3emb":
        embedder = Qwen3EmbeddingEmbedder()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 3) Extract embeddings
    print(f"\n📊 Extracting embeddings with {args.pooling} pooling...")
    emb = embedder.embed_texts(texts, batch_size=args.embed_batch, pooling=args.pooling)
    print(f"✓ Embeddings ready: shape={emb.shape}")

    # 4) KNN (top-k)
    nn_idx, nn_score = topk_neighbors_all(
        emb,
        k=args.k,
        query_block=args.query_block,
    )

    # 5) Save JSONL: each line contains item meta + its neighbors
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    print(f"\n💾 Saving to {args.out_jsonl}...")
    
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in range(N):
            neighbors = [
                {"global_id": int(nn_idx[i, j]), "cosine": float(nn_score[i, j])}
                for j in range(args.k)
            ]
            row = {
                **meta[i],
                "neighbors": neighbors,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6) Print statistics
    print("\n" + "=" * 80)
    print("📈 Statistics")
    print("=" * 80)
    print(f"Total samples:       {N}")
    print(f"Embedding dimension: {emb.shape[1]}")
    print(f"Mean cosine score:   {nn_score.mean():.4f}")
    print(f"Min cosine score:    {nn_score.min():.4f}")
    print(f"Max cosine score:    {nn_score.max():.4f}")
    print("=" * 80)
    print(f"✅ Done! Output: {args.out_jsonl}")


if __name__ == "__main__":
    main()

