#!/usr/bin/env python3
"""
Topic Label Evaluation via Text Embedding Retrieval

Clean implementation that:
1. Computes all similarities using a single consistent method
2. Properly handles ranking without numerical precision bugs
3. Provides meaningful metrics for both correct and incorrect retrievals
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TopicDataset(Enum):
    """Which topic dataset to use."""
    TEN_THOUSAND = "keenanpepper/ten-thousand-things"
    FIFTY_THOUSAND = "keenanpepper/fifty-thousand-things"


class IndexStrategy(Enum):
    """Strategy for what text to embed and index for each topic."""
    TITLE_ONLY = "title_only"
    TITLE_PLUS_FIRST_LABEL = "title_plus_first_label"
    TITLE_PLUS_ALL_LABELS = "title_plus_all_labels"
    MEAN_OF_ALL = "mean_of_all"


def format_topic_document(title: str, labels: list[str], strategy: IndexStrategy) -> str:
    """Format a topic into a text document for embedding."""
    if strategy == IndexStrategy.TITLE_ONLY:
        return title
    elif strategy == IndexStrategy.TITLE_PLUS_FIRST_LABEL:
        return f"{title}: {labels[0]}"
    elif strategy == IndexStrategy.TITLE_PLUS_ALL_LABELS:
        label_bullets = "\n".join(f"* {label}" for label in labels)
        return f"{title}, who/which could also be described as:\n{label_bullets}"
    elif strategy == IndexStrategy.MEAN_OF_ALL:
        return title  # Handled specially in build_index
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


@dataclass
class TopicRetrievalConfig:
    """Configuration for topic retrieval evaluation."""
    dataset: TopicDataset = TopicDataset.FIFTY_THOUSAND
    embedding_model: str = "thenlper/gte-large"
    device: str = "cuda"
    batch_size: int = 256
    normalize_embeddings: bool = True
    index_strategy: IndexStrategy = IndexStrategy.TITLE_ONLY
    
    @property
    def dataset_name(self) -> str:
        return self.dataset.value


class TopicRetrievalIndex:
    """
    Efficient index for topic embeddings with GPU-accelerated retrieval.
    """
    
    def __init__(self, config: Optional[TopicRetrievalConfig] = None):
        self.config = config or TopicRetrievalConfig()
        self.device = torch.device(self.config.device)
        
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.device
        )
        
        self.titles: list[str] = []
        self.labels: list[list[str]] = []
        self.topic_embeddings: Optional[torch.Tensor] = None
        
    def load_dataset(self) -> None:
        """Load the topic dataset from HuggingFace."""
        print(f"Loading dataset: {self.config.dataset_name}")
        ds = load_dataset(self.config.dataset_name, split="train")
        
        self.titles = list(ds["original_title"])
        self.labels = list(ds["labels"])
        
        label_counts = [len(labels) for labels in self.labels]
        min_labels, max_labels = min(label_counts), max(label_counts)
        if min_labels == max_labels:
            print(f"Loaded {len(self.titles)} topics, each with {min_labels} labels")
        else:
            mean_labels = sum(label_counts) / len(label_counts)
            print(f"Loaded {len(self.titles)} topics with {min_labels}-{max_labels} labels each (mean: {mean_labels:.1f})")
    
    def _embed_texts(self, texts: list[str], show_progress: bool = True) -> torch.Tensor:
        """
        Embed a list of texts efficiently in batches.
        
        Returns:
            Tensor of shape [len(texts), embed_dim] on GPU
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.config.device,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return embeddings
    
    def build_index(self) -> None:
        """Build the embedding index using the configured strategy."""
        if not self.titles:
            self.load_dataset()
        
        strategy = self.config.index_strategy
        print(f"Building index with strategy: {strategy.value}")
        
        if strategy == IndexStrategy.MEAN_OF_ALL:
            print("Embedding title + all labels separately and averaging...")
            
            all_texts = []
            topic_boundaries = [0]
            
            for title, labels in zip(self.titles, self.labels):
                all_texts.append(title)
                all_texts.extend(labels)
                topic_boundaries.append(len(all_texts))
            
            print(f"  Embedding {len(all_texts)} total texts...")
            all_embs = self._embed_texts(all_texts)
            
            print("  Averaging embeddings per topic...")
            topic_embeddings = []
            for i in range(len(self.titles)):
                start, end = topic_boundaries[i], topic_boundaries[i + 1]
                topic_emb = all_embs[start:end].mean(dim=0)
                topic_embeddings.append(topic_emb)
            
            self.topic_embeddings = torch.stack(topic_embeddings, dim=0)
            
            if self.config.normalize_embeddings:
                self.topic_embeddings = F.normalize(self.topic_embeddings, p=2, dim=1)
        else:
            documents = [
                format_topic_document(title, labels, strategy)
                for title, labels in zip(self.titles, self.labels)
            ]
            
            print(f"\nExample indexed document:\n---\n{documents[0]}\n---\n")
            self.topic_embeddings = self._embed_texts(documents)
        
        print(f"Built index with shape: {self.topic_embeddings.shape}")
    
    def save_index(self, path: str | Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.topic_embeddings.cpu(), path / "embeddings.pt")
        
        metadata = {
            "titles": self.titles,
            "labels": self.labels,
            "config": {
                "dataset": self.config.dataset.value,
                "embedding_model": self.config.embedding_model,
                "normalize_embeddings": self.config.normalize_embeddings,
                "index_strategy": self.config.index_strategy.value,
            }
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        print(f"Saved index to {path}")
    
    def load_index(self, path: str | Path) -> None:
        """Load a pre-built index from disk."""
        path = Path(path)
        
        self.topic_embeddings = torch.load(path / "embeddings.pt", weights_only=True).to(self.device)
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.titles = metadata["titles"]
        self.labels = metadata["labels"]
        
        print(f"Loaded index from {path}")
        print(f"Index shape: {self.topic_embeddings.shape}")
    
    def get_default_cache_path(self) -> Path:
        """Get the default cache path based on config."""
        dataset_name = self.config.dataset.name.lower()
        strategy_name = self.config.index_strategy.value
        return Path(__file__).parent / f"index_cache_{dataset_name}_{strategy_name}"
    
    def build_or_load_index(self, cache_path: str | Path | None = None, force_rebuild: bool = False) -> None:
        """Build the index, or load from cache if available."""
        if cache_path is None:
            cache_path = self.get_default_cache_path()
        else:
            cache_path = Path(cache_path)
        
        if not force_rebuild and (cache_path / "embeddings.pt").exists():
            print(f"Loading cached index from {cache_path}")
            self.load_index(cache_path)
        else:
            print(f"Building index (will cache to {cache_path})")
            self.build_index()
            self.save_index(cache_path)
    
    def query_batch(
        self, 
        texts: list[str], 
        k: int = 10,
        show_progress: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the top-K most similar topics for a batch of query texts.
        
        Returns:
            (indices, scores): Tensors of shape [batch_size, k]
        """
        if self.topic_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embeddings = self._embed_texts(texts, show_progress=show_progress)
        similarities = torch.mm(query_embeddings, self.topic_embeddings.T)
        scores, indices = torch.topk(similarities, k=min(k, len(self.titles)), dim=1)
        
        return indices, scores


def evaluate_labels(
    index: TopicRetrievalIndex,
    labels: list[str],
    ground_truth_indices: list[int],
    k_values: list[int] = [1, 5, 10, 20, 50, 100],
    batch_size: int = 512,
) -> dict:
    """
    Evaluate a list of labels using embedding retrieval.
    
    This is the main evaluation function with proper numerical handling.
    
    Args:
        index: Pre-built TopicRetrievalIndex
        labels: List of generated labels to evaluate
        ground_truth_indices: Index of the correct topic for each label
        k_values: K values for recall computation
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with evaluation results
    """
    n_labels = len(labels)
    assert len(ground_truth_indices) == n_labels
    
    max_k = max(k_values)
    gt_indices = torch.tensor(ground_truth_indices, device=index.device)
    
    print(f"Evaluating {n_labels} labels...")
    print("Computing embeddings and similarities in batches...")
    
    # Pre-allocate tensors to avoid concatenation overhead
    top_k_scores = torch.zeros(n_labels, max_k, device=index.device, dtype=torch.float32)
    top_k_indices = torch.zeros(n_labels, max_k, device=index.device, dtype=torch.long)
    correct_scores = torch.zeros(n_labels, device=index.device, dtype=torch.float32)
    ranks = torch.zeros(n_labels, device=index.device, dtype=torch.long)
    
    # Process in batches to avoid OOM - don't keep full similarity matrix in memory!
    for i in tqdm(range(0, n_labels, batch_size), desc="Processing batches"):
        batch_start = i
        batch_end = min(i + batch_size, n_labels)
        batch_labels = labels[batch_start:batch_end]
        batch_gt_indices = gt_indices[batch_start:batch_end]
        
        # Embed this batch
        batch_embeddings = index._embed_texts(batch_labels, show_progress=False)
        
        # Compute similarities to ALL topics using ONE method (matmul)
        # This is the single source of truth for all similarity scores
        batch_similarities = torch.mm(batch_embeddings, index.topic_embeddings.T)
        # Shape: [batch_size, n_topics]
        
        # Extract what we need from this batch, then free the similarity matrix
        
        # 1. Get top-K predictions
        batch_top_k_scores, batch_top_k_indices = torch.topk(batch_similarities, k=max_k, dim=1)
        top_k_scores[batch_start:batch_end] = batch_top_k_scores
        top_k_indices[batch_start:batch_end] = batch_top_k_indices
        
        # 2. Extract correct scores from the SAME similarity matrix (no numerical issues)
        batch_size_actual = batch_similarities.shape[0]
        batch_correct_scores = batch_similarities[torch.arange(batch_size_actual, device=index.device), batch_gt_indices]
        correct_scores[batch_start:batch_end] = batch_correct_scores
        
        # 3. Compute ranks using belt-and-suspenders approach
        # Count how many OTHER topics (excluding correct topic) have higher similarity
        batch_correct_scores_expanded = batch_correct_scores.unsqueeze(1)  # [batch_size, 1]
        
        # Create mask to exclude the correct topic from each row
        batch_mask = torch.ones_like(batch_similarities, dtype=torch.bool)
        batch_mask[torch.arange(batch_size_actual, device=index.device), batch_gt_indices] = False
        
        # Count how many OTHER topics beat the correct score
        batch_better_than_correct = (batch_similarities > batch_correct_scores_expanded) & batch_mask
        batch_ranks = batch_better_than_correct.sum(dim=1) + 1
        ranks[batch_start:batch_end] = batch_ranks
        
        # Free memory from this batch
        del batch_similarities, batch_better_than_correct, batch_mask
        torch.cuda.empty_cache()
    
    # Get rank 1 predictions
    predicted_indices = top_k_indices[:, 0]
    predicted_scores = top_k_scores[:, 0]
    
    # Compute recall@K
    recalls = {}
    for k in k_values:
        # Check if ground truth is in top-K
        hits = (top_k_indices[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
        recalls[k] = hits.float().mean().item()
    
    # Build per-label results
    per_label_results = []
    reciprocal_ranks = []
    margins = []
    confidence_gaps = []
    
    for i in range(n_labels):
        gt_idx = ground_truth_indices[i]
        pred_idx = predicted_indices[i].item()
        rank = ranks[i].item()
        
        correct_score = correct_scores[i].item()
        predicted_score = predicted_scores[i].item()
        
        # Margin: how much better/worse is correct vs rank 1?
        # Positive = correct is better (correct answer is rank 1)
        # Negative = rank 1 is better (we got it wrong)
        margin = correct_score - predicted_score
        margins.append(margin)
        
        # Confidence gap: how much better is rank 1 vs rank 2?
        # This measures how distinctive the top prediction is
        if max_k >= 2:
            rank2_score = top_k_scores[i, 1].item()
            confidence_gap = predicted_score - rank2_score
            confidence_gaps.append(confidence_gap)
        else:
            confidence_gap = None
        
        # Reciprocal rank
        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        
        # Find rank within top-K (for backwards compatibility)
        rank_in_topk = None
        for r in range(min(max_k, len(top_k_indices[i]))):
            if top_k_indices[i, r].item() == gt_idx:
                rank_in_topk = r + 1
                break
        
        per_label_results.append({
            "label": labels[i],
            "correct_topic": index.titles[gt_idx],
            "predicted_topic": index.titles[pred_idx],
            "correct_score": correct_score,
            "predicted_score": predicted_score,
            "rank": rank,
            "rank_in_top_k": rank_in_topk,
            "margin": margin,
            "confidence_gap": confidence_gap,
            "reciprocal_rank": rr,
            "is_correct": (rank == 1),
        })
    
    # Compute aggregate metrics
    mrr = sum(reciprocal_ranks) / n_labels
    mean_correct_score = sum(r["correct_score"] for r in per_label_results) / n_labels
    mean_margin = sum(margins) / n_labels
    mean_confidence_gap = sum(confidence_gaps) / len(confidence_gaps) if confidence_gaps else 0.0
    
    return {
        "n_labels": n_labels,
        "recalls": recalls,
        "mrr": mrr,
        "mean_correct_score": mean_correct_score,
        "mean_margin": mean_margin,
        "mean_confidence_gap": mean_confidence_gap,
        "per_label_results": per_label_results,
    }


def print_eval_summary(results: dict, name: str = "Evaluation") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Total labels: {results['n_labels']}")
    
    print("\nRecall@K:")
    for k, recall in sorted(results["recalls"].items()):
        print(f"  R@{k:<4}: {recall:.4f} ({recall * 100:.2f}%)")
    
    print("\nAdditional metrics:")
    print(f"  MRR (Mean Reciprocal Rank):  {results['mrr']:.4f}")
    print(f"  Mean correct score:          {results['mean_correct_score']:.4f}")
    print(f"  Mean margin:                 {results['mean_margin']:+.4f}")
    print(f"  Mean confidence gap:         {results['mean_confidence_gap']:.4f}")


if __name__ == "__main__":
    # Demo: Build index and test
    config = TopicRetrievalConfig()
    index = TopicRetrievalIndex(config)
    index.build_or_load_index()
    
    print("\nTesting with a few example queries...")
    test_labels = [
        index.titles[0],  # Use actual title
        index.labels[1][0],  # Use synthetic label
        "a made up description that won't match anything well",
    ]
    test_ground_truth = [0, 1, 2]
    
    results = evaluate_labels(index, test_labels, test_ground_truth)
    print_eval_summary(results, "Demo Test")
