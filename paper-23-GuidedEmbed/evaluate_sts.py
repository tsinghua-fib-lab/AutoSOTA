import torch
import numpy as np
from scipy.stats import spearmanr
import logging
from data import load_data, load_embeddings
from evaluate_cluster import get_trained_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def create_sts_pairs(texts, labels, n_pairs=50000, random_seed=42):
    """
    Create STS text pair dataset
    
    Args:
        texts: List of texts
        labels: List of labels
        n_pairs: Number of text pairs to generate
        random_seed: Random seed
    
    Returns:
        list of dicts: Dataset containing text1, text2 and score
    """
    np.random.seed(random_seed)
    n_texts = len(texts)
    
    # Create all possible text pair combinations
    logger.info("Generating text pairs...")
    labels = np.array(labels)
    
    # Calculate required number of similar and dissimilar pairs
    n_similar = n_pairs // 2
    n_dissimilar = n_pairs - n_similar
    
    # Generate similar pairs (same label)
    logger.info("Generating similar text pairs...")
    similar_pairs = []
    while len(similar_pairs) < n_similar:
        i, j = np.random.randint(0, n_texts, 2)
        if labels[i] == labels[j] and (i, j) not in similar_pairs:
            similar_pairs.append((i, j))
    
    # Generate dissimilar pairs (different labels)
    logger.info("Generating dissimilar text pairs...")
    dissimilar_pairs = []
    while len(dissimilar_pairs) < n_dissimilar:
        i, j = np.random.randint(0, n_texts, 2)
        if labels[i] != labels[j] and (i, j) not in dissimilar_pairs:
            dissimilar_pairs.append((i, j))
    
    # Merge all pairs and create dataset
    all_pairs = similar_pairs + dissimilar_pairs
    
    dataset = []
    for i, j in tqdm(all_pairs, desc="Creating dataset"):
        score = 1.0 if labels[i] == labels[j] else 0.0
        dataset.append({
            'text1_id': i,
            'text2_id': j,
            'score': score
        })
    
    return dataset

def compute_cosine_scores(embeddings1, embeddings2):
    """Calculate cosine similarity between two sets of embedding vectors"""
    return np.array([
        cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ])

def evaluate_sts(embeddings1, embeddings2, ground_truth_scores):
    """Evaluate STS performance"""
    # Calculate cosine similarity
    cosine_scores = compute_cosine_scores(embeddings1, embeddings2)
    
    # Calculate Spearman correlation coefficient
    correlation, _ = spearmanr(cosine_scores, ground_truth_scores)

    return correlation

def main():
    # Load original data
    logger.info("Loading original data...")
    data = load_data()
    
    # Create STS dataset
    logger.info("Creating STS dataset...")
    sts_dataset = create_sts_pairs(
        texts=data['texts'],
        labels=data['labels'],
        n_pairs=50000
    )
    
    # Prepare text pairs and ground truth scores
    text1_id_list = [item['text1_id'] for item in sts_dataset]
    text2_id_list = [item['text2_id'] for item in sts_dataset]
    ground_truth_scores = np.array([item['score'] for item in sts_dataset])
    
    # Get base embeddings
    logger.info("Generating base embeddings...")
    base_embeddings = load_embeddings(indices=data['indices'])
    print("base_embeddings.shape: ", base_embeddings.shape)
    base_embeddings1 = base_embeddings[text1_id_list]
    base_embeddings2 = base_embeddings[text2_id_list]
    
    # Evaluate base embeddings
    base_correlation = evaluate_sts(
        base_embeddings1, 
        base_embeddings2, 
        ground_truth_scores
    )
    logger.info(f"Base model Spearman correlation: {base_correlation:.4f}")
    
    # Get transformed embeddings
    logger.info("Generating transformed embeddings...")
    transformed_embeddings = get_trained_embeddings(base_embeddings)
    transformed_embeddings1 = transformed_embeddings[text1_id_list]
    transformed_embeddings2 = transformed_embeddings[text2_id_list]
    
    # Evaluate transformed embeddings
    transformed_correlation = evaluate_sts(
        transformed_embeddings1,
        transformed_embeddings2,
        ground_truth_scores
    )
    logger.info(f"Transformed model Spearman correlation: {transformed_correlation:.4f}")
    

if __name__ == "__main__":
    main() 