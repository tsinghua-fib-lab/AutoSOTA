import numpy as np
import logging
from tqdm import tqdm
import os
from data import load_data, load_embeddings
from evaluate_cluster import get_trained_embeddings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def create_triplets(texts, labels, n_triplets=50000, random_seed=42):
    """
    Create triplet dataset (anchor, positive, negative)
    
    Args:
        texts: list of texts
        labels: list of labels
        n_triplets: number of triplets to generate
        random_seed: random seed
    
    Returns:
        list of dicts: dataset containing anchor, positive, negative
    """
    np.random.seed(random_seed)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    logger.info("Generating triplets...")
    triplets = []
    triplets_history = {}
    
    # Create sample index dictionary for each label
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    with tqdm(total=n_triplets, desc="Generating triplets") as pbar:
        while len(triplets) < n_triplets:
            # Randomly select anchor's label
            anchor_label = np.random.choice(unique_labels)
            # Get labels different from anchor_label
            negative_labels = [l for l in unique_labels if l != anchor_label]
            negative_label = np.random.choice(negative_labels)
            
            # Select anchor and positive from the same label
            anchor_candidates = label_to_indices[anchor_label]
            if len(anchor_candidates) < 2:
                continue
            anchor_idx, positive_idx = np.random.choice(anchor_candidates, size=2, replace=False)
            
            # Select negative from different labels
            negative_candidates = label_to_indices[negative_label]
            if len(negative_candidates) < 1:
                continue
            negative_idx = np.random.choice(negative_candidates)
            
            triplet_check_num_1 = anchor_idx * 1e10 + positive_idx * 1e5 + negative_idx
            triplet_check_num_2 = positive_idx * 1e10 + anchor_idx * 1e5 + negative_idx
            if triplet_check_num_1 not in triplets_history and triplet_check_num_2 not in triplets_history:
                triplets_history[triplet_check_num_1] = 1
                triplets_history[triplet_check_num_2] = 1
                triplets.append({
                    'anchor_id': int(anchor_idx),
                    'positive_id': int(positive_idx),
                    'negative_id': int(negative_idx)
                })
                pbar.update(1)
    
    return triplets

def evaluate_similarity(embeddings, triplets):
    """
    Evaluate similarity accuracy of embeddings using Euclidean distance
    
    Args:
        embeddings: embedding matrix
        triplets: triplet dataset
    
    Returns:
        float: accuracy
    """
    correct = 0
    total = len(triplets)
    
    for triplet in triplets:
        anchor_emb = embeddings[triplet['anchor_id']]
        positive_emb = embeddings[triplet['positive_id']]
        negative_emb = embeddings[triplet['negative_id']]
        
        # Calculate Euclidean distance
        pos_dist = np.linalg.norm(anchor_emb - positive_emb)
        neg_dist = np.linalg.norm(anchor_emb - negative_emb)
        
        # The smaller the distance, the more similar
        if pos_dist < neg_dist:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def main():
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data = load_data()
    
    # Create triplet dataset
    triplets = create_triplets(
        texts=data['texts'],
        labels=data['labels'],
        n_triplets=50000
    )
    
    # Evaluate base embeddings
    logger.info("Evaluating base embeddings...")
    base_embeddings = load_embeddings(indices=data['indices'])
    base_accuracy = evaluate_similarity(base_embeddings, triplets)
    logger.info(f"Base model similarity accuracy: {base_accuracy:.4f}")
    
    # Evaluate transformed embeddings
    logger.info("Evaluating transformed embeddings...")
    transformed_embeddings = get_trained_embeddings(base_embeddings)
    transformed_accuracy = evaluate_similarity(transformed_embeddings, triplets)
    logger.info(f"Transformed similarity accuracy: {transformed_accuracy:.4f}")
    

if __name__ == "__main__":
    main() 