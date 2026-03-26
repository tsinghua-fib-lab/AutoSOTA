import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import os
import logging
from data import load_data, load_embeddings
from evaluate import load_trained_model, get_transformed_embeddings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_trained_embeddings(embeddings, model_path = 'cache/model.pth'):
    # Load trained model
    logger.info("Loading trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path)
    model = load_trained_model(
        model_path,
        input_dim=embeddings.shape[1],
        embedding_dim=checkpoint.get('embedding_dim', 64)
    )
    
    # Get transformed embeddings
    logger.info("Transforming embeddings...")
    transformed_embeddings = get_transformed_embeddings(model, embeddings)
    return transformed_embeddings

def evaluate_clustering(embeddings, labels, n_clusters=None):
    """Evaluate clustering performance of embeddings"""
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    nmi_score = normalized_mutual_info_score(labels, cluster_labels)
    return nmi_score


def main():
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    
    # Load training and test data
    logger.info("Loading data...")
    test_data = load_data()
    
    # Get universal embeddings for the dataset
    logger.info("Evaluating universal embeddings...")
    base_embeddings = load_embeddings(indices=test_data['indices'])
    base_nmi = evaluate_clustering(base_embeddings, test_data['labels'])
    logger.info(f"NMI score for base embeddings: {base_nmi:.4f}")

    # Get trained embeddings for the dataset
    logger.info("Generating trained embeddings...")
    transformed_embeddings = get_trained_embeddings(base_embeddings)
    transformed_nmi = evaluate_clustering(transformed_embeddings, test_data['labels'])
    logger.info(f"NMI score for trained embeddings: {transformed_nmi:.4f}")

if __name__ == "__main__":
    main() 