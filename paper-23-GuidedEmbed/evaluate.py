import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from model import GSTransformModel
from data import load_data, load_embeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model(model_path, input_dim, embedding_dim):
    """Load trained model"""
    model = GSTransformModel(input_dim, embedding_dim).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_transformed_embeddings(model, base_embeddings):
    """Transform base embeddings using the model"""
    transformed_embeddings = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(base_embeddings), batch_size):
            batch = torch.tensor(
                base_embeddings[i:i + batch_size], 
                dtype=torch.float32
            ).to(device)
            z, _ = model(batch)
            transformed_embeddings.append(z.cpu().numpy())
    
    return np.vstack(transformed_embeddings)

def evaluate_clustering(embeddings, labels, n_clusters=None):
    """Evaluate clustering performance of embeddings"""
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # Use KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate NMI score
    nmi_score = normalized_mutual_info_score(labels, cluster_labels)
    return nmi_score

def plot_nmi_comparison(scores, save_path):
    """Plot NMI score comparison"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Base Embeddings', 'Transformed Embeddings'], 
                  [scores['base'], scores['transformed']], 
                  color=['blue', 'orange'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('NMI Score Comparison')
    plt.ylabel('NMI Score')
    plt.ylim(0, 1)  # NMI score range is 0-1
    plt.savefig(save_path)
    plt.close()

def visualize_embeddings(embeddings, labels, label_names, title, save_path, use_pca=True):
    """
    Visualize embeddings with dimensionality reduction
    Args:
        embeddings: embedding array
        labels: label array
        label_names: list of label names
        title: chart title
        save_path: save path
        use_pca: whether to use PCA (otherwise t-SNE)
    """
    print(f"\nData shape before dimension reduction: {embeddings.shape}")
    
    # Dimensionality reduction using PCA
    if use_pca:
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    else:
        # t-SNE reduction
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced_embeddings = reducer.fit_transform(embeddings)
    
    print(f"Data shape after dimension reduction: {reduced_embeddings.shape}")

    # Visualization
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                         c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Principal Component 1' if use_pca else 't-SNE Dimension 1')
    plt.ylabel('Principal Component 2' if use_pca else 't-SNE Dimension 2')

    # Add legend
    unique_labels = np.unique(labels)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=plt.cm.tab20(i/20), 
                                label=f'{label_names[i]}', markersize=10)
                      for i in unique_labels]
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5), fontsize='small')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Visualization results saved to {save_path}")

def main():
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    
    # Load IMDB data
    logger.info("Loading IMDB data...")
    data_dict = load_data()
    test_labels = data_dict['labels']
    label_mapping = data_dict['label_mapping']
    label_names = list(label_mapping.keys())  # Get list of label names
    
    # Get embeddings for all texts
    logger.info("Getting base embeddings...")
    test_embeddings = load_embeddings(indices=data_dict['indices'])
    
    # Calculate NMI score and visualize base embeddings
    logger.info("Evaluating base embeddings...")
    base_nmi = evaluate_clustering(test_embeddings, test_labels)
    logger.info(f"Based embeddings NMI score: {base_nmi:.4f}")
    
    # Visualize base embeddings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    visualize_embeddings(
        embeddings=test_embeddings,
        labels=test_labels,
        label_names=label_names,
        title='Base Embeddings Visualization',
        save_path=f'./results/base_visualization_{timestamp}.png',
        use_pca=False
    )
    
    # Load trained model
    logger.info("Loading trained model...")
    model_path = 'cache/model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path)
    model = load_trained_model(
        model_path,
        input_dim=test_embeddings.shape[1],
        embedding_dim=checkpoint.get('embedding_dim', 64)
    )
    
    # Get transformed embeddings
    logger.info("Transforming embeddings...")
    transformed_embeddings = get_transformed_embeddings(model, test_embeddings)
    
    # Calculate NMI score and visualize transformed embeddings
    logger.info("Evaluating transformed embeddings...")
    transformed_nmi = evaluate_clustering(transformed_embeddings, test_labels)
    logger.info(f"Transformed embeddings NMI score: {transformed_nmi:.4f}")
    
    # Visualize transformed embeddings
    visualize_embeddings(
        embeddings=transformed_embeddings,
        labels=test_labels,
        label_names=label_names,
        title='Transformed Embeddings Visualization',
        save_path=f'./results/transformed_visualization_{timestamp}.png',
        use_pca=False
    )
    
    # Save NMI comparison results
    scores = {
        'base': base_nmi,
        'transformed': transformed_nmi
    }
    
    plot_nmi_comparison(scores, f'./results/nmi_comparison_{timestamp}.png')
    logger.info(f"Result charts saved to: ./results/")
    
    # Output detailed evaluation results
    logger.info("\n=== Detailed Evaluation Results ===")
    logger.info(f"Dataset size: {len(test_embeddings)}")
    logger.info(f"Number of categories: {len(label_mapping)}")
    logger.info(f"Base embedding dimension: {test_embeddings.shape[1]}")
    logger.info(f"Transformed embedding dimension: {transformed_embeddings.shape[1]}")
    logger.info(f"NMI improvement: {transformed_nmi - base_nmi:.4f}")

if __name__ == "__main__":
    main() 