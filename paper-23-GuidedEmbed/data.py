import os
import numpy as np
import pickle
import logging
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_huggingface_data():
    """Load dataset from huggingface"""
    dataset = load_dataset("dataset_name")
    
    data_dict = {
        idx: {
            'sentence': row['text'],
            'label': row['label']
        }
        for idx, row in enumerate(dataset)
    }
    # Print dataset length
    logger.info(f"Dataset length: {len(dataset)}")

    return data_dict

def get_model_path():
    """Return current model path"""
    return "mixedbread-ai/mxbai-embed-large-v1"

def generate_embeddings(texts, batch_size=64, cache_path=None, num_gpus=None):
    logger.info(f"Generating embeddings...")
    
    # Detect available GPU count
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Using GPUs: {num_gpus}")
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(get_model_path(), trust_remote_code=True)
    base_model = AutoModel.from_pretrained(get_model_path(), trust_remote_code=True)
    
    # If multiple GPUs are available, use DataParallel
    if num_gpus > 1:
        base_model = torch.nn.DataParallel(base_model, device_ids=list(range(num_gpus)))
        # Adjust batch size based on GPU count
        effective_batch_size = batch_size * num_gpus
    else:
        effective_batch_size = batch_size
    
    logger.info(f"Effective batch size: {effective_batch_size}")
    base_model = base_model.to(device)
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), effective_batch_size)):
        batch_texts = texts[i:i + effective_batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, 
                         return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = base_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    if cache_path:
        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved to: {cache_path}")
    
    return embeddings

def load_data():
    """Load data and generate embeddings"""
    sentences = load_huggingface_data()
    # Extract text and method labels
    texts = [sentence['sentence'] for sentence in sentences.values()]
    labels = [sentence['label'] for sentence in sentences.values()]

    # Generate embeddings
    embeddings_path = './cache/embeddings.pkl'
    if not os.path.exists(embeddings_path):
        generate_embeddings(texts, cache_path=embeddings_path)

    # Create label mapping
    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    indices = list(range(len(texts)))
    label_ids = [label_mapping[labels[i]] for i in indices]
    
    # Build return data
    data_dict = {
        'texts': texts,
        'labels': label_ids,
        'indices': indices,
        'label_mapping': label_mapping,
    }
    
    return data_dict

def load_embeddings(indices=None, cache_path = './cache/embeddings.pkl'):
    # Check cache
    if os.path.exists(cache_path):
        logger.info(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            all_embeddings = pickle.load(f)
            
        # If indices provided, only return specified vectors
        if indices is not None:
            print("Loading embeddings for specified indices")
            return all_embeddings[indices]
        return all_embeddings
    else:
        raise ValueError(f"Cache file does not exist: {cache_path}")

def main():
    # Delete embeddings.pkl file in cache directory, confirm with user before deletion
    if os.path.exists('./cache/embeddings.pkl'):
        print("Cache file exists, delete? (y/n)")
        if input() == 'y':
            os.remove('./cache/embeddings.pkl')
    data = load_data()
    
    # Print some information
    logger.info(f"Loaded {len(data['texts'])} items")
    embeddings = load_embeddings(cache_path='./cache/embeddings.pkl')
    logger.info(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main() 