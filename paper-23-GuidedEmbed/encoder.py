import os
import torch
import logging
import numpy as np
from typing import List, Optional
import pickle

from data import generate_embeddings
from label_construction import TextClassificationWorkflow
from model import GSTransformModel
from evaluate import get_transformed_embeddings

from train import train_with_validation, get_label_mapping
from torch.utils.data import DataLoader, random_split   
from model import ExampleDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GSTransform_Encoder:
    def __init__(
        self,
        cache_dir: str = "./cache",
        embedding_dim: int = 64,
        random_seed: int = 42
    ):
        """Initialize GSTransform encoder
        
        Args:
            cache_dir: Cache directory path
            embedding_dim: Dimension of transformed embeddings
            random_seed: Random seed
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        
        # Initialize workflow
        self.workflow = TextClassificationWorkflow(
            cache_dir=cache_dir,
            random_seed=random_seed
        )
        
        self.base_embeddings = None
        self.texts = None
        self.model = None
        
    def pre_encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        cache_name: Optional[str] = None
    ) -> np.ndarray:
        """Encode texts using base embedding model
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            cache_name: Cache file name (optional)
        
        Returns:
            np.ndarray: Base embeddings for the texts
        """
    
        # Store texts
        self.texts = texts
        
        cache_path = None
        if cache_name:
            cache_path = os.path.join(self.cache_dir, f"{cache_name}_base_embeddings.pkl")
            # Check if cache exists
            if os.path.exists(cache_path):
                logger.info(f"Loading embeddings from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    self.base_embeddings = pickle.load(f)
                return self.base_embeddings
            
        self.base_embeddings = generate_embeddings(
            texts=texts,
            batch_size=batch_size,
            cache_path=cache_path
        )
        return self.base_embeddings
    
    def transform(
        self,
        instruction: str,
        cache_name: Optional[str] = None
    ) -> np.ndarray:
        """Transform embeddings based on user instruction
        
        Args:
            instruction: User instruction
            cache_name: Cache file name (optional)
            
        Returns:
            np.ndarray: Transformed embeddings
        """
        if self.base_embeddings is None or self.texts is None:
            raise ValueError("Please run pre_encode first before transform")
        
        # Randomly select 3000 texts from self.texts with random seed 42
        rng = np.random.RandomState(42)
        selected_indices = rng.permutation(len(self.texts))[:3000]
        selected_texts = [self.texts[i] for i in selected_indices]
        
        # Run label construction workflow
        result = self.workflow.run(
            instruction=instruction,
            texts=selected_texts,
            indices=selected_indices,
            n_clusters=50,
            cache_key=cache_name
        )
        
        # Prepare model
        input_dim = self.base_embeddings.shape[1]
        self.model = GSTransformModel(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim
        ).to(device)
        
        # Model save path
        model_path = os.path.join(
            self.cache_dir,
            f"model_{cache_name}.pth" if cache_name else "model.pth"
        )

        # Prepare training data
        train_indices = [int(i) for i in result['indices']]
        train_embeddings = self.base_embeddings[train_indices]
        
        label_texts = [result['classifications'][str(i)] for i in train_indices]
        label_mapping = get_label_mapping(label_texts)
        train_labels = [label_mapping[label] for label in label_texts]

        # Create dataset
        dataset = ExampleDataset(train_embeddings, train_labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64)
        
        # Train model
        train_losses, val_losses = train_with_validation(
            self.model,
            train_dataloader,
            val_dataloader,
            num_epochs=500,
            learning_rate=0.0001,
            lambda1=1.0,
            lambda2=1.0,
            margin=10.0,
            patience=10
        )
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': input_dim,
            'embedding_dim': self.embedding_dim,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'label_mapping': result['categories']
        }, model_path)
        
        # Get transformed embeddings
        transformed_embeddings = get_transformed_embeddings(
            self.model,
            self.base_embeddings
        )
        
        return transformed_embeddings
