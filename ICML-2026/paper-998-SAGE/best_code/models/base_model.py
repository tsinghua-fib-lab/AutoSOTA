"""
Base Model Class
Defines unified model interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np


class BaseModel(ABC):
    """Base class for all models"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config['model_name']
        self.model_type = config['model_type']
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load model"""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Make prediction on input

        Args:
            input_data: Input data

        Returns:
            Prediction result dictionary containing:
            - 'prediction': Predicted label/output
            - 'confidence': Confidence score
            - 'top5_predictions': Top-5 predictions
            - 'top5_confidences': Top-5 confidence scores
            - 'raw_output': Raw output (optional)
        """
        pass

    def batch_predict(self, input_list: List[Any], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Batch prediction

        Args:
            input_list: List of input data
            batch_size: Batch size

        Returns:
            List of prediction results
        """
        results = []
        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i+batch_size]
            for item in batch:
                results.append(self.predict(item))
        return results
    
    def get_top_k_predictions(self, logits: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Get Top-K predictions

        Args:
            logits: Model output logits
            k: K value for Top-K

        Returns:
            (top_k_indices, top_k_probs): Top-K indices and probabilities
        """
        # Convert to probabilities
        if len(logits.shape) == 1:
            probs = self._softmax(logits)
        else:
            probs = logits
            
        # Get Top-K
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_probs = probs[top_k_indices]
        
        return top_k_indices.tolist(), top_k_probs.tolist()
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
