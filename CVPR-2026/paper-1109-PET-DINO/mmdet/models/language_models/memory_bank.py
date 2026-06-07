import torch
import pickle
import random
import json
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class MemoryBank:
    def __init__(self, max_length, dim, label_map_file, **kwargs):
        """Initialize the Memory Bank.

        Args:
            max_length (int): Maximum number of features stored per class.
            dim (int): Dimension of the feature vectors.
            label_map_file (str): Path to the JSON file containing label IDs.
        """
        self.max_length = max_length
        self.dim = dim

        with open(label_map_file, 'r') as f:
            content = json.load(f)
        
        self.bank = {}
        self.ema_features = {}
        for class_id, class_name in content.items():
            class_id = int(class_id)
            self.bank[class_id] = deque(maxlen=max_length)
            self.ema_features[class_id] = None
        
        self.ema_decay = kwargs.get('ema_decay', 0.9)
        self.use_ema_feature = kwargs.get('use_ema_feature', False)
        self.use_mean_feature = kwargs.get('use_mean_feature', False)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, features, labels, ignore_labels=[]):
        """Update the Memory Bank and EMA features.

        Args:
            features (torch.Tensor): Feature vectors of shape [batch_size, dim].
            labels (torch.Tensor): Class labels of shape [batch_size].
            ignore_labels (list): Labels to ignore during update.
        """
        features = features.detach()
        labels = labels.detach()
        assert features.shape[0] == labels.shape[0]
        
        for feature, label in zip(features, labels):
            class_id = int(label)
            if class_id not in ignore_labels:
                if class_id in self.bank:
                    self.bank[class_id].append(feature)
                    
                    if self.ema_features[class_id] is None:
                        self.ema_features[class_id] = feature.clone()
                    else:
                        self.ema_features[class_id] = (1.0 - self.ema_decay) * self.ema_features[class_id] + \
                                                      self.ema_decay * feature
                else:
                    raise ValueError(f"invalid Class ID {class_id} attempt to update bank.")        
    
    def get_class_features(self, class_id):
        """Get all features of a given class.

        Args:
            class_id (int): Class ID.

        Returns:
            torch.Tensor: All features of the class, shape [num_features, dim].
        """
        if class_id not in self.bank:
            raise ValueError(f"Class ID {class_id} is not in memory bank.")
            
        if len(self.bank[class_id]) == 0:
            return torch.empty(0, self.dim)
        
        return torch.stack(list(self.bank[class_id]), dim=0)

    def get_class_ema_feature(self, class_id):
        """Get the EMA feature of a given class.

        Args:
            class_id (int): Class ID.

        Returns:
            torch.Tensor: The EMA feature of shape [dim].
        """
        if class_id not in self.ema_features:
            raise ValueError(f"Class ID {class_id} is not in memory bank.")
            
        if self.ema_features[class_id] is None:
            raise ValueError(f"No EMA feature available for class {class_id}")
            
        return self.ema_features[class_id]

    def get_class_any_feature(self, class_id, return_index=False):
        """Get a random feature from a given class.

        Args:
            class_id (int): Class ID.
            return_index (bool): Whether to also return the index.

        Returns:
            torch.Tensor: A feature vector of shape [dim].
            int (optional): Index in the queue (when return_index=True).
        """
        if class_id not in self.bank:
            raise ValueError(f"Class ID {class_id} is not in memory bank.")
            
        class_features = self.bank[class_id]
        
        if len(class_features) == 0:
            raise ValueError(f"No features available for class {class_id}")
            
        # Randomly select a feature
        idx = random.randrange(len(class_features))
        feature = class_features[idx]
        
        if return_index:
            # Return feature and its index in the queue
            return feature, idx
        else:
            # Return feature only
            return feature

    def get_class_mean_feature(self, class_id):
        """Get the mean feature of a given class.

        Returns:
            torch.Tensor: Mean feature of shape [dim].
        """
        if class_id not in self.bank:
            raise ValueError(f"Class ID {class_id} is not in memory bank.")
            
        class_features = self.get_class_features(class_id)

        if len(class_features) == 0:
            raise ValueError(f"No features available for class {class_id}")
        
        class_mean_feature = class_features.mean(dim=0)

        return class_mean_feature
    
    def get_FeatureNum_valid_classes(self, min_FeatureNum):
        """Get classes that have at least `min_FeatureNum` features.

        Returns:
            list: Classes meeting the minimum feature count requirement.
        """
        not_empty_classes_list = []

        for key, queue in self.bank.items():
            if len(queue) >= min_FeatureNum:
                not_empty_classes_list.append(key)
        
        return not_empty_classes_list

    def save(self, file_path):
        """Save the Memory Bank to a file (including EMA features).

        Args:
            file_path (str): File save path.
        """
        bank_data = {
            'dim': self.dim,
            'max_length': self.max_length,
            'ema_decay': self.ema_decay,
            'bank': {},
            'ema_features': {}
        }
        
        for class_id, queue in self.bank.items():
            if len(queue) > 0:
                features = torch.stack(list(queue), dim=0)
                bank_data['bank'][class_id] = features.numpy()
            else:
                bank_data['bank'][class_id] = np.array([])

            if self.ema_features[class_id] is not None:
                bank_data['ema_features'][class_id] = self.ema_features[class_id].numpy()
            else:
                bank_data['ema_features'][class_id] = None

        with open(file_path, 'wb') as f:
            pickle.dump(bank_data, f)
        logger.info(f"Memory bank saved to {file_path}")

    def load(self, file_path):
        """Load the Memory Bank from a file (including EMA features).

        Args:
            file_path (str): File path to load from.
        """
        with open(file_path, 'rb') as f:
            bank_data = pickle.load(f)
            
        # Check parameter compatibility
        if bank_data['dim'] != self.dim:
            logger.warning(f"Feature dimension mismatch (saved: {bank_data['dim']}, current: {self.dim})")
        
        if bank_data['max_length'] != self.max_length:
            logger.warning(f"Max length mismatch (saved: {bank_data['max_length']}, current: {self.max_length})")
        
        # Update EMA decay if present in saved data
        if 'ema_decay' in bank_data:
            self.ema_decay = bank_data['ema_decay']
            logger.info(f"Loaded EMA decay: {self.ema_decay}")
        
        # Load bank data
        for class_id, features in bank_data['bank'].items():
            class_id = int(class_id)
            
            if class_id not in self.bank:
                self.bank[class_id] = deque(maxlen=self.max_length)
                self.ema_features[class_id] = None
                logger.info(f"Added new class {class_id} from saved bank")
            
            # Load features
            if features.size > 0:
                features_tensor = torch.tensor(features)
                # Clear existing queue and add new features
                self.bank[class_id] = deque(maxlen=self.max_length)
                for feature in features_tensor:
                    self.bank[class_id].append(feature)
        
        # Load EMA features
        if 'ema_features' in bank_data:
            for class_id, ema_feature in bank_data['ema_features'].items():
                class_id = int(class_id)
                if class_id in self.ema_features:
                    if ema_feature is not None:
                        self.ema_features[class_id] = torch.tensor(ema_feature)
                    else:
                        self.ema_features[class_id] = None
                else:
                    # Handle classes present in file but not in current bank
                    self.ema_features[class_id] = None
                    logger.warning(f"EMA feature for class {class_id} loaded but class not in current bank")

        logger.info(f"Memory bank loaded from {file_path}")
        logger.info(f"Loaded {sum(len(q) for q in self.bank.values())} features across {len(self.bank)} classes")
        logger.info(f"Loaded {sum(1 for e in self.ema_features.values() if e is not None)} EMA features")

    def __str__(self):
        """Return statistics of the Memory Bank."""
        sizes = []
        ema_count = 0
        for class_id in self.bank:
            sizes.append(len(self.bank[class_id]))
            if self.ema_features[class_id] is not None:
                ema_count += 1
        
        if sizes:  # ensure list is not empty
            min_size = min(sizes)
            max_size = max(sizes)
            total_features = sum(sizes)
            num_classes = len(self.bank)
            avg_size = total_features / num_classes if num_classes > 0 else 0
            
            return (
                f"MemoryBank(dim={self.dim}, classes={num_classes}, max_per_class={self.max_length})\n"
                f"Total features: {total_features}\n"
                f"EMA features: {ema_count}/{num_classes}\n"
                f"Features per class: min={min_size}, avg={avg_size:.1f}, max={max_size}"
            )
        else:
            return "MemoryBank is empty"

