import torch
from collections import defaultdict
import numpy as np


class BatchSampler:
    def __init__(self, samples: torch.Tensor, labels: torch.Tensor, batch_size: int, num_class_eff=None, num_class_eff_min=None, num_class_eff_max=None, online=False):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size

        self.label_dict = defaultdict(list)
        self.classes = sorted(list(set(labels.tolist())))

        for i, label in enumerate(labels):
            self.label_dict[label.item()].append(i)

        self.num_class = len(self.classes)

        self.num_class_eff_min = num_class_eff_min
        self.num_class_eff_max = num_class_eff_max
        self.max_samples_per_class = [len(self.label_dict[label]) for label in self.classes]
        self.num_class_eff = num_class_eff if num_class_eff_min is None and num_class_eff is not None else self.num_class if num_class_eff_min is None else None
        self.online = online
        self.remaining_indices_per_class = None

        if self.online:
            # In online mode, initialize remaining_indices_per_class with all available indices per class
            self.reset_remaining_indices()

    def reset_remaining_indices(self):
        """Resets the remaining indices for online sampling."""
        self.remaining_indices_per_class = {label: indices[:] for label, indices in self.label_dict.items()}
        for label in self.remaining_indices_per_class:
            np.random.shuffle(self.remaining_indices_per_class[label])
    
    def generate_indices(self) -> list[int] | None:
        final_indices = []

        # Determine the number of classes to sample (num_class_eff)
        if self.num_class_eff is None and self.num_class_eff_min is not None and self.num_class_eff_max is not None:
            num_class_eff = np.random.randint(self.num_class_eff_min, self.num_class_eff_max + 1)
        else:
            num_class_eff = self.num_class_eff
         
        # Offline setting: select classes first
        if num_class_eff > 0 and num_class_eff < self.num_class:
            selected_classes = np.random.choice(self.classes, num_class_eff, replace=False)
            
        else:
            selected_classes = self.classes

        # Gather all indices from the selected classes
        all_indices = []
        for label in selected_classes:
            all_indices.extend(self.label_dict[label])
            
        if self.batch_size == -1 or len(all_indices) < self.batch_size:
            final_indices = all_indices
        else:

            # Randomly select batch_size indices from all available indices
            final_indices = np.random.choice(all_indices, self.batch_size, replace=False).tolist()

        np.random.shuffle(final_indices)

        return final_indices
    
    
    
class OnlineSampler:
    def __init__(self, samples: torch.Tensor, labels: torch.Tensor, gamma: float, slots:int, batch_size: int, num_class_eff=None, num_class_eff_min=None, num_class_eff_max=None):
        
        self.label_dict = defaultdict(list)
        self.classes = sorted(list(set(labels.tolist())))

        for i, label in enumerate(labels):
            self.label_dict[label.item()].append(i)
            
        self.labels = list(self.label_dict.keys())
        self.labels.sort()

        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)

           
           
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class_eff
        self.c = 0
        
        if self.gamma == -1:
           self.create_indices_class(labels)
        else:
            self.create_indices(labels)
        
        
    def generate_indices(self) -> list[int] | None:
        if self.c + self.batch_size <= len(self.indices):
            batch = self.indices[self.c:self.c + self.batch_size]
            self.c += self.batch_size
            return batch
        else:
            return None
    
    def create_indices(self, test_labels):
            
        selected_classes = self.classes
        num_selected_classes = len(selected_classes)
        
        final_indices = []
        rng = np.random.default_rng()
        label_distribution = rng.dirichlet([self.gamma] * self.num_slots, self.num_class)
        slot_indices = [[] for _ in range(self.num_slots)]
        
        
        for l,label in enumerate(selected_classes):
            
            indices = np.array(self.label_dict[label])
            partition = label_distribution[self.labels.index(label),:]
            samples_per_splits = np.split(indices, (np.cumsum(partition)[:-1] * len(indices)).astype(int))
            for s, ids in enumerate(samples_per_splits):
                slot_indices[s].extend(ids)
            
        for s_ids in slot_indices:
            permutation = np.random.permutation(range(len(s_ids)))
            ids = []
            for i in permutation:
                ids.extend(s_ids[i] if isinstance(s_ids[i], list) else [s_ids[i]])
            final_indices.extend(ids)
        self.indices = final_indices    
    
    def create_indices_class(self, test_labels):
        # When gamma == -1, we want to order the classes separately and shuffle both classes and samples
        final_indices = []
    
        # Shuffle the order of the classes
        shuffled_classes = np.random.permutation(self.classes)
    
        for label in shuffled_classes:
            indices = np.array(self.label_dict[label])
            # Shuffle the samples within each class
            np.random.shuffle(indices)
            final_indices.extend(indices)
    
        self.indices = final_indices
