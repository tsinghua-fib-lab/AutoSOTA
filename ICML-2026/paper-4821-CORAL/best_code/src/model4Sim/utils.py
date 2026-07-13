import numpy as np
import torch
import random
from torch.utils.data import Dataset

def get_num_node_unique(data_list):
    sequences, _, _, targets,_,_ = data_list
    item_set = set()
    for seq in sequences:
        item_set.update(seq)
    item_set.update(targets)
    if 0 in item_set:
        item_set.remove(0)
    return len(item_set)

class Data(Dataset):
    def __init__(self, raw_data, num_items, max_len, mask_prob, mode='train'):
        """
        SASRec Data Preparation with CORAL Support.
        args:
            raw_data: [sequences, _, cats, targets, _, ratings]
        """
        self.raw_inputs = raw_data[0]
        self.raw_cats = raw_data[2]    # Added: Category Sequences
        self.raw_targets = raw_data[3]
        self.raw_ratings = raw_data[5] # Added: Rating Sequences
        
        self.num_items = num_items
        self.max_len = max_len
        self.mode = mode
        
        # Prepare Data
        processed = self._prepare_sasrec_data()
        self.inputs = np.asarray(processed[0])
        self.mask = np.asarray(processed[1])
        self.targets = np.asarray(processed[2])
        
        # New attributes for CORAL
        self.cat_inputs = np.asarray(processed[3])
        self.rate_inputs = np.asarray(processed[4])
        self.user_ids = np.asarray(processed[5]) # Track original user index
        
        self.length = len(self.inputs)

    def _prepare_sasrec_data(self):
        new_inputs = []
        new_masks = []
        new_targets = []
        new_cats = []
        new_rates = []
        new_users = []
        
        for i in range(len(self.raw_inputs)):
            seq = self.raw_inputs[i]
            cat_seq = self.raw_cats[i]
            rate_seq = self.raw_ratings[i]
            
            # Remove existing padding (0) to handle cleanly
            # We assume seq, cat_seq, rate_seq are aligned
            seq_clean = []
            cat_clean = []
            rate_clean = []
            
            for idx, item in enumerate(seq):
                if item != 0:
                    seq_clean.append(item)
                    cat_clean.append(cat_seq[idx])
                    rate_clean.append(rate_seq[idx])
            
            if len(seq_clean) < 2:
                continue
                
            if self.mode == 'train':
                # SASRec Training: Input is seq[:-1], Target is seq[1:]
                
                if len(seq_clean) > self.max_len + 1:
                    seq_clean = seq_clean[-(self.max_len + 1):]
                    cat_clean = cat_clean[-(self.max_len + 1):]
                    rate_clean = rate_clean[-(self.max_len + 1):]
                
                input_seq = seq_clean[:-1]
                target_seq = seq_clean[1:]
                
                # We don't necessarily need cats/rates for training SASRec loss, 
                # but we pad them to keep structure consistent if needed later
                input_cat = cat_clean[:-1]
                input_rate = rate_clean[:-1]
                
                pad_len = self.max_len - len(input_seq)
                if pad_len > 0:
                    input_seq = input_seq + [0] * pad_len
                    target_seq = target_seq + [0] * pad_len
                    input_cat = input_cat + [0] * pad_len
                    input_rate = input_rate + [0] * pad_len
                else:
                    input_seq = input_seq[-self.max_len:]
                    target_seq = target_seq[-self.max_len:]
                    input_cat = input_cat[-self.max_len:]
                    input_rate = input_rate[-self.max_len:]
                
                new_inputs.append(input_seq)
                new_targets.append(target_seq)
                new_masks.append([1 if x != 0 else 0 for x in input_seq])
                new_cats.append(input_cat)
                new_rates.append(input_rate)
                new_users.append(i)
                
            else:
                # Test Mode: Input is full sequence (up to max_len)
                target_item = self.raw_targets[i]
                
                input_seq = seq_clean
                input_cat = cat_clean
                input_rate = rate_clean
                
                if len(input_seq) > self.max_len:
                    input_seq = input_seq[-self.max_len:]
                    input_cat = input_cat[-self.max_len:]
                    input_rate = input_rate[-self.max_len:]
                
                pad_len = self.max_len - len(input_seq)
                mask_seq = [1] * len(input_seq) + [0] * pad_len
                
                input_seq = input_seq + [0] * pad_len
                input_cat = input_cat + [0] * pad_len
                input_rate = input_rate + [0] * pad_len
                
                new_inputs.append(input_seq)
                new_targets.append(target_item)
                new_masks.append(mask_seq)
                new_cats.append(input_cat)
                new_rates.append(input_rate)
                new_users.append(i)
                
        return [new_inputs, new_masks, new_targets, new_cats, new_rates, new_users]

    def __getitem__(self, index):
        return [
            torch.tensor(self.inputs[index], dtype=torch.long), 
            torch.tensor(self.mask[index], dtype=torch.long), 
            torch.tensor(self.targets[index], dtype=torch.long),
            # Supplementary for CORAL
            torch.tensor(self.cat_inputs[index], dtype=torch.long),
            torch.tensor(self.rate_inputs[index], dtype=torch.float),
            torch.tensor(self.user_ids[index], dtype=torch.long)
        ]

    def __len__(self):
        return self.length