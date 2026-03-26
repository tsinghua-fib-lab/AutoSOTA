# %%
import os
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

class STSDataset(Dataset):
    def __init__(self, tokenizer, file_name, max_seq_len=128, data_type='interactive', shuffle=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data_type = data_type
        self.ori_json = self.load_train(file_name)
        self.random_list = list(range(len(self.ori_json)))
        if shuffle:
            random.shuffle(self.random_list)
    
    def load_train(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_json = f.read()
        ori_json = json.loads(ori_json)
        return ori_json
    
    def __getitem__(self, idx):
        idx = self.random_list[idx]
        item = self.ori_json[idx]
        s1 = item['text1']
        s2 = item['text2']
        labels = float(item['label'])
        if self.data_type == 'interactive':
            T = self.tokenizer(s1, s2, add_special_tokens=True, max_length=self.max_seq_len, padding='max_length', truncation=True)
            input_ids = torch.tensor(T['input_ids'])
            attn_mask = torch.tensor(T['attention_mask'])
            token_type_ids = torch.tensor(T['token_type_ids'])
            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'token_type_ids': token_type_ids,
                'labels': torch.tensor(labels)
            }
        elif self.data_type == 'siamese':
            left_length = self.max_seq_len // 2
            if left_length < self.max_seq_len / 2:
                left_length += 1
            right_length = self.max_seq_len - left_length
            T1 = self.tokenizer(s1, add_special_tokens=True, max_length=left_length, padding='max_length', truncation=True)
            T2 = self.tokenizer(s2, add_special_tokens=True, max_length=right_length, padding='max_length', truncation=True)
            ss1 = torch.tensor(T1['input_ids'])
            mask1 = torch.tensor(T1['attention_mask'])
            tid1 = torch.tensor(T1['token_type_ids'])
            ss2 = torch.tensor(T2['input_ids'])
            mask2 = torch.tensor(T2['attention_mask'])
            tid2 = torch.ones(ss2.shape).long()
            return {
                'input_ids': torch.cat([ss1, ss2]),
                'attention_mask': torch.cat([mask1, mask2]),
                'token_type_ids': torch.cat([tid1, tid2]),
                'labels': torch.tensor(labels)
            }
        elif self.data_type == 'copy':
            left_length = self.max_seq_len // 2
            if left_length < self.max_seq_len / 2:
                left_length += 1
            right_length = self.max_seq_len - left_length
            T1 = self.tokenizer(s1, add_special_tokens=True, max_length=left_length, padding='max_length', truncation=True)
            T2 = self.tokenizer(s1, add_special_tokens=True, max_length=right_length, padding='max_length', truncation=True)
            ss1 = torch.tensor(T1['input_ids'])
            mask1 = torch.tensor(T1['attention_mask'])
            tid1 = torch.tensor(T1['token_type_ids'])
            ss2 = torch.tensor(T2['input_ids'])
            mask2 = torch.tensor(T2['attention_mask'])
            tid2 = torch.ones(ss2.shape).long()
            return {
                'input_ids': torch.cat([ss1, ss2]),
                'attention_mask': torch.cat([mask1, mask2]),
                'token_type_ids': torch.cat([tid1, tid2]),
                'labels': torch.tensor(1)
            }
    
    def __len__(self):
        return len(self.ori_json)
