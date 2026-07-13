

import os
import json
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import logging

GLUE_TASKS = {
    'cola': {'num_labels': 2, 'metric': 'matthews_corrcoef', 'is_regression': False, 'eval_split': 'validation'},
    'sst2': {'num_labels': 2, 'metric': 'accuracy', 'is_regression': False, 'eval_split': 'validation'},
    'mrpc': {'num_labels': 2, 'metric': 'f1', 'is_regression': False, 'eval_split': 'validation'},
    'qqp': {'num_labels': 2, 'metric': 'f1', 'is_regression': False, 'eval_split': 'validation'},
    'stsb': {'num_labels': 1, 'metric': 'pearson_spearman', 'is_regression': True, 'eval_split': 'validation'},
    'mnli': {'num_labels': 3, 'metric': 'accuracy', 'is_regression': False, 'eval_split': 'validation_matched'},
    'mnli-mm': {'num_labels': 3, 'metric': 'accuracy', 'is_regression': False, 'eval_split': 'validation_mismatched'},
    'qnli': {'num_labels': 2, 'metric': 'accuracy', 'is_regression': False, 'eval_split': 'validation'},
    'rte': {'num_labels': 2, 'metric': 'accuracy', 'is_regression': False, 'eval_split': 'validation'},
    'wnli': {'num_labels': 2, 'metric': 'accuracy', 'is_regression': False, 'eval_split': 'validation'},
}

class GLUEDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_length=512, task_name='sst2'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name.lower()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        if self.task_name in ['cola', 'sst2']:

            text = example['sentence']
        elif self.task_name == 'mrpc':

            text = f"{example['sentence1']} [SEP] {example['sentence2']}"
        elif self.task_name == 'qqp':

            text = f"{example['question1']} [SEP] {example['question2']}"
        elif self.task_name == 'stsb':

            text = f"{example['sentence1']} [SEP] {example['sentence2']}"
        elif self.task_name in ['mnli', 'mnli-mm']:

            text = f"{example['premise']} [SEP] {example['hypothesis']}"
        elif self.task_name == 'qnli':

            text = f"{example['question']} [SEP] {example['sentence']}"
        elif self.task_name == 'rte':

            text = f"{example['sentence1']} [SEP] {example['sentence2']}"
        elif self.task_name == 'wnli':

            text = f"{example['sentence1']} [SEP] {example['sentence2']}"
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

        if hasattr(self.tokenizer, 'encode'):

            tokens = self.tokenizer.encode(text).ids
        else:

            tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:

            tokens = tokens + [0] * (self.max_length - len(tokens))

        if self.task_name == 'stsb':

            label = float(example['label'])
        else:

            label = int(example['label']) if example['label'] != -1 else 0

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float if GLUE_TASKS[self.task_name]['is_regression'] else torch.long)
        }

class GLUEClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pooled_output):

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_glue_data(task_name, tokenizer, max_length=512, cache_dir=None, seed=42):

    task_name = task_name.lower()

    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unsupported GLUE task: {task_name}")

    task_config = GLUE_TASKS[task_name]
    eval_split = task_config['eval_split']

    if task_name == 'mnli-mm':
        dataset_name = 'mnli'
    else:
        dataset_name = task_name

    try:

        dataset = load_dataset('glue', dataset_name, cache_dir=cache_dir)

        train_data = dataset['train'].shuffle(seed=seed)
        eval_data = dataset[eval_split].shuffle(seed=seed + 1000)

        train_dataset = GLUEDataset(train_data, tokenizer, max_length, task_name)
        eval_dataset = GLUEDataset(eval_data, tokenizer, max_length, task_name)

        return train_dataset, eval_dataset

    except Exception as e:
        print(f"加载GLUE数据集失败: {e}")
        raise

def compute_glue_metrics(task_name, predictions, labels):

    import numpy as np

    task_name = task_name.lower()
    task_config = GLUE_TASKS[task_name]

    if task_config['is_regression']:

        if len(set(predictions)) <= 1:
            pearson_corr = 0.0
            spearman_corr = 0.0
        else:
            try:
                pearson_corr = pearsonr(predictions, labels)[0]
                spearman_corr = spearmanr(predictions, labels)[0]

                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
            except Exception:

                pearson_corr = 0.0
                spearman_corr = 0.0

        return {
            'pearson': float(pearson_corr),
            'spearman': float(spearman_corr),
            'pearson_spearman': float((pearson_corr + spearman_corr) / 2)
        }
    else:

        if predictions.ndim > 1 and predictions.shape[1] > 1:

            predictions = np.argmax(predictions, axis=1)
        else:

            predictions = predictions.astype(int)

        acc = accuracy_score(labels, predictions)

        if task_config['metric'] == 'matthews_corrcoef':
            mcc = matthews_corrcoef(labels, predictions)

            if np.isnan(mcc):
                mcc = 0.0
            return {'accuracy': float(acc), 'matthews_corrcoef': float(mcc)}
        elif task_config['metric'] == 'f1':
            f1 = f1_score(labels, predictions, average='binary' if task_config['num_labels'] == 2 else 'macro')
            return {'accuracy': float(acc), 'f1': float(f1)}
        else:
            return {'accuracy': float(acc)}

def create_glue_collate_fn(pad_token_id=0):

    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'labels': labels}
    return collate_fn

class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.001, maximize=True):
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_better(self, score):
        if self.maximize:
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
