# dataloader.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_dataloaders(batch_size=16):
    """
    Returns train, valid, and test dataloaders for QNLI.
    """
    # 1. Load QNLI dataset
    dataset = load_dataset('glue', 'qnli')
    
    train_data = dataset['train']
    validation_data = dataset['validation']
    
    train_df = train_data.to_pandas()
    validation_df = validation_data.to_pandas()
    
    # Merge to perform your custom 70/15/15 split logic
    merged_df = pd.concat([train_df, validation_df], ignore_index=True)
    
    train_df, temp_df = train_test_split(merged_df, test_size=0.3, random_state=42)
    test_df, valid_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # 2. Extract PAIRS: QNLI has 'question' and 'sentence' columns
    train_questions = list(train_df['question'])
    train_sentences = list(train_df['sentence'])
    
    valid_questions = list(valid_df['question'])
    valid_sentences = list(valid_df['sentence'])
    
    test_questions  = list(test_df['question'])
    test_sentences  = list(test_df['sentence'])
    
    # 3. Tokenize Pairs (pass both lists to tokenizer)
    train_encodings = tokenizer(train_questions, train_sentences, truncation=True, padding=True)
    valid_encodings = tokenizer(valid_questions, valid_sentences, truncation=True, padding=True)
    test_encodings  = tokenizer(test_questions,  test_sentences,  truncation=True, padding=True)
    
    train_labels = list(train_df['label'])
    test_labels  = list(test_df['label'])
    valid_labels = list(valid_df['label'])
    
    train_dataset = Custom_Dataset(train_encodings, train_labels)
    valid_dataset = Custom_Dataset(valid_encodings, valid_labels)
    test_dataset  = Custom_Dataset(test_encodings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader