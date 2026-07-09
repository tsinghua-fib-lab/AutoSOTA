#!/usr/bin/env python3
"""
Simple PLATE Example: Fine-tuning on a text dataset

This example demonstrates how to use PLATE with just one hyperparameter (r).

Installation:
    pip install -e .
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from plate import PLATEConfig, get_plate_model


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def prepare_dataset(tokenizer, dataset_name="middle_english", max_samples=5000, max_length=512):
    """Load and prepare a dataset."""
    print(f"Loading {dataset_name}...")
    
    if dataset_name == "middle_english":
        # EN-ME dataset - use only the Middle English text
        ds = load_dataset("Qilex/EN-ME", split="train")
        texts = []
        for ex in ds:
            if len(texts) >= max_samples:
                break
            me_text = ex['translation']['me']
            if len(me_text.strip()) > 20:  # Filter very short texts
                texts.append(me_text)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    return TextDataset(encodings)


def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-3B"
    batch_size = 4
    learning_rate = 5e-4
    num_epochs = 1
    max_samples = 500
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # Configure PLATE
    print(f"\nConfiguring PLATE")
    config = PLATEConfig(
        r=128,
        col_tau=0.9,  # Input orthogonality threshold
        plate_alpha=1.0,  # Scaling factor
        max_rank=512,
        plate_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply PLATE adapter
    model = get_plate_model(model, config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(tokenizer, dataset_name="middle_english", max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
    
    print("\n✅ Training complete!")
    print(f"Model saved with PLATE adapter. Trainable params: {trainable:,}")


if __name__ == "__main__":
    main()
