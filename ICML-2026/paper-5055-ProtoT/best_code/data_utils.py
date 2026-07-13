"""
FineWeb dataset preparation with BPE tokenization (including training the BPE tokenizer)

This script handles:
1. Loading and preprocessing FineWeb-Edu dataset
2. Training a custom BPE tokenizer on the data
3. Creating variable-length training sequences
"""

import os
import re
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# BPE tokenizer imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import normalizers
from tokenizers.processors import TemplateProcessing


def create_causal_collate_fn(pad_idx, max_seq_len):
    def collate_fn(batch):
        inputs = [item[:-1] for item in batch]
        targets = [item[1:] for item in batch]
        
        # Pad to fixed length instead of dynamic padding
        padded_inputs = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
        padded_targets = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
        
        # Fill with pad_idx initially
        padded_inputs.fill_(pad_idx)
        padded_targets.fill_(pad_idx)
        
        # Copy actual sequences
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            seq_len = min(len(inp), max_seq_len)
            padded_inputs[i, :seq_len] = inp[:seq_len]
            padded_targets[i, :seq_len] = tgt[:seq_len]
            
        return padded_inputs, padded_targets
    return collate_fn


class NPZDataset(Dataset):
    """Efficient dataset using pre-tokenized NPZ files."""
    
    def __init__(self, npz_path, seq_length=512, max_num_datapoints=None):
        print(f"📂 Loading pre-tokenized data from {npz_path}")
        data = np.load(npz_path)
        self.tokens = data['tokens']
        self.offsets = data['offsets'] if 'offsets' in data else None
        self.seq_length = seq_length
        self.max_num_datapoints = max_num_datapoints
        
        if self.offsets is not None:
            self.num_docs = len(self.offsets) - 1
            print(f"✅ Loaded {len(self.tokens):,} tokens from {self.num_docs:,} documents")
        else:
            print(f"✅ Loaded {len(self.tokens):,} tokens")
    
    def __len__(self):
        if self.offsets is not None:
            num_datapoints = self.num_docs
        else:
            num_datapoints = len(self.tokens) // self.seq_length
        if self.max_num_datapoints is not None:
            num_datapoints = min(num_datapoints, self.max_num_datapoints)
        return num_datapoints

    def __getitem__(self, idx):
        if self.offsets is not None:
            # Document-based sampling
            start_idx = self.offsets[idx]
            end_idx = self.offsets[idx + 1]
            doc_tokens = self.tokens[start_idx:end_idx]
            
            # Truncate or pad
            if len(doc_tokens) > self.seq_length:
                tokens = doc_tokens[:self.seq_length]
            else:
                # Pad with token 0
                tokens = np.concatenate([doc_tokens, np.zeros(self.seq_length - len(doc_tokens), dtype=np.int32)])
        else:
            # Fixed-length sampling
            start_idx = idx * self.seq_length
            tokens = self.tokens[start_idx:start_idx + self.seq_length]

        return torch.tensor(tokens, dtype=torch.long)


class FineWebBPETokenizer:
    """
    Custom BPE tokenizer for FineWeb dataset with variable-length document handling.
    """
    
    def __init__(self, vocab_size=16000, cache_dir="tok"):
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<eot>"]  # Added end-of-text token
        self.specials = None
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.eot_token = "<eot>"  # End of text/document
        
    def train_tokenizer(self, documents, force_retrain=False):
        """Train BPE tokenizer on FineWeb documents."""
        os.makedirs(self.cache_dir, exist_ok=True)
        tokenizer_path = os.path.join(self.cache_dir, f"fineweb_bpe_{self.vocab_size}.json")
        
        if os.path.exists(tokenizer_path) and not force_retrain:
            print(f"📂 Loading cached FineWeb BPE tokenizer from {tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self._setup_special_tokens()
            return
        
        print(f"🔧 Training BPE tokenizer on FineWeb data (vocab_size={self.vocab_size})...")
        
        # Initialize BPE model
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        
        # Enhanced normalization for web text
        tokenizer.normalizer = normalizers.Sequence([
            NFD(),           # Unicode normalization
            Lowercase(),     # Lowercase
            StripAccents()   # Remove accents
        ])
        
        # Use whitespace pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()
        
        # BPE trainer with special handling for web content
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=3,  # Higher min frequency for web data
            continuing_subword_prefix="",
            end_of_word_suffix="</w>",
            show_progress=True
        )
        
        # Prepare training data
        print("📝 Preparing training texts...")
        training_texts = []
        for doc in tqdm(documents[:10000], desc="Processing documents for tokenizer training"):  # Use subset for training
            if isinstance(doc, dict):
                text = doc.get('text', '')
            else:
                text = str(doc)
            
            # Clean and prepare text
            text = self._clean_text(text)
            if len(text) > 50:  # Only use substantial texts
                training_texts.append(text)
        
        print(f"📊 Training tokenizer on {len(training_texts)} documents...")
        
        # Write training data temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            for text in training_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            # Train the tokenizer
            tokenizer.train([temp_file], trainer)
            
            # Set up post-processor for special tokens
            tokenizer.post_processor = TemplateProcessing(
                single=f"{self.bos_token} $A {self.eos_token}",
                special_tokens=[
                    (self.bos_token, tokenizer.token_to_id(self.bos_token)),
                    (self.eos_token, tokenizer.token_to_id(self.eos_token)),
                    (self.eot_token, tokenizer.token_to_id(self.eot_token)),
                ],
            )
            
            # Save tokenizer
            tokenizer.save(tokenizer_path)
            print(f"💾 Saved FineWeb BPE tokenizer to {tokenizer_path}")
            
        finally:
            os.unlink(temp_file)
        
        self.tokenizer = tokenizer
        self._setup_special_tokens()
        print(f"✅ FineWeb BPE tokenizer ready with {self.get_vocab_size()} tokens")
    
    def _clean_text(self, text):
        """Clean web text for tokenization."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove very long sequences of repeated characters
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def _setup_special_tokens(self):
        """Configure special tokens and padding."""
        if self.tokenizer is None:
            return
        
        # Validate that pad token has ID 0 as expected
        pad_id = self.tokenizer.token_to_id(self.pad_token)
        if pad_id != 0:
            print(f"⚠️  Warning: Pad token '{self.pad_token}' has ID {pad_id}, not 0. This may cause issues.")
        
        self.tokenizer.enable_padding(
            pad_id=pad_id,
            pad_token=self.pad_token,
        )
        self.tokenizer.enable_truncation(max_length=1024)
        # Dict of special tokens for easy access
        self.specials = {tok: self.tokenizer.token_to_id(tok) for tok in self.special_tokens}

        # To remove the </w> word suffix in decoding
        self.tokenizer.decoder = BPEDecoder(suffix="</w>")
    
    def get_pad_id(self):
        """Get the pad token ID."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.token_to_id(self.pad_token)
    
    def encode_document(self, text, max_length=None, add_eot=True):
        """
        Encode a document with proper handling of variable lengths.
        
        Args:
            text: Document text
            max_length: Maximum sequence length
            add_eot: Whether to add end-of-text token
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        
        # Clean the text
        text = self._clean_text(text)
        
        # Encode the text
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        
        # Add end-of-text token if requested
        if add_eot:
            eot_id = self.tokenizer.token_to_id(self.eot_token)
            token_ids.append(eot_id)
        
        # Handle length constraints
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    
    def clean_decode(self, text: str) -> str:
        # collapse spaces before punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        # collapse spaces around apostrophes
        text = re.sub(r"\s*'\s*", "'", text)
        # collapse spaces around quotes
        # text = re.sub(r'\s*"\s*', '"', text)
        return text.strip()
    
    def decode(self, token_ids):
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        text = self.tokenizer.decode(token_ids)
        return self.clean_decode(text) 

    def encode(self, text):
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        return self.tokenizer.encode(text).ids
