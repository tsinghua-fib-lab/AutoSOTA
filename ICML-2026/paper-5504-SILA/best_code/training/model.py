#!/usr/bin/env python3
"""Language model wrapper with soft token injection for SelfIE adapter training."""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from selfie_adapters.projection import create_projection_module


class SelfIEModel:
    """
    Wraps a language model with projection module for SelfIE adapter training.
    
    Handles:
    - Loading and freezing the language model
    - Creating the projection module (adapter)
    - Soft token injection at <|reserved_special_token_0|> positions
    - Computing cross-entropy loss on target labels
    """
    
    def __init__(self, config, model_dim: int, cache_dir: Optional[str] = None):
        """
        Args:
            config: Configuration object
            model_dim: Model dimension (must match vectors)
            cache_dir: Optional HuggingFace cache directory
        """
        self.config = config
        self.model_dim = model_dim
        
        # Load language model
        print(f"\nLoading language model: {config.model.name}")
        cache_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        load_kwargs = {
            "low_cpu_mem_usage": True,
            "dtype": "auto",
            **cache_kwargs,
        }
        
        # Add 8-bit quantization if requested
        if config.model.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["load_in_8bit"] = True
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                load_kwargs.pop("dtype", None)
                print("✓ Using 8-bit quantization (bitsandbytes)")
            except ImportError:
                print("Warning: bitsandbytes not available, falling back to regular loading")
        
        load_kwargs["device_map"] = config.model.device_map
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name, **cache_kwargs)
        self.tokenizer.clean_up_tokenization_spaces = False
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Handle injection token (add if not in vocabulary)
        injection_token = config.soft_prompt.reserved_token
        num_added = 0
        
        if injection_token not in self.tokenizer.get_vocab():
            num_added = self.tokenizer.add_tokens([injection_token], special_tokens=True)
            print(f"✓ Added injection token '{injection_token}' to tokenizer vocabulary")
        else:
            print(f"✓ Injection token '{injection_token}' already in tokenizer vocabulary")
        
        print(f"Loading model with dtype='auto' and device_map='{config.model.device_map}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            **load_kwargs,
        )
        
        # Resize model embeddings if we added a token
        if num_added > 0:
            tokenizer_vocab_size = len(self.tokenizer)
            model_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            
            if tokenizer_vocab_size > model_vocab_size:
                self.model.resize_token_embeddings(tokenizer_vocab_size)
                print(f"✓ Resized model embeddings to {tokenizer_vocab_size} tokens")
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Loaded and frozen {total_params:,} language model parameters")
        
        # Enable gradient checkpointing if requested
        if config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✓ Enabled gradient checkpointing")
        
        # Get first device
        self.first_device = next(self.model.parameters()).device
        print(f"✓ First device: {self.first_device}")
        
        # Create projection module
        print("\nCreating projection module...")
        self.projection = create_projection_module(
            projection_type=config.projection.type,
            dim=model_dim,
            normalize_input=config.projection.normalize_input,
            device=self.first_device,
            init_scale=config.projection.init_scale,
            low_rank_rank=config.projection.low_rank_rank,
            low_rank_init_factor=config.projection.low_rank_init_factor,
        )
        
        print("✓ Projection module kept in float32 for training stability")
        
        # Prepare template
        self.template = config.soft_prompt.template
        self.injection_token = config.soft_prompt.reserved_token
        self._prepare_template()
    
    def _prepare_template(self):
        """Pre-compute template tokens and injection positions."""
        self.template_tokens = self.tokenizer(
            self.template,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        inject_token_id = self.tokenizer.convert_tokens_to_ids(self.injection_token)
        self.inject_positions = [
            i for i, token_id in enumerate(self.template_tokens["input_ids"][0])
            if token_id == inject_token_id
        ]
        
        print("\nTemplate prepared:")
        print(f"  Length: {self.template_tokens['input_ids'].shape[1]} tokens")
        print(f"  Injection token: {self.injection_token}")
        print(f"  Injection positions: {self.inject_positions}")
    
    def compute_loss(
        self,
        vectors: torch.Tensor,
        target_labels: List[str],
        label_smoothing: float = 0.0,
        max_loss: float = float("inf"),
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute cross-entropy loss for predicting target labels.
        
        Args:
            vectors: Input vectors of shape (batch_size, model_dim)
            target_labels: List of target label strings (with closing quote and EOS)
            label_smoothing: Label smoothing factor
            max_loss: Maximum loss value to clamp to
        
        Returns:
            Tuple of (loss tensor, statistics dict)
        """
        batch_size = vectors.shape[0]
        
        vectors = vectors.to(device=self.first_device, dtype=torch.float32)
        soft_tokens = self.projection(vectors)
        
        model_dtype = next(self.model.parameters()).dtype
        soft_tokens = soft_tokens.to(dtype=model_dtype)
        
        template_input_ids = self.template_tokens["input_ids"].to(self.first_device)
        with torch.no_grad():
            embed_layer = self.model.get_input_embeddings()
            template_embeds = embed_layer(template_input_ids)
        
        all_sequences = []
        all_target_lengths = []
        
        for i in range(batch_size):
            modified_embeds = template_embeds.clone()
            for pos in self.inject_positions:
                modified_embeds[0, pos, :] = soft_tokens[i]
            
            target_tokens = self.tokenizer(
                target_labels[i],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.first_device)
            
            with torch.no_grad():
                target_embeds = embed_layer(target_tokens["input_ids"])
            
            full_sequence = torch.cat([modified_embeds, target_embeds], dim=1)
            
            all_sequences.append(full_sequence)
            all_target_lengths.append(target_tokens["input_ids"].shape[1])
        
        # Pad sequences
        max_seq_len = max(seq.shape[1] for seq in all_sequences)
        padded_sequences = []
        attention_masks = []
        
        for seq in all_sequences:
            seq_len = seq.shape[1]
            if seq_len < max_seq_len:
                padding = torch.zeros(
                    1, max_seq_len - seq_len, seq.shape[2],
                    device=seq.device, dtype=seq.dtype
                )
                padded_seq = torch.cat([seq, padding], dim=1)
            else:
                padded_seq = seq
            
            mask = torch.cat([
                torch.ones(1, seq_len, device=seq.device, dtype=torch.long),
                torch.zeros(1, max_seq_len - seq_len, device=seq.device, dtype=torch.long),
            ], dim=1)
            
            padded_sequences.append(padded_seq)
            attention_masks.append(mask)
        
        batched_embeds = torch.cat(padded_sequences, dim=0)
        batched_masks = torch.cat(attention_masks, dim=0)
        
        outputs = self.model(
            inputs_embeds=batched_embeds,
            attention_mask=batched_masks,
        )
        
        template_len = self.template_tokens["input_ids"].shape[1]
        all_losses = []
        total_valid_tokens = 0
        total_clamped_tokens = 0
        
        loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        
        for i in range(batch_size):
            target_len = all_target_lengths[i]
            target_logits = outputs.logits[i, template_len-1 : template_len+target_len-1]
            
            target_tokens = self.tokenizer(
                target_labels[i],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.first_device)
            target_ids = target_tokens["input_ids"][0]
            
            token_losses = loss_fn(target_logits, target_ids)
            clamped_losses = torch.clamp(token_losses, max=max_loss)
            sequence_loss = clamped_losses.mean()
            
            all_losses.append(sequence_loss)
            total_valid_tokens += len(token_losses)
            total_clamped_tokens += int((token_losses > max_loss).sum().item())
        
        final_loss = torch.stack(all_losses).mean()
        
        with torch.no_grad():
            soft_token_norms = torch.norm(soft_tokens, p=2, dim=-1)
            stats = {
                "mean_soft_token_norm": soft_token_norms.mean().item(),
                "max_soft_token_norm": soft_token_norms.max().item(),
                "batch_size": batch_size,
                "total_valid_tokens": total_valid_tokens,
                "total_clamped_tokens": total_clamped_tokens,
                "max_seq_len": max_seq_len,
                "template_len": template_len,
                "mean_target_len": sum(all_target_lengths) / len(all_target_lengths),
                "max_target_len": max(all_target_lengths),
            }
            
            if hasattr(self.projection, "get_scale"):
                stats["scale"] = self.projection.get_scale()
            if hasattr(self.projection, "get_bias_norm"):
                stats["bias_norm"] = self.projection.get_bias_norm()
        
        return final_loss, stats
    
    def generate_descriptions(
        self,
        vectors: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> List[str]:
        """Generate natural language descriptions for vectors."""
        batch_size = vectors.shape[0]
        
        vectors = vectors.to(device=self.first_device, dtype=torch.float32)
        
        with torch.no_grad():
            soft_tokens = self.projection(vectors)
            model_dtype = next(self.model.parameters()).dtype
            soft_tokens = soft_tokens.to(dtype=model_dtype)
        
        template_input_ids = self.template_tokens["input_ids"].to(self.first_device)
        with torch.no_grad():
            embed_layer = self.model.get_input_embeddings()
            template_embeds = embed_layer(template_input_ids)
        
        all_inputs_embeds = []
        
        for i in range(batch_size):
            modified_embeds = template_embeds.clone()
            for pos in self.inject_positions:
                modified_embeds[0, pos, :] = soft_tokens[i]
            all_inputs_embeds.append(modified_embeds)
        
        batched_embeds = torch.cat(all_inputs_embeds, dim=0)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=batched_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_texts = []
        for output_ids in outputs:
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def parameters(self):
        """Return trainable parameters (only projection parameters)."""
        return self.projection.parameters()
