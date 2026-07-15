#!/usr/bin/env python3
"""
Label Generator for SAE Reflective Coherence Training.
Maps SAE decoder vectors to natural language descriptions using soft prompts.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import LabelGeneratorConfig
from selfie_adapters import load_adapter


class AdapterProjection:
    """Wrapper that loads and uses SelfIE adapters for projection."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.adapter = load_adapter(checkpoint_path, device=str(device))
        self.metadata = self.adapter.get_metadata()
        self.model_dim = self.adapter.model_dim

    def forward(self, sae_vectors: torch.Tensor) -> torch.Tensor:
        # Disable adapter's built-in normalization since evaluation_functions.py
        # handles normalization before scaling (to make scale_values work correctly)
        return self.adapter.transform(sae_vectors, normalize_input=False)
    
    def __call__(self, sae_vectors: torch.Tensor) -> torch.Tensor:
        return self.forward(sae_vectors)
    
    def num_parameters(self) -> int:
        return self.metadata['num_parameters']
    
    def get_metadata(self) -> dict:
        return self.metadata
    
    def get_projection_stats(self) -> dict:
        return self.adapter.get_projection_stats()


class IdentityProjection:
    """Simple identity projection for baseline runs."""
    
    def __init__(self, model_dim: int, device: torch.device):
        self.model_dim = model_dim
        self.device = device

    def forward(self, sae_vectors: torch.Tensor) -> torch.Tensor:
        return sae_vectors.to(self.device)
    
    def __call__(self, sae_vectors: torch.Tensor) -> torch.Tensor:
        return self.forward(sae_vectors)
    
    def num_parameters(self) -> int:
        return 0
    
    def get_metadata(self) -> dict:
        return {'projection_type': 'identity', 'model_dim': self.model_dim, 'num_parameters': 0}
    
    def get_projection_stats(self) -> dict:
        return {'scale': 1.0, 'bias_norm': 0.0}


def create_projection_module(checkpoint_path: str, device: torch.device, model_dim: int = None, **kwargs):
    """Factory function to create projection modules."""
    if device == "auto" or str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    
    if checkpoint_path is None:
        if model_dim is None:
            raise ValueError("model_dim is required when using identity projection")
        return IdentityProjection(model_dim=model_dim, device=device)
    
    return AdapterProjection(checkpoint_path=checkpoint_path, device=device)


class LabelGenerator(nn.Module):
    """
    Neural network that maps SAE decoder vectors to text descriptions.

    Architecture:
    - Single linear projection from model_dim to model_dim
    - Creates one soft token that is injected at each reserved token position in the template
    - Template uses explicit special token syntax with reserved token placeholders (configurable)
    """

    def __init__(
        self,
        model_dim: int,
        base_model,  # ObservableLanguageModel for generation
        config: Optional[LabelGeneratorConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = LabelGeneratorConfig()

        # Validate num_soft_tokens == 1
        if config.num_soft_tokens != 1:
            raise ValueError(
                f"num_soft_tokens must be 1 for reserved token approach, got {config.num_soft_tokens}. "
                f"This implementation assumes a single soft token is injected at multiple reserved token positions."
            )

        self.config = config
        self.model_dim = model_dim
        self.base_model = base_model
        
        # Get device from base_model (already resolved to embedding layer's device for multi-GPU)
        self.device = base_model.device
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        # Get adapter checkpoint path
        adapter_checkpoint_path = getattr(config, "adapter_checkpoint_path", None)

        # Create adapter-based projection (or identity projection if checkpoint_path is None)
        self.projection = create_projection_module(
            checkpoint_path=adapter_checkpoint_path,
            device=self.device,
            model_dim=model_dim,  # Required for identity projection fallback
        )

        # Print projection info
        metadata = self.projection.get_metadata()
        print(
            f"Using {metadata['projection_type']} adapter with {self.projection.num_parameters():,} parameters"
        )

    def forward(self, sae_vectors: torch.Tensor) -> list[str]:
        """
        Generate descriptions for a batch of SAE vectors.

        Args:
            sae_vectors: Batch of SAE vectors, shape (batch_size, model_dim)

        Returns:
            List of generated description strings
        """
        # Create soft tokens for the entire batch at once
        batch_soft_tokens = self._create_soft_tokens_batch(sae_vectors)

        # Generate descriptions for the entire batch
        descriptions = self._generate_with_soft_prompt_batch(batch_soft_tokens)

        return descriptions

    def _create_soft_tokens_batch(self, sae_vectors: torch.Tensor) -> torch.Tensor:
        """
        Create soft tokens by projecting SAE vectors through the adapter.
        Since we use a single soft token injected at multiple positions, we create
        one soft token per batch item.

        Args:
            sae_vectors: Batch of SAE vectors of shape (batch_size, model_dim)

        Returns:
            Soft tokens of shape (batch_size, 1, model_dim)
        """
        # Apply adapter transformation
        assert self.config.num_soft_tokens == 1, (
            "Only one soft token is supported for now"
        )
        # Adapter handles normalization, device, dtype automatically
        soft_tokens = self.projection(sae_vectors)  # (batch_size, model_dim)

        # Add dimension for single soft token: (batch_size, 1, model_dim)
        soft_tokens = soft_tokens.unsqueeze(1)

        # Ensure dtype matches the base model
        if hasattr(self.base_model, "dtype") and self.base_model.dtype is not None:
            soft_tokens = soft_tokens.to(dtype=self.base_model.dtype)

        return soft_tokens

    def _generate_with_soft_prompt_batch(
        self, batch_soft_tokens: torch.Tensor
    ) -> list[str]:
        """
        Generate descriptions using soft tokens injected at reserved token positions.

        Args:
            batch_soft_tokens: Tensor of shape (batch_size, 1, model_dim)

        Returns:
            List of generated description strings
        """
        template = self.config.template
        reserved_token = self.config.reserved_token
        batch_size = batch_soft_tokens.shape[0]

        # Check if template contains reserved tokens
        if reserved_token not in template:
            raise ValueError(
                f"Template must contain {reserved_token} tokens for injection. "
                f"Current template: {template}"
            )

        # Count number of reserved tokens
        num_inject_positions = template.count(reserved_token)

        # Tokenize template with special tokens
        template_tokens = self.base_model.tokenizer(
            template, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # Get special token ID for position finding
        inject_token_id = self.base_model.tokenizer.convert_tokens_to_ids(
            reserved_token
        )

        # Get template embeddings
        with torch.no_grad():
            template_embeds = self.base_model._original_model.model.embed_tokens(
                template_tokens["input_ids"]
            )  # Shape: (1, template_length, hidden_size)

        # Find positions of reserved tokens in the tokenized sequence
        inject_positions = []

        for i, token_id in enumerate(template_tokens["input_ids"][0]):
            if token_id == inject_token_id:
                inject_positions.append(i)

        if len(inject_positions) != num_inject_positions:
            raise ValueError(
                f"Mismatch between {reserved_token} count in text ({num_inject_positions}) "
                f"and tokenized positions ({len(inject_positions)}). "
                f"The tokenizer may not recognize {reserved_token} as a single token."
            )

        # Expand template embeddings to match batch size
        template_embeds = template_embeds.expand(batch_size, -1, -1)
        # Shape: (batch_size, template_length, hidden_size)

        # Ensure soft tokens match dtype and device
        batch_soft_tokens = batch_soft_tokens.to(
            dtype=template_embeds.dtype, device=template_embeds.device
        )

        # Create modified embeddings by replacing reserved token positions with soft tokens
        modified_embeds = template_embeds.clone()

        # Replace each reserved token position with the soft token
        for pos in inject_positions:
            modified_embeds[:, pos, :] = batch_soft_tokens[
                :, 0, :
            ]  # Use single soft token

        # Create attention mask for the modified sequences
        seq_length = modified_embeds.shape[1]
        attention_mask = torch.ones(
            (batch_size, seq_length), dtype=torch.long, device=self.device
        )

        # Generate using the modified embeddings
        with torch.no_grad():
            outputs = self.base_model._original_model.generate(
                inputs_embeds=modified_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_generation_length,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.base_model.tokenizer.pad_token_id,
                eos_token_id=self.base_model.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens for each sample in the batch
        descriptions = []
        for i in range(batch_size):
            # Based on evidence: generate() with inputs_embeds returns only new tokens
            # No need to slice off the input portion
            new_tokens = outputs[i]

            generated_text = self.base_model.tokenizer.decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Extract label based on config setting
            # Template ends with 'The meaning of "..." is "' so we want the label text
            if self.config.strip_last_quote:
                # New behavior: strip the last closing quote (allows quotes inside labels)
                if '"' in generated_text:
                    label = generated_text.rsplit('"', 1)[0]
                else:
                    # No closing quote found, use the whole generation
                    label = generated_text
            else:
                # Legacy behavior: take everything up to the first closing quote
                if '"' in generated_text:
                    label = generated_text.split('"')[0]
                else:
                    # No closing quote found, use the whole generation
                    label = generated_text

            descriptions.append(label)

        return descriptions

    def compute_loss(
        self,
        sae_vectors: torch.Tensor,
        target_labels: List[str],
        latent_ids: Optional[List[int]] = None,
        label_smoothing: float = 0.0,
        max_loss: float = float("inf"),
    ) -> Tuple[torch.Tensor, dict]:
        """
        Efficient loss computation for batches of individual (sae_vector, target_label) pairs.

        This is now the main loss function - it processes flattened batches where each element
        is a single (SAE vector, target label) pair. Batch size = number of forward passes.

        Args:
            sae_vectors: Batch of SAE vectors (batch_size, model_dim)
            target_labels: List of target label strings (length = batch_size)
            latent_ids: Optional list of latent IDs for grouping loss computation
            label_smoothing: Label smoothing factor
            max_loss: Maximum loss value to prevent NaN gradients

        Returns:
            Tuple of (loss tensor, statistics dict)
        """
        batch_size = sae_vectors.shape[0]

        if len(target_labels) != batch_size:
            raise ValueError(
                f"target_labels length {len(target_labels)} doesn't match batch_size {batch_size}"
            )

        # DEBUG: Log EOS token on first call (to avoid spam)
        if not hasattr(self, "_eos_token_logged"):
            eos_token = self.base_model.tokenizer.eos_token
            print(f"DEBUG: Using EOS token: '{eos_token}' (repr: {repr(eos_token)})")
            self._eos_token_logged = True

        # Create soft tokens for the batch
        batch_soft_tokens = self._create_soft_tokens_batch(sae_vectors)

        # Get template setup
        template = self.config.template
        reserved_token = self.config.reserved_token
        template_tokens = self.base_model.tokenizer(
            template, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        inject_token_id = self.base_model.tokenizer.convert_tokens_to_ids(
            reserved_token
        )
        inject_positions = [
            i
            for i, token_id in enumerate(template_tokens["input_ids"][0])
            if token_id == inject_token_id
        ]

        # Get template embeddings
        with torch.no_grad():
            template_embeds = self.base_model._original_model.model.embed_tokens(
                template_tokens["input_ids"]
            )  # (1, template_length, hidden_size)

        # Build all sequences for batched processing
        all_sequences = []
        all_target_lengths = []

        for i in range(batch_size):
            # Get soft token for this SAE vector
            soft_token = batch_soft_tokens[i : i + 1]  # (1, 1, hidden_size)

            # Create modified template with soft token injected
            modified_template = (
                template_embeds.clone()
            )  # (1, template_length, hidden_size)
            for pos in inject_positions:
                modified_template[:, pos, :] = soft_token[:, 0, :]

            # Tokenize target
            target_text = target_labels[i] + '"' + self.base_model.tokenizer.eos_token
            target_tokens = self.base_model.tokenizer(
                target_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            # Get target embeddings
            with torch.no_grad():
                target_embeds = self.base_model._original_model.model.embed_tokens(
                    target_tokens["input_ids"]
                )  # (1, target_length, hidden_size)

            # Create full sequence
            full_sequence = torch.cat([modified_template, target_embeds], dim=1)
            all_sequences.append(full_sequence)
            all_target_lengths.append(target_tokens["input_ids"].shape[1])

        if not all_sequences:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        # Pad all sequences to the same length for batching
        max_seq_len = max(seq.shape[1] for seq in all_sequences)
        padded_sequences = []
        attention_masks = []

        for seq in all_sequences:
            seq_len = seq.shape[1]
            if seq_len < max_seq_len:
                # Pad with zeros
                padding = torch.zeros(
                    1,
                    max_seq_len - seq_len,
                    seq.shape[2],
                    device=seq.device,
                    dtype=seq.dtype,
                )
                padded_seq = torch.cat([seq, padding], dim=1)
            else:
                padded_seq = seq

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat(
                [
                    torch.ones(1, seq_len, device=seq.device, dtype=torch.long),
                    torch.zeros(
                        1, max_seq_len - seq_len, device=seq.device, dtype=torch.long
                    ),
                ],
                dim=1,
            )

            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)

        # Stack into single batch
        batched_sequences = torch.cat(
            padded_sequences, dim=0
        )  # (batch_size, max_seq_len, hidden_size)
        batched_attention_masks = torch.cat(
            attention_masks, dim=0
        )  # (batch_size, max_seq_len)

        print(
            f"Processing {batch_size} sequences in 1 forward pass (max_len={max_seq_len})"
        )

        # Single forward pass for all sequences
        outputs = self.base_model._original_model(
            inputs_embeds=batched_sequences,
            attention_mask=batched_attention_masks,
        )

        # Calculate losses for each sequence
        prompt_length = template_embeds.shape[1]
        all_losses = []
        all_token_losses = []
        total_clamped_tokens = 0
        total_valid_tokens = 0

        for i in range(batch_size):
            target_length = all_target_lengths[i]

            # Extract logits for this sequence's target portion
            target_logits = outputs.logits[
                i, prompt_length - 1 : prompt_length + target_length - 1
            ]

            # Get target tokens for this sequence
            target_text = target_labels[i] + '"' + self.base_model.tokenizer.eos_token
            target_tokens = self.base_model.tokenizer(
                target_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            # Compute loss for this sequence
            target_logits_flat = target_logits.view(-1, target_logits.shape[-1])
            target_ids_flat = target_tokens["input_ids"].view(-1)

            loss_fn = torch.nn.CrossEntropyLoss(
                reduction="none", label_smoothing=label_smoothing
            )
            token_losses = loss_fn(target_logits_flat, target_ids_flat)

            # Clamp and average
            clamped_losses = torch.clamp(token_losses, max=max_loss)
            sequence_loss = clamped_losses.mean()
            all_losses.append(sequence_loss)

            # Collect diagnostics
            all_token_losses.extend(token_losses.detach().cpu().tolist())
            total_clamped_tokens += int((token_losses > max_loss).sum().item())
            total_valid_tokens += len(token_losses)

        # If latent_ids provided, group by latent and average within each latent
        if latent_ids is not None:
            # Group losses by latent ID
            latent_losses = {}
            for loss, latent_id in zip(all_losses, latent_ids):
                if latent_id not in latent_losses:
                    latent_losses[latent_id] = []
                latent_losses[latent_id].append(loss)

            # Average within each latent, then across latents
            latent_averaged_losses = [
                torch.stack(losses).mean() for losses in latent_losses.values()
            ]
            final_loss = torch.stack(latent_averaged_losses).mean()
        else:
            # Simple average across all sequences
            final_loss = torch.stack(all_losses).mean()

        # Calculate statistics
        with torch.no_grad():
            token_norms = torch.norm(batch_soft_tokens, dim=-1)
            norm_stats = {
                "mean": token_norms.mean().item(),
                "max": token_norms.max().item(),
                "num_clamped_tokens": total_clamped_tokens,
                "total_valid_tokens": total_valid_tokens,
                "batch_size": batch_size,
                "max_sequence_length": max_seq_len,
                "unique_latents": len(set(latent_ids)) if latent_ids else batch_size,
            }

            if all_token_losses:
                sorted_losses = sorted(all_token_losses, reverse=True)
                norm_stats["top_10_token_losses"] = sorted_losses[:10]

        return final_loss, norm_stats
