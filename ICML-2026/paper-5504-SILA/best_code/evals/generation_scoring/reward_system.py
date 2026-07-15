#!/usr/bin/env python3
"""
Reward System for SAE Reflective Coherence Training.
Computes rewards based on target SAE latent activation strength.
"""

import gc
import re
import traceback
from typing import Any, List, Optional, Union

import torch
from config import ModelConfig, RewardConfig
from nnsight import LanguageModel
from sae_core import load_sae


class SAERewardSystem:
    """
    Reward system that computes rewards based on target SAE latent activation strength.

    The workflow is:
    1. Take a natural language label and target latent index
    2. Substitute label into prompt template
    3. Generate text using the model
    4. Measure SAE activations during generation using nnsight hooks
    5. Extract activation strength of target latent
    6. Return activation strength as reward
    """

    def __init__(
        self,
        model_config: ModelConfig,
        reward_config: Optional[RewardConfig] = None,
        existing_model: Any = None,
        existing_tokenizer: Any = None,
    ):
        self.model_config = model_config
        self.reward_config = reward_config or RewardConfig()

        # Require existing model and tokenizer (no fallback loading)
        if existing_model is None or existing_tokenizer is None:
            raise ValueError(
                "existing_model and existing_tokenizer are required parameters"
            )

        self.existing_model = existing_model
        self.existing_tokenizer = existing_tokenizer

        # These will be initialized in setup()
        self.model = None
        self.tokenizer = None
        self.sae = None
        self.hf_model = None  # Store reference to original HF model for generation
    
    def _get_tensor_device(self):
        """Get the device string for tensor operations. Returns 'cuda' for 'auto' multi-GPU setups."""
        return "cuda" if self.model_config.device == "auto" else self.model_config.device

    def setup(self):
        """Initialize model, tokenizer, and SAE. Call this before using the reward system."""
        print("✓ Using existing model and tokenizer")

        # Store reference to original HF model for generation
        self.hf_model = self.existing_model

        # Create nnsight wrapper around the existing transformers model
        self.model = LanguageModel(
            self.existing_model,  # Use the existing transformers model directly
            device_map=self.model_config.device,
            dtype=getattr(torch, self.model_config.dtype),
        )
        self.tokenizer = self.existing_tokenizer

        # Ensure nnsight model has access to the tokenizer
        # This is critical for generation to work properly
        self.model.tokenizer = self.existing_tokenizer

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load SAE using SAELens library
        try:
            print(f"Loading SAE: {self.model_config.sae_release}/{self.model_config.sae_id}")
            self.sae = self._load_sae_from_huggingface()
            print("✓ SAE loaded successfully")
        except Exception as e:
            print(f"Error: Could not load SAE: {e}")
            raise e

    def _load_sae_from_huggingface(self):
        """Load SAE using SAELens library."""
        # Determine SAE device: if multi-GPU (device="auto"), use first GPU
        sae_device = "cuda:0" if self.model_config.device == "auto" else self.model_config.device
        
        # Use SAELens to load the SAE
        sae = load_sae(
            release=self.model_config.sae_release,
            sae_id=self.model_config.sae_id,
            device=sae_device,
        )

        return sae

    # Note: The original non-batched compute_reward method has been removed.
    # Use compute_batch_rewards() for all reward computation.

    def compute_batch_rewards(
        self,
        labels: List[str],
        target_latent_indices: Union[List[int], int],
        batch_size: Optional[int] = None,
        return_detailed_activations: bool = False,
    ) -> List[Union[float, dict]]:
        """
        Compute rewards for a batch of labels in parallel for efficient training.

        Args:
            labels: List of natural language labels
            target_latent_indices: List of target latent indices (one per label)
                                 or single index to use for all labels
            batch_size: Optional batch size for processing (uses len(labels) if None)
            return_detailed_activations: If True, return detailed per-token activation info

        Returns:
            If return_detailed_activations=False: List of rewards (one per label)
            If return_detailed_activations=True: List of dicts with reward + per-token activations
        """
        if isinstance(target_latent_indices, int):
            target_latent_indices = [target_latent_indices] * len(labels)

        if len(labels) != len(target_latent_indices):
            raise ValueError(
                "Number of labels must match number of target latent indices"
            )

        if batch_size is None:
            batch_size = len(labels)

        # Process in chunks if batch is too large
        all_rewards: List[Union[float, dict]] = []
        for i in range(0, len(labels), batch_size):
            chunk_labels = labels[i : i + batch_size]
            chunk_indices = target_latent_indices[i : i + batch_size]

            print(
                f"Processing batch chunk {i // batch_size + 1} with {len(chunk_labels)} items"
            )
            chunk_rewards = self._compute_batch_chunk_parallel(
                chunk_labels, chunk_indices, return_detailed_activations
            )
            all_rewards.extend(chunk_rewards)

        return all_rewards

    def _compute_batch_chunk_parallel(
        self,
        labels: List[str],
        target_latent_indices: List[int],
        return_detailed_activations: bool = False,
    ) -> List[Union[float, dict]]:
        """
        Compute rewards for a single batch chunk in parallel.

        This is the core parallel processing method.
        """
        if self.model is None:
            raise RuntimeError("Must call setup() before computing rewards")

        batch_size = len(labels)
        print(f"Processing {batch_size} labels in parallel")

        # Step 1: Prepare batch prompts
        batch_prompts = []
        batch_system_messages = []

        for label in labels:
            # Use conversation mode (only supported mode)
            prompt = self.reward_config.conversation_prompt_template.replace("_", label)
            system_message = self.reward_config.conversation_system_message

            batch_prompts.append(prompt)
            batch_system_messages.append(system_message)

        # Step 2: Apply chat template and tokenize batch directly
        if self.reward_config.use_chat_template:
            # Build batch of message lists
            batch_messages = []
            for prompt, system_message in zip(batch_prompts, batch_system_messages):
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                batch_messages.append(messages)

            try:
                # Set padding side for generation (decoder-only models work better with left padding)
                self.tokenizer.padding_side = "left"

                # Apply chat template directly to batch with tokenization
                batch_inputs = self.tokenizer.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,  # Explicitly request attention mask
                )

                # Handle different return formats FIRST (before calling .items())
                if isinstance(batch_inputs, torch.Tensor):
                    # If only tensor returned, create attention mask manually
                    input_ids = batch_inputs
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                    batch_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                elif (
                    isinstance(batch_inputs, dict)
                    and "attention_mask" not in batch_inputs
                ):
                    # If dict but no attention mask, create one
                    attention_mask = (
                        batch_inputs["input_ids"] != self.tokenizer.pad_token_id
                    ).long()
                    batch_inputs["attention_mask"] = attention_mask

                # Now move to device (batch_inputs is guaranteed to be a dict)
                # Use _get_tensor_device() to handle device="auto" (returns "cuda" for multi-GPU)
                batch_inputs = {
                    k: v.to(self._get_tensor_device()) for k, v in batch_inputs.items()
                }

            except Exception as e:
                raise RuntimeError(
                    f"Batch chat template application failed: {e}"
                ) from e
        else:
            raise Exception("Use chat template formatting - plain text not supported")

        print(
            f"Batch input shape: {batch_inputs['input_ids'].shape}, attention mask: {batch_inputs['attention_mask'].shape}"
        )

        try:
            # Clear GPU cache before batch processing
            if (
                torch.cuda.is_available()
                and self.reward_config.aggressive_memory_cleanup
            ):
                torch.cuda.empty_cache()

            # Step 4: Batch generation
            with torch.no_grad():
                print(
                    f"Generating {self.reward_config.max_new_tokens} tokens for batch..."
                )

                # Use the stored HF model reference for generation (no NNsight tracing)
                batch_generated_ids = self.hf_model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs[
                        "attention_mask"
                    ],  # Pass attention mask to avoid warnings
                    max_new_tokens=self.reward_config.max_new_tokens,
                    temperature=self.reward_config.temperature,
                    do_sample=self.reward_config.do_sample,
                    top_p=self.reward_config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                print("Done generating reward tokens")

            # Step 5: Extract new tokens for each item in batch
            # With left padding, we need the full original length (including padding)
            # to correctly extract only the newly generated tokens
            original_input_length = batch_inputs["input_ids"].shape[1]
            batch_new_tokens = []
            batch_generated_texts = []

            for i in range(batch_size):
                generated_sequence = batch_generated_ids[i]
                new_tokens = generated_sequence[original_input_length:]

                # Decode for display/processing
                generated_text = self.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                batch_new_tokens.append(new_tokens)
                batch_generated_texts.append(generated_text)

            # Count non-padding tokens for each sequence
            non_padding_lengths = []
            for tokens in batch_new_tokens:
                non_padding_count = (tokens != self.tokenizer.pad_token_id).sum().item()
                non_padding_lengths.append(non_padding_count)

            mean_length = sum(non_padding_lengths) / len(non_padding_lengths) if non_padding_lengths else 0
            min_length = min(non_padding_lengths) if non_padding_lengths else 0
            max_length = max(non_padding_lengths) if non_padding_lengths else 0
            print(
                f"Generated texts for batch (non-padding token lengths: mean={mean_length:.2f}, min={min_length}, max={max_length})"
            )

            # Log individual generations for debugging
            print(f"\n{'=' * 50}")
            if self.reward_config.full_debug_mode:
                print(
                    f"INDIVIDUAL GENERATIONS (Batch of {batch_size}) - FULL DEBUG MODE"
                )
                print(f"{'=' * 50}")
                # Show ALL completions when full_debug_mode is enabled
                num_to_log = len(labels)
                show_full_details = True
            else:
                print(f"INDIVIDUAL GENERATIONS (Batch of {batch_size})")
                print(f"{'=' * 50}")
                # Show limited completions when full_debug_mode is disabled
                num_to_log = min(len(labels), self.reward_config.num_debug_samples)
                show_full_details = False

            for i in range(num_to_log):
                label = labels[i]
                generated_text = batch_generated_texts[i]
                target_latent = target_latent_indices[i]
                print(
                    f"Generation {i + 1} (Label: '{label}' → Target Latent {target_latent}):"
                )
                print(f"  Generated: {repr(generated_text)}")
                if show_full_details:
                    print(f"  Generated (full text): {generated_text}")
                    print(f"  Generated (length): {len(generated_text)} chars")
                    print("")  # Add empty line for readability

            if not self.reward_config.full_debug_mode and len(labels) > num_to_log:
                print(
                    f"... and {len(labels) - num_to_log} more generations (set full_debug_mode=True to see all)"
                )

            # Step 6: Process conversations (only supported mode)
            batch_final_token_ids = self._process_batch_conversations(
                batch_generated_texts
            )

            print("Computing SAE activations")
            # Step 7: Batch SAE analysis
            batch_rewards = self._compute_batch_sae_activations(
                batch_final_token_ids,
                target_latent_indices,
                return_detailed_activations,
                batch_generated_texts=batch_generated_texts,
            )
            print("Done computing SAE activations")

            # Clean up batch generation results
            if hasattr(batch_generated_ids, "is_cuda") and batch_generated_ids.is_cuda:
                batch_generated_ids = batch_generated_ids.cpu()
            del batch_generated_ids, batch_inputs

            return batch_rewards

        except Exception as e:
            print(f"Error in batch reward computation: {e}")
            print(traceback.format_exc())
            raise RuntimeError(f"Batch reward computation failed: {e}") from e

        finally:
            # Always clean up GPU memory
            if (
                torch.cuda.is_available()
                and self.reward_config.aggressive_memory_cleanup
            ):
                torch.cuda.empty_cache()
                gc.collect()

    def _process_batch_conversations(
        self, batch_generated_texts: List[str]
    ) -> List[torch.Tensor]:
        """Process batch of generated conversations and return tokenized results."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        batch_final_token_ids = []

        for i, generated_text in enumerate(batch_generated_texts):
            try:
                # Parse meta-conversation (show details based on full_debug_mode setting)
                if self.reward_config.full_debug_mode:
                    print(
                        f"🔄 BATCH PROCESSING - Item {i + 1}/{len(batch_generated_texts)}"
                    )
                parsed_conversation = self._parse_meta_conversation(
                    generated_text, debug=self.reward_config.full_debug_mode
                )

                if not parsed_conversation:
                    raise ValueError("No valid conversation found")

                # Convert to chat template and tokenize directly
                final_inputs = self.tokenizer.apply_chat_template(
                    parsed_conversation,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )

                # Handle different return formats
                if isinstance(final_inputs, torch.Tensor):
                    final_inputs = {"input_ids": final_inputs}

                # Move to device (use _get_tensor_device() to handle device="auto")
                final_inputs = {
                    k: v.to(self._get_tensor_device()) for k, v in final_inputs.items()
                }

                batch_final_token_ids.append(final_inputs["input_ids"].squeeze(0))

            except Exception as e:
                print(f"❌ Conversation parsing failed for item {i}: {e}")
                print(f"   Generated text: {repr(generated_text)}")
                raise RuntimeError(
                    f"Failed to parse conversation for item {i}: {e}"
                ) from e

        return batch_final_token_ids

    def _compute_batch_sae_activations(
        self,
        batch_token_ids: List[torch.Tensor],
        target_latent_indices: List[int],
        return_detailed_activations: bool = False,
        batch_generated_texts: Optional[List[str]] = None,
    ) -> List[Union[float, dict]]:
        """
        Compute SAE activations for a batch of token sequences with full batching and memory optimization.

        This implementation follows the spec: NO loops over sequences or tokens.
        Everything is done in a single batched forward pass with proper padding and masking.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        if self.sae is None:
            raise RuntimeError("SAE not initialized")

        try:
            batch_size = len(batch_token_ids)
            if batch_size == 0:
                return []

            print(
                f"Computing SAE activations for {batch_size} sequences in fully batched mode"
            )

            # Step 1: Pad all sequences to the same length and create attention masks
            max_seq_len = max(len(seq) for seq in batch_token_ids)
            padded_sequences = []
            attention_masks = []

            for token_ids in batch_token_ids:
                # Add batch dimension if needed
                if token_ids.dim() == 1:
                    token_ids = token_ids.unsqueeze(0)

                seq_len = token_ids.shape[1]

                # Pad sequence to max length
                if seq_len < max_seq_len:
                    padding = torch.full(
                        (1, max_seq_len - seq_len),
                        self.tokenizer.pad_token_id,
                        device=token_ids.device,
                        dtype=token_ids.dtype,
                    )
                    padded_seq = torch.cat([token_ids, padding], dim=1)
                else:
                    padded_seq = token_ids

                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = torch.cat(
                    [
                        torch.ones(
                            1, seq_len, device=token_ids.device, dtype=torch.bool
                        ),
                        torch.zeros(
                            1,
                            max_seq_len - seq_len,
                            device=token_ids.device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )

                padded_sequences.append(padded_seq)
                attention_masks.append(attention_mask)

            # Stack into single batch tensor
            batch_input_ids = torch.cat(
                padded_sequences, dim=0
            )  # (batch_size, max_seq_len)
            batch_attention_mask = torch.cat(
                attention_masks, dim=0
            )  # (batch_size, max_seq_len)

            print(f"Batched input shape: {batch_input_ids.shape}")

            # Step 2: Single forward pass to get hidden states for entire batch
            with torch.no_grad():
                with self.model.trace(batch_input_ids, scan=False):
                    target_layer = self._get_target_layer()
                    batch_hidden_states = target_layer.output.save()

            # batch_hidden_states shape: (batch_size, max_seq_len, d_model)
            print(f"Hidden states shape: {batch_hidden_states.shape}")

            # Step 3: Single SAE forward pass for entire batch
            with torch.no_grad():
                # Ensure dtype and device match SAE (important for multi-GPU setups)
                batch_hidden_states = batch_hidden_states.to(
                    device=self.sae.W_enc.device,
                    dtype=self.sae.W_enc.dtype
                )

                # Compute SAE activations for entire batch at once
                batch_activations = self.sae.encode(batch_hidden_states)
                # Shape: (batch_size, max_seq_len, d_sae)

                print(f"SAE activations shape: {batch_activations.shape}")

                # Step 4: Extract target latent activations using fancy indexing
                device = batch_activations.device
                batch_indices = torch.arange(batch_size, device=device)
                target_indices_tensor = torch.tensor(
                    target_latent_indices, device=device
                )

                # Extract target activations: (batch_size, max_seq_len)
                target_activations = batch_activations[
                    batch_indices[:, None],
                    torch.arange(max_seq_len, device=device)[None, :],
                    target_indices_tensor[:, None],
                ]

                # Apply attention mask to zero out padded tokens
                target_activations = target_activations * batch_attention_mask.to(
                    target_activations.dtype
                )

                # Step 5: Compute per-sequence rewards and detailed results
                batch_results = []

                for i in range(batch_size):
                    # Get activations for this sequence
                    seq_activations = target_activations[i]  # (max_seq_len,)
                    seq_attention_mask = batch_attention_mask[i]  # (max_seq_len,)

                    # Extract only non-padded tokens
                    valid_activations = seq_activations[seq_attention_mask]

                    # Convert to list for JSON serialization
                    per_token_activations = valid_activations.cpu().tolist()

                    # Note: We intentionally do NOT compute a "reward" field here.
                    # The misleading "mean activation" metric was removed.
                    # Use compute_mean_max_hit_rate.py to compute the proper binary hit rate.

                    if return_detailed_activations:
                        result = {
                            "per_token_activations": per_token_activations,
                            "num_tokens": len(per_token_activations),
                            "target_latent_index": target_latent_indices[i],
                            "generated_text": batch_generated_texts[i]
                            if batch_generated_texts
                            else None,
                        }
                        batch_results.append(result)
                    else:
                        # For non-detailed mode, just return the activations
                        batch_results.append({"per_token_activations": per_token_activations})

                # Clean up batch tensors
                del batch_activations, target_activations, batch_hidden_states

        except Exception as e:
            print(f"❌ Batch SAE activation computation failed: {e}")
            print(traceback.format_exc())

            # Return error results for all items in batch
            batch_results = []
            for i in range(len(batch_token_ids)):
                if return_detailed_activations:
                    error_result = {
                        "error": str(e),
                        "reward": None,
                        "per_token_activations": None,
                        "num_tokens": 0,
                        "target_latent_index": target_latent_indices[i]
                        if i < len(target_latent_indices)
                        else -1,
                        "generated_text": batch_generated_texts[i]
                        if batch_generated_texts and i < len(batch_generated_texts)
                        else None,
                    }
                    batch_results.append(error_result)
                else:
                    batch_results.append(None)  # Use None to indicate error, never 0.0

        # Print summary statistics for successful computations only
        successful_results = [
            r
            for r in batch_results
            if (r is not None and (not isinstance(r, dict) or r.get("error") is None))
        ]

        print(f"Computed {len(successful_results)} successful activation results")

        errors = len(batch_results) - len(successful_results)
        if errors > 0:
            print(f"⚠️  {errors} sequences failed with errors")

        return batch_results

    def _get_target_layer(self):
        """Get the target layer for SAE hook based on model config."""
        # Simple: use layer number directly (no string parsing needed)
        layer_num = self.model_config.sae_layer_number
        return self.model.model.layers[layer_num]

    def _parse_meta_conversation(
        self, meta_text: str, debug: bool = True
    ) -> List[dict]:
        """
        Parse meta-conversation format into standard chat format, preserving original order.

        Converts: "[USER] Hello\n[ASSISTANT] Hi there!" or "[USER] Hello [ASSISTANT] Hi there!"
        To: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

        Preserves the exact order the model generated, including potential multiple
        consecutive messages from the same role.

        Uses flexible line-based parsing that treats [USER] and [ASSISTANT] tags as optional.
        """

        conversation: List[dict] = []

        # Split by lines and also by inline tags to handle both formats
        # First, split by newlines to handle line-based format
        lines = meta_text.split("\n")

        # Then process each line and also check for inline tags
        text_parts = []
        for line in lines:
            # Split by tags within the line
            parts = re.split(r"(\[(?:USER|ASSISTANT)\])", line)
            text_parts.extend(parts)

        current_role = None
        current_content: List[str] = []

        for part in text_parts:
            part = part.strip()
            if not part:
                continue

            # Check if this part is a role tag
            role_match = re.match(r"\[(USER|ASSISTANT)\]", part)
            if role_match:
                # Save previous message if we have content
                if current_role is not None and current_content:
                    content = " ".join(current_content).strip()
                    if content:
                        role = "user" if current_role == "USER" else "assistant"
                        conversation.append({"role": role, "content": content})

                # Start new message
                current_role = role_match.group(1)
                current_content = []
            else:
                # This is content text
                if current_role is not None:
                    # We have a role, add this content to it
                    current_content.append(part)
                else:
                    # No role specified yet, try to infer or default to user
                    # For lenient parsing, assume first content is from user if no role specified
                    if not conversation:
                        current_role = "USER"
                        current_content = [part]
                    else:
                        # Add to last role or default to alternating
                        last_role = conversation[-1]["role"]
                        current_role = "ASSISTANT" if last_role == "user" else "USER"
                        current_content = [part]

        # Don't forget the last message
        if current_role is not None and current_content:
            content = " ".join(current_content).strip()
            if content:
                role = "user" if current_role == "USER" else "assistant"
                conversation.append({"role": role, "content": content})

        # If we still have no conversation, fail
        if not conversation and meta_text.strip():
            print(f"WARNING: Failed to parse meta-conversation: {meta_text}")
            print("Returning original text as a single message")
            return [{"role": "assistant", "content": meta_text}]

        # Debug logging to verify parsing results
        if debug:
            print(f"   Raw input: {repr(meta_text)}")
            if hasattr(self, "reward_config") and self.reward_config.full_debug_mode:
                print(f"   Raw input (full text): {meta_text}")
                print(f"   Parsed {len(conversation)} messages (FULL DEBUG MODE):")
                # Show ALL messages when full_debug_mode is enabled
                num_msgs_to_show = len(conversation)
                show_full_content = True
            else:
                print(f"   Parsed {len(conversation)} messages:")
                # Show limited messages when full_debug_mode is disabled
                num_msgs_to_show = min(len(conversation), 4)  # Fallback limit
                show_full_content = False

            for i in range(num_msgs_to_show):
                msg = conversation[i]
                print(f"     [{i + 1}] {msg['role'].upper()}: {repr(msg['content'])}")
                if show_full_content:
                    print(
                        f"     [{i + 1}] {msg['role'].upper()} (full): {msg['content']}"
                    )

            if not show_full_content and len(conversation) > num_msgs_to_show:
                print(
                    f"     ... and {len(conversation) - num_msgs_to_show} more messages"
                )

            if show_full_content:
                print("   End of parsed conversation")
            print("")  # Add empty line for readability

        return conversation

    def clear_gpu_memory(self):
        """Manually clear GPU memory and force garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Get memory usage before cleanup
            memory_before = torch.cuda.memory_allocated() / 1024**3

            # Force garbage collection
            gc.collect()

            # Clear cache again after GC
            torch.cuda.empty_cache()

            # Report memory usage
            memory_after = torch.cuda.memory_allocated() / 1024**3
            print(
                f"GPU memory cleanup: {memory_before:.2f} GB -> {memory_after:.2f} GB"
            )
        else:
            print("No CUDA available, skipping GPU memory cleanup")
