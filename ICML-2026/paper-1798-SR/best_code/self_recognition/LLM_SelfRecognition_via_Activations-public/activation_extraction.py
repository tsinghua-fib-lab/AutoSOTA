"""Activation extraction utilities for transformer models."""

from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm


class ActivationExtractor:
    """Extract activations from specific layers of the transformer model."""

    def __init__(self, model, layer_idx: Optional[List[int]], include_prompt_tokens: bool = False):
        self.model = model
        self.include_prompt_tokens = include_prompt_tokens

        if hasattr(self.model, "transformer"):
            self.layers = self.model.transformer.h
        elif hasattr(self.model, "model"):
            if hasattr(model.model, "layers"):
                self.layers = self.model.model.layers
            elif hasattr(model.model, "language_model"):
                self.layers = self.model.model.language_model.layers
        else:
            raise ValueError("Could not find transformer, language_model, or model attribute.")

        resolved_layer_indices = list(range(len(self.layers))) if layer_idx is None else list(layer_idx)
        if not resolved_layer_indices:
            raise ValueError("layer_idx cannot be empty. Must specify at least one layer index.")

        self.layer_indices = resolved_layer_indices

        self.activations = {}
        self.hooks = []

        for idx in self.layer_indices:
            hook = self.layers[idx].register_forward_hook(
                lambda module, inputs, output, layer_idx=idx: self._hook_fn(module, inputs, output, layer_idx)
            )
            self.hooks.append(hook)

    def _hook_fn(self, module, inputs, output, layer_idx):
        if isinstance(output, tuple):
            layer_output = output[0]
        else:
            layer_output = output

        self.activations[layer_idx] = layer_output.detach().cpu()

    def extract_activations(
        self,
        texts: List[str],
        prompts: List[str] | None,
        tokenizer,
        device,
        token_indices: Optional[List[int]] = None,
        mode: str = "some",
        compute_perplexity: bool = False,
        max_input_tokens: int | None = None,
        prompt_indices: List[int] | None = None,
    ) -> pd.DataFrame:
        """Extract activations for a list of texts."""
        df_rows = []

        if mode not in {"some", "mean", "last_period"}:
            raise ValueError(f"Invalid mode: {mode}. Expected one of 'some', 'mean', 'last_period'.")

        if token_indices is None:
            token_indices = [-1]

        if mode == "last_period":
            candidates = [".", " ."]
            period_token_sequences = []
            for cand in candidates:
                ids = tokenizer.encode(cand, add_special_tokens=False)
                if len(ids) > 0:
                    period_token_sequences.append(ids)

        if prompts is None:
            prompts = [""] * len(texts)

        if prompt_indices is None:
            prompt_indices = list(range(len(texts)))

        prompt_lengths = []
        for prompt in prompts:
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", padding=False, truncation=True, add_special_tokens=False
            )["input_ids"].to(device)
            prompt_lengths.append(int(prompt_ids.shape[1]))

        for text, prompt, prompt_len, p_idx in tqdm(
            zip(texts, prompts, prompt_lengths, prompt_indices),
            total=len(texts),
            desc="Extracting activations",
        ):
            full_text = f"{prompt} {text}".strip()

            original_trunc_side = getattr(tokenizer, "truncation_side", None)
            if original_trunc_side is not None:
                tokenizer.truncation_side = "left"
            tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
            if max_input_tokens is not None:
                tokenizer_kwargs["max_length"] = max_input_tokens
            inputs = tokenizer(full_text, **tokenizer_kwargs)
            if original_trunc_side is not None:
                tokenizer.truncation_side = original_trunc_side
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            seq_length = inputs["attention_mask"].sum(dim=1).cpu().item()

            completion_ids = tokenizer.encode(" " + text, add_special_tokens=False)
            comp_len = min(len(completion_ids), seq_length)
            prompt_len = max(0, seq_length - comp_len)

            if compute_perplexity:
                logits = outputs.logits
                shift_logits = logits[:, :-1, :]
                shift_labels = inputs["input_ids"][:, 1:]
                shift_mask = inputs["attention_mask"][:, 1:]

                positions = torch.arange(shift_labels.shape[1], device=device)
                completion_pos_mask = (positions >= prompt_len).unsqueeze(0).expand_as(shift_labels)

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                nll = torch.nn.functional.nll_loss(
                    log_probs.reshape(-1, log_probs.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).reshape(shift_labels.shape)

                effective_mask = (shift_mask.bool() & completion_pos_mask).to(nll.dtype)
                nll = nll * effective_mask
                token_count = int(effective_mask.sum().item())
                if token_count > 0:
                    mean_nll = nll.sum() / token_count
                    perplexity_value = float(torch.exp(mean_nll).cpu().item())
                else:
                    perplexity_value = float("nan")
            else:
                perplexity_value = None

            if mode == "last_period":
                input_ids_seq = inputs["input_ids"][0, :seq_length].tolist()
                prompt_len_for_period = prompt_len

                last_match_end = -1
                for seq in period_token_sequences:
                    L = len(seq)
                    for i in range(prompt_len_for_period, len(input_ids_seq) - L + 1):
                        if input_ids_seq[i:i + L] == seq:
                            last_match_end = i + L - 1

                if last_match_end == -1:
                    continue

                last_period_idx = last_match_end

            for layer_idx in self.layer_indices:
                layer_tensor = self.activations[layer_idx][0, :seq_length, :]
                if layer_idx < 0:
                    actual_layer_idx = len(self.layers) + layer_idx
                else:
                    actual_layer_idx = layer_idx

                if mode == "mean":
                    if self.include_prompt_tokens:
                        mean_activation = layer_tensor.mean(dim=0)
                    else:
                        completion_tensor = layer_tensor[prompt_len:, :]
                        if completion_tensor.shape[0] > 0:
                            mean_activation = completion_tensor.mean(dim=0)
                        else:
                            mean_activation = layer_tensor[-1, :]

                    row = {
                        "text": full_text,
                        "layer_idx": actual_layer_idx,
                        "token_idx": seq_length - 1,
                        "activations": mean_activation.float().numpy(),
                        "prompt_idx": p_idx,
                    }
                    if compute_perplexity:
                        row["perplexity"] = perplexity_value
                    df_rows.append(row)
                elif mode == "last_period":
                    activation = layer_tensor[last_period_idx, :].float().numpy()
                    row = {
                        "text": full_text,
                        "layer_idx": actual_layer_idx,
                        "token_idx": last_period_idx,
                        "activations": activation,
                        "prompt_idx": p_idx,
                    }
                    if compute_perplexity:
                        row["perplexity"] = perplexity_value
                    df_rows.append(row)
                else:  # mode == "some"
                    for token_idx in token_indices:
                        if token_idx < 0:
                            actual_idx = seq_length + token_idx
                        else:
                            actual_idx = token_idx

                        if actual_idx < 0 or actual_idx >= seq_length:
                            continue

                        activation = layer_tensor[actual_idx, :].float().numpy()

                        row = {
                            "text": full_text,
                            "layer_idx": actual_layer_idx,
                            "token_idx": actual_idx,
                            "activations": activation,
                            "prompt_idx": p_idx,
                        }
                        if compute_perplexity:
                            row["perplexity"] = perplexity_value
                        df_rows.append(row)
        return pd.DataFrame(df_rows)

    def cleanup(self):
        """Remove all forward hooks."""
        for hook in self.hooks:
            hook.remove()


def extract_activations_for_texts(
    texts: dict,
    config: dict,
    model,
    tokenizer,
    device,
) -> tuple:
    """Extract activations for both LLM and human completions."""
    print("\n>>> Extracting activations from LLM")
    extractor = ActivationExtractor(model, layer_idx=config["LAYERS_TO_ANALYZE"])
    config["LAYERS_TO_ANALYZE"] = list(extractor.layer_indices)
    print(f"Extracting activations from layers {config['LAYERS_TO_ANALYZE']}")

    activation_prompts = texts["prompts"] if config["USE_PROMPT_FOR_ACTIVATIONS"] else None

    prompt_indices = list(range(len(texts["llm_completions"])))

    print("Extracting activations for AI-generated texts...")
    ai_activations = extractor.extract_activations(
        texts["llm_completions"], activation_prompts, tokenizer, device,
        token_indices=config["TOKENS_TO_ANALYZE"],
        mode=config["EXTRACTION_MODE"],
        compute_perplexity=config["COMPUTE_PERPLEXITY"],
        max_input_tokens=config.get("MAX_INPUT_TOKENS", None),
        prompt_indices=prompt_indices,
    )

    print("Extracting activations for human-authored texts...")
    human_activations = extractor.extract_activations(
        texts["human_completions"], activation_prompts, tokenizer, device,
        token_indices=config["TOKENS_TO_ANALYZE"],
        mode=config["EXTRACTION_MODE"],
        compute_perplexity=config["COMPUTE_PERPLEXITY"],
        max_input_tokens=config.get("MAX_INPUT_TOKENS", None),
        prompt_indices=prompt_indices,
    )

    extractor.cleanup()
    print("Activation extraction complete!")

    ai_activations["source_type"] = "llm"
    ai_activations["label"] = 0
    human_activations["source_type"] = "human"
    human_activations["label"] = 1

    return ai_activations, human_activations
