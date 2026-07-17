"""Text generation utilities for creating AI completions."""

from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from data_loader import (
    create_prompts_and_completions,
    xlsum_system_prompt,
    xsum_system_prompt,
)


def supports_apply_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "apply_chat_template", None) is not None


def _to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return {k: v.to(device) for k, v in batch.items()}


def _build_tokenized_chat_inputs(
    batch_prompts: List[str],
    tokenizer,
    system_prompt: Optional[str],
    max_input_tokens: int | None,
):
    tokenized = []
    truncation_side = getattr(tokenizer, "truncation_side", "right")
    for prompt in batch_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, dict):
            ids = ids.get("input_ids", ids)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if max_input_tokens is not None and len(ids) > max_input_tokens:
            if truncation_side == "left":
                ids = ids[-max_input_tokens:]
            else:
                ids = ids[:max_input_tokens]
        tokenized.append(torch.tensor(ids, dtype=torch.long))

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    padding_side = getattr(tokenizer, "padding_side", "right")
    max_len = max((t.numel() for t in tokenized), default=0)
    input_ids = torch.full((len(tokenized), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(tokenized), max_len), dtype=torch.long)
    for i, ids in enumerate(tokenized):
        if max_len == 0:
            continue
        seq_len = ids.numel()
        if padding_side == "left":
            start = max_len - seq_len
            input_ids[i, start:] = ids
            attention_mask[i, start:] = 1
        else:
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "attention_mask": attention_mask}, pad_token_id


def generate_completions(
    prompts: List[str],
    model,
    tokenizer,
    device,
    temperature: float = 0.7,
    top_p: float = 1.0,
    min_new_tokens: int = 190,
    max_new_tokens: int = 210,
    gen_kwargs: Optional[dict] = None,
    batch_size: int = 8,
    use_chat_template: bool = True,
    max_input_tokens: int | None = None,
    use_kv_cache: bool = True,
    empty_cache_between_batches: bool = True,
    system_prompt: Optional[str] = None,
) -> tuple[list[str], list[torch.Tensor]]:
    """Generate text completions for given prompts."""
    completions = []
    completions_tokenized = []
    has_chat_template = supports_apply_chat_template(tokenizer)
    if use_chat_template:
        if has_chat_template:
            print("Using chat template for generation")
        else:
            print("Tokenizer has no chat template; using raw prompts")

    tokenizer_class = type(tokenizer).__name__
    model_name = str(getattr(model, "name_or_path", ""))
    tokenize_during_generation = (
        "ministral-3" in model_name.lower()
        or "MistralCommonBackend" in tokenizer_class
    )

    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Generating completions"):
        batch_prompts = prompts[start_idx:start_idx + batch_size]

        inputs = None
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        if use_chat_template and has_chat_template:
            if tokenize_during_generation:
                inputs, pad_token_id = _build_tokenized_chat_inputs(
                    batch_prompts=batch_prompts,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    max_input_tokens=max_input_tokens,
                )
            else:
                batch_texts = []
                for p in batch_prompts:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": p})
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    batch_texts.append(rendered)
        else:
            batch_texts = batch_prompts

        if inputs is None:
            tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
            if max_input_tokens is not None:
                tokenizer_kwargs["max_length"] = max_input_tokens
            inputs = tokenizer(batch_texts, **tokenizer_kwargs)

        inputs = _to_device(inputs, device)

        with torch.inference_mode():
            effective_gen_kwargs = dict(gen_kwargs or {})
            effective_gen_kwargs.setdefault("return_dict_in_generate", False)
            effective_gen_kwargs.setdefault("output_scores", False)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                use_cache=use_kv_cache,
                **effective_gen_kwargs
            )

        padding_side = getattr(tokenizer, "padding_side", "right")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).tolist()
        else:
            input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

        outputs = outputs.detach().cpu()

        for i in range(outputs.shape[0]):
            if padding_side == "left":
                start_index = inputs["input_ids"].shape[1]
            else:
                start_index = int(input_lengths[i])

            new_tokens = outputs[i, start_index:]

            if pad_token_id is not None:
                new_tokens = new_tokens[new_tokens != pad_token_id]

            completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            completions.append(completion)
            completions_tokenized.append(new_tokens)

        del inputs
        del attention_mask
        if empty_cache_between_batches and torch.cuda.is_available() and (str(device) == "cuda"):
            torch.cuda.empty_cache()

    return completions, completions_tokenized


def generate_texts(
    config: dict,
    model,
    tokenizer,
    device,
) -> Dict[str, List]:
    """Generate prompts and completions from the configured dataset."""
    print("\n>>> Loading dataset and creating prompt-completion pairs")
    prompts, human_completions, human_tokens = create_prompts_and_completions(
        tokenizer=tokenizer,
        dataset_name=config["DATASET_NAME"],
        subset=config.get("DATASET_SUBSET", None),
        num_samples=config["NUM_SAMPLES"],
        completion_length=config["TARGET_COMPLETION_TOKENS"],
        prompt_length=config["TARGET_PROMPT_TOKENS"],
        max_doc_length=config.get("MAX_DOC_LENGTH", None),
        id_prefix=config.get("ID_PREFIX_FILTER", None),
    )

    print("\n>>> Generating AI completions")
    if config["DATASET_NAME"] == "c4":
        min_new_tokens = config["TARGET_COMPLETION_TOKENS"] - config["TARGET_COMPLETION_TOKENS_DIFF"]
        max_new_tokens = config["TARGET_COMPLETION_TOKENS"] + config["TARGET_COMPLETION_TOKENS_DIFF"]
        sys_prompt = None
    elif config["DATASET_NAME"] == "xsum":
        min_new_tokens = 5
        max_new_tokens = 50
        sys_prompt = xsum_system_prompt() if config["USE_CHAT_TEMPLATE"] else None
    elif config["DATASET_NAME"] == "xl-sum":
        min_new_tokens = 10
        max_new_tokens = 60
        sys_prompt = xlsum_system_prompt() if config["USE_CHAT_TEMPLATE"] else None
    else:
        raise ValueError(f"Invalid dataset name: {config['DATASET_NAME']}")

    llm_completions, llm_tokens = generate_completions(
        prompts, model, tokenizer, device,
        temperature=config["TEMPERATURE"],
        top_p=config.get("TOP_P", 1.0),
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        batch_size=config["BATCH_SIZE"],
        max_input_tokens=config.get("MAX_INPUT_TOKENS", None),
        use_chat_template=config["USE_CHAT_TEMPLATE"],
        system_prompt=sys_prompt,
    )

    return {
        "prompts": prompts,
        "human_completions": human_completions,
        "llm_completions": llm_completions,
        "human_tokens": human_tokens,
        "llm_tokens": llm_tokens,
    }
