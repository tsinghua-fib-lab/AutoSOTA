"""Inference runner."""

import torch
from typing import List, Optional

from models.qwen import AttentionPatch


def run_inference(
    model,
    processor,
    samples: List,
    dataset,
    max_new_tokens: int = 128,
    attention_patch: Optional[AttentionPatch] = None,
):
    """
    Run inference on a batch of samples.

    Args:
        model: The VLM model
        processor: The processor
        samples: List of dataset samples
        dataset: Dataset object (for building messages)
        max_new_tokens: Max tokens to generate
        attention_patch: Optional AttentionPatch for extracting attention

    Returns:
        predictions: List of prediction strings
        attention: Extracted attention tensor (if attention_patch provided)
        attention_info: Attention metadata (if attention_patch provided)
    """
    messages_batch = [dataset.build_messages(s) for s in samples]

    inputs = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=len(samples) > 1,
    )
    inputs = inputs.to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Generation kwargs
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if attention_patch is not None:
        gen_kwargs["output_attentions"] = True
        gen_kwargs["return_dict_in_generate"] = True

    outputs = model.generate(**inputs, **gen_kwargs)

    # Extract attention if patch provided
    attention, attention_info = None, None
    if attention_patch is not None:
        attention = attention_patch.extract(outputs, inputs.input_ids, input_len)
        attention_info = attention_patch.attention_info
        generated_ids = outputs.sequences
    else:
        generated_ids = outputs

    # Decode predictions
    predictions = processor.batch_decode(
        [ids[input_len:] for ids in generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return predictions, attention, attention_info
