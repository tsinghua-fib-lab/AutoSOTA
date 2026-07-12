# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dataset loading utilities for PD scheduler experiments.

Functions for loading prompts from various datasets (Alpaca, ShareGPT, LMSYS)
and applying chat templates.
"""

import json
import os
import random
import warnings

import numpy as np

# Suppress tokenizer length warnings
warnings.filterwarnings(
    "ignore", message="Token indices sequence length is longer than"
)

# HuggingFace datasets
try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Constants
WORD_TO_TOKEN_RATIO = 1.3
DEFAULT_SHAREGPT_PATH = "./ShareGPT_V3_unfiltered_cleaned_split.json"


def parse_alpaca_text(text: str) -> tuple[str, str]:
    """Parse Alpaca text format to extract input and output."""
    if "### Response:" in text:
        parts = text.split("### Response:", 1)
        input_text = parts[0].strip()
        output_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        input_text = text
        output_text = ""
    return input_text, output_text


def load_alpaca_prompts(
    max_samples: int = 1000, tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from Alpaca dataset.

    Args:
        max_samples: Maximum number of samples to load.
        tokenizer: Optional tokenizer for accurate token counting.

    Returns:
        Tuple of (prompts, input_lengths, output_lengths).
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading tatsu-lab/alpaca from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        text = item.get("text", "")
        if text:
            input_text, output_text = parse_alpaca_text(text)

            if input_text and output_text:
                prompts.append(input_text)

                if use_tokenizer:
                    input_len = len(tokenizer.encode(input_text))
                    output_len = len(tokenizer.encode(output_text))
                else:
                    input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                    output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

                input_lengths.append(input_len)
                output_lengths.append(output_len)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
    print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_sharegpt_prompts(
    json_path: str = DEFAULT_SHAREGPT_PATH,
    max_samples: int = 1000,
    tokenizer=None,
    output_len_cap: int | None = None,
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from ShareGPT dataset.

    Args:
        json_path: Path to ShareGPT JSON file.
        max_samples: Maximum number of samples to load.
        tokenizer: Optional tokenizer for accurate token counting.
        output_len_cap: Per-prompt output length cap (truncates dataset-native
            output_len at this value). Paper §4.1 / REPRODUCE.md ShareGPT
            protocol specifies cap ≤500. Default None = no cap.

    Returns:
        Tuple of (prompts, input_lengths, output_lengths).
    """
    print(f"Loading ShareGPT from {json_path}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ShareGPT file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    sample_count = 0

    for item in data:
        if sample_count >= max_samples:
            break

        conversations = item.get("conversations", [])
        if not conversations:
            continue

        i = 0
        while i < len(conversations) and sample_count < max_samples:
            turn = conversations[i]

            if turn.get("from") == "human":
                input_text = turn.get("value", "").strip()
                output_text = ""
                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    if next_turn.get("from") == "gpt":
                        output_text = next_turn.get("value", "").strip()
                        i += 1

                if input_text and output_text:
                    prompts.append(input_text)

                    if use_tokenizer:
                        input_len = len(tokenizer.encode(input_text))
                        output_len = len(tokenizer.encode(output_text))
                    else:
                        input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                        output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

                    if output_len_cap is not None:
                        output_len = min(output_len, output_len_cap)

                    input_lengths.append(input_len)
                    output_lengths.append(output_len)
                    sample_count += 1

            i += 1

    print(f"Loaded {len(prompts)} prompts from ShareGPT")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_lmsys_prompts(
    max_samples: int = 1000, tokenizer=None
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from LMSYS-Chat-1M dataset.

    Args:
        max_samples: Maximum number of samples to load.
        tokenizer: Optional tokenizer for accurate token counting.

    Returns:
        Tuple of (prompts, input_lengths, output_lengths).
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading lmsys/lmsys-chat-1m from HuggingFace...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    sample_count = 0

    for item in dataset:
        if sample_count >= max_samples:
            break

        conversation = item.get("conversation", [])
        if not conversation:
            continue

        input_text = ""
        output_text = ""

        for turn in conversation:
            role = turn.get("role", "")
            content = turn.get("content", "").strip()

            if role == "user" and not input_text:
                input_text = content
            elif role == "assistant" and input_text and not output_text:
                output_text = content
                break

        if input_text and output_text:
            prompts.append(input_text)

            if use_tokenizer:
                input_len = len(tokenizer.encode(input_text))
                output_len = len(tokenizer.encode(output_text))
            else:
                input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                output_len = int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)

            input_lengths.append(input_len)
            output_lengths.append(output_len)
            sample_count += 1

    print(f"Loaded {len(prompts)} prompts from LMSYS-Chat-1M")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_longbench_prompts(
    max_samples: int = 1000,
    tokenizer=None,
    min_input_len: int = 1000,
    max_input_len: int = 4000,
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from THUDM/LongBench-v2 dataset.

    This dataset contains long-context tasks, ideal for testing prefill-heavy
    scenarios. Long contexts are split into chunks of [min_input_len, max_input_len]
    tokens, each chunk paired with the same question.

    Args:
        max_samples: Maximum number of samples to load.
        tokenizer: Optional tokenizer for accurate token counting.
        min_input_len: Minimum input length in tokens.
        max_input_len: Maximum input length in tokens.

    Returns:
        Tuple of (prompts, input_lengths, output_lengths).
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading THUDM/LongBench-v2 from HuggingFace...")
    print(f"  Input length range: [{min_input_len}, {max_input_len}] tokens")
    print("  Long contexts will be chunked with random sizes in this range")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    total_chunks = 0
    original_samples = 0

    for item in dataset:
        if len(prompts) >= max_samples:
            break

        context = item.get("context", "")
        question = item.get("question", "")
        choice_a = item.get("choice_A", "")
        choice_b = item.get("choice_B", "")
        choice_c = item.get("choice_C", "")
        choice_d = item.get("choice_D", "")

        if not context or not question:
            continue

        original_samples += 1

        # Build the question suffix (fixed part)
        choices_text = ""
        if choice_a or choice_b or choice_c or choice_d:
            choices_text = (
                f"\n\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}"
            )
        question_suffix = f"\n\nQuestion: {question}{choices_text}\n\nAnswer:"

        # Calculate suffix length
        if use_tokenizer:
            suffix_len = len(tokenizer.encode(question_suffix))
        else:
            suffix_len = int(len(question_suffix.split()) * WORD_TO_TOKEN_RATIO)

        # Minimum context length needed
        min_context_len = min_input_len - suffix_len
        max_context_len = max_input_len - suffix_len

        if min_context_len < 100:
            continue

        # Tokenize context for chunking with random sizes
        if use_tokenizer:
            context_tokens = tokenizer.encode(context)
            context_len = len(context_tokens)

            # Split context into chunks with random sizes
            chunks = []
            pos = 0
            while pos < context_len:
                # Randomly pick chunk size for this iteration
                target_context_len = random.randint(min_context_len, max_context_len)
                end = min(pos + target_context_len, context_len)
                chunk_tokens = context_tokens[pos:end]

                # Only add if chunk is large enough
                if len(chunk_tokens) >= min_context_len:
                    chunk_text = tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)

                pos = end
        else:
            # Word-based chunking with random sizes
            words = context.split()
            min_words_per_chunk = int(min_context_len / WORD_TO_TOKEN_RATIO)
            max_words_per_chunk = int(max_context_len / WORD_TO_TOKEN_RATIO)

            chunks = []
            pos = 0
            while pos < len(words):
                # Randomly pick chunk size for this iteration
                words_per_chunk = random.randint(
                    min_words_per_chunk, max_words_per_chunk
                )
                end = min(pos + words_per_chunk, len(words))
                chunk_words = words[pos:end]

                # Only add if chunk is large enough
                if len(chunk_words) >= min_words_per_chunk:
                    chunk_text = " ".join(chunk_words)
                    chunks.append(chunk_text)

                pos = end

        # Create samples from chunks
        for chunk in chunks:
            if len(prompts) >= max_samples:
                break

            input_text = f"{chunk}{question_suffix}"

            # Calculate input length
            if use_tokenizer:
                input_len = len(tokenizer.encode(input_text))
            else:
                input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)

            # Filter by input length range
            if input_len < min_input_len or input_len > max_input_len:
                continue

            # Output is just the answer (A/B/C/D), very short
            output_len = 5

            prompts.append(input_text)
            input_lengths.append(input_len)
            output_lengths.append(output_len)
            total_chunks += 1

        # Progress indicator
        if len(prompts) % 1000 == 0 and len(prompts) > 0:
            print(
                f"  Loaded {len(prompts)} samples from "
                f"{original_samples} original contexts..."
            )

    print(f"Loaded {len(prompts)} prompts from LongBench-v2")
    print(f"  Original contexts: {original_samples}")
    print(f"  Total chunks created: {total_chunks}")
    if len(prompts) > 0:
        print(f"  Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"  Average output length: {np.mean(output_lengths):.1f} tokens")

    return prompts, input_lengths, output_lengths


def load_numina_math_prompts(
    max_samples: int = 1000,
    tokenizer=None,
    min_output_len: int = 0,
    output_len_cap: int | None = None,
    step_by_step: bool = True,
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from AI-MO/NuminaMath-CoT dataset.

    This dataset contains 850k math problems with detailed chain-of-thought
    solutions, making it ideal for testing decode-heavy scenarios.

    Args:
        max_samples: Maximum number of samples to load.
        tokenizer: Optional tokenizer for accurate token counting.
        min_output_len: Minimum output length in tokens (filter shorter samples).
        output_len_cap: Per-prompt output length cap (truncates dataset-native
            output_len at this value). Paper §4.1 protocol caps NuminaMath at
            4000 tokens. Default None = no cap.
        step_by_step: If True, append "Please think step by step." to prompts.

    Returns:
        Tuple of (prompts, input_lengths, output_lengths).
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading AI-MO/NuminaMath-CoT from HuggingFace...")
    print(f"  min_output_len filter: {min_output_len} tokens")
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    suffix = "\n\nPlease think step by step." if step_by_step else ""

    filtered_count = 0
    processed_count = 0

    for item in dataset:
        if len(prompts) >= max_samples:
            break

        processed_count += 1
        messages = item.get("messages", [])

        # Extract user question and assistant response
        user_content = None
        assistant_content = None

        for msg in messages:
            if msg.get("role") == "user" and user_content is None:
                user_content = msg.get("content", "").strip()
            elif msg.get("role") == "assistant" and assistant_content is None:
                assistant_content = msg.get("content", "").strip()

        if not user_content or not assistant_content:
            continue

        # Calculate lengths
        input_text = user_content + suffix

        if use_tokenizer:
            input_len = len(tokenizer.encode(input_text))
            output_len = len(tokenizer.encode(assistant_content))
        else:
            input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
            output_len = int(len(assistant_content.split()) * WORD_TO_TOKEN_RATIO)

        # Filter by minimum output length
        if output_len < min_output_len:
            filtered_count += 1
            continue

        if output_len_cap is not None:
            output_len = min(output_len, output_len_cap)

        prompts.append(input_text)
        input_lengths.append(input_len)
        output_lengths.append(output_len)

        # Progress indicator
        if len(prompts) % 1000 == 0:
            print(
                f"  Loaded {len(prompts)} samples "
                f"(processed {processed_count}, filtered {filtered_count})"
            )

    print(f"Loaded {len(prompts)} prompts from NuminaMath-CoT")
    print(
        f"  Processed: {processed_count}, "
        f"Filtered (output < {min_output_len}): {filtered_count}"
    )
    if len(prompts) > 0:
        print(f"  Average input length: {np.mean(input_lengths):.1f} tokens")
        print(f"  Average output length: {np.mean(output_lengths):.1f} tokens")
        print(f"  Step-by-step mode: {step_by_step}")

    return prompts, input_lengths, output_lengths


def load_processbench_prompts(
    max_samples: int = 1000,
    tokenizer=None,
    split: str = "gsm8k",
    step_by_step: bool = True,
) -> tuple[list[str], list[int], list[int]]:
    """
    Load prompts from Qwen/ProcessBench dataset.

    This dataset is useful for testing decode-heavy scenarios because
    adding "Please think step by step" triggers long chain-of-thought outputs.

    Args:
        max_samples: Maximum number of samples to load.
        tokenizer: Optional tokenizer for accurate token counting.
        split: Dataset split to use (gsm8k, math, etc.).
        step_by_step: If True, append "Please think step by step." to prompts
                      for decode-heavy testing.

    Returns:
        Tuple of (prompts, input_lengths, output_lengths).
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading Qwen/ProcessBench (split={split}) from HuggingFace...")
    dataset = load_dataset("Qwen/ProcessBench", split=split)

    prompts = []
    input_lengths = []
    output_lengths = []

    use_tokenizer = tokenizer is not None
    suffix = "\n\nPlease think step by step." if step_by_step else ""

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        problem = item.get("problem", "")
        steps = item.get("steps", [])

        if problem:
            # Build the prompt
            input_text = problem + suffix

            # Estimate output from the steps (chain of thought)
            output_text = "\n".join(steps) if steps else ""

            prompts.append(input_text)

            if use_tokenizer:
                input_len = len(tokenizer.encode(input_text))
                # Output length is estimated from steps
                output_len = len(tokenizer.encode(output_text)) if output_text else 256
            else:
                input_len = int(len(input_text.split()) * WORD_TO_TOKEN_RATIO)
                output_len = (
                    int(len(output_text.split()) * WORD_TO_TOKEN_RATIO)
                    if output_text
                    else 256
                )

            input_lengths.append(input_len)
            output_lengths.append(output_len)

    print(f"Loaded {len(prompts)} prompts from ProcessBench ({split})")
    if len(prompts) > 0:
        print(f"Average input length: {np.mean(input_lengths):.1f} tokens")
        print(
            f"Average output length: {np.mean(output_lengths):.1f} tokens (estimated)"
        )
        print(f"Step-by-step mode: {step_by_step}")

    return prompts, input_lengths, output_lengths


def load_wildchat_conversations(
    max_conversations: int = 500,
    tokenizer=None,
    min_turns: int = 8,
) -> list[dict]:
    """
    Load multi-turn conversations from allenai/WildChat-1M dataset.

    This dataset contains real multi-turn conversations, ideal for testing
    prefix cache effectiveness. Each conversation with n turns generates
    n requests, where each subsequent request shares a prefix with previous ones.

    Args:
        max_conversations: Maximum number of conversations to load.
        tokenizer: Optional tokenizer for token counting.
        min_turns: Minimum number of turns required (filter shorter conversations).

    Returns:
        List of conversations in format:
        [{"id": "conv_id", "messages": [{"role": "user/assistant", "content": "..."}]}]

    Note:
        This format is compatible with vLLM's multi-turn benchmark:
        python benchmarks/multi_turn/benchmark_serving_multi_turn_threaded.py
    """
    if not HF_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading allenai/WildChat-1M from HuggingFace...")
    print(f"  min_turns filter: {min_turns}")
    # Use streaming to avoid loading entire dataset
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    conversations = []
    processed_count = 0
    filtered_count = 0

    for item in dataset:
        if len(conversations) >= max_conversations:
            break

        processed_count += 1
        conv_data = item.get("conversation", [])
        turn_count = item.get("turn", 0)

        # Filter by minimum turns
        if turn_count < min_turns:
            filtered_count += 1
            continue

        # Extract messages
        messages = []
        for msg in conv_data:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()

            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Validate conversation structure
        if len(messages) < min_turns * 2 - 1:  # At least min_turns user messages
            filtered_count += 1
            continue

        # Ensure conversation starts with user
        if messages and messages[0]["role"] != "user":
            messages = messages[1:]

        # Ensure alternating user/assistant pattern
        valid_messages = []
        expected_role = "user"
        for msg in messages:
            if msg["role"] == expected_role:
                valid_messages.append(msg)
                expected_role = "assistant" if expected_role == "user" else "user"

        # Need at least min_turns complete rounds (user + assistant)
        if len(valid_messages) < min_turns * 2:
            filtered_count += 1
            continue

        conv_id = item.get("conversation_hash", f"conv_{len(conversations)}")
        conversations.append({"id": conv_id, "messages": valid_messages})

        # Progress indicator
        if len(conversations) % 100 == 0:
            print(
                f"  Loaded {len(conversations)} conversations "
                f"(processed {processed_count}, filtered {filtered_count})"
            )

    print(f"Loaded {len(conversations)} conversations from WildChat-1M")
    print(
        f"  Processed: {processed_count}, "
        f"Filtered (turns < {min_turns}): {filtered_count}"
    )

    if conversations:
        turn_counts = [len(c["messages"]) for c in conversations]
        print(f"  Average turns per conversation: {np.mean(turn_counts):.1f}")
        print(f"  Turn range: [{np.min(turn_counts)}, {np.max(turn_counts)}]")

        # Calculate token statistics if tokenizer provided
        if tokenizer:
            total_tokens = []
            for conv in conversations[:100]:  # Sample first 100
                tokens = sum(
                    len(tokenizer.encode(m["content"])) for m in conv["messages"]
                )
                total_tokens.append(tokens)
            print(
                f"  Avg tokens per conversation (sample): {np.mean(total_tokens):.1f}"
            )

    return conversations


def apply_chat_template(
    prompts: list[str], tokenizer, enable_thinking: bool = True
) -> list[str]:
    """
    Apply chat template to prompts.

    Args:
        prompts: List of raw prompt strings.
        tokenizer: Tokenizer with apply_chat_template method.
        enable_thinking: Whether to enable thinking mode (for Qwen3 models).

    Returns:
        List of formatted prompts with chat template applied.
    """
    print(f"Applying chat template (enable_thinking={enable_thinking})...")
    formatted_prompts = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        formatted_prompts.append(text)

    print(f"Applied chat template to {len(formatted_prompts)} prompts")
    return formatted_prompts
