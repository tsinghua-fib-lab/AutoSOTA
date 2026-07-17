"""Dataset loading and prompt construction utilities."""

from datasets import load_dataset


def xsum_template(text: str, word_min: int = 10, word_max: int = 25) -> str:
    """User prompt for XSUM summarization."""
    return (
        f"Summarize the article in exactly one sentence ({word_min}–{word_max} words). "
        "No preface, labels, quotes, or line breaks.\n\n"
        "ARTICLE START\n"
        f"{text}\n"
        "ARTICLE END\n\n"
        "Answer: "
    )


def xsum_system_prompt(word_min: int = 10, word_max: int = 25) -> str:
    """System prompt for single-sentence summaries."""
    return (
        "You write a single-sentence summary of news articles.\n"
        "Rules:\n"
        "- Output exactly one sentence ending with a single period.\n"
        f"- {word_min}–{word_max} words.\n"
        "- No preface, labels, quotes, or line breaks.\n"
        "- Focus on the main event and key entities only."
    )


def xlsum_template(text: str, word_min: int = 10, word_max: int = 25) -> str:
    """User prompt for XL-SUM-style summarization."""
    return (
        f"Summarize the article in one sentence or two at most ({word_min}–{word_max} words).\n"
        "Prefer a single sentence if possible.\n"
        "No preface, labels, quotes, bullet points, or line breaks.\n\n"
        "ARTICLE START\n"
        f"{text}\n"
        "ARTICLE END\n\n"
        "Answer: "
    )


def xlsum_system_prompt() -> str:
    """System prompt for XL-SUM-style summaries."""
    return (
        "You write very short summaries of news articles.\n"
        "Rules:\n"
        "- Output only one sentence whenever possible (but never more than two).\n"
        "- Respect the requested word range.\n"
        "- The final sentence must end with a single period.\n"
        "- No preface, labels, quotes, bullet points, or line breaks.\n"
        "- Use a neutral, journalistic tone.\n"
        "- Focus on the main event and key entities only."
    )


def create_prompts_and_completions(
    tokenizer,
    dataset_name,
    num_samples,
    subset=None,
    completion_length=40,
    prompt_length=20,
    max_doc_length=None,
    id_prefix=None,
):
    """Load dataset and create prompt/completion pairs."""
    print(f"Loading and processing {dataset_name} dataset...")

    if dataset_name == "c4":
        if not subset:
            raise ValueError('Dataset "c4" requires a subset (e.g., "realnewslike"). Please set dataset.subset.')
        print(f"Target prompt:completion length in tokens -> {prompt_length}:{completion_length}")
        dataset = load_dataset("allenai/c4", subset, split="train", streaming=True)
    elif dataset_name == "xsum":
        dataset = load_dataset("EdinburghNLP/xsum", split="train", streaming=True)
    elif dataset_name == "xl-sum":
        if not subset:
            raise ValueError('Dataset "xl-sum" requires a subset (e.g., "english"). Please set dataset.subset.')
        dataset = load_dataset("csebuetnlp/xlsum", subset, split="train", streaming=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    prompts = []
    human_completions = []
    human_completions_tokenized = []
    
    if dataset_name == "c4":
        for example in dataset:
            text = example["text"].strip()
            if len(text) < 100:
                continue

            all_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(all_tokens) >= (prompt_length + completion_length):
                prompt_tokens = all_tokens[:prompt_length]
                prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                completion_tokens = all_tokens[prompt_length:prompt_length + completion_length]
                completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

                prompts.append(prompt)
                human_completions.append(completion)
                human_completions_tokenized.append(completion_tokens)

                if len(prompts) >= num_samples:
                    break
    
    elif dataset_name == "xsum":
        collected = 0
        for example in dataset:
            doc = (example.get("document") or "").strip()
            summary = (example.get("summary") or "").strip()
            if not doc or not summary:
                continue
            if max_doc_length is not None and len(doc) > max_doc_length:
                continue
            prompts.append(xsum_template(doc))
            human_completions.append(summary)
            collected += 1
            if collected >= num_samples:
                break
        
        human_completions_tokenized = tokenizer(human_completions, add_special_tokens=False)["input_ids"]
    
    elif dataset_name == "xl-sum":
        collected = 0
        for example in dataset:
            title = (example.get("title") or "").strip()
            text = (example.get("text") or "").strip()
            summary = (example.get("summary") or "").strip()
            example_id = (example.get("id") or "").strip()
            if not text or not summary:
                continue
            if id_prefix and not example_id.startswith(id_prefix):
                continue
            if max_doc_length is not None and len(text) > max_doc_length:
                continue
            text = "Title: " + title + "\n\n" + text
            prompts.append(xlsum_template(text))
            human_completions.append(summary)
            collected += 1
            if collected >= num_samples:
                break
            
        human_completions_tokenized = tokenizer(human_completions, add_special_tokens=False)["input_ids"]
    
    if len(prompts) < num_samples:
        print(f"Warning: Could only collect {len(prompts)} samples out of requested {num_samples}.")

    print(f"Loaded {len(prompts)} prompt-completion pairs from {dataset_name} dataset")

    return prompts, human_completions, human_completions_tokenized
