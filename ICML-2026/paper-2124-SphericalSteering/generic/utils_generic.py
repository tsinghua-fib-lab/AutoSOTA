"""
Utility functions for generic MC task evaluation with Spherical Steering.

Supports: COPA, StoryCloze, MMLU (global), Winogrande, BoolQ

Prompt style: Simple "Q: {question} A: {answer}" format.
"""

import torch
import numpy as np
import random
from baukit import TraceDict
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


# ==================== Random Seeds ====================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Model Paths ====================

HF_NAMES = {
    'llama3.1-8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
}


# ==================== MMLU Categories ====================

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics", "high_school_physics",
        "high_school_statistics", "machine_learning", "number_theory", "physics"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social_Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
        "virology"
    ]
}


# ==================== Prompt Formatting ====================

def format_prompt(question, answer=None):
    """
    Prompt: "Q: {question} A: {answer}"
    """
    if answer is not None:
        return f"Q: {question} A: {answer}"
    else:
        return f"Q: {question} A:"


# ============================================================
# Data Loaders for Prototype Extraction (Training)
# Returns: prompts, labels, q_indices
# ============================================================

def get_copa_data(seed=42):
    """
    Load COPA (super_glue) train split (400 samples).
    For each item: correct choice → label=1, incorrect → label=0.
    """
    print(f"Loading COPA for prototype extraction...")
    dataset = load_dataset("super_glue", "copa", split="train")
    dataset = dataset.shuffle(seed=seed)

    prompts, labels, q_indices = [], [], []

    for i, item in enumerate(dataset):
        premise = item["premise"]
        q_type = item["question"]
        if q_type == "cause":
            question = f"{premise} What was the cause of this?"
        else:
            question = f"{premise} What happened as a result?"

        choices = [item["choice1"], item["choice2"]]
        correct_label = item["label"]  # 0 or 1

        # Correct
        prompts.append(format_prompt(question, choices[correct_label]))
        labels.append(1)
        q_indices.append(i)
        # Incorrect
        prompts.append(format_prompt(question, choices[1 - correct_label]))
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


def get_storycloze_data(split='train', num_samples=None):
    """
    Load XStoryCloze (English).
    For each story: correct ending → label=1, incorrect → label=0.
    """
    print(f"Loading XStoryCloze (en, split={split})...")
    hf_split = 'train' if split == 'train' else 'eval'
    dataset = load_dataset("juletxara/xstory_cloze", "en")[hf_split]

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts, labels, q_indices = [], [], []

    for i, item in enumerate(dataset):
        story = " ".join([
            item['input_sentence_1'], item['input_sentence_2'],
            item['input_sentence_3'], item['input_sentence_4']
        ])
        opt1 = item['sentence_quiz1']
        opt2 = item['sentence_quiz2']
        correct_idx = item['answer_right_ending']  # 1 or 2

        if correct_idx == 1:
            correct_text, incorrect_text = opt1, opt2
        else:
            correct_text, incorrect_text = opt2, opt1

        prompts.append(format_prompt(story, correct_text))
        labels.append(1)
        q_indices.append(i)
        prompts.append(format_prompt(story, incorrect_text))
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


def get_mmlu_global_data(split='train', seed=42):
    """
    Load MMLU Global Balanced Dataset (all 4 categories combined).

    Split strategy (per category):
      - train: indices 0-500   (500/cat, 2000 total)
      - test:  indices 500-1000 (500/cat, 2000 total)

    Creates a single balanced dataset for computing ONE global prototype
    and ONE micro-average accuracy across all of MMLU.
    """
    print(f"Loading MMLU Global Balanced (split={split})...")

    all_prompts, all_labels, all_q_indices = [], [], []
    global_q_idx = 0

    for cat_name, subsets in MMLU_CATEGORIES.items():
        loaded = []
        for sub in subsets:
            try:
                loaded.append(load_dataset("cais/mmlu", sub, split='test'))
            except Exception as e:
                print(f"  Warning: Failed to load {sub}: {e}")
        if not loaded:
            continue

        cat_dataset = concatenate_datasets(loaded).shuffle(seed=seed)
        total = len(cat_dataset)

        if split == 'train':
            start_idx, end_idx = 0, min(500, total)
        elif split == 'test':
            start_idx, end_idx = 500, min(1000, total)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")

        if start_idx >= total:
            continue

        dataset = cat_dataset.select(range(start_idx, end_idx))
        print(f"  {cat_name}: {len(dataset)} questions")

        for item in dataset:
            question = item['question']
            choices = item['choices']
            correct_idx = item['answer']

            all_prompts.append(format_prompt(question, choices[correct_idx]))
            all_labels.append(1)
            all_q_indices.append(global_q_idx)
            for j, c in enumerate(choices):
                if j != correct_idx:
                    all_prompts.append(format_prompt(question, c))
                    all_labels.append(0)
                    all_q_indices.append(global_q_idx)
            global_q_idx += 1

    print(f"  Total: {len(all_prompts)} samples from {global_q_idx} questions")
    return all_prompts, np.array(all_labels), np.array(all_q_indices)


def get_winogrande_data(split='train', num_samples=None, seed=42):
    """
    Load Winogrande (winogrande_xl).
    For each item: correct option → label=1, incorrect → label=0.
    """
    print(f"Loading Winogrande (split={split})...")
    hf_split = 'train' if split == 'train' else 'validation'
    dataset = load_dataset("winogrande", "winogrande_xl")[hf_split]

    if split == 'train':
        dataset = dataset.shuffle(seed=seed)

    # Default to 2000 train samples if not specified
    if num_samples is None and split == 'train':
        num_samples = 2000
        print("  Using default num_samples=2000 for Winogrande train split")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    prompts, labels, q_indices = [], [], []
    for i, item in enumerate(dataset):
        sentence = item['sentence']
        opt1 = item['option1']
        opt2 = item['option2']
        correct_idx = item['answer']  # '1' or '2'

        if correct_idx == '1':
            correct_text, incorrect_text = opt1, opt2
        else:
            correct_text, incorrect_text = opt2, opt1

        prompts.append(format_prompt(sentence, correct_text))
        labels.append(1)
        q_indices.append(i)
        prompts.append(format_prompt(sentence, incorrect_text))
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


def get_boolq_data(num_questions=2000, seed=42):
    """
    Load BoolQ (super_glue) train split.
    Options: ["no", "yes"], label: 0→no, 1→yes.
    """
    print(f"Loading BoolQ for prototype extraction...")
    dataset = load_dataset("super_glue", "boolq", split="train")
    dataset = dataset.shuffle(seed=seed)

    if num_questions is not None:
        dataset = dataset.select(range(min(num_questions, len(dataset))))

    options = ["no", "yes"]
    prompts, labels, q_indices = [], [], []

    for i, item in enumerate(dataset):
        passage = item["passage"]
        question = item["question"]
        label = int(item["label"])  # 0 or 1

        base = f"Passage: {passage}\nQuestion: {question}\nA:"

        prompts.append(base + " " + options[label])
        labels.append(1)
        q_indices.append(i)
        prompts.append(base + " " + options[1 - label])
        labels.append(0)
        q_indices.append(i)

    print(f"  Total: {len(prompts)} samples from {len(dataset)} questions")
    return prompts, np.array(labels), np.array(q_indices)


# ==================== Unified Data Loader ====================

def get_dataset_data(dataset_name, split='train', num_samples=None, seed=42):
    """Unified data loader interface."""
    if dataset_name == 'copa':
        return get_copa_data(seed=seed)
    elif dataset_name == 'storycloze':
        return get_storycloze_data(split, num_samples)
    elif dataset_name == 'mmlu_global':
        return get_mmlu_global_data(split, seed)
    elif dataset_name == 'winogrande':
        return get_winogrande_data(split, num_samples, seed)
    elif dataset_name == 'boolq':
        return get_boolq_data(num_questions=num_samples or 2000, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: copa, storycloze, mmlu_global, winogrande, boolq")


# ==================== Feature Extraction ====================

def get_layer_activations(model, tokenizer, prompts, layer_idx, device):
    """
    Extract last-token activations from a specific layer using baukit.
    Batch size = 1 for precision.
    """
    layer_name = f"model.layers.{layer_idx}"
    activations = []

    print(f"Extracting activations from {layer_name}...")
    model.eval()

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting"):
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            with TraceDict(model, [layer_name]) as ret:
                model(**inputs)

            layer_output = ret[layer_name].output
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]

            last_token_act = layer_output[0, -1, :].cpu().numpy()
            activations.append(last_token_act)

    return np.array(activations)
