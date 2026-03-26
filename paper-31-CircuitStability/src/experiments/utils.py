import tqdm
import random
import transformers
import torch
import numpy as np
from functools import wraps

from cdatasets import DatasetBuilder, PromptFormatter

import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM


def seed_everything(seed: int = 42):
    random.seed(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_key_value_pairs(pairs):
    """Convert a list of key=value strings into a dictionary."""
    params = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid argument format: {pair}. Expected key=value.")
        key, value = pair.split("=", 1)
        # Attempt to convert to int or float if applicable\
        try:
            params[key] = eval(value)
        except:
            params[key] = value
    return params


def make_dataset(dataset_id, dataset_params, formatter_id, formatter_params):
    db = DatasetBuilder.get_strategy(dataset_id)
    formatter = PromptFormatter.get_strategy(formatter_id, **formatter_params)
    for k, v in dataset_params.items():
        db.set_param(k, v)
    dataset = db.build()
    dataset.get_questions()
    dataset.format_questions(formatter)
    return dataset


@torch.inference_mode()
def eval_pass(model, dataloader, max_new_tokens=15):
    model.eval()
    inputs, out_texts, labels = [], [], []
    for clean_prompt, _, label in tqdm.tqdm(dataloader):
        tokens, _, _, _ = clean_prompt
        outputs = model.generate(tokens, max_new_tokens=max_new_tokens, verbose=False)
        decoded_texts = model.to_string(outputs)
        out_texts.extend(decoded_texts)
        labels.extend(label)
        inputs.extend(model.to_string(tokens))
    return inputs, out_texts, labels


@torch.inference_mode()
def eval_choice(model, dataloader, choices):
    model.eval()
    inputs, out_texts, labels = [], [], []
    choice_logit_idxs = [model.to_single_token(c) for c in choices]
    idx2choice = {i: c for i, c in enumerate(choices)}
    for clean_prompt, _, label in tqdm.tqdm(dataloader):
        tokens, _, _, _ = clean_prompt
        outputs = model(tokens)
        logits = outputs[:, -1, choice_logit_idxs]
        probs = F.softmax(logits, dim=-1)
        answers = [idx2choice[i.item()] for i in probs.argmax(dim=-1)]
        out_texts.extend(answers)
        labels.extend(label)
        inputs.extend(model.to_string(tokens))
    return inputs, out_texts, labels


def kl_metric(model, logits, clean_logits, input_length, labels):
    """Compute the KL divergence between the model's logits and the clean logits.
    Here the clean logits are taken as the target distribution.

    Args:
        model (HookedTransformer)
        logits (torch.Tensor: [batch, seq_len, vocab_size])
        clean_logits (torch.Tensor: [batch, seq_len, vocab_size])
    Return:
        float
    """
    logit_probs = F.log_softmax(logits, dim=-1)
    clean_probs = F.softmax(clean_logits, dim=-1)
    return F.kl_div(logit_probs, clean_probs, reduction="batchmean", log_target=True)


def perplexity(model, logits, clean_logits, input_length, labels):
    """Compute the perplexity of the model's logits given the clean labels.
    It should be that the length of the tokenized labels is equal to the length
    of the logits.

    Args:
        model (HookedTransformer)
        logits (torch.Tensor: [batch, seq_len, vocab_size])
        labels (torch.Tensor: [batch, seq_len])
    """
    logit_probs = F.log_softmax(logits, dim=-1)
    label_toks = model.to_tokens(labels, prepend_bos=True, padding_side="left").to(
        logits.device
    )
    correct_logit_probs = logit_probs.gather(-1, label_toks.unsqueeze(-1)).squeeze(-1)
    nll = -correct_logit_probs
    return nll.mean()


def kl_all_pos(model, logits, clean_logits, input_length, labels):
    clean_probs = F.softmax(clean_logits, dim=-1)

    log_clean = F.log_softmax(clean_logits, dim=-1)
    log_logits = F.log_softmax(logits, dim=-1)

    kl_div = torch.sum(clean_probs * (log_clean - log_logits), dim=-1)
    return kl_div.mean()


def extraction_schema(extract_fn, model, **kwargs):
    def decorator(metric_fn):
        @wraps(metric_fn)
        def wrapper(logits, clean_logits, input_length, labels, model=model):
            logits, clean_logits, labels = extract_fn(
                model, logits, clean_logits, input_length, labels, **kwargs
            )
            return metric_fn(model, logits, clean_logits, input_length, labels)

        return wrapper

    return decorator


def extract_last_token(model, logits, clean_logits, input_length, labels):
    """Extract the last token from the logits and clean logits."""
    return logits[:, -1], clean_logits[:, -1], labels


def extract_equal_sign(model, logits, clean_logits, input_length, labels):
    """Extract the logits from the equal sign to the end of the sequence."""

    def get_equal_pos(str_tokens):
        for i in range(len(str_tokens) - 1, -1, -1):
            if "=" in str_tokens[i]:
                return i
        return -1

    str_tokens = model.to_str_tokens(labels)
    equal_sign_pos = torch.LongTensor(list(map(get_equal_pos, str_tokens))).to(
        logits.device
    )
    return logits[..., equal_sign_pos:], clean_logits[..., equal_sign_pos:], labels


def extract_none(model, logits, clean_logits, input_length, labels):
    return logits, clean_logits, labels


def get_metric(metric_id):
    if metric_id == "kl":
        return kl_metric
    elif metric_id == "perplexity":
        return perplexity
    else:
        raise ValueError(f"Invalid metric id: {metric_id}")


def get_extraction(extraction_id):
    if extraction_id == "last_token":
        return extract_last_token
    elif extraction_id == "equal_sign":
        return extract_equal_sign
    elif extraction_id == "none":
        return extract_none
    else:
        raise ValueError(f"Invalid extraction id: {extraction_id}")


def load_model(
    base_model: str = "pythia-160m",
    variant: str = None,
    checkpoint: int = 143000,
    cache: str = "model_cache",
    device: torch.device = torch.device("cuda"),
    large_model: bool = False,
) -> HookedTransformer:
    """
    Load a transformer model from a pretrained base model or variant.

    Args:
        BASE_MODEL (str): The name of the base model.
        VARIANT (str): The name of the model variant (if applicable).
        CHECKPOINT (int): The checkpoint value for the model.
        CACHE (str): The directory to cache the model.
        device (torch.device): The device to load the model onto.

    Returns:
        HookedTransformer: The loaded transformer model.
    """
    if not variant:

        if large_model:
            model_type = torch.bfloat16
        else:
            model_type = None

        model = HookedTransformer.from_pretrained(
            base_model,
            checkpoint_value=checkpoint,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
            # refactor_factored_attn_matrices=False,
            dtype=model_type,
            **{"cache_dir": cache},
        )
    elif not variant and large_model:
        if large_model:
            model_type = torch.bfloat16
        else:
            model_type = None
        revision = f"step{checkpoint}"
        source_model = (
            AutoModelForCausalLM.from_pretrained(
                f"EleutherAI/{base_model}", revision=revision, cache_dir=cache
            )
            .to(model_type)
            .to("cpu")
        )
        print(
            f"Loaded model {variant} at {revision}; now loading into HookedTransformer"
        )
        model = HookedTransformer.from_pretrained(
            base_model,
            hf_model=source_model,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
            dtype=model_type,
            **{"cache_dir": cache},
        )
    else:
        if large_model:
            model_type = torch.bfloat16
        else:
            model_type = None

        revision = f"step{checkpoint}"
        source_model = AutoModelForCausalLM.from_pretrained(
            variant, revision=revision, cache_dir=cache
        ).to(
            "cpu"
        )  # .to(torch.bfloat16)
        print(
            f"Loaded model {variant} at {revision}; now loading into HookedTransformer"
        )
        model = HookedTransformer.from_pretrained(
            base_model,
            hf_model=source_model,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
            dtype=model_type,
            **{"cache_dir": cache},
        )

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model
