from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def model2family(model_name: str):
    if "gpt2" in model_name:
        return "gpt2"
    elif "pythia" in model_name:
        return "pythia"
    else:
        raise ValueError(f"Couldn't find model family for model: {model_name}")


def kl_div(logits, clean_logits, input_length, labels, mean=True):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    clean_logits = clean_logits[idx, input_length - 1]

    logprobs = torch.log_softmax(logits, dim=-1)
    clean_logprobs = torch.log_softmax(clean_logits, dim=-1)

    results = torch.nn.functional.kl_div(
        logprobs, clean_logprobs, log_target=True, reduction="none"
    )
    return results.mean() if mean else results


def precision_at_k(
    clean_logits, corrupted_logits, input_length, labels, k=1, mean=True
):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)
    predictions = torch.argmax(clean_probs, dim=-1).cpu()

    results = []
    for i, (ls, _) in enumerate(labels):
        r = torch.sum((ls == predictions[i]).float())
        results.append(r)
    results = torch.stack(results)
    return results.mean() if mean else results


def prob_diff_hypernymy(
    clean_logits,
    corrupted_logits,
    input_length,
    labels,
    mean=True,
    loss=False,
    logits=False,
):
    batch_size = clean_logits.size(0)
    idx = torch.arange(batch_size, device=clean_logits.device)

    clean_logits = clean_logits[idx, input_length - 1]
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if logits:
        clean_probs = clean_logits

    results = []
    for i, (ls, corrupted_ls) in enumerate(labels):
        r = (
            clean_probs[i][ls.to(clean_probs.device)].sum()
            - clean_probs[i][corrupted_ls.to(clean_probs.device)].sum()
        )
        results.append(r)
    results = torch.stack(results)
    if loss:
        results = -results
    return results.mean() if mean else results


def batch(iterable, n: int = 1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def get_singular_and_plural(model, strict=False) -> Tuple[torch.Tensor, torch.Tensor]:
    _TOKENIZER = model.tokenizer
    tokenizer_length = model.cfg.d_vocab_out

    df: pd.DataFrame = pd.read_csv("../data/sva/combined_verb_list.csv")
    singular = df["sing"].to_list()
    plural = df["plur"].to_list()
    singular_set = set(singular)
    plural_set = set(plural)
    verb_set = singular_set | plural_set
    assert len(singular_set & plural_set) == 0, f"{singular_set & plural_set}"
    singular_indices, plural_indices = [], []

    for i in range(tokenizer_length):
        token = _TOKENIZER._convert_id_to_token(i)
        if token is not None:
            if token[0] == "Ġ":
                token = token[1:]
                if token in verb_set:
                    if token in singular_set:
                        singular_indices.append(i)
                    else:  # token in plural_set:
                        idx = plural.index(token)
                        third_person_present = singular[idx]
                        third_person_present_tokenized = _TOKENIZER(
                            f" {third_person_present}", add_special_tokens=False
                        )["input_ids"]
                        if (
                            len(third_person_present_tokenized) == 1
                            and third_person_present_tokenized[0]
                            != _TOKENIZER.unk_token_id
                        ):
                            plural_indices.append(i)
                        elif not strict:
                            plural_indices.append(i)

    return torch.tensor(singular_indices, device=model.cfg.device), torch.tensor(
        plural_indices, device=model.cfg.device
    )


def get_sva_prob_diff(model, strict=True) -> torch.Tensor:
    singular_indices, plural_indices = get_singular_and_plural(model, strict=strict)

    def sva_prob_diff(
        logits, clean_logits, input_length, labels, loss=False, mean=True
    ):
        batch_size = clean_logits.size(0)
        idx = torch.arange(batch_size, device=clean_logits.device)
        probs = F.softmax(logits[idx, input_length - 1], dim=-1)
        singular = probs[:, singular_indices].sum(-1)
        plural = probs[:, plural_indices].sum(-1)

        correct_form_prob_diff = torch.where(
            labels == 0, singular - plural, plural - singular
        )
        if loss:
            correct_form_prob_diff = -correct_form_prob_diff
        if mean:
            return correct_form_prob_diff.mean()
        else:
            return correct_form_prob_diff

    return sva_prob_diff


def inflow_outflow_difference(g, absolute: bool = True):
    diffs = []
    for name, node in g.nodes.items():
        if "logits" in name or "input" in name:
            continue
        diff = sum(edge.score for edge in node.child_edges) - sum(
            edge.score for edge in node.parent_edges
        )
        if absolute:
            diff = abs(diff)
        diffs.append(diff)
    diffs = np.array(diff)
    logit_inflow = sum(edge.score for edge in g.logits[0].parent_edges)
    input_outflow = sum(edge.score for edge in g.nodes["input"].child_edges)
    return diffs.mean(), logit_inflow, input_outflow
