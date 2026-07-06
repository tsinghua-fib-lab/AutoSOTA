from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal, Sequence
from collections import Counter
import re
import string
import torch
import numpy as np

from transformers import AutoTokenizer


class LargeModel(Enum):
    opt_125m = "opt-125m"
    opt_350m = "opt-350m"
    opt_1p3b = "opt-1.3b"
    opt_2p7b = "opt-2.7b"
    opt_6p7b = "opt-6.7b"
    opt_13b = "opt-13b"
    opt_30b = "opt-30b"
    gemma_3_270m = "gemma-3-270m"
    llama_32_1b = "llama-3.2-1b"


class ModelDtype(Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


# LLM
SUPPORTED_LLM = {
    "opt-125m": "facebook/opt-125m",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",
    "opt-30b": "facebook/opt-30b",
    "gemma-3-270m": "google/gemma-3-270m",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1b",
}


def get_hf_tokenizer(hf_model_name):
    return AutoTokenizer.from_pretrained(
        hf_model_name, padding_side="left", truncate_side="left"
    )


class CustomLMDataset(torch.utils.data.DataLoader):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = self.texts[idx]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        # left_truncation
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
        return torch.tensor(input_ids, dtype=torch.long)


class CustomLMGenerationDataset(torch.utils.data.DataLoader):
    def __init__(self, texts, golds, tokenizer, max_length):
        assert len(texts) == len(golds)
        self.texts = texts
        self.golds = golds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = self.texts[idx]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        # left_truncation
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
        return torch.tensor(input_ids, dtype=torch.long), (
            len(input_ids),
            self.golds[idx],
        )


class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError

    def verbalize(self, sample):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        raise NotImplementedError


class ClassificationTemplate(Template):
    verbalizer = {0: "0", 1: "1"}

    def get_verbalizer_id(self, tokenizer):
        return {k: tokenizer.encode(v)[-1] for k, v in self.verbalizer.items()}

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class SST2Template(ClassificationTemplate):
    verbalizer = {0: " bad", 1: " good"}

    def verbalize_for_pred(self, sample):
        text = sample["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class QQPTemplate(ClassificationTemplate):
    verbalizer = {0: " No", 1: " Yes"}

    def verbalize_for_pred(self, sample):
        q1 = sample["question1"].strip()
        q2 = sample["question1"].strip()
        return f"Question 1: {q1}\n Question 2: {q2}\n Are they semantically equivalent? Yes or No?\n"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class BoolQTemplate(ClassificationTemplate):
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        passage = sample["passage"]
        question = sample["question"]
        return f"{passage}\nQuestion: {question}\nIs that correct? Yes or No?\n"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


class RTETemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def verbalize_for_pred(self, sample):
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        return f'{premise}\nDoes this mean that "{hypothesis}" is true? Yes or No?\n'


class MultiRCTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        paragraph = sample["paragraph"]
        question = sample["question"]
        answer = sample["answer"]
        return f'{paragraph}\nQuestion: {question}\nI found this answer "{answer}". Is that correct? Yes or No?\n'


class CBTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def verbalize_for_pred(self, sample):
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        return (
            f'Suppose {premise} Can we infer that "{hypothesis}"? Yes, No, or Maybe?\n'
        )


class WICTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        sent1 = sample["sentence1"]
        sent2 = sample["sentence2"]
        word = sample["word"]
        return f'Does the word "{word}" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n'


class WSCTemplate(ClassificationTemplate):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def verbalize_for_pred(self, sample):
        text = sample["text"]
        span1 = sample["span1_text"]
        span2 = sample["span2_text"]
        return f'{text}\nIn the previous sentence, does the pronoun "{span2.lower()}" refer to {span1}? Yes or No?\n'


class SQuADTemplate(Template):
    def encode(self, sample):
        prompt = "Answer concisely in a few words:"
        question = sample["question"].strip()
        title = sample["title"]
        context = sample["context"]
        return f"Title: {title}\nContext: {context}\nQuestion: {question}\n{prompt}"

    def verbalize(self, sample):
        prompt = "Answer concisely in a few words:"
        question = sample["question"].strip()
        title = sample["title"]
        context = sample["context"]
        # There are multiple answers. For the prompt we only take the first one
        answer = sample["answers"]["text"][0]
        return f"Title: {title}\nContext: {context}\nQuestion: {question}\n{prompt}{answer}"


class DROPTemplate(Template):
    def encode(self, sample):
        prompt = "Answer:"
        question = sample["question"].strip()
        passage = sample["passage"]
        return f"Passage: {passage}\nQuestion: {question}\{prompt}:"

    def verbalize(self, sample):
        prompt = "Answer:"
        question = sample["question"].strip()
        passage = sample["passage"]
        # There are multiple answers. for the prompt we only take the first one
        answer = sample["answers_spans"]["spans"][0]
        return f"Passage: {passage}\nQuestion: {question}\n{prompt}{answer}"


class XSUMTemplate(Template):
    def encode(self, sample):
        prompt = "Summarize this in one sentence:"
        document = sample["document"]
        return f"Document: {document}\n{prompt}:"

    def verbalize(self, sample):
        prompt = "Summarize this in one sentence:"
        document = sample["document"]
        summary = sample["summary"]
        return f"Document: {document}\n{prompt}{summary}"


class LmClassificationTask(Enum):
    sst2 = "sst2"
    rte = "rte"
    multirc = "multirc"
    cb = "cb"
    wic = "wic"
    wsc = "wsc"
    boolq = "boolq"
    qqp = "qqp"


class LmGenerationTask(Enum):
    squad = "squad"
    drop = "drop"
    xsum = "xsum"


LM_DATASET_MAP = {
    LmClassificationTask.sst2.name: "glue",
    LmClassificationTask.rte.name: "super_glue",
    LmClassificationTask.multirc.name: "super_glue",
    LmClassificationTask.cb.name: "super_glue",
    LmClassificationTask.wic.name: "super_glue",
    LmClassificationTask.wsc.name: "super_glue",
    LmClassificationTask.boolq.name: "super_glue",
    LmClassificationTask.qqp.name: "glue",
    LmGenerationTask.squad.name: "squad",
    LmGenerationTask.drop.name: "drop",
    LmGenerationTask.xsum.name: "xsum",
}

LM_TEMPLATE_MAP = {
    LmClassificationTask.sst2.name: SST2Template,
    LmClassificationTask.rte.name: RTETemplate,
    LmClassificationTask.multirc.name: MultiRCTemplate,
    LmClassificationTask.cb.name: CBTemplate,
    LmClassificationTask.wic.name: WICTemplate,
    LmClassificationTask.wsc.name: WSCTemplate,
    LmClassificationTask.boolq.name: BoolQTemplate,
    LmClassificationTask.qqp.name: QQPTemplate,
    LmGenerationTask.squad.name: SQuADTemplate,
    LmGenerationTask.drop.name: DROPTemplate,
    LmGenerationTask.xsum.name: XSUMTemplate,
}


@dataclass
class LLMBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device=None, dtype=None):
        self.input_ids = self.input_ids.to(device=device)
        self.attention_mask = self.attention_mask.to(device=device)
        return self


def get_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        # Pad sequences to the max length in the batch
        padded_batch = tokenizer.pad(
            {"input_ids": batch},
            padding="max_length",
            max_length=max_length,
            # max_length=max_length,
            return_tensors="pt",
        )
        input_ids = padded_batch["input_ids"]
        attention_mask = padded_batch["attention_mask"]
        return (
            LLMBatchInput(input_ids[:, :(-1)], attention_mask[:, :(-1)]),
            input_ids[:, 1:],
        )  # Prepare input and target sequences

    return collate_fn


def get_collate_fn_for_gen_model(tokenizer, max_length):
    def collate_fn(batch):
        inputs, golds = zip(*batch)
        # Pad sequences to the max length in the batch
        padded_batch = tokenizer.pad(
            {"input_ids": inputs},
            padding="max_length",
            # max_length=max_length,
            return_tensors="pt",
        )
        input_ids = padded_batch["input_ids"]
        attention_mask = padded_batch["attention_mask"]
        return (LLMBatchInput(input_ids, attention_mask), golds)

    return collate_fn


def get_lm_loss(
    loss_type: Literal["full_sentence", "last_token", "accuracy", "f1"],
    *,
    verbalizer_id_map: dict[int, int] | None = None,
    tokenizer=None,
):
    if loss_type == "f1":  # notice this is not a real score
        return partial(f1_batch_score, tokenizer=tokenizer)

    assert verbalizer_id_map is not None
    n_candidate = len(verbalizer_id_map)
    verbalizer_id_list = [verbalizer_id_map[i] for i in range(n_candidate)]
    if loss_type == "full_sentence":
        return full_sentence_cross_entropy_loss
    elif loss_type == "last_token":
        return partial(
            last_token_cross_entropy_loss,
            verbalizer_id_map=verbalizer_id_map,
            verbalizer_id_list=verbalizer_id_list,
        )
    elif loss_type == "accuracy":
        return partial(
            last_token_accuracy,
            verbalizer_id_map=verbalizer_id_map,
            verbalizer_id_list=verbalizer_id_list,
        )


def full_sentence_cross_entropy_loss(batch_pred, sentence_label_tokens):
    logits = batch_pred.logits
    # Flatten the logits and labels for calculating loss
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = sentence_label_tokens.contiguous().view(-1)

    # Calculate the loss
    loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)
    return loss


def last_token_cross_entropy_loss(
    batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list
):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(
        -1, len(verbalizer_id_list)
    )
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)

    loss = torch.nn.functional.cross_entropy(
        last_token_batch_pred, last_token_label, label_smoothing=0.1
    )
    return loss


def last_token_accuracy(
    batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list
):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(
        -1, len(verbalizer_id_list)
    )
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)

    pred = last_token_batch_pred.max(1, keepdim=True)[1]
    return pred.eq(last_token_label.view_as(pred)).cpu().float().mean()


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred: str, gold: list[str]) -> float:
    if gold[0].lower() in ["cannotanswer", "no answer"]:
        return int(normalize_answer(gold[0]) == normalize_answer(pred))
    else:
        all_f1s: list[float] = []
        for ans in gold:
            prediction_tokens = normalize_answer(pred).split()
            ground_truth_tokens = normalize_answer(ans).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                all_f1s.append(0.0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                all_f1s.append((2 * precision * recall) / (precision + recall))
        return float(np.max(all_f1s))


def f1_batch_score(
    batch_pred: torch.Tensor, golden_outputs: Sequence[tuple[int, str]], tokenizer
) -> torch.Tensor:
    assert batch_pred.shape[0] == len(golden_outputs)
    f1s = []
    start_pos = max(pos_and_gold[0] for pos_and_gold in golden_outputs)

    for pred, pos_and_gold in zip(batch_pred, golden_outputs):
        _, gold_sentence = pos_and_gold
        pred_sentence = tokenizer.decode(
            pred[start_pos:], skip_special_tokens=True
        ).strip()
        f1 = f1_score(pred_sentence, [gold_sentence])
        f1s.append(f1)
        # print(f"==============\n{f1=}\npred = {pred_sentence}\ngold = {gold_sentence}")
    return torch.tensor(np.mean(f1s), dtype=torch.float32)
