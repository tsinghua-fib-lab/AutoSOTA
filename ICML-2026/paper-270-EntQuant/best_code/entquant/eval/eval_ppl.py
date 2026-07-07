import os
import random
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .evaluator import ModelEvaluator
from .utils import eval_mode

logger = getLogger(__name__)

# Path for locally saved datasets (for offline mode)
if os.getenv("LOCAL_DATASET_PATH") is not None:
    LOCAL_DATASET_PATH = Path(os.getenv("LOCAL_DATASET_PATH"))
else:
    LOCAL_DATASET_PATH = Path(__file__).parent / "local_datasets"


def load_dataset_auto(
    name: str,
    config: str | None = None,
    split: str = "train",
    data_files: dict[str, str] | None = None,
):
    """
    Load dataset from local disk if available, otherwise from HuggingFace Hub.
    Automatically saves to disk after downloading for future offline use.

    Args:
        name: Dataset name (e.g., "wikitext", "ptb_text_only", "allenai/c4")
        config: Dataset configuration (e.g., "wikitext-2-raw-v1", "penn_treebank")
        split: Dataset split (e.g., "train", "test", "validation")
        data_files: Optional data files dict for datasets like C4

    Returns:
        The loaded dataset.
    """
    # Construct local path
    path_parts = [name.replace("/", "_")]
    if config:
        path_parts.append(config)
    if data_files:
        # Include a hash of data_files to differentiate C4 variants
        files_str = "_".join(sorted(data_files.values())).replace("/", "_").replace(".", "_")
        path_parts.append(files_str)
    path_parts.append(split)
    local_path = LOCAL_DATASET_PATH / "_".join(path_parts)

    # Try loading from local disk first
    if local_path.exists():
        logger.info(f"Loading dataset from local cache: {local_path}")
        return load_from_disk(str(local_path))

    # Load from HuggingFace Hub
    logger.info(f"Downloading dataset: {name} (config={config}, split={split})")
    if data_files:
        ds = load_dataset(name, data_files=data_files, split=split)
    elif config:
        ds = load_dataset(name, config, split=split)
    else:
        ds = load_dataset(name, split=split)

    # Save for future offline use
    local_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(local_path))
    logger.info(f"Saved dataset to local cache: {local_path}")

    return ds


class PPLModelEvaluator(ModelEvaluator):
    """
    Evaluates the perplexity of a model on given test dataset.
    The returned dictionary has the following keys:
     - ppl/<dataset_name>: The average perplexity of the model on <dataset_name>
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset_names: tuple[str] | list[str],
        ctx_length: int | None = None,
    ):
        """
        Args:
            tokenizer: Required to process samples of the datasets.
            dataset_names: Names of the datasets to evaluate on. Currently supported: "wikitext", "ptb", "c4".
            ctx_length: Each dataset is fed to the model in chunks of this length.
                If None, the model's max context length is used.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_names = dataset_names
        self.ctx_length = ctx_length

    def __call__(
        self,
        model: nn.Module | PreTrainedModel,
        prefix: str | None = "ppl",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert isinstance(model, PreTrainedModel), (
            f"Expected PreTrainedModel, got {type(model).__name__}. If using EntQuantModel, pass entquant_model.model."
        )
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            for dataset_name in self.dataset_names:
                logger.info(f"Validating PPL on {dataset_name}...")
                ppl = evaluate_ppl(
                    dataset_name=dataset_name, model=model, tokenizer=self.tokenizer, ctx_length=self.ctx_length
                )
                logger.info(f"Validated PPL on {dataset_name}: {ppl}")
                results[prefix + dataset_name] = ppl
        return results


def evaluate_ppl(
    dataset_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ctx_length: int | None,
    n_samples: int = 128,
    seed: int = 0,
) -> float:
    """
    Script to evaluate the perplexity of a model on a given dataset where the samples are fed to the model in chunks.

    Adapted from https://github.com/locuslab/wanda/blob/main/lora_ft/evaluate_ppl.py.

    Args:
        dataset_name: Name of the dataset to evaluate on. Currently supported: "wikitext2", "ptb", "c4", etc.
        model: An `PreTrainedModel` that supports causal language modeling.
        tokenizer: Suitable tokenizer for the model.
        ctx_length: Chunk size to feed to the model. If None, the model's max context length is used.
        n_samples: Number of samples for calibration (used by get_loaders).
        seed: Random seed for reproducibility.

    Returns: The average perplexity of the model on the dataset.
    """
    # Get max context length of model if ctx_length is not provided
    if ctx_length is None:
        ctx_length = model.config.max_position_embeddings

    # Use get_loaders to handle data preprocessing - we use the testenc for perplexity evaluation
    _, testenc = get_loaders(name=dataset_name, tokenizer=tokenizer, nsamples=n_samples, seed=seed, seqlen=ctx_length)

    # Feed the encoded string to the model in chunks of size ctx_length and measure the perplexity on each chunk
    input_ids = testenc.input_ids
    n_iterations = input_ids.numel() // ctx_length
    nlls = []
    for i in tqdm(range(n_iterations)):
        batch = input_ids[:, (i * ctx_length) : ((i + 1) * ctx_length)].to(model.device)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * ctx_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (n_iterations * ctx_length))

    return ppl.item()


# Everything below is adopted from https://github.com/LeanModels/LeanQuant/blob/master/datautils.py


def get_wikitext2(nsamples: int, seed: int, seqlen: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    traindata = load_dataset_auto("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset_auto("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples: int, seed: int, seqlen: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    traindata = load_dataset_auto("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset_auto("ptb_text_only", "penn_treebank", split="validation")

    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples: int, seed: int, seqlen: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    traindata = load_dataset_auto(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset_auto(
        "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples: int, seed: int, seqlen: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    traindata = load_dataset_auto("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset_auto("ptb_text_only", "penn_treebank", split="test")

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples: int, seed: int, seqlen: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    traindata = load_dataset_auto(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset_auto(
        "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_c4_full(nsamples: int, seed: int, seqlen: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    traindata = load_dataset_auto(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset_auto(
        "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )

    np.random.seed(seed)
    idx_perm = np.random.permutation(np.arange(len(traindata))).tolist()

    trainloader = []
    for i in idx_perm:
        trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] >= seqlen:
            inp = trainenc.input_ids[:, :seqlen]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            if len(trainloader) >= nsamples:
                break

    logger.info(f"Collected {len(trainloader)} calibration samples.")

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        if "full" in name:
            return get_c4_full(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {name}")
