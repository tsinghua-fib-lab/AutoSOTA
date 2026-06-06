import copy
import logging
import os
import random
from typing import Any

import torch
import transformers
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from dart_math.utils import (
    IGNORE_IDX,
    PROJ_HOME,
    PromptTemplate,
    get_pathname_from_name_or_path,
)


# %% ../nbs/01_train.ipynb 0
def tokenize_fn(
    tokenizer: transformers.PreTrainedTokenizer,  # (HF) tokenizer.
    strings: list[str],  # Strings to tokenize.
) -> dict[str, list[torch.Tensor | int]]:
    """Tokenize a list of strings."""

    tokenized_dict = tokenizer(
        strings,
        padding=False,  # No padding for cache
        max_length=tokenizer.model_max_length,
        truncation=True,
        # ValueError: False is not a valid TensorType, please select one of ['pt', 'tf', 'np', 'jax', 'mlx']
        # return_tensors= # Return list[list]
    )
    input_ids_list = tokenized_dict["input_ids"]
    # No need to consider padding because `padding=False`
    input_id_lens = [len(input_id) for input_id in input_ids_list]

    input_ids_pt_list = [
        torch.tensor(input_ids, dtype=torch.int) for input_ids in input_ids_list
    ]

    return {"input_ids": input_ids_pt_list, "input_id_lens": input_id_lens}


def preprocess(
    tokenizer: transformers.PreTrainedTokenizer,
    sources: list[str],
    targets: list[str],
    verbose: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Preprocess the `sources` and `targets` for training.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer.
    sources : list[str]
        Sources as input to the model.
    targets : list[str]
        Targets as output of the model.
    verbose : bool, default: True
        Whether to show some examples.

    Returns
    -------
    dict[str, list[torch.Tensor]]
        `{"input_ids": input_ids, "labels": labels}`
    """

    examples = [s + t for s, t in zip(sources, targets)]

    logging.info("Tokenizing examples ...")
    examples_tokenized = tokenize_fn(tokenizer, examples)

    logging.info("Tokenizing sources ...")
    sources_tokenized = tokenize_fn(tokenizer, sources)

    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)

    for label_id_seq, source_len in zip(labels, sources_tokenized["input_id_lens"]):
        # Mask out the prompttokens for loss computation
        label_id_seq[:source_len] = IGNORE_IDX
        # No need to consider padding because `padding=False`
        # label_id_seq[label_id_seq == self.tokenizer.pad_token_id] = IGNORE_IDX

    if verbose:
        # Show some masked examples
        rand_idx = random.randint(0, len(input_ids) - 1)
        eg_example = input_ids[rand_idx]
        eg_example_text = tokenizer.decode(eg_example)
        eg_label = labels[rand_idx]
        eg_label_text = tokenizer.decode(eg_label[eg_label != IGNORE_IDX])
        logging.info(
            f"Masked labels.\neg_example_text:\n\n{eg_example_text}\n\neg_label_text:\n{eg_label_text}"
        )

    return {"input_ids": input_ids, "labels": labels}


class TokenizedSupervisedDataset(Dataset):
    """Tokenized dataset for supervised fine-tuning.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer. `None` for empty dataset.
    input_ids : list[torch.Tensor], default: []
        List of input token ID sequences.
    labels : list[torch.Tensor], default: []
        List of label sequences.
    attention_mask : list[torch.Tensor], default: []
        List of attention mask sequences.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        input_ids: list[torch.Tensor] = None,
        labels: list[torch.Tensor] = None,
        attention_mask: list[torch.Tensor] = None,
    ):

        Dataset.__init__(self)

        if input_ids is None:
            input_ids = []
        if labels is None:
            labels = []
        if attention_mask is None:
            attention_mask = []

        self.tokenizer = tokenizer

        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

    @staticmethod
    def load_from_raw_dset(
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        query_field: str = "query",
        resp_field: str = "response",
        prompt_template: str | dict[str, str] | PromptTemplate = "alpaca",
    ) -> "TokenizedSupervisedDataset":
        """Load a dataset from a file and tokenize it.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            (HF) tokenizer.
        data_path : str
            Dataset ID or path.
        query_field : str, default: "query"
            Field name for query.
        resp_field : str, default: "response"
            Field name for response.
        prompt_template : str | dict[str, str] | PromptTemplate, default: "alpaca"
            ID / File path / PromptTemplate object of prompt template.

        Returns
        -------
        TokenizedSupervisedDataset
        """
        logging.info("Loading from raw dataset ...")
        dset = load_dataset(data_path)["train"]

        queries = dset[query_field]
        resps = dset[resp_field]

        logging.debug("Formatting inputs ...")
        if not isinstance(prompt_template, PromptTemplate):
            prompt_template = PromptTemplate.load_from_id_or_path(prompt_template)
        sources = [prompt_template.make_full_prompt(query) for query in queries]

        targets = [f"{resp}{tokenizer.eos_token}" for resp in resps]

        data_dict = preprocess(tokenizer, sources, targets)
        input_ids, labels = data_dict["input_ids"], data_dict["labels"]

        return TokenizedSupervisedDataset(
            tokenizer=tokenizer,
            input_ids=input_ids,
            labels=labels,
            attention_mask=[
                input_id_seq.ne(tokenizer.pad_token_id) for input_id_seq in input_ids
            ],
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(
        self,
        i: int,
    ) -> dict[str, torch.Tensor]:
        """Get a data point.

        Parameters
        ----------
        i : int
            `dataset[i]`

        Returns
        -------
        dict[str, torch.Tensor]
            `{"input_ids": input_ids[i], "labels": labels[i], "attention_mask": attention_mask[i]}`
        """
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
        }

    def concat(
        self,
        datasets: list["TokenizedSupervisedDataset"],
    ) -> None:
        """Concatenate `TokenizedSupervisedDataset` instances to the current dataset.
        datasets : list[TokenizedSupervisedDataset]
            List of tokenized datasets to concatenate.
            Each dataset should have the following fields at least: `"input_ids"`, `"labels"`, and `"attention_mask"`.
        """
        self.input_ids += sum([ds.input_ids for ds in datasets], [])
        self.labels += sum([ds.labels for ds in datasets], [])
        self.attention_mask += sum([ds.attention_mask for ds in datasets], [])

    def shuffle(self, seed: int = 42) -> None:
        """Shuffle the dataset."""
        random.seed(seed)
        rand_idxs = list(range(len(self)))
        random.shuffle(rand_idxs)
        logging.debug(f"rand_indices[:10] = {rand_idxs[:10]}")
        logging.debug(f"rand_indices[-10:] = {rand_idxs[-10:]}")
        self.input_ids = [self.input_ids[i] for i in rand_idxs]
        self.labels = [self.labels[i] for i in rand_idxs]
        self.attention_mask = [self.attention_mask[i] for i in rand_idxs]

    def pad(self) -> None:
        """Pad the dataset to the same length of the longest data point."""
        max_len = max([len(input_id) for input_id in self.input_ids])
        for i in range(len(self.input_ids)):
            pad_len = max_len - len(self.input_ids[i])
            self.input_ids[i] = torch.cat(
                [
                    self.input_ids[i],
                    torch.tensor([self.tokenizer.pad_token_id] * pad_len),
                ]
            )
            self.labels[i] = torch.cat(
                [self.labels[i], torch.tensor([IGNORE_IDX] * pad_len)]
            )
            self.attention_mask[i] = torch.cat(
                [self.attention_mask[i], torch.tensor([0] * pad_len)]
            )


class PackedDataset(Dataset):
    """Packed dataset containing computation sequences.

    Parameters
    ----------
    dataset : Dataset | HFDataset
        Original tokenized dataset, which should have the following fields at least: `"input_ids"`, `"labels"`, and `"attention_mask"`.
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer.
    pack_len : int
        Maximum length of packed compuation sequence in token.
    shuffle_seed : int, default: 42
        Seed for shuffling the dataset before packing. `None` / Negative values mean no shuffling.
    """

    def __init__(
        self,
        dataset: Dataset | HFDataset,
        tokenizer: transformers.PreTrainedTokenizer,
        pack_len: int,
        shuffle_seed: int = 42,
    ):
        Dataset.__init__(self)

        self.pack_len = pack_len
        self.tokenizer = tokenizer

        self.lens = []
        self.dps = []

        dp_idxs = list(range(len(dataset)))
        if shuffle_seed is not None and shuffle_seed >= 0:
            random.seed(shuffle_seed)
            random.shuffle(dp_idxs)
            logging.debug(
                f"Shuffled dataset with seed = {shuffle_seed}, getting {dp_idxs[:5]}..."
            )

        for i_dp in dp_idxs:
            raw_dp = dataset[i_dp]
            input_len = torch.sum(raw_dp["attention_mask"]).item()

            raw_dp["input_ids"] = PackedDataset.extract_ids(
                raw_dp["input_ids"], input_len, tokenizer.padding_side
            )

            if "labels" not in raw_dp:  # Create labels if not existed
                labels = raw_dp["input_ids"].clone()
                # Mask pad_token
                labels[labels == tokenizer.pad_token_id] = IGNORE_IDX
                raw_dp["labels"] = labels.tolist()
            else:  # Extract labels
                raw_dp["labels"] = PackedDataset.extract_ids(
                    raw_dp["labels"], input_len, tokenizer.padding_side
                )

            self.dps.append(raw_dp)
            self.lens.append(input_len)

        max_input_len = max(self.lens)
        assert (
            self.pack_len >= max_input_len
        ), f"pack_len must be >= max(input lens), found pack_len={self.pack_len}, max_input_len={max_input_len}"
        self.groups = PackedDataset.pack_dps_by_len(self.lens, self.pack_len)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.groups)))]
        group = self.groups[idx]
        group_dps = [self.dps[index] for index in group]
        return PackedDataset.pack_dps_FA(group_dps, self.tokenizer, self.pack_len)

    @staticmethod
    def extract_ids(
        ids: list[int],
        input_len: int,
        padding_side: str,
    ) -> dict[str, Any]:
        """Extract `input_ids` and `labels` from a padded data point

        Parameters
        ----------
        ids : list[int]
            (Padded) list of token IDs.
        input_len : int
            Length of input.
        padding_side : str
            Padding side of the tokenizer. Must be 'left' or 'right'.

        Returns
        -------
        dict[str, Any]
            Extracted token IDs.
        """
        assert padding_side in [
            "left",
            "right",
        ], "padding_side must be 'left' or 'right'"
        return ids[:input_len] if padding_side == "right" else ids[-input_len:]

    @staticmethod
    def pack_dps_by_len(lens: list[int], pack_len: int) -> list[list[int]]:
        """Pack data points into groups (each group is a new data point), will be used by PackedDataset, to reduce number of data points in training.
        Given lens of data points, we pack them into groups such that the sum of lens
        in each group is less than `pack_len`. Each group will be considered as a data point (packed data point)
        This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
        There are many algorithms to implement this, but here we use the simple algorithm.
        We will pack/merge a consecutive list of data points until reaching the `pack_len`

        Parameters
        ----------
        lens : list[int]
            Lengths of data points.
        pack_len : int
            Maximum length of packed compuation sequence in token.

        Returns
        -------
        list[list[int]]
            Length groups of packed data points.
        """
        groups = []
        current_packed_len = 0
        current_group = []
        for i in range(len(lens)):
            cur_len = lens[i]
            if cur_len + current_packed_len <= pack_len:
                current_packed_len += lens[i]
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
                current_packed_len = cur_len
        if len(current_group) > 0:
            groups.append(current_group)
        return groups

    @staticmethod
    def pack_dps_FA(
        dps: list[dict[str, list[int]]],
        tokenizer: transformers.PreTrainedTokenizer,
        pack_len: int,
    ) -> dict[str, torch.Tensor]:
        """Pack data points (for Flash Attention)

        Parameters
        ----------
        dps : list[dict[str, list[int]]]
            Data points, each of which should have the following fields at least: `"input_ids"`, `"labels"`, `"attention_mask"`.
        tokenizer : transformers.PreTrainedTokenizer
            (HF) tokenizer.
        pack_len : int
            Maximum length of packed compuation sequence in token.

        Returns
        -------
        dict[str, torch.Tensor]
            Packed data point tensors.
        """
        input_ids = []
        lens = []
        label_ids = []
        attention_mask = []

        for index, item in enumerate(dps):
            input_ids += item["input_ids"]

            labels = list(item["labels"])
            # The first token should not be used to compute loss
            labels[0] = IGNORE_IDX
            label_ids += labels
            lens.append(len(item["input_ids"]))
            attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

        pad_leng = pack_len - len(input_ids)  # padding to model_max_len
        if tokenizer.padding_side == "right":
            input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
            label_ids = label_ids + [IGNORE_IDX for _ in range(pad_leng)]
            attention_mask = attention_mask + [0 for _ in range(pad_leng)]
        else:
            input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
            label_ids = [IGNORE_IDX for _ in range(pad_leng)] + label_ids
            attention_mask = [0 for _ in range(pad_leng)] + attention_mask

        assert len(input_ids) == len(label_ids) == len(attention_mask) == pack_len
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(label_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def stat(self) -> None:
        """Print out the statistics of the packed dataset.
        Original -> Packed:
        1. Number of data/computation sequences;
        2. Average effective length of compution sequences.
        """
        print(
            f"Number of sequences: {len(self.dps)} data/computation sequences -> {len(self.groups)} computation sequences"
        )
        original_avg_len = sum(self.lens) / len(self.lens)

        packed_lens = []
        for group in self.groups:
            lens = [self.lens[index] for index in group]
            packed_lens.append(sum(lens))

        avg_packed_len = sum(packed_lens) / len(packed_lens)
        print(
            f"Average effective length of compution sequences: {original_avg_len} -> {avg_packed_len}"
        )


def get_tokenized_cache_fname(
    ds_name_or_path: str | list[str], tokenizer_name_or_path: str
) -> str:
    ds_pathname = get_pathname_from_name_or_path(ds_name_or_path)
    tokenizer_pathname = get_pathname_from_name_or_path(tokenizer_name_or_path)
    return f"{ds_pathname}-{tokenizer_pathname}-tokenized.pt"


DEF_TOK_CACHE_HOME = os.path.join(PROJ_HOME, "data/cache-tokenized")


def make_supervised_dset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str | list[str],
    query_field: str | list[str] = "query",
    resp_field: str | list[str] = "response",
    tokenized_cache_home: str = DEF_TOK_CACHE_HOME,
    shuffle_seed: int = 42,
    pack_len: int = None,
    prompt_template: str | dict[str, str] | PromptTemplate = "alpaca",
) -> TokenizedSupervisedDataset | PackedDataset:
    """Make dataset for supervised fine-tuning.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        (HF) tokenizer.
    data_path : str | list[str]
        Dataset ID or path.
    query_field : str | list[str], default: "query"
        Field name for query.
    resp_field : str | list[str], default: "response"
        Field name for response.
    tokenized_cache_home : str, default: DEF_TOK_CACHE_HOME
        Path to the tokenized cache home. Useful when repeatedly training on large datasets. None or "" means no cache.
    shuffle_seed : int, default: 42
        Seed for shuffling the dataset before packing. None or negative means no shuffling.
    pack_len : int, default: None
        Maximum length of packed computation sequence in token. None / Non-positive means no packing.
    prompt_template : str | dict[str, str] | PromptTemplate, default: "alpaca"
        ID / File path / PromptTemplate object of prompt template.

    Returns
    -------
    TokenizedSupervisedDataset | PackedDataset
        Dataset ready for input to `Trainer`, containing the following fields at least: `"input_ids"`, `"labels"`, and `"attention_mask"`.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    data_paths = [data_path] if isinstance(data_path, str) else data_path
    query_fields = [query_field] if isinstance(query_field, str) else query_field
    resp_fields = [resp_field] if isinstance(resp_field, str) else resp_field

    tokenized_train_datasets = []
    for data_path, query_field, resp_field in zip(
        data_paths, query_fields, resp_fields
    ):
        # No cache, just tokenize
        if tokenized_cache_home is None or tokenized_cache_home == "":
            tokenized_train_dataset = TokenizedSupervisedDataset.load_from_raw_dset(
                tokenizer=tokenizer,
                data_path=data_path,
                query_field=query_field,
                resp_field=resp_field,
                prompt_template=prompt_template,
            )
        else:  # Cache tokenized datasets
            os.makedirs(tokenized_cache_home, exist_ok=True)
            data_cache_path = os.path.join(
                tokenized_cache_home,
                get_tokenized_cache_fname(data_path, tokenizer.name_or_path),
            )
            logging.info(f"Trying to load data from {data_cache_path} ...")
            try:  # Load from cache if exists
                tokenized_train_dataset = torch.load(data_cache_path)
            # Cache miss -> tokenize the dataset and cache it
            except Exception as e_load:
                logging.debug(e_load)
                try:
                    os.remove(data_cache_path)
                except Exception as e_rm:
                    logging.debug(e_rm)
                if not os.path.exists(data_cache_path):
                    from torch.distributed import barrier

                    # If this is not rank 0, stay here, wait for rank 0 to process the data
                    if local_rank != 0:
                        print(
                            f"[Process {local_rank}] Waiting for main process to prepare the training data"
                        )
                        barrier()  # When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.
                        logging.info(
                            f"[Process {local_rank}] Loading data from {data_cache_path}"
                        )
                        tokenized_train_dataset = torch.load(data_cache_path)
                    else:  # Rank 0 processes the data and saves to `data_cache_path`
                        # The way we read dataset is:
                        # Rank 0 will process the dataset and save the result to data_cache_path, other ranks will read from the data_cache_path

                        tokenized_train_dataset = (
                            TokenizedSupervisedDataset.load_from_raw_dset(
                                tokenizer=tokenizer,
                                data_path=data_path,
                                query_field=query_field,
                                resp_field=resp_field,
                                prompt_template=prompt_template,
                            )
                        )

                        torch.save(tokenized_train_dataset, data_cache_path)

                        logging.info(
                            f"process: {local_rank} finishes processing data and saves to {data_cache_path}"
                        )
                        world_size = int(os.environ.get("WORLD_SIZE", 1))
                        if world_size > 1:
                            barrier()
                else:  # Cache existing
                    logging.info(
                        f"[Process {local_rank}] Loading data from {data_cache_path}"
                    )
                    tokenized_train_dataset = torch.load(data_cache_path)

        tokenized_train_datasets.append(tokenized_train_dataset)

    train_dataset = tokenized_train_datasets[0]
    train_dataset.concat(tokenized_train_datasets[1:])
    del tokenized_train_datasets

    # Shuffle the dataset if necessary
    if shuffle_seed is not None and shuffle_seed >= 0:
        train_dataset.shuffle(seed=shuffle_seed)
    else:  # We disable shuffling by default but this is possibly not desired
        if len(data_paths) > 1:
            logging.warning(
                f"Concatenating {len(data_paths)} datasets, but `shuffle_seed` is not set."
            )  # Except for curriculum learning, where we need specific order

    # Pack the dataset if specified
    if pack_len is None or pack_len <= 0:
        logging.debug(
            "No packing is applied to the dataset. Padding the dataset to the same length ..."
        )
        train_dataset.pad()
    else:
        logging.debug(f"Packing dataset with pack_len = {pack_len} ...")
        train_dataset = PackedDataset(
            dataset=tokenized_train_dataset,
            tokenizer=tokenizer,
            pack_len=pack_len,
        )

    # For consistency checking
    logging.info(f"len(train_dataset): {len(train_dataset)}")
    logging.info(f"[Process {local_rank}] train_dataset[-1]: {train_dataset[-1]}")

    return train_dataset


MODEL_MODULES2PATCH = {
    "llama": transformers.models.llama.modeling_llama,
    "mistral": transformers.models.mistral.modeling_mistral,
    "mixtral": transformers.models.mixtral.modeling_mixtral,
}


def monkey_patch4pack(
    name_or_cls_or_obj: str | type | transformers.PretrainedConfig,
) -> None:
    """Monkey patch the modeling module for packing. Must be called before instantiating the model.

    Parameters
    ----------
    name_or_cls_or_obj : str | type | transformers.PretrainedConfig
        Name containing the model name like "llama" / "mistral" / ...
    """

    if isinstance(name_or_cls_or_obj, str):
        name = name_or_cls_or_obj
    elif isinstance(name_or_cls_or_obj, type):
        name = name_or_cls_or_obj.__name__
    else:
        name = type(name_or_cls_or_obj).__name__

    name = name.lower()

    def get_max_seqlen_in_batch(attention_mask):
        max_num = torch.max(attention_mask)
        # attention_mask: B x N
        counts = []
        for i in range(1, max_num + 1):
            # shape: B, count length of data point maksed with i
            counts.append(torch.sum(attention_mask == i, axis=-1))
        result = torch.stack(counts, axis=1)
        result = result.flatten()
        return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)

    def get_unpad_data(attention_mask):
        # attention_mask.sum(dim=-1, dtype=torch.int32)
        seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

    patched = False
    for model_type, model_module in MODEL_MODULES2PATCH.items():
        if model_type in name:
            model_module._get_unpad_data = get_unpad_data
            patched = True
            break

    if not patched:
        logging.error(
            f"Model {name} not supported by monkey patching for packing. For now, packing only supports models: Mistral, Llama, Mixtral"
        )
        import sys

        sys.exit(1)
