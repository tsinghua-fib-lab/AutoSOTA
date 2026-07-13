import os
import re
import sys
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Union, Any, Optional, Tuple, List
from tqdm.auto import tqdm
from torch.utils.data import Dataset

import urllib.request

from autoregltl.ltl.vocab import MergedLTLVocab, EncDecVocab
from autoregltl.ltl.parser import ParseError, ltl_formula, ltl_trace


def download_dataset(dataset_name, split, dataset_dir):
    url_lookup = {
        'na-5-ts-35-nf-1m-lbt-sat': 'https://storage.googleapis.com/deepltl_data/data/sat/na-5-ts-35-nf-1m-lbt-sat/',
        'na-5-ts-35-50-nf-20k-lbt-sat': 'https://storage.googleapis.com/deepltl_data/data/sat/na-5-ts-35-50-nf-20k-lbt-sat/',
    }

    # Check if split already exists
    split_file = os.path.join(dataset_dir, split + '.txt')
    if not os.path.isfile(split_file):
        if dataset_name not in url_lookup:
            print("Cannot download this dataset:", dataset_name)
            return
        os.makedirs(dataset_dir, exist_ok=True)
        print(f'Downloading dataset {dataset_name}/{split}')
        urllib.request.urlretrieve(url_lookup[dataset_name] + split + '.txt', split_file)


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def read_pairs(
    filename,
    max_formula_length,
    max_trace_length,
    max_samples = None,
    min_aps = None,
    max_aps = None,
):
    """
    Expects data file to have formula\ntrace\n format
    """
    if min_aps is not None or max_aps is not None:
        min_aps = min_aps if min_aps is not None else 0
        max_aps = max_aps if max_aps is not None else float('inf')
        def ap_filter(formula):
            aps = len({f for f in formula.replace("xor", "") if f.islower()})
            return aps < min_aps or aps > max_aps
    else:
        ap_filter = lambda x: False

    if max_formula_length >= 0:
        def formula_filter(formula):
            return len(formula.replace("<->", "=").replace("xor", "^")) > max_formula_length
    else:
        formula_filter = lambda x: False

    filtered = 0
    pairs = []
    with open(filename, 'r') as file:  # expect formula\ntrace\n format
        for formula_line in file:
            if formula_line == '\n':
                break
            formula_line = formula_line.strip()
            trace_line = next(file).strip()  # get second line
            if formula_filter(formula_line) or \
               (max_trace_length >= 0 and len(trace_line) > max_trace_length) or \
               ap_filter(formula_line):
                filtered += 1
                continue
            pairs.append((trace_line, formula_line))

    if max_samples is not None:
        pairs = pairs[:max_samples]

    print("Filtered out", filtered, "samples")
    return pairs


class RawLTLDataset(SeqDataset):
    """Dataset that consists of pairs of a LTL formula and a satisfying trace."""
    def __init__(
        self, 
        filename,
        max_formula_length,
        max_trace_length,
        max_samples = None,
        min_aps = None,
        max_aps = None,
    ):
        pairs = read_pairs(filename, max_formula_length, max_trace_length, max_samples, min_aps, max_aps)
        super().__init__(pairs)


class DecoderLTLDataset(SeqDataset):
    def __init__(
        self, 
        filename,
        vocab: MergedLTLVocab,
        max_formula_length,
        max_trace_length,
        max_samples = None,
        min_aps = None,
        max_aps = None,
    ):
        pairs = read_pairs(filename, max_formula_length, max_trace_length, max_samples, min_aps, max_aps)

        def process_pair(trace_str, formula_str):
            trace = vocab.encode_trace(trace_str)
            formula = vocab.encode_ltl(formula_str)
            # No need to feed EOS in input
            input_ids = trace + formula
            labels = ([-100] * (len(trace)-1)) + formula + [vocab.eos_id]
            return torch.tensor(input_ids), torch.tensor(labels)

        data = [process_pair(*pair) for pair in tqdm(pairs, desc=os.path.basename(filename))]
        super().__init__(data)


@dataclass
class DecoderLTLCollator:
    pad_token_id: int = 0
    ignore_label: int = -100

    def __call__(self, instances):
        input_ids, labels = zip(*instances)  # Unzip
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_label)
        return {"input_ids": input_ids, "labels": labels}


class EncDecLTLDataset(SeqDataset):
    def __init__(
        self, 
        filename,
        vocab: EncDecVocab,
        max_formula_length,
        max_trace_length,
        tree_pos_enc = False,
        max_samples = None,
        min_aps = None,
        max_aps = None,
        pairs = None,
    ):
        if pairs is None:
            pairs = read_pairs(filename, max_formula_length, max_trace_length, max_samples, min_aps, max_aps)

        if isinstance(vocab, EncDecVocab):
            def process_pair0(trace_str, formula_str):
                # Input is ltl, output is trace
                trace = vocab.inp.encode(trace_str, prepend_start_token=False)
                formula = vocab.out.encode(formula_str, prepend_start_token=False)
                return torch.tensor(formula), torch.tensor(trace)
        elif isinstance(vocab, MergedLTLVocab):
            def process_pair0(trace_str, formula_str):
                # Input is ltl, output is trace
                trace = vocab.encode_trace(trace_str, eos=True)
                formula = vocab.encode_ltl(formula_str, eos=True)
                return torch.tensor(formula), torch.tensor(trace)
        else:
            raise ValueError(f"Unsupported vocab type: {type(vocab)}")
        
        if tree_pos_enc:
            def process_pair(trace_str, formula_str):
                formula = ltl_formula(formula_str, 'network-polish')
                position_list = formula.binary_position_list(format='lbt', add_first=True)
                # pad to max length
                max_length = max([len(l) for l in position_list])
                padded_position_list = [l + [0] * (max_length - len(l)) for l in position_list]
                pe = torch.tensor(padded_position_list, dtype=torch.float32)
                return process_pair0(trace_str, formula_str) + (pe,)
        else:
            process_pair = process_pair0

        if filename is not None:
            iterator = tqdm(pairs, desc=os.path.basename(filename))
        else:
            iterator = pairs
        data = [process_pair(*pair) for pair in iterator]
        super().__init__(data)


@dataclass
class EncDecLTLCollator:
    """
    A collate function that pads the input sequences to the longest sequence in the batch.
    """
    d_embed_enc: Optional[int] = None

    def __call__(self, batch):
        """
        Args:
            batch: list of (input, target) or (input, target, positional_encoding)

        Returns:
            A dictionary:
                input_ids: int tensor with shape (batch_size, input_length)
                target_ids: int tensor with shape (batch_size, target_length)
                pe: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding
        """
        # Pad by adding zeros to the end
        max_input_length = max(map(lambda x: x[0].size(0), batch))
        xs = torch.stack([F.pad(x[0], (0, max_input_length - x[0].size(0)), "constant", 0) for x in batch], dim=0)
        max_target_length = max(map(lambda x: x[1].size(0), batch))
        ys = torch.stack([F.pad(x[1], (0, max_target_length - x[1].size(0)), "constant", 0) for x in batch], dim=0)
        pe = None
        if self.d_embed_enc is not None:
            pe = torch.stack([F.pad(x[2], (0, self.d_embed_enc - x[2].size(-1), 0, max_input_length - x[2].size(-2)), "constant", 0) for x in batch], dim=0)
        return {
            "input_ids": xs,
            "target_ids": ys,
            "pe": pe,
        }


def get_dataset_vocab(args, config=None):
    if args.ds_name is None:
        raise ValueError('No dataset is specified')

    aps = ['a', 'b', 'c', 'd', 'e']
    if args.ds_name == 'prop-60-no-derived':
        aps += ['f', 'g', 'h', 'i', 'j']
    
    # Match the 6 in ltlx-6ap-etc
    # \b is word boundary
    if matches := re.findall(r'\b(\d+)ap\b', args.ds_name):
        ap_count = int(matches[0])
        aps = [chr(i) for i in range(ord('a'), ord('z')+1)][:ap_count]
    
    if args.vocab_aps:
        if args.vocab_aps < len(aps):
            raise ValueError(f"This dataset requires at least {len(aps)} aps")
        elif args.vocab_aps > len(aps):
            aps = [chr(i) for i in range(ord('a'), ord('z')+1)][:args.vocab_aps]

    kwargs = {} if args.decoder_only else {"use_start_token": True, "use_pad_token": True}
    vocab = MergedLTLVocab(aps=aps, **kwargs)
    print(f"[get_dataset_vocab] Vocab size: {vocab.size()} (merged tokens)")
    return vocab


def get_dataset(args, split, dataset_class, **kwargs):
    """
    Returns:
        the datasets corresponding to the given split according to args
    """
    # Dataset specification
    if args.ds_name is None:
        sys.exit('No dataset specified\n')
    else:
        if args.ds_name == 'ltl-35' or args.ds_name == 'prop-35':
            max_formula_length = 35
            dataset_name = 'na-5-ts-35-nf-1m-lbt-sat'
        elif args.ds_name == 'ltl-50-test' or args.ds_name == 'prop-50-test':
            max_formula_length = 50
            if split != 'test':
                sys.exit(f'Dataset {args.ds_name} can only be used in test mode\n')
            dataset_name = 'na-5-ts-35-50-nf-20k-lbt-sat'
        elif os.path.exists(os.path.join(args.data_dir, args.ds_name)):
            max_formula_length = -1
            dataset_name = args.ds_name
        else:
            sys.exit(f'{args.ds_name} is not a valid dataset\n')

    # data_dir = data_dir if data_dir is not None else os.path.join(os.path.dirname(__file__), '../../../data')
    dataset_dir = os.path.join(args.data_dir, dataset_name)

    download_dataset(dataset_name, split, dataset_dir)

    dataset_args = {
        'max_formula_length': max_formula_length,
        'max_trace_length': args.max_trace_length,
        'min_aps': args.exact_aps if args.exact_aps is not None else args.min_aps,
        'max_aps': args.exact_aps if args.exact_aps is not None else args.max_aps,
    }
    dataset_args.update(kwargs)

    return dataset_class(os.path.join(dataset_dir, split + '.txt'), **dataset_args)