import glob
import json
import math
import random
import sys

import numpy as np
import torch
from dahuffman import HuffmanCodec
from tqdm import tqdm
from transformers import AutoTokenizer


class Binarization:
    def __init__(
        self,
        tokenizer,
        devices="cpu",
        use_huffman_coding=True,
        huffman_coding_path=None,
        corpus=None,
        save_path="huffman_coding.tsv",
    ):
        # Check consistency of arguments
        if use_huffman_coding and (
            huffman_coding_path is None and corpus is None
        ):
            raise Exception(
                "In order to use Huffman coding, you need to either specify an existing huffman coding path, or a corpus of text"
            )

        # Init variables
        self.tokenizer = tokenizer
        if type(devices) == list:
            self.devices = devices
            self.device = devices[0]
        else:
            self.devices = [devices]
            self.device = devices

        vocab = self.tokenizer.get_vocab()
        self.V = len(vocab)
        inv_vocab = {vocab[k]: k for k in vocab}
        self.huffman = use_huffman_coding

        if not use_huffman_coding:
            self.L = int(math.ceil(np.log2(self.V)))
            self.avg_bit_length = self.L
            self.token_to_bits = [self.int_to_bits(i) for i in range(self.V)]
        elif huffman_coding_path is not None:
            # Load existing coding
            with open(huffman_coding_path) as infile:
                self.token_to_bits = [
                    tuple([int(b) for b in l.split(" ")])
                    for l in infile.read().split("\n")
                    if len(l)
                ]
                self.avg_bit_length = int(self.token_to_bits[-1][0])
                self.token_to_bits = self.token_to_bits[:-1]
            self.L = len(self.token_to_bits[0])
        else:
            (
                self.token_to_bits,
                self.avg_bit_length,
            ) = Binarization._build_huffman(corpus, tokenizer, save_path)
            self.avg_bit_length = int(self.avg_bit_length)
            self.L = len(self.token_to_bits[0])

        self.bits_to_token = {
            b: idx for idx, b in enumerate(self.token_to_bits)
        }
        self.bits_to_text = {
            b: inv_vocab[t] for b, t in self.bits_to_token.items()
        }
        self.text_to_bits = {k: self.token_to_bits[vocab[k]] for k in vocab}
        self.token_to_bits_tensor = {
            device: torch.tensor(self.token_to_bits).to(device)
            for device in self.devices
        }

    def get_token_to_bits_tensor(self, device):
        if device not in self.token_to_bits_tensor:
            self.token_to_bits_tensor[device] = self.token_to_bits_tensor[
                self.device
            ].to(device)
        return self.token_to_bits_tensor[device]

    def is_leaf(self, encodings):
        if not self.huffman:
            if (
                (type(encodings) == torch.Tensor and len(encodings.shape) == 1)
                or type(encodings) == list
                and type(encodings[0]) != list
            ):
                return False
            elif type(encodings) == torch.Tensor and len(encodings.shape) > 1:
                return (
                    torch.zeros((encodings.shape[0]))
                    .to(encodings.device)
                    .bool()
                )
            else:
                return [False for _ in range(len(encodings))]
        else:
            if type(encodings) == list and type(encodings[0]) != list:
                return tuple(encodings) in self.bits_to_token
            elif type(encodings) == list:
                return [tuple(e) in self.bits_to_token for e in encodings]
            else:
                return (
                    torch.Tensor([self.is_leaf(encodings.tolist())])
                    .squeeze()
                    .to(encodings.device)
                )

    @staticmethod
    def _build_huffman(corpus, tokenizer, save_path=None, article_count=100000):
        # Get frequencies of tokens in corpus and build huffman encoding

        counts = torch.zeros((len(tokenizer)))
        data = []
        for filename in glob.glob("{}/*.json".format(corpus)):
            with open(filename) as infile:
                data.extend(json.load(infile))

        articles = random.sample(data, article_count)

        for txt in tqdm(
            articles, desc="Encoding wikipedia articles", total=article_count
        ):
            encoded = tokenizer(txt, return_tensors="pt").input_ids.squeeze()
            counts += torch.nn.functional.one_hot(encoded, len(tokenizer)).sum(
                dim=0
            )

        freq_dict = {k: counts[k].item() for k in range(len(tokenizer))}

        hc = HuffmanCodec.from_frequencies(
            freq_dict, eof=tokenizer.eos_token_id
        )
        codewords_str = [
            (symbol, bin(val)[2:].rjust(bits, "0"))
            for symbol, (bits, val) in hc.get_code_table().items()
        ]
        codewords_str = sorted(codewords_str, key=lambda x: x[0])
        max_len = max(len(v[1]) for v in codewords_str)
        codewords = [
            tuple(
                [int(b) for b in v[1]]
                + [-1 for _ in range(max_len - len(v[1]))]
            )
            for v in codewords_str
        ]

        abl = Binarization._get_average_length(
            articles, tokenizer, codewords, save_path
        )

        if save_path:
            with open(save_path, "w") as outfile:
                outfile.write(
                    "\n".join(" ".join(str(b) for b in l) for l in codewords)
                    + f"\n{abl}\n"
                )

        return codewords, abl

    @staticmethod
    def _get_average_length(text, tokenizer, codewords, save_path=None):
        codewords = torch.tensor(codewords).cuda()
        total_bits = 0
        total_tokens = 0
        for txt in tqdm(text, desc="Encoding text", total=len(text)):
            encoded = tokenizer(txt, return_tensors="pt").input_ids.squeeze()
            total_tokens += len(encoded)
            encodings = codewords[encoded.cuda()]
            mask = encodings >= 0
            total_bits += len(encodings.flatten()[mask.flatten()])

        val = int(total_bits // total_tokens)
        if save_path:
            with open(save_path, "a") as outfile:
                outfile.write(str(val) + "\n")

        return val

    def int_to_bits(self, i):
        raw_binary = "{0:b}".format(i)
        raw_binary = list(reversed(raw_binary))
        if len(raw_binary) < self.L:
            raw_binary += ["0" for _ in range(self.L - len(raw_binary))]

        return tuple([int(i) for i in raw_binary])

    def bits_to_int(self, b):
        return int("".join(str(i) for i in reversed(b)), 2)

    def to_bit(self, tokens):
        if type(tokens) == torch.Tensor:
            t = self.get_token_to_bits_tensor(tokens.device)
            return t[tokens]

        if type(tokens) == str:
            return self.text_to_bits[tokens]

        if type(tokens) == int:
            return self.token_to_bits[tokens]

        if type(tokens[0]) == str:
            return [self.text_to_bits[v] for v in tokens]

        return [self.token_to_bits[v] for v in tokens]

    def to_token(self, bits):
        if type(bits) == torch.Tensor and len(bits.shape) > 1:
            return torch.tensor(
                [self.bits_to_token[tuple(v)] for v in bits.tolist()]
            ).to(bits.device)
        elif type(bits) == torch.Tensor:
            return torch.tensor([self.bits_to_token[tuple(bits.tolist())]]).to(
                bits.device
            )

        if type(bits) == tuple:
            return self.bits_to_token[bits]
        else:
            return [self.bits_to_token[b] for b in bits]


def generate_huffman_coding():
    tokenizer, corpus, output_path = tuple(sys.argv[1:4])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer, fast=False, add_eos_token=True
    )
    codewords, abl = Binarization._build_huffman(corpus, tokenizer, output_path)
    print("Average length: {}".format(abl))
    return codewords


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    binarization = Binarization(tokenizer)

    # Test that conversion functions work
    for i in [random.randint(0, len(tokenizer) - 1) for _ in range(10)]:
        assert binarization.bits_to_int(binarization.int_to_bits(i)) == i

    # Test that the to-token and to-bit functions are inverses
    bits_lists = [
        binarization.int_to_bits(random.randint(0, len(tokenizer) - 1))
        for _ in range(10)
    ]
    assert all(
        [binarization.to_bit(binarization.to_token(b)) == b for b in bits_lists]
    )
    assert all(
        [
            bits_lists[i] == v
            for i, v in enumerate(
                binarization.to_bit(binarization.to_token(bits_lists))
            )
        ]
    )

    # Test on Tensor objects
    token_tensors = torch.randint(0, len(tokenizer), (10,))
    assert all(
        [
            binarization.to_token(binarization.to_bit(t)).item() == t.item()
            for t in token_tensors
        ]
    )

    bit_tensors = torch.stack([torch.tensor(b) for b in bits_lists])
    assert torch.equal(
        binarization.to_bit(binarization.to_token(bit_tensors)), bit_tensors
    )
