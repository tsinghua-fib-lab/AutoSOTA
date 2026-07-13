import torch
from transformers import LogitsProcessor

from typing import Dict, Union, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from autoregltl.ltl.vocab import MergedLTLVocab, EncDecVocab
from autoregltl.ltl.vocab import EOS_TOKEN, PAD_TOKEN, START_TOKEN

class LTLSyntaxEnforcerConfig(LogitsProcessor):
    def __init__(self, vocab):
        self.operand_count = {
            # Logical
            '!': 1,
            '&': 2,
            # Unused
            '|': 2,

            # Temporal
            'X': 1,
            'U': 2,
            # Unused
            'F': 1,
            'G': 1,
            'R': 2,
            'W': 2,
            'M': 2,
        }

        if isinstance(vocab, MergedLTLVocab):
            self.vocab = vocab
            self.vocab_size = self.vocab.trace_size()
            self.operand_count[EOS_TOKEN] = 1  # Ignoring pad tokens
        elif isinstance(vocab, EncDecVocab):
            self.vocab = vocab.out
            self.vocab_size = self.vocab.size()
            # Ignoring these tokens
            self.operand_count[PAD_TOKEN] = 1
            self.operand_count[START_TOKEN] = 1
        else:
            raise ValueError("Unsupported vocab type")
        self.token_list = self.vocab.token_list

        self.operand_count.update({op: 0 for op in self.vocab.aps + self.vocab.consts})
    
    def enforce(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Enforces the LTL syntax constraints on the output logits without keeping state.

        Args:
            input_ids: The input token IDs.
            logits: The output logits before applying the syntax
        """
        batch_size = input_ids.size(0)
        def get_expected_statements(tokens):
            expected = 1
            start = True
            for token in tokens:
                if token >= self.vocab_size:  # Ignore trace tokens
                    continue
                char = self.token_list[token]
                if expected == 0:
                    # Weird stuff happens in HF transformers' beam search implementation violating this:
                    # assert char == EOS_TOKEN
                    break
                operand_count = self.operand_count[char]
                expected += operand_count - 1
            return expected
        
        for i, tokens in enumerate(input_ids.tolist()):
            expected = get_expected_statements(tokens)
            if expected == 0:
                self.vocab.force_eos(logits, i)
            else:
                self.vocab.disallow_eos(logits, i)
        
        return logits

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return self.enforce(input_ids, logits)