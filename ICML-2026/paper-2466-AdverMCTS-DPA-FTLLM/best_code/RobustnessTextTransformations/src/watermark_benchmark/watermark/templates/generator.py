from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor
from transformers import PreTrainedTokenizerBase

from watermark_benchmark.utils.classes import VerifierOutput
from watermark_benchmark.watermark.templates.random import BaseRandomness
from watermark_benchmark.watermark.templates.verifier import Verifier


class Watermark(ABC):
    """
    Abstract base class for watermarking algorithms.

    Attributes:
        rng: An instance of a random number generator.
        verifiers: A list of verification algorithms.
        tokenizer: An instance of the tokenizer.
        temp: A temperature parameter used in the watermarking process.

    Methods to implement:
        * _process
        Use the logits, previous tokens and generation ids to generate the next token watemarked logits

    Other methods:
        * process
        wrapper around _process

        * verify:
        calls the verification procedure in each of the verifiers

        * verify_text:
        encodes the list of texts and calls verify
    """

    def __init__(
        self,
        rng: BaseRandomness,
        verifiers: Union[List[Verifier], Verifier],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        temp: float = 1.0,
    ):
        self.rng = rng
        self.verifiers = verifiers
        self.tokenizer = tokenizer
        self.temp = temp
        if isinstance(self.verifiers, Verifier):
            self.verifiers = [self.verifiers]

    def process(
        self,
        logits: Tensor,
        previous_tokens: Tensor,
        ids: Tensor,
    ) -> Tensor:
        """
        Abstract method for processing logits.

        Args:
            logits: A tensor of logits.
            previous_tokens: A tensor of previous tokens.
            ids: A tensor of ids.

        Returns:
            A tensor of processed logits.
        """
        return self._process(logits, previous_tokens, ids)

    @abstractmethod
    def _process(self, logits, previous_tokens, ids):
        pass

    def reset(self):
        """
        Resets the random number generator.
        """
        self.rng.reset()

    def verify(
        self, tokens, skip_edit=False, meta=None, **kwargs
    ) -> Dict[Tuple[float, str, str], VerifierOutput]:
        """
        Verifies the watermark in the given tokens.

        Args:
            tokens: A tensor of tokens.
            index: The index of the token to verify.
            exact: Whether to use exact matching.
            skip_edit: Whether to skip edit distance verification.

        Returns:
            A list of tuples containing the verifier ID and the verification result.
        """
        rtn = {}
        for v in self.verifiers:
            if "method" in v.__dict__ and v.method != "regular":
                kwargs["exact"] = False
            rtn[v.id()] = v.verify(tokens, meta=meta, **kwargs)
        return rtn

    def verify_text(self, text, skip_edit=False, meta=None, **kwargs):
        """
        Verifies the watermark in the given text.

        Args:
            text: The text to verify.
            index: The index of the token to verify.
            exact: Whether to use exact matching.
            skip_edit: Whether to skip edit distance verification.

        Returns:
            A list of tuples containing the verifier ID and the verification result.
        """
        tokens = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).to(self.rng.device)
        return self.verify(tokens, skip_edit=skip_edit, meta=meta, **kwargs)
