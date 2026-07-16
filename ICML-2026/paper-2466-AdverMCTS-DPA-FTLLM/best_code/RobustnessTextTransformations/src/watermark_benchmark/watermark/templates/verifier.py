import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import hash_cpp
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from watermark_benchmark.utils import adapt_arguments
from watermark_benchmark.utils.classes import VerifierOutput
from watermark_benchmark.watermark.templates.random import (
    BaseRandomness,
    EmbeddedRandomness,
    Randomness,
)


class Verifier(ABC):
    """
    Abstract base class for watermark verifiers.

    Attributes:
        * rng: An instance of a random number generator.
        * pvalue: The p-value of the verifier.
        * tokenizer: An instance of the tokenizer used for the LLM.

    Methods to implement:
        * _verify:
        Verifies if a given sequence of tokens contains a watermark. Returns a VerifierOutput object,
        that contains the detection pvalue for each subsequence of the input tokens.

    Other methods:
        * verify:
        wrapper around _verify

        * id:
        returns a unique identifier for the verifier
    """

    def __init__(
        self,
        rng: Optional[BaseRandomness] = None,
        pvalue: Optional[float] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Initializes the Verifier object.
        """
        self.pvalue = pvalue
        self.rng = rng
        self.tokenizer = tokenizer

    def verify(
        self, tokens: torch.Tensor, index: int = 0, **kwargs
    ) -> VerifierOutput:
        """
        Verifies if a given sequence of tokens contains a watermark.
        """
        if self.rng is not None:
            tokens = tokens.to(self.rng.device)
        tokens = tokens.reshape(-1)

        if not len(tokens.shape) or not len(tokens) or tokens.nelement() < 2:
            return VerifierOutput()

        kwargs["index"] = index
        return adapt_arguments(self._verify, kwargs, tokens)

    @abstractmethod
    def _verify(self, tokens: torch.Tensor, index: int = 0) -> VerifierOutput:
        """
        Verifies if the given text contains a watermark.

        Args:
            tokens (torch.Tensor): The text to verify.
            index (int, optional): The index of the text. Defaults to 0. Used to recover the secret key in the rng.

        Returns:
            list: A list of tuples containing the verification results.
        """

    def id(self) -> Tuple[float, str, str]:
        """
        Returns a unique identifier for the verifier.

        :return: A tuple containing the p-value, the type of the verifier, and the version of the verifier.
        :rtype: tuple
        """
        return (self.pvalue, "theoretical", "standard")


class EmpiricalVerifier(Verifier):
    """
    EmpiricalVerifier is a subclass of Verifier that implements the empirical verification method.
    It uses a score matrix to detect watermarks in a given text.
    """

    @abstractmethod
    def __init__(
        self,
        rng: Randomness,
        pvalue: float,
        tokenizer: PreTrainedTokenizerBase,
        method: str,
        gamma: float,
        log: bool,
    ):
        super().__init__(rng, pvalue, tokenizer)
        self.method = method
        self.precomputed = False
        self.gamma_edit = (
            gamma if not log else (np.log(gamma) if gamma > 0 else -math.inf)
        )
        self.rand_size = self.rng.vocab_size
        self.precomputed_results = None

    @abstractmethod
    def score_matrix(self, tokens, random_values, index=0, meta=None):
        pass

    @abstractmethod
    def random_score_matrix(
        self, tokens, random_shape, shared_randomness, index=0, meta=None
    ):
        pass

    def detect(self, scores):
        """
        Detects the watermark in the given score matrix using the specified method.

        Args:
            scores (torch.Tensor): The score matrix.

        Returns:
            torch.Tensor: The detected watermark.
        """
        if self.method == "regular":
            A = self.regular_distance(scores)
        else:
            A = self.levenstein_distance(scores)
        return A.min(axis=0)

    def regular_distance(self, scores):
        """
        Computes the regular distance between the scores.

        Args:
            scores (torch.Tensor): The score matrix.

        Returns:
            torch.Tensor: The computed distance.
        """
        KL, SL = scores.shape
        if not isinstance(self.rng, EmbeddedRandomness):
            indices = (
                torch.vstack(
                    (
                        (
                            (
                                torch.arange(KL)
                                .reshape(-1, 1)
                                .repeat(1, SL)
                                .flatten()
                                + torch.arange(SL).repeat(KL)
                            )
                            % KL
                        ),
                        torch.arange(SL).repeat(KL) % SL,
                    )
                )
                .t()
                .reshape(KL, -1, 2)
                .to(scores.device)
            )
            rslt = scores[indices[:, :, 0], indices[:, :, 1]].cumsum(axis=1)
        else:
            rslt = (
                scores[torch.arange(SL), torch.arange(SL)]
                .cumsum(0)
                .unsqueeze(0)
            )
        return rslt

    def levenstein_distance(self, scores):
        """
        Computes the Levenstein distance between the scores.

        Args:
            scores (torch.Tensor): The score matrix.

        Returns:
            torch.Tensor: The computed distance.
        """
        KL, SL = scores.shape
        if not isinstance(self.rng, EmbeddedRandomness):
            container = (
                torch.zeros((KL, SL + 1, SL + 1)).float().to(self.rng.device)
            )
        else:
            container = (
                torch.zeros((1, SL + 1, SL + 1)).float().to(self.rng.device)
            )

        # Set initial values
        container[:, 0, :] = (
            (torch.arange(SL + 1) * self.gamma_edit)
            .to(self.rng.device)
            .unsqueeze(0)
            .expand(container.shape[0], -1)
        )
        container[:, :, 0] = (
            (torch.arange(SL + 1) * self.gamma_edit)
            .to(self.rng.device)
            .unsqueeze(0)
            .expand(container.shape[0], -1)
        )

        # Compute
        container = hash_cpp.levenshtein(scores, container, self.gamma_edit)
        return container[:, torch.arange(1, SL + 1), torch.arange(1, SL + 1)]

    def pre_compute_baseline(self, max_len=1024, runs=200):
        """
        Pre-computes a set of baseline scores to speed up the verification process.

        Args:
            max_len (int, optional): The maximum length of the text. Defaults to 1024.
            runs (int, optional): The number of runs to perform. Defaults to 200.
        """
        self.precomputed_results = torch.zeros((runs, max_len)).to(
            self.rng.device
        )
        tokens = torch.randint(0, self.rng.vocab_size, (max_len,))
        if not isinstance(self.rng, EmbeddedRandomness):
            shared_randomness = (
                torch.arange(self.rng.key_len)
                .repeat(1 + max_len // self.rng.key_len)[:max_len]
                .to(self.rng.device)
            )
            L = self.rng.key_len
        else:
            shared_randomness = torch.arange(max_len).to(self.rng.device)
            L = max_len

        for run in range(runs):
            scores = self.random_score_matrix(
                tokens, (1, L, self.rng.vocab_size), shared_randomness
            )
            self.precomputed_results[run] = self.detect(scores)[0]

        self.precomputed = True

    def _verify(self, tokens, index=0, exact=False, meta=None):
        """
        Verifies if the given text contains a watermark.

        Args:
            tokens (torch.Tensor): The text to verify.
            index (int, optional): The index of the text. Defaults to 0.
            exact (bool, optional): Whether to perform an exact verification. Defaults to False.

        Returns:
            list: A list of tuples containing the verification results.
        """
        verifier_output = VerifierOutput()
        tokens = tokens.unsqueeze(0)

        if not isinstance(self.rng, EmbeddedRandomness):
            xi = self.rng.xi[index].to(self.rng.device).unsqueeze(0)
            scores = self.score_matrix(tokens, xi, index=index, meta=meta)
        else:
            if self.rand_size > 1:
                randomness = torch.cat(
                    tuple(
                        self.rng.rand_range(
                            self.rng.get_seed(tokens[:, :i], [index]),
                            self.rand_size,
                        )
                        for i in range(tokens.shape[-1])
                    ),
                    axis=0,
                ).unsqueeze(0)
            else:
                randomness = torch.cat(
                    tuple(
                        self.rng.rand_index(
                            self.rng.get_seed(tokens[:, :i], [index]), 0
                        ).reshape(1, 1)
                        for i in range(tokens.shape[-1])
                    ),
                    axis=0,
                ).unsqueeze(0)

            xi = randomness
            scores = self.score_matrix(
                tokens, randomness, index=index, meta=meta
            )

        if scores is None:
            return verifier_output

        test_result = self.detect(scores)[0]
        p_val = torch.zeros_like(test_result).to(self.rng.device)

        if exact:
            rc = 100
            # Before simlating random seeds, we need to figure out which tokens will share the same randomness
            if not isinstance(self.rng, EmbeddedRandomness):
                shared_randomness = torch.arange(self.rng.key_len)
            else:
                _, shared_randomness = xi[0, :, 0].unique(return_inverse=True)
                shared_randomness = shared_randomness.to(self.rng.device)

            # rv = torch.cuda.FloatTensor(100, xi.shape[1], tokens.shape[-1]).uniform_(0,1).to(self.rng.device)
            for _ in range(rc):
                scores_alt = self.random_score_matrix(
                    tokens, xi.shape, shared_randomness, index=index, meta=meta
                )
                null_result = self.detect(scores_alt)[0]
                p_val += null_result < test_result

        else:
            rc = 100
            if not self.precomputed:
                self.pre_compute_baseline()
            null_result = self.precomputed_results[
                torch.randperm(self.precomputed_results.shape[0])[:100].to(
                    self.rng.device
                ),
                : test_result.shape[-1],
            ]
            if null_result.shape[-1] < test_result.shape[-1]:
                test_result = test_result[: null_result.shape[-1]]
            p_val = (null_result < test_result).sum(axis=0)

        for idx, val in enumerate(p_val.cpu().numpy()):
            verifier_output.update(idx, val / rc)

        self.rng.reset()
        return verifier_output

    def id(self):
        """
        Returns the ID of the verifier.

        Returns:
            tuple: The ID of the verifier.
        """
        return (self.pvalue, "empirical", self.method)
