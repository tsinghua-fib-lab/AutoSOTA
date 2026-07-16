import math
import random

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from watermark_benchmark.utils.classes import VerifierOutput
from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import (
    EmpiricalVerifier,
    Verifier,
)


class ExponentialGenerator(Watermark):
    """
    A watermark generator that uses an exponential distribution to select the next token. See Aaronson et al. (2023) for more details.

    Args:
        rng (RandomNumberGenerator): A random number generator.
        verifiers (List[Verifier]): A list of verifiers to check the watermark.
        tokenizer (Tokenizer): A tokenizer to convert text to tokens.
        temp (float): A temperature value to control the randomness of the selection.
        skip_prob (float): A probability value to skip the watermark generation process.
    """

    def __init__(self, rng, verifiers, tokenizer, temp, skip_prob):
        super().__init__(rng, verifiers, tokenizer, temp)
        self.skip_prob = skip_prob

    def _process(self, logits, previous_tokens, ids):

        # Truncate unused logits. Return as is with skip probability
        local_logits = logits[:, : self.rng.vocab_size] / self.temp
        if random.random() < self.skip_prob:
            return logits

        # Compute probabilities and get random values
        probs = F.softmax(local_logits, dim=-1)
        hash_values = self.rng.rand_range(
            self.rng.get_seed(previous_tokens, ids=ids),
            self.rng.vocab_size,
            device=probs.device,
        )
        hash_values = torch.div(-torch.log(hash_values), probs)

        # Get next token, and update logit
        next_token = hash_values.argmin(dim=-1)

        local_logits[:] = -math.inf
        local_logits[torch.arange(local_logits.shape[0]), next_token] = 0

        return local_logits


class ExponentialVerifier(Verifier):
    """
    Implements a verifier for the exponential watermarking scheme.

    Args:
        rng (RandomNumberGenerator): The random number generator to use.
        pvalue (float): The p-value threshold for the verifier.
        tokenizer (Tokenizer): The tokenizer to use.
        log (bool): Whether to use logarithmic scoring or not.
    """

    def __init__(self, rng, pvalue, tokenizer, log):
        super().__init__(rng, pvalue, tokenizer)
        self.log = log

    def _verify(self, tokens, index=0, meta=None):
        cumul = []
        seen = set()

        try:
            for i, tok in enumerate(tokens):
                prev_values = tokens[:i]
                seed = self.rng.get_seed(prev_values, [index])
                hv = self.rng.rand_index(seed, tok).item()

                if (seed[0], hv) in seen:
                    continue

                seen.add((seed[0], hv))
                cumul.append((hv, i))
            assert len(cumul)
        except Exception:
            return VerifierOutput()

        return_value, ctr, ctn = VerifierOutput(), 0, 0
        for i, val in enumerate(cumul):
            ctn += 1
            ctr += val[0] if not self.log else -np.log(max(0.00001, 1 - val[0]))
            if not self.log:
                # pval = tfp.distributions.Bates(ctn).survival_function(ctr/ctn)
                pval = scipy.stats.norm.sf(
                    ctr / ctn, loc=0.5, scale=1 / math.sqrt(12 * ctn)
                )
            else:
                # pval = s(ctr, loc=0.5, scale=1/math.sqrt(12*(ctn))) if not self.log else scipy.stats.gamma.sf(ctr, ctn)
                pval = scipy.stats.gamma.sf(ctr, ctn)
            return_value.update(i + 1, pval)
        return return_value

    def id(self):
        base = super().id()
        return base[:2] + ("log",) if self.log else base


class ExponentialEmpiricalVerifier(EmpiricalVerifier):
    """
    A class for verifying watermarks using the Exponential scheme.

    Inherits from EmpiricalVerifier.

    Args:
        rng (torch.Generator): A random number generator.
        pvalue (float): The p-value threshold for the verification test.
        tokenizer (Tokenizer): A Tokenizer object for tokenizing text.
        method (str): The watermarking method used.
        log (bool): Whether to use logarithmic scores or not.
        gamma (float): The gamma parameter for the Exponential scheme.

    Attributes:
        log (bool): Whether to use logarithmic scores or not.
    """

    def __init__(self, rng, pvalue, tokenizer, method, log, gamma):
        log = True
        super().__init__(rng, pvalue, tokenizer, method, gamma, log=log)
        self.log = log

    def score_matrix(self, tokens, random_values, index=0, meta=None):
        """Prepare all possible overlapping of random values (shape KEY_LEN x VOCAB_SIZE) and tokens (shape SEQ_LEN)"""
        tokens = tokens.reshape(-1)
        if not tokens.nelement():
            return None
        random_values = random_values.squeeze()
        if len(random_values.shape) == 1:
            # Key length of 1
            random_values = random_values.unsqueeze(0)
        return (
            torch.log(1 - random_values[:, tokens])
            if self.log
            else 1 - random_values[:, tokens]
        )

    def random_score_matrix(
        self, tokens, random_shape, shared_randomness, index=0, meta=None
    ):
        """Produce a random score vector (faster to directly sample the random scores than to sample all random values)"""
        # To do: repeat random values when token context is the same
        _, L, V = random_shape
        tokens = tokens.reshape(-1)
        if not tokens.nelement():
            return None
        random_values = torch.cuda.FloatTensor(
            L, V, device=self.rng.device
        ).uniform_(0, 1)[shared_randomness, :]
        return (
            torch.log(1 - random_values[:, tokens])
            if self.log
            else 1 - random_values[:, tokens]
        )

    def id(self):
        base = super().id()
        return base[:2] + ("log",) if self.log else base
