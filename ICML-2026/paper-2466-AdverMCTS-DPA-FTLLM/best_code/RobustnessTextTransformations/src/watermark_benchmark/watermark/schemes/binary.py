import math
import random

import scipy
import torch
import torch.nn.functional as F

from watermark_benchmark.utils.bit_tokenizer import Binarization
from watermark_benchmark.utils.classes import VerifierOutput
from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.random import ExternalRandomness
from watermark_benchmark.watermark.templates.verifier import (
    EmpiricalVerifier,
    Verifier,
)


class BinaryGenerator(Watermark):
    """
    A watermarking scheme that generates binary tokens. See Christ et al. (2023) for more details.

    Args:
        rng (RandomNumberGenerator): A random number generator.
        verifier (Verifier): A verifier object.
        tokenizer (Tokenizer): A tokenizer object.
        temp (float): A temperature value for softmax.
        binarizer (Binarization): A binarizer object.
        skip_prob (float): A probability value for skipping the watermarking process.

    Attributes:
        skip_prob (float): A probability value for skipping the watermarking process.
        base_len (int): The length of the previous tokens.
        binarizer (Binarization): A binarizer object.
    """

    def __init__(self, rng, verifier, tokenizer, temp, binarizer, skip_prob):
        super().__init__(rng, verifier, tokenizer, temp)
        self.skip_prob = skip_prob
        self.base_len = -1
        if binarizer is None:
            self.binarizer = Binarization(tokenizer, rng.devices)
        else:
            self.binarizer = binarizer

    def reset(self):
        super().reset()
        self.base_len = -1

    def _process(self, logits, previous_tokens, ids):
        previous_tokens = self.rng.normalize_previous_values(previous_tokens)

        # Truncate unused logits
        if random.random() < self.skip_prob:
            return logits

        logits = logits[:, : self.rng.vocab_size] / self.temp
        probs = F.softmax(logits, dim=-1)

        if self.base_len < 0:
            self.base_len = previous_tokens.shape[1]

        N, _ = logits.shape

        representations = self.binarizer.get_token_to_bits_tensor(probs.device)
        choice = torch.zeros((N, self.binarizer.L)).int().to(logits.device) - 1
        seeds = self.rng.get_seed(previous_tokens, ids)

        # Select next binary token
        branch_filter = torch.ones((N,)).to(probs.device).bool()
        for bit_index in range(self.binarizer.L):
            # If all choices are leaves, stop. We can tell we are on a leaf if the only remaining element has bit -1
            prob_sum = probs.sum(axis=1)
            prob_done = (
                probs[:, representations[:, bit_index] == -1].sum(axis=1)
                / prob_sum
            )
            branch_filter[prob_done > 0.5] = False
            if not branch_filter.sum():
                break

            # Compute probability and get randomness
            p = (
                probs[:, representations[:, bit_index] == 1].sum(axis=1)
                / prob_sum
            )
            h = self.rng.rand_index(seeds, bit_index, device=probs.device)

            choice[branch_filter, bit_index] = (h < p).int()[branch_filter]

            # Set probability of stale branches to 0
            criteria = (
                representations.expand(N, *representations.shape)[
                    :, :, bit_index
                ]
                != choice[:, bit_index].expand(self.binarizer.V, N).t()
            )
            probs[criteria] = 0

        # Convert to token
        try:
            choice = self.binarizer.to_token(choice)
        except Exception:
            # Sometimes (very rarely) the token does not exist. We need to debug this, but for now we just return the original logits.
            return logits

        next_token = choice.to(probs.device)
        logits[:] = -math.inf
        logits[torch.arange(logits.shape[0]), next_token] = 0

        return logits


class BinaryVerifier(Verifier):
    """
    Verifier for binary watermarking schemes.

    Args:
        rng (RandomNumberGenerator): Random number generator.
        pvalue (float): P-value threshold for the statistical test.
        tokenizer (Tokenizer): Tokenizer object.
        binarizer (Binarizer): Binarizer object.
        skip_prob (float): Probability of skipping a token during verification.

    Attributes:
        skip_prob (float): Probability of skipping a token during verification.
        binarizer (Binarizer): Binarizer object.
    """

    def __init__(self, rng, pvalue, tokenizer, binarizer, skip_prob):
        super().__init__(rng, pvalue, tokenizer)
        self.skip_prob = skip_prob
        if binarizer is None:
            self.binarizer = Binarization(tokenizer, rng.devices)
        else:
            self.binarizer = binarizer

    def _verify(self, tokens, index=0, meta=None):
        return_value = VerifierOutput()
        binary_tokens = self.binarizer.to_bit(tokens).squeeze()
        mask = binary_tokens >= 0
        max_bitlen = mask.sum(axis=1).max()
        binary_tokens = binary_tokens[:, :max_bitlen]
        mask = mask[:, :max_bitlen]
        ctn = mask.sum(axis=1)
        xi = []
        for i in range(tokens.shape[-1]):
            prev_values = tokens[:i]
            bitlen = ctn[i].item()
            seed = self.rng.get_seed(prev_values, [index])
            xi.append(
                [self.rng.rand_index(seed, i).item() for i in range(bitlen)]
                + [-1 for _ in range(max_bitlen - bitlen)]
            )

        xi = torch.Tensor(xi).to(self.rng.device)

        v = (
            -(xi * binary_tokens + (1 - xi) * (1 - binary_tokens)).abs().log()
            * mask
        )
        cumul = v.sum(axis=-1).cumsum(0).tolist()
        ctn = mask.sum(axis=1).cumsum(0).tolist()

        # Compute average
        for i, v in enumerate(cumul):
            c = ctn[i]
            likelihood = scipy.stats.gamma.sf(v, c)
            return_value.update(i, likelihood)

        return return_value


class BinaryEmpiricalVerifier(EmpiricalVerifier):
    """
    A verifier for binary watermarking schemes that uses empirical testing to detect watermarks.

    Args:
        rng (RandomnessProvider): A randomness provider.
        pvalue (float): The p-value threshold for the statistical test.
        tokenizer (Tokenizer): A tokenizer object.
        method (str): The detection method to use.
        binarizer (Binarizer, optional): A binarizer object. Defaults to None.
        skip_prob (float): The probability of skipping a token during detection.
        gamma (float): The gamma parameter for the statistical test.

    Attributes:
        skip_prob (float): The probability of skipping a token during detection.
        binarizer (Binarizer): A binarizer object.
    """

    def __init__(
        self, rng, pvalue, tokenizer, method, binarizer, skip_prob, gamma
    ):
        super().__init__(rng, pvalue, tokenizer, method, gamma, log=True)
        self.skip_prob = skip_prob
        if binarizer is None:
            self.binarizer = Binarization(tokenizer, rng.devices)
        else:
            self.binarizer = binarizer

    def score_matrix(self, tokens, random_values, index=0, meta=None):
        if not tokens.nelement():
            return None

        binary_tokens = self.binarizer.to_bit(
            tokens.to(self.rng.device)
        ).squeeze()
        mask = binary_tokens >= 0

        # Truncate tokens to max token bit length. For everyting to fit on GPUs, we set a maximum on the truncated length.
        max_bitlen = mask.sum(axis=1).max()
        if isinstance(max_bitlen, torch.Tensor):
            max_bitlen = int(max_bitlen.item())
        KL, SL = random_values.shape[1], binary_tokens.shape[0]
        indices = (
            torch.vstack(
                (
                    (
                        (
                            torch.arange(KL, device=self.rng.device)
                            .reshape(-1, 1)
                            .repeat(1, SL)
                            .flatten()
                        )
                    ),
                    torch.arange(SL, device=self.rng.device).repeat(KL) % SL,
                )
            )
            .t()
            .reshape(KL, -1, 2)
            .to(self.rng.device)
        )

        rslt = None
        for r in range(1 + (max_bitlen // 100)):
            range_low = r * 100
            range_high = min((r + 1) * 100, max_bitlen)
            if range_high == range_low:
                continue
            binary_tokens_local = binary_tokens[:, range_low:range_high]
            mask_local = mask[:, range_low:range_high]
            try:
                xi_local = random_values[0, :, range_low:range_high].reshape(
                    -1, range_high - range_low
                )
            except:
                print(random_values.shape)
                print(range_low)
                print(range_high)

            # Binary tokens has shape SL x L, xi has shape KL x L. We only want to sum coordinates that are not -1.
            v = (
                (
                    xi_local[indices[:, :, 0]]
                    * binary_tokens_local[indices[:, :, 1]]
                    + (1 - binary_tokens_local[indices[:, :, 1]])
                    * (1 - xi_local[indices[:, :, 0]])
                ).abs()
            ).log() * mask_local[indices[:, :, 1]]
            if rslt is None:
                rslt = v.sum(axis=-1)
            else:
                rslt += v.sum(axis=-1)

        return rslt / mask.sum(axis=-1)

    def random_score_matrix(
        self,
        tokens,
        random_shape,
        shared_randomness,
        binary_tokens=None,
        index=0,
        meta=None,
    ):
        """Produce a random score vector (faster to directly sample the random scores than to sample all random values)"""
        _, L, _ = random_shape
        if binary_tokens is None:
            binary_tokens = self.binarizer.to_bit(
                tokens.to(self.rng.device)
            ).squeeze()
        mask = binary_tokens >= 0

        # Truncate tokens to max token bit length. For everyting to fit on GPUs, we set a maximum on the truncated length.
        max_bitlen = mask.sum(axis=1).max()
        KL, SL = L, binary_tokens.shape[0]
        indices = (
            torch.vstack(
                (
                    (
                        (
                            torch.arange(KL, device=self.rng.device)
                            .reshape(-1, 1)
                            .repeat(1, SL)
                            .flatten()
                        )
                    ),
                    torch.arange(SL, device=self.rng.device).repeat(KL) % SL,
                )
            )
            .t()
            .reshape(KL, -1, 2)
        )
        xi = torch.cuda.FloatTensor(
            L, max_bitlen, device=self.rng.device
        ).uniform_(0, 1)[shared_randomness, :]

        rslt = None
        for r in range(1 + (max_bitlen // 100)):
            range_low = r * 100
            range_high = min((r + 1) * 100, max_bitlen)
            if range_high == range_low:
                continue
            binary_tokens_local = binary_tokens[:, range_low:range_high]
            mask_local = mask[:, range_low:range_high]
            xi_local = xi[:, range_low:range_high].reshape(
                -1, range_high - range_low
            )
            # Binary tokens has shape SL x L, xi has shape KL x L. We only want to sum coordinates that are not -1.

            v = (
                (
                    xi_local[indices[:, :, 0]]
                    * binary_tokens_local[indices[:, :, 1]]
                    + (1 - binary_tokens_local[indices[:, :, 1]])
                    * (1 - xi_local[indices[:, :, 0]])
                ).abs()
            ).log() * mask_local[indices[:, :, 1]]
            if rslt is None:
                rslt = v.sum(axis=-1)
            else:
                rslt += v.sum(axis=-1)

        return rslt / mask.sum(axis=1)

    def pre_compute_baseline(self, max_len=1200, runs=250):
        # Special version for binary schemes
        self.test_results = torch.zeros((runs, max_len)).to(self.rng.device)
        binary_tokens = (
            torch.randint(0, 2, (max_len * self.binarizer.avg_bit_length,))
            .to(self.rng.device)
            .reshape(max_len, self.binarizer.avg_bit_length)
        )
        if type(self.rng) == ExternalRandomness:
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
                None,
                (1, L, self.binarizer.avg_bit_length),
                shared_randomness,
                binary_tokens,
            )
            self.test_results[run] = self.detect(scores)[0]

        self.precomputed = True
