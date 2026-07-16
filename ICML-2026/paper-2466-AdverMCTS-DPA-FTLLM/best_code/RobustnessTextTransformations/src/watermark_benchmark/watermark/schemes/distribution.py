import scipy
import torch

from watermark_benchmark.utils.classes import VerifierOutput
from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import (
    EmpiricalVerifier,
    Verifier,
)


class DistributionShiftGeneration(Watermark):
    """
    A watermarking scheme that adds a delta value to the logits of certain tokens in the input sequence. See Kirchenbauer et al. (2023) for more details.

    Args:
        rng (RandomNumberGenerator): A random number generator object.
        verifier (Verifier): A verifier object.
        tokenizer (Tokenizer): A tokenizer object.
        temp (float): Temperature parameter for softmax function.
        delta (float): Value to add to logits of selected tokens.
        gamma (float): Proportion of tokens to select for watermarking.

    Attributes:
        delta (float): Value to add to logits of selected tokens.
        gamma (float): Proportion of tokens to select for watermarking.
        temp (float): Temperature parameter for softmax function.
    """

    def __init__(self, rng, verifier, tokenizer, temp, delta, gamma):
        super().__init__(rng, verifier, tokenizer, temp)
        self.delta = delta
        self.gamma = gamma
        self.temp = temp

    def _process(self, logits, previous_tokens, ids):
        """
        Applies the watermarking scheme to the input logits.

        Args:
            logits (torch.Tensor): The input logits.
            previous_tokens (torch.Tensor): The previous tokens in the sequence.
            ids (torch.Tensor): The IDs of the previous tokens.

        Returns:
            torch.Tensor: The logits with the watermark applied.
        """

        # Truncate unused logits
        logits = logits[:, : self.rng.vocab_size]

        N, _ = logits.shape

        # Get greenlist and update logits
        seeds = self.rng.rand_index(self.rng.get_seed(previous_tokens, ids), 0)
        greenlist = self.rng.green_list(seeds, self.gamma)
        logits[
            torch.arange(N).unsqueeze(1).expand(-1, greenlist.size(1)),
            greenlist,
        ] += self.delta

        return logits


class DistributionShiftVerifier(Verifier):
    """
    A verifier that checks for distribution shift in a sequence of tokens.

    Args:
        rng (RandomNumberGenerator): A random number generator.
        pvalue (float): The p-value threshold for the binomial test.
        tokenizer (Tokenizer): A tokenizer for the sequence of tokens.
        gamma (float): The proportion of tokens that are allowed to be different.

    Attributes:
        gamma (float): The proportion of tokens that are allowed to be different.
    """

    def __init__(self, rng, pvalue, tokenizer, gamma):
        super().__init__(rng, pvalue, tokenizer)
        self.gamma = gamma

    def _verify(self, tokens, index=0, meta=None):
        cumul = []
        seen = set()

        for i, _ in enumerate(tokens):
            prev_values = tokens[:i]
            current_token = tokens[i].item()

            seeds = self.rng.rand_index(
                self.rng.get_seed(prev_values, [index]), 0
            )
            greenlist = self.rng.green_list(seeds, self.gamma)

            if (current_token, seeds.item()) in seen:
                continue

            seen.add((current_token, seeds.item()))

            if current_token in set(greenlist.squeeze().cpu().numpy()):
                cumul.append(1)
            else:
                cumul.append(0)

        if not len(cumul):
            return VerifierOutput()

        ctr = 0
        return_value = VerifierOutput()
        for i, val in enumerate(cumul):
            ctr += val
            cnt = i + 1
            nd = scipy.stats.binomtest(
                ctr, cnt, self.gamma, alternative="greater"
            ).pvalue
            return_value.update(i, nd)

        return return_value


class DistributionShiftEmpiricalVerifier(EmpiricalVerifier):
    """
    A class for verifying the distribution shift of a watermark using empirical testing.

    Inherits from EmpiricalVerifier.

    Args:
        rng (RandomNumberGenerator): A random number generator object.
        pvalue (float): The p-value threshold for the statistical test.
        tokenizer (Tokenizer): A tokenizer object.
        method (str): The method used to generate the watermark.
        gamma_watermark (float): The gamma value for the watermark.
        gamma_edit_distance (float): The gamma value for the edit distance.

    Methods:
        score_matrix(tokens, random_values, index=0): Computes the score matrix for the given tokens and random values.
        random_score_matrix(tokens, random_shape, shared_randomness, index=0): Produces a random score matrix.
    """

    def __init__(
        self,
        rng,
        pvalue,
        tokenizer,
        method,
        gamma_watermark,
        gamma_edit_distance,
    ):
        super().__init__(
            rng, pvalue, tokenizer, method, gamma_edit_distance, False
        )
        self.gamma = gamma_watermark
        self.rand_size = 1

    def score_matrix(self, tokens, random_values, index=0, meta=None):
        _, L, _ = random_values.shape
        random_values = random_values[0, :, 0].reshape(1, L).cpu()

        tokens = tokens.squeeze().to(self.rng.device)
        if not tokens.nelement():
            return None

        greenlists = torch.stack(
            [
                self.rng.green_list(
                    random_values[:, i], self.gamma, True
                ).squeeze()
                for i in range(L)
            ]
        )
        greenlists = greenlists.repeat(1 + L // len(tokens), 1)[
            : len(tokens), :
        ].to(self.rng.device)
        rtn = 1 - (greenlists[:, tokens].float())
        return rtn.float()

    def random_score_matrix(
        self, tokens, random_shape, shared_randomness, index=0, meta=None
    ):
        """Produce a random score vector (faster to directly sample the random scores than to sample all random values)"""
        _, L, _ = random_shape
        val = (
            torch.cuda.FloatTensor(
                L, self.rng.vocab_size, device=self.rng.device
            )
            .uniform_(0, 1)[shared_randomness, :]
            .lt(self.gamma)
        )
        return 1 - (val[:, tokens.squeeze().to(self.rng.device)].float())

        # random_values = torch.rand((1,L), dtype=torch.float32).to(self.rng.device)
        # tokens = tokens.squeeze()
        # greenlists = [set(self.rng.green_list(random_values[:, i%L], self.gamma).squeeze().cpu().numpy()) for i in range(len(tokens))]
        # return torch.tensor([[0 if t.item() in g else 1 for t in tokens] for g in greenlists]).to(self.rng.device)
