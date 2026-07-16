"""
Example use of the benchmark on a custom watermark.
We define a basic watermark that selects tokens 
based on a hash-based greenlist as an example.
"""

from typing import List, Optional, Union

import hash_cpp
import scipy
import torch
from transformers import PreTrainedTokenizerBase
from watermark_benchmark import (
    Binarization,
    ConfigSpec,
    VerifierOutput,
    WatermarkSpec,
    full_pipeline,
)
from watermark_benchmark.watermark import (
    BaseRandomness,
    Verifier,
    Watermark,
)


# Define the watermark randomness
class ExampleRandomness(BaseRandomness):
    """Example randomness implementation, only requires a secret key and does not depend on generated text"""

    def _get_seed(self, previous_values, ids=None):
        N, _ = previous_values.shape
        if ids is None:
            ids = [0 for _ in range(N)]

        return [str(self.get_secret(i)) for i in ids]

    def _rand_index(self, seeds, index, device=None):
        if device is None:
            device = self.device
        return hash_cpp.index_hash(seeds, index).to(device)


# Define the watermark verifier
class ExampleVerifier(Verifier):
    def _verify(self, tokens, index=0):
        cumul = []

        for i, _ in enumerate(tokens):
            prev_values = tokens[:i]
            current_token = tokens[i].item()

            seeds = self.rng.rand_index(
                self.rng.get_seed(prev_values, [index]), 0
            )
            hv = self.rng.rand_index(seeds, current_token).item()
            if hv > 0.5:
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
                ctr, cnt, 0.5, alternative="greater"
            ).pvalue
            return_value.update(i, nd)

        return return_value


# Define the watermark
class ExampleWatermark(Watermark):
    def _process(self, logits, previous_tokens, ids):
        logits = logits[:, : self.rng.vocab_size]
        hash_values = self.rng.rand_range(
            self.rng.get_seed(previous_tokens, ids),
            self.rng.vocab_size,
            logits.device,
        )
        logits[hash_values < 0.5] -= 5
        return logits


# Define the builder function. Return None for unhandled cases
def custom_builder(
    watermark_spec: Optional[WatermarkSpec],
    tokenizer: Optional[PreTrainedTokenizerBase],
    binarizer: Optional[Binarization],
    device: Union[str, int, torch.device] = "cpu",
    key: Union[str, List[str]] = None,
) -> Optional[Watermark]:
    if watermark_spec is None or watermark_spec.generator != "example":
        return None
    else:
        rng = ExampleRandomness(key, device)
        return ExampleWatermark(
            rng,
            ExampleVerifier(rng, watermark_spec.pvalue, tokenizer),
            tokenizer,
            watermark_spec.temp,
        )


def main():
    config = ConfigSpec()
    config.devices = [i for i in range(torch.cuda.device_count())]

    watermarks = []
    for temp in [0, 0.3, 0.7, 1]:
        spec = WatermarkSpec()
        spec.generator = "example"
        spec.temp = temp
        watermarks.append(spec)

    return full_pipeline(
        config, watermarks, custom_builder=custom_builder, run_validation=True
    )
