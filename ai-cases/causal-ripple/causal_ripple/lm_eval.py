"""lm-evaluation-harness integration for historical HumanEval and MBPP."""

import os
import random

import accelerate
import numpy as np
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .cascade import visible_cascade
from .config import MODEL_ID, MODEL_REVISION, get_config
from .sampling import generate


@register_model("causal_ripple")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path=MODEL_ID,
        revision=MODEL_REVISION,
        profile="base",
        remasking="ripple",
        steps=128,
        gen_length=512,
        block_length=512,
        batch_size=1,
        seed=1234,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("The reported code experiments use batch size one")
        self.batch_size = 1
        self.accelerator = accelerate.Accelerator()
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self._device = torch.device(f"cuda:{self._rank}" if self._world_size > 1 else device)
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = dict(trust_remote_code=True, revision=revision)
        self.model = (
            AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **loader
            )
            .to(self._device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **loader)
        self.config = get_config(profile, remasking)
        self.steps, self.gen_length, self.block_length = (
            int(steps),
            int(gen_length),
            int(block_length),
        )
        self.remasking = remasking

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError("This integration evaluates code-generation tasks")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This integration evaluates code-generation tasks")

    def generate_until(self, requests):
        results = []
        for request in tqdm(requests, desc="Generating code"):
            text, options = request.args
            prompt = torch.tensor([self.tokenizer(text)["input_ids"]], device=self.device)
            kwargs = dict(
                model=self.model,
                prompt=prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                tokenizer=self.tokenizer,
                config=self.config,
                temperature=0.0,
            )
            output, _, _ = (
                visible_cascade(**kwargs)
                if self.remasking == "visible_cascade"
                else generate(remasking=self.remasking, **kwargs)
            )
            answer = self.tokenizer.decode(output[0, prompt.shape[1] :], skip_special_tokens=False)
            for stop in options["until"]:
                if stop in answer:
                    answer = answer.split(stop)[0]
            ids = self.tokenizer(answer)["input_ids"]
            results.append(self.tokenizer.decode(ids, skip_special_tokens=True))
            if self.world_size > 1:
                self.accelerator.wait_for_everyone()
        return results


def main():
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cli_evaluate()


if __name__ == "__main__":
    main()
