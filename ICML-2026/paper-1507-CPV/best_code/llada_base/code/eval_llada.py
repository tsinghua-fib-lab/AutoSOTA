'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import sys
import warnings
from pathlib import Path

# Suppress dill pickling warnings for dynamically loaded model classes
warnings.filterwarnings("ignore", category=UserWarning, module="dill._dill")
# Also filter the specific PicklingWarning
try:
    from dill._dill import PicklingWarning
    warnings.filterwarnings("ignore", category=PicklingWarning)
except ImportError:
    pass

# Prefer local lm_eval over the system-installed one
_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_LM_EVAL = _SCRIPT_DIR / "lm_eval"
if _LOCAL_LM_EVAL.exists():
    sys.path.insert(0, str(_SCRIPT_DIR))

import accelerate
import torch
import re
import time
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from generate import generate
from modeling_llada_kv_cover import LLaDAModelLM
from contextual_cover import COVERGenerator


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set CUDA random seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        # Method selection: "ori" for original generation or "cover" for COVER.
        method="ori",
        # COVER-specific parameters
        tau_draft=0.5,
        max_unmask_per_step=5,
        use_low_conf_reverify=True,
        max_reverify_per_step=5,
        max_reverify_times=1,
        use_kv_cache_for_reverify=True,
        use_attention_score=False,
        debug=False,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
            method: Generation method - "ori" for original generation or "cover" for COVER.
            tau_draft: Confidence threshold for drafting and verification (COVER only).
            max_unmask_per_step: Maximum positions to unmask per step (COVER only).
            use_low_conf_reverify: Whether to use low-confidence re-verification (COVER only).
            max_reverify_per_step: Maximum positions to re-verify per step (COVER only).
            max_reverify_times: Maximum re-verification times per position (COVER only).
            use_kv_cache_for_reverify: Whether to use KV cache for re-verification (COVER only).
            use_attention_score: Whether to use attention score for seed selection (COVER only).
            debug: Enable debug output (COVER only).
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        self.model_path = model_path
        self.method = method.lower().strip() if isinstance(method, str) else "ori"
        if self.method not in {"cover", "ori"}:
            raise ValueError(f"Unknown method '{self.method}'. Expected 'cover' or 'ori'.")

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        # Use the COVER model when running COVER; otherwise use the original model.
        if self.method == "cover":
            self.model = LLaDAModelLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                **model_kwargs
            )
        elif self.method == "ori":
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)

        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

        # COVER-specific parameters
        self.tau_draft = float(tau_draft)
        self.max_unmask_per_step = int(max_unmask_per_step)
        self.use_low_conf_reverify = bool(use_low_conf_reverify)
        self.max_reverify_per_step = int(max_reverify_per_step)
        self.max_reverify_times = int(max_reverify_times)
        self.use_kv_cache_for_reverify = bool(use_kv_cache_for_reverify)
        self.use_attention_score = bool(use_attention_score)
        self.debug = bool(debug)

        # Initialize COVER generator if using COVER.
        self._cover_generator = None
        if self.method == "cover":
            self._cover_generator = COVERGenerator(self.model, self.tokenizer, mask_id=self.mask_id)
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for idx, elem in enumerate(tqdm(ds, desc=f"Generating (method={self.method})...")):
            req = requests[idx]
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            t0 = time.perf_counter()

            if self.method == "cover":
                # Use COVER generation.
                result = self._cover_generator.generate_causal(
                    prompt=prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    tau_draft=self.tau_draft,
                    max_unmask_per_step=self.max_unmask_per_step,
                    temperature=0.0,
                    use_low_conf_reverify=self.use_low_conf_reverify,
                    max_reverify_per_step=self.max_reverify_per_step,
                    max_reverify_times=self.max_reverify_times,
                    use_kv_cache_for_reverify=self.use_kv_cache_for_reverify,
                    use_attention_score=self.use_attention_score,
                    debug=self.debug,
                    track_flip_flop=True,
                )
                generated_output, actual_steps = result[0], result[1]
                flip_flop_stats = result[2] if len(result) > 2 else None
            elif self.method == "ori":
                # Use original generation.
                generated_output = generate(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id
                )
                actual_steps = self.steps
                flip_flop_stats = None
            else:
                raise ValueError(f"Unknown method '{self.method}'. Expected 'cover' or 'ori'.")

            t1 = time.perf_counter()

            # Save inference statistics to the request object for logging
            req.inference_steps = int(actual_steps) if actual_steps is not None else None
            req.inference_time_seconds = float(t1 - t0)

            # Save flip-flop statistics if available
            if flip_flop_stats is not None:
                req.flip_flop_count = flip_flop_stats["flip_flop_count"][0] if flip_flop_stats["flip_flop_count"] else 0
                req.total_unmask_count = flip_flop_stats["total_unmask_count"][0] if flip_flop_stats["total_unmask_count"] else 0
                req.total_remask_count = flip_flop_stats["total_remask_count"][0] if flip_flop_stats["total_remask_count"] else 0
                req.replace_count = flip_flop_stats.get("replace_count", [0])[0] if flip_flop_stats.get("replace_count") else 0
                req.changed_after_remask_count = flip_flop_stats.get("changed_after_remask_count", [0])[0] if flip_flop_stats.get("changed_after_remask_count") else 0
                req.keep_count = flip_flop_stats.get("keep_count", [0])[0] if flip_flop_stats.get("keep_count") else 0
            else:
                req.flip_flop_count = None
                req.total_unmask_count = None
                req.total_remask_count = None
                req.replace_count = None
                req.changed_after_remask_count = None
                req.keep_count = None

            generated_answer = self.tokenizer.decode(generated_output[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
