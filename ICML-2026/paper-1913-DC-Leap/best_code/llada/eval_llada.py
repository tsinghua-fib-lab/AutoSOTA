# This code is adapted from: https://github.com/ML-GSAI/LLaDA
'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from model.modeling_llada import LLaDAModelLM
from dataclasses import asdict
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from generate import generate, generate_with_dc_leap
import time
import datetime


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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
        steps=256,
        gen_length=256,
        block_length=32,
        remasking='low_confidence',
        commit_thres=0.70,
        draft_thres=0.98,
        max_window_size=128,
        device="cuda",
        method="original",         
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Instruct model path.
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
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = LLaDAModelLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.cache = None
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        self.commit_thres = commit_thres
        self.draft_thres = draft_thres
        self.max_window_size = max_window_size
        #self.num_distal_blocks = num_distal_blocks
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        
        self.method = method
        # verify input
        acceptable_methods = ["original", "dc_leap"]
        assert method in acceptable_methods, f"method must be one of {acceptable_methods}"
        
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

    @torch.no_grad()
    def loglikelihood(self, requests):
        start = time.perf_counter()
        total_tokens = 0

        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            nonlocal total_tokens
            total_tokens += len(prefix) + len(target)
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds).map(_tokenize).with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Computing likelihood..."):
            prefix = elem["prefix"]
            target = elem["target"]
            ll = self.get_loglikelihood(prefix, target)
            is_greedy = self.suffix_greedy_prediction(prefix, target)
            out.append((ll, bool(is_greedy)))

        elapsed = time.perf_counter() - start
        tok_p_sec = total_tokens / elapsed if elapsed else 0
        
        if out:
            print(f"[Local] tokens_per_second: {tok_p_sec:.2f}")
            print(f"[Local] seconds_per_sample: {elapsed / len(requests):.4f}")

        torch.cuda.empty_cache()
        return out          


    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        out = []
        num_tokens = 0
        total_raw_tokens = 0
        start_time = time.perf_counter()
        
        for req in tqdm(requests, desc="Generating..."):
            question = req.args[0]
            if (not self.is_instruct) or ('task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval')):
                user_input = question
                input_ids = self.tokenizer(user_input)['input_ids']
            else:
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input)['input_ids']

            stop_tokens = req.args[1]['until']
            input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
            input_len = input_ids.shape[1]
            if self.method == "original":
                generated_answer = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0., cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id)
            
            elif self.method == "dc_leap":
                generated_answer = generate_with_dc_leap(self.model, input_ids, steps=self.steps, commit_thres=self.commit_thres, draft_thres=self.draft_thres, gen_length=self.gen_length, block_length=self.block_length, max_window_size=self.max_window_size, temperature=0., cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id)
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
            raw_generated_len = generated_answer.shape[1] - input_len
            generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=False)
            
            total_raw_tokens += raw_generated_len

            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
            num_tokens += generated_answer_ids.numel()
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        print(f"[Local] Number of tokens: {total_raw_tokens}")
        print(f"[Local] Generation time: {elapsed:.2f} seconds")
        print(f"[Local] Tokens per second: {total_raw_tokens / elapsed:.2f}")

        accelerator = getattr(self, 'accelerator', None)
        if accelerator is not None and accelerator.num_processes > 1:
            try:
                accelerator.wait_for_everyone()
                total_elapsed = accelerator.reduce(torch.tensor(elapsed, device=accelerator.device), reduction="sum").item()
                total_num_tokens = accelerator.reduce(torch.tensor(num_tokens, device=accelerator.device), reduction="sum").item()
                if accelerator.is_main_process:
                    print(f"[Global] Number of tokens: {total_num_tokens}")
                    print(f"[Global] Generation time: {total_elapsed:.2f} seconds")
                    print(f"[Global] Tokens per second: {total_num_tokens / total_elapsed:.2f}")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[Warning] Failed to aggregate multi-GPU stats: {e}")

        return out 


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    