import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from transformers import logging, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
logging.set_verbosity_error() # Set logging to show only errors (this suppresses warnings)
import gc
from functools import partial
from .data import get_test_dataloaders, get_tokenizer
import .losses
import .utils.seq_utils as sutils


class Eval:
    def __init__(self, cfg, sde, **kwargs):
        self.cfg = cfg
        self.sde = sde
        self.distributed = kwargs.get('distributed', True)

        self.tokenizer = get_tokenizer(cfg.data.train)
        # Load gpt2 tokenizer if the current tokenizer is not gpt2 tokenizer
        # E.g., text8 and lm1b
        if cfg.data.train in ["text8", "lm1b"]:
            self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        
        token_size = cfg.tokens
        base_max_length = math.ceil(math.log(cfg.original_tokens) / math.log(token_size))
        self._base_n = partial(sutils.base_n, base=token_size, max_length=base_max_length)

        self.gen_ppl_batch_size = 16 # Adjust this for your GPU memory


    def eval(self, sample=None, drift_model=None):
        result_dict = {}
        self.device = drift_model.device

        # Measure token entropy
        if self.cfg.eval.entropy:
            result_dict["entropy"] = self.entropy(sample)

        if self.cfg.data.train == "text8":
            try:
                if self.cfg.eval.nll:
                    result_dict['bpc'] = self.bpc(drift_model)
                if self.cfg.eval.eval_model=="gptj":
                    result_dict['gptj_nll'] = self.gptj_nll(sample)
            except Exception as e:
                print(e)
            return result_dict
        
        # Measure generative perplexity
        if self.cfg.eval.gen_ppl:
            try:
                result_dict['gpt2_ppl'] = self.gpt2_ppl(sample)
            except Exception as e:
                print(e)
        if self.cfg.eval.nll:
            try:
                result_dict["nll"] = self.nll(drift_model)
            except Exception as e:
                print(e)
                
        return result_dict
    

    # Token Entropy
    def entropy(self, sample):
        if self.cfg.data.train in ["text8"]:
            sample = self.gpt_tokenizer(self.tokenizer.batch_decode(sample)).input_ids

        sample_stats = {}
        for tokens in sample:
            for token in tokens:
                if isinstance(token, torch.Tensor):
                    token = token.item()
                if token not in sample_stats:
                    sample_stats[token] = 0
                sample_stats[token] += 1

        sample_probs = {}
        total_sample_stats = sum(sample_stats.values())

        sample_entropy = 0
        for k, v in sample_stats.items():
            p = v / total_sample_stats
            sample_probs[k] = p
            if p > 0:
                sample_entropy += -p * np.log(p)

        entropy = torch.tensor(sample_entropy, device=self.device)

        if self.distributed:
            dist.all_reduce(entropy)
            entropy /= dist.get_world_size()
        
        return entropy


    def mc_estimate(self, drift_model, batch):
        loss_fn = losses.elbo_loss_fn(
            self.sde, False, simul_steps=self.cfg.eval.simul_steps
        )
        return loss_fn(drift_model, batch)


    # Upper bound of negative log-likelihood (nll)
    def nll(self, drift_model):
        test_ds = get_test_dataloaders(self.cfg)
        test_iter = iter(test_ds)
        
        with torch.no_grad():
            nelbo, N = 0., 0
            for _ in tqdm(range(self.cfg.eval.n_iters), desc='[NLL]', position=0, leave=False):
                batch = next(test_iter)['input_ids'].to(self.device)
                batch = self._base_n(batch)
                batch = F.one_hot(batch, num_classes=self.sde.manifold.dim+1).to(torch.float32)
                
                nelbo_batch = self.mc_estimate(drift_model, batch).sum()
                nelbo += nelbo_batch
                N += batch.shape[0]

            nelbo = nelbo / N

            if self.distributed:
                dist.all_reduce(nelbo)
                nelbo /= dist.get_world_size()
        del batch
        return nelbo
    
    
    # Bits-per-character (bpc)
    def bpc(self, drift_model):
        nelbo = self.nll(drift_model)
        return nelbo / self.cfg.model.length / np.log(2)


    # GPT-J generative ppl
    def gptj_nll(self, sample):
        batch_size = self.gen_ppl_batch_size
        with torch.no_grad():
            eval_model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", torch_dtype=torch.bfloat16, device_map="auto"
            )
            gen_nll, N = 0., 0
            for k in tqdm(range(len(sample) // batch_size), leave=False, desc='[GPTJ NLL]'):
                batch = sample[batch_size*k:batch_size*(k+1)]
            
                output = self.gpt_tokenizer(
                    self.tokenizer.batch_decode(batch), padding='longest', return_tensors='pt'
                ).to(self.device)
                input_ids = output.input_ids
                ori_len = output.attention_mask.sum(dim=1)

                logits = eval_model(
                    input_ids=input_ids,
                    attention_mask=output.attention_mask,
                    labels=input_ids, 
                    return_dict=True
                ).logits
                logits = logits.transpose(-1, -2)

                for j in range(len(logits)):
                    s = input_ids[j][:ori_len[j]].unsqueeze(0)
                    l = logits[j][...,:ori_len[j]].unsqueeze(0)
                    gen_nll += F.cross_entropy(l[..., :-1], s[..., 1:], reduction="none").mean()
                    N += 1
            gen_nll /= N
            if self.distributed:
                dist.all_reduce(gen_nll)
                gen_nll /= dist.get_world_size()

            del eval_model, logits, output, sample, l
            gc.collect()
            torch.cuda.empty_cache()
        return gen_nll
    

    # GPT2 generative ppl
    def gpt2_ppl(self, sample):
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(self.device).eval()
        batch_size = self.gen_ppl_batch_size
        gen_ppl, N = 0., 0

        with torch.no_grad():
            if self.cfg.data.train in ["text8", "lm1b"]:
                for k in tqdm(range(len(sample) // batch_size), leave=False, desc='[Gen. PPL]'):
                    batch = sample[batch_size*k:batch_size*(k+1)]
                
                    output = self.gpt_tokenizer(
                        self.tokenizer.batch_decode(batch), padding='longest', return_tensors='pt'
                    ).to(self.device)
                    input_ids = output.input_ids
                    ori_len = output.attention_mask.sum(dim=1)

                    logits = eval_model(
                        input_ids=input_ids,
                        attention_mask=output.attention_mask,
                        labels=input_ids, 
                        return_dict=True
                    ).logits
                    logits = logits.transpose(-1, -2)

                    for j in range(len(logits)):
                        s = input_ids[j][:ori_len[j]].unsqueeze(0)
                        l = logits[j][...,:ori_len[j]].unsqueeze(0)
                        gen_ppl += F.cross_entropy(
                            l[..., :-1], s[..., 1:], reduction="none"
                        ).mean().exp()
                        N += 1
                gen_ppl /= N

            else:
                _sample = torch.ones(
                    (len(sample), self.cfg.model.length), device=self.device, dtype=torch.long
                ) * self.tokenizer.pad_token_id
                attention_mask = torch.zeros_like(_sample)
                for i, s in enumerate(sample):
                    _sample[i, :len(s)] = s
                    attention_mask[i, :len(s)] = 1
                sample = _sample.clip(max=self.tokenizer.vocab_size-1)

                for k in tqdm(range(len(sample) // batch_size), leave=False, desc='[Gen. PPL]'):
                    input_ids = sample[batch_size*k:batch_size*(k+1)]
                    ori_len = attention_mask[batch_size*k:batch_size*(k+1)].sum(dim=1)

                    logits = eval_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask[batch_size*k:batch_size*(k+1)],
                        labels=input_ids, 
                        return_dict=True
                    ).logits
                    logits = logits.transpose(-1, -2)

                    for j in range(len(logits)):
                        s = input_ids[j][:ori_len[j]].unsqueeze(0)
                        l = logits[j][...,:ori_len[j]].unsqueeze(0)
                        gen_ppl += F.cross_entropy(
                            l[..., :-1], s[..., 1:], reduction="none"
                        ).mean().exp()
                        N += 1
                gen_ppl /= N

            if self.distributed:
                dist.all_reduce(gen_ppl)
                gen_ppl /= dist.get_world_size()
            
            del eval_model, logits
            gc.collect()
            torch.cuda.empty_cache()
        return gen_ppl