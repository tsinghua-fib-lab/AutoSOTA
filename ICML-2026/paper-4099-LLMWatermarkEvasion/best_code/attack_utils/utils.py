from typing import Any, List, Optional

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    LlamaTokenizerFast,
    LogitsProcessorList,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import partial

import openai


from tqdm import tqdm

import sys

sys.path.append(".")
sys.path.append("./attack_utils")

from logger import ProgressLogger, print

import random

import tiktoken

full_model_names = {
    "gpt4o": "gpt-4o-2024-08-06",
    "gpt4o-mini": "gpt-4o-mini-2024-07-18",
}

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)



def is_decoder_only_model(model_name: str) -> bool:
    return any(
        [
            (t in model_name.lower())
            for t in ["gpt", "opt", "bloom", "llama", "mistral", "gemma"]
        ]
    )


def is_chat_model(model_name: str) -> bool:
    return "Instruct" in model_name or "chat" in model_name or "it" in model_name


class HfModel:
    def __init__(self, model_cfg, sys_prompt=None, cache_dir=None) -> None:
        self.device, self.seed = model_cfg.device, model_cfg.seed
        self.cfg = model_cfg
        self.cache_dir = cache_dir
        self.model: Optional[AutoModel] = None
        self.end_inst_tag = "[/INST]"
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer()
        self.sys_prompt = sys_prompt
        self._load_model()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        # Load tokenizer
        if "Llama-3.1" in self.cfg.name or "Llama-3.2" in self.cfg.name:
            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.name, padding_side="left", cache_dir=self.cache_dir
            )
            tokenizer.pad_token = tokenizer.eos_token
        elif "llama" in self.cfg.name:
            tokenizer = LlamaTokenizerFast.from_pretrained(
                self.cfg.name, torch_dtype=torch.float16, padding_side="left", cache_dir=self.cache_dir
            )
            tokenizer.pad_token = tokenizer.eos_token
        elif "dipper" in self.cfg.name:
            tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", cache_dir=self.cache_dir)
            # Used to be T5Tokenizer but we need this for offset_mapping
        elif "pegasus" in self.cfg.name:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.name, cache_dir=self.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.name, padding_side="left", cache_dir=self.cache_dir
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.period_token_id = tokenizer(
            "Game over.", add_special_tokens=False
        ).input_ids[-1]

        return tokenizer

    def _load_model(self) -> None:
        if "gpt-oss-20b" in self.cfg.name:
            from transformers import GptOssForCausalLM, Mxfp4Config
            quantization_config = Mxfp4Config(dequantize=True)
            self.model = GptOssForCausalLM.from_pretrained(
                self.cfg.name,
                quantization_config=quantization_config,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=self.cache_dir
            )

        elif is_decoder_only_model(self.cfg.name) or "Qwen" in self.cfg.name or "Phi" in self.cfg.name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name,
                torch_dtype=torch.float16 if self.cfg.use_fp16 else torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir
            )
        elif "dipper" in self.cfg.name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.cfg.name)
        else:
            raise ValueError(f"Unknown model name: {self.cfg.name}")

        self.model.eval()

    def _chatify(
        self, prompt: str, completion: Optional[str] = None, sys_prompt=None
    ) -> str:
        if completion is not None:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            if "Qwen" in self.cfg.name or "gpt-oss-20b" in self.cfg.name:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"{sys_prompt}\n\n{prompt}"}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

    def _encode_batch(
        self,
        prompts: List[str],
        completions: Optional[List[str]] = None,
        prevent_truncation: bool = False,
        return_offsets: bool = False,
        return_inputs: bool = False,
    ) -> BatchEncoding:
        # Tokenize
        if is_chat_model(self.cfg.name):
            if completions is not None:
                inputs = [self._chatify(p, c) for p, c in zip(prompts, completions)]
            else:
                if self.sys_prompt is not None:
                    inputs = [
                        self._chatify(p, sys_prompt=self.sys_prompt) for p in prompts
                    ]
                else:
                    inputs = [self._chatify(p) for p in prompts]

            # Some templates are wrong (e.g., mistral) and add a trailing space
            inputs = [i.strip() for i in inputs]

            add_special_tokens = False  # already done by chatify
        else:
            inputs = prompts
            add_special_tokens = True
        batchenc: BatchEncoding = self.tokenizer(
            inputs,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=not prevent_truncation,
            padding=len(inputs) > 1,
            max_length=self.cfg.prompt_max_len,
            return_offsets_mapping=return_offsets,
        ).to(self.device)

        if batchenc["input_ids"].shape[-1] == self.cfg.prompt_max_len:
            print(
                f"At least one prompt was truncated to {self.cfg.prompt_max_len}",
                tag="!!!",
                color="red",
                tag_color="red",
            )
        if return_inputs:
            # Needed for PPL computation
            return inputs, batchenc
        else:
            return batchenc

    def generate(
        self,
        prompts: List[str],
        logit_processors: LogitsProcessorList = None,
        reseed: bool = True,
        return_model_inputs: bool = False,
        return_dict_in_generate=False
    ) -> Any:
        if self.model is None:
            raise RuntimeError("Call _load_model before using the model")
        inputs, batchenc = self._encode_batch(
            prompts, return_offsets=True, return_inputs=True
        )
        print(f"Model Input with length {batchenc['input_ids'].shape[1]}", color="blue")

        actual_inputs = self._get_actual_input_str(inputs, batchenc)
        if hasattr(batchenc, "offset_mapping"):
            del batchenc["offset_mapping"]
        if reseed:
            torch.manual_seed(self.seed)

        ProgressLogger.start("Calling model.generate")

        completions = self.model.generate(
            **batchenc,
            max_new_tokens=self.cfg.response_max_len,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.use_sampling,
            temperature=self.cfg.sampling_temp,
            length_penalty=self.cfg.length_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            repetition_penalty=self.cfg.repetition_penalty,
            logits_processor=logit_processors,
            output_scores=True if return_dict_in_generate else False,
            return_dict_in_generate=return_dict_in_generate
        )
        ProgressLogger.stop()

        if return_dict_in_generate:
            scores = completions.scores
            scores = [i.detach().cpu() for i in scores]
            completions = completions.sequences
        
        if is_decoder_only_model(self.cfg.name):
            completions = completions[:, batchenc["input_ids"].shape[-1] :]
        elif "Qwen" in self.cfg.name:
            completions = completions[:, batchenc["input_ids"].shape[-1] :]

        if completions.shape[-1] == self.cfg.response_max_len:
            print(
                f"At least one response seems to be truncated to {self.cfg.response_max_len}",
                tag="!!!",
                color="red",
                tag_color="red",
            )

        completions_str: List[str] = self.tokenizer.batch_decode(
            completions, skip_special_tokens=True
        )
        print(f"Model Output with shape {completions.shape}", color="purple")

        if return_model_inputs:
            return completions_str, actual_inputs
        if return_dict_in_generate:
            return completions_str, scores, completions
        return completions_str

    def _get_actual_input_str(
        self, inputs: List[str], batchenc: BatchEncoding
    ) -> List[str]:
        last_idx = -1 if "dipper" not in self.cfg.name else -2
        idxs = batchenc["offset_mapping"][
            :, last_idx, 1
        ]  # cutoff was here - Location of last token in the input_text
        actual_inputs = []
        for inp, idx in zip(inputs, idxs):
            inp = inp[: min(len(inp) - 8, idx)]
            inp = inp[10:]

            actual_inputs.append(inp)
        return actual_inputs

    def get_logits(self, input_ids):
        return self.model(input_ids=input_ids)


class APIModel:
    def __init__(self, model_name: str, temperature=0.1) -> None:
        if model_name not in ["gpt4o", "gpt3.5", "gpt4o-mini", "gpt5"]:
            raise RuntimeError(f"Unknown model name: {model_name}")
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name
        self.full_model_name = full_model_names[model_name]
        self.temperature = temperature
        self.enc = tiktoken.encoding_for_model(self.full_model_name)
    
    def get_gpt_token_ids(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def _query_api(
        self, sysprompt: str, ID: int, prompt: str, response_format: Any = None, max_tokens:int = 1500
    ) -> str:
        try:
            payload = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": prompt},
            ]
            # Use max_completion_tokens for newer models like GPT-5
            if self.full_model_name.startswith("gpt-5"):
                # GPT-5 only supports default temperature (1), so omit the parameter
                request = partial(
                    self.client.chat.completions.create,
                    model=self.full_model_name,
                    messages=payload,
                    max_completion_tokens=max_tokens,
                )
            else:
                request = partial(
                    self.client.chat.completions.create,
                    model=self.full_model_name,
                    messages=payload,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
            if response_format is None:
                response = request().choices[0].message.content
            else:
                response = (
                    request(response_format=response_format)
                    .choices[0]
                    .message.content
                )
        except Exception as e:
            # we need to know on which ID we crashed
            # Store ID in a custom attribute for Python 3.10 compatibility
            e.__notes__ = [str(ID)]
            raise e
        return response
    
    def _query_api_with_logit_bias(
        self, sysprompt: str, ID: int, prompt: str, response_format: Any = None, logit_bias: float = 0.0, logit_bias_tokens: List[int] = None, max_tokens:int = 1500
    ) -> str:
        try:
            payload = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": prompt},
            ]
            logit_bias_dict = {token: logit_bias for token in logit_bias_tokens}
            # Use max_completion_tokens for newer models like GPT-5
            if self.full_model_name.startswith("gpt-5"):
                # GPT-5 only supports default temperature (1), so omit the parameter
                request = partial(
                    self.client.chat.completions.create,
                    model=self.full_model_name,
                    messages=payload,
                    max_completion_tokens=max_tokens,
                    logit_bias=logit_bias_dict
                )
            else:
                request = partial(
                    self.client.chat.completions.create,
                    model=self.full_model_name,
                    messages=payload,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias_dict
                )
            if response_format is None:
                response = request().choices[0].message.content
            else:
                response = (
                    request(response_format=response_format)
                    .choices[0]
                    .message.content
                )
        except Exception as e:
            # we need to know on which ID we crashed
            # Store ID in a custom attribute for Python 3.10 compatibility
            e.__notes__ = [str(ID)]
            raise e
        return response

    def generate(
        self, sysprompt: str, prompts: List[str], response_format: Any = None, max_tokens:int = 1500
    ):
        max_workers = 16
        base_timeout = 240
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(prompts)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(prompts):
                results = executor.map(
                    lambda id: (
                        id,
                        prompts[id],
                        self._query_api(
                            sysprompt, id, prompts[id], response_format=response_format, max_tokens=max_tokens
                        ),
                    ),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Queries",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield answer
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except openai.RateLimitError as r:
                    print(f"Rate Limit: {r}")
                    time.sleep(10)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    if (
                        getattr(e, "type", "default") == "invalid_prompt"
                        and len(ids_to_do) == 1
                    ):
                        # We skip invalid prompts for GPT -> This only works with bs=1
                        ids_to_do = []
                        yield "invalid_prompt_error"
                    elif "maximum context length" in e.message:
                        # how much are we over?
                        pattern = r"(\d+) tokens. However, you requested (\d+)"
                        matches = re.findall(pattern, e.message)
                        max_len, curr_len = matches[0]
                        chars_to_cut = max(
                            (int(curr_len) - int(max_len)) * 4, 400
                        )  # estimate but never below 400 (~100 toks)

                        #  presumably the first ID is the one that crashed?
                        curr_id = int(e.__notes__[0])
                        assert curr_id in ids_to_do

                        # split and shorten
                        toks = prompts[curr_id].split("</documents>")
                        if len(toks) == 2:
                            toks[0] = toks[0][:-chars_to_cut]  # drop the last chars_to_cut chars
                            prompts[curr_id] = f"{toks[0]}</documents>{toks[1]}"
                            print(
                                f"Shortened prompt {curr_id} by {chars_to_cut} chars."
                            )
                        else:
                            print(
                                f"Tried to shorten the prompt but there was an error....Got {len(toks)} toks."
                            )
                        time.sleep(3)
                    else:
                        time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(base_timeout, timeout)
                retry_ctr += 1
    
    def generate_with_logit_bias(
        self, sysprompt: str, prompts: List[str], response_format: Any = None, logit_bias: float = 0.0, logit_bias_tokens: List[int] = None, max_tokens:int = 1500
    ):
        max_workers = 16
        base_timeout = 240
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(prompts)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(prompts):
                if logit_bias_tokens is not None:
                    results = executor.map(
                        lambda id: (
                            id,
                            prompts[id],
                            self._query_api_with_logit_bias(
                                sysprompt, id, prompts[id], response_format=response_format, logit_bias=logit_bias, logit_bias_tokens=logit_bias_tokens, max_tokens=max_tokens
                            ),
                        ),
                        ids_to_do,
                        timeout=timeout,
                    )
                else:
                    results = executor.map(
                        lambda id: (
                            id,
                            prompts[id],
                            self._query_api(
                                sysprompt, id, prompts[id], response_format=response_format, max_tokens=max_tokens
                            ),
                        ),
                        ids_to_do,
                        timeout=timeout,
                    )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Queries",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield answer
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except openai.RateLimitError as r:
                    print(f"Rate Limit: {r}")
                    time.sleep(10)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    if (
                        getattr(e, "type", "default") == "invalid_prompt"
                        and len(ids_to_do) == 1
                    ):
                        # We skip invalid prompts for GPT -> This only works with bs=1
                        ids_to_do = []
                        yield "invalid_prompt_error"
                    elif "maximum context length" in e.message:
                        # how much are we over?
                        pattern = r"(\d+) tokens. However, you requested (\d+)"
                        matches = re.findall(pattern, e.message)
                        max_len, curr_len = matches[0]
                        chars_to_cut = max(
                            (int(curr_len) - int(max_len)) * 4, 400
                        )  # estimate but never below 400 (~100 toks)

                        #  presumably the first ID is the one that crashed?
                        curr_id = int(e.__notes__[0])
                        assert curr_id in ids_to_do

                        # split and shorten
                        toks = prompts[curr_id].split("</documents>")
                        if len(toks) == 2:
                            toks[0] = toks[0][:-chars_to_cut]  # drop the last chars_to_cut chars
                            prompts[curr_id] = f"{toks[0]}</documents>{toks[1]}"
                            print(
                                f"Shortened prompt {curr_id} by {chars_to_cut} chars."
                            )
                        else:
                            print(
                                f"Tried to shorten the prompt but there was an error....Got {len(toks)} toks."
                            )
                        time.sleep(3)
                    else:
                        time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(base_timeout, timeout)
                retry_ctr += 1

