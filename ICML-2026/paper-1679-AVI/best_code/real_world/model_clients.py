

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict

from .config import HuggingFaceConfig, ModelConfig

logger = logging.getLogger(__name__)


class BaseModelClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.name = config.name

    def generate(self, question: str) -> str:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.config)


class StubModelClient(BaseModelClient):


    def generate(self, question: str) -> str:
        return f"[STUB:{self.name}] {question}"


class HFModelClient(BaseModelClient):

    def __init__(self, config: ModelConfig, hf_config: HuggingFaceConfig) -> None:
        super().__init__(config)
        self.hf_config = hf_config
        self._load_model()

    def _load_model(self) -> None:
        import os
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        if self.hf_config.cache_dir:
            os.environ["HF_HOME"] = self.hf_config.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.hf_config.cache_dir
            os.environ["HF_HUB_CACHE"] = self.hf_config.cache_dir
            logger.info(f"Set model cache directory: {self.hf_config.cache_dir}")

        seq2seq_models = [
            "blenderbot", "t5", "bart", "pegasus", "mbart", "marian",
            "fsmt", "bigbird_pegasus", "led", "longt5"
        ]
        self.is_seq2seq = any(model_type in self.name.lower() for model_type in seq2seq_models)
        
        common_kwargs = {}
        if self.hf_config.cache_dir:
            cache_dir_str = str(self.hf_config.cache_dir)
            common_kwargs["cache_dir"] = cache_dir_str

        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.hf_config.trust_remote_code,
            **common_kwargs,
        }
        model_kwargs: Dict[str, Any] = {
            "device_map": self.hf_config.device_map,
            "trust_remote_code": self.hf_config.trust_remote_code,
            "low_cpu_mem_usage": self.hf_config.low_cpu_mem_usage,
            **common_kwargs,
        }

        if self.hf_config.max_memory:
            model_kwargs["max_memory"] = self.hf_config.max_memory

        if self.hf_config.revision:
            tokenizer_kwargs["revision"] = self.hf_config.revision
            model_kwargs["revision"] = self.hf_config.revision

        dtype = None
        quantization = self.hf_config.quantization

        if quantization:
            try:
                from bitsandbytes import __version__  # noqa: F401
                from transformers import BitsAndBytesConfig
                import torch

                compute_dtype = getattr(torch, self.hf_config.dtype, torch.float16)
                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type="nf4",
                    )
                elif quantization == "8bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    raise ValueError("Quantization only supports '4bit' or '8bit'")
                model_kwargs["quantization_config"] = quant_config
            except ImportError as exc:
                raise ImportError(
                    "Need to install bitsandbytes to use quantization loading."
                ) from exc
        else:
            import torch

            dtype = getattr(torch, self.hf_config.dtype, torch.float16)
            model_kwargs["torch_dtype"] = dtype

        logger.info("Loading model %s (seq2seq=%s)", self.name, self.is_seq2seq)
        
        tokenizer_loaded = False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.name,
                **tokenizer_kwargs,
            )
            tokenizer_loaded = True
            logger.info("✅ Successfully loaded tokenizer with full parameters")
        except (TypeError, ValueError, OSError) as e:
            logger.warning(f"Failed to load tokenizer with full parameters: {e}")
        
        if not tokenizer_loaded:
            try:
                logger.info("Trying strategy 2: using slow tokenizer...")
                slow_tokenizer_kwargs = tokenizer_kwargs.copy()
                slow_tokenizer_kwargs["use_fast"] = False
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name,
                    **slow_tokenizer_kwargs,
                )
                tokenizer_loaded = True
                logger.info("✅ Successfully loaded slow tokenizer")
            except Exception as e2:
                logger.warning(f"Failed to load slow tokenizer: {e2}")
        
        if not tokenizer_loaded:
            try:
                logger.info("Trying strategy 3: using environment variables (without cache_dir)...")
                minimal_kwargs = {
                    "trust_remote_code": self.hf_config.trust_remote_code,
                }
                if self.hf_config.revision:
                    minimal_kwargs["revision"] = self.hf_config.revision
                    
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name,
                    **minimal_kwargs,
                )
                tokenizer_loaded = True
                logger.info("✅ Successfully loaded tokenizer with environment variables")
            except Exception as e3:
                logger.warning(f"Failed to load tokenizer with environment variables: {e3}")
        
        if not tokenizer_loaded:
            try:
                logger.info("Trying strategy 4: slow tokenizer + environment variables...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name,
                    use_fast=False,
                    trust_remote_code=self.hf_config.trust_remote_code,
                )
                tokenizer_loaded = True
                logger.info("✅ Successfully loaded slow tokenizer (without cache_dir)")
            except Exception as e4:
                logger.error(f"❌ All strategies failed, cannot load tokenizer")
                logger.error(f"Last error: {e4}")
                logger.error(f"Suggestions: 1) Check if the model name is correct")
                logger.error(f"     2) Check network connection")
                logger.error(f"     3) Try to manually download model to cache directory")
                raise RuntimeError(f"Cannot load tokenizer: {self.name}") from e4
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        ModelClass = AutoModelForSeq2SeqLM if self.is_seq2seq else AutoModelForCausalLM
        
        # Fix: Force eager attention for Qwen models to avoid SdpaAttention NaN
        is_qwen = 'qwen' in self.name.lower()
        if is_qwen and 'attn_implementation' not in model_kwargs:
            model_kwargs['attn_implementation'] = 'eager'
            logger.info('Qwen model detected: forcing attn_implementation=eager to avoid NaN')
        
        try:
            self.model = ModelClass.from_pretrained(
                self.name,
                **model_kwargs,
            )
        except (TypeError, ValueError, OSError) as e:
            logger.warning(f"Failed to load model with full parameters: {e}")
            logger.info("Trying to load model with environment variables...")
            
            minimal_model_kwargs = {
                "device_map": self.hf_config.device_map,
                "trust_remote_code": self.hf_config.trust_remote_code,
                "low_cpu_mem_usage": self.hf_config.low_cpu_mem_usage,
            }
            
            if self.hf_config.max_memory:
                minimal_model_kwargs["max_memory"] = self.hf_config.max_memory
            if self.hf_config.revision:
                minimal_model_kwargs["revision"] = self.hf_config.revision
            if "torch_dtype" in model_kwargs:
                minimal_model_kwargs["torch_dtype"] = model_kwargs["torch_dtype"]
            if "quantization_config" in model_kwargs:
                minimal_model_kwargs["quantization_config"] = model_kwargs["quantization_config"]
            if "attn_implementation" in model_kwargs:
                minimal_model_kwargs["attn_implementation"] = model_kwargs["attn_implementation"]
            
            try:
                self.model = ModelClass.from_pretrained(
                    self.name,
                    **minimal_model_kwargs,
                )
                logger.info("✅ Successfully loaded model with environment variables")
            except Exception as e2:
                logger.error(f"❌ Failed to load model completely: {e2}")
                raise

    def _format_prompt(self, question: str) -> str:
        if self.is_seq2seq:
            return question
        
        if self.config.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": question})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        template = self.config.resolved_prompt_template
        return template.format(question=question)

    def generate(self, question: str) -> str:
        import torch

        do_sample = self.config.temperature > 0
        generation_kwargs = dict(self.config.generation_kwargs)
        generation_kwargs.setdefault("max_new_tokens", self.config.max_new_tokens)
        generation_kwargs.setdefault("temperature", self.config.temperature)
        generation_kwargs.setdefault("top_p", self.config.top_p)
        generation_kwargs.setdefault("do_sample", do_sample)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        prompt = self._format_prompt(question)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
        )
        
        model_inputs = {}
        if "input_ids" in encoded:
            model_inputs["input_ids"] = encoded["input_ids"].to(self.model.device)
        if "attention_mask" in encoded:
            model_inputs["attention_mask"] = encoded["attention_mask"].to(self.model.device)
        
        if "token_type_ids" in encoded and hasattr(self.model.config, "type_vocab_size"):
            if self.model.config.type_vocab_size > 1:
                model_inputs["token_type_ids"] = encoded["token_type_ids"].to(self.model.device)

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
            except RuntimeError as e:
                if 'inf' in str(e) or 'nan' in str(e) or 'multinomial' in str(e):
                    import logging
                    _log = logging.getLogger(__name__)
                    _log.debug(f'Multinomial error, retrying with greedy sampling: {e}')
                    greedy_kwargs = dict(generation_kwargs)
                    greedy_kwargs['do_sample'] = False
                    greedy_kwargs.pop('temperature', None)
                    greedy_kwargs.pop('top_p', None)
                    outputs = self.model.generate(
                        **model_inputs,
                        **greedy_kwargs,
                    )
                else:
                    raise

        if self.is_seq2seq:
            generated_tokens = outputs[0]
        else:
            generated_tokens = outputs[0][model_inputs["input_ids"].shape[-1]:]
        
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        for stop_word in self.config.stop_sequences:
            if stop_word in text:
                text = text.split(stop_word)[0]
                break
        return text.strip()


def build_model_client(
    config: ModelConfig,
    hf_config: HuggingFaceConfig,
) -> BaseModelClient:
    provider = (config.provider or "hf").lower()
    if provider == "stub":
        return StubModelClient(config)
    if provider == "hf":
        return HFModelClient(config, hf_config)
    raise ValueError(f"Unknown model provider: {config.provider}")


