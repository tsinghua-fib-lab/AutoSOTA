"""Base language model wrapper used by decoding pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize_attn_implementation(attn_implementation: Optional[str]) -> Optional[str]:
    if attn_implementation is None:
        return None
    text = str(attn_implementation).strip()
    return text or None


@dataclass
class TokenizerEncoding:
    input_ids: Tensor
    attention_mask: Optional[Tensor]


class BaseLanguageModel:
    """Lightweight wrapper around a HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        self.device = _resolve_device(device)
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt or "You are a helpful assistant."
        dtype = getattr(torch, torch_dtype) if torch_dtype else None
        attn_impl = _normalize_attn_implementation(attn_implementation)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        # Decoder-only models (e.g., Qwen) should use left padding for generation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Many causal models do not define pad tokens. Fall back to EOS so batching works.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()
        
        # 检测是否支持 chat template
        self._has_chat_template = (
            hasattr(self.tokenizer, "chat_template") 
            and self.tokenizer.chat_template is not None
        ) or hasattr(self.tokenizer, "apply_chat_template")

    def encode_prompt(self, prompt: str) -> TokenizerEncoding:
        # 如果是 Chat 模型且启用 chat template，使用 apply_chat_template
        if self.use_chat_template and self._has_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                return TokenizerEncoding(
                    input_ids=input_ids.to(self.device),
                    attention_mask=torch.ones_like(input_ids).to(self.device),
                )
            except Exception:
                # 如果 apply_chat_template 失败，回退到普通编码
                pass
        
        # 普通编码（completion 模型或回退）
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
        )
        attention_mask = encoded.get("attention_mask", None)
        return TokenizerEncoding(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
        )

    def decode(self, token_ids: Tensor) -> str:
        token_list = token_ids.tolist() if isinstance(token_ids, Tensor) else token_ids
        return self.tokenizer.decode(
            token_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def max_positions(self) -> Optional[int]:
        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            return max_pos
        tok_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_limit, int) and 0 < tok_limit < 10**6:
            return tok_limit
        return None

    def step(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional = None,
    ) -> Dict[str, Tensor]:
        # 使用 KV cache 时只传最后一个 token 的 input_ids
        step_input_ids = input_ids if past_key_values is None else input_ids[:, -1:]
        # 但 attention_mask 必须是完整序列的（KV cache 需要知道完整的 attention 信息）
        step_attention_mask = attention_mask  # 始终使用完整的 attention_mask
        with torch.inference_mode():
            outputs = self.model(
                input_ids=step_input_ids,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        return {
            "logits": logits.squeeze(0),
            "past_key_values": outputs.past_key_values,
        }

    def expand_inputs(
        self,
        encoding: TokenizerEncoding,
        next_token_id: int,
    ) -> TokenizerEncoding:
        next_token = torch.tensor([[next_token_id]], device=self.device)
        input_ids = torch.cat((encoding.input_ids, next_token), dim=-1)
        attention_mask = None
        if encoding.attention_mask is not None:
            attention_one = torch.ones_like(next_token)
            attention_mask = torch.cat((encoding.attention_mask, attention_one), dim=-1)
        return TokenizerEncoding(input_ids=input_ids, attention_mask=attention_mask)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)
