from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


@dataclass(frozen=True)
class TextBuildConfig:
    model_name_or_path: str
    model_arch: str = "auto"  # "llama" | "t5" | "auto"
    device: str = "cuda"
    dtype: str | None = None  # "fp16" | "bf16" | "fp32" | None
    model_kind: str = "causal_lm"  # "causal_lm" | "sequence_classification"
    num_labels: int = 3
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True


class TextLM(nn.Module):
    """
    Thin wrapper around a HuggingFace text model with unified eval APIs.
    """

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _is_spiece_parse_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("spiece.model" in msg) and ("error parsing line" in msg or "sentencepiece" in msg)

    @staticmethod
    def build(cfg: TextBuildConfig) -> TextLM:
        try:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except Exception as e:
            raise ImportError("Text model support requires transformers to be installed.") from e

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if cfg.dtype is not None and cfg.dtype not in dtype_map:
            raise ValueError(f"Unknown dtype '{cfg.dtype}'. Choose from: {sorted(dtype_map)}")
        torch_dtype = dtype_map.get(cfg.dtype, None)

        arch = str(cfg.model_arch).strip().lower()
        if arch not in {"llama", "t5", "auto"}:
            raise ValueError("model_arch must be one of: llama, t5, auto")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name_or_path,
                trust_remote_code=bool(cfg.trust_remote_code),
                use_fast=bool(cfg.use_fast_tokenizer),
            )
        except Exception as e:
            if bool(cfg.use_fast_tokenizer) and TextLM._is_spiece_parse_error(e):
                print(
                    "[warn] Fast tokenizer failed to parse sentencepiece model; "
                    "falling back to slow tokenizer (use_fast=False)."
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model_name_or_path,
                    trust_remote_code=bool(cfg.trust_remote_code),
                    use_fast=False,
                )
            else:
                raise
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_cfg = AutoConfig.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=bool(cfg.trust_remote_code),
        )
        is_encoder_decoder = bool(getattr(hf_cfg, "is_encoder_decoder", False))

        kind = str(cfg.model_kind).strip().lower()
        common = {
            "pretrained_model_name_or_path": cfg.model_name_or_path,
            "trust_remote_code": bool(cfg.trust_remote_code),
            "torch_dtype": torch_dtype,
        }
        if kind == "causal_lm":
            use_seq2seq = (arch == "t5") or (arch == "auto" and is_encoder_decoder)
            if use_seq2seq:
                model = AutoModelForSeq2SeqLM.from_pretrained(**common)
            else:
                model = AutoModelForCausalLM.from_pretrained(**common)
        elif kind == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                **common,
                num_labels=int(cfg.num_labels),
            )
        else:
            raise ValueError("model_kind must be one of: causal_lm, sequence_classification")

        # Sequence-classification forward with batch_size > 1 requires pad_token_id.
        if getattr(model.config, "pad_token_id", None) is None:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = tokenizer.bos_token_id
            if pad_id is None:
                pad_id = 0
            model.config.pad_token_id = int(pad_id)

        model = model.to(cfg.device)
        model.eval()

        return TextLM(model=model, tokenizer=tokenizer)

    def _is_encoder_decoder(self) -> bool:
        return bool(getattr(self.model.config, "is_encoder_decoder", False))

    def _context_limit(self) -> int:
        raw = getattr(self.model.config, "max_position_embeddings", None)
        if raw is None:
            raw = getattr(self.model.config, "n_positions", 4096)
        try:
            return max(32, int(raw))
        except Exception:
            return 4096

    def _encode_text(self, text: str) -> list[int]:
        toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if isinstance(toks, list):
            return [int(t) for t in toks]
        raise TypeError(f"Unexpected tokenizer output type: {type(toks)}")

    def _ensure_prefix(self, prefix_ids: list[int]) -> list[int]:
        if prefix_ids:
            return prefix_ids
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if bos_id is not None:
            return [int(bos_id)]
        if eos_id is not None:
            return [int(eos_id)]
        return [0]

    @torch.no_grad()
    def _suffix_logprob_causal(
        self,
        *,
        prefix_ids: list[int],
        suffix_ids: list[int],
        device: str,
        max_prompt_tokens: int | None,
    ) -> float:
        if not suffix_ids:
            return float("-inf")

        pfx = list(prefix_ids)
        sfx = list(suffix_ids)
        pfx = self._ensure_prefix(pfx)

        ctx_limit = self._context_limit()
        if max_prompt_tokens is not None:
            pfx = pfx[-max(1, int(max_prompt_tokens)) :]

        max_prefix = max(1, ctx_limit - len(sfx))
        if len(pfx) > max_prefix:
            pfx = pfx[-max_prefix:]

        ids = torch.tensor([pfx + sfx], dtype=torch.long, device=device)
        logits = self.model(input_ids=ids).logits[0]  # [L, V]
        if logits.ndim != 2:
            raise ValueError(
                "Prompt scoring requires causal-LM token logits with shape [L, V], "
                f"but got shape {tuple(logits.shape)}. "
                "Use model_kind='causal_lm' for prompt mode."
            )
        log_probs = torch.log_softmax(logits, dim=-1)

        prefix_len = len(pfx)
        score = 0.0
        for j, tok in enumerate(sfx):
            pos = prefix_len + j - 1
            score += float(log_probs[pos, int(tok)].item())
        return score

    @torch.no_grad()
    def _suffix_logprob_encoder_decoder(
        self,
        *,
        prefix_ids: list[int],
        suffix_ids: list[int],
        device: str,
        max_prompt_tokens: int | None,
    ) -> float:
        if not suffix_ids:
            return float("-inf")

        enc = list(prefix_ids)
        if max_prompt_tokens is not None:
            enc = enc[-max(1, int(max_prompt_tokens)) :]
        if not enc:
            enc = self._ensure_prefix(enc)

        enc_ids = torch.tensor([enc], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(enc_ids)

        start_id = getattr(self.model.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = self.tokenizer.pad_token_id
        if start_id is None:
            start_id = self.tokenizer.eos_token_id
        if start_id is None:
            start_id = self.tokenizer.bos_token_id
        if start_id is None:
            start_id = 0

        dec_in = [int(start_id)] + [int(x) for x in suffix_ids[:-1]]
        dec_ids = torch.tensor([dec_in], dtype=torch.long, device=device)

        logits = self.model(
            input_ids=enc_ids,
            attention_mask=attention_mask,
            decoder_input_ids=dec_ids,
        ).logits[0]
        if logits.ndim != 2:
            raise ValueError(
                "Prompt scoring requires seq2seq token logits with shape [T, V], "
                f"but got shape {tuple(logits.shape)}. "
                "Use model_kind='causal_lm' for prompt mode."
            )
        log_probs = torch.log_softmax(logits, dim=-1)

        score = 0.0
        for j, tok in enumerate(suffix_ids):
            score += float(log_probs[j, int(tok)].item())
        return score

    @torch.no_grad()
    def nli_accuracy(
        self,
        *,
        examples: list[Any],
        label_texts: list[str],
        prompt_template: str,
        device: str | None = None,
        max_prompt_tokens: int | None = None,
        print_every: int | None = None,
    ) -> float:
        """
        Evaluate accuracy for NLI examples.
        Each example must expose: premise, hypothesis, label.
        """
        if not examples:
            raise ValueError("No examples provided.")
        if not label_texts:
            raise ValueError("No label_texts provided.")

        dev = device or str(next(self.model.parameters()).device)
        self.model.eval()

        label_prefix = "" if self._is_encoder_decoder() else " "
        answer_token_ids = [self._encode_text(label_prefix + txt.strip()) for txt in label_texts]
        if any(len(ids) == 0 for ids in answer_token_ids):
            raise ValueError("At least one label_text tokenized to an empty token list.")

        correct = 0
        total = 0

        for i, ex in tqdm(enumerate(examples), total=len(examples), desc="Evaluating NLI examples"):
            premise = str(ex.premise).strip()
            hypothesis = str(ex.hypothesis).strip()
            gold = int(ex.label)

            prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)
            prompt_ids = self._encode_text(prompt)

            if self._is_encoder_decoder():
                scores = [
                    self._suffix_logprob_encoder_decoder(
                        prefix_ids=prompt_ids,
                        suffix_ids=ans_ids,
                        device=dev,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    for ans_ids in answer_token_ids
                ]
            else:
                scores = [
                    self._suffix_logprob_causal(
                        prefix_ids=prompt_ids,
                        suffix_ids=ans_ids,
                        device=dev,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    for ans_ids in answer_token_ids
                ]
            pred = int(max(range(len(scores)), key=lambda j: scores[j]))

            if pred == gold:
                correct += 1
            total += 1

            if print_every is not None and int(print_every) > 0 and (i + 1) % int(print_every) == 0:
                print(f"  progress: {i + 1}/{len(examples)}")

        if total <= 0:
            raise ValueError("No valid examples were evaluated.")
        return float(correct) / float(total)

    @torch.no_grad()
    def sequence_classification_accuracy(
        self,
        loader: Any,
        *,
        device: str | None = None,
        mask_class: list[int] | None = None,
        print_every: int | None = None,
    ) -> float:
        """
        Evaluate classification accuracy from model logits.
        Expects model output logits of shape [B, C].
        """
        dev = device or str(next(self.model.parameters()).device)
        self.model.eval()

        mask = None if mask_class is None else [int(x) for x in mask_class]
        correct = 0
        total = 0

        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            labels = batch["labels"].to(dev).long()

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            if logits.ndim != 2:
                raise ValueError(
                    f"Head-logits mode expects [B, C] logits, got shape {tuple(logits.shape)}. "
                    "Use prompt mode for generative evaluation."
                )

            if mask is not None:
                idx = torch.tensor(mask, device=logits.device, dtype=torch.long)
                masked = logits.index_select(dim=1, index=idx)
                pred_local = masked.argmax(dim=-1)
                pred = idx[pred_local]
            else:
                pred = logits.argmax(dim=-1)

            correct += int((pred == labels).sum().item())
            total += int(labels.numel())

            if print_every is not None and int(print_every) > 0 and (i + 1) % int(print_every) == 0:
                print(f"  progress: {i + 1}")

        if total <= 0:
            raise ValueError("No samples were evaluated.")
        return float(correct) / float(total)
