import math
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from shared_utils import (
    DEFAULT_GENERATE_KWARGS,
    DEFAULT_PROMPT_TEMPLATE,
)


class LLMAttributionEvaluator():
    def __init__(
        self, 
        model: Any, 
        tokenizer: Any, 
        generate_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.generated_ids = None
        self.prompt_ids = None
        
        self.model.eval()
    
    def format_prompt(self, prompt) -> str:
        modified_prompt = DEFAULT_PROMPT_TEMPLATE.format(context = prompt, query = "")
        formatted_prompt = [{"role": "user", "content": modified_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            formatted_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        return formatted_prompt

    # Query the model for its generation
    # This internally saves the input and generated token ids
    def response(self, prompt) -> Tuple[str, str]:
        formatted_prompt = self.format_prompt(" " + prompt)

        model_input = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens = False).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(model_input.input_ids, **self.generate_kwargs) # [1, num_prompt_tokens + num_generations]
            # Get only the prompt tokens (excluding the prompt)
            self.prompt_ids = outputs[:, :model_input.input_ids.shape[1]] # [1, num_prompt_tokens]
            # Get only the generated tokens (excluding the prompt)
            self.generated_ids = outputs[:, model_input.input_ids.shape[1]:] # [1, num_generations]

        return self.tokenizer.decode(self.generated_ids[0], skip_special_tokens=True), self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    #  we want to evaluate the probability of producing a reponse given a prompt
    def compute_logprob_response_given_prompt(self, prompt_ids, response_ids) -> torch.Tensor:
        """
        Compute log-probabilities of `response_ids` given `prompt_ids`.

        prompt_ids: [B, N]
        response_ids: [B, M]
        Returns: [B, M]
        """
        # concat prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)   # [B, N+M]
        attention_mask = torch.ones_like(input_ids)

        # Get model outputs
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, seq_len, vocab_size]

        # Compute log-probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]

        # Only consider response tokens
        response_start = prompt_ids.shape[1]

        # Align logits to predict each y_t from y_{<t}
        logits_for_response = log_probs[:, response_start - 1: -1, :]  # [B, M, vocab]

        # Gather log-probs for the actual response tokens
        gathered = logits_for_response.gather(2, response_ids.unsqueeze(-1))  # [B, M, 1]
        return gathered.squeeze(-1)  # [B, M]

    def _ensure_pad_token_id(self) -> int:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id; cannot define baseline token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return int(self.tokenizer.pad_token_id)

    def _find_subsequence_start(self, haystack: torch.Tensor, needle: torch.Tensor) -> Optional[int]:
        if haystack.ndim != 1 or needle.ndim != 1:
            raise ValueError("Expected 1D tensors for subsequence matching.")
        if needle.numel() == 0:
            return 0
        hay_len = int(haystack.numel())
        needle_len = int(needle.numel())
        if needle_len > hay_len:
            return None
        for i in range(hay_len - needle_len + 1):
            if torch.equal(haystack[i : i + needle_len], needle):
                return i
        return None

    def get_topk_tokens(self, attr_matrix, text_list, topk = 10) -> torch.Tensor:
        input_len = len(text_list)
        input_col_sums = attr_matrix.sum(0).clamp(0)[0 : input_len]
        topk_cols = torch.topk(input_col_sums, topk)[1]

        return torch.sort(topk_cols)[0]

    def add_dummy_facts_to_prompt(self, text_sentences) -> List[str]:
        # create dummy fact sentences
        dummy_sentences = []
        for i in range(len(text_sentences)):
            dummy_sentences.append(" Unrelated Sentence.")

        # Interleave the dummy facts
        result = []
        for x, y in zip(text_sentences, dummy_sentences):
            result.append(x)
            result.append(y)

        # add back on the last sentence that we left out
        return result

    def faithfulness_test(
        self,
        attribution: torch.Tensor,
        prompt: str,
        generation: str,
        *,
        k: int = 20,
    ) -> Tuple[float, float, float]:
        """Token-level MAS/RISE faithfulness via guided deletion in k perturbation steps (no optimization).

        attribution: [R, P] token attribution on *prompt-side tokens* only.
        prompt: raw prompt string (NOT sentence-segmented).
        generation: target generation string (think + output); scored as generation + eos.
        k: number of perturbation steps; each step perturbs ~1/k of prompt tokens.
        """

        def auc(arr: np.ndarray) -> float:
            return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / max(1, (arr.shape[0] - 1))

        pad_token_id = self._ensure_pad_token_id()

        # Leading-space convention must match attribution path (" " + prompt).
        user_prompt = " " + prompt
        formatted_prompt = self.format_prompt(user_prompt)

        # Tokenize (CPU for span finding, then move to device).
        formatted_ids = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        user_ids = self.tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        user_start = self._find_subsequence_start(formatted_ids[0], user_ids[0])
        if user_start is None:
            raise RuntimeError("Failed to locate user prompt token span inside formatted chat prompt.")

        prompt_ids = formatted_ids.to(self.device)
        prompt_ids_perturbed = prompt_ids.clone()
        generation_ids = self.tokenizer(
            generation + self.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)

        # Compute guided deletion ordering over prompt-side tokens.
        attr_cpu = attribution.detach().cpu()
        w = attr_cpu.sum(0)
        sorted_attr_indices = torch.argsort(w, descending=True)
        attr_sum = float(w.sum().item())

        P = int(w.numel())
        if int(user_ids.shape[1]) != P:
            raise ValueError(
                "Prompt-side attribution length does not match tokenized user prompt length: "
                f"attr P={P}, user_prompt P={int(user_ids.shape[1])}."
            )
        if P > 0:
            steps = int(k) if k is not None else 0
            if steps <= 0:
                steps = 1
            steps = min(steps, P)
        else:
            steps = 0

        scores = np.zeros(steps + 1, dtype=np.float64)
        density = np.zeros(steps + 1, dtype=np.float64)

        scores[0] = self.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
        density[0] = 1.0

        if P == 0:
            return auc(scores), auc(scores), auc(scores)

        if attr_sum <= 0:
            density = np.linspace(1.0, 0.0, steps + 1)

        base = P // steps
        remainder = P % steps
        start = 0
        for step in range(steps):
            size = base + (1 if step < remainder else 0)
            group = sorted_attr_indices[start : start + size]
            start += size

            for idx in group:
                j = int(idx.item())
                prompt_ids_perturbed[0, user_start + j] = pad_token_id
            scores[step + 1] = (
                self.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
            )
            if attr_sum > 0:
                dec = float(w.index_select(0, group).sum().item()) / attr_sum
                density[step + 1] = density[step] - dec

        min_normalized_pred = 1.0
        normalized_model_response = scores.copy()
        for i in range(len(scores)):
            normalized_pred = (normalized_model_response[i] - scores[-1]) / (abs(scores[0] - scores[-1]))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            min_normalized_pred = min(min_normalized_pred, normalized_pred)
            normalized_model_response[i] = min_normalized_pred

        alignment_penalty = np.abs(normalized_model_response - density)
        corrected_scores = normalized_model_response + alignment_penalty
        corrected_scores = corrected_scores.clip(0.0, 1.0)
        corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))

        if np.isnan(corrected_scores).any():
            corrected_scores = np.linspace(1.0, 0.0, len(scores))

        return auc(normalized_model_response), auc(corrected_scores), auc(normalized_model_response + alignment_penalty)

    def evaluate_attr_recovery(
        self,
        attribution: torch.Tensor,
        *,
        prompt_len: int,
        gold_prompt_token_indices: List[int],
        top_fraction: float = 0.1,
    ) -> float:
        """Recall of gold prompt tokens among top-attributed prompt tokens.

        Ranking excludes model-generated tokens by restricting to prompt-side tokens [0, prompt_len).
        """
        if attribution.ndim != 2:
            raise ValueError("Expected 2D token-level attribution matrix [G, P+G].")
        if prompt_len <= 0:
            return float("nan")
        if int(attribution.shape[1]) < int(prompt_len):
            raise ValueError(
                "prompt_len exceeds attribution width: "
                f"prompt_len={int(prompt_len)} attribution_cols={int(attribution.shape[1])}."
            )

        gold: set[int] = set()
        for raw in gold_prompt_token_indices or []:
            try:
                idx = int(raw)
            except Exception:
                continue
            if 0 <= idx < int(prompt_len):
                gold.add(idx)
        if not gold:
            return float("nan")

        w = torch.nan_to_num(attribution[:, :prompt_len].sum(0).to(dtype=torch.float32), nan=0.0).clamp(min=0.0)
        k = max(1, int(math.ceil(float(prompt_len) * float(top_fraction))))
        k = min(k, int(prompt_len))
        topk = torch.topk(w, k, largest=True).indices.tolist()
        hit = len(set(topk).intersection(gold))
        return float(hit) / float(len(gold))

    
