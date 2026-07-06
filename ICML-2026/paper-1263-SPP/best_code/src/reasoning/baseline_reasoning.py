"""
Baseline reasoning module
Implements a reasoning procedure fully consistent with the baseline (single-step reasoning)
Supports multi-turn reasoning for scenario-level tasks, including a History mechanism
Supports prompt templates for 4 question types
Uses the official chat template format
"""

import torch
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import math

from ..utils import get_logger
from ..model import BaseModel
from .official_templates import OfficialTemplateBuilder

logger = get_logger(__name__)


class BaselineReasoning:
    """
    Baseline reasoner
    A reasoning procedure fully consistent with the baseline; does not involve complex strategies such as Tree, CoT, or 1/N
    Supports multi-turn reasoning for scenario-level tasks, including a History mechanism
    """

    def __init__(
        self,
        base_model: BaseModel,
        max_steps: int = 1,  # Maximum number of reasoning steps (default 1 step, consistent with baseline)
        step_selection_strategy: str = "fixed",  # "fixed" or "difficulty_based"
        difficulty_threshold: float = 0.5,  # Difficulty threshold (used when step_selection_strategy="difficulty_based")
        max_history_turns: int = 2,  # Maximum number of history turns to keep (default 2, i.e. the previous 1-2 turns)
        use_history: bool = True,  # Whether to use the history mechanism
    ):
        """
        Args:
            base_model: Base model
            max_steps: Maximum number of reasoning steps
            step_selection_strategy: Step-count selection strategy
                - "fixed": Fixed number of steps (consistent with baseline)
                - "difficulty_based": Adjust the number of steps based on difficulty
            difficulty_threshold: Difficulty threshold (use more steps when exceeded)
            max_history_turns: Maximum number of history turns to keep
            use_history: Whether to use the history mechanism
        """
        self.base_model = base_model
        # Use 1-step reasoning by default (simplest, consistent with baseline)
        self.max_steps = max_steps if max_steps > 0 else 1
        self.step_selection_strategy = step_selection_strategy
        self.difficulty_threshold = difficulty_threshold
        self.max_history_turns = max_history_turns
        self.use_history = use_history

        # MoE router stats (optional, populated per step when available)
        self._last_moe_router_stats: Optional[Dict] = None
        
        # History memory (stores the QA content of previous turns)
        self.previous_turns_qa: List[Dict] = []

        # Initialize the official template builder
        # Obtain the tokenizer path (from base_model)
        tokenizer_path = base_model.model_name
        self.template_builder = OfficialTemplateBuilder(
            model_name=base_model.model_name,
            tokenizer_path=tokenizer_path
        )
        
        logger.info(f"Initializing Baseline reasoner (with History mechanism and official templates):")
        logger.info(f"  Max steps: {self.max_steps} (default 1 step, single-step reasoning)")
        logger.info(f"  Step selection strategy: {step_selection_strategy} (default fixed, fixed number of steps)")
        logger.info(f"  Use History: {use_history} (keep at most {max_history_turns} turns)")
        logger.info(f"  Use official template: {self.template_builder.model_type}")
        if step_selection_strategy == "difficulty_based":
            logger.info(f"  Difficulty threshold: {difficulty_threshold}")
    
    def reason(
        self,
        prompt: str,
        difficulty: Optional[float] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        task_type: Optional[str] = None,
        is_first_turn: bool = False,
    ) -> Dict:
        """
        Run reasoning (supports the History mechanism and prompt templates for 4 question types)

        Args:
            prompt: Input prompt
            difficulty: Question difficulty (0-1, optional)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: top-p sampling parameter
            task_type: Question type (multiple_choice, factual, code, reasoning)
            is_first_turn: Whether this is the first turn (used to reset history)

        Returns:
            {
                'result': str,  # Final result
                'steps': List[Dict],  # Reasoning steps
                'num_steps': int,  # Actual number of steps
                'inference_time': float,  # Inference time
            }
        """
        import time
        start_time = time.time()

        # If this is the first turn, reset history
        if is_first_turn:
            self.previous_turns_qa = []
            logger.info("Resetting History (first turn)")

        # Determine the number of reasoning steps
        if self.step_selection_strategy == "difficulty_based" and difficulty is not None:
            num_steps = self._determine_steps_by_difficulty(difficulty)
        else:
            num_steps = self.max_steps

        logger.debug(f"Number of reasoning steps: {num_steps} (strategy: {self.step_selection_strategy})")

        # Build the prompt including history (using the official template)
        full_prompt = self._build_prompt(prompt, task_type)

        # Debug: log the built full prompt (disabled, using debug level)
        logger.debug(f"Built full prompt (first 500 chars): {full_prompt[:500]}")
        logger.debug(f"Full prompt length: {len(full_prompt)} chars, task_type={task_type}")

        # Run reasoning
        # Note: inference_time only measures pure inference time (the time of model.generate()), excluding prompt building, history updates, etc.
        steps = []
        current_prompt = full_prompt
        total_pure_inference_time = 0.0  # Accumulated pure inference time
        total_wall_time = 0.0  # Accumulated end-to-end time (including tokenization/decoding, etc.)
        peak_memory_mb_allocated = 0.0  # Peak GPU memory over the end-to-end process (allocated)
        peak_memory_mb_reserved = 0.0  # Peak GPU memory over the end-to-end process (reserved)

        for step_idx in range(num_steps):
            self._last_moe_router_stats = None
            # Single-step reasoning (consistent with baseline)
            # _single_step_reasoning returns (cleaned_text, pure_inference_time, peak_allocated_mb, peak_reserved_mb, wall_time)
            step_result, step_inference_time, step_peak_allocated_mb, step_peak_reserved_mb, step_wall_time = self._single_step_reasoning(
                current_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                task_type=task_type  # Pass task_type
            )
            total_pure_inference_time += step_inference_time  # Accumulate pure inference time
            total_wall_time += step_wall_time
            peak_memory_mb_allocated = max(peak_memory_mb_allocated, step_peak_allocated_mb)
            peak_memory_mb_reserved = max(peak_memory_mb_reserved, step_peak_reserved_mb)

            steps.append({
                'step': step_idx + 1,
                'prompt': current_prompt,
                'result': step_result,  # step_result is a string, not a tuple
            })

            # For multi-step reasoning, use the current result as input for the next step
            if step_idx < num_steps - 1:
                current_prompt = f"{current_prompt}\n\n{step_result}"

        # Final result
        final_result = steps[-1]['result'] if steps else ""

        # Update History Memory (store the QA content of the current turn)
        if self.use_history:
            hist_entry = {
                'question': prompt,
                'answer': final_result
            }
            self.previous_turns_qa.append(hist_entry)

            # Limit history length (keep the most recent max_history_turns turns)
            if len(self.previous_turns_qa) > self.max_history_turns:
                self.previous_turns_qa = self.previous_turns_qa[-self.max_history_turns:]

            logger.debug(f"History updated: stored QA content (currently {len(self.previous_turns_qa)} turns)")

        # inference_time only includes pure inference time (the time of model.generate()), excluding prompt building, history updates, etc.
        inference_time = total_pure_inference_time
        
        return {
            'result': final_result,
            'steps': steps,
            'num_steps': num_steps,
            'inference_time': inference_time,
            'wall_time': total_wall_time,
            'peak_memory_mb_allocated': peak_memory_mb_allocated,
            'peak_memory_mb_reserved': peak_memory_mb_reserved,
            'moe_router_stats': self._last_moe_router_stats,
        }

    def _maybe_collect_moe_router_stats(
        self,
        input_len: int,
        full_ids: torch.Tensor,
    ) -> Optional[Dict]:
        """
        Collect compact MoE router statistics (Mixtral-style models).

        We do a single forward pass on the full sequence (prompt + generated tokens)
        with output_router_logits enabled (when supported), then aggregate expert usage
        on generated tokens only to avoid prompt bias.
        """
        model = getattr(self.base_model, "model", None)
        if model is None:
            return None
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None

        num_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
        top_k = getattr(cfg, "num_experts_per_tok", None)
        if not num_experts or not top_k:
            return None

        try:
            attn = torch.ones_like(full_ids, dtype=torch.long, device=full_ids.device)
            with torch.no_grad():
                out = model(
                    input_ids=full_ids,
                    attention_mask=attn,
                    output_router_logits=True,
                    return_dict=True,
                )
        except Exception as e:
            logger.warning(f"MoE router stats: forward failed: {type(e).__name__}: {e}")
            return None

        router_logits = getattr(out, "router_logits", None)
        if router_logits is None:
            return None

        # router_logits is typically a tuple/list over MoE layers, each [B, T, E]
        try:
            counts = [0] * int(num_experts)
            entropies = []
            tok_start = int(input_len)
            tok_end = int(full_ids.shape[1])
            gen_len = max(0, tok_end - tok_start)
            if gen_len == 0:
                return {
                    "num_experts": int(num_experts),
                    "top_k": int(top_k),
                    "generated_tokens": 0,
                }

            for layer_logits in router_logits:
                if layer_logits is None:
                    continue
                # Accept a few common shapes:
                # - [B, T, E]
                # - [T, E]
                # - [B*T, E] (flattened)
                lg = None
                if layer_logits.ndim == 3:
                    lg = layer_logits[0, tok_start:tok_end, :]
                elif layer_logits.ndim == 2:
                    if layer_logits.shape[0] == tok_end:
                        lg = layer_logits[tok_start:tok_end, :]
                    else:
                        # fall back: assume flattened [B*T, E] with B=1
                        if layer_logits.shape[0] >= tok_end:
                            lg = layer_logits[tok_start:tok_end, :]
                if lg is None:
                    continue
                if lg.numel() == 0:
                    continue
                # top-k experts per token
                topk_idx = torch.topk(lg, k=int(top_k), dim=-1).indices  # [gen_len, k]
                for i in range(topk_idx.shape[0]):
                    for j in range(topk_idx.shape[1]):
                        e = int(topk_idx[i, j].item())
                        if 0 <= e < len(counts):
                            counts[e] += 1
                # entropy over experts (softmax on logits)
                probs = torch.softmax(lg.float(), dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # [gen_len]
                entropies.extend(ent.detach().cpu().tolist())

            ent_mean = float(sum(entropies) / len(entropies)) if entropies else 0.0
            ent_var = float(sum((x - ent_mean) ** 2 for x in entropies) / max(1, (len(entropies) - 1))) if len(entropies) > 1 else 0.0
            ent_std = math.sqrt(ent_var)

            return {
                "num_experts": int(num_experts),
                "top_k": int(top_k),
                "generated_tokens": int(gen_len),
                "expert_counts": counts,
                "entropy_mean": ent_mean,
                "entropy_std": float(ent_std),
            }
        except Exception as e:
            logger.warning(f"MoE router stats: aggregation failed: {type(e).__name__}: {e}")
            return None
    
    def _build_prompt(self, prompt: str, task_type: Optional[str] = None) -> str:
        """
        Build the prompt (using the official template, including history and format constraints)

        Args:
            prompt: Original prompt (user question)
            task_type: Question type (multiple_choice, factual, code, reasoning)

        Returns:
            The full prompt (in the official template format)
        """
        # Convert previous_turns_qa into the history format required by the official template
        history = []
        if self.use_history and self.previous_turns_qa:
            for hist in self.previous_turns_qa:
                # Add user message
                history.append({
                    "role": "user",
                    "content": hist.get('question', '')
                })
                # Add assistant reply
                history.append({
                    "role": "assistant",
                    "content": hist.get('answer', '')
                })

        # Use the official template builder
        full_prompt = self.template_builder.build_prompt(
            user_message=prompt,
            task_type=task_type,
            history=history,
            system_prompt=None,  # Use the default system prompt
            add_format_constraints=True
        )

        return full_prompt

    def _build_multiple_choice_prompt(self, prompt: str) -> str:
        """Build the prompt for multiple-choice questions"""
        full_prompt = f"{prompt}\n\n"

        # Add the QA content of previous turns (History)
        if self.use_history and self.previous_turns_qa:
            full_prompt += "Previous turns (for reference only):\n"
            for i, hist in enumerate(self.previous_turns_qa):
                full_prompt += f"Q{i+1}: {hist['question']}\n"
                full_prompt += f"A{i+1}: {hist['answer']}\n"
            full_prompt += "\n"

        # Format requirements for multiple-choice questions (very strict, emphasizing conciseness)
        full_prompt += "CRITICAL: This is a multiple-choice question.\n"
        full_prompt += "You MUST provide ONLY a single letter (A, B, C, D, etc.) as your answer.\n"
        full_prompt += "NO explanations. NO thinking process. NO sentences. NO additional content.\n"
        full_prompt += "ONLY the letter itself. Be concise and precise.\n"
        full_prompt += "You MUST provide an answer. Even if you are uncertain, provide your best guess.\n"
        full_prompt += "Answer:"
        
        return full_prompt
    
    def _build_factual_prompt(self, prompt: str) -> str:
        """Build the prompt for factual questions"""
        full_prompt = f"{prompt}\n\n"

        # Add the QA content of previous turns (History)
        if self.use_history and self.previous_turns_qa:
            full_prompt += "Previous turns (for reference only):\n"
            for i, hist in enumerate(self.previous_turns_qa):
                full_prompt += f"Q{i+1}: {hist['question']}\n"
                full_prompt += f"A{i+1}: {hist['answer']}\n"
            full_prompt += "\n"

        # Format requirements for factual questions (emphasizing conciseness and precision)
        full_prompt += "CRITICAL: This is a FACTUAL question.\n"
        full_prompt += "Provide ONLY the key noun, phrase, or short answer.\n"
        full_prompt += "NO complete sentences. NO explanations. NO thinking process. NO additional content.\n"
        full_prompt += "ONLY the essential information. Be concise and precise.\n"
        full_prompt += "You MUST provide an answer. Even if you are uncertain, provide your best guess.\n"
        full_prompt += "Answer:"
        
        return full_prompt
    
    def _build_code_prompt(self, prompt: str) -> str:
        """Build the prompt for code questions"""
        full_prompt = f"{prompt}\n\n"

        # Add the QA content of previous turns (History)
        if self.use_history and self.previous_turns_qa:
            full_prompt += "Previous turns (for reference only):\n"
            for i, hist in enumerate(self.previous_turns_qa):
                full_prompt += f"Q{i+1}: {hist['question']}\n"
                full_prompt += f"A{i+1}: {hist['answer']}\n"
            full_prompt += "\n"

        # Format requirements for code questions (emphasizing conciseness and precision)
        full_prompt += "CRITICAL: This is a CODE question.\n"
        full_prompt += "Provide ONLY the code.\n"
        full_prompt += "NO markdown. NO comments. NO explanations. NO thinking process. NO additional content.\n"
        full_prompt += "ONLY the code itself. Be concise and precise.\n"
        full_prompt += "You MUST provide an answer. Even if you are uncertain, provide your best attempt.\n"
        full_prompt += "Code:"
        
        return full_prompt
    
    def _build_reasoning_prompt(self, prompt: str) -> str:
        """Build the prompt for reasoning questions"""
        full_prompt = f"{prompt}\n\n"

        # Add the QA content of previous turns (History)
        if self.use_history and self.previous_turns_qa:
            full_prompt += "Previous turns (for reference only):\n"
            for i, hist in enumerate(self.previous_turns_qa):
                full_prompt += f"Q{i+1}: {hist['question']}\n"
                full_prompt += f"A{i+1}: {hist['answer']}\n"
            full_prompt += "\n"

        # Format requirements for reasoning questions (emphasizing conciseness and precision)
        full_prompt += "CRITICAL: This is a REASONING question.\n"
        full_prompt += "Provide a direct and concise answer to the question.\n"
        full_prompt += "NO lengthy explanations. NO reasoning steps. NO thinking process. NO additional content.\n"
        full_prompt += "ONLY the answer itself. Be concise and precise.\n"
        full_prompt += "You MUST provide an answer. Even if you are uncertain, provide your best guess.\n"
        full_prompt += "Answer:"
        
        return full_prompt
    
    def _build_default_prompt(self, prompt: str) -> str:
        """Build the default prompt"""
        full_prompt = f"{prompt}\n\n"

        # Add the QA content of previous turns (History)
        if self.use_history and self.previous_turns_qa:
            full_prompt += "Previous turns (for reference only):\n"
            for i, hist in enumerate(self.previous_turns_qa):
                full_prompt += f"Q{i+1}: {hist['question']}\n"
                full_prompt += f"A{i+1}: {hist['answer']}\n"
            full_prompt += "\n"
        
        full_prompt += "CRITICAL: Provide ONLY the answer.\n"
        full_prompt += "NO explanations. NO thinking process. NO additional content.\n"
        full_prompt += "You MUST provide an answer. Even if you are uncertain, provide your best guess.\n"
        full_prompt += "Answer:"
        
        return full_prompt
    
    def _get_format_constraint(self, task_type: Optional[str] = None) -> str:
        """
        Return the format constraint based on question type (adapted from the benchmark project)

        Args:
            task_type: Question type (multiple_choice, factual, code, reasoning)

        Returns:
            Format constraint string
        """
        if task_type == 'multiple_choice':
            return """Answer with only the letter (A, B, C, D, etc.):"""
        elif task_type == 'factual':
            return """Answer briefly:"""
        elif task_type == 'code':
            return """CRITICAL: Provide ONLY the code. NO markdown, NO comments, NO explanations, NO thinking process.

CORRECT:
def calculate_sum(a, b):
    return a + b

Code:"""
        elif task_type == 'reasoning':
            return """Answer:"""
        else:
            return "Answer:"
    
    def _single_step_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        task_type: Optional[str] = None,  # Add task_type parameter
    ) -> Tuple[str, float, float, float, float]:
        """
        Single-step reasoning (consistent with baseline)

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: top-p sampling parameter

        Returns:
            (reasoning result text, pure inference time, peak_allocated_mb, peak_reserved_mb, wall_time)
        """
        # Debug: log the incoming prompt (disabled, using debug level)
        logger.debug(f"Prompt passed to _single_step_reasoning (first 500 chars): {prompt[:500]}")
        logger.debug(f"Incoming prompt length: {len(prompt)} chars")

        step_start_time = 0.0
        peak_memory_allocated_mb = 0.0
        peak_memory_reserved_mb = 0.0

        # Generate using base_model
        import time
        step_start_time = time.time()
        inputs = self.base_model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.base_model.device)

        # Debug: log the length after tokenization
        input_length = inputs['input_ids'].shape[1]
        logger.info(f"Length after tokenization: {input_length} tokens")

        # Stop sequences: minimal configuration to avoid stopping too early
        # Strategy: rely mainly on the prompt and EM match; do not set overly strict stop conditions
        # Note: HuggingFace's generate requires stopping_criteria to use stop sequences
        # For simplicity, we rely mainly on max_new_tokens and the EOS token; stop sequences are handled in post-processing
        # The stop sequences defined here are for post-processing, not for stopping during generation
        stop_sequences = [
            "\n\nCRITICAL:",  # If the model repeats the format markers, it may be starting to repeat itself
            "\n\nPrevious turns",  # If the model starts repeating the history format, it may be going off track
        ]

        # Set max_new_tokens based on question type
        # Strategy: allow the answer length to be about 2x the reference answer to avoid overly severe truncation
        # Rely more on the prompt to make the model answer concisely, rather than forced truncation
        if task_type == 'multiple_choice':
            # Multiple choice: although only 1 letter is needed, the model may produce filler, so give it enough space
            # Set a large enough max_new_tokens so the model can generate a complete answer (even with filler)
            # The answer extraction logic will extract the true answer from the filler
            effective_max_tokens = 60  # Increased from 50 to 60, slightly higher to handle more filler
            effective_temperature = 0.0  # Use greedy decoding to ensure precise output
        elif task_type == 'factual':
            # Factual: the reference answer is usually 10-30 words, but the model may produce filler, so increase slightly
            effective_max_tokens = 60  # Increased from 50 to 60, slightly higher to handle more filler
            effective_temperature = 0.1  # Lower, to ensure precise output
        elif task_type == 'code':
            # Code: the reference answer may be 50-100 lines, 2x is about 100-200 lines, roughly 400-600 tokens
            effective_max_tokens = 550  # Increased from 500 to 550, slightly higher
            effective_temperature = 0.3  # Slightly higher, code needs some creativity
        else:  # reasoning
            # Reasoning: the reference answer may be 20-50 words, but the model may produce filler, so it needs enough space
            # Increase max_new_tokens to ensure a complete answer can be generated even with filler
            effective_max_tokens = 180  # Increased from 150 to 180, slightly higher to handle more filler
            effective_temperature = 0.1  # Lower, to ensure precise reasoning

        # Generate
        # Strategy: rely mainly on the prompt to make the model answer concisely, rather than forced truncation
        # Use moderate parameters and let the model generate naturally, handling answer quality via EM match
        # Record pure inference time (only the time of model.generate(), excluding hooks overhead)
        import time

        # Reset the hooks time counter (if any)
        if hasattr(self.base_model, 'mask_hooks_time'):
            self.base_model.mask_hooks_time = 0.0

        # Record peak GPU memory during inference (allocated/reserved)
        if torch.cuda.is_available() and self.base_model.device.type == 'cuda':
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                # If reset fails, still ensure inference can run
                pass

        inference_start_time = time.time()
        with torch.no_grad():
            # Very low temperatures still go through multinomial; under int4/fp16 + head mask the logits
            # occasionally contain nan/inf, which can trigger a CUDA device-side assert. When temperature < 0.15,
            # use greedy directly, consistent with the "near-deterministic" objective.
            do_sample = effective_temperature >= 0.15
            gen_kw = dict(
                max_new_tokens=effective_max_tokens,
                temperature=effective_temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.base_model.tokenizer.eos_token_id,
                eos_token_id=self.base_model.tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=0.9 if task_type == 'multiple_choice' else 1.0,
            )
            try:
                outputs = self.base_model.model.generate(**inputs, **gen_kw)
            except Exception as e:
                # Sampling can hit CUDA multinomial asserts when the probability tensor is invalid
                # (e.g. inf/nan/<0) under fp16 + head masking; greedy decoding avoids multinomial.
                logger.warning(
                    "generate failed (%s); retrying with greedy decoding (do_sample=False)",
                    e,
                )
                if torch.cuda.is_available() and self.base_model.device.type == "cuda":
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                gen_kw = dict(
                    max_new_tokens=effective_max_tokens,
                    do_sample=False,
                    pad_token_id=self.base_model.tokenizer.eos_token_id,
                    eos_token_id=self.base_model.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    length_penalty=0.9 if task_type == 'multiple_choice' else 1.0,
                )
                outputs = self.base_model.model.generate(**inputs, **gen_kw)
        total_time = time.time() - inference_start_time

        # Subtract the hooks overhead (if any) to ensure only pure model inference time is measured
        hooks_time = getattr(self.base_model, 'mask_hooks_time', 0.0)
        pure_inference_time = total_time - hooks_time

        # If the result is negative, the hooks overhead is small or the measurement is inaccurate; use the original time
        if pure_inference_time < 0:
            pure_inference_time = total_time

        # After generation, synchronize and read peak GPU memory
        if torch.cuda.is_available() and self.base_model.device.type == 'cuda':
            try:
                torch.cuda.synchronize()
                peak_memory_allocated_mb = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
                peak_memory_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024 ** 2)
            except Exception:
                peak_memory_allocated_mb = 0.0
                peak_memory_reserved_mb = 0.0
        
        sequences = outputs

        # Decode
        generated_text = self.base_model.tokenizer.decode(
            sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Optional: collect MoE router stats via a post-hoc forward pass
        if isinstance(sequences, torch.Tensor) and sequences.ndim == 2:
            try:
                full_ids = sequences.to(self.base_model.device)
                self._last_moe_router_stats = self._maybe_collect_moe_router_stats(
                    input_len=int(inputs["input_ids"].shape[1]),
                    full_ids=full_ids,
                )
            except Exception:
                self._last_moe_router_stats = None
        else:
            self._last_moe_router_stats = None
        
        # Log the raw generated text (full output)
        if len(generated_text) > 0:
            logger.info(f"Raw generated text (full): {generated_text[:200]}..." if len(generated_text) > 200 else f"Raw generated text (full): {generated_text}")
        else:
            logger.warning(f"The model did not generate any text! task_type={task_type}, max_new_tokens={effective_max_tokens}, temperature={effective_temperature}")

        # Minimal post-processing: only handle obvious formatting issues, avoiding empty answers
        # Strategy: if stop sequences already exist, post-processing should be very light, only removing obvious repeated format markers
        # Rely mainly on EM match to handle answer quality, rather than post-processing
        cleaned_text = self._minimal_cleanup(generated_text, task_type, original_text=generated_text)

        # If cleanup yields an empty string, return the original text (to avoid empty answers)
        if not cleaned_text:
            cleaned_text = generated_text.strip()

        # End-to-end wall-clock time (including tokenization/decoding/post-processing, etc.)
        wall_time = time.time() - step_start_time

        # Return the result and pure inference time (only the time of model.generate()) plus peak GPU memory / end-to-end time
        return cleaned_text, pure_inference_time, peak_memory_allocated_mb, peak_memory_reserved_mb, wall_time
    
    def _minimal_cleanup(self, text: str, task_type: Optional[str] = None, original_text: Optional[str] = None) -> str:
        """
        Minimal cleanup: handle obvious formatting issues; in particular, multiple-choice answers need truncation after a newline

        Strategy:
        1. For multiple choice: truncate after the first `\n\n` or `\n` (keep the first option letter)
        2. For other types: only remove obvious repeated format markers
        3. Code: do not truncate at newlines (code needs newlines)
        4. Rely mainly on EM match to handle answer quality

        Args:
            text: Generated text
            task_type: Question type
            original_text: Original text (returned if cleanup yields an empty string)

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        text = text.strip()
        if not text:
            return ""

        # For multiple choice: extract the answer letter (A-Z) from the whole text; do not over-truncate
        # Even if there is filler, make sure the answer can still be extracted
        if task_type == 'multiple_choice':
            import re
            # Do not truncate at newlines! Search for the answer across the whole text

            # Prefer "answer"-related patterns (most reliable)
            # Note: order matters; more specific patterns come first
            patterns = [
                r'[Aa]nswer\s+to\s+.*?[:\s]+([A-Z])',  # "answer to your question is: D" (most specific)
                r'[Tt]he\s+[Aa]nswer\s+is[:\s]+([A-Z])',  # "The answer is D"
                r'[Cc]orrect\s+[Aa]nswer\s+is[:\s]+([A-Z])',  # "Correct answer is D"
                r'[Aa]nswer[:\s]+([A-Z])',  # "Answer: D" or "answer is D"
                r'[Ii]s[:\s]+([A-Z])\b',  # "is: D" (note the \b to avoid matching the "i" in "is")
                r'[Oo]ption\s+([A-Z])',  # "Option A"
                r'[Cc]hoice\s+([A-Z])',  # "Choice B"
                r'[Cc]hoose\s+([A-Z])',  # "Choose A"
                r'[Ww]ould\s+[Cc]hoose\s+([A-Z])',  # "would choose A"
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    letter = match.group(1)
                    # Make sure the match is a single letter A-Z
                    if letter and len(letter) == 1 and letter.isalpha() and letter.isupper():
                        return letter

            # If not found, try to find a standalone letter (A-Z) in the whole text
            # Prefer a letter at the end of the sentence (usually the answer)
            # First look for a letter among the last few words
            words = text.split()
            if words:
                # Check whether any of the last 3 words is a standalone letter
                for word in words[-3:]:
                    if len(word) == 1 and word.isalpha() and word.isupper():
                        return word

            # If still not found, find all standalone letters in the whole text
            matches = list(re.finditer(r'\b([A-Z])\b', text))
            if matches:
                # Take the last match (usually at the answer position)
                return matches[-1].group(1)

            # If still not found, try searching after removing common filler prefixes
            cleaned = re.sub(r'^(Sure!|Okay!|Here\'s|Got it!|Here goes|Here we go|Sure thing!).*?[:,\s]+', '', text, flags=re.IGNORECASE)
            cleaned = cleaned.strip('.,:;!? ')
            # Look for a letter in the cleaned text
            letter_match = re.search(r'\b([A-Z])\b', cleaned)
            if letter_match:
                return letter_match.group(1)

            # Finally, if the cleaned text is a single character and a letter, return it
            if len(cleaned) == 1 and cleaned.isalpha() and cleaned.isupper():
                return cleaned

            # If all methods fail, return the original text (at least keep some information)
            return text.strip()

        # For code: do not truncate at newlines, but markdown markers can be removed
        # Code needs newlines, so do not apply the newline-truncation logic
        if task_type == 'code':
            # Only remove markdown code-block markers, keeping the code content
            import re
            text = re.sub(r'```\w*\n?', '', text)
            text = re.sub(r'```', '', text)
            return text.strip()

        # For other types (factual, reasoning): extract the answer and remove filler prefixes
        # Note: the answer may be on the first line, or on the second or third line (after filler lines)
        if task_type in ['factual', 'reasoning']:
            import re
            # Define pure-filler patterns (used for detection)
            pure_filler_patterns = [
                r'^(Here\'s my (answer|response))$',
                r'^(my (answer|response))$',
                r'^(The answer (is|to))$',
                r'^(answer|response)$',
                r'^(Here\'s|Here is)$',
            ]
            
            # Strategy: first try to remove filler prefixes from the whole text (including across lines)
            # This handles cases like "Sure, I can do that! Here's my response:\n\nThe answer is 42"
            # Use a more direct approach: find the content after the first colon or newline
            # 1. First try to match and remove leading filler (including patterns like "Sure, I can do that! Here's my response:")
            # 2. If the cleaned text still starts with "Here's my response" or "Here's my answer", keep removing
            full_cleaned = text

            # Step 1: remove leading filler prefixes (including "Sure, I can do that!", etc.)
            full_cleaned = re.sub(
                r'^(Sure!|Okay!|Got it!|Here goes|Here we go|Sure thing!|Sure, I can do that!).*?[:,\s]+',
                '',
                full_cleaned,
                flags=re.IGNORECASE | re.DOTALL
            )

            # Step 2: if it still starts with "Here's my response:" or "Here's my answer:", remove it (including the colon and the following newline)
            full_cleaned = re.sub(
                r'^(Here\'s my (answer|response)|Here\'s|The answer (is|to))[:,\s]*\s*\n*\s*',
                '',
                full_cleaned,
                flags=re.IGNORECASE | re.DOTALL
            )

            # Remove leading whitespace (including newlines, colons, commas, etc.)
            full_cleaned = full_cleaned.strip('.,:;!? \n\r\t')

            # Check whether the cleaned text is still pure filler
            is_pure_filler_full = any(re.match(pattern, full_cleaned, re.IGNORECASE) for pattern in pure_filler_patterns) if full_cleaned else False
            if is_pure_filler_full:
                full_cleaned = ""  # If it is pure filler, clear it

            # If the cleaned whole text is valid, use it
            if full_cleaned and len(full_cleaned) >= 2:
                # For the reasoning type, if the cleaned text is long, take only the first 300 chars
                if task_type == 'reasoning' and len(full_cleaned) > 300:
                    return full_cleaned[:300].strip()
                return full_cleaned

            # If cleaning the whole text failed, try processing only the first line (for backward compatibility)
            first_line = text.split('\n')[0] if '\n' in text else text
            cleaned = re.sub(r'^(Sure!|Okay!|Here\'s|Got it!|Here goes|Here we go|Sure thing!|Sure, I can do that!|Here\'s my answer|Here\'s my response|The answer is|The answer to).*?[:,\s]+', '', first_line, flags=re.IGNORECASE)
            cleaned = cleaned.strip('.,:;!? ')

            is_pure_filler = any(re.match(pattern, cleaned, re.IGNORECASE) for pattern in pure_filler_patterns)
            if is_pure_filler:
                cleaned = ""

            # If the first line is empty or too short after cleaning, try extracting from the whole text (the answer may be on the second line)
            if not cleaned or len(cleaned) < 2:
                # Try extracting from the whole text (after removing all filler prefixes)
                full_cleaned = re.sub(r'^(Sure!|Okay!|Here\'s|Got it!|Here goes|Here we go|Sure thing!|Sure, I can do that!|Here\'s my answer|Here\'s my response|The answer is|The answer to).*?[:,\s]+', '', text, flags=re.IGNORECASE)
                full_cleaned = full_cleaned.strip('.,:;!? ')
                # If the cleaned whole text is valid, use it
                if full_cleaned and len(full_cleaned) >= 2:
                    # For the reasoning type, if the cleaned text is long, take only the first 300 chars
                    if task_type == 'reasoning' and len(full_cleaned) > 300:
                        return full_cleaned[:300].strip()
                    return full_cleaned
                else:
                    # If it is still empty after cleaning, try more aggressive cleaning: remove all known filler patterns
                    # Try to find the first meaningful word (not filler)
                    aggressive_cleaned = re.sub(r'^(Sure!|Okay!|Here\'s|Got it!|Here goes|Here we go|Sure thing!|Sure, I can do that!|Here\'s my answer|Here\'s my response|The answer is|The answer to|I can do that|I\'d be happy to help|I can help).*?[:,\s]+', '', text, flags=re.IGNORECASE)
                    aggressive_cleaned = aggressive_cleaned.strip('.,:;!? ')

                    # Check whether it is still pure filler after aggressive cleaning
                    is_pure_filler_aggressive = any(re.match(pattern, aggressive_cleaned, re.IGNORECASE) for pattern in pure_filler_patterns) if aggressive_cleaned else False
                    if is_pure_filler_aggressive:
                        aggressive_cleaned = ""  # If it is pure filler, clear it

                    if aggressive_cleaned and len(aggressive_cleaned) >= 2:
                        if task_type == 'reasoning' and len(aggressive_cleaned) > 300:
                            return aggressive_cleaned[:300].strip()
                        return aggressive_cleaned
                    # If all cleaning failed, return the original text (at least keep some information, rather than an empty string)
                    # But return only the first 300 chars to avoid being too long
                    if task_type == 'reasoning' and len(text) > 300:
                        return text[:300].strip()
                    return text.strip()

            # If the cleaned text is valid, return it
            # For the reasoning type, if the text is long, take only the first 300 chars
            if task_type == 'reasoning' and len(cleaned) > 300:
                return cleaned[:300].strip()
            return cleaned

        # Default: only remove obvious leading format markers
        if text.startswith("CRITICAL:"):
            lines = text.split('\n')
            if len(lines) > 1:
                text = '\n'.join(lines[1:]).strip()

        if text.startswith("Answer:"):
            text = text[len("Answer:"):].strip()

        # If the text is empty after cleaning, return the original text (to avoid empty answers)
        if not text and original_text:
            return original_text.strip()

        return text
    
    def _extract_answer(self, text: str, task_type: Optional[str] = None) -> str:
        """
        Extract the answer portion from the generated text

        Args:
            text: The full generated text
            task_type: Question type

        Returns:
            The extracted answer
        """
        if not text:
            return ""

        # Remove leading/trailing whitespace
        text = text.strip()

        # If the text is empty, return directly
        if not text:
            return ""

        # Check whether it contains only underscores or dashes (and is short)
        # Note: do not over-filter; only filter clearly invalid underscore answers
        text_no_underscore = text.strip('_').strip('-').strip()
        if len(text_no_underscore) == 0 and len(text) < 10:
            # Only underscores and very short (<10 chars); treat as an invalid answer
            return ""

        # If it contains format-constraint keywords, extract the content before them
        # But be careful: do not over-truncate; only truncate when it is clearly a format constraint
        # Loosen the condition: only truncate on keywords that appear after a newline, and far enough into the text (>20 chars)
        stop_keywords = [
            "\n\nBased on the above information",  # Only truncate when it appears after a newline
            "\nCRITICAL:",
            "\nCORRECT:",
            "\nWRONG:",
            "\nComment:",
            "\nPrevious question",
            "\nPrevious answer"
        ]

        for keyword in stop_keywords:
            if keyword in text:
                # Find the first occurrence and extract the content before it
                idx = text.find(keyword)
                # Only truncate when far enough into the text (>20 chars), to avoid truncating the format constraint in the prompt
                if idx > 20:
                    text = text[:idx].strip()
                    break

        # If it contains "Answer:", extract the content after it
        # But be careful: if "Answer:" is at the start, it may be part of the prompt and should not be extracted
        # Loosen the condition: only extract when it appears in the middle of the text (>20 chars)
        if "Answer:" in text:
            # Check whether "Answer:" is in the middle of the text (not part of the prompt)
            idx = text.find("Answer:")
            if idx > 20:  # Only extract when in the middle of the text (>20 chars from the start)
                parts = text.split("Answer:", 1)
                if len(parts) > 1:
                    text = parts[1].strip()

        # Remove common format markers
        text = text.replace("\n\n", "\n").strip()

        # Check again: if only underscores remain after extraction, return an empty string
        # But be careful: do not over-filter; only filter clearly invalid underscore answers
        # Only treat as invalid if the text is very short (<10 chars) and contains only underscores
        if not text:
            return ""

        # Check whether it contains only underscores or dashes (and is short)
        text_no_underscore = text.strip('_').strip('-').strip()
        if len(text_no_underscore) == 0 and len(text) < 10:
            # Only underscores and very short (<10 chars); treat as an invalid answer
            return ""

        # Perform basic extraction based on task_type
        import re

        if task_type == 'multiple_choice':
            # Simplify: extract any A-F letter from the text
            # Prefer: an option letter followed by a period, comma, space, or end of string
            match = re.search(r'\b([A-F])(?:\.|,|\s|$)', text)
            if match:
                return match.group(1)

            # Second best: match any A-F letter
            match = re.search(r'\b([A-F])\b', text)
            if match:
                return match.group(1)

            # Last: extract from common phrases
            match = re.search(r'(?:answer|correct|choice|option|is)\s+([A-F])\b', text, re.IGNORECASE)
            if match:
                return match.group(1)

            # If none found, return an empty string
            return ""
        elif task_type == 'factual':
            # Take the first line, but do not hard-limit to 50 chars (it may truncate the correct answer)
            lines = text.split('\n')
            if lines:
                text = lines[0].strip()
            # Remove common leading phrases
            text = re.sub(r'^(The correct answer|The answer|Answer|Correct answer)[:.]?\s*', '', text, flags=re.IGNORECASE)
            text = text.strip()
            # Limit the length, but not too short (100 chars should be enough)
            if len(text) > 100:
                text = text[:100].strip()
        elif task_type == 'code':
            # Code may be long, so keep more
            if len(text) > 500:
                text = text[:500].strip()
        else:  # reasoning
            # Take the first 300 chars (increased, to avoid truncation)
            if len(text) > 300:
                text = text[:300].strip()

        # Remove duplicated answers (if the model generated repetitions)
        words = text.split()
        if len(words) > 20:  # If it is too long, it may be a repetition
            # Try to find a repetition pattern
            for i in range(1, min(10, len(words) // 2)):
                if words[:i] == words[i:2*i]:
                    text = ' '.join(words[:i])
                    break

        return text.strip()
    
    def _determine_steps_by_difficulty(self, difficulty: float) -> int:
        """
        Determine the number of reasoning steps based on difficulty

        Args:
            difficulty: Question difficulty (0-1)

        Returns:
            Number of reasoning steps
        """
        if difficulty >= self.difficulty_threshold:
            # Hard questions: use more steps
            return min(self.max_steps, 3)  # At most 3 steps
        else:
            # Easy questions: use fewer steps
            return 1
    
    def batch_reason(
        self,
        prompts: List[str],
        difficulties: Optional[List[float]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        task_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Batch reasoning (supports multi-turn reasoning for scenario-level tasks)

        Args:
            prompts: List of input prompts
            difficulties: List of difficulties (optional)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: top-p sampling parameter
            task_types: List of question types (optional)

        Returns:
            List of reasoning results
        """
        results = []
        for i, prompt in enumerate(prompts):
            difficulty = difficulties[i] if difficulties and i < len(difficulties) else None
            task_type = task_types[i] if task_types and i < len(task_types) else None
            is_first_turn = (i == 0)  # The first turn resets history
            
            result = self.reason(
                prompt,
                difficulty=difficulty,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                task_type=task_type,
                is_first_turn=is_first_turn
            )
            results.append(result)
        return results
    
    def reset_history(self):
        """Reset History"""
        self.previous_turns_qa = []
        logger.info("History has been reset")

