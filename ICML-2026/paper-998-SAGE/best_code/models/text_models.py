
from typing import Dict, Any, List, Tuple
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from .base_model import BaseModel
import pdb

# ============================================================
#   Llama 3.1 8B  —  local causal LM
# ============================================================

class LlamaTextClassifier(BaseModel):
    """Llama-3.1-8B for text classification via log-likelihood"""

    def __init__(self, config: Dict[str, Any], class_names: List[str]):
        super().__init__(config)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        self.calib_T = float(config.get("calib_T", 1.0))
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model, self.tokenizer = None, None

    def load_model(self):
        print(f"Loading Llama model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        print("Model loaded")

    @torch.no_grad()
    # def predict(self, text: str) -> Dict[str, Any]:
    def predict(self, text: str, choices: list = None) -> Dict[str, Any]:  # Added choices parameter
        if self.model is None: self.load_model()
        # Build different prompts based on whether choices are provided
        if choices:
            # Multiple-choice format
            options = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            class_list = ", ".join(self.class_names)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content":
                 f"Question: {text}\n\n"
                 f"Options:\n{options}\n\n"
                 f"Answer with ONLY the letter ({class_list}):"}
            ]
        else:
            # Original classification format
            class_list = ", ".join(self.class_names)
            messages = [
                {"role": "system", "content": "You are a concise text classifier."},
                {"role": "user", "content":
                f"Classify the following text into one of these categories: {class_list}.\n"
                f"Answer only with the category name.\n\nText: {text}\n\nCategory:"}
            ]
        prefix = self._build_prefix(messages)
        avg_logprobs = self._avg_loglik_per_label(prefix, self.class_names)

        # # Added: generate actual text
        # input_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)["input_ids"]
        # output_ids = self.model.generate(
        #     input_ids,
        #     max_new_tokens=10,
        #     do_sample=False,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id
        # )
        # generated_ids = output_ids[0, input_ids.shape[1]:]
        # generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        logits = torch.tensor(avg_logprobs) / self.calib_T
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred = int(np.argmax(probs))

        topk = np.argsort(-probs)[:min(5, self.num_classes)]
        return {
            "prediction": pred,
            "prediction_name": self.class_names[pred],
            "confidence": float(probs[pred]),
            "top5_predictions": topk.tolist(),
            "top5_prediction_names": [self.class_names[i] for i in topk],
            "top5_confidences": probs[topk].tolist(),
            "raw_output": {"label_avg_logprobs": avg_logprobs, "calib_T": self.calib_T}
            # "raw_output": {
            #     "label_avg_logprobs": avg_logprobs, 
            #     "calib_T": self.calib_T,
            #     "generated_text": generated_text,  # Added generated text
            #     "prompt": prefix  # Optional: also save the prompt
            # }
        }

    # ----- internals -----
    def _build_prefix(self, messages):
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            sys, user = messages[0]["content"], messages[1]["content"]
            return f"<|system|>\n{sys}\n<|user|>\n{user}\n<|assistant|>\n"

    @torch.no_grad()
    def _print_generation(self, prefix, max_new_tokens=50):
        """Generate and print the model's full output (for debugging)"""
        print("\n" + "="*80)
        print("DEBUG: Model Generation Output")
        print("="*80)
        print(f"Prefix:\n{prefix}")
        print("-"*80)

        input_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)["input_ids"]

        # Generate text
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode generated portion
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Generated Text:\n{generated_text}")
        print("="*80 + "\n")

    @torch.no_grad()
    def _avg_loglik_per_label(self, prefix, labels):
        # Run inference on prefix only once
        input_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)["input_ids"]
        logits = self.model(input_ids=input_ids).logits

        # Get logits at the last position (i.e., first token prediction after assistant)
        next_token_logits = logits[0, -1, :]  # shape: [vocab_size]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)

        # Get the probability of the corresponding token for each class
        out_scores = []
        for lab in labels:
            # Tokenize the class name (usually with a space prefix)
            comp = " " + lab
            comp_ids = self.tokenizer(comp, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            if len(comp_ids) == 1:
                # Single token case: directly take the log prob of that token
                token_id = comp_ids[0].item()
                out_scores.append(log_probs[token_id].item())
            else:
                # Multi-token case: need greedy decoding or just take the first token
                # Here we take the first token's probability as an approximation
                token_id = comp_ids[0].item()
                out_scores.append(log_probs[token_id].item())

        return out_scores


# ============================================================
#   Qwen3-8B  —  local causal LM
# ============================================================

class Qwen3TextClassifier(BaseModel):
    """Qwen3-8B text classifier (same scoring as Llama)"""

    def __init__(self, config, class_names):
        super().__init__(config)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.get("model_name", "Qwen/Qwen3-8B")
        self.calib_T = float(config.get("calib_T", 1.0))
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model, self.tokenizer = None, None

    def load_model(self):
        print(f"Loading Qwen3 model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda": self.model.to(self.device)
        self.model.eval(); print("Model loaded")

    @torch.no_grad()
    # def predict(self, text):
    def predict(self, text, choices: list = None):  # Added choices parameter
        if self.model is None: self.load_model()
        # Build different prompts based on whether choices are provided
        if choices:
            # Multiple-choice format
            options = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            class_list = ", ".join(self.class_names)
            msgs = [
                {"role": "system", "content": "You are a helpful assistant. Answer directly."},
                {"role": "user", "content":
                 f"Question: {text}\n\n"
                 f"Options:\n{options}\n\n"
                 f"Answer with ONLY the letter ({class_list}):"}
            ]
        else:
            class_list = ", ".join(self.class_names)
            msgs = [
                {"role": "system", "content": "You are a helpful text classifier. Answer directly."},
                {"role": "user", "content":
                f"Classify this text into one of these categories: {class_list}\n"
                f"Answer ONLY with the category name, nothing else.\n\nText: {text}\n\nCategory:"}
            ]
        prefix = self._build_prefix(msgs)
        
        # Debug: print model's full generated text
        # self._print_generation(prefix)
        
        avg_logprobs = self._avg_loglik_per_label(prefix, self.class_names)
        logits = torch.tensor(avg_logprobs) / self.calib_T
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred = int(np.argmax(probs))
        topk = np.argsort(-probs)[:min(5, self.num_classes)]
        return {
            "prediction": pred,
            "prediction_name": self.class_names[pred],
            "confidence": float(probs[pred]),
            "top5_predictions": topk.tolist(),
            "top5_prediction_names": [self.class_names[i] for i in topk],
            "top5_confidences": probs[topk].tolist(),
            "raw_output": {"label_avg_logprobs": avg_logprobs, "calib_T": self.calib_T}
        }

    def _build_prefix(self, msgs):
        # Manually build prompt for full format control, avoiding triggering thinking mode
        sys_content = msgs[0]["content"]
        usr_content = msgs[1]["content"]
        
        # Use simplified format, not using official chat_template
        # Pre-write empty <think></think> tags to make the model skip the thinking phase and output the answer directly
        return f"<|im_start|>system\n{sys_content}<|im_end|>\n<|im_start|>user\n{usr_content}<|im_end|>\n<|im_start|>assistant\n<think></think>\n"

    @torch.no_grad()
    def _print_generation(self, prefix, max_new_tokens=50):
        """Generate and print the model's full output (for debugging)"""
        print("\n" + "="*80)
        print("DEBUG: Model Generation Output")
        print("="*80)
        print(f"Prefix:\n{prefix}")
        print("-"*80)

        input_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)["input_ids"]

        # Generate text
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode generated portion
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Generated Text:\n{generated_text}")
        print("="*80 + "\n")

    @torch.no_grad()
    def _avg_loglik_per_label(self, prefix, labels):
        # Run inference on prefix only once
        input_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)["input_ids"]
        logits = self.model(input_ids=input_ids).logits

        # Get logits at the last position (i.e., first token prediction after assistant)
        next_token_logits = logits[0, -1, :]  # shape: [vocab_size]
        # log_probs = torch.log_softmax(next_token_logits, dim=-1)

        
        # Get the probability of the corresponding token for each class
        scores = []
        for lab in labels:
            # Tokenize the class name (usually with a space prefix)
            comp = " " + lab
            comp_ids = self.tokenizer(comp, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if len(comp_ids) == 1:
                # Single token case: directly take the log prob of that token
                token_id = comp_ids[0].item()
                scores.append(next_token_logits[token_id].item())
            else:
                # Multi-token case: need greedy decoding or just take the first token
                # Here we take the first token's probability as an approximation
                token_id = comp_ids[0].item()
                scores.append(next_token_logits[token_id].item())
        # pdb.set_trace()
        # print(scores)
        return scores
# class OpenAITextClassifier(BaseModel):
#     """OpenAI GPT-4o via log-likelihood per label"""

#     def __init__(self, config, class_names):
#         super().__init__(config)
#         self.class_names = class_names
#         self.num_classes = len(class_names)
#         self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
#         self.model_name = config.get("model_name", "gpt-4o-mini")
#         self.calib_T = float(config.get("calib_T", 1.0))
#         self.client = OpenAI(api_key=self.api_key)

#     def load_model(self):
#         print(f"Using OpenAI model: {self.model_name}")

#     def predict(self, text):
#         if not self.api_key:
#             raise ValueError("OPENAI_API_KEY not found.")
        
#         class_list = ", ".join(self.class_names)
#         messages = [
#             {"role": "system", "content": "You are a helpful text classifier."},
#             {"role": "user", "content":
#              f"Classify this text into one of these categories: {class_list}\n"
#              f"Answer only with the category name.\n\nText: {text}\n\nCategory:"}
#         ]
        
#         # Method 1: Use logprobs to get the first token's probability distribution
#         try:
#             avg_logprobs = self._score_with_first_token_logprobs(messages)
#         except Exception as e:
#             print(f"Logprobs method failed: {e}, falling back to generation matching")
#             # Method 2: Fallback - generate and match
#             avg_logprobs = self._score_with_generation(messages)
        
#         logits = torch.tensor(avg_logprobs) / self.calib_T
#         probs = torch.softmax(logits, dim=0).cpu().numpy()
        
#         pred = int(np.argmax(probs))
#         topk = np.argsort(-probs)[:min(5, self.num_classes)]
        
#         return {
#             "prediction": pred,
#             "prediction_name": self.class_names[pred],
#             "confidence": float(probs[pred]),
#             "top5_predictions": topk.tolist(),
#             "top5_prediction_names": [self.class_names[i] for i in topk],
#             "top5_confidences": probs[topk].tolist(),
#             "raw_output": {
#                 "label_avg_logprobs": avg_logprobs,
#                 "calib_T": self.calib_T
#             }
#         }
    
#     def _score_with_first_token_logprobs(self, messages):
#         """
#         Correct method: Get the first token's logprobs after classification prompt,
#         then extract each label's probability from them.
#         """
#         # Let model generate naturally, get the first token's logprobs
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=messages,
#             temperature=0,
#             max_tokens=1,  # Only need the first token
#             logprobs=True,
#             top_logprobs=20  # Get top-20 tokens
#         )
        
#         if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
#             raise ValueError("No logprobs returned from API")
        
#         # First token's logprobs distribution
#         first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        
#         # Build a dictionary: token -> logprob
#         token_to_logprob = {
#             lp.token.lower().strip(): lp.logprob 
#             for lp in first_token_logprobs
#         }
        
#         # Find the corresponding logprob for each class
#         label_logprobs = []
#         for label in self.class_names:
#             label_lower = label.lower().strip()
            
#             # Try to match the label
#             found_logprob = None
            
#             # Exact match
#             if label_lower in token_to_logprob:
#                 found_logprob = token_to_logprob[label_lower]
#             # Partial match (label may have a space prefix)
#             elif f" {label_lower}" in token_to_logprob:
#                 found_logprob = token_to_logprob[f" {label_lower}"]
#             # Fuzzy match
#             else:
#                 for token, logprob in token_to_logprob.items():
#                     if label_lower in token or token in label_lower:
#                         found_logprob = logprob
#                         break
            
#             # If not found, assign a very low logprob
#             if found_logprob is None:
#                 found_logprob = -20.0  # Very low probability
            
#             label_logprobs.append(found_logprob)
        
#         return label_logprobs
    
#     def _score_with_generation(self, messages):
#         """
#         Fallback method: generate output and use match scores
#         """
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=messages,
#             temperature=0,
#             max_tokens=5
#         )
        
#         generated = response.choices[0].message.content.strip().lower()
        
#         # Match scores (converted to log space for consistency)
#         scores = []
#         for label in self.class_names:
#             label_lower = label.lower().strip()
#             if label_lower == generated:
#                 scores.append(0.0)  # log(1) = 0
#             elif label_lower in generated:
#                 scores.append(-1.0)  # log(0.37) ≈ -1
#             elif generated in label_lower:
#                 scores.append(-2.0)  # log(0.14) ≈ -2
#             else:
#                 scores.append(-5.0)  # log(0.007) ≈ -5
        
#         return scores
class OpenAITextClassifier(BaseModel):
    """OpenAI GPT model used as a text classifier via numeric-choice log-likelihood."""

    def __init__(self, config, class_names):
        super().__init__(config)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model_name = config.get("model_name", "gpt-4o-mini")
        self.calib_T = float(config.get("calib_T", 1.0))
        self.client = OpenAI(api_key=self.api_key)

    def load_model(self):
        """No explicit model loading needed for OpenAI API."""
        print(f"Using OpenAI model: {self.model_name}")

    # ----------------------- Public inference API ----------------------- #
    # def predict(self, text: str):
    def predict(self, text: str, choices: list = None):
        """
        Run classification and return:
        - prediction (int index)
        - prediction_name (label string)
        - confidence (softmax prob of predicted class)
        - top5 (indices, names, probs)
        - raw_output: numeric logprobs before temperature scaling
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        # Build different prompts based on whether choices are provided
        if choices:
            messages = self._build_multiple_choice_messages(text, choices)
        else:
            messages = self._build_messages(text)

        # Try logprobs-based scoring first
        try:
            label_logprobs = self._score_with_first_token_logprobs(messages)
        except Exception as e:
            print(f"[OpenAITextClassifier] Logprobs method failed: {e}, falling back to generation.")
            label_logprobs = self._score_with_generation(messages)

        # Convert logprobs to calibrated logits and probs
        logits = torch.tensor(label_logprobs, dtype=torch.float32) / self.calib_T
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        pred = int(np.argmax(probs))
        topk_count = min(5, self.num_classes)
        topk = np.argsort(-probs)[:topk_count]

        return {
            "prediction": pred,
            "prediction_name": self.class_names[pred],
            "confidence": float(probs[pred]),
            "top5_predictions": topk.tolist(),
            "top5_prediction_names": [self.class_names[i] for i in topk],
            "top5_confidences": probs[topk].tolist(),
            "raw_output": {
                "label_logprobs": [float(x) for x in label_logprobs],
                "calib_T": self.calib_T
            }
        }

    # ----------------------- Prompt construction ----------------------- #
    # Added: multiple-choice prompt
    def _build_multiple_choice_messages(self, question: str, choices: list):
        """Multiple-choice format (using 1234 mapped to ABCD)"""
        options = "\n".join([f"{i+1}) {choices[i]}" for i in range(len(choices))])
        user_content = (
            "You are answering a multiple-choice question.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            f"Answer ONLY with the NUMBER (1-{self.num_classes}) of the correct option:"
        )
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]

    def _build_messages(self, text: str):
        """
        Build messages where model chooses ONE number from 1..num_classes.
        This makes first-token logprobs directly correspond to class probabilities.
        """
        # Build numbered menu
        options = "\n".join(
            f"{i + 1}) {label}" for i, label in enumerate(self.class_names)
        )

        user_content = (
            "You are a text classifier.\n"
            "Classify the following text into ONE of the categories below.\n\n"
            f"{options}\n\n"
            f"Answer ONLY with the NUMBER (1-{self.num_classes}), "
            "without any extra text.\n\n"
            f"Text:\n{text}\n\n"
            "Category (number only):"
        )

        messages = [
            {"role": "system", "content": "You are a helpful text classification assistant."},
            {"role": "user", "content": user_content}
        ]
        return messages

    # ----------------------- Scoring via logprobs ----------------------- #
    def _score_with_first_token_logprobs(self, messages):
        """
        Use first generated token's logprobs as logits over numeric options 1..K.
        This assumes the first token is the digit '1', '2', ..., 'K'.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=1,          # we only need the first token
            logprobs=True,
            top_logprobs=max(self.num_classes, 10)  # ensure all digits are likely covered
        )

        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            raise ValueError("No logprobs returned from API.")

        first_token_info = choice.logprobs.content[0]
        top_logprobs = first_token_info.top_logprobs

        # Build token -> logprob dict; keep raw token string (no lower/strip needed for digits)
        token_to_logprob = {lp.token: lp.logprob for lp in top_logprobs}

        label_logprobs = []
        for i in range(self.num_classes):
            digit = str(i + 1)  # '1', '2', ..., 'K'

            # Try direct match; if not found, try with leading space (depending on tokenizer)
            if digit in token_to_logprob:
                lp = token_to_logprob[digit]
            elif (" " + digit) in token_to_logprob:
                lp = token_to_logprob[" " + digit]
            else:
                # Very unlikely if prompt is well-formed, but keep a small floor
                lp = -20.0

            label_logprobs.append(lp)

        return label_logprobs

    # ----------------------- Fallback: plain generation ----------------------- #
    def _score_with_generation(self, messages):
        """
        Fallback method: generate one short answer and map it to a numeric class.
        We then build pseudo-logprobs: correct number -> 0, others -> negative penalties.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=3  # number + maybe newline
        )
        raw = response.choices[0].message.content.strip()

        # Extract first digit in the answer
        chosen_idx = None
        for ch in raw:
            if ch.isdigit():
                idx = int(ch) - 1
                if 0 <= idx < self.num_classes:
                    chosen_idx = idx
                    break

        # Build pseudo-logprobs
        scores = []
        for i in range(self.num_classes):
            if chosen_idx is None:
                # model didn't give a valid number → almost uniform but low confidence
                scores.append(-2.0)
            elif i == chosen_idx:
                scores.append(0.0)    # log(1) for the chosen class
            else:
                scores.append(-3.0)   # log(~0.05) for others

        return scores


class MinistralTextClassifier(BaseModel):
    """Ministral-8B-Instruct-2410 for text classification via log-likelihood."""

    def __init__(self, config: Dict[str, Any], class_names: List[str]):
        super().__init__(config)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # default to Ministral 8B Instruct
        self.model_name = config.get(
            "model_name",
            "mistralai/Ministral-8B-Instruct-2410"
        )
        self.calib_T = float(config.get("calib_T", 1.0))
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model, self.tokenizer = None, None

    def load_model(self):
        """Load Ministral tokenizer and model."""
        print(f"Loading Ministral model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # use bfloat16 on GPU if possible (recommended), else float32
        if self.device == "cuda":
            torch_dtype = torch.bfloat16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()
        print("Ministral model loaded.")

    @torch.no_grad()
    # def predict(self, text: str) -> Dict[str, Any]:
    def predict(self, text: str, choices: list = None) -> Dict[str, Any]: 
        """Classify text using single-step log-likelihood over label first tokens."""
        if self.model is None:
            self.load_model()
        # Build different prompts based on whether choices are provided
        if choices:
            # Multiple-choice format
            options = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            class_list = ", ".join(self.class_names)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Question: {text}\n\n"
                        f"Options:\n{options}\n\n"
                        f"Answer with ONLY the letter ({class_list}):"
                    ),
                },
            ]
        else:
            class_list = ", ".join(self.class_names)
            messages = [
                {"role": "system", "content": "You are a concise text classifier."},
                {
                    "role": "user",
                    "content": (
                        f"Classify the following text into one of these categories: {class_list}.\n"
                        f"Answer only with the category name.\n\n"
                        f"Text: {text}\n\nCategory:"
                    ),
                },
            ]

        prefix = self._build_prefix(messages)
        label_scores = self._first_token_loglik_per_label(prefix, self.class_names)

        logits = torch.tensor(label_scores, dtype=torch.float32) / self.calib_T
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        pred = int(np.argmax(probs))
        topk_idx = np.argsort(-probs)[: min(5, self.num_classes)]

        return {
            "prediction": pred,
            "prediction_name": self.class_names[pred],
            "confidence": float(probs[pred]),
            "top5_predictions": topk_idx.tolist(),
            "top5_prediction_names": [self.class_names[i] for i in topk_idx],
            "top5_confidences": probs[topk_idx].tolist(),
            "raw_output": {
                "label_scores": [float(x) for x in label_scores],
                "calib_T": self.calib_T,
            },
        }

    # ---------------- internal helpers ---------------- #

    def _build_prefix(self, messages: List[Dict[str, str]]) -> str:
        """Use HF chat template if available, otherwise fall back to a simple format."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            system = messages[0]["content"]
            user = messages[1]["content"]
            return (
                f"<|system|>\n{system}\n"
                f"<|user|>\n{user}\n"
                f"<|assistant|>\n"
            )

    @torch.no_grad()
    def _first_token_loglik_per_label(self, prefix: str, labels: List[str]) -> List[float]:
        """
        Run one forward pass on the prefix, take the logits at the last position
        and use the logit of each label's first token (with a leading space) as its score.
        """
        inputs = self.tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        outputs = self.model(**inputs)
        # logits shape: [1, seq_len, vocab_size]
        last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        scores = []
        for label in labels:
            completion = " " + label
            comp_ids = self.tokenizer(
                completion,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]

            # use first token id as label anchor
            token_id = comp_ids[0].item()
            scores.append(last_logits[token_id].item())

        return scores
