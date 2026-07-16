# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# attacks.py
# Description: Watermark-removal attack framework. A shared base
#   harness loads the detector, loops over watermark schemes, rewrites
#   each sample, detects, saves results, and computes TPR-at-fixed-FPR.
#   A new attack only needs to subclass WatermarkRemovalAttack and
#   implement load_models() / rewrite() / format_results() / output_path().
# =================================================================

import json
import os
import sys

import numpy as np
import torch
import transformers

sys.path.append(".")
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

from attack_utils.utils import HfModel, APIModel
from evaluation.tools.text_editor import DipperParaphraser
from evaluation.dataset import C4Dataset
from evaluation.tpr import load_or_compute_human_text_results, evaluate_detectability, attach_detectability
from evaluation.paths import attack_result_path, watermarked_response_path

import yaml
from types import SimpleNamespace
from tqdm import tqdm


SUPPORTED_WATERMARKS = [
    "KGW", "Unigram", "SWEET", "EWD", "SIR", "DIP", "Unbiased",
    "UPV", "TS", "SynthID", "EXP", "EXPGumbel", "EXPEdit", "ITSEdit",
]

# Watermark detector model, standardized across generation and detection so the
# positive (attack) and negative (human-text) score distributions stay scale-consistent.
DETECTOR_MODEL = "/models/opt-1.3b"
DETECTOR_MAX_NEW_TOKENS = 230

SYSPROMPT = """You are an expert paraphrasing system designed to rewrite text while maintaining complete semantic fidelity. Follow these precise guidelines:

CORE REQUIREMENTS:
- Preserve 100% of the original meaning and information content
- Maintain identical factual accuracy, tone, and intent
- Keep the same level of formality/informality as the source
- Retain all specific details, numbers, names, and technical terms exactly as provided

TRANSFORMATION RULES:
- Use synonyms and alternative phrasings where semantically equivalent
- Vary sentence structure and length for natural flow
- Reorganize clauses and phrases without altering logical relationships
- Maintain coherence and readability in the target language
- Ensure each transformed sentence conveys the same scope and specificity as the original

STRICT PROHIBITIONS:
- Do NOT add interpretations, assumptions, elaborations, or external knowledge
- Do NOT introduce information not explicitly present in the original text
- Do NOT amplify, exaggerate, or minimize any claims or statements
- Do NOT omit any information present in the original
- Do NOT change the perspective, viewpoint, or stance
- Do NOT include explanations, commentary, or meta-text
- Do NOT expand on implicit meanings or draw inferences
- Do NOT add contextual information or background details

CONTENT BOUNDARIES:
- Work ONLY with information explicitly stated in the source text
- If the original is vague or general, keep the paraphrase equally vague or general
- If the original is specific, maintain that exact level of specificity
- Do not fill in gaps or provide additional details, even if they seem logical

OUTPUT FORMAT:
- Provide only the paraphrased text
- Match the original format (paragraphs, lists, etc.)
- No prefacing remarks, explanations, or additional content

QUALITY CHECK:
Before outputting, verify that:
1. Someone reading only your paraphrase would understand exactly the same information as someone reading the original text
2. No new information has been introduced
3. No original information has been lost or altered
4. The scope and specificity remain identical
"""


# =================================================================
# Attack primitives
# =================================================================

class SelfInformationCalculator:
    """BIRA self-information: ranks tokens by self-information and returns the
    high-self-information tokens (above the percentile) to ban / bias against."""

    def __init__(self, model):
        self.model = model.model
        self.tokenizer = model.tokenizer

    def calculate_self_information(self, text: str):
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs).detach()

        input_ids = encoding["input_ids"]
        input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)

        tokens = [token_ for token_ in input_ids.squeeze().tolist()[1:]]

        self_info_values = (
            self_info[:, :-1]
            .gather(-1, input_ids_expaned)
            .squeeze(-1)
            .squeeze(0)
            .tolist()
        )
        assert len(tokens) == len(
            self_info_values
        ), f"length of tokens is not equal to length of info"
        return tokens, self_info_values

    def transform_tokens(self, text, threshold_low):
        tokens, self_info_values = self.calculate_self_information(text)

        sorted_pairs = sorted(zip(self_info_values, tokens), reverse=True)
        self_info_values, tokens = zip(*sorted_pairs)
        self_info_values = list(self_info_values)
        tokens = list(tokens)

        percentile = np.percentile(self_info_values, threshold_low)
        proxy_suppression_set = []
        proxy_suppression_texts = []

        for idx, (token, self_info) in enumerate(zip(tokens, self_info_values)):
            if self_info >= percentile:
                text = self.tokenizer.decode(token)

                if token not in proxy_suppression_set:
                    proxy_suppression_set.append(token)
                    proxy_suppression_texts.append(text)

        return proxy_suppression_set, proxy_suppression_texts


class HighSurprisalLogitProcessor(LogitsProcessor):
    """Applies the BIRA logit bias (beta) to the high-surprisal (high self-information)
    tokens in the proxy suppression set."""

    def __init__(self, proxy_suppression_sets, beta):
        super().__init__()
        self.proxy_suppression_sets = proxy_suppression_sets
        self.beta = beta

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.Tensor:
        for b, proxy_suppression_set in enumerate(self.proxy_suppression_sets):
            if input_ids[b].shape[0] < 1:
                continue
            logits[b, proxy_suppression_set] += self.beta

        return logits


def check_last_ngram_repeats(
    text: str,
    tokenizer,
    h: int = 450,
    threshold: float = 0.25,
    n_max: int = 1,   # use 1..3 by default
) -> bool:
    """
    Detects text collapse by measuring distinct-n diversity in a recent window.

    Flags collapse if min(distinct-1 .. distinct-n_max) in the last `h` tokens
    is below `threshold`.
    """
    # Tokenize (avoid special tokens if supported)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)

    if not ids:
        return False

    w = ids[-min(h, len(ids)):]

    def distinct_ratio(arr, n: int) -> float:
        m = len(arr) - n + 1
        if m <= 0:
            return 1.0  # not enough context → treat as fully diverse (avoid false positives)
        if n == 1:
            units = arr
        else:
            units = [tuple(arr[i:i+n]) for i in range(m)]
        return len(set(units)) / m

    distinct = distinct_ratio(w, n_max)

    return distinct < threshold


def load_dipper_paraphraser(name="dipper-1", cached_dir=None):
    if name == "dipper-1":
        lex_diversity = 60
        order_diversity = 0
        sent_interval = 1
        max_new_tokens = 120
        do_sample = True
        top_p = 0.75
        top_k = None
    elif name == "dipper-2":
        lex_diversity = 60
        order_diversity = 40
        sent_interval = 1
        max_new_tokens = 120
        do_sample = True
        top_p = 0.75
        top_k = None
    else:
        raise ValueError(f"Invalid name: {name}")

    dipper_paraphraser = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('google/t5-v1_1-xxl', cache_dir=cached_dir),
                            model=T5ForConditionalGeneration.from_pretrained('kalpeshk2011/dipper-paraphraser-xxl', device_map='auto', cache_dir=cached_dir),
                            lex_diversity=lex_diversity, order_diversity=order_diversity, sent_interval=sent_interval,
                            max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, top_k=top_k)
    return dipper_paraphraser


# ---- SIRA primitives (self-information blanking + prompt builders) ----
# SIRA's SelfInformationCalculator is a DIFFERENT class than BIRA's above: it
# blanks high-self-information spans rather than banning tokens.

def fill_paraphrase_prompt(input_text):
    return (
        "You are a paraphraser. You are given an input passage 'INPUT'. "
        "You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should be diverse and different "
        "as much as possible from 'INPUT' and should not copy any part verbatim from 'INPUT'. "
        "However, 'OUTPUT' should preserve the information in the INPUT. "
        "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: "
        + input_text
    )


def fill_attack_prompt(text, blank_text):
    PROMPT = """You will be shown one reference paragraph and one incomplete paragraph.
    Your task is to write a complete paragraph using incomplete paragraph.
    The complete paragraph should have similar length with reference paragraph.
    You need to include all the information in the reference. 
    but do not take the expression and words in the reference paragraph.
    You should only answer the complete paragraph.
    """
    PROMPT += "reference:" + text + "\n"
    PROMPT += "incomplete paragraph:" + blank_text + "\n"
    return PROMPT


class SIRASelfInformationCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._prepare_model()

    def _prepare_model(self):
        self.model.eval()
        print("SelfInformationCalculator: Model and tokenizer loaded successfully.")

    def calculate_self_information(self, text: str):
        with torch.no_grad():
            if not text.strip():
                return [], []
            encoding = self.tokenizer(
                text, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)
            if encoding["input_ids"].shape[1] == 0:
                return [], []
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)

        input_ids = encoding["input_ids"]
        if input_ids.shape[1] <= 1:
            return [], []

        input_ids_expanded = input_ids[:, 1:].unsqueeze(-1)

        tokens = [
            self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]
        ]
        self_info_values = (
            self_info[:, :-1]
            .gather(-1, input_ids_expanded)
            .squeeze(-1)
            .squeeze(0)
            .tolist()
        )

        return tokens, self_info_values

    def transform_tokens(self, tokens, self_info_values, threshold_low):
        if not tokens or not self_info_values:
            return []
        percentile = np.percentile(self_info_values, threshold_low)
        transformed_tokens = []
        current_low_si_chunk = []

        for token, self_info in zip(tokens, self_info_values):
            if self_info <= percentile:
                current_low_si_chunk.append(token)
            else:
                if current_low_si_chunk:
                    transformed_tokens.append(f"({' '.join(current_low_si_chunk)})")
                    current_low_si_chunk = []
                transformed_tokens.append("_")

        if current_low_si_chunk:
            transformed_tokens.append(f"({' '.join(current_low_si_chunk)})")

        return transformed_tokens


class WatermarkRemovalAttack:
    """Base harness shared by all watermark-removal attacks.

    Subclasses implement four hooks:
      load_models()            -- load attack-specific models once; set self.model_name
      rewrite(text)            -- attack one sample -> (attacked_text, meta_dict)
      format_results(records)  -- build the list of dicts written to the output JSON
      output_path(algorithm)   -- the result JSON path for one watermark scheme
    """

    def __init__(self, args):
        self.args = args
        self.model_name = None  # set by load_models()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- hooks implemented by subclasses ----
    def load_models(self):
        raise NotImplementedError

    def rewrite(self, text):
        raise NotImplementedError

    def format_results(self, records):
        raise NotImplementedError

    def output_path(self, algorithm_name):
        raise NotImplementedError

    # ---- shared harness ----
    def run(self):
        args = self.args
        for algorithm in args.algorithms:
            assert algorithm in SUPPORTED_WATERMARKS

        self.load_models()

        # Watermark detector (opt-1.3b). Only the evaluation pass needs it — NOT
        # rewriting — so it is built lazily here. Using an attack purely to rewrite
        # text (see examples/attack_example.py) never loads it and needs no watermark.
        detector_model = AutoModelForCausalLM.from_pretrained(DETECTOR_MODEL).to(self.device)
        detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_MODEL)
        detector_config = TransformersConfig(
            model=detector_model,
            tokenizer=detector_tokenizer,
            vocab_size=detector_model.config.vocab_size,
            device=self.device,
            max_new_tokens=DETECTOR_MAX_NEW_TOKENS,
            do_sample=True,
            no_repeat_ngram_size=4,
        )

        for algorithm_name in args.algorithms:
            myWatermark = AutoWatermark.load(
                f"{algorithm_name}",
                algorithm_config=f"config/{algorithm_name}.json",
                transformers_config=detector_config,
            )

            dataset_path = watermarked_response_path(args.input_path, algorithm_name)
            num_data = args.num_data
            with open(dataset_path, "r") as f:
                lines = f.readlines()[:num_data]
            dataset = [json.loads(line)["watermarked_text"] for line in lines]

            print(f"\nAttack algorithms: {args.attack_algorithms}\n")
            print(f"# of test data: {num_data}\n")

            records = []
            two_pass_count = 0
            two_pass_success = 0
            for text in tqdm(dataset[:num_data], desc="Removing watermarked text"):
                attacked_text, meta = self.rewrite(text)
                detect_result = myWatermark.detect_watermark(attacked_text)
                
                # ALGO-1: Two-pass re-attack with stronger beta for detected samples
                if (detect_result["is_watermarked"] 
                    and not meta.get("is_degenerated", False)
                    and args.attack_algorithms == "BIRA"):
                    two_pass_count += 1
                    original_beta = args.beta
                    second_beta = max(args.beta * 1.5, -8.0)
                    args.beta = second_beta
                    attacked_text2, meta2 = self.rewrite(text)
                    detect_result2 = myWatermark.detect_watermark(attacked_text2)
                    args.beta = original_beta
                    
                    if (not detect_result2["is_watermarked"] 
                        and not meta2.get("is_degenerated", False)):
                        attacked_text = attacked_text2
                        detect_result = detect_result2
                        meta = {"generation_iter": meta["generation_iter"],
                                "is_degenerated": meta2["is_degenerated"],
                                "second_pass": True, "first_pass_iter": meta["generation_iter"]}
                        two_pass_success += 1
                
                print(f"\ndetect_result: {detect_result}")
                records.append({
                    "watermarked_text": text,
                    "attacked_text": attacked_text,
                    "is_watermarked": True if detect_result["is_watermarked"] else False,
                    "score": detect_result["score"],
                    "meta": meta,
                })

            detection_scores = [r["score"] for r in records]
            avg_ASR = 1 - sum(r["is_watermarked"] for r in records) / num_data
            if two_pass_count > 0:
                print(f"\n[ALGO-1] Two-pass triggered on {two_pass_count} samples, "
                      f"succeeded on {two_pass_success}")
            print(f"\navg_ASR: {avg_ASR:.2f}")

            total_results = self.format_results(records)

            # ---- Detectability evaluation (TPR-at-fixed-FPR and/or best-threshold F1) ----
            # Reuse the same myWatermark detector (opt-1.3b) so the positive (attack)
            # and negative (human-text) score distributions are scale-consistent. The
            # metrics are embedded in the attack-result JSON itself, next to avg_ASR.
            human_text_dataset = C4Dataset(args.dataset_path, max_samples=500)
            human_text_results = load_or_compute_human_text_results(
                myWatermark, human_text_dataset, args.human_text_result_save_dir, algorithm_name
            )
            detectability = evaluate_detectability(
                detection_scores, human_text_results, args.labels,
                rules=args.rules, target_fprs=args.target_fprs,
            )
            total_results = attach_detectability(total_results, detectability)

            output_path_name = self.output_path(algorithm_name)
            os.makedirs(os.path.dirname(output_path_name), exist_ok=True)
            print(f"{output_path_name}")
            with open(output_path_name, "w") as f:
                json.dump(total_results, f, indent=4)
            for name, metric in detectability.items():
                print(f"  {name}: {metric}")


class BIRAAttack(WatermarkRemovalAttack):
    """BIRA (and the vanilla_paraphrasing baseline).

    Marks high-self-information tokens and biases the rewriter away from them,
    iterating (raising beta) while the output collapses into repetition.
      backend=hf : local model, bias applied via a logit processor.
      backend=api: API model, bias applied via logit_bias; a local auxiliary
                   model computes the self-information mask.
    vanilla_paraphrasing is the no-mask, single-pass mode.
    """

    def __init__(self, args):
        super().__init__(args)
        self.backend = args.backend
        self.threshold_low = args.percentile
        if args.attack_algorithms == "vanilla_paraphrasing" and self.backend == "api":
            args.beta = 0

    def load_models(self):
        args = self.args
        if self.backend == "hf":
            with open(args.model_cfg_path, "r") as f:
                model_cfg = SimpleNamespace(**yaml.safe_load(f))
            model_cfg.use_sampling = args.use_sampling
            if not model_cfg.use_sampling:
                model_cfg.sampling_temp = 1.0
            self.gen_model = HfModel(model_cfg=model_cfg, sys_prompt=SYSPROMPT)
            self.aux_model = self.gen_model
            self.model_name = model_cfg.name.split("/")[-1]
        else:  # api
            if args.attack_algorithms == "BIRA":
                with open(args.model_cfg_path, "r") as f:
                    model_cfg = SimpleNamespace(**yaml.safe_load(f))
                self.aux_model = HfModel(model_cfg=model_cfg, sys_prompt=SYSPROMPT)
            else:
                self.aux_model = None
            self.inference_model = APIModel(model_name=args.api_model_name)
            self.model_name = args.api_model_name

    def rewrite(self, text):
        args = self.args
        prompts = [f"\nDocuments:\n{text}\n"]
        contexts = [text]

        processors = None
        logit_bias_tokens = None
        max_iter = args.max_iter

        if args.attack_algorithms == "BIRA":
            calculator = SelfInformationCalculator(model=self.aux_model)
            high_surprisal_token_lists = []
            high_surprisal_text_lists = []
            for c in contexts:
                high_surprisal_token, high_surprisal_text = calculator.transform_tokens(
                    c, threshold_low=self.threshold_low
                )
                high_surprisal_token_lists.append(high_surprisal_token)
                high_surprisal_text_lists.append(high_surprisal_text)

            if self.backend == "hf":
                processors = HighSurprisalLogitProcessor(
                    proxy_suppression_sets=high_surprisal_token_lists,
                    beta=args.beta,
                )
            else:
                logit_bias_tokens = [
                    self.inference_model.get_gpt_token_ids(t) for t in high_surprisal_text_lists[0]
                ]
                logit_bias_tokens = sum(logit_bias_tokens, [])[:300]
        else:  # vanilla_paraphrasing
            max_iter = 1

        beta = args.beta
        iteration = 0
        for _ in range(max_iter):
            iteration += 1
            if self.backend == "hf":
                if processors is not None:
                    processors.beta = beta
                results = self.gen_model.generate(
                    prompts,
                    logit_processors=LogitsProcessorList([processors]) if processors is not None else None,
                )[0].strip()
                is_degenerated = check_last_ngram_repeats(results, tokenizer=self.gen_model.tokenizer)
            else:  # api
                iter_results = self.inference_model.generate_with_logit_bias(
                    sysprompt=SYSPROMPT,
                    prompts=prompts,
                    response_format=None,
                    logit_bias=beta,
                    logit_bias_tokens=logit_bias_tokens,
                )
                for results in iter_results:
                    if args.attack_algorithms == "BIRA":
                        is_degenerated = check_last_ngram_repeats(results, tokenizer=self.aux_model.tokenizer)
                    else:
                        is_degenerated = False

            if not is_degenerated:
                break

            beta += args.learning_rate
            print(f"Learning_rate: {args.learning_rate}")
            print(f"\nReducing beta: {beta}")

        return results, {"generation_iter": iteration, "is_degenerated": is_degenerated}

    def format_results(self, records):
        num_data = self.args.num_data
        items = [
            {
                "d_idx": d_idx,
                "watermarked_text": r["watermarked_text"],
                "attacked_text": r["attacked_text"],
                "is_watermarked": r["is_watermarked"],
                "score": r["score"],
                "generation_iter": r["meta"]["generation_iter"],
                "is_degenerated": r["meta"]["is_degenerated"],
            }
            for d_idx, r in enumerate(records)
        ]
        total_suspicious = sum(it["is_degenerated"] for it in items)
        avg_ASR = 1 - sum(r["is_watermarked"] for r in records) / num_data
        avg_iteration = sum(r["meta"]["generation_iter"] for r in records) / num_data
        return items + [
            {"total_suspicious": total_suspicious},
            {"avg_ASR": avg_ASR},
            {"avg_iteration": avg_iteration},
        ]

    def output_path(self, algorithm_name):
        args = self.args
        return attack_result_path(
            args.result_save_dir, args.attack_algorithms, algorithm_name, self.model_name,
            num_data=args.num_data, beta=args.beta, percentile=args.percentile,
        )


class DipperAttack(WatermarkRemovalAttack):
    """DIPPER paraphrase attack (dipper-1 / dipper-2)."""

    def load_models(self):
        self.model = load_dipper_paraphraser(name=self.args.attack_algorithms)
        # dipper has no rewriting LLM; model_name is just the TPR output-path label.
        self.model_name = self.args.model_name

    def rewrite(self, text):
        results = self.model.edit(text, reference="")
        return results, {}

    def format_results(self, records):
        num_data = self.args.num_data
        items = [
            {
                "d_idx": d_idx,
                "watermarked_text": r["watermarked_text"],
                "attacked_text": r["attacked_text"],
                "is_watermarked": r["is_watermarked"],
                "score": r["score"],
            }
            for d_idx, r in enumerate(records)
        ]
        avg_ASR = 1 - sum(r["is_watermarked"] for r in records) / num_data
        return items + [{"avg_ASR": avg_ASR}]

    def output_path(self, algorithm_name):
        args = self.args
        return attack_result_path(
            args.result_save_dir, args.attack_algorithms, algorithm_name, num_data=args.num_data,
        )


class SIRAAttack(WatermarkRemovalAttack):
    """SIRA (Self-Information Rewrite Attack).

    Per sample, three per-document-independent stages:
      1. paraphrase the watermarked text          -> reference text
      2. self-information blanking (mask high-SI)  -> blanked text
      3. attack: fill the blanks using the reference -> attacked text
    Stages 1 & 3 use a chat text-generation pipeline (greedy); stage 2 uses the
    SIRA SelfInformationCalculator. The base harness then detects / saves / TPRs.
    """

    @staticmethod
    def _extract(outputs):
        """Pull the assistant content out of a transformers chat-pipeline output."""
        generated = ""
        if outputs and outputs[0]["generated_text"]:
            gen = outputs[0]["generated_text"]
            if isinstance(gen, list) and len(gen) > 0:
                last = gen[-1]
                generated = last["content"] if isinstance(last, dict) and "content" in last else str(last)
            elif isinstance(gen, str):
                generated = str(gen)
        return generated

    def load_models(self):
        args = self.args
        self.model_name = args.model_name
        # Paraphrase/blanking model (stages 1 & 2): loaded once and shared by the
        # generation pipeline and the self-information calculator.
        para_model = AutoModelForCausalLM.from_pretrained(
            args.paraphrase_model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        para_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_model_path)
        self.paraphrase_pipeline = transformers.pipeline(
            "text-generation", model=para_model, tokenizer=para_tokenizer
        )
        self.calculator = SIRASelfInformationCalculator(model=para_model, tokenizer=para_tokenizer)
        # Attack model (stage 3): reuse the paraphrase model if it is the same path.
        if args.attack_model_path == args.paraphrase_model_path:
            self.attack_pipeline = self.paraphrase_pipeline
        else:
            attack_model = AutoModelForCausalLM.from_pretrained(
                args.attack_model_path, device_map="auto", torch_dtype=torch.bfloat16
            )
            attack_tokenizer = AutoTokenizer.from_pretrained(args.attack_model_path)
            self.attack_pipeline = transformers.pipeline(
                "text-generation", model=attack_model, tokenizer=attack_tokenizer
            )

    def rewrite(self, text):
        args = self.args

        # Stage 1: paraphrase -> reference text
        para_messages = [
            {"role": "system", "content": "You are a helpful rewriter."},
            {"role": "user", "content": fill_paraphrase_prompt(text)},
        ]
        ref_text = self._extract(
            self.paraphrase_pipeline(para_messages, max_new_tokens=args.max_new_tokens, do_sample=False)
        )

        # Stage 2: self-information blanking
        tokens, self_info_values = self.calculator.calculate_self_information(text)
        if not tokens:
            blank_text = ""
        else:
            blank_text = "".join(self.calculator.transform_tokens(tokens, self_info_values, args.threshold))

        # Stage 3: attack — fill the blanks using the reference text
        attack_messages = [{"role": "user", "content": fill_attack_prompt(ref_text, blank_text)}]
        attack_text = self._extract(
            self.attack_pipeline(attack_messages, max_new_tokens=args.max_new_tokens, do_sample=False)
        )

        return attack_text, {}

    def format_results(self, records):
        num_data = self.args.num_data
        items = [
            {
                "watermarked_text": r["watermarked_text"],
                "SIRA_text": r["attacked_text"],
                "is_watermarked": r["is_watermarked"],
                "score": r["score"],
            }
            for r in records
        ]
        avg_ASR = 1 - sum(r["is_watermarked"] for r in records) / num_data
        return items + [{"avg_ASR": avg_ASR}]

    def output_path(self, algorithm_name):
        return attack_result_path(
            self.args.result_save_dir, "SIRA", algorithm_name, self.args.model_name,
            num_data=self.args.num_data,
        )


# attack_algorithms value -> Attack class
ATTACK_REGISTRY = {
    "BIRA": BIRAAttack,
    "vanilla_paraphrasing": BIRAAttack,
    "dipper-1": DipperAttack,
    "dipper-2": DipperAttack,
    "SIRA": SIRAAttack,
}
