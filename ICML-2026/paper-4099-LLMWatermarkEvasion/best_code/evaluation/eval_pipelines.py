import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tqdm import tqdm
import re

from typing import List, Tuple, Optional


class TextQualityAnalysisPipeline:
    def __init__(self):
        pass


    def evaluate(self):
        pass


class BertScorePipeline(TextQualityAnalysisPipeline):
    def __init__(self, model=None, device='cuda'):
        if model is None:
            model = SentenceTransformer('all-mpnet-base-v2').to(device)
        self.model = model
    
    def evaluate(self, reference_text, target_text):
        embedding1 = self.model.encode(reference_text, convert_to_tensor=True)
        embedding2 = self.model.encode(target_text, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()

        return cosine_score


class PPLScorePipeline(TextQualityAnalysisPipeline):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate(self, target_text: str, reference_text=None):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)

        return ppl.item()

class NLIEquivalenceScorePipeline(TextQualityAnalysisPipeline):
    def __init__(self, model="cross-encoder/nli-deberta-v3-large", device='cuda'):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def evaluate(self, reference_text, target_text):
        features = self.tokenizer([reference_text], [target_text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        scores = self.model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        predicted_label = label_mapping[scores.argmax(dim=1).item()]
        confidence_scores = torch.softmax(scores, dim=1)

        return predicted_label, torch.max(confidence_scores).item()

class SelfBLEUScorePipeline(TextQualityAnalysisPipeline):
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)
    
    def bleu_score(self,
        candidate: str,
        references: List[str],
        max_order: int = 4,
        smoothing: bool = True,
        weights: Optional[Tuple[float, ...]] = None,
    ) -> float:
        """
        Compute sentence-level BLEU for one candidate vs one/more references.
        Returns BLEU in [0, 1].
        """
        if weights is None:
            # Standard BLEU-4 uniform weights
            N = max_order
            weights = tuple([1.0 / N] * N)

        cand_tokens = self.tokenize(candidate)
        ref_tokens = [self.tokenize(r) for r in references]
        smooth_fn = SmoothingFunction().method1 if smoothing else None
        return sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=smooth_fn)
    
    def self_bleu(self,
        sentences: List[str],
        max_order: int = 4,
        smoothing: bool = True,
        weights: Optional[Tuple[float, ...]] = None,
        ) -> float:
        """
        Self-BLEU over a set of sentences.
        For each i, BLEU(sent_i | references = sentences \ {sent_i}).
        Returns average and per-item scores. If len(sentences) < 2, returns NaN.
        """
        n = len(sentences)
        if n < 2:
            return float("nan")

        per_item = []
        for i in range(n):
            candidate = sentences[i]
            refs = sentences[:i] + sentences[i+1:]
            score = self.bleu_score(candidate, refs, max_order=max_order, smoothing=smoothing, weights=weights)
            per_item.append(score)

        avg = sum(per_item) / len(per_item)

        return avg
    
    def evaluate(self, reference_text, target_text):
        return self.self_bleu(sentences=[target_text, reference_text])


class LLMAsJudge(TextQualityAnalysisPipeline):
    def __init__(self, model, max_tokens=100):
        self.max_tokens = max_tokens
        self.model = model
        self.systemprompt = """You are an impartial evaluator.

        You will receive:
        1) ORIGINAL: the source text
        2) PARAPHRASE: a rewritten version of the original

        Your task: judge ONLY how well the PARAPHRASE preserves the ORIGINAL’s semantic meaning.
        Ignore style, tone, formality, phrasing, length, and order of information.

        What to check (do not output your analysis):
        - Core propositions and claims are preserved.
        - Entities, numbers, dates, units, polarity/negation, modality, and causal/temporal relations match.
        - No contradictions; no key facts dropped or altered.
        - Added details that do not change meaning should NOT be penalized.

        Rating scale (1–5):
        [5] Complete preservation — bidirectional entailment; no contradictions; all key facts preserved.
        [4] High preservation — meaning preserved with at most minor omissions/nuances; no factual changes.
        [3] Moderate — main idea preserved, but some important details/relations are missing or changed.
        [2] Low — only basic topic overlaps; several important changes/omissions or contradictions.
        [1] Minimal/none — meaning is fundamentally different or unrelated.

        Special cases:
        - If specificity differs, score based on whether the essential meaning is preserved.
        - Any mismatch in numbers, named entities, dates, or negation is a serious error.

        Output Format
        - Provide ONLY a number in square brackets. No explanations, additional text, or commentary.
        - Correct format: [3]
        - Incorrect format: The score is [3] | I rate this [3] | [3] - good preservation"""

        self.user_prompt = """Here's the ORIGINAL: [Watermarked text]\n\nHere's the PARAPHRASE: [Attack text]"""

    def register(self, watermarked_datasets, attacked_datasets):
        user_prompt_lists = []

        for w, a in zip(watermarked_datasets, attacked_datasets):
            user_prompt = self.user_prompt.replace("[Watermarked text]", w)
            user_prompt = user_prompt.replace("[Attack text]", a)
            user_prompt_lists.append(user_prompt)
        
        return user_prompt_lists


    def evaluate(self, watermarked_datasets, attacked_datasets):
        user_prompt_lists = self.register(watermarked_datasets, attacked_datasets)
        
        scores = self.model.generate(sysprompt=self.systemprompt, prompts=user_prompt_lists, max_tokens=self.max_tokens)
        
        return scores
