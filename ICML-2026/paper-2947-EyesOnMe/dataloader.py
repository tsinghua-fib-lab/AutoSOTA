import json
import re
from collections import Counter
from typing import List, Tuple
import config


class Dataset():
    def __init__(
        self,
        question_path: str = config.QUESTIONS_PATH,
        passage_path: str = config.PASSAGES_PATH,
        trigger_ratio: Tuple[float, float] = (config.TRIGGER_RATIO_MIN, config.TRIGGER_RATIO_MAX),
        num_test_triggers: int = config.NUM_TEST_TRIGGERS,
        filtered_word=None
    ):
        """
        Load questions and passages, pair them, filter by trigger word frequency,
        and select trigger words.

        Attributes set on self:
            full_dataset: list of (question, passage) pairs
            filtered_dataset: list of query strings matching the trigger word
            filtered_words: list of candidate trigger strings
        """
        with open(question_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        with open(passage_path, "r", encoding="utf-8") as f:
            passages = json.load(f)

        if filtered_word:
            filtered_words = [filtered_word]
        else:
            sentence_word_sets = [set(re.findall(r"\w+", q.lower())) for q in questions]
            word_counts = Counter(word for s in sentence_word_sets for word in s)
            total_queries = len(questions)
            min_ratio, max_ratio = trigger_ratio
            filtered_words = [
                word for word, count in word_counts.items()
                if count > min_ratio * total_queries and count < max_ratio * total_queries
            ]

        full_dataset = list(zip(questions, passages))
        filtered_dataset = [
            (q, p) for q, p in full_dataset
            if any(trigger in q.lower() for trigger in filtered_words)
        ]

        with open(config.QUERY_DICT_PATH, 'r') as f:
            filtered_dataset = json.load(f)[filtered_word][:num_test_triggers]

        self.full_dataset = full_dataset
        self.filtered_dataset = filtered_dataset
        self.filtered_words = filtered_words[:num_test_triggers]
