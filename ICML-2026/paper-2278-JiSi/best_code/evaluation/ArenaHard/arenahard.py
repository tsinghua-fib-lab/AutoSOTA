import os
import re
import json
from typing import Dict, Optional, Any

from datasets import Dataset, disable_progress_bars
from loguru import logger

from evaluation.base_evaluator import BaseEvaluator
from evaluation.ArenaHard.prompts import JUDGE_SETTINGS, PROMPT_TEMPLATE, REGEX_PATTERNS
from generators.generator import DirectGenerator

disable_progress_bars()

DATA_DIR = "data/ArenaHard"
BASELINE_ANSWER_DIR = "evaluation/ArenaHard/model_answer"


class ArenaHardEvaluator(BaseEvaluator):
    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        super().__init__(grader_cache_config)
        self.task = "arenahard"
        self.seed = 42

        # Initialize arena grader (uses ARENA_GRADER_* environment variables)
        self.arena_grader = DirectGenerator(
            model_name=os.getenv("ARENA_GRADER_MODEL_NAME", "deepseek"),
            base_url=os.getenv("ARENA_GRADER_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("ARENA_GRADER_API_KEY", ""),
            temperature=0.0,
            top_p=1.0,
            timeout=500,
            cache_config=grader_cache_config
        )

        # Load baseline answers for all categories
        self.baseline_answers = self._load_baseline_answers()

    def _load_baseline_answers(self) -> Dict[str, Dict[str, Any]]:
        """
        Load baseline model answers for all categories

        Returns:
            {baseline_model_name: {uid: answer_data}}
        """
        baseline_answers = {}

        # Get unique baseline models from JUDGE_SETTINGS
        baseline_models = set(
            settings["baseline"]
            for settings in JUDGE_SETTINGS.values()
        )

        for baseline_model in baseline_models:
            answer_file = os.path.join(BASELINE_ANSWER_DIR, f"{baseline_model}.jsonl")

            if not os.path.exists(answer_file):
                logger.warning(f"Baseline answer file not found: {answer_file}")
                continue

            # Load answers into dict indexed by uid
            answers = {}
            with open(answer_file, 'r', encoding='utf-8') as f:
                for line in f:
                    answer = json.loads(line.strip())
                    uid = answer.get('uid', '')
                    answers[uid] = answer

            baseline_answers[baseline_model] = answers
            logger.info(f"Loaded {len(answers)} baseline answers for {baseline_model}")

        return baseline_answers

    def load_data(self, split: str = "test"):
        # ArenaHard only has one dataset version (v2)
        data = self.load_jsonl(os.path.join(DATA_DIR, f"arena-hard-v2.jsonl"))

        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))

        # Add origin_query field for consistency with other evaluators
        data = data.map(lambda x: {**x, 'origin_query': x.get('prompt', '')})

        return data

    def get_valid_splits(self):
        return ["test"]

    def format_prompt(self, item: Dict) -> Dict:
        # ArenaHard uses the question directly as prompt
        return {
            "prompt": item.get('prompt', item.get('question', '')),
            "uid": item.get('uid', ''),
            "category": item.get('category', ''),
            "subcategory": item.get('subcategory', ''),
            "language": item.get('language', 'en')
        }

    def _get_score(self, judgment_text: str) -> Optional[str]:
        """
        Extract score label from judgment text using regex patterns

        Args:
            judgment_text: The judge model's response

        Returns:
            Score label like "A>>B", "A>B", "A=B", "B>A", "B>>A", or None
        """
        for pattern_str in REGEX_PATTERNS:
            pattern = re.compile(pattern_str)
            matches = pattern.findall(judgment_text.upper())
            matches = [m for m in matches if m != ""]

            if len(set(matches)) > 0:
                return matches[-1].strip("\n")

        return None

    def _pairwise_judgment(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        category: str
    ) -> Optional[str]:
        """
        Perform one round of pairwise judgment

        Args:
            question: The user question/prompt
            answer_a: Assistant A's answer
            answer_b: Assistant B's answer
            category: Question category (for system prompt)

        Returns:
            Score label like "A>>B", "A>B", etc., or None if failed
        """
        # Get system prompt for this category
        system_prompt = JUDGE_SETTINGS[category]["system_prompt"]

        # Format user prompt
        user_prompt = PROMPT_TEMPLATE.format(
            QUESTION=question,
            ANSWER_A=answer_a,
            ANSWER_B=answer_b
        )

        # Call arena grader (with automatic caching and retry)
        try:
            # DirectGenerator doesn't have generate_with_messages, so we combine into a single prompt
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            result = self.arena_grader.generate(full_prompt)

            # Extract score from response
            score = self._get_score(result.output)
            return score

        except Exception as e:
            logger.error(f"Arena grader failed: {e}")
            return None

    def _calculate_final_score(self, round1_label: Optional[str], round2_label: Optional[str]) -> float:
        """
        Calculate final score using point system

        Args:
            round1_label: Round 1 judgment (baseline=A, model=B)
            round2_label: Round 2 judgment (model=A, baseline=B)

        Returns:
            1.0 if better than baseline, 0.5 if equal, 0.0 if worse or invalid
        """
        # Round 1: baseline=A, model=B
        round1_map = {
            "B>>A": 3,   # model significantly better
            "B>A": 1,    # model slightly better
            "A=B": 0,    # tie
            "B=A": 0,    # tie
            "A>B": -1,   # baseline slightly better
            "A>>B": -3,  # baseline significantly better
        }

        # Round 2: model=A, baseline=B
        round2_map = {
            "A>>B": 3,   # model significantly better
            "A>B": 1,    # model slightly better
            "A=B": 0,    # tie
            "B=A": 0,    # tie
            "B>A": -1,   # baseline slightly better
            "B>>A": -3,  # baseline significantly better
        }

        # Handle invalid labels
        if round1_label is None or round2_label is None:
            return 0.0

        score1 = round1_map.get(round1_label, 0)
        score2 = round2_map.get(round2_label, 0)

        total = score1 + score2

        # Convert to 0/0.5/1
        if total > 0:
            return 1.0
        elif total == 0:
            return 0.5
        else:
            return 0.0

    def evaluate(self, data, output_text, **kwargs):
        """
        Evaluate ArenaHard output using pairwise comparison with baseline

        Args:
            data: Dictionary containing uid, prompt, category, etc.
            output_text: The model's response

        Returns:
            Dictionary with id, prediction, ground_truth, and is_correct (score)
        """
        uid = data.get('uid', '')
        question = data.get('prompt', '')
        category = data.get('category', 'hard_prompt')

        # Get baseline model for this category
        baseline_model = JUDGE_SETTINGS.get(category, JUDGE_SETTINGS['hard_prompt'])['baseline']

        # Get baseline answer
        if baseline_model not in self.baseline_answers:
            logger.error(f"Baseline model {baseline_model} not found in loaded answers")
            return {
                "id": uid,
                "prediction": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": {"answer": output_text}}
                ],
                "ground_truth": None,
                "is_correct": 0.0
            }

        baseline_answer_dict = self.baseline_answers[baseline_model].get(uid)
        if not baseline_answer_dict:
            logger.warning(f"Baseline answer not found for uid={uid}, baseline={baseline_model}")
            return {
                "id": uid,
                "prediction": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": {"answer": output_text}}
                ],
                "ground_truth": None,
                "is_correct": 0.0
            }

        # Extract baseline answer text
        baseline_answer = baseline_answer_dict['messages'][-1]['content']['answer']

        # Round 1: baseline=A, model=B
        round1_label = self._pairwise_judgment(
            question=question,
            answer_a=baseline_answer,
            answer_b=output_text,
            category=category
        )

        # Round 2: model=A, baseline=B
        round2_label = self._pairwise_judgment(
            question=question,
            answer_a=output_text,
            answer_b=baseline_answer,
            category=category
        )

        # Calculate final score
        score = self._calculate_final_score(round1_label, round2_label)

        # Format prediction as standard message format
        prediction = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": {"answer": output_text}}
        ]

        return {
            "id": uid,
            "prediction": prediction,
            "ground_truth": None,  # ArenaHard has no ground truth
            "is_correct": score  # 0.0, 0.5, or 1.0
        }
