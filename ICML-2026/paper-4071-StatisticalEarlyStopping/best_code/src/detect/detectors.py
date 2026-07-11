from abc import ABC, abstractmethod
from typing import Optional, Tuple

from src.detect.evaluation_judge_prompts import (
    LLM_JUDGE_ABSTENTION_PROMPT,
)
from src.inference import Inference
import logging

logger = logging.getLogger(__name__)

class Detector(ABC):
    """Determines whether a response constitutes an abstention"""

    def __init__(self):
        pass

    @abstractmethod
    def detect(
        self, **kwargs
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Detects abstention in model's response, returns is_abstention label
        which is bool or None (in case detector wasn't able to determine abstention label)
        and optionally a longer Detector's response (relevant for LLM judge).
        """
        pass


################################################################################
##############################Abstention detector##############################
################################################################################


class LLMJudgeAbstentionDetector(Detector):
    USE_RESPONSE_WITH_REASONING = False

    def __init__(
        self, judge_model: Inference = Inference(model_name="Qwen/Qwen3-8B", temperature=0), *args, **kwargs
    ):
        self._judge_model = judge_model

    def detect(self, question = "", model_answer = "", **kwargs) -> Tuple[Optional[bool], Optional[str]]:
        # Reference answers are list or None

        judge_prompt = LLM_JUDGE_ABSTENTION_PROMPT.format(
            question=question,
            model_answer=model_answer,
        )

        # Temporary workaround for batched inference
        judge_response = self._judge_model.get_response([judge_prompt], enable_thinking=False)[0]
        judge_response = judge_response.lower().strip(" .,\n")
        if judge_response not in ["yes", "no"]:
            logger.warning(
                f"\nUnexpected judge response:\n{judge_response}"
                f"\nJudge prompt:\n{judge_prompt}"
            )
            return None, judge_response
        return "yes" in judge_response.lower(), judge_response

    def batch_detect(self, questions, model_answers, **kwargs):
        judge_prompts = [
            LLM_JUDGE_ABSTENTION_PROMPT.format(
                question=question,
                model_answer=model_answer,
            )
            for question, model_answer in zip(questions, model_answers)
        ]

        # Temporary workaround for batched inference
        judge_responses = self._judge_model.get_response(judge_prompts, enable_thinking=False)
        results = []
        for judge_response in judge_responses:
            judge_response = judge_response.lower().strip(" .,\n")
            if judge_response not in ["yes", "no"]:
                logger.warning(
                    f"\nUnexpected judge response:\n{judge_response}"
                )
                results.append((None, judge_response))
            else:
                results.append((judge_response == "yes", judge_response))
        return results


if __name__ == "__main__":
    # Example usage
    question = "How does the number of stars in that region of the sky change with distance (per unit range of distance, r)?\nA. ~ r^4\nB. ~ r^2\nC. ~ r^5\nD. ~ r^3"
    model_answer = "To determine how the number of stars in a region of the sky changes with distance \\( r \\) per unit range of distance, we need to consider the volume and density of stars.\n\n1. **Volume of a Sphere**: The volume of a spherical region of radius \\( r \\) is given by \\( \\frac{4}{3}\\pi r^3 \\).\n2. **Density of Stars**: The number of stars per unit volume (density) is constant. Therefore, the number of stars in a region of radius \\( r \\) is proportional to the volume, which is \\( r^3 \\).\n3. **Change with Distance**: If we consider the number of stars in a thin spherical shell at radius \\( r \\), the number of stars in the shell is proportional to the surface area of the shell, which is \\( 4\\pi r^2 \\). The change in the number of stars as the radius increases by \\( dr \\) is proportional to \\( r^2 \\).\n\nThus, the number of stars in a region of the sky changes with distance per unit range of distance \\( r \\) is proportional to \\( r^2 \\).\n\n\\[\n\\boxed{B}\n\\]"

    # Another example
    question2 = "If the total number of customers by the third day was 500, how many customers did she count on the third day?"
    answer2 = "The problem does not provide enough information to determine the number of customers on the third day. To find the number of customers on the third day, we would need the total number of customers on the first and second days, or some other data about the distribution of customers over the three days. Without this information, it's impossible to calculate the exact number of customers on the third day.\n\nHowever, if we were to assume that the number of customers is the same each day, the average number of customers per day would be approximately 166.666, but since the number of customers must be a whole number, this is not feasible. Therefore, the problem is missing necessary details to provide a precise answer.\n\nAnswer: \\boxed{166} (Note: This is an approximation and assumes a uniform distribution, but it's not a whole number.)"

    llm_judge_detector = LLMJudgeAbstentionDetector()
    # Test batch detection
    questions_batch = [question, question2]
    answers_batch = [model_answer, answer2]
    batch_results = llm_judge_detector.batch_detect(questions_batch, answers_batch)
    for i, (is_abstention, judge_response) in enumerate(batch_results):
        print(f"LLM Judge (Batch) - Question {i+1}: Is abstention: {is_abstention}, Judge response: {judge_response}")