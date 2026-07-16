import json
import re
import string
from tqdm import tqdm

class Evaluator:
    def __init__(
            self,
            result_dir: str = "../results/",
    ):
        self.result_dir = result_dir

    def _normalize_answer(
            self,
            answer: str
    ) -> str:
        """
        Normalize a given string by applying the following transformations:
        1. Convert the string to lowercase.
        2. Remove punctuation characters.
        3. Remove the articles "a", "an", and "the".
        4. Normalize whitespace by collapsing multiple spaces into one.

        Args:
            answer (str): The input string to be normalized.

        Returns:
            str: The normalized string.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(answer))))

    def exact_match(
            self,
            answer1: str,
            answer2: str,
    ):
        clean_answer1 = answer1.strip().replace(" ", "").lower()
        clean_answer2 = answer2.strip().replace(" ", "").lower()
        if clean_answer1 == clean_answer2 or clean_answer1 in clean_answer2 or clean_answer2 in clean_answer1:
            return True
        else:
            return False

    def f1_score(
            self,
            ground_truth: str,
            prediction: str,
    ):
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        prediction_tokens = self._normalize_answer(prediction).split()
        common = set(ground_truth_tokens) & set(prediction_tokens)

        if not common:
            return 0.0

        precision = len(common) / len(prediction_tokens)
        recall = len(common) / len(ground_truth_tokens)

        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    def llm_judge(
            self,
            question: str,
            ground_truth: str,
            prediction: str,
            llm,
    ) -> bool:
        # Construct evaluation prompt
        prompt = f"""
        You are an expert evaluator. Your task is to determine if the predicted answer is semantically equivalent to the ground truth answer for the given question.

        Question: {question}
        Ground Truth Answer: {ground_truth}
        Predicted Answer: {prediction}

        Instructions:
        - Compare the predicted answer and the ground truth answer in the context of the question.
        - They are considered equivalent if they convey the same meaning, even if the wording is different.
        - Respond in JSON format with two keys:
            "is_equivalent": true or false,
            "explanation": a brief explanation for your decision.

        Example response:
        {{
            "is_equivalent": true,
            "explanation": "Both answers correctly state that the capital of France is Paris."
        }}

        Important: Only output the JSON object and nothing else.
        """

        response_text = ""
        try:
            # Get evaluation from LLM
            response = llm.complete(prompt)
            response_text = response.text.strip()

            # [New Fix 1]: Remove potentially existing Markdown code block symbols
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON response
            result = json.loads(response_text)
            is_equivalent = result.get("is_equivalent", False)

            return bool(is_equivalent)

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            response_lower = response_text.lower()

            # [Fix 2]: Prioritize judging negative words and remove "equivalent," which is very likely to cause misjudgments.
            # Because "not equivalent" contains "not" while "is_equivalent" does not, this is safer.
            if "not equivalent" in response_lower or '"is_equivalent": false' in response_lower or "false" in response_lower or "no" in response_lower:
                return False
            elif '"is_equivalent": true' in response_lower or "true" in response_lower or "yes" in response_lower:
                return True
            else:
                # Default to False if uncertain
                return False

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return False


    def old_llm_judge(
            self,
            question: str,
            ground_truth: str,
            prediction: str,
            llm,
    ) -> bool:
        """
        Evaluate whether the predicted answer is semantically equivalent to the ground truth answer for the given question.

        Args:
            question: The question text
            ground_truth: The ground truth answer text
            prediction: The predicted answer text
            llm: LLM object from llama_index framework

        Returns:
            bool: True if semantically equivalent, False otherwise
        """
        # Construct evaluation prompt
        prompt = f"""
        You are an expert evaluator. Your task is to determine if the predicted answer is semantically equivalent to the ground truth answer for the given question.

        Question: {question}
        Ground Truth Answer: {ground_truth}
        Predicted Answer: {prediction}

        Instructions:
        - Compare the predicted answer and the ground truth answer in the context of the question.
        - They are considered equivalent if they convey the same meaning, even if the wording is different.
        - Respond in JSON format with two keys:
            "is_equivalent": true or false,
            "explanation": a brief explanation for your decision.

        Example response:
        {{
            "is_equivalent": true,
            "explanation": "Both answers correctly state that the capital of France is Paris."
        }}

        Important: Only output the JSON object and nothing else.
        """

        response_text = ""
        try:
            # Get evaluation from LLM
            response = llm.complete(prompt)
            response_text = response.text.strip()

            # Parse JSON response
            result = json.loads(response_text)
            is_equivalent = result.get("is_equivalent", False)

            return bool(is_equivalent)

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            response_lower = response_text.lower()
            if "true" in response_lower or "yes" in response_lower or "equivalent" in response_lower:
                return True
            elif "false" in response_lower or "no" in response_lower or "not equivalent" in response_lower:
                return False
            else:
                # Default to False if uncertain
                return False

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return False

    def evaluate_file_by_em(
            self,
            json_file_name: str,
            answer1_name: str,
            answer2_name: str,
    ):
        file_path = self.result_dir + json_file_name + ".json"
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Data has been successfully loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: File {file_path} does not exist")
            raise
        except json.JSONDecodeError:
            print(f"Error: file {file_path} is not a valid JSON format")
            raise
        except Exception as e:
            print(f"Error loading file: {e}")
            raise

        true_number = 0
        true_index_set = set()
        for i, item in enumerate(data):
            answer1 = item[answer1_name]
            answer2 = item[answer2_name]
            if self.exact_match(answer1, answer2):
                true_number = true_number + 1
                true_index_set.add(i)
        true_ratio = true_number / len(data)
        return true_ratio, true_index_set

    def evaluate_file_by_llm_judge(
            self,
            json_file_name: str,
            question_name: str,
            ground_truth_name: str,
            prediction_name: str,
            llm,
    ):
        file_path = self.result_dir + json_file_name + ".json"
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Data has been successfully loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: File {file_path} does not exist")
            raise
        except json.JSONDecodeError:
            print(f"Error: file {file_path} is not a valid JSON format")
            raise
        except Exception as e:
            print(f"Error loading file: {e}")
            raise

        true_number = 0
        true_index_set = set()
        for i, item in tqdm(enumerate(data)):
            question = item[question_name]
            ground_truth = item[ground_truth_name]
            prediction = item[prediction_name]
            if self.llm_judge(question, ground_truth, prediction, llm):
                true_number = true_number + 1
                true_index_set.add(i)
        true_ratio = true_number / len(data)
        return true_ratio, true_index_set


if __name__ == "__main__":
    pass
