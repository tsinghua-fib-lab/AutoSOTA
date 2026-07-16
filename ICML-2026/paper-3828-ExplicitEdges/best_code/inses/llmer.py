import json
import re
from typing import Optional, List, Dict, Any
from tqdm import tqdm


llm_only_prompt = """You are a helpful assistant that answers questions based on your own knowledge."""

llm_CoT_prompt = """You are a helpful assistant that answers questions based on your own knowledge.
Below are several examples of chain of thought. 
You can refer to these examples to think about the question and give the correct answer. 
Your answer must be returned in JSON format with two fields: "reasoning" and "answer".
The "reasoning" field should contain your step-by-step reasoning process, and the "answer" field should contain the final answer.
The "answer" field should be as concise as possible and should not contain unnecessary explanations.

Examples of Chain of Thought:
"""

# CoT examples
llm_CoT_examples = """
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.
Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.
Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.
Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.
Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.
Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.
"""



class LLMer:
    def __init__(
            self,
            llm_client,
            result_dir: str = "../results/",
    ):
        self.llm_client = llm_client
        self.result_dir = result_dir

    def _parse_json_response(self, response_text: str) -> Dict[str, str]:
        """Parse the JSON response returned by the model"""
        # Try parsing JSON directly
        try:
            result = json.loads(response_text)
            if "answer" in result:  # and "reasoning" in result:
                return result
        except json.JSONDecodeError:
            pass

        # If direct parsing fails, try extracting the JSON portion
        try:
            # Find the start and end positions of the JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                if "answer" in result:  # and "reasoning" in result:
                    return result
        except (json.JSONDecodeError, AttributeError):
            pass

        # If JSON parsing fails, try extracting the reasoning and answer from the text
        try:
            # Extract the reasoning section (from the beginning to before "The answer is")
            reasoning_match = re.search(r"^(.*?)The answer is", response_text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text

            # Extract the answer portion (the content within the curly braces)
            answer_match = re.search(r"The answer is \{([^}]+)\}", response_text)
            answer = answer_match.group(1).strip() if answer_match else "Answer not found"

            return {
                "reasoning": reasoning,
                "answer": answer
            }
        except Exception:
            # If all extraction methods fail, return the original response
            return {
                "reasoning": "Failed to parse response",
                "answer": response_text
            }

    def answer_by_llm(
            self,
            question: str,
    ):
        """
        Use GPT to evaluate retrieval results as a specific role, return a strict JSON result.
        """

        # llm only
        prompt = llm_only_prompt

        # Add current question
        prompt += f"\nQ: {question}\n"
        prompt += "Please provide your response in the following JSON format:\n"

        # llm only prompt
        prompt += '{\n  "answer": "Your final answer"\n}'

        try:
            response = self.llm_client.complete(
                prompt=prompt,
                max_tokens=1024,
            )
        except Exception as e:
            print(f"LLM error: {e}")
            raise

        # Response parsing
        result = self._parse_json_response(response.text)

        return result

    def answer_by_llm_CoT(
            self,
            question: str,
    ):
        """
        Use GPT to evaluate retrieval results as a specific role, return a strict JSON result.
        """

        # append CoT examples
        prompt = llm_CoT_prompt
        prompt += llm_CoT_examples

        # append query
        prompt += f"\nQ: {question}\n"
        prompt += "Please provide your response in the following JSON format:\n"
        # CoT prompt
        prompt += '{\n  "reasoning": "Your step-by-step reasoning process",\n  "answer": "Your final answer"\n}'

        try:
            response = self.llm_client.complete(
                prompt=prompt,
                max_tokens=1024,
            )
        except Exception as e:
            print(f"LLM error: {e}")
            raise

        # Response parsing
        result = self._parse_json_response(response.text)

        return result

    def run_on_dataset(self, dataset_name: str, sample_size: int, llm_only: bool=True):
        from data_loader import DataLoader
        qa, context = DataLoader(dataset_name=dataset_name, sample_size=sample_size).load()
        result_list = []
        for item in tqdm(qa):
            question = item['question']
            try:
                if llm_only:
                    llm_answer = self.answer_by_llm(question)
                else:
                    llm_answer = self.answer_by_llm_CoT(question)
            except Exception as e:
                llm_answer = {"reasoning": "Exception !!!", "answer": str(e)}
                print("Exception question: ", question)
            result = {**item, "llm_answer": llm_answer['answer']}
            result_list.append(result)

        file_path = self.result_dir + dataset_name + "LLM.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(result_list, file, ensure_ascii=False, indent=4)
            print(f"data has been successfully saved to: {file_path}")
        except Exception as e:
            print(f"file save error: {e}")


if __name__ == "__main__":
    pass
