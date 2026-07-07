import os
import re
from openai import OpenAI


class LLMHandler:
    def __init__(self, model_type="deepseek", api_key=None, base_url=None):
        self.model_type = model_type

        if model_type == "deepseek":
            self.client = OpenAI(
                api_key=api_key or os.environ.get('DEEPSEEK_API_KEY'),
                base_url=base_url or "https://api.deepseek.com"
            )
            self.model_name = "deepseek-chat"
        # elif model_type == "gemini": ...

    def ask(self, prompt, system_prompt="You are a helpful assistant", temperature=0.7):
        if self.model_type == "deepseek":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content
        return None

    def extract_smiles_and_rationale(self, response_text):
        rationale = "No rationale provided."
        new_smiles = None

        # Match Rationale
        rat_match = re.search(r"Rationale:\s*(.*?)(?=New SMILES:|$)", response_text, re.S | re.I)
        if rat_match:
            rationale = rat_match.group(1).strip()

        # Match New SMILES
        smiles_match = re.search(r"New SMILES:\s*([^\s\n]+)", response_text, re.I)
        if smiles_match:
            new_smiles = smiles_match.group(1).strip()

        return rationale, new_smiles