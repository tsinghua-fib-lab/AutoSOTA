"""
Automated LLM Expert Evaluation System (DeepSeek Version)
=========================================================
Simulates expert chemists using DeepSeek's API (OpenAI-compatible) to evaluate 
solubility predictions and explanations in a multi-turn pipeline.

Requirements:
pip install openai pandas tqdm
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for LLM API access"""
    # DeepSeek API Key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    
    # Model selection: 'deepseek-chat' (V3) or 'deepseek-reasoner' (R1)
    # Using 'deepseek-chat' as default for standard chat interactions
    DEEPSEEK_MODEL = "deepseek-reasoner"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    # Input Data
    INPUT_FILE = "file - explanations_summary.csv"

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT = """# SYSTEM PROMPT FOR LLM-BASED EXPERT EVALUATION OF SOLUBILITY PREDICTIONS

You are a computational chemistry expert with deep knowledge of molecular solubility, thermodynamics, and structure-property relationships. You will evaluate solubility predictions and their explanations from the solubility prediction model.

## YOUR EXPERTISE:
- PhD-level understanding of physical chemistry, thermodynamics, and molecular interactions
- Extensive experience with solubility prediction in drug discovery
- Familiarity with QSPR models, molecular descriptors, and computational methods
- Critical evaluation skills for assessing chemical reasoning and plausibility

## EVALUATION TASK:

You will be presented with solute-solvent pairs and asked to evaluate:
1. LogS Solubility Value given Solute and Solvent SMILES
1. Initial prediction plausibility (before reading explanation)
2. Explanation quality (chemical coherence and grounding)
3. Final agreement (after reading explanation)

## CRITICAL INSTRUCTIONS:

### 1. EVALUATE LIKE A REAL EXPERT:
- Base judgments on chemical principles (polarity, H-bonding, entropy, lattice energy)
- Consider molecular structure, functional groups, and thermodynamic factors
- Be appropriately skeptical—don't accept claims without mechanistic reasoning
- Recognize when uncertainty is high (complex systems, competing effects)

### 2. RATING SCALE (1-5):
- **5 = Strongly Agree**: Prediction/explanation is chemically sound and well-justified
- **4 = Agree**: Generally reasonable, minor concerns or uncertainties
- **3 = Neutral**: Plausible but significant ambiguity or missing information
- **2 = Disagree**: Major chemical concerns, weak justification
- **1 = Strongly Disagree**: Contradicts fundamental principles or clearly wrong

### 3. WHAT TO LOOK FOR IN PREDICTIONS:

**For Initial Agreement (Q1/Q2):**
- Does the predicted solubility class (highly soluble, moderately soluble, etc.) match your chemical intuition?
- Consider: "Like dissolves like," polar/nonpolar matching, H-bond donor/acceptor compatibility
- Factor in molecular weight, complexity, and temperature effects

**For Explanation Quality (Q3):**
- **Chemical accuracy**: Are the cited features (TPSA, MolLogP, H-bonding) correctly interpreted?
- **Mechanistic grounding**: Does it explain WHY these features matter (not just THAT they matter)?
- **Consistency**: Do the features support the predicted direction?
- **Red flags**: Contradictions, missing key factors, generic statements without specifics

**For Post-Explanation Agreement (Q4):**
- Did the explanation provide new insight or resolve uncertainty?
- Did it expose flaws in the prediction?
- Are you more or less confident after reading it?

### 4. SPECIFIC GUIDANCE BY SOLUBILITY CLASS:
[Standard solubility class definitions implied]

### 5. COMMON PITFALLS TO AVOID:
❌ **Don't just agree with everything** - real experts are critical
❌ **Don't ignore temperature** - higher T generally increases solubility (van't Hoff)
❌ **Don't accept generic explanations** - "polar molecule" is not enough; WHY is it polar?
❌ **Don't change your rating unless explanation provides real insight** - if it just restates features, Q4 = Q2

### 7. RESPONSE FORMAT:

For each solute-solvent pair:
```
Question 1: Predict the solubility (select from: Very Highly Soluble, Highly Soluble, Moderately Soluble, Poorly Soluble, Highly Insoluble)

[Your prediction with 1-2 sentence justification]

Question 2: How much do you agree with the model's prediction of [X] at [T]K?
Rating: [1-5]

[Brief justification: What chemical features support or contradict this?]

Question 3: How much do you agree with the explanation?
Rating: [1-5]

[Evaluate: Is it chemically accurate? Does it provide mechanistic insight? Any red flags?]

Question 4: After reading the explanation, how much do you agree with the prediction?
Rating: [1-5]

[Did the explanation change your view? If yes, why? If no, why not?]
```

## EXAMPLE EVALUATION:
[Standard example omitted for brevity but implied]

## FINAL REMINDERS:
1. **Be honest and critical**
2. **Use chemical reasoning**
3. **Allow updates**
4. **Vary responses**
5. **Maintain consistency**

Your goal is to provide authentic expert-level evaluation that reflects real chemical reasoning, not to validate or invalidate the model systematically.

## CRITICAL: RESPONSE FORMAT
You must response with ONLY valid JSON.
"""

# ============================================================================
# LLM CLIENTS
# ============================================================================

class DeepSeekClient:
    """DeepSeek API client (via OpenAI SDK) with Manual Chat History"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.history = []
        except ImportError:
            raise ImportError("Please install: pip install openai")
    
    def start_chat(self):
        # Reset history
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        return self

    def send_message(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        try:
            # DeepSeek supports JSON mode ensuring valid structure
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.5,
                response_format={"type": "json_object"} 
            )
            response_text = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": response_text})
            return response_text
        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")

# ============================================================================
# SURVEY PIPELINE
# ============================================================================

class SurveyPipeline:
    
    def __init__(self, client, client_type="deepseek"):
        self.client = client
        self.client_type = client_type
        self.api_calls = 0

    def _extract_json(self, response_text: str) -> Dict:
        """Extract JSON from response"""
        text = response_text.strip()
        # Remove code fences
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback regex
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON: {response_text}")

    def run_mixture_evaluation(self, mixture_data: pd.Series) -> Dict:
        """Runs the 3-step pipeline for a single mixture"""
        
        results = {}
        chat = self.client.start_chat()
        
        # --- Step 1: Blind Prediction ---
        prompt_q1 = f"""
Step 1: Evaluation
Please analyze the solubility of the following solute in the given solvent.

Solute SMILES: {mixture_data['Solute']}
Solvent SMILES: {mixture_data['Solvent']}
Temperature: {mixture_data['Temperature']} K

Based on the structure and conditions, predict the solubility class.
Output JSON ONLY:
{{
    "Q1_prediction": "Your predicted Class (Very Highly Soluble, Highly Soluble, Moderately Soluble, Poorly Soluble, Highly Insoluble)",
    "Q1_reasoning": "Brief chemical justification"
}}
"""
        resp_q1 = self._send(chat, prompt_q1)
        results.update(self._extract_json(resp_q1))
        
        # --- Step 2: Agreement with Model Prediction ---
        model_pred_logS = mixture_data['Predicted_LogS']
        # Infer class from logS for context if needed
        model_class = "Unknown"
        if model_pred_logS >= 0: model_class = "Very Highly Soluble"
        elif model_pred_logS >= -1: model_class = "Highly Soluble"
        elif model_pred_logS >= -2.5: model_class = "Moderately Soluble"
        elif model_pred_logS >= -4: model_class = "Poorly Soluble"
        else: model_class = "Highly Insoluble"

        prompt_q2 = f"""
Step 2: Compare with Model Prediction
The computational model predicted:
LogS: {model_pred_logS}
Class: {model_class}

How much do you agree with this prediction?
Output JSON ONLY:
{{
    "Q2_rating": <Integer 1-5>,
    "Q2_reasoning": "Why you agree or disagree"
}}
"""
        resp_q2 = self._send(chat, prompt_q2)
        results.update(self._extract_json(resp_q2))

        # --- Step 3: Agreement with Explanation ---
        explanation = mixture_data['Explanation']
        prompt_q3 = f"""
Step 3: Evaluate Explanation
Here is the model's explanation for its prediction:
"{explanation}"

A. Rate how much you agree with this explanation's logic.
B. Given this explanation (assuming it's true), rate how much you agree with the prediction now.

Output JSON ONLY:
{{
    "Q3_explanation_rating": <Integer 1-5>,
    "Q3_prediction_agreement_given_explanation": <Integer 1-5>,
    "Q3_reasoning": "Critique of the explanation"
}}
"""
        resp_q3 = self._send(chat, prompt_q3)
        results.update(self._extract_json(resp_q3))
        
        return results

    def _send(self, chat, prompt):
        self.api_calls += 1
        return self.client.send_message(prompt)

# ============================================================================
# EXECUTION
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} mixtures from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def main():
    # 1. Load and Randomize Data
    df = load_data(Config.INPUT_FILE)
    if df.empty:
        return
        
    # Randomize
    df = df.sample(frac=1).reset_index(drop=True)
    print("Data randomized.")

    if Config.DEEPSEEK_API_KEY and Config.DEEPSEEK_API_KEY != "your-deepseek-key-here":
        client = DeepSeekClient(Config.DEEPSEEK_API_KEY, Config.DEEPSEEK_MODEL, Config.DEEPSEEK_BASE_URL)
        client_type = "deepseek"
        print(f"Using DeepSeek Client ({Config.DEEPSEEK_MODEL}).")
    else:
        print("No valid API Key found for DeepSeek. Please set DEEPSEEK_API_KEY.")
        return

    pipeline = SurveyPipeline(client, client_type)
    
    all_results = []
    output_file = "llm_survey_pipeline_results_deepseek.json"
    
    print(f"Starting survey on {len(df)} mixtures...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = pipeline.run_mixture_evaluation(row)
            
            # Combine Input + Output
            full_record = {
                "mixture_id": idx,
                "solute": row['Solute'],
                "solvent": row['Solvent'],
                **result
            }
            all_results.append(full_record)
            
            # Incremental Save
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
                
            time.sleep(1) # Rate limit safety
            
        except Exception as e:
            print(f"Error processing mixture {idx}: {e}")
            continue

    print(f"\nSurvey Complete. Results saved to {output_file}")
    print(f"Total API Calls: {pipeline.api_calls}")

if __name__ == "__main__":
    main()
