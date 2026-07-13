"""
Generate Ready-to-Use Manual Prompts
====================================
Reads the CSV and generates a markdown file with filled-in prompts for each mixture.
This allows the user to simply copy-paste blocks into ChatGPT.
"""

import pandas as pd
import os

def generate_prompts():
    input_file = "file - explanations_summary.csv"
    output_file = "ready_to_use_prompts.md"
    
    try:
        df = pd.read_csv(input_file)
        # Randomize to match the survey behavior, or keep deterministic?
        # User might want to match the "Sample ID" order if they are doing manual checks.
        # But survey.py randomizes. Let's keep it sequential for manual sanity, 
        # or we can verify if user wants random. Sequential is safer for manual tracking.
        # Use Sample ID if available or just index.
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    with open(output_file, "w") as f:
        f.write("# Ready-to-Use Prompts for ChatGPT Evaluation\n\n")
        f.write("**Instructions:**\n")
        f.write("1. Check `manual_prompts_for_chatgpt.md` for the SYSTEM PROMPT and paste that first.\n")
        f.write("2. For each mixture below, copy the prompts in order (Step 1 -> Step 2 -> Step 3).\n\n")
        f.write("---\n\n")

        for idx, row in df.iterrows():
            f.write(f"# Mixture {idx}\n\n")
            
            # --- Step 1 ---
            f.write(f"### Step 1: Blind Prediction\n")
            f.write("```text\n")
            f.write(f"Step 1: Evaluation\n")
            f.write(f"Please analyze the solubility of the following solute in the given solvent.\n\n")
            f.write(f"Solute SMILES: {row['Solute']}\n")
            f.write(f"Solvent SMILES: {row['Solvent']}\n")
            f.write(f"Temperature: {row['Temperature']} K\n\n")
            f.write(f"Based on the structure and conditions, predict the solubility class.\n")
            f.write(f"Output JSON ONLY:\n")
            f.write("{\n")
            f.write('    "Q1_prediction": "Your predicted Class (Very Highly Soluble, Highly Soluble, Moderately Soluble, Poorly Soluble, Highly Insoluble)",\n')
            f.write('    "Q1_reasoning": "Brief chemical justification"\n')
            f.write("}\n")
            f.write("```\n\n")

            # --- Step 2 ---
            model_pred_logS = row['Predicted_LogS']
            # Logic for class string
            if model_pred_logS >= 0: model_class = "Very Highly Soluble"
            elif model_pred_logS >= -1: model_class = "Highly Soluble"
            elif model_pred_logS >= -2.5: model_class = "Moderately Soluble"
            elif model_pred_logS >= -4: model_class = "Poorly Soluble"
            else: model_class = "Highly Insoluble"

            f.write(f"### Step 2: Model Agreement\n")
            f.write("```text\n")
            f.write(f"Step 2: Compare with Model Prediction\n")
            f.write(f"The computational model predicted:\n")
            f.write(f"LogS: {model_pred_logS}\n")
            f.write(f"Class: {model_class}\n\n")
            f.write(f"How much do you agree with this prediction?\n")
            f.write(f"Output JSON ONLY:\n")
            f.write("{\n")
            f.write('    "Q2_rating": <Integer 1-5>,\n')
            f.write('    "Q2_reasoning": "Why you agree or disagree"\n')
            f.write("}\n")
            f.write("```\n\n")

            # --- Step 3 ---
            f.write(f"### Step 3: Explanation Agreement\n")
            f.write("```text\n")
            f.write(f"Step 3: Evaluate Explanation\n")
            f.write(f"Here is the model's explanation for its prediction:\n")
            f.write(f'"{row["Explanation"]}"\n\n')
            f.write(f"A. Rate how much you agree with this explanation's logic.\n")
            f.write(f"B. Given this explanation (assuming it's true), rate how much you agree with the prediction now.\n\n")
            f.write(f"Output JSON ONLY:\n")
            f.write("{\n")
            f.write('    "Q3_explanation_rating": <Integer 1-5>,\n')
            f.write('    "Q3_prediction_agreement_given_explanation": <Integer 1-5>,\n')
            f.write('    "Q3_reasoning": "Critique of the explanation"\n')
            f.write("}\n")
            f.write("```\n\n")
            f.write("---\n\n")

    print(f"Generated {output_file} with {len(df)} mixtures.")

if __name__ == "__main__":
    generate_prompts()
