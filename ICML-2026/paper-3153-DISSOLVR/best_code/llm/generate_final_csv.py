"""
Generate Final Combined CSV
===========================
Combines:
1. file - explanations_summary.csv (Base Data)
2. actual_solubility.txt (Ground Truth)
3. combined.json (ChatGPT Results)
4. llm_survey_pipeline_results.json (Gemini Results)
5. llm_survey_pipeline_results_claude.json (Claude Results)
6. llm_survey_pipeline_results_deepseek.json (DeepSeek Results)

Produces:
Solubility Research Form (Responses) - Form responses 1 (2).csv
"""

import pandas as pd
import json
import re
import os

def parse_actual_solubility(filepath):
    """Parses actual_solubility.txt -> {id: logs}"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    
    pattern = r"Sample ID:\s*(\d+)\s*\nSolubility \(LogS\):\s*([\d\.-]+)"
    matches = re.findall(pattern, content)
    return {int(sid): float(logs) for sid, logs in matches}

def load_json_results(filepath):
    """Loads a JSON result file and returns a dict mapped by Original_Sample_ID (or inferred)"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
        
    mapped_data = {}
    
    # Handle combined.json (ChatGPT) which has a 'mixtures' key
    if isinstance(data, dict) and "mixtures" in data:
        for item in data["mixtures"]:
            # Mixture number 0 corresponds to CSV index 0
            idx = item.get("mixture_number")
            mapped_data[idx] = item
    
    # Handle list-based results (Gemini/Claude/DeepSeek)
    elif isinstance(data, list):
        for item in data:
            # Prefer Original_Sample_ID if available
            if "Original_Sample_ID" in item:
                idx = item["Original_Sample_ID"]
                mapped_data[idx] = item
            elif "mixture_id" in item:
                # Fallback to mixture_id if update_results.py wasn't run.
                # The index might not match, but we need the record in the dict 
                # so the solute/solvent lookup fallback in main() can find it.
                idx = item["mixture_id"]
                mapped_data[f"temp_{idx}"] = item
                
    return mapped_data

def main():
    csv_file = "file - explanations_summary.csv"
    actual_file = "actual_solubility.txt"
    chatgpt_file = "combined.json"
    gemini_file = "llm_survey_pipeline_results.json"
    claude_file = "llm_survey_pipeline_results_claude.json"
    deepseek_file = "llm_survey_pipeline_results_deepseek.json"
    
    output_csv = "Solubility Research Form (Responses) - Form responses 1 (2).csv"

    # 1. Load Base Data
    try:
        base_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Base CSV not found.")
        return

    # 2. Load Ground Truth
    actual_map = parse_actual_solubility(actual_file)

    # 3. Load LLM Results
    gpt_data = load_json_results(chatgpt_file)
    gemini_data = load_json_results(gemini_file)
    claude_data = load_json_results(claude_file)
    deepseek_data = load_json_results(deepseek_file)

    print(f"Base Rows: {len(base_df)}")
    print(f"ChatGPT Records: {len(gpt_data)}")
    print(f"Gemini Records: {len(gemini_data)}")
    print(f"Claude Records: {len(claude_data)}")
    print(f"DeepSeek Records: {len(deepseek_data)}")

    # 4. Construct Final DataFrame
    final_rows = []

    for idx, row in base_df.iterrows():
        # -- Basic Info --
        solute = row['Solute']
        solvent = row['Solvent']
        temp = row['Temperature']
        pred_logs = row['Predicted_LogS']
        explanation = row['Explanation']
        
        # -- Actual LogS & Error --
        actual_logs = actual_map.get(idx, None)
        
        error = None
        quality = "Neutral"
        
        if actual_logs is not None:
            # Ensure pred_logs is float
            try:
                p_val = float(pred_logs)
                error = abs(p_val - actual_logs)
                
                if error < 0.3:
                    quality = "Good"
                elif error > 1.0:
                    quality = "Bad"
            except:
                pass

        new_row = {
            "Mixture ID": idx,
            "Solute": solute,
            "Solvent": solvent,
            "Temperature": temp,
            "Predicted LogS": pred_logs,
            "Actual LogS": actual_logs if actual_logs is not None else "",
            "Absolute Error": error if error is not None else "",
            "Mixture Quality": quality,
            "Explanation": explanation
        }

        # -- Helper to find Record by ID or Solute/Solvent --
        def find_record(mapping, current_id, solute_str, solvent_str):
            # 1. Try ID
            if current_id in mapping:
                return mapping[current_id]
            # 2. Try Fallback search by string components
            for k, v in mapping.items():
                if v.get('solute') == solute_str and v.get('solvent') == solvent_str:
                    return v
            return {}

        g_rec = find_record(gemini_data, idx, solute, solvent)
        c_rec = find_record(claude_data, idx, solute, solvent)
        ds_rec = find_record(deepseek_data, idx, solute, solvent)
        gpt_rec = gpt_data.get(idx, {}) # ChatGPT relies on mixture_number/index

        # -- Gemini Columns --
        new_row["Gemini Q1 Prediction"] = g_rec.get("Q1_prediction", "")
        new_row["Gemini Q1 Reasoning"] = g_rec.get("Q1_reasoning", "")
        new_row["Gemini Q2 Rating"] = g_rec.get("Q2_rating", "")
        new_row["Gemini Q2 Reasoning"] = g_rec.get("Q2_reasoning", "")
        new_row["Gemini Q3 Explanation Rating"] = g_rec.get("Q3_explanation_rating", "")
        new_row["Gemini Q3 Prediction Agreement"] = g_rec.get("Q3_prediction_agreement_given_explanation", "")
        new_row["Gemini Q3 Reasoning"] = g_rec.get("Q3_reasoning", "")

        # -- Claude Columns --
        new_row["Claude Q1 Prediction"] = c_rec.get("Q1_prediction", "")
        new_row["Claude Q1 Reasoning"] = c_rec.get("Q1_reasoning", "")
        new_row["Claude Q2 Rating"] = c_rec.get("Q2_rating", "")
        new_row["Claude Q2 Reasoning"] = c_rec.get("Q2_reasoning", "")
        new_row["Claude Q3 Explanation Rating"] = c_rec.get("Q3_explanation_rating", "")
        new_row["Claude Q3 Prediction Agreement"] = c_rec.get("Q3_prediction_agreement_given_explanation", "")
        new_row["Claude Q3 Reasoning"] = c_rec.get("Q3_reasoning", "")

        # -- DeepSeek Columns --
        new_row["DeepSeek Q1 Prediction"] = ds_rec.get("Q1_prediction", "")
        new_row["DeepSeek Q1 Reasoning"] = ds_rec.get("Q1_reasoning", "")
        new_row["DeepSeek Q2 Rating"] = ds_rec.get("Q2_rating", "")
        new_row["DeepSeek Q2 Reasoning"] = ds_rec.get("Q2_reasoning", "")
        new_row["DeepSeek Q3 Explanation Rating"] = ds_rec.get("Q3_explanation_rating", "")
        new_row["DeepSeek Q3 Prediction Agreement"] = ds_rec.get("Q3_prediction_agreement_given_explanation", "")
        new_row["DeepSeek Q3 Reasoning"] = ds_rec.get("Q3_reasoning", "")

        # -- ChatGPT Columns --
        q1_obj = gpt_rec.get("Q1", {})
        q2_obj = gpt_rec.get("Q2", {})
        q3_obj = gpt_rec.get("Q3", {})

        new_row["ChatGPT Q1 Prediction"] = q1_obj.get("Q1_prediction", "")
        new_row["ChatGPT Q1 Reasoning"] = q1_obj.get("Q1_reasoning", "")
        new_row["ChatGPT Q2 Rating"] = q2_obj.get("Q2_rating", "")
        new_row["ChatGPT Q2 Reasoning"] = q2_obj.get("Q2_reasoning", "")
        new_row["ChatGPT Q3 Explanation Rating"] = q3_obj.get("Q3_explanation_rating", "")
        new_row["ChatGPT Q3 Prediction Agreement"] = q3_obj.get("Q3_prediction_agreement_given_explanation", "")
        new_row["ChatGPT Q3 Reasoning"] = q3_obj.get("Q3_reasoning", "")
        
        final_rows.append(new_row)

    # 5. Create Final DataFrame and Save
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(output_csv, index=False)
    print(f"Successfully saved combined data to '{output_csv}'")

if __name__ == "__main__":
    main()
