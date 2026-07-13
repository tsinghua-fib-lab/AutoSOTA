import json
import pandas as pd
import re

def parse_actual_solubility(filepath):
    """
    Parses the actual_solubility.txt file.
    Returns a dict mapping Sample ID (int) -> LogS (float).
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Regex to find Sample ID and Solubility blocks
    # Pattern looks for:
    # --------------------------------------------------
    # Sample ID: 2
    # Solubility (LogS): -5.545129023869357
    # Source: Test
    
    pattern = r"Sample ID:\s*(\d+)\s*\nSolubility \(LogS\):\s*([\d\.-]+)"
    matches = re.findall(pattern, content)
    
    actual_map = {}
    for sample_id, logs in matches:
        actual_map[int(sample_id)] = float(logs)
        
    return actual_map

def main():
    survey_results_path = "llm_survey_pipeline_results_claude.json"
    csv_path = "file - explanations_summary.csv"
    actual_solubility_path = "actual_solubility.txt"
    
    # 1. Load Data
    try:
        with open(survey_results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {survey_results_path} not found.")
        return

    try:
        csv_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    actual_map = parse_actual_solubility(actual_solubility_path)
    
    print(f"Loaded {len(results)} survey results.")
    print(f"Loaded {len(csv_df)} CSV rows.")
    print(f"Loaded {len(actual_map)} actual solubility records.")
    
    updated_count = 0
    
    # 2. Iterate through results and match
    for record in results:
        solute = record.get('solute')
        solvent = record.get('solvent')
        
        # Find matching row in CSV
        # We need to be careful about float precision if matching logic involved floats, 
        # but Solute/Solvent strings should be robust enough.
        
        match = csv_df[(csv_df['Solute'] == solute) & (csv_df['Solvent'] == solvent)]
        
        if len(match) == 0:
            print(f"Warning: No match found for Solute={solute[:10]}... Solvent={solvent}")
            continue
        elif len(match) > 1:
            print(f"Warning: Multiple matches found for Solute={solute[:10]}... Solvent={solvent}. Taking first.")
        
        # Original Row Index (Sample ID)
        original_id = match.index[0]
        
        # Get Predicted LogS from CSV
        predicted_logs = float(match.iloc[0]['Predicted_LogS'])
        
        # Get Actual LogS from map
        if original_id in actual_map:
            actual_logs = actual_map[original_id]
        else:
            print(f"Warning: Sample ID {original_id} not found in actual_solubility.txt")
            actual_logs = None
            
        # Update record
        record['Original_Sample_ID'] = int(original_id)
        record['Predicted_LogS'] = predicted_logs
        record['Actual_LogS'] = actual_logs
        
        updated_count += 1

    # 3. Save updated results
    with open(survey_results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSuccessfully updated {updated_count} records in {survey_results_path}")

if __name__ == "__main__":
    main()
