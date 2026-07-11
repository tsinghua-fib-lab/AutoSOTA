import pandas as pd
import json
import os
from config import MODELS_LIST

# Define the base directories, models, and data types
BASE_DIRS = ['results/', 'results_prompt_intervention/', 'results_prompt_intervention_criticism/']
MODELS_LIST = MODELS_LIST
DATA_LIST =  ["gpqa", "hle"] #['mmlu', 'umwp', 'mc', 'mip'] #
SAVE_DIR = 'cross' if DATA_LIST == ['gpqa', 'hle'] else 'math'

# Loop through each base directory
for base_dir in BASE_DIRS:
    print(f"Processing directory: {base_dir}")
    all_results = []

    # Loop through each model
    for model_path in MODELS_LIST:
        # Loop through each data type
        for data_name in DATA_LIST:
            file_name = f"{data_name}_metrics.jsonl"
            # Construct the full file path
            file_path = os.path.join(base_dir, model_path, file_name)

            # Initialize totals and counters
            total_thinking_lengths_ill_posed = 0
            total_correct_well_posed = 0
            total_keyword_abstention_ill_posed = 0
            total_llm_abstention_ill_posed = 0
            number_of_records = 0
            
            avg_thinking_lengths_ill_posed = float('nan')
            avg_correct_well_posed = float('nan')
            avg_keyword_abstention_ill_posed = float('nan')
            avg_llm_abstention_ill_posed = float('nan')

            try:
                # Open and read the jsonl file line by line
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip(): # Ensure line is not empty
                            try:
                                data = json.loads(line)
                                
                                # Accumulate values
                                total_thinking_lengths_ill_posed += data.get("thinking_lengths_ill_posed", 0)
                                total_correct_well_posed += 1 if data.get("correct_well_posed", False) else 0
                                total_keyword_abstention_ill_posed += 1 if data.get("keyword_abstention_ill_posed", False) else 0
                                total_llm_abstention_ill_posed += 1 if data.get("llm_abstention_ill_posed", False) else 0
                                
                                number_of_records += 1
                            except json.JSONDecodeError:
                                print(f"Warning: Skipping malformed JSON line in {file_path}: {line.strip()}")

                # Calculate averages if records were found
                if number_of_records > 0:
                    avg_thinking_lengths_ill_posed = total_thinking_lengths_ill_posed / number_of_records
                    avg_correct_well_posed = total_correct_well_posed / number_of_records
                    avg_keyword_abstention_ill_posed = total_keyword_abstention_ill_posed / number_of_records
                    avg_llm_abstention_ill_posed = total_llm_abstention_ill_posed / number_of_records

            except FileNotFoundError:
                print(f"Warning: File not found, skipping: {file_path}")
            except ZeroDivisionError:
                 print(f"Warning: ZeroDivisionError (empty file?): {file_path}")
            except Exception as e:
                print(f"An unexpected error occurred with {file_path}: {e}")

            # Store the results for this combination
            all_results.append({
                "model_name": model_path,
                "dataset": data_name,
                "avg_thinking_lengths_ill_posed": avg_thinking_lengths_ill_posed,
                "avg_correct_well_posed": avg_correct_well_posed,
                "avg_llm_abstention_ill_posed": avg_llm_abstention_ill_posed
            })

    # Create a DataFrame from the results
    df = pd.DataFrame(all_results)
    
    # Generate a CSV filename for the base directory
    # Cleans up the directory name to make a valid filename
    csv_filename = base_dir.replace('/', '_').strip('_') + "_summary.csv"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    csv_filename = os.path.join(SAVE_DIR, csv_filename)
    
    # Save the DataFrame to CSV
    df.to_csv(csv_filename, index=False)
    print(f"Successfully created CSV file: {csv_filename}")

print("All processing complete.")