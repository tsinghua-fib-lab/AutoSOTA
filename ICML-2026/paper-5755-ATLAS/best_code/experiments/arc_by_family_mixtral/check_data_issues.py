import pandas as pd
import os

def check_constant(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        
        # Identify constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        print(f"Constant columns ({len(constant_cols)}): {constant_cols}")
        
        # Also check for mostly NA columns or other weirdness
        na_cols = [col for col in df.columns if df[col].isna().all()]
        print(f"All-NA columns: {na_cols}")

    except Exception as e:
        print(f"Error reading file: {e}")

check_constant("data/arc_response_matrix_other.csv")
check_constant("data/arc_response_matrix_mixtral.csv")
