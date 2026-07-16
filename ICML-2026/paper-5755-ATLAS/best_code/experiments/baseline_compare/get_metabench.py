from datasets import load_dataset
import pandas as pd
import csv
import re

# DATASET_NAME = 'GSM8K'
# DATASET_NAME = 'ARC'
# DATASET_NAME = 'HellaSwag'
# DATASET_NAME = 'Winogrande'
DATASET_NAME = 'TruthfulQA'


ds = load_dataset("HCAI/metabench", DATASET_NAME)

# subject = 'anatomy'
# subject = 'astronomy'
# subject = 'high_school_macroeconomics'


# Create a list to store the matched item IDs
item_ids_primary = ds['primary']['metabench_idx']
item_ids_secondary = ds['secondary']['metabench_idx']

# Create DataFrames for the data
df_primary = pd.DataFrame({'item_id': item_ids_primary})
df_secondary = pd.DataFrame({'item_id': item_ids_secondary})

# Save the results to a CSV file
output_path_primary = f'metabench_{DATASET_NAME}_item_ids_primary.csv'
df_primary.to_csv(output_path_primary, index=False)

output_path_secondary = f'metabench_{DATASET_NAME}_item_ids_secondary.csv'
df_secondary.to_csv(output_path_secondary, index=False)

print(f"Saved {len(item_ids_primary)} questions with item IDs to {output_path_primary}")
print(f"Sample of the data:\n{df_primary.head()}")

print(f"Saved {len(item_ids_secondary)} questions with item IDs to {output_path_secondary}")
print(f"Sample of the data:\n{df_secondary.head()}")
