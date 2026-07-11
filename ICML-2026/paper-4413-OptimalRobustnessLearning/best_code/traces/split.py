import pandas as pd
import numpy as np
import os

dataset = 'citi'

file_path = os.path.join(dataset, f'{dataset}_all.csv')
df = pd.read_csv(file_path, header=None)

total_rows = len(df)
train_size = int(0.8 * total_rows)
val_size = int(0.1 * total_rows)
test_size = total_rows - train_size - val_size

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size+val_size]
test_data = df.iloc[train_size+val_size:]

train_data.to_csv(os.path.join(dataset, f'{dataset}_train.csv'), index=False, header=None)
val_data.to_csv(os.path.join(dataset, f'{dataset}_valid.csv'), index=False, header=None)
test_data.to_csv(os.path.join(dataset, f'{dataset}_test.csv'), index=False, header=None)

print(f"Data split into train, validation, and test sets.")
