import numpy as np, pandas as pd

def load_processed(csv_path="data/jigsaw_cost.csv"):
    df = pd.read_csv(csv_path)
    df.rename(columns={'weight':'abs_delta'}, inplace=True)
    df = df.dropna(subset=['comment_text']).copy()
    # signed Δ helper
    if 'delta_signed' not in df.columns:
        df['delta_signed'] = df['abs_delta'] * np.where(df['y_star']==1, 1, -1)
    return df

