import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import category_encoders as ce
from folktables import ACSDataSource


def load_folk(device='cpu'):
    states_subset = ["CA", "TX", "FL", "NY", "PA"]
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

    all_X = []
    all_y = []
    all_groups = []
    all_state_labels = []

    print("--- Starting Sequential Data Collection ---")
    for state in states_subset:
        print(f"Processing {state}...")
        raw_df = data_source.get_data(states=[state], download=True)

        filtered_df = raw_df[(raw_df['AGEP'] >= 16) & (raw_df['WKHP'] >= 1)].copy()

        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

        all_X.append(filtered_df[features])
        all_y.append(filtered_df['PINCP'].values)
        all_groups.append(filtered_df['RAC1P'].values)
        all_state_labels.extend([state] * len(filtered_df))

        del raw_df
        del filtered_df

    X_final = pd.concat(all_X, ignore_index=True)
    y_final = np.arcsinh(np.concatenate(all_y))
    group_final = np.concatenate(all_groups)
    state_series = pd.Series(all_state_labels)

    num_cols = ['AGEP', 'WKHP']
    scaler = StandardScaler()
    X_final[num_cols] = scaler.fit_transform(X_final[num_cols])

    cat_cols = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
    encoder = ce.BinaryEncoder(cols=cat_cols)
    X_encoded = encoder.fit_transform(X_final)

    train_indices = state_series[state_series.isin(["CA", "TX", "FL", "PA"])].index.tolist()
    ny_indices = state_series[state_series == "NY"].index.tolist()

    ny_train_idx, ny_test_idx = train_test_split(ny_indices, train_size=0.10, random_state=42)

    final_train_idx = train_indices + ny_train_idx
    final_test_idx = ny_test_idx

    X_tensor = torch.tensor(X_encoded.values, dtype=torch.float32).to(device)
    y_tensor = torch.from_numpy(y_final.astype(np.float32)).unsqueeze(1).to(device)
    g_tensor = torch.from_numpy(group_final.astype(np.int64)).to(device)

    train_data = {
        'X': X_tensor[final_train_idx],
        'y': y_tensor[final_train_idx],
        'g': g_tensor[final_train_idx]
    }
    test_data = {
        'X': X_tensor[final_test_idx],
        'y': y_tensor[final_test_idx],
        'g': g_tensor[final_test_idx]
    }

    torch.save(train_data, 'folk_train.pt')
    torch.save(test_data, 'folk_test.pt')

    print(f"Saved: Train size {train_data['X'].shape[0]}, Test size {test_data['X'].shape[0]}")
    return train_data, test_data


def train_linreg(X, y, filename='linreg_weights2.pt'):
    model = LinearRegression(fit_intercept=True).fit(X, y)
    state_dict = {
        'weights': torch.tensor(model.coef_, dtype=torch.float32),
        'bias': torch.tensor(model.intercept_, dtype=torch.float32)
    }
    torch.save(state_dict, filename)
    return state_dict


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int, with_alpha: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        state_dict = torch.load('linreg_weights2.pt')
        with torch.no_grad():
            self.linear.weight.copy_(state_dict['weights'])
            self.linear.bias.copy_(state_dict['bias'])

        if with_alpha:
            self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
