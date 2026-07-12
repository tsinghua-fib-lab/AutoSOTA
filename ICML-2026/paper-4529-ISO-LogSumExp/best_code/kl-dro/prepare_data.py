"""Download sklearn-compatible California Housing data via OpenML and save."""
import numpy as np
from sklearn.datasets import fetch_openml

print("Fetching California Housing from OpenML...")
housing = fetch_openml(name='california_housing', version=1, as_frame=False, parser='auto')

# OpenML columns: longitude, latitude, housing_median_age, total_rooms,
#   total_bedrooms, population, households, median_income, ocean_proximity
# Drop ocean_proximity (categorical) and convert remaining to float
numeric_cols = [0, 1, 2, 3, 4, 5, 6, 7]
raw_numeric = housing.data[:, numeric_cols].astype(float)
longitude = raw_numeric[:, 0]
latitude = raw_numeric[:, 1]
housing_median_age = raw_numeric[:, 2]
total_rooms = raw_numeric[:, 3]
total_bedrooms = raw_numeric[:, 4]
population = raw_numeric[:, 5]
households = raw_numeric[:, 6]
median_income = raw_numeric[:, 7]

target = housing.target.astype(float) / 100_000.0  # to $100k

# Handle missing total_bedrooms (207 missing values in OpenML)
# Impute with median, same as sklearn's fetch_california_housing
total_bedrooms_clean = total_bedrooms.copy()
nan_mask = np.isnan(total_bedrooms_clean)
total_bedrooms_clean[nan_mask] = np.nanmedian(total_bedrooms_clean)
print(f"Imputed {nan_mask.sum()} missing total_bedrooms with median {np.nanmedian(total_bedrooms_clean):.1f}")

# Derived features (same as sklearn's fetch_california_housing)
ave_rooms = total_rooms / households
ave_bedrms = total_bedrooms_clean / households
ave_occup = population / households

# sklearn feature order: MedInc, HouseAge, AveRooms, AveBedrms,
#   Population, AveOccup, Latitude, Longitude
sklearn_data = np.column_stack([
    median_income, housing_median_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude
])

print(f"Data shape: {sklearn_data.shape}")
print(f"Target shape: {target.shape}")
print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
print(f"Target mean: {target.mean():.4f}")

np.save('/datasets/california_housing_data.npy', sklearn_data)
np.save('/datasets/california_housing_target.npy', target)
print("Saved sklearn-compatible data to /datasets/")
