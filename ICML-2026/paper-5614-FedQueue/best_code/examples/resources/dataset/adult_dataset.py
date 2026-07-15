import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from appfl.misc.data import (
    Dataset,
    iid_partition,
    dirichlet_noniid_partition,
    class_noniid_partition,
)


def introduce_incompleteness(data, incompleteness_level):
    """Introduce incompleteness to the dataset."""
    incomplete_data = data.copy()
    n_samples, n_features = data.shape

    # Calculate the number of values to replace with NaN
    n_missing = int(incompleteness_level * n_samples * n_features)

    # Randomly select indices to set as NaN
    missing_indices = np.random.choice(n_samples * n_features, n_missing, replace=False)

    for index in missing_indices:
        sample_index = index // n_features
        feature_index = index % n_features
        incomplete_data[sample_index, feature_index] = np.nan

    return incomplete_data


def impute_data(data, method="mean"):
    """Impute missing values using the specified method."""
    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"Invalid imputation method: {method}")

    return imputer.fit_transform(data)


def get_adult(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    test_size: float = 0.2,
    **kwargs,
):
    # Get the download directory for dataset
    dir = os.path.join(os.getcwd(), "datasets", "RawData")
    os.makedirs(dir, exist_ok=True)

    # Download the data if not already available
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset_path = os.path.join(dir, "adult.csv")
    if not os.path.exists(dataset_path):
        pd.read_csv(url, header=None).to_csv(dataset_path, index=False)

    # Load and preprocess the data
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data = pd.read_csv(dataset_path, names=column_names, skipinitialspace=True)

    # Clean the income column
    data["income"] = (
        data["income"].str.strip().replace({"<=50K.": "<=50K", ">50K.": ">50K"})
    )

    # Remove any rows with unexpected income values
    valid_incomes = ["<=50K", ">50K"]
    data = data[data["income"].isin(valid_incomes)]

    # Encode categorical variables
    le = LabelEncoder()
    for column in data.select_dtypes(include=["object"]).columns:
        if column != "income":
            data[column] = le.fit_transform(data[column])

    # Handle income classes
    income_le = LabelEncoder()
    data["income"] = income_le.fit_transform(data["income"])

    # Split features and target
    X = data.drop("income", axis=1).values
    y = data["income"].values

    # Introduce incompleteness if required
    incompleteness_level = kwargs.get("incompleteness_level", 0.0)
    X_incomplete = introduce_incompleteness(X.astype(float), incompleteness_level)

    # Impute missing data
    imputation_method = kwargs.get("imputation_method", "mean")
    X_imputed = impute_data(X_incomplete, method=imputation_method)

    # Balance the classes manually if balance_classes is True
    if kwargs.get("balance_classes"):
        data_imputed = pd.DataFrame(
            X_imputed, columns=data.drop("income", axis=1).columns
        )
        data_imputed["income"] = y

        class_0 = data_imputed[data_imputed["income"] == 0]
        class_1 = data_imputed[data_imputed["income"] == 1]

        if len(class_0) > len(class_1):
            class_1_oversampled = class_1.sample(
                len(class_0), replace=True, random_state=42
            )
            balanced_data = pd.concat([class_0, class_1_oversampled], axis=0)
        else:
            class_0_oversampled = class_0.sample(
                len(class_1), replace=True, random_state=42
            )
            balanced_data = pd.concat([class_1, class_0_oversampled], axis=0)

        # Shuffle the dataset
        balanced_data = balanced_data.sample(frac=1, random_state=42)

        # Use balanced dataset
        X_imputed = balanced_data.drop("income", axis=1).values
        y = balanced_data["income"].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create Dataset objects
    train_data_raw = Dataset(X_train_tensor, y_train_tensor)
    test_dataset = Dataset(X_test_tensor, y_test_tensor)

    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_noniid":
        train_datasets = dirichlet_noniid_partition(
            train_data_raw, num_clients, **kwargs
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    return train_datasets[client_id], test_dataset
