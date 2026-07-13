"""
Dataset loaders for the ECSEL experiments.

Each loader reads a raw classification dataset from ``wd`` (a working-directory
path ending in a separator), applies dataset-specific preprocessing (encoding
categorical columns, dropping missing rows, mapping the target to 0-indexed
integer classes), and returns a :class:`pandas.DataFrame` whose feature columns
are renamed ``X_0, X_1, ...`` and whose target column is named ``output``.

Encoding conventions
---------------------
- Nominal categorical features are one-hot encoded.
- Ordinal categorical features are mapped to integer codes.
- Targets are mapped to integers starting at 0.
- Rows with missing values are dropped.
"""

import pandas as pd


def iris(wd):
    """Load the Iris dataset (150 x 5).

    Source: https://archive.ics.uci.edu/ml/datasets/iris

    Features are the four continuous measurements (sepal length, sepal width,
    petal length, petal width). The three-class target (Setosa, Versicolour,
    Virginica) is encoded as integer codes.

    Parameters
    ----------
    wd : str
        Working-directory path containing ``iris.csv`` (semicolon-separated,
        with a header row).

    Returns
    -------
    pd.DataFrame
        Columns ``X_0`` .. ``X_3`` and ``output``.
    """
    df = pd.read_csv(wd + 'iris.csv', header=0, sep=';')
    print(f"True shape: {df.shape}")

    # Separate target column.
    target_col = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # Encode the target as integer class codes.
    target_encoded = target_col.astype('category').cat.codes

    # Combine features and target.
    df = pd.concat([features, target_encoded.rename('output')], axis=1)

    # Rename feature columns to the standard X_i format.
    feature_cols = df.columns[:-1]
    df = df.rename(columns={col: f'X_{i}' for i, col in enumerate(feature_cols)})

    print(f"Shape: {df.shape}")
    print("Target mapping:", dict(enumerate(target_col.astype('category').cat.categories)))

    return df


def seeds(wd):
    """Load the Seeds dataset (wheat-kernel geometry).

    Source: https://archive.ics.uci.edu/ml/datasets/seeds

    All features are continuous geometric measurements. The three-class target
    (Kama, Rosa, Canadian wheat varieties) is provided as 1-indexed labels and
    kept as integers here (1=Kama, 2=Rosa, 3=Canadian per the source).

    Parameters
    ----------
    wd : str
        Working-directory path containing ``seeds.csv`` (tab-separated, no
        header row).

    Returns
    -------
    pd.DataFrame
        Columns ``X_0`` .. ``X_{n-1}`` and ``output``.
    """
    df = pd.read_csv(wd + 'seeds.csv', header=None, sep='\t')
    print(f"True shape: {df.shape}")

    # Separate target column (convert floats to ints).
    target = df.iloc[:, -1].astype(int)
    features = df.iloc[:, :-1]

    print("Target mapping:")
    print("  0: Kama seed, 1: Rosa seed, 2: Canadian seed")

    # Combine features and target, then rename.
    df = pd.concat([features, target], axis=1)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['output']

    print(f"Processed shape: {df.shape}")
    print("Class distribution:", df['output'].value_counts().to_dict())
    return df


def ilpd(wd):
    """Load the Indian Liver Patient Dataset (ILPD, 583 x 11).

    Source: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset

    Features are age, gender, and several blood-chemistry measurements (total
    and direct bilirubin, alkaline phosphotase, the aminotransferases, total
    proteins, albumin, and the albumin/globulin ratio). Gender is mapped to an
    integer (1=Female, 0=Male). The binary target is remapped so that
    1=liver disease becomes 0 and 2=healthy becomes 1.

    Parameters
    ----------
    wd : str
        Working-directory path containing ``ILPD.csv`` (comma-separated, no
        header row).

    Returns
    -------
    pd.DataFrame
        Columns ``X_0`` .. ``X_9`` and ``output`` (0 = liver disease,
        1 = healthy).
    """
    df = pd.read_csv(wd + 'ILPD.csv', header=None, sep=',')
    print("True Shape: ", df.shape)
    df = df.dropna()  # Drop rows with missing values.

    # Separate target and features.
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # Map gender to an integer code (the only categorical feature).
    gender_col = features.columns[1]
    features[gender_col] = features[gender_col].map({'Female': 1, 'Male': 0})

    # Combine features and target, then rename.
    df = pd.concat([features, target], axis=1)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['output']

    # Encode the target as 0/1.
    df['output'] = df['output'].map({1: 0, 2: 1})
    print("Target mapping: 0: liver disease, 1: healthy")

    return df


def hearts(wd):
    """Load the Heart Disease dataset (14 attributes).

    Source: https://archive.ics.uci.edu/ml/datasets/heart+disease

    Uses the 13 standard predictors plus the binary target. The nominal
    features ``cp`` (chest-pain type) and ``thal`` are one-hot encoded, while
    ordered-severity features such as ``restecg`` and ``slope`` are left as
    ordinal integers. The target is binary: 0 = no significant heart disease,
    1 = significant heart disease.

    Parameters
    ----------
    wd : str
        Working-directory path containing ``hearts.csv`` (comma-separated, with
        a header row).

    Returns
    -------
    pd.DataFrame
        Columns ``X_0`` .. ``X_{n-1}`` and ``output``.
    """
    df = pd.read_csv(wd + 'hearts.csv', header=0, sep=',')
    print(f"True shape: {df.shape}")
    df = df.dropna()  # Drop rows with missing values.

    # Separate target and features.
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # One-hot encode nominal features only; ordered features stay ordinal.
    categorical_cols = ['cp', 'thal']

    print("\nOne-Hot Encoding Mapping:")
    print("=" * 60)
    for col in categorical_cols:
        if col in features.columns:
            unique_vals = sorted(features[col].unique())
            print(f"\n{col}: {unique_vals}")
            if col == 'cp':
                print("  1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic")
            elif col == 'thal':
                print("  3=normal, 6=fixed defect, 7=reversible defect")

    print("\nKeeping as ordinal (not one-hot encoded):")
    print("  restecg: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy")
    print("  slope: 1=upsloping, 2=flat, 3=downsloping")

    features = pd.get_dummies(features, columns=categorical_cols, drop_first=False)

    print("\nNew one-hot encoded columns:")
    print("=" * 60)
    for col in categorical_cols:
        encoded_cols = [c for c in features.columns if c.startswith(col + '_')]
        if encoded_cols:
            print(f"\n{col} -> {encoded_cols}")

    # Combine features and target, then rename.
    df = pd.concat([features, target], axis=1)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['output']

    print("Target mapping: 0: no significant disease, 1: sign. heart disease")

    print(f"\nFinal shape: {df.shape}")
    print("=" * 60)

    return df

def transfusion(wd):
    """Load the Blood Transfusion Service Center dataset.

    Source: https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center

    All features are integer-valued (recency, frequency, monetary, and time
    of donations), so no categorical encoding is needed. The binary target
    indicates whether the donor gave blood in a target window.

    Parameters
    ----------
    wd : str
        Working-directory path containing ``transfusion.csv`` (comma-separated,
        with a header row).

    Returns
    -------
    pd.DataFrame
        Columns ``X_0`` .. ``X_{n-1}`` and ``output`` (0 = not donating,
        1 = donating).
    """
    df = pd.read_csv(wd + 'transfusion.csv', header=0, sep=',')
    print("True shape: ", df.shape)
    df = df.dropna()  # Drop rows with missing values.

    # Separate target and features (no encoding needed; all integer).
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # Combine and rename to the standard X_i / output format.
    df = pd.concat([features, target], axis=1)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['output']

    print("Shape: ", df.shape)
    print("Target mapping: 0: not donating blood, 1: donating blood")
    return df

def loan(wd):
    """Load the loan default dataset.

    The raw data has 21 features (42 after one-hot encoding in the source file).
    The target column ``good_bad`` is moved to the end and renamed ``output``.

    Parameters
    ----------
    wd : str
        Working-directory path containing ``loan.csv`` (comma-separated, with a
        header row).

    Returns
    -------
    pd.DataFrame
        Columns ``X_0`` .. ``X_{n-1}`` and ``output`` (0 = good loan,
        1 = bad loan).
    """
    df = pd.read_csv(wd + 'loan.csv', header=0, sep=',')

    # Move the target to the end, then rename to the standard X_i / output format.
    target_col = 'good_bad'
    features = df.drop(columns=[target_col])
    target = df[target_col]
    df = pd.concat([features, target], axis=1)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['output']

    df = df.dropna()  # Drop rows with missing values.

    print("Shape: ", df.shape)
    print("Target mapping: 0 = good loan, 1 = bad loan")
    return df
def compas(wd):
    """Load the COMPAS recidivism dataset (3518 x 8 after preprocessing).

    Source: ProPublica COMPAS analysis
    https://github.com/propublica/compas-analysis

    Features: age, sex (Male=1), race (African-American=1), priors_count,
    c_charge_degree (F=1), juv_fel_count, juv_misd_count.
    Binary target: two_year_recid (0 = no recidivism, 1 = recidivism).

    Parameters
    ----------
    wd : str
        Working-directory path (unused; data is loaded from /datasets).

    Returns
    -------
    pd.DataFrame
        Columns X_0 .. X_6 and output.
    """
    import pandas as pd

    data_path = "/datasets/compas-scores-two-years.csv"
    df = pd.read_csv(data_path)
    print(f"True shape: {df.shape}")

    # Apply standard ML fairness preprocessing
    df = df[(df["days_b_screening_arrest"] >= -30) & (df["days_b_screening_arrest"] <= 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["race"].isin(["African-American", "Caucasian"])]
    df = df.dropna(subset=["c_offense_date"])

    # Select features
    features = df[["age", "sex", "race", "priors_count", "c_charge_degree",
                    "juv_fel_count", "juv_misd_count"]].copy()

    # Encode categorical features
    features["sex"] = (features["sex"] == "Male").astype(int)
    features["race"] = (features["race"] == "African-American").astype(int)
    features["c_charge_degree"] = (features["c_charge_degree"] == "F").astype(int)

    # Target
    target = df["two_year_recid"].values

    # Combine and rename
    result = pd.concat([features, pd.Series(target, name="output")], axis=1)
    result.columns = ["X_" + str(i) for i in range(len(result.columns) - 1)] + ["output"]
    result = result.reset_index(drop=True)

    print(f"Processed shape: {result.shape}")
    print("Target mapping: 0 = No recidivism, 1 = Recidivism")
    print("Class distribution:", result["output"].value_counts().to_dict())
    return result
