from sklearn.linear_model import LogisticRegression
from scipy import stats


def fit_models(df, predictors):
    flt_str = {"S": "index==index", "A": "S==1", "Y0": "S==1 & A==0", "Y1": "S==1 & A==1"}
    models = {}

    for key, value in flt_str.items():
        model = LogisticRegression(max_iter=5000, tol=1e-6)
        model.fit(df.query(value)[predictors], df.query(value)[key])
        models[key] = model

    return models


def pearsonr(df, col_x, col_y, n, random_state=None):
    if col_y == "SE_R":
        df_new = df.query("index==index")[[col_x, col_y]].dropna()
    elif col_y == "SE_S":
        df_new = df.query("R==0")[[col_x, col_y]].dropna()
    elif col_y == "SE_A":
        df_new = df.query("R==0 & S==1")[[col_x, col_y]].dropna()
    elif col_y == "SE_Y0":
        df_new = df.query("R==0 & S==1 & A==0")[[col_x, col_y]].dropna()
    elif col_y == "SE_Y1":
        df_new = df.query("R==0 & S==1 & A==1")[[col_x, col_y]].dropna()
    else:
        raise Exception(f"Make sure {col_y} is in ['SE_R', 'SE_S', 'SE_A', 'SE_Y0', 'SE_Y1']")

    df_new = df_new.sample(n=min(n, len(df_new)), random_state=random_state)
    r, p = stats.pearsonr(df_new[col_x], df_new[col_y])

    return r, p