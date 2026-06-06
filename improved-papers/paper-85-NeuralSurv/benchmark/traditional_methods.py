import numpy as np

from sklearn.model_selection import GridSearchCV, ShuffleSplit

from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.ensemble import (
    RandomSurvivalForest,
)
from sksurv.svm import FastSurvivalSVM

from sksurv.metrics import concordance_index_censored


def fit_CoxPHFitter(data_train, data_val, data_test, new_times):

    print("\nFitting CoxPH ...")

    df_train = data_train["pd.DataFrame"]
    df_test = data_test["pd.DataFrame"]

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_train, duration_col="time", event_col="event")

    hz_test = cph.predict_partial_hazard(df_test)
    surv_test = cph.predict_survival_function(df_test, times=df_test["time"]).T
    surv_new_times = cph.predict_survival_function(df_test, times=new_times).T

    return hz_test.to_numpy(), surv_test.to_numpy(), surv_new_times.to_numpy()


def fit_WeibullAFTFitter(data_train, data_val, data_test, new_times):

    print("\nFitting Weibull AFTF ...")

    df_train = data_train["pd.DataFrame"]
    df_test = data_test["pd.DataFrame"]

    aft = WeibullAFTFitter()
    aft.fit(df_train, duration_col="time", event_col="event")

    hz_test = aft.predict_hazard(df_test, times=df_test["time"]).T
    surv_test = aft.predict_survival_function(df_test, times=df_test["time"]).T
    surv_new_times = aft.predict_survival_function(df_test, times=new_times).T

    return hz_test.to_numpy(), surv_test.to_numpy(), surv_new_times.to_numpy()


def fit_RandomSurvivalForest(
    data_train, data_val, data_test, new_times, random_state=4
):
    # Used random_state = 0, 1, 2, 3, 4

    print("\nFitting Random Survival Forest (RSF)...")

    df_train = data_train["pd.DataFrame"]
    df_test = data_test["pd.DataFrame"]

    X_train = df_train.drop(columns=["event", "time"])
    X_test = df_test.drop(columns=["event", "time"])

    y_train = list(zip(df_train["event"].astype(bool), df_train["time"]))
    y_train = np.array(y_train, dtype=[("cens", "bool"), ("time", "float")])

    rsf = RandomSurvivalForest(
        n_estimators=1000,
        min_samples_split=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=random_state,
    )
    rsf.fit(X_train, y_train)

    risk_test = rsf.predict(X_test)
    survs = rsf.predict_survival_function(X_test)

    max_train_time = df_train["time"].max()
    test_time_capped = df_test["time"].where(
        df_test["time"] <= max_train_time, max_train_time
    )  # max test time <= max test train
    new_times_capped = np.where(new_times <= max_train_time, new_times, max_train_time)

    surv_test = np.vstack([fn(test_time_capped) for fn in survs])
    surv_new_times = np.vstack([fn(new_times_capped) for fn in survs])

    return risk_test, surv_test, surv_new_times


def fit_FastSurvivalSVM(data_train, data_val, data_test, new_times, random_state=4):
    # Used random_state = 0, 1, 2, 3, 4

    print("\nFitting Support Vector Machine (SVM)...")

    df_train = data_train["pd.DataFrame"]
    df_test = data_test["pd.DataFrame"]

    X_train = df_train.drop(columns=["event", "time"])
    X_test = df_test.drop(columns=["event", "time"])

    y_train = list(zip(df_train["event"].astype(bool), df_train["time"]))
    y_train = np.array(y_train, dtype=[("cens", "bool"), ("time", "float")])

    svm = FastSurvivalSVM(max_iter=1000, random_state=random_state, tol=1e-05)

    # hyperparameters tuning
    def score_survival_model(model, X, y):
        prediction = model.predict(X)
        result = concordance_index_censored(y["cens"], y["time"], prediction)
        return result[0]

    param_grid = {"alpha": [2.0**v for v in range(-12, 13, 2)]}
    cv = ShuffleSplit(n_splits=100, test_size=0.5, random_state=random_state)
    gcv = GridSearchCV(
        svm, param_grid, scoring=score_survival_model, n_jobs=1, refit=False, cv=cv
    )
    gcv = gcv.fit(X_train, y_train)

    # fit with best hyperparameters
    svm.set_params(**gcv.best_params_)
    svm.fit(X_train, y_train)

    risk_test = svm.predict(X_test)

    return risk_test, None, None
