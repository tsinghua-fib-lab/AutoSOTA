import collections
import csv
import itertools
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from multiprocessing import cpu_count
from time import time

import numpy as np
import sympy
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

try:
    from bitween import milp_gurobi
except ImportError:
    milp_gurobi = None
try:
    from bitween import milp_pulp
except ImportError:
    milp_pulp = None
from bitween.analyzer import validate_preconditions
from bitween.checker import fuzz_and_check
from bitween.config import Config, Correctness, FullMILP, Method, MILPSolver
from bitween.fuzzer import fuzz_and_trace
from bitween.miscs import Symbolic, TimeoutFunction, getLogger
from bitween.reducer import Reducer
from bitween.sampler import Distribution, Domain, sample
from bitween.utilities import pp
from bitween.verifier import fuzz_and_verify
from bitween.z3utils import Z3

sympy.init_printing(use_unicode=False, wrap_line=False)

config = Config()
log = getLogger(__name__, config.logger_level, empty_format=True)

TABLE_WIDTH = config.property_table_width

__version__ = "1.0.0"


@dataclass(frozen=True)
class Equation:
    expr: sympy.Expr
    error: float
    pivot: str
    sample_size: int
    model_desc: str
    dimension: int
    note: str


@dataclass(frozen=True)
class Model:
    model: object
    score: float
    model_type: str
    params: dict
    coefficients: np.ndarray
    intercept: float
    X_test: np.ndarray
    y_test: np.ndarray


def load_input_data(file_path):
    with open(file_path, "r") as file:
        input_data = file.read()
    return input_data


def parse_dig_vtrace_file(input_data) -> dict:
    # Splitting the input data into lines
    lines = input_data.strip().split("\n")

    # Dictionary to store the data and variable names for each trace type
    traces = {}

    # Processing each line
    for line in lines:
        parts = line.split(";")
        parts = [part.strip() for part in parts]
        trace_type = parts[0]
        values = parts[1:]

        # Handle header lines
        if "I " in values[0]:
            # Strip 'I ' from the names and store them as variable names
            traces[trace_type] = {"terms": [name[2:] for name in values], "data": []}
            continue

        # Check if the trace type is already initialized
        if trace_type not in traces:
            continue

        # Append the data to the respective list in the trace, converting to numeric types
        traces[trace_type]["data"].append(np.fromstring("; ".join(values), sep="; "))

    # Convert data lists to numpy arrays
    for trace in traces:
        traces[trace]["data"] = np.array(traces[trace]["data"])
        # traces[trace]["data"].setflags(write=False)

    return traces


def generate_monomials(terms, degree):
    # Generate all combinations of terms up to the given degree
    monomials = []
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(terms, d):
            monomials.append("||".join(combo))
    return monomials


def calculate_monomial_data(terms, monomials, data):
    # Split each monomial and calculate its value for each row in the data
    monomial_values = np.ones((data.shape[0], len(monomials)))
    for i, monomial in enumerate(monomials):
        for term in monomial.split("||"):
            term_index = terms.index(term)
            monomial_values[:, i] *= data[:, term_index]
    return monomial_values


def process_trace(terms: list[str], data, degree):
    monomials = generate_monomials(terms, degree)
    extended_terms = terms + monomials[len(terms) :]
    monomial_data = calculate_monomial_data(terms, monomials, data)
    # append ones column for the constant term to the extended data and a constant term to the extended terms
    extended_terms.append("1")
    # recover || to *
    extended_terms = [term.replace("||", "*") for term in extended_terms]
    extended_data = np.hstack((monomial_data, np.ones((data.shape[0], 1))))
    return extended_terms, extended_data


# NOTE: Property Test
def property_test(term_coefs: dict[str, float], extended_terms, extended_data) -> float:
    # Evaluate the equation for each row in entire data
    error = 0.0
    for row in extended_data:
        lhs = 0
        for term, coef in term_coefs.items():
            lhs += coef * row[extended_terms.index(term)]
        error += abs(lhs)
    return error / extended_data.shape[0]


def find_model(pivot, terms, data, test_size=0.2):
    sample_size = data.shape[0]

    if config.use_sample_rate_regression and config.sample_rate_regression > 1:
        # select a random subset of the data based on the number of terms * sample rate
        sample_size = int(len(terms) * config.sample_rate_regression)
        threshold = config.sample_threshold_regression
        if sample_size < threshold:
            if data.shape[0] > threshold:
                sample_size = threshold
            else:  # use all data
                sample_size = data.shape[0]

        if sample_size < data.shape[0]:
            sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
            data = data[sample_indices]
        else:
            sample_size = data.shape[0]

    X = data
    y = data[:, terms.index(pivot)]

    X_train, X_test, y_train, y_test = train_test_split(
        np.delete(data, terms.index(pivot), axis=1),
        y,
        test_size=test_size,
        # random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    coefficients = model.coef_
    intercept = model.intercept_
    sample_size = X_train.shape[0]

    return {
        "model": model,
        "score": score,
        "model_type": "Linear",
        "params": "",  # NOTE: No parameters for simple linear regression
        "coefficients": coefficients,
        "intercept": intercept,
        "sample_size": sample_size,
        # "X_test": X_test,
        # "y_test": y_test,
        "X_test": X,  # Include the entire X for evaluating the equation
        "y_test": y,  # Include the entire y for evaluating the equation
    }


def find_model_w_lassocv(pivot, terms, data, test_size=0.2):
    sample_size = data.shape[0]

    if config.use_sample_rate_regression and config.sample_rate_regression > 1:
        # select a random subset of the data based on the number of terms * sample rate
        sample_size = int(len(terms) * config.sample_rate_regression)
        threshold = config.sample_threshold_regression
        if sample_size < threshold:
            if data.shape[0] > threshold:
                sample_size = threshold
            else:  # use all data
                sample_size = data.shape[0]

        if sample_size < data.shape[0]:
            sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
            data = data[sample_indices]
        else:
            sample_size = data.shape[0]

    X = data
    y = data[:, terms.index(pivot)]

    X_train, X_test, y_train, y_test = train_test_split(
        np.delete(data, terms.index(pivot), axis=1),
        y,
        test_size=test_size,
        # random_state=42,
    )

    model = LassoCV(cv=5, random_state=42, alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    coefficients = model.coef_
    intercept = model.intercept_
    sample_size = X_train.shape[0]

    return {
        "model": model,
        "score": score,
        "model_type": "Lasso",
        "params": {"alpha": model.alpha_},
        "coefficients": coefficients,
        "intercept": intercept,
        "sample_size": sample_size,
        # "X_test": X_test,
        # "y_test": y_test,
        "X_test": X,  # Include the entire X for evaluating the equation
        "y_test": y,  # Include the entire y for evaluating the equation
    }


def sfs_heuristics(extended_terms, extended_data, degree=2, test_size=0.2):

    def get_rate(
        degree=degree,
        initial_rate=config.selector_initial_rate,
        decay_rate=config.selector_decay_rate,
    ):
        """
        Calculate the rate for a given degree using an exponential decay formula.

        :param degree: The degree for which to calculate the rate.
        :param initial_rate: The initial rate at degree 1.
        :param decay_rate: The rate of decay per degree increase.
        :return: The calculated rate.
        """
        return initial_rate * ((1 - decay_rate) ** (degree - 1))

    # NOTE Include the constant term (intercept=False in LinearRegression)
    # X = extended_data[:, :-1]  # Exclude the constant term

    sample_size = extended_data.shape[0]

    if config.use_sample_rate_regression and config.sample_rate_regression > 1:
        # select a random subset of the extended_data based on the number of terms * sample rate
        sample_size = int(len(extended_terms) * config.sample_rate_regression)
        threshold = config.sample_threshold_regression
        if sample_size < threshold:
            if extended_data.shape[0] > threshold:
                sample_size = threshold
            else:  # use all data
                sample_size = extended_data.shape[0]

        if sample_size < extended_data.shape[0]:
            sample_indices = np.random.choice(
                extended_data.shape[0], sample_size, replace=False
            )
            extended_data = extended_data[sample_indices]
        else:
            sample_size = extended_data.shape[0]

    def fit_model(target_idx):
        y = extended_data[:, target_idx]
        # Exclude the target variable from the features
        X_ = np.delete(extended_data, target_idx, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_, y, test_size=test_size, random_state=42
        )
        # extract the target variable from the extended_terms and keep the rest
        pivot = extended_terms[target_idx]
        features = np.array([term for term in extended_terms if term != pivot])

        # initial best model info
        best_error = np.inf
        best_score = -np.inf
        best_model = None
        best_intercept = None
        best_coefficients = None

        # TODO observe this hyperparameter
        # max_features = int(X_train.shape[1] * get_rate())
        max_features = config.selector_max_features
        # Define the range of features to select
        last_mse = np.inf
        # n_features = max_features
        counter = collections.Counter()
        models = []
        for i in range(2):
            if i == 1:
                # remove the max counted term from terms and data
                dominant_term = counter.most_common(1)[0][0]
                idx = extended_terms.index(dominant_term)
                X_ = np.delete(X_, idx, axis=1)
                features = np.delete(features, np.where(features == dominant_term))
                X_train = np.delete(X_train, idx, axis=1)
                log.info(f"\nRemoved {dominant_term} from the {features}")

            if not config.selector_parallel:
                log.info("-" * 80)

            for n_features in range(1, max_features + 1):
                # for n_features in range(max_features, max_features + 1):
                log.info(f"\nEvaluating model with {n_features} features selected:")
                # Define the feature selector with the current number of features
                selector = SequentialFeatureSelector(
                    LinearRegression(fit_intercept=False),
                    n_features_to_select=n_features,
                    cv=3,  # TODO check this, 5 is the recommended value
                    # n_jobs=-1,
                )
                # Define the pipeline with the current feature selector
                # cachedir = mkdtemp()
                pipe = Pipeline(
                    [
                        ("selector", selector),
                        ("linear", LinearRegression(fit_intercept=False)),
                    ],
                    # memory=cachedir,
                )

                # Fit the pipeline to the training data
                pipe.fit(X_train, y_train)

                # Get the selected features and their coefficients
                mask = pipe.named_steps["selector"].get_support()
                selected_features = features[mask]
                model = pipe.named_steps["linear"]
                coefficients = model.coef_

                # Count the frequency of each term
                counter.update(selected_features)

                log.info(f"Selected features: {selected_features}")
                # Construct the polynomial equation
                equation = sympy.Rational(0)
                extended_coefficients = np.zeros(features.shape[0])
                feature_list = features.tolist()
                for feature, coeff in zip(selected_features, coefficients):
                    idx = feature_list.index(feature)
                    extended_coefficients[idx] = coeff
                    # TODO be careful with rounding
                    coeff = Fraction(coeff).limit_denominator(10 * config.precision)
                    # coeff = round(coeff, config.precision)
                    if feature == "1" and coeff != 0:
                        # equation += sympy.Rational(coeff)
                        equation += coeff
                    elif coeff != 0:
                        # equation += sympy.Rational(coeff) * sympy.Symbol(feature)
                        equation += coeff * sympy.Symbol(feature)

                # equation = equation.rstrip(" + ")  # remove the last plus sign
                log.info(f"Equation: {pivot} = {equation}")
                equation = sympy.Symbol(pivot) - equation

                # Predict and evaluate the model on the test set
                # mse = mean_squared_error(y_test, model.predict(X_test[:, mask]))
                # NOTE use the entire X and y for evaluating the equation
                # mse = mean_squared_error(y, model.predict(X_[:, mask]))
                mse = mean_absolute_error(y, model.predict(X_[:, mask]))
                if mse <= best_error:
                    best_score = mse
                    best_model = model
                    best_intercept = model.intercept_
                    best_coefficients = extended_coefficients

                log.info(f"Mean Absolute Error on Test Data: {pp(mse)}")

                if mse < 1e-10:  # TODO check this threshold
                    log.info(
                        f"Model for {pivot}: {equation.evalf()} (Found a perfect model)"
                    )
                    break

                # if mse - last_mse > 1e-4:  # TODO check this threshold
                #     log.info(f"mse increased from {last_mse} to {mse}, stopping the search.")
                #     break
                # last_mse = mse

            if i == 1:
                params = {"blocked": f"{dominant_term}"}
            else:
                params = ""

            models.append(
                {
                    "model": best_model,
                    "score": best_score,
                    "model_type": "ForwardSelection",
                    "params": params,
                    "intercept": best_intercept,
                    "coefficients": best_coefficients,
                    "sample_size": sample_size,
                    "X_test": extended_data,  # NOTE: Entire X for evaluating the eq.
                    "y_test": extended_data[:, target_idx],  # NOTE: Entire y
                }
            )

        return (pivot, models)

    # Create a model for each term in extended_terms, excluding the constant '1'
    results = Parallel(n_jobs=-1 if config.selector_parallel else 1)(
        delayed(fit_model)(i) for i in range(len(extended_terms) - 1)
    )

    return {pivot: content for pivot, content in results}


def linear_regression_heuristics(extended_terms, extended_data, test_size=0.2):
    X = extended_data[:, :-1]  # Exclude the constant term
    models = {}

    def fit_model(target_idx):
        y = extended_data[:, target_idx]
        # Exclude the target variable from the features
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, target_idx, axis=1), y, test_size=test_size, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        # Insert 0 for the target variable's coefficient
        coefficients = model.coef_
        intercept = model.intercept_

        return (
            extended_terms[target_idx],
            model,
            score,
            coefficients,
            intercept,
            # X_test,
            # y_test,
            X,  # Include the entire X for evaluating the equation
            y,  # Include the entire y for evaluating the equation
        )

    # Create a model for each term in extended_terms, excluding the constant '1'
    results = Parallel(n_jobs=-1)(
        delayed(fit_model)(i) for i in range(len(extended_terms) - 1)
    )

    sample_size = int(extended_data.shape[0] * (1 - test_size))
    for term, model, score, coefficients, intercept, X_test, y_test in results:
        models[term] = {
            "model": model,
            "score": score,
            "model_type": "Linear",
            "params": "",  # NOTE: No parameters for simple linear regression
            "coefficients": coefficients,
            "intercept": intercept,
            "sample_size": sample_size,
            "X_test": X_test,
            "y_test": y_test,
        }

    return models


def multiple_regression_heuristics(extended_terms, extended_data, test_size=0.2):
    X_ = extended_data[:, :-1]  # Exclude the constant term

    # Define the models and parameters for grid search
    model_params = {
        "Linear": {
            "model": LinearRegression(),
            # "params": {},
            # "params": {"positive": [True, False]},
            # "params": {"fit_intercept": [True, False]},
            "params": {"fit_intercept": [False, True], "positive": [True, False]},
        },
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {
                "alpha": [1e-3, 1e-2, 1e-1, 100, 1000],
                "fit_intercept": [True, False],
            },
        },
        "Lasso": {
            "model": Lasso(random_state=42),
            "params": {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 100, 1000],
                "fit_intercept": [True, False],
            },
        },
        # "ElasticNet": {
        #     "model": ElasticNet(random_state=42),
        #     "params": {
        #         "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        #         "l1_ratio": [0.1, 0.5, 0.9, 0.95, 0.99, 1],
        #     },
        # },
    }

    sample_size = extended_data.shape[0]

    if config.use_sample_rate_regression and config.sample_rate_regression > 1:
        # select a random subset of the extended_data based on the number of terms * sample rate
        sample_size = int(len(extended_terms) * config.sample_rate_regression)
        threshold = config.sample_threshold_regression
        if sample_size < threshold:
            if extended_data.shape[0] > threshold:
                sample_size = threshold
            else:  # use all data
                sample_size = extended_data.shape[0]

        if sample_size < extended_data.shape[0]:
            sample_indices = np.random.choice(
                extended_data.shape[0], sample_size, replace=False
            )
            extended_data = extended_data[sample_indices]
        else:
            sample_size = extended_data.shape[0]

    X = extended_data[:, :-1]  # Exclude the constant term

    def fit_model(i):
        y = extended_data[:, i]
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, i, axis=1), y, test_size=test_size, random_state=None
        )

        best_score = -np.inf
        best_model = None
        best_model_name = ""
        best_params = {}
        best_intercept = None
        best_coefficients = None
        cv = config.cross_validation
        scoring = str(config.regression_score)
        for model_name, mp in model_params.items():
            clf = GridSearchCV(
                mp["model"], mp["params"], cv=cv, scoring=scoring, n_jobs=-1
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore")
                try:
                    clf.fit(X_train, y_train)
                    if clf.best_score_ > best_score:
                        best_score = clf.best_score_
                        best_model = clf.best_estimator_
                        best_model_name = model_name
                        best_params = clf.best_params_
                        best_intercept = best_model.intercept_
                        best_coefficients = best_model.coef_
                except Exception as e:
                    log.debug(
                        f"Model for {extended_terms[i]}: {model_name}({mp['params']})"
                    )
                    log.debug(e)

        return extended_terms[i], {
            "model": best_model,
            "score": best_score,
            "model_type": best_model_name,
            "params": best_params,
            "intercept": best_intercept,
            "coefficients": best_coefficients,
            "sample_size": sample_size,
            "X_test": X_,  # NOTE: Include the entire X for evaluating the eq.
            "y_test": X_[:, i],  # NOTE: Include the entire y for evaluating the eq.
        }

    results = Parallel(n_jobs=-1)(
        delayed(fit_model)(i) for i in range(len(extended_terms) - 1)
    )

    # return {term: content for term, content in results}
    models = {}
    for term, content in results:
        if content["model"] is None:
            log.debug(f"Model for {term}: No model found")
            continue
        models[term] = content
    return models


def infer_equations(  # noqa F811
    models,
    extended_terms,
    extended_data,
    coeff_threshold=config.coeff_threshold,
    coeff_cutoff=config.coeff_cutoff,
    intercept_cutoff=config.intercept_cutoff,
    epsilon=config.epsilon,
    objective_threshold=config.objective_threshold,
):

    # NOTE: MILP-based Synthesis
    def milp_synthesis_wrapper(
        pivot: str,
        selected_terms: list[str],
        selected_data: np.ndarray,
        model_desc: str,
        blocked: str,
        bound: int = config.bound,
    ):
        str_ = ""
        optimal = None

        dims = len(selected_terms)
        sample_size = selected_data.shape[0]
        if config.use_sample_rate_milp and config.sample_rate_milp > 1:
            sample_size = int(len(selected_terms) * config.sample_rate_milp)
            threshold = config.sample_threshold_milp
            if sample_size < threshold:
                if selected_data.shape[0] > threshold:
                    sample_size = threshold
                else:  # use all data
                    sample_size = selected_data.shape[0]

            if sample_size < selected_data.shape[0]:
                sample_indices = np.random.choice(
                    selected_data.shape[0], sample_size, replace=False
                )
                selected_data = selected_data[sample_indices]
            else:
                sample_size = selected_data.shape[0]

        if config.milp_solver == MILPSolver.GUROBI:
            optimal = milp_gurobi.OPTIMAL
            status, expr, obj, term_coefs, _ = milp_gurobi.milp_synthesis(
                selected_data,
                selected_terms,
                pivot,
                bound=bound,
                timeout=config.milp_timeout,
                blocked=blocked,
                solver=config.milp_solver,
            )
        else:
            optimal = milp_pulp.OPTIMAL
            status, expr, obj, term_coefs, _ = milp_pulp.milp_synthesis(
                selected_data,
                selected_terms,
                pivot,
                bound=bound,
                timeout=config.milp_timeout,
                blocked=blocked,
                solver=config.milp_solver,
            )

        if status == optimal:
            if abs(obj) < objective_threshold:  # check if the objective is small enough
                equation = sympy.sympify(expr)

                s_eq = str(equation.evalf())
                if len(s_eq) > TABLE_WIDTH:
                    s_eq = s_eq[:TABLE_WIDTH] + " ..."
                    s_eq = f"{s_eq + ' = 0':<{TABLE_WIDTH+4}} "
                else:
                    s_eq = f"{s_eq + ' = 0':<{TABLE_WIDTH+8}} "

                # NOTE: Property Test: Evaluate the equation for each row in entire data
                error = property_test(term_coefs, extended_terms, extended_data)

                s_err = f"err: {round(error, 2):<5.2f}"
                s_obj = f"obj: {round(obj, 2):.2f}"
                str_ += s_eq + f"({s_err}); ({model_desc}); ({s_obj}); [{pivot}] **\n"

                return (
                    Equation(
                        equation, error, pivot, sample_size, model_desc, dims, str_
                    ),
                    None,
                    None,
                )
            else:
                str_ += f"MILP for {pivot}: Objective too large: {obj}\n"
                return (
                    Equation(None, None, pivot, sample_size, model_desc, dims, str_),
                    None,
                    None,
                )
        else:
            str_ += f"MILP for {pivot}: No solution found\n"
            return (
                Equation(None, None, pivot, sample_size, model_desc, dims, str_),
                None,
                None,
            )

    def infer_equation(pivot, model, extended_terms, extended_data, model_desc):
        str_ = ""
        sample_size = model["sample_size"]
        dims = len(extended_terms)

        # NOTE: preparing an equation from the regression model
        adaptive_intercept_cutoff = max(intercept_cutoff, int(5 * np.abs(model["intercept"]))) if model["intercept"] is not None else intercept_cutoff
        if config.use_cutoff and np.abs(model["intercept"]) >= adaptive_intercept_cutoff:
            # str += f"Model for {pivot}: Intercept = {model['intercept']}!\n"
            return (
                Equation(None, None, pivot, sample_size, model_desc, dims, str_),
                None,
                None,
            )

        rhs = 0
        coeff_terms = {}
        selected_terms = [pivot]  # Include LHS term
        terms = [item for item in extended_terms if item != pivot]

        # check all coefficients with adaptive cutoff
        adaptive_cutoff = max(coeff_cutoff, int(5 * np.max(np.abs(model["coefficients"])))) if len(model["coefficients"]) > 0 else coeff_cutoff
        if config.use_cutoff and np.any(np.abs(model["coefficients"]) > adaptive_cutoff):
            # str += f"Model for {pivot}: Large Coefficient!\n"
            return (
                Equation(None, None, pivot, sample_size, model_desc, dims, str_),
                None,
                None,
            )

        for i, coefficient in enumerate(model["coefficients"]):
            if abs(coefficient) >= coeff_threshold:
                # TODO be careful with rounding
                # coeff = round(coefficient, config.precision)
                coeff = Fraction(coefficient).limit_denominator(10 * config.precision)
                if coeff != 0:
                    # rhs += sympy.Rational(coeff) * sympy.Symbol(terms[i])
                    rhs += coeff * sympy.Symbol(terms[i])
                    coeff_terms[terms[i]] = coefficient
                    # Include term in selected list
                    selected_terms.append(terms[i])

        # Add the constant term (intercept)
        # TODO be careful with rounding
        intercept = round(model["intercept"], config.precision)
        if intercept > coeff_threshold:
            # rhs += sympy.Rational(intercept)
            rhs += Fraction(intercept).limit_denominator(10 * config.precision)

        equation = sympy.Symbol(pivot) - rhs
        # equation = sympy.simplify(equation)
        s_eq = str(equation.evalf())
        if len(s_eq) > TABLE_WIDTH:
            s_eq = s_eq[:TABLE_WIDTH] + " ..."
            s_eq = f"{s_eq + ' = 0':<{TABLE_WIDTH+4}} "
        else:
            s_eq = f"{s_eq + ' = 0':<{TABLE_WIDTH+8}} "

        str_ += s_eq

        # log.info(f"\nModel for {pivot}: {s_eq}")

        # NOTE: Property Test: Evaluate the equation for each row in X_test
        X_test = model["X_test"]
        y_test = model["y_test"]

        rhs_values = np.zeros(y_test.shape[0])
        for i, row in enumerate(X_test):
            for t, coeff in coeff_terms.items():
                rhs_values[i] += coeff * row[extended_terms.index(t)]
            if intercept > coeff_threshold:
                rhs_values[i] += intercept

        me = np.mean(np.abs(rhs_values - y_test))
        str_ += f"(err: {round(me, 2):<5.2f}): ({model_desc}); [{dims}] [{pivot}]"
        if me < epsilon:
            str_ += "***\n"
        else:
            str_ += "\n"

        if config.milp_enabled is not True and config.regression_refinement is not True:
            return (
                Equation(equation, me, pivot, sample_size, model_desc, dims, str_),
                None,
                None,
            )

        # NOTE: MILP Synthesis based on the regression model
        # Selecting respective columns from data
        selected_indices = [extended_terms.index(term) for term in selected_terms]
        selected_data = extended_data[:, selected_indices]

        return (
            Equation(equation, me, pivot, sample_size, model_desc, dims, str_),
            selected_terms,
            selected_data,
        )

    # NOTE: Inference starts here
    results = []
    milp_input = []
    term_frequencies = collections.Counter()
    for pivot, model in models.items():
        model_dsc = f"{model['model_type']}({model['params']})"
        equation, term, data = infer_equation(
            pivot, model, extended_terms, extended_data, model_dsc
        )
        if equation.expr is not None:
            results.append(equation)
            # milp_input.append((pivot, term, data))  # NOTE: MILP input is collected here
            if equation.error >= epsilon:
                milp_input.append((pivot, term, data))
            # TODO: is this a good idea?
            if equation.error < epsilon:
                continue

            # NOTE: start Regression Refinement
            if config.regression_refinement and len(term) > 1:
                model_1 = find_model(pivot, term, data)
                model_dsc = f"Refine({model_1['model_type']}({model_1['params']})).1"
                equation, term_1, data_1 = infer_equation(
                    pivot, model_1, term, data, model_dsc
                )
                if equation.expr is not None:
                    results.append(equation)
                if term_1 is not None and len(term_1) > 1:
                    model_2 = find_model(pivot, term_1, data_1)
                    model_dsc = (
                        f"Refine({model_2['model_type']}({model_2['params']})).2"
                    )
                    equation, term_2, data_2 = infer_equation(
                        pivot, model_2, term_1, data_1, model_dsc
                    )
                    if equation.expr is not None:
                        results.append(equation)
                    if (
                        term_2 is not None
                        and len(term_2) > 2
                        and len(term_1) > len(term_2)
                    ):
                        model_3 = find_model(pivot, term_2, data_2)
                        model_dsc = (
                            f"Refine({model_3['model_type']}({model_3['params']})).3"
                        )
                        equation, _, _ = infer_equation(
                            pivot, model_3, term_2, data_2, model_dsc
                        )
                        if equation.expr is not None:
                            results.append(equation)

    # collect term frequencies from inferred equations (results)
    for equation in results:
        if not equation.model_desc.startswith("Milp"):
            term_frequencies.update([str(s) for s in equation.expr.free_symbols])
    # log.info(f"Term Frequencies: {term_frequencies}\n")

    # NOTE: MILP Synthesis based on the regression model
    if config.milp_enabled:
        # NOTE: Parallel MILP Synthesis based on the regression model
        solver = str(config.milp_solver)
        blocked = None  # NOTE: No blocked term for now
        if config.milp_parallel:
            model_desc = f"Milp({solver})"
            milp_results = Parallel(n_jobs=-1)(
                delayed(milp_synthesis_wrapper)(*mi, model_desc, blocked)
                for mi in milp_input
            )
            for equation, _, _ in milp_results:
                results.append(equation)

        else:
            model_desc = f"Milp({solver})"
            milp_results = [
                milp_synthesis_wrapper(*mi, model_desc, blocked) for mi in milp_input
            ]
            for equation, _, _ in milp_results:
                results.append(equation)

    # NOTE: Run a milp over the most frequent terms, pivot all terms
    if (
        config.milp_enabled
        and config.milp_freq_refine_enabled
        and len(term_frequencies) > 5
    ):
        terms = []  # there is no "1" in term_frequencies
        # keep the relative order of the terms
        for term in extended_terms:
            if term in term_frequencies:
                terms.append(term)
        selected_indices = [extended_terms.index(term) for term in terms]
        selected_data = extended_data[:, selected_indices]
        model_desc = f"Milp({config.milp_solver})+Freqs"
        inputs = []
        for pivot in terms:
            inputs.append((pivot, terms, selected_data, model_desc))

        blocked = None  # NOTE: No blocked term for now
        milp_results = Parallel(n_jobs=-1)(
            delayed(milp_synthesis_wrapper)(*mi, blocked) for mi in inputs
        )
        for equation, _, _ in milp_results:
            results.append(equation)

    # NOTE: EAGER MILP over Full Model that includes all terms
    if config.milp_enabled and config.full_milp_strategy != FullMILP.NEVER:
        if (
            config.full_milp_strategy == FullMILP.AUTO
            and extended_data.shape[0] < config.full_milp_threshold
        ) or config.full_milp_strategy == FullMILP.ALWAYS:
            model_desc = f"Milp({solver})+Full"

            milp_input = []
            for pivot in extended_terms[:-1]:
                terms = extended_terms.copy()
                terms.pop()
                data = np.delete(extended_data, -1, axis=1)
                milp_input.append((pivot, terms, data, model_desc))

            blocked = None  # NOTE: No blocked term for now
            milp_results = Parallel(n_jobs=-1)(
                delayed(milp_synthesis_wrapper)(*mi, blocked) for mi in milp_input
            )
            for equation, _, _ in milp_results:
                results.append(equation)

    # NOTE: Linear Equation Solver based on the regression model
    if config.linear_solver_enabled:
        # NOTE: Linear Solver

        milp_input = []
        for pivot in extended_terms[:-1]:
            terms = extended_terms.copy()
            data = extended_data.copy()
            milp_input.append((pivot, terms, data))

        lin_solve_results = Parallel(n_jobs=-1)(
            delayed(Symbolic.linear_solve)(*mi) for mi in milp_input
        )
        dims = len(extended_terms)
        for expr, sample_size in lin_solve_results:
            if expr is not None and expr != 0:
                equation = Equation(expr, 0, pivot, sample_size, "LinSolve", dims, "")
            results.append(equation)

    # remove None values
    return [
        r for r in results if r is not None and r.error is not None
    ], term_frequencies


# Function to round the coefficients of an expression
def round_coefficients(expr, decimals):
    return expr.xreplace({n: round(n, decimals) for n in expr.atoms(sympy.Number)})


def find_models_with_pysr(extended_terms, extended_data, test_size=0.2):
    from pysr import PySRRegressor

    config.precision = 2  # NOTE: Force the precision to 2 for PySR

    X = extended_data[:, :-1]  # Exclude the constant term
    results = []

    def symbolic_regression(target_idx):
        pivot = extended_terms[target_idx]
        terms = [term for term in extended_terms if term != pivot]
        pysr_vars_map = []
        for i, term in enumerate(terms):
            pysr_vars_map.append((sympy.Symbol(f"x{i}"), sympy.Symbol(term)))
        y = extended_data[:, target_idx]
        # Exclude the target variable from the features
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, target_idx, axis=1), y, test_size=test_size, random_state=42
        )
        iterations = 50
        model_name = f"PySR(iter={iterations})"
        timeout = config.pysr_timeout / (len(extended_terms) - 1)
        model = PySRRegressor(
            niterations=iterations,  # < Increase me for better results
            binary_operators=["+", "*"],
            # unary_operators=[
            #     "cos",
            #     "exp",
            #     "sin",
            #     "sqrt",
            #     "inv(x) = 1/x",
            #     # ^ Custom operator (julia syntax)
            # ],
            populations=max(15, cpu_count() * 2),
            timeout_in_seconds=timeout,
            # extra_sympy_mappings={"inv": lambda x: 1 / x},
            # ^ Define operator for SymPy as well
            # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            # ^ Custom loss function (julia syntax)
            select_k_features=config.selector_max_features,  # NOTE: this is important
            # ^ Train on only the 4 most important features
            progress=False,
            temp_equation_file=True,
            print_precision=3,
        )
        model.fit(X_train, y_train)
        log.info(model)
        rhs = model.sympy().subs(pysr_vars_map)
        log.info(f"pysr: {pivot} = {rhs}")
        rhs = round_coefficients(sympy.simplify(rhs.expand()), config.precision)
        log.info(f"{pivot} = {rhs}")
        equation = sympy.simplify(sympy.Symbol(pivot) - rhs)
        log.info(str(equation) + " = 0")
        if equation == 0:
            log.warning(f"SR found a zero equation for {pivot}")
            return None
        feature_size = X_train.shape[1]
        sample_size = X_train.shape[0]
        mse = mean_squared_error(y_test, model.predict(X_test))
        return Equation(equation, mse, pivot, sample_size, model_name, feature_size, "")

    # Create a model for each term in extended_terms, excluding the constant '1'
    for i in range(len(extended_terms) - 1):
        equation = symbolic_regression(i)
        if equation is not None and equation.expr is not None:
            results.append(equation)

    return results


def find_models_with_kan(extended_terms, extended_data, test_size=0.2):
    pass


def find_models_with_eager_milp(extended_terms, extended_data, test_size=0.2):

    selected_data = extended_data[:, :-1]  # Exclude the constant term
    selected_terms = extended_terms[:-1]  # Exclude the constant term
    results = []

    # NOTE: MILP-based Synthesis
    def milp_synthesis_wrapper(
        pivot: str,
        selected_terms: list[str],
        selected_data: np.ndarray,
        model_desc: str,
        blocked: str,
        bound: int = config.bound,
    ):
        str_ = ""
        optimal = None

        dims = len(selected_terms)
        sample_size = selected_data.shape[0]
        if config.use_sample_rate_milp and config.sample_rate_milp > 1:
            sample_size = int(len(selected_terms) * config.sample_rate_milp)
            threshold = config.sample_threshold_milp
            if sample_size < threshold:
                if selected_data.shape[0] > threshold:
                    sample_size = threshold
                else:  # use all data
                    sample_size = selected_data.shape[0]

            if sample_size < selected_data.shape[0]:
                sample_indices = np.random.choice(
                    selected_data.shape[0], sample_size, replace=False
                )
                selected_data = selected_data[sample_indices]
            else:
                sample_size = selected_data.shape[0]

        if config.milp_solver == MILPSolver.GUROBI:
            optimal = milp_gurobi.OPTIMAL
            status, expr, obj, term_coefs, _ = milp_gurobi.milp_synthesis(
                selected_data,
                selected_terms,
                pivot,
                bound=bound,
                timeout=config.milp_timeout,
                blocked=blocked,
                solver=config.milp_solver,
            )
        else:
            optimal = milp_pulp.OPTIMAL
            status, expr, obj, term_coefs, _ = milp_pulp.milp_synthesis(
                selected_data,
                selected_terms,
                pivot,
                bound=bound,
                timeout=config.milp_timeout,
                blocked=blocked,
                solver=config.milp_solver,
            )

        if status == optimal:
            if (
                abs(obj) < config.objective_threshold
            ):  # check if the objective is small enough
                equation = sympy.sympify(expr)

                s_eq = str(equation.evalf())
                if len(s_eq) > TABLE_WIDTH:
                    s_eq = s_eq[:TABLE_WIDTH] + " ..."
                    s_eq = f"{s_eq + ' = 0':<{TABLE_WIDTH+4}} "
                else:
                    s_eq = f"{s_eq + ' = 0':<{TABLE_WIDTH+8}} "

                # NOTE: Property Test: Evaluate the equation for each row in entire data
                error = property_test(term_coefs, extended_terms, extended_data)

                s_err = f"err: {round(error, 2):<5.2f}"
                s_obj = f"obj: {round(obj, 2):.2f}"
                str_ += s_eq + f"({s_err}); ({model_desc}); ({s_obj}); [{pivot}] **\n"

                return (
                    Equation(
                        equation, error, pivot, sample_size, model_desc, dims, str_
                    ),
                    None,
                    None,
                )
            else:
                str_ += f"MILP for {pivot}: Objective too large: {obj}\n"
                return (
                    Equation(None, None, pivot, sample_size, model_desc, dims, str_),
                    None,
                    None,
                )
        else:
            str_ += f"MILP for {pivot}: No solution found\n"
            return (
                Equation(None, None, pivot, sample_size, model_desc, dims, str_),
                None,
                None,
            )

    # Create a model for each term in extended_terms, excluding the constant '1'
    for i in range(len(selected_terms)):
        equation = milp_synthesis_wrapper(
            selected_terms[i],
            selected_terms.copy(),
            selected_data.copy(),
            "EagerMILP",
            None,
        )
        if (
            equation is not None
            and equation[0] is not None
            and equation[0].expr is not None
        ):
            results.append(equation[0])

    return results


def find_models_with_gplearn(extended_terms, extended_data, test_size=0.2):
    from gplearn.genetic import SymbolicRegressor

    config.precision = 2  # NOTE: Force the precision to 2 for GPlearn

    X = extended_data[:, :-1]  # Exclude the constant term
    results = []

    locals = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "neg": lambda x: -x,
        "inv": lambda x: 1 / x,
        "pow": lambda x, y: x**y,
        "sqrt": sympy.sqrt,
        "log": sympy.log,
        "abs": sympy.Abs,
        "max": sympy.Max,
        "min": sympy.Min,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
    }

    def symbolic_regression(target_idx):
        pivot = extended_terms[target_idx]
        terms = [term for term in extended_terms if term != pivot]
        gplearn_vars_map = []
        for i, term in enumerate(terms):
            gplearn_vars_map.append((sympy.Symbol(f"X{i}"), sympy.Symbol(term)))
        y = extended_data[:, target_idx]
        # Exclude the target variable from the features
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, target_idx, axis=1), y, test_size=test_size, random_state=42
        )
        model_name = "gplearn"
        model = SymbolicRegressor(
            population_size=1000,
            generations=20,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
            parsimony_coefficient=0.01,
            # random_state=0,
            function_set=("add", "sub", "mul"),
            n_jobs=-1,
        )
        log.info("-" * 80)
        model.fit(X_train, y_train)
        log.info(f"gplearn: {model._program}")
        s_eq = str(model._program)
        if len(s_eq) > 400:
            log.warning(f"SR expression too long: {s_eq[:400]} ...")
            return None
        rhs = sympy.sympify(str(model._program), locals=locals).expand()
        rhs = rhs.subs(gplearn_vars_map)
        log.info(f"gplearn: {pivot} = {rhs}")
        rhs = round_coefficients(rhs, config.precision)
        log.info(f"{pivot} = {rhs}")
        equation = sympy.simplify(sympy.Symbol(pivot) - rhs)
        s_eq = str(equation)
        log.info(s_eq + " = 0")
        if equation == 0:
            log.warning(f"SR found a zero equation for {pivot}")
            return None
        feature_size = X_train.shape[1]
        sample_size = X_train.shape[0]
        mse = mean_squared_error(y_test, model.predict(X_test))
        log.info(f"MSE: {mse}")
        return Equation(equation, mse, pivot, sample_size, model_name, feature_size, "")

    # Create a model for each term in extended_terms, excluding the constant '1'
    for i in range(len(extended_terms) - 1):
        equation = symbolic_regression(i)
        if equation is not None and equation.expr is not None:
            results.append(equation)

    return results


def find_models_with_rag_sr(extended_terms, extended_data, test_size=0.2):
    from bitween.rag_sr import RAGSRRegressor

    config.precision = 2

    X = extended_data[:, :-1]  # Exclude the constant term
    results = []

    locals = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "neg": lambda x: -x,
        "inv": lambda x: 1 / x,
        "pow": lambda x, y: x**y,
        "sqrt": sympy.sqrt,
        "log": sympy.log,
        "abs": sympy.Abs,
        "max": sympy.Max,
        "min": sympy.Min,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
    }

    def _regression(target_idx):
        pivot = extended_terms[target_idx]
        terms = [term for term in extended_terms if term != pivot]

        rag_sr_vars_map = []
        for i, term in enumerate(terms):
            rag_sr_vars_map.append((sympy.Symbol(f"X{i}"), sympy.Symbol(term)))
        y = extended_data[:, target_idx]
        # Exclude the target variable from the features
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, target_idx, axis=1), y, test_size=test_size, random_state=42
        )

        model_name = "RAG_SR"
        model = RAGSRRegressor()
        log.info("-" * 80)
        model.fit(X_train, y_train, categorical_features=np.zeros(X_train.shape[1]))
        s_eq = str(model.model())
        log.info(f"RAG_SR: {s_eq}")

        if len(s_eq) > 400:
            log.warning(f"SR expression too long: {s_eq[:400]} ...")
            return None

        rhs = sympy.sympify(s_eq, locals=locals).expand()
        rhs = rhs.subs(rag_sr_vars_map)
        log.info(f"RAG_SR: {pivot} = {rhs}")
        rhs = round_coefficients(rhs, config.precision)

        equation = sympy.simplify(sympy.Symbol(pivot) - rhs)
        s_eq = str(equation)
        log.info(s_eq + " = 0")
        if equation == 0:
            log.warning(f"SR found a zero equation for {pivot}")
            return None

        feature_size = X_train.shape[1]
        sample_size = X_train.shape[0]
        mse = mean_squared_error(y_test, model.predict(X_test))
        log.info(f"MSE: {mse}")

        return Equation(equation, mse, pivot, sample_size, model_name, feature_size, "")

    # Create a model for each term in extended_terms, excluding the constant '1'
    for i in range(len(extended_terms) - 1):
        equation = _regression(i)
        if equation is not None and equation.expr is not None:
            results.append(equation)

    return results


def find_models_with_parfam(extended_terms, extended_data, test_size=0.2):
    from parfam import ParFamWrapper

    config.precision = 2

    X = extended_data[:, :-1]  # Exclude the constant term
    results = []

    locals = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "neg": lambda x: -x,
        "inv": lambda x: 1 / x,
        "pow": lambda x, y: x**y,
        "sqrt": sympy.sqrt,
        "log": sympy.log,
        "abs": sympy.Abs,
        "max": sympy.Max,
        "min": sympy.Min,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
    }

    def symbolic_regression(target_idx):
        pivot = extended_terms[target_idx]
        terms = [term for term in extended_terms if term != pivot]
        parfam_vars_map = []
        for i, term in enumerate(terms):
            parfam_vars_map.append((sympy.Symbol(f"x{i}"), sympy.Symbol(term)))
        y = extended_data[:, target_idx]
        # Exclude the target variable from the features
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, target_idx, axis=1), y, test_size=test_size, random_state=42
        )

        model_name = "PARFAM"
        model = ParFamWrapper(iterate=True, config_name="big")
        log.info("-" * 80)
        model.fit(X_train, y_train)
        s_eq = str(model.formula_reduced)
        log.info(f"PARFAM: {s_eq}")

        if len(s_eq) > 400:
            log.warning(f"SR expression too long: {s_eq[:400]} ...")
            return None

        rhs = sympy.sympify(s_eq, locals=locals).expand()
        rhs = rhs.subs(parfam_vars_map)
        log.info(f"PARFAM: {pivot} = {rhs}")
        rhs = round_coefficients(rhs, config.precision)

        equation = sympy.simplify(sympy.Symbol(pivot) - rhs)
        s_eq = str(equation)
        log.info(s_eq + " = 0")
        if equation == 0:
            log.warning(f"SR found a zero equation for {pivot}")
            return None

        feature_size = X_train.shape[1]
        sample_size = X_train.shape[0]
        mse = mean_squared_error(y_test, model.predict(X_test))
        log.info(f"MSE: {mse}")

        return Equation(equation, mse, pivot, sample_size, model_name, feature_size, "")

    # Create a model for each term in extended_terms, excluding the constant '1'
    for i in range(len(extended_terms) - 1):
        equation = symbolic_regression(i)
        if equation is not None and equation.expr is not None:
            results.append(equation)

    return results


def find_models_with_metasymnet(extended_terms, extended_data, test_size=0.2):
    from bitween.metasymnet import metasymnet

    config.precision = 2

    X = extended_data[:, :-1]  # Exclude the constant term
    results = []

    common_locals = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "neg": lambda x: -x,
        "inv": lambda x: 1 / x,
        "pow": lambda x, y: x**y,
    }

    locals = {
        **common_locals,
        "sqrt": sympy.sqrt,
        "log": sympy.log,
        "abs": sympy.Abs,
        "max": sympy.Max,
        "min": sympy.Min,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
    }

    eval_locals = {
        **common_locals,
        "sqrt": math.sqrt,
        "log": math.log,
        "abs": abs,
        "max": max,
        "min": min,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }

    def _regression(target_idx):
        pivot = extended_terms[target_idx]
        terms = [term for term in extended_terms if term != pivot and term != "1"]

        msn_vars_pairs = []
        for i, term in enumerate(terms):
            msn_vars_pairs.append((sympy.Symbol(f"x{i}"), sympy.Symbol(term)))
        y = extended_data[:, target_idx]

        # Exclude the target variable from the features
        X_train, X_test, y_train, y_test = train_test_split(
            np.delete(X, target_idx, axis=1), y, test_size=test_size, random_state=42
        )

        model_name = "MetaSymNet"
        log.info("-" * 80)
        try:
            netstruct = ["s", "x", "x"]
            expr = metasymnet(
                X=np.transpose(X_train),
                Y=y_train,
                netstruct=netstruct,
                iters=10,
            )
        except Exception as e:
            log.error(f"Error during metasymnet: {e}")
            return None

        s_eq = str(expr)
        log.info(f"MetaSymNet: {s_eq}")
        if len(s_eq) > 400:
            log.warning(f"SR expression too long: {s_eq[:400]} ...")
            return None

        rhs = sympy.sympify(s_eq, locals=locals).expand()
        rhs = rhs.subs(msn_vars_pairs)
        log.info(f"MetaSymNet: {pivot} = {rhs}")
        rhs = round_coefficients(rhs, config.precision)
        log.info(f"{pivot} = {rhs}")

        equation = sympy.simplify(sympy.Symbol(pivot) - rhs)
        s_eq = str(equation)
        log.info(s_eq + " = 0")
        if equation == 0:
            log.warning(f"SR found a zero equation for {pivot}")
            return None

        sample_size = X_train.shape[0]
        feature_size = X_train.shape[1]

        try:
            y_pred = []

            for idx in range(X_test.shape[0]):
                namespace = {
                    **eval_locals,
                    **{
                        str(p[0]): X_test[idx, ip]
                        for ip, p in enumerate(msn_vars_pairs)
                    },
                }

                y_val = eval(str(expr), None, namespace)
                y_pred.append(y_val)
        except Exception as e:
            log.error(f"Error computing y_pred: {e}")

        mse = math.inf
        if len(y_pred) == X_test.shape[0]:
            try:
                mse = mean_squared_error(y_test, y_pred)
            except Exception as e:
                log.error(f"Error computing MSE: {e}")

        log.info(f"MSE: {mse}")

        return Equation(equation, mse, pivot, sample_size, model_name, feature_size, "")

    # Create a model for each term in extended_terms, excluding the constant '1'
    for i in range(len(extended_terms) - 1):
        equation = _regression(i)
        if equation is not None and equation.expr is not None:
            results.append(equation)

    return results


def bitween(file_path: str = None) -> tuple[dict, dict, dict]:

    if file_path is None:
        file_path = config.file_path
        if file_path is None:
            raise ValueError("No file path provided")

    st = time()
    samples = 0

    input_data = load_input_data(file_path)
    trace_data = parse_dig_vtrace_file(input_data)

    # TODO: check this
    loc_symbols = {}
    _str = ""
    results = collections.defaultdict(list)
    original_pysr_timeout = config.pysr_timeout

    for loc, data in trace_data.items():
        terms = data["terms"]
        # TODO: check this
        loc_symbols[loc] = {term: sympy.Symbol(term) for term in terms}  # TODO: check
        data = data["data"]
        samples += data.shape[0]

        _str += f"\nLocation: {loc}\n"
        _str += f"Terms: {terms}\n"
        _str += f"Shape: {data.shape}\n"

        initial_degree = 1
        if config.method == Method.FORWARD_SELECTION:
            initial_degree = config.degree

        if config.method == Method.PYSR:
            chunks = len(trace_data) + config.degree + 1 - initial_degree
            config.pysr_timeout = original_pysr_timeout / chunks

        for degree in range(initial_degree, config.degree + 1):
            _str += f"\nDegree: {degree}\n"
            extended_terms, extended_data = process_trace(terms, data, degree)
            trace_data[loc]["extended_terms"] = extended_terms

            _str += f"{extended_terms}; size: {len(extended_terms)}\n"

            if config.method == Method.MULTIPLE_REGRESSION:
                # (Option 1) use cross validation to find the best model for each term
                models = multiple_regression_heuristics(extended_terms, extended_data)
            elif config.method == Method.SIMPLE_REGRESSION:
                # (Option 2) use simple linear regression to find a model for each term
                models = linear_regression_heuristics(extended_terms, extended_data)
            elif config.method == Method.FORWARD_SELECTION:
                # (Option 3) use forward selection to find a model for each term
                models = sfs_heuristics(extended_terms, extended_data, degree)
            elif config.method == Method.EAGER_MILP:
                # (Option 4) for ablation study
                result = find_models_with_eager_milp(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            elif config.method == Method.PYSR:
                result = find_models_with_pysr(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            elif config.method == Method.GPLEARN:
                result = find_models_with_gplearn(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            elif config.method == Method.KAN:
                result = find_models_with_kan(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            elif config.method == Method.RAG_SR:
                result = find_models_with_rag_sr(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            elif config.method == Method.PARFAM:
                result = find_models_with_parfam(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            elif config.method == Method.MetaSymNet:
                result = find_models_with_metasymnet(extended_terms, extended_data)
                results[loc].extend(result)
                continue
            else:
                raise ValueError("Invalid initial method")

            # Display the models and their equations
            for pivot, content in models.items():
                # check if content is a list because of forward selection
                if isinstance(content, list):
                    for c in content:
                        _str += f"Model for {pivot}: Score = {c['score']}, "
                        _str += f"{c['model_type']}({c['params']})\n"
                else:
                    _str += f"Model for {pivot}: Score = {content['score']}, "
                    _str += f"{content['model_type']}({content['params']})\n"

            _str += "\n"

            term_freq = None
            if config.method == Method.FORWARD_SELECTION:
                for pivot, contents in models.items():
                    for content in contents:
                        result, term_freq = infer_equations(
                            {pivot: content}, extended_terms, extended_data
                        )
                        results[loc].extend(result)
                        term_freq.update(term_freq)
            else:
                result, term_freq = infer_equations(
                    models, extended_terms, extended_data
                )

            _str += f"Term Frequencies: {term_freq}; Size: {len(term_freq)}\n\n"
            results[loc].extend(result)
            for r in result:
                _str += r.note

    log.info(_str)

    def sanitize_source(s):
        return (
            s.replace("'", "")
            # .replace("alpha", "a")
            .replace("{", "")
            .replace("}", "")
            .replace("fit_intercept", "intercept")
            .replace("True", "T")
            .replace("False", "F")
        )

    # NOTE: Reporting--Display the inferred equalities
    log.info("\nInferred Equalities:")
    assertion_dict = {}
    error_dict = {}
    sample_dict = {}
    for loc, result in results.items():
        log.info(
            f"\nLocation: {loc}; "
            f"Samples: {trace_data[loc]['data'].shape[0]}; "
            f"Query Functions: {trace_data[loc]['terms']}"
        )
        p_width = config.property_table_width
        # NOTE: a dirty hack to simplify the equation for display
        # based on given function calls and variables
        equations = [eq.expr.evalf() for eq in result]
        # TODO: check this.
        equations = Reducer.find_and_substitute_terms(equations, loc_symbols[loc])
        # NOTE: simplify equations
        equations = [sympy.nsimplify(eq) for eq in equations]
        good_fit = set()
        good_fit_error = []
        samples_used = []
        max_p = 4  # pivot
        max_m = 15  # model
        max_e = 30  # equation
        max_error = 3  # error
        max_s = 2  # sample size
        init_d = str(len(trace_data[loc]["extended_terms"]))  # initial dimension
        max_d = len(init_d)  # dimension
        for i, eq in enumerate(result):
            if eq.error < config.epsilon:
                eql_s = str(equations[i])
                max_m = max(max_m, len(sanitize_source(eq.model_desc)) + 1)
                max_e = max(max_e, len(eql_s) + 4)
                max_error = max(max_error, len(str(round(eq.error, 3))))
                max_p = max(max_p, len(eq.pivot))
                max_s = max(max_s, len(str(eq.sample_size)))
                max_d = max(max_d, len(str(eq.dimension)))
                if len(eql_s) > p_width:
                    max_e = p_width
        ruler = "-" * (max_p + max_m + max_d + max_e + max_error + max_s + 16)
        # print the header
        log.info(ruler)
        log.info(
            f"{'Model':<{max_m}} | "
            f"{'Term':^{max_p}} | "
            f"{init_d:<{max_d}} | "
            f"{'Invariant/Property':<{max_e}} | "
            f"{'Err':<{max_error}} | {'n':^{max_s}} |"
        )
        log.info(ruler)
        for i, eq in enumerate(result):
            # NOTE: add the equation to the good_fit set as long as the error is less
            # than epsilon and the equation is not zero
            if eq.error < config.epsilon and not isinstance(
                equations[i], sympy.core.numbers.Zero
            ):
                s_eq = str(equations[i])
                if len(s_eq) > p_width:
                    s_eq = s_eq[: (p_width - 3)] + "..."
                log.info(
                    f"{sanitize_source(eq.model_desc):<{max_m}} | "
                    f"{eq.pivot:^{max_p}} | "
                    f"{eq.dimension:<{max_d}} | "
                    f"{s_eq:<{max_e}} | "
                    f"{round(eq.error, 3):<{max_error}} | "
                    f"{eq.sample_size:^{max_s}} |"
                )
                good_fit.add(equations[i])  # NOTE: This is very crucial
                good_fit_error.append(eq.error)
                samples_used.append(eq.sample_size)
        log.info(ruler)

        # NOTE: Reductions

        log.info(f"Total time before the last reduction: {time() - st:.2f}s")

        log.info("Reduced Equalities:")
        # TODO: which one is better?
        # equations = Symbolic.refine(good_fit)
        equations = Reducer.merge_equations(list(good_fit))

        # NOTE: relate the location to the equations
        assertion_dict[loc] = equations
        error_dict[loc] = np.mean(good_fit_error) if len(good_fit_error) > 0 else -1.0
        sample_dict[loc] = np.mean(samples_used) if len(samples_used) > 0 else -1.0

        for eq in equations:
            log.info(f"{eq} = 0")

        log.debug(f"Analysis Time: {time() - st:.2f}s")

        if config.slow_simplify:
            equations = Z3._simplify_slow(equations, [], loc)
            for eq in equations:
                log.info(f"{eq} = 0")

        if config.consistency_check:
            log.info("\nChecking Consistency of Equations:")

            try:
                log.info(f"1. Solve algebraically: {sympy.solve(equations)}")
            except NotImplementedError:
                log.info("1. Solve algebraically: Could not solve")

            log.info("2. Check satisfiability: ")
            sat, r = Z3.check_sat(equations)
            log.info(f"{sat}, {r}")

    log.info(f"{samples} samples is used.")

    for loc, eqs in assertion_dict.items():
        for i, eq in enumerate(eqs):
            assertion_dict[loc][i] = sympy.Eq(eq, 0)

    return assertion_dict, error_dict, sample_dict


def _empty_bitween_result() -> tuple[dict, dict, dict]:
    return ({"vtrace1": []}, {"vtrace1": -1.0}, {"vtrace1": -1.0})


def bitween_with_timeout(
    trace_file: str,
    timeout_sec: float = 600.0,
) -> tuple[dict, dict, dict, str]:

    if config.method == Method.PYSR:
        log.warning(
            f"PySR detected: timeout ({timeout_sec}s) is distributed among "
            "PySR regressions due to limiitations"
        )

        try:
            config.pysr_timeout = timeout_sec
            out = bitween(trace_file)

        except Exception as e:
            error_msg = str(e)

    else:
        out, error_msg = TimeoutFunction.call_for(
            timeout_sec=timeout_sec,
            func=bitween,
            args=(trace_file,),
        )

    if error_msg:
        empty_out = _empty_bitween_result()
        res = (*empty_out, error_msg)
    else:
        res = (*out, "")

    return res


def get_vars(template: list[str]):
    vars = set()
    for t in template:
        expr = sympy.sympify(t)
        vars.update(expr.free_symbols)
    return vars


def infer_invariants(
    file_path: str,  # path to the C file
    func_name: str,  # name of the function to infer invariants
    max_degree: int = 2,  # maximum degree
    n: int = 20,  # number of iterations
    epsilon: float = 0.001,  # error threshold
    milp: MILPSolver = None,
    bound: int = None,
    method: Method = Method.MULTIPLE_REGRESSION,
    correctness: Correctness = Correctness.NONE,
):
    """This is prepared for web interface of Bitween"""

    if max_degree > 3:
        raise ValueError("Degree greater than 3 is not supported for web interface")

    config.degree = max_degree
    config.epsilon = epsilon
    if milp:
        config.milp_enabled = True
        config.milp_solver = milp
    else:
        config.milp_enabled = False

    if bound:
        config.bound = bound

    config.method = method

    # Load the vtrace, vassume, and vdistr data
    trace_file = fuzz_and_trace(file_path, func_name, n)

    equations, error, samples = bitween(trace_file)

    if correctness == Correctness.VERIFICATION:
        fuzz_and_verify(file_path, func_name, equations)
    elif correctness == Correctness.FUZZING:
        fuzz_and_check(file_path, func_name, equations, n * 10)
    elif correctness == Correctness.NONE:
        pass
    else:
        raise ValueError("Invalid correctness option")

    return equations, error, samples


def infer_invariants_and_check_correctness(
    file_path: str,  # path to the C file
    func_name: str,  # name of the function to infer invariants
    max_degree: int = 2,  # maximum degree
    n: int = 20,  # number of iterations
    epsilon: float = 0.001,  # error threshold
    milp: MILPSolver = None,
    bound: int = None,
    method: Method = Method.MULTIPLE_REGRESSION,
) -> tuple[dict, dict, dict]:
    """
    Infers invariants from given C program having vtraces, vassumes,
    and vdistrs, and verifies the inferred invariants using symoblic
    execution or fuzzing.
    """

    config.degree = max_degree
    config.epsilon = epsilon
    if milp:
        config.milp_enabled = True
        config.milp_solver = milp
    else:
        config.milp_enabled = False

    if bound:
        config.bound = bound

    config.method = method

    # Load the vtrace, vassume, and vdistr data
    trace_file = fuzz_and_trace(file_path, func_name, n)

    equations, error, samples = bitween(trace_file)

    fuzz_and_check(file_path, func_name, equations, n * 10)

    return equations, error, samples


def infer_invariants_and_verify_correctness(
    file_path: str,  # path to the C file
    func_name: str,  # name of the function to infer invariants
    max_degree: int = 2,  # maximum degree
    n: int = 20,  # number of iterations
    epsilon: float = 0.001,  # error threshold
    milp: MILPSolver = None,
    bound: int = None,
    method: Method = Method.MULTIPLE_REGRESSION,
) -> tuple[dict, dict, dict]:
    config.degree = max_degree
    config.epsilon = epsilon
    if milp:
        config.milp_enabled = True
        config.milp_solver = milp
    else:
        config.milp_enabled = False

    if bound:
        config.bound = bound

    config.method = method

    # Load the vtrace, vassume, and vdistr data
    trace_file = fuzz_and_trace(file_path, func_name, n)

    equations, error, samples = bitween(trace_file)

    fuzz_and_verify(file_path, func_name, equations)

    return equations, error, samples


def infer_property(
    domain: Domain,  # domain of the samples
    distribution: Distribution,  # distribution of the samples
    exprs: list[str],  # function calls to function basis
    template: list[str],  # template for function basis
    functions: list[callable],
    max_degree: int = 2,  # maximum degree
    n: int = 30,  # number of samples
    epsilon: float = 0.001,  # error threshold
    preconditions: dict[str, callable] = None,  # function preconditions
    milp: MILPSolver = None,
    var_bound: int = None,
    method: Method = Method.MULTIPLE_REGRESSION,
    trace_file: str = "trace.csv",
    max_attempts: int = None,
) -> tuple[dict, dict, dict, str]:

    config.degree = max_degree
    config.epsilon = epsilon
    config.method = method
    if milp:
        config.milp_enabled = True
        config.milp_solver = milp
    else:
        config.milp_enabled = False

    if var_bound:
        config.bound = var_bound

    empty_out = _empty_bitween_result()

    # get a dictionary of functions
    functions = {func.__name__: func for func in functions}
    # get the list of variables
    variables = get_vars(template)

    evals = [["vtrace1"] + [f"I {term}" for term in template]]

    curr_iter = 0
    total_iter = 0
    max_attempts = max_attempts or 10 * n

    while curr_iter < n:

        if total_iter == max_attempts:
            error_msg = (
                f"Produced only {curr_iter} samples instead of {n}, "
                f"despite trying {max_attempts} attempts"
            )
            log.error(error_msg)
            return (*empty_out, error_msg)

        total_iter += 1
        variables = sample(distribution, list(variables))
        ok, error_msg = validate_preconditions(
            exprs,
            preconditions,
            variables,
        )
        if not ok:
            log.warn(error_msg)
            continue

        eval_row = ["vtrace1"]
        try:
            for expr in exprs:
                eval_row.append(
                    eval(
                        expr,
                        math.__dict__ | {"sign": np.sign},
                        functions | variables,
                    )
                )

            # if any of the evaluation is None, skip the row
            if None in eval_row:
                continue

            evals.append(eval_row)

        except ZeroDivisionError as e:
            log.error(f"ZeroDivisionError: {e}, {variables}")
            continue

        except ValueError as e:
            log.error(f"ValueError: {e}, {variables}")
            continue

        curr_iter += 1

    log.info(
        f"Produced {len(evals) - 1} samples in " f"{total_iter}/{max_attempts} attempts"
    )

    # Write the data to a CSV file
    with open(trace_file, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(evals)

    return (*bitween(trace_file), "")


def infer_property_with_timeout(
    domain: Domain,
    distribution: Distribution,
    exprs: list[str],
    template: list[str],
    functions: list[callable],
    max_degree: int = 2,
    n: int = 30,
    epsilon: float = 0.001,
    preconditions: dict[str, callable] = None,
    milp: MILPSolver = None,
    var_bound: int = None,
    method: Method = Method.MULTIPLE_REGRESSION,
    trace_file: str = "trace.csv",
    max_attempts: int = None,
    timeout_sec: float = 600.0,
) -> tuple[dict, dict, dict, str]:

    if method == Method.PYSR:
        log.warning(
            f"PySR detected: timeout ({timeout_sec}s) is distributed among "
            "PySR regressions due to limiitations"
        )

        try:
            config.pysr_timeout = timeout_sec
            res = infer_property(
                domain,
                distribution,
                exprs,
                template,
                functions,
                max_degree,
                n,
                epsilon,
                preconditions,
                milp,
                var_bound,
                method,
                trace_file,
                max_attempts,
            )
            error_msg = res[-1]

        except Exception as e:
            error_msg = str(e)

    else:
        res, error_msg = TimeoutFunction.call_for(
            timeout_sec=timeout_sec,
            func=infer_property,
            args=(
                domain,
                distribution,
                exprs,
                template,
                functions,
                max_degree,
                n,
                epsilon,
                preconditions,
                milp,
                var_bound,
                method,
                trace_file,
                max_attempts,
            ),
        )

    if error_msg:
        empty_out = _empty_bitween_result()
        res = (*empty_out, error_msg)

    return res


if __name__ == "__main__":  # noqa E123
    bitween("benchmarks/bitween/dig/bresenham.dig.traces.csv")
