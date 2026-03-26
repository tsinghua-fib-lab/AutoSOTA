# The Hillstrom dataset from sklift

from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklift.datasets import fetch_hillstrom

from .base import RCTUpliftDatasetCatalog


class HillstromDataset(RCTUpliftDatasetCatalog):
    """
    This is a marketting dataset from Kevin Hillstrom Dataset MineThatData
    (https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics)
    it is a three-arm experiments with 64,000 datapoints and 3 different treatments:

    1. 1/3 were randomly chosen to receive an email campaign featuring men's merchandise
    2. 1/3 were randomly chosen to receive an email campaign featuring women's merchandise
    3. 1/3 were randomly chosen to not receive an email campaign at all

    During a period of two weeks following the e-mail campaign, results were tracked.

    The customer information includes (these are essentially the covariates):

    - Recency: Months since last purchase.
    - History_Segment: Categorization of dollars spent in the past year.
    - Mens 0/1 indicator: 1 = customer purchased Mens merchandise in the past year.
    - Womens 0/1 indicator: 1 = customer purchased Womens merchandise in the past year.
    - Zipcode: Categorization of customer zipcode.
    - Newbie 0/1 indicator: 1 = customer has registered in the past twelve months.
    - Channel: Describes the channels the customer purchased from in the past year.

    The outcome is chosen among the following:

    - Visit 0/1 indicator: 1 = customer visited the website in the following two weeks.
    - Conversion 0/1 indicator: 1 = customer purchased merchandise in the following two weeks.
    - Spend: Actual dollars spent in the following two weeks.

    Now we turn this into a proper evaluation dataset to evaluate a CATE estimator with the purpose
    of evaluating the Qini index. To do so, we turn the base 64,000 datapoints into a suite of 'n_tables'
    tasks where we randomly subsample the rows into tables of size `table_size`. Since this is a three-arm
    experiment, we have three different treatments:
    1. No email (control) vs. email either men's or women's merchandise (treatment)
    2. Email men's merchandise (control) vs. women's merchandise (treatment)

    We also perform a very primitive form of feature preprocessing where we turn the categorical variables into
    ordinal variables using a LabelEncoder. For the History_Segment variable, we turn it into three features:
    1. The category number
    2. The minimum spend
    3. The maximum spend
    """

    ARM_TYPE = Literal["Womens E-Mail", "Mens E-Mail", "No E-Mail"]

    def __init__(
        self,
        seed: int = 42,
        n_folds: int = 5,
        outcome_col: Literal["spend", "conversion", "visit"] = "visit",
        control_arm: Tuple[ARM_TYPE] | ARM_TYPE = "No E-Mail",
        treatment_arm: Tuple[ARM_TYPE] | ARM_TYPE = ("Womens E-Mail", "Mens E-Mail"),
        subsample_max_rows: int | None = None,
    ):
        base_rng = np.random.default_rng(seed)
        self.X_df, self.y_df, self.t_df = fetch_hillstrom(return_X_y_t=True, target_col=outcome_col)

        # (1) handle covariates
        all_X: pd.DataFrame = self.X_df.copy()
        for col in ["zip_code", "channel"]:
            le = LabelEncoder()
            all_X[col] = le.fit_transform(self.X_df[col].astype(str))

        def parse(history_segment: str):
            # parse a string of style "%d) $%d - $%d" into a tuple of (item_id, price, quantity)
            if "-" not in history_segment:
                parts = history_segment.split("+")
                parts[0] = parts[0].strip()
                item_id = int(parts[0].split(") ")[0])
                left = float(parts[0].split("$")[1].replace(",", ""))
                right = left * 2
            else:
                parts = history_segment.split(" - ")
                item_id = int(parts[0].split(") ")[0])
                left = float(parts[0].split("$")[1].replace(",", ""))
                right = float(parts[1].split("$")[1].replace(",", ""))
            return item_id, left, right

        # run parse on the 'history' column in X1 and create new columns for item_id, price, quantity
        all_X["history_segment"] = all_X["history_segment"].apply(lambda x: parse(x))
        all_X["item_id"] = all_X["history_segment"].apply(lambda x: x[0])
        all_X["price"] = all_X["history_segment"].apply(lambda x: x[1])
        all_X["quantity"] = all_X["history_segment"].apply(lambda x: x[2])
        all_X = all_X.drop(columns=["history_segment"])
        all_X: np.ndarray = all_X.values

        # (2) handle outcomes
        all_y = self.y_df.values.copy()

        # (3) handle treatments
        if isinstance(control_arm, str):
            control_arm = (control_arm,)
        if isinstance(treatment_arm, str):
            treatment_arm = (treatment_arm,)
        control_arm = set(list(control_arm))
        treatment_arm = set(list(treatment_arm))
        if len(control_arm.intersection(treatment_arm)) > 0:
            raise ValueError("Control and treatment arms cannot intersect")

        def decide(s: str):
            if s in control_arm:
                return 0
            elif s in treatment_arm:
                return 1
            else:
                return -1

        map_dict = {
            "Womens E-Mail": decide("Womens E-Mail"),
            "Mens E-Mail": decide("Mens E-Mail"),
            "No E-Mail": decide("No E-Mail"),
        }
        t_encoded = self.t_df.astype(str).replace(map_dict).values
        valid_rows = np.where(t_encoded > -1, True, False)
        all_t = t_encoded.astype(np.float32)

        # (4) remove invalid rows
        all_X = all_X[valid_rows].astype(np.float32)
        all_y = all_y[valid_rows].astype(np.float32)
        all_t = all_t[valid_rows].astype(np.float32)

        super().__init__(
            n_folds=n_folds,
            name="Hillsrom",
            all_X=all_X,
            all_y=all_y,
            all_t=all_t,
            base_rng=base_rng,
            subsample_max_rows=subsample_max_rows,
        )
