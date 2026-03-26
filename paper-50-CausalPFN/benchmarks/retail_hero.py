import gc
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from sklift.datasets import fetch_x5

from .base import RCTUpliftDatasetCatalog


class X5Dataset(RCTUpliftDatasetCatalog):
    """
    Uplift X5 dataset (https://ods.ai/competitions/x5-retailhero-uplift-modeling/data)
    The features contain client ids and enrollment dates, but it also contains a larger
    transactions table. We use the feature extraction method from:
    (https://nbviewer.org/github/kirrlix1994/Retail_hero/blob/master/Retail_hero_contest_2nd_place_solution.ipynb)
    to merge and aggregate the features into a single table.

    The dataset contains a binary target variable (conversion) and a binary treatment variable (treatment)
    whether or not the customer has been enrolled in a marketing campaign.
    """

    def __init__(
        self,
        seed: int = 42,
        n_folds: int = 5,
        subsample_max_rows: int | None = None,
    ):
        base_rng = np.random.default_rng(seed)
        dataset = fetch_x5()
        data, target, treatment = dataset.data, dataset.target, dataset.treatment
        df_clients = data["clients"].copy()
        df_purchases = data["purchases"].copy()
        df_train = data["train"].copy()
        # parse a column in df_clients
        for col in ["first_redeem_date", "first_issue_date"]:
            df_clients[col] = pd.to_datetime(df_clients[col], format="%Y-%m-%d %H:%M:%S")
            df_clients[col] = df_clients[col].fillna(datetime(2019, 3, 19, 0, 0))
            df_clients.loc[:, col] = (df_clients[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1d")

        # two -level aggregation:

        # 1. aggregate to (client, transaction) take one row only
        #  ('express_points_spent', 'purchase_sum' - same for one transaction)

        # 2. aggregate all transactions to client.

        df_purch = (
            df_purchases.groupby(["client_id", "transaction_id"])[["express_points_spent", "purchase_sum"]]
            .last()
            .groupby("client_id")
            .agg({"express_points_spent": ["mean", "sum"], "purchase_sum": ["sum"]})
        )

        # set readable column names:
        df_purch.columns = ["express_spent_mean", "express_points_spent_sum", "purchase_sum__sum"]

        #'regular_points_received_sum_last_m'

        reg_points_last_m = (
            df_purchases[df_purchases["transaction_datetime"] > "2019-02-18"]
            .groupby(["client_id", "transaction_id"])["regular_points_received"]
            .last()
            .groupby("client_id")
            .sum()
        )

        reg_points_last_m = pd.DataFrame({"regular_points_received_sum_last_m": reg_points_last_m})

        df_purchases["transaction_datetime"] = pd.to_datetime(
            df_purchases["transaction_datetime"], format="%Y-%m-%d %H:%M:%S"
        )

        df_purchases.loc[:, "purch_day"] = (
            df_purchases["transaction_datetime"] - pd.Timestamp("1970-01-01")
        ) // pd.Timedelta("1d")
        # 'after_redeem_sum'

        df_purch_joined = pd.merge(
            df_purchases[["client_id", "purch_day", "transaction_id", "purchase_sum"]],
            df_clients.reset_index()[["client_id", "first_redeem_date"]],
            on="client_id",
            how="left",
        )

        df_purch_joined = df_purch_joined.assign(
            date_diff=df_purch_joined["first_redeem_date"] - df_purch_joined["purch_day"]
        )

        df_purch_agg = (
            df_purch_joined[df_purch_joined["date_diff"] <= 0]
            .groupby(["client_id", "transaction_id"])
            .last()
            .groupby("client_id")["purchase_sum"]
            .sum()
        )

        after_redeem_sum = pd.DataFrame(data={"after_redeem_sum": df_purch_agg})

        del df_purch_joined, df_purch_agg
        gc.collect()

        df_purch_delta_agg = df_purchases.groupby("client_id").agg({"purch_day": ["max", "min"]})

        df_purch_delta = pd.DataFrame(
            data=df_purch_delta_agg["purch_day"]["max"] - df_purch_delta_agg["purch_day"]["min"] + 1,
            columns=["purch_delta"],
        )

        del df_purch_delta_agg
        gc.collect()

        df_feats = pd.concat(
            [
                df_clients.set_index("client_id")[["first_redeem_date"]],
                df_purch,
                df_purch_delta,
                reg_points_last_m,
                after_redeem_sum,
            ],
            axis=1,
            sort=False,
        )

        df_feats = df_feats.assign(
            avg_spent_perday=df_feats["purchase_sum__sum"] / df_feats["purch_delta"],
            after_redeem_sum_perday=df_feats["after_redeem_sum"] / df_feats["purch_delta"],
        ).drop(["purch_delta", "purchase_sum__sum", "after_redeem_sum"], axis=1)

        # replace the Nan with average
        df_feats = df_feats.fillna(df_feats.mean())

        merged_df = pd.merge(df_train, df_feats, on="client_id", how="left")
        # drop the client_id
        merged_df = merged_df.drop(["client_id"], axis=1)
        all_X = merged_df.values.astype("float32")
        all_y = target.values.astype("float32")
        all_t = treatment.values.astype("float32")

        super().__init__(
            name="X5",
            all_X=all_X,
            all_y=all_y,
            all_t=all_t,
            base_rng=base_rng,
            n_folds=n_folds,
            subsample_max_rows=subsample_max_rows,
        )
