import pandas as pd
import numpy as np


def subsample_to_current_us_distribution(
    df,
    income_col="income_bracket",
    education_col="education_level",
    random_state=42,
):
    """
    Subsample a dataframe so that the joint distribution of
    (income bracket, education level) approximately matches
    current U.S. distributions.

    Assumptions:
    - Education targets are based on 2024 CPS educational-attainment totals.
    - Income targets are a percentile-style mapping for your custom labels:
        Very Low      = bottom 20%
        Low           = next 20%
        Lower-Middle  = next 15%
        Middle        = next 15%
        Upper-Middle  = next 15%
        High          = next 10%
        Very High     = top 5%
    - Because your education labels include MBA and Postdoc, which are not
      standard standalone CPS categories, the "advanced degree" mass is split
      across ['Master’s', 'MBA', 'PhD', 'Postdoc'] in proportion to their
      prevalence in your dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the two columns.
    income_col : str
        Column name for income bracket.
    education_col : str
        Column name for education level.
    random_state : int
        RNG seed.

    Returns
    -------
    sampled_df : pandas.DataFrame
        Subsampled dataframe.
    diagnostics : dict
        Useful objects: target marginals, observed table, target table, etc.
    """
    rng = np.random.default_rng(random_state)

    # Keep only rows where both variables are present and recognized
    valid_income = [
        "Upper-Middle",
        "Middle",
        "Very Low",
        "Low",
        "High",
        "Very High",
        "Lower-Middle",
    ]
    valid_edu = [
        "No Formal Education",
        "Bachelor’s",
        "Postdoc",
        "Master’s",
        "Diploma",
        "High School",
        "MBA",
        "PhD",
        "Associate’s",
    ]

    x = df.copy()
    x = x[
        x[income_col].isin(valid_income) & x[education_col].isin(valid_edu)
    ].copy()

    if x.empty:
        raise ValueError(
            "No rows remain after filtering to recognized income/education"
            "categories."
        )

    # ------------------------------------------------------------------
    # 1) TARGET EDUCATION DISTRIBUTION
    # ------------------------------------------------------------------
    # Based on CPS 2024 Table 2 (Both Sexes, 25+), mapped to your labels.
    #
    # Direct CPS-style categories:
    # - No Formal Education = None-8th + 9th-11th
    # - High School = High school graduate
    # - Diploma = Some college, no degree
    # - Associate’s = Associate's degree
    # - Bachelor’s = Bachelor's degree
    #
    # Advanced degrees in CPS are Master's + Professional + Doctoral.
    # Your labels split this into Master’s / MBA / PhD / Postdoc, so we
    # allocate that advanced-degree mass in proportion to how those 4 labels
    # appear in your dataframe.
    #
    # 2024 CPS counts (thousands), Both Sexes, age 25+:
    # None-8th grade      =  8043
    # 9th-11th grade      = 11600
    # High school grad    = 64010
    # Some college        = 32170
    # Associate's         = 25060
    # Bachelor's          = 54560
    # Master's            = 26010
    # Professional        =  3211
    # Doctoral            =  5113
    # Total               = 229800
    #
    fixed_edu_counts = {
        "No Formal Education": 8043 + 11600,
        "High School": 64010,
        "Diploma": 32170,
        "Associate’s": 25060,
        "Bachelor’s": 54560,
    }

    advanced_total = 26010 + 3211 + 5113  # Master's + Professional + Doctoral

    advanced_labels = ["Master’s", "MBA", "PhD", "Postdoc"]
    advanced_counts_in_df = (
        x[education_col]
        .value_counts()
        .reindex(advanced_labels, fill_value=0)
        .astype(float)
    )

    if advanced_counts_in_df.sum() == 0:
        # If the dataframe has none of these, put all advanced mass on Master’s
        advanced_shares = pd.Series(
            [1.0, 0.0, 0.0, 0.0], index=advanced_labels
        )
    else:
        advanced_shares = advanced_counts_in_df / advanced_counts_in_df.sum()

    edu_target = pd.Series(fixed_edu_counts, dtype=float)
    edu_target = pd.concat([edu_target, advanced_shares * advanced_total])
    edu_target = edu_target.reindex(valid_edu, fill_value=0.0)
    edu_target = edu_target / edu_target.sum()

    # ------------------------------------------------------------------
    # 2) TARGET INCOME DISTRIBUTION
    # ------------------------------------------------------------------
    # Your income labels are custom, so use a percentile-style mapping
    # consistent with CPS quintiles + top 5%.
    income_target = pd.Series(
        {
            "Very Low": 0.20,
            "Low": 0.20,
            "Lower-Middle": 0.15,
            "Middle": 0.15,
            "Upper-Middle": 0.15,
            "High": 0.10,
            "Very High": 0.05,
        },
        dtype=float,
    )
    income_target = income_target.reindex(valid_income)

    # ------------------------------------------------------------------
    # 3) OBSERVED CONTINGENCY TABLE
    # ------------------------------------------------------------------
    obs = (
        pd.crosstab(x[income_col], x[education_col])
        .reindex(index=valid_income, columns=valid_edu, fill_value=0)
        .astype(float)
    )

    if obs.values.sum() == 0:
        raise ValueError("Observed contingency table is empty.")

    # ------------------------------------------------------------------
    # 4) ITERATIVE PROPORTIONAL FITTING (RAKING)
    # ------------------------------------------------------------------
    def ipf(seed_table, target_rows, target_cols, max_iter=5000, tol=1e-10):
        m = seed_table.astype(float).copy()

        # structural zeros stay zero
        zero_mask = m == 0

        # start from positive seed for nonzero cells
        if m.values.sum() == 0:
            raise ValueError("Seed table sums to zero.")

        m = m / m.sum()

        for _ in range(max_iter):
            old = m.copy()

            # row scaling
            row_sums = m.sum(axis=1)
            row_factors = np.divide(
                target_rows.values,
                row_sums.values,
                out=np.ones_like(target_rows.values, dtype=float),
                where=row_sums.values > 0,
            )
            m = (m.T * row_factors).T

            # col scaling
            col_sums = m.sum(axis=0)
            col_factors = np.divide(
                target_cols.values,
                col_sums.values,
                out=np.ones_like(target_cols.values, dtype=float),
                where=col_sums.values > 0,
            )
            m = m * col_factors

            # restore structural zeros
            m[zero_mask] = 0.0

            # renormalize
            s = m.values.sum()
            if s > 0:
                m /= s

            if np.max(np.abs(m.values - old.values)) < tol:
                break

        return m

    target_prob_table = ipf(obs, income_target, edu_target)

    # ------------------------------------------------------------------
    # 5) SCALE TO A FEASIBLE SAMPLE SIZE
    # ------------------------------------------------------------------
    # Need target counts <= observed counts in every populated cell.
    positive_mask = target_prob_table > 0
    feasible_scale = np.min(
        obs[positive_mask] / target_prob_table[positive_mask]
    )
    feasible_n = int(np.floor(feasible_scale))

    if feasible_n <= 0:
        raise ValueError("Could not find a feasible positive subsample size.")

    target_counts_float = target_prob_table * feasible_n
    target_counts = np.floor(target_counts_float).astype(int)

    # Largest-remainder top-up, while respecting observed availability
    remainder = (
        (target_counts_float - target_counts)
        .stack()
        .sort_values(ascending=False)
    )
    shortfall = feasible_n - int(target_counts.values.sum())

    for (inc, edu), _ in remainder.items():
        if shortfall <= 0:
            break
        if target_counts.loc[inc, edu] < obs.loc[inc, edu]:
            target_counts.loc[inc, edu] += 1
            shortfall -= 1

    # ------------------------------------------------------------------
    # 6) SAMPLE WITHOUT REPLACEMENT WITHIN EACH CELL
    # ------------------------------------------------------------------
    sampled_parts = []
    for inc in valid_income:
        for edu in valid_edu:
            n_take = int(target_counts.loc[inc, edu])
            if n_take <= 0:
                continue

            pool = x[(x[income_col] == inc) & (x[education_col] == edu)]
            if len(pool) < n_take:
                # safety fallback; should not happen after feasibility scaling
                n_take = len(pool)

            chosen_idx = rng.choice(
                pool.index.to_numpy(), size=n_take, replace=False
            )
            sampled_parts.append(x.loc[chosen_idx])

    sampled_df = (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    diagnostics = {
        "target_income_distribution": income_target,
        "target_education_distribution": edu_target,
        "observed_table": obs,
        "target_probability_table": target_prob_table,
        "target_counts": target_counts,
        "sampled_table": pd.crosstab(
            sampled_df[income_col], sampled_df[education_col]
        ).reindex(index=valid_income, columns=valid_edu, fill_value=0),
        "sampled_n": len(sampled_df),
    }

    return sampled_df, diagnostics
