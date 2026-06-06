# Description: Statistical tests for suitability filter
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# Statistical tests and helper functions


def t_test(sample1, sample2, equal_var=False):
    """
    Perform a two-sample t-test for two samples.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    equal_var: if False, uses Welch's t-test.
    Returns: t-statistic, p-value
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return {"t_statistic": t_stat, "p_value": p_value}


def non_inferiority_ztest(array1, array2, margin=0, increase_good=True, alpha=0.05):
    """
    Perform a non-inferiority z-test for two arrays.
    Use when: large sample size, approximately normal data distribution, assumes known population variances
    array1: array of values for sample 1
    array2: array of values for sample 2
    margin: non-inferiority margin (threshold for difference in means)
    increase_good: if True, Ho: mean2 <= mean1 - threshold. Else Ho: mean2 >= mean1 + threshold.
    alpha: significance level
    Returns: mean_diff, z_score, p_value, reject_null
    """

    # Calculate the mean and standard deviation of both arrays
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    std1 = np.std(array1, ddof=1)
    std2 = np.std(array2, ddof=1)

    # Calculate the difference in means
    if increase_good:
        mean_diff = mean1 - mean2
    else:
        mean_diff = mean2 - mean1

    # Calculate the standard error of the difference
    se_diff = np.sqrt((std1**2 / len(array1)) + (std2**2 / len(array2)))

    # Calculate the Z-score
    z_score = (mean_diff - margin) / se_diff

    # Calculate the p-value
    p_value = stats.norm.cdf(z_score)

    return {
        "mean_diff": mean_diff,
        "z_score": z_score,
        "p_value": p_value,
        "reject_null": p_value < alpha,
    }


def non_inferiority_ttest(
    sample1, sample2, margin=0, increase_good=True, equal_var=False, alpha=0.05
):
    """
    Perform a non-inferiority t-test for two samples.
    Use when: small sample size, unequal population variances, adjusts for dof, accounts for sample size differences
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    margin: non-inferiority margin (threshold for difference in means)
    equal_var: if False, uses Welch's t-test.
    increase_good: if True, Ho: mean2 <= mean1 - threshold. Else Ho: mean2 >= mean1 + threshold.
    Returns: t_statistic, p_value, reject_null
    """
    if increase_good:
        sample2_diff = sample2 + margin
    else:
        sample2_diff = sample2 - margin

    # Perform two-sided t-test
    t_stat, p_value_two_sided = stats.ttest_ind(
        sample1, sample2_diff, equal_var=equal_var
    )

    is_neg = t_stat < 0

    # Adjust for one-sided test (upper-tailed)
    if increase_good and is_neg or not increase_good and not is_neg:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (p_value_two_sided / 2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value_one_sided,
        "reject_null": p_value_one_sided < alpha,
    }


def satterthwaite_dof(s1, n1, s2, n2):
    """Calculate the Satterthwaite degrees of freedom."""
    numerator = (s1**2 / n1 + s2**2 / n2) ** 2
    denominator = ((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1))
    return numerator / denominator


def equivalence_test(sample1, sample2, threshold_low, threshold_upp, equal_var=False):
    """
    Perform a corrected custom TOST with Satterthwaite's degrees of freedom.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    threshold_low: lower bound of the equivalence interval.
    threshold_upp: upper bound of the equivalence interval.
    equal_var: if False, uses Welch's t-test.
    Returns: t-statistic and p-values for the lower and upper bound tests, and the degrees of freedom.
    """

    # Calculate means and standard deviations
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    n1, n2 = len(sample1), len(sample2)
    mean_diff = mean1 - mean2
    low = threshold_low
    upp = threshold_upp

    # Calculate standard error of the difference
    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)

    # Satterthwaite's degrees of freedom
    dof = satterthwaite_dof(std1, n1, std2, n2)

    # Lower bound test
    t_stat_low = (mean_diff - low) / se_diff
    p_value_low = 1 - stats.t.cdf(t_stat_low, df=dof)

    # Upper bound test
    t_stat_upp = (mean_diff - upp) / se_diff
    p_value_upp = stats.t.cdf(t_stat_upp, df=dof)

    # Return the results
    return {
        "t_statistic_low": t_stat_low,
        "p_value_low": p_value_low,
        "t_statistic_upp": t_stat_upp,
        "p_value_upp": p_value_upp,
        "dof": dof,
    }


# Combining p-values from tests
def harmonic_mean_pvalue(p_values):
    """
    Calculate the Harmonic Mean p-value (HMP) from an array of p-values.

    Parameters:
    p_values (array-like): A list or numpy array of p-values to combine.

    Returns:
    float: The combined Harmonic Mean p-value.
    """
    p_values = np.array(p_values)

    if np.any(p_values <= 0) or np.any(p_values > 1):
        raise ValueError("P-values should be in the range (0, 1].")

    # Calculate Harmonic Mean p-value
    k = len(p_values)
    hmp = k / np.sum(1.0 / p_values)

    return hmp


def stouffer_zscore(p_values, weights=None):
    """
    Calculate the combined p-value using Stouffer's Z-score method.

    Parameters:
    p_values (array-like): List of p-values to combine.
    weights (array-like, optional): Weights for each p-value, typically related to sample size.

    Returns:
    float: Combined p-value.
    """
    # Convert p-values to Z-scores
    z_scores = stats.norm.ppf(1 - np.array(p_values))

    # If no weights are provided, use equal weights
    if weights is None:
        weights = np.ones_like(p_values)

    # Calculate combined Z-score
    combined_z = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))

    # Convert combined Z-score back to a p-value
    combined_p_value = 1 - stats.norm.cdf(combined_z)

    return combined_p_value


def glass_delta(sample1, sample2, margin, increase_good=True):
    """
    Calculate Glass's delta effect size for two samples.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    margin: non-inferiority margin
    increase_good: if True, Ho: mean2 <= mean1. Else Ho: mean2 >= mean1.
    Returns: Glass's delta effect size
    """
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1 = np.std(sample1, ddof=1)

    if increase_good:
        glass_delta = (mean2 - (mean1 - margin)) / std1
    else:
        glass_delta = (mean2 - (mean1 + margin)) / std1

    return glass_delta


def cohen_d(sample1, sample2, margin, increase_good=True):
    """
    Calculate Cohen's d effect size for two samples.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    margin: non-inferiority margin
    increase_good: if True, Ho: mean2 <= mean1. Else Ho: mean2 >= mean1.
    Returns: Cohen's d effect size
    """
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    n1, n2 = len(sample1), len(sample2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if increase_good:
        cohens_d = (mean2 - (mean1 - margin)) / pooled_std
    else:
        cohens_d = (mean2 - (mean1 + margin)) / pooled_std

    return cohens_d


def sample_size_non_inferiority_ttest(
    sample1,
    sample2,
    power,
    margin=0,
    increase_good=True,
    alpha=0.05,
    effect_method="cohen",
):
    """
    Calculate the required sample size for a given power level.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    power: desired power level (probability of rejecting the null hypothesis when it is false)
    margin: non-inferiority margin
    increase_good: if True, Ho: mean2 <= mean1. Else Ho: mean2 >= mean1.
    alpha: significance level
    effect_method: method to calculate effect size. Options: "cohen" or "glass"
    Returns: required sample size for sample2
    """
    if effect_method == "cohen":
        effect_size = cohen_d(sample1, sample2, margin, increase_good)
    elif effect_method == "glass":
        effect_size = glass_delta(sample1, sample2, margin, increase_good)
    else:
        raise ValueError("Invalid effect size method.")

    if abs(effect_size) < 1e-6:
        raise ValueError("Effect size too small for power analysis.")

    analysis = TTestIndPower()

    ratio = analysis.solve_power(
        effect_size=effect_size,
        nobs1=len(sample1),
        alpha=alpha,
        power=power,
        alternative="larger" if increase_good else "smaller",
        ratio=None,
    )

    return int(np.ceil(len(sample1) * ratio))


def power_non_inferiority_ttest(
    sample1, sample2, margin=0, alpha=0.05, increase_good=True, effect_method="cohen"
):
    """
    Calculate the power of a non-inferiority t-test.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    margin: non-inferiority margin
    alpha: significance level
    increase_good: if True, Ho: mean2 <= mean1. Else Ho: mean2 >= mean1.
    effect_method: method to calculate effect size. Options: "cohen" or "glass"
    Returns: power of the non-inferiority t-test
    """
    if effect_method == "cohen":
        effect_size = cohen_d(sample1, sample2, margin, increase_good)
    elif effect_method == "glass":
        effect_size = glass_delta(sample1, sample2, margin, increase_good)
    else:
        raise ValueError("Invalid effect size method.")

    if abs(effect_size) < 1e-6:
        raise ValueError("Effect size too small for power analysis.")

    analysis = TTestIndPower()

    power = analysis.solve_power(
        effect_size=effect_size,
        nobs1=len(sample1),
        alpha=alpha,
        alternative="larger" if increase_good else "smaller",
        ratio=len(sample2) / len(sample1),
    )

    return power


def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


def weighted_variance(data, weights, weighted_mean):
    return np.sum(weights * (data - weighted_mean) ** 2) / np.sum(weights)


def non_inferiority_weighted_ttest(
    sample1, weights1, sample2, weights2, margin=0, increase_good=True, alpha=0.05
):
    """
    Perform a non-inferiority t-test for two weighted samples.

    sample1, sample2: arrays of values for samples 1 and 2
    weights1, weights2: weights associated with sample1 and sample2 respectively
    margin: non-inferiority margin (threshold for difference in means)
    increase_good: if True, Ho: mean2 <= mean1 - threshold. Else Ho: mean2 >= mean1 + threshold.
    alpha: significance level

    Returns: t_statistic, p_value, reject_null
    """
    # Calculate weighted means
    mean1 = weighted_mean(sample1, weights1)
    mean2 = weighted_mean(sample2, weights2)

    # Adjust for non-inferiority margin
    if increase_good:
        mean2_diff = mean2 + margin
    else:
        mean2_diff = mean2 - margin

    # Calculate weighted variances
    var1 = weighted_variance(sample1, weights1, mean1)
    var2 = weighted_variance(sample2, weights2, mean2)

    # Calculate effective sample sizes for each sample
    n1 = np.sum(weights1) ** 2 / np.sum(weights1**2)
    n2 = np.sum(weights2) ** 2 / np.sum(weights2**2)

    # Calculate the weighted t-statistic
    pooled_se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2_diff) / pooled_se

    # Degrees of freedom for Welch's t-test (since variances are likely unequal)
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1**2) / (n1**2 * (n1 - 1)) + (var2**2) / (n2**2 * (n2 - 1))
    )

    # Calculate the one-sided p-value
    p_value_two_sided = 2 * stats.t.cdf(-abs(t_stat), df)
    is_neg = t_stat < 0

    if increase_good and is_neg or not increase_good and not is_neg:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (p_value_two_sided / 2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value_one_sided,
        "reject_null": p_value_one_sided < alpha,
    }
