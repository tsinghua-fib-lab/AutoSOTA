from scipy.stats import pearsonr

def correlation_between_length_and_density():
    print("***** Correlation analysis (pearsonr) ******")
    lengths = [1.209, 1.117, 1.117, 1.278, 1.166, 1.788, 1.549, 1.408, 2.249, 2.031, 2.124, 1.304, 2.13, 2.89, 2.443, 2.956, 2.218, 1.73, 1.674, 1.415, 1.867, 2.073, 2.154, 2.401, 2.223, 2.157, 2.426, 1.41, 2.247, 2.842, 3.141, 3.449]
    densities = [0.886, 0.854, 0.86, 0.808, 0.899, 0.798, 0.752, 0.73, 0.713, 0.698, 0.575, 0.772, 0.704, 0.656, 0.484, 0.534, 0.853, 0.808, 0.812, 0.751, 0.875, 0.819, 0.633, 0.717, 0.761, 0.774, 0.752, 0.777, 0.887, 0.729, 0.586, 0.567]
    missing = [1.0 - density for density in densities]
    ratings = [0.907, 0.908, 0.885, 0.76, 0.902, 0.795, 0.838, 0.775, 0.906, 0.778, 0.838, 0.72, 0.893, 0.768, 0.739, 0.752, 0.939, 0.867, 0.897, 0.774, 0.94, 0.865, 0.744, 0.791, 0.838, 0.838, 0.832, 0.642, 0.902, 0.772, 0.662, 0.701]

    p = pearsonr(lengths, missing)
    print(f"Length - Missing Ratio\n\tcorrelation: {p.statistic:.3f}\n\tp-value: {p.pvalue:.3f}")

    p = pearsonr(missing, ratings)
    print(f"Missing Ratio - Rating\n\tcorrelation: {p.statistic:.3f}\n\tp-value: {p.pvalue:.3f}")


if __name__ == '__main__':
    correlation_between_length_and_density()