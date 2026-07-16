# Theta Estimation Update for p-IRT Scripts

## Summary

The current p-IRT scripts (`pirt_tinybenchmark.r`, `pirt_metabench.r`, `pirt_random_100.r`) use a simplified `estimate_theta_simple()` function that performs basic logit transformation. 

The proper approach (used in `metabench_theta.r`, `tinyarc_theta.r`, `random_100.r`) uses the `mirt` package with EAP (Expected A Posteriori) method for more accurate theta estimation.

## Current Implementation (Simplified)

```r
estimate_theta_simple <- function(responses, item_params_subset) {
  # Simple logit transformation
  p_obs <- mean(responses, na.rm = TRUE)
  avg_c <- mean(item_params_subset$c, na.rm = TRUE)
  p_adj <- (p_obs - avg_c) / (1 - avg_c)
  theta <- log(p_adj / (1 - p_adj))
  return(theta)
}
```

**Issues:**
- Not true IRT estimation
- Doesn't account for item information
- Less accurate for extreme scores
- No standard error

## Proper Implementation (mirt-based)

```r
source("theta_estimation_utils.r")

# Replace estimate_theta_simple() with:
result <- estimate_theta_mirt(responses, item_params_subset)
theta <- result$theta
se <- result$se
```

**Advantages:**
- Uses full 3PL IRT model
- Accounts for item discrimination and difficulty
- EAP method with proper prior
- Provides standard errors
- Handles edge cases robustly

## Files to Update

### 1. `pirt_tinybenchmark.r`

**Line ~162** (in `compute_pirt_accuracy` function):
```r
# OLD:
theta_l <- estimate_theta_simple(subset_responses, item_params_subset)

# NEW:
theta_result <- estimate_theta_mirt(subset_responses, item_params_subset)
theta_l <- theta_result$theta
```

### 2. `pirt_metabench.r`

**Line ~166** (in `compute_pirt_accuracy` function):
```r
# OLD:
theta_l <- estimate_theta_simple(subset_responses, item_params_subset)

# NEW:
theta_result <- estimate_theta_mirt(subset_responses, item_params_subset)
theta_l <- theta_result$theta
```

### 3. `pirt_random_100.r`

**Line ~172** (in `compute_pirt_accuracy` function):
```r
# OLD:
theta_l <- estimate_theta_simple(subset_responses, item_params_subset)

# NEW:
theta_result <- estimate_theta_mirt(subset_responses, item_params_subset)
theta_l <- theta_result$theta
```

## Additional Changes Required

### Add library loading at top of each script:

```r
# Load theta estimation utilities
source(paste0(BASE_DIR, "/theta_estimation_utils.r"))
```

### Remove the old estimate_theta_simple function

Delete the entire function definition from each script (currently ~80-130 lines depending on script).

## Testing

After updates, test with:

```bash
# Test TinyBenchmark
Rscript pirt_tinybenchmark.r --benchmark=arc

# Test Metabench  
Rscript pirt_metabench.r --benchmark=arc --version=secondary

# Test Random 100
Rscript pirt_random_100.r --benchmark=arc
```

Check that:
1. Scripts run without errors
2. Theta values are reasonable (-3 to 3 range)
3. `avg_predicted` values are NOT 0.5 (unless truly no parameters)
4. Results are similar or better than simplified method

## Expected Improvements

- **More accurate theta estimates** especially for extreme scorers
- **Better predictions** for unobserved items
- **Lower RMSE** when comparing pirt_accuracy vs actual_accuracy
- **Proper standard errors** (can be used for future confidence intervals)

## Rollback Plan

If mirt-based estimation causes issues:
1. Keep the original `estimate_theta_simple()` function
2. Add it as a fallback in `theta_estimation_utils.r`
3. The current implementation already has this fallback built-in

## Notes

- The `mirt` package must be installed: `install.packages("mirt")`
- Estimation will be slightly slower (uses EM algorithm)
- For large datasets, consider caching theta estimates
- The fallback to simple method is automatic if mirt fails
