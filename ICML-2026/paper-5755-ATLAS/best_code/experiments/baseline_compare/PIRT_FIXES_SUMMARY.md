# p-IRT Scripts Fixes Summary

## Issues Fixed

### 1. Item ID Format Mismatch (All Scripts)
**Problem**: Item parameters had "X" prefix (e.g., `X1`, `X2`) while response matrices had no prefix (e.g., `1`, `2`), causing unobserved item predictions to default to 0.5.

**Fix**: Added ID normalization in all three scripts:
```r
# Normalize item IDs: remove "X" prefix if present
item_ids <- gsub("^X", "", item_ids)
```

**Files Fixed**:
- `pirt_metabench.r` (line 47)
- `pirt_random_100.r` (line 43)
- `pirt_tinybenchmark.r` (line 46)

---

### 2. TinyBenchmark Response Matrix (pirt_tinybenchmark.r)
**Problem**: Script used calibration response matrix with 3,747 models instead of test set with 417 models.

**Before**:
```r
RESPONSE_FILE <- paste0(BASE_DIR, "/tinybenchmark_calibration/response_matrix_tiny", BENCHMARK, ".csv")
```

**After**:
```r
RESPONSE_FILE <- paste0("/store01/nchawla/pli9/llmbenchmark/select_model/gaussian_sampled_", BENCHMARK, "_response_matrix_test.csv")
```

**Impact**: Now uses same test set as metabench and random_100 for fair comparison.

---

### 3. TinyBenchmark n_all_items Bug (pirt_tinybenchmark.r)
**Problem**: Script treated 100 TinyBenchmark items as the full dataset, resulting in:
- `n_all_items = 100` (should be 844 for ARC)
- `n_unobserved_items = 0` (should be 744)
- `avg_predicted = 0` (no prediction needed)

**Fix**: Changed to use full item parameters for n_all_items:
```r
# Before
n_all_items <- length(subset_item_ids)  # 100

# After
n_all_items <- length(item_params$item_id)  # 844
```

---

### 4. Theta Estimation Method (All Scripts)
**Problem**: Used simplified `estimate_theta_simple()` with basic logit transformation instead of proper IRT estimation.

**Fix**: 
1. Created `theta_estimation_utils.r` with mirt-based EAP estimation
2. Updated all scripts to use `estimate_theta_mirt()`

**Before**:
```r
theta_l <- estimate_theta_simple(subset_responses, item_params_subset)
```

**After**:
```r
theta_result <- estimate_theta_mirt(subset_responses, item_params_subset)
theta_l <- theta_result$theta
```

**Files Updated**:
- `pirt_tinybenchmark.r` (line 214)
- `pirt_metabench.r` (line 178)
- `pirt_random_100.r` (line 179)

---

### 5. TinyBenchmark Item Selection (pirt_tinybenchmark.r)
**Problem**: Script didn't properly load and filter TinyBenchmark selected items from the full response matrix.

**Fix**: Added proper item loading and column matching:
```r
# Load selected items
selected_items_df <- read.csv(SELECTED_ITEMS_FILE, stringsAsFactors = FALSE)
selected_item_ids <- as.character(selected_items_df$item_id)

# Match in response matrix with normalization
selected_normalized <- gsub("^X", "", gsub("[^0-9A-Za-z_]", "", selected_item_ids))
response_normalized <- gsub("^X", "", gsub("[^0-9A-Za-z_]", "", all_response_item_ids))
subset_cols <- which(response_normalized %in% selected_normalized)

# Extract only selected columns
subset_responses <- as.numeric(response_matrix[model_idx, subset_cols + 1])
```

---

## Summary of Expected Improvements

### Before Fixes:
- ❌ `avg_predicted = 0.5` for metabench/random_100 (all unobserved items missing)
- ❌ `avg_predicted = 0` for tinybenchmark (wrong calculation)
- ❌ TinyBenchmark used wrong response matrix (3,747 models vs 417)
- ❌ Simplified theta estimation (not true IRT)

### After Fixes:
- ✅ `avg_predicted` uses proper IRT predictions (varies by model)
- ✅ TinyBenchmark correctly predicts on 744 unobserved items
- ✅ All scripts use same 417-model test set
- ✅ Proper mirt-based EAP theta estimation
- ✅ Item ID matching works correctly

---

## Files Created/Modified

### Created:
1. `theta_estimation_utils.r` - Shared mirt-based theta estimation
2. `THETA_ESTIMATION_UPDATE.md` - Theta estimation documentation
3. `PIRT_FIXES_SUMMARY.md` - This file

### Modified:
1. `pirt_tinybenchmark.r`
   - Fixed response matrix path
   - Added item selection logic
   - Fixed n_all_items calculation
   - Added mirt theta estimation
   - Normalized item IDs

2. `pirt_metabench.r`
   - Normalized item IDs
   - Added mirt theta estimation

3. `pirt_random_100.r`
   - Normalized item IDs
   - Added mirt theta estimation

---

## Testing

Test all three scripts:
```bash
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare

# TinyBenchmark (should now have 417 models, not 3747)
Rscript pirt_tinybenchmark.r --benchmark=arc

# Metabench (avg_predicted should NOT be 0.5)
Rscript pirt_metabench.r --benchmark=arc --version=secondary

# Random 100 (avg_predicted should NOT be 0.5)
Rscript pirt_random_100.r --benchmark=arc
```

### Expected Output Changes:

**TinyBenchmark**:
- Models: 417 (was 3747)
- `n_subset_items`: 99-100
- `n_all_items`: 844
- `avg_predicted`: 0.52-0.68 (was 0)

**Metabench**:
- `avg_predicted`: 0.55-0.62 (was 0.5)

**Random 100**:
- `avg_predicted`: 0.55-0.62 (was 0.5)

---

## Dependencies

The updated scripts require:
- `mirt` package: `install.packages("mirt")`
- All scripts must be run from `/store01/nchawla/pli9/llmbenchmark/baseline_compare/`

---

## Notes

- The fixes maintain backward compatibility with existing output format
- Fallback to simple theta estimation if mirt fails
- All item ID normalizations handle multiple formats
- Standard errors are now available (not currently used)
