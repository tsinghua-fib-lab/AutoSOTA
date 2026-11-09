# Spatial cross-validation for benthic habitat ML

*Per-class indicator variogram radii, pre-flight screening, and one negative result*

I want to share three methodological things from the GeoHab 2026 MLWG competition that I think transfer beyond this dataset: a protocol for fitting per-class indicator variograms at multiple lag windows, a pre-flight screening procedure for spatial leave-one-out cross-validation, and one negative result on object-based features fed to a tabular model.

All three are grounded in the Refuge Cove dataset from Ierodiaconou et al. (2018). I'm holding back competition-specific details (feature identities, submission scores, model parameters) because they aren't needed to follow the methods. The companion notebook for my entry is public if you want the full pipeline.

---

## Part I — Per-class indicator variogram LOOCV

### Why fixed-radius BlockKFold isn't enough

Spatial autocorrelation in benthic habitat data is well documented. Ierodiaconou et al. (2018, Appendix 2) report spatial autocorrelation up to approximately 250 m in the raw AUV video transects at Refuge Cove (Moran's I analysis); their validation sample spacing of 50 m was designed to reduce this. When I ran StratifiedKFold (k=5) on this dataset, I got a weighted F1 of 0.989 (std 0.003). Switching to BlockKFold at 150 m spacing dropped that to 0.635 (std 0.208). I interpreted the 0.35 gap as spatial leakage: under random splits, held-out points find near-twins in the training set, and the F1 reflects local similarity rather than learned structure.

The high BlockKFold variance (std 0.208) surprised me as much as the gap itself. At 150 m blocks on a 0.72 km² bay, fold composition varies a lot. Some folds contain little FMAT or SGAM. A single CV point estimate doesn't tell the whole story here.

I wish I had quantified this earlier in the competition. BlockKFold was a real improvement, but it applies one exclusion distance to all classes, and the habitats at Refuge Cove don't cooperate with that assumption. ALG occupies granitic reef whose extent is controlled by bedrock geometry, and NVB covers broad sand flats shaped by sediment transport across the bay. These are substrate-controlled patches, spatially extensive. FMAT and SGZ, by contrast, form smaller patches driven by local nutrient flux, light availability, and bioturbation on shared sandy substrate. A single block size cannot honestly evaluate a model on classes with such different spatial footprints, which is what led me to per-class exclusion radii grounded in indicator variograms.

*Figure omitted from the source-only archive: Figure 1. Three CV schemes compared on representative held-out points for ALG (top), SGZ (middle), and NVB (bottom). Left: StratifiedKFold (k=5), no spatial exclusion. Centre: BlockKFold at 150 m spacing, dashed red square shows exclusion block. Right: per-class variogram LOOCV, red circle shows class-specific exclusion radius. Grey points are excluded from training; blue points are eligible. The NVB panel (bottom right) illustrates the scale problem: a 414 m exclusion radius covers most of the 0.72 km² study area..*

### Multi-lag indicator variogram protocol

The idea behind per-class exclusion radii is simple: fit an indicator variogram for each class and read off the range as the distance over which class membership is spatially autocorrelated. In practice, the fitting turned out to be the hard part.

My first attempt used a single fit window at max_lag=600 m. Two classes (ALG and FMAT) hit the upper fitting bound at 1500 m, which told me the fit was capturing regional nonstationarity rather than local autocorrelation structure. For leave-one-out cross-validation (LOOCV) exclusion radii, I needed the short-lag range: the distance over which knowing a point's class still tells you something about its neighbours. Long-lag trend is the wrong thing to fit.

To see why the long-lag fits failed, consider what an indicator variogram measures. For a given class, gamma at lag h is the probability that two points separated by distance h have different class memberships. At short lags, nearby points tend to share a class, so gamma is low. At longer lags, gamma rises toward the sill p(1−p). The range is where it plateaus: beyond it, class membership at one point tells you nothing about the other.

On a small site like Refuge Cove (1222 m diagonal), long lags create a problem. If ALG concentrates along reef margins and NVB dominates the central sand flats, then at 400–600 m lags you're comparing points from entirely different parts of the bay. The variogram keeps rising past the theoretical sill because these distant pairs carry the bay's large-scale habitat zonation on top of local autocorrelation. The fitter accommodates that rising tail by pushing the range estimate toward the fitting bound. The resulting "range" reflects the scale of the spatial trend, not the local dependence I needed for an exclusion radius.

The fix was a multi-window sensitivity test. I ran each class through three fitting windows with shared lag spacing (xlag=40 m, chosen to give at least 3–4 bins within the shortest window while maintaining min_pairs=30 per bin at the site's point density; xltol=20 m): Run A at approximately 160 m max lag, Run B at 240 m, and Run C at 400 m. The question is how the fitted range behaves as the window widens.

Three patterns emerged. FMAT showed the clearest diagnostic: it converged at 186 m in the shortest window, then the range shot upward at wider windows (300 m, 456 m) as nonstationarity took over — a hockey-stick effect. That tells me the short-lag fit captured genuine local structure, and the wider windows are chasing trend. SGZ showed the pattern I was hoping for across the board: two independent windows (240 m and 400 m) converging on nearly identical ranges (191 m and 187 m), confirming stable short-lag structure. ALG and NVB were harder to read. Both hit the upper fitting bound at the 160 m and 240 m windows, meaning those windows were simply too narrow to resolve their spatial structure. They only converged at the 400 m window (278 m and 414 m respectively), which gives me a single estimate rather than a stability test. The 5×3 variogram panel (Figure 2) shows all fifteen fits; I'd encourage readers to judge them visually rather than taking the table on faith.

*Figure omitted from the source-only archive: Figure 2. Indicator variograms for all five classes (rows) at three max-lag windows: 160 m, 240 m, and 400 m (columns). Coloured dots are experimental semivariance values with pair counts annotated; solid curves are fitted Gaussian or spherical models. Dashed horizontal lines show the theoretical sill p(1−p). Green vertical dashed lines mark the fitted range. Bound-hit symbols flag cases where the fitted range reached the upper fitting limit. FMAT (row 2) converges at 186 m already in the 160 m window; SGAM (row 4) collapses to a lower-bound artifact across all windows due to near-zero indicator variance at n=170..*

I used stability bands to formalize the decision: if the ratio of fitted ranges between two windows was 15% or less, I called it stable. Between 15% and 25%, borderline, requiring manual review. Above 25%, unstable, and I did not assign a radius from that fit. These thresholds are practitioner's choices calibrated on this dataset, not universal cutoffs; a different site with different point density and extent might need different bands.

Here is what came out:

| Class | Radius (m) | Basis | Confidence |
|-------|-----------|-------|-----------|
| ALG | 278 | Run C (400 m window) convergence | Medium; treat as upper bound |
| FMAT | 186 | Run A (160 m window) convergence | High; textbook short-lag signal |
| NVB | 414 | Run C (400 m window) convergence | Medium; treat as upper bound |
| SGAM | 190 | Ecological proxy (= SGZ) | Low; direct fit failed, n=170 |
| SGZ | 190 | Run B + C convergence (ratio 0.02) | High; two independent windows agree |

Two results worth noting. FMAT and SGZ both resolved cleanly at short lag distances (186 m and 190 m respectively), with FMAT converging already at the 160 m window and SGZ showing near-identical ranges at 240 m and 400 m. These are the classes I'm most confident about.

ALG and NVB did not plateau within the 160 m or 240 m windows; both hit upper fitting bounds in Runs A and B. They only resolved at the 400 m window (278 m and 414 m respectively). These are usable radii but I treat them as upper bounds, since a 400 m window on a 1222 m diagonal site may still be influenced by nonstationarity.

SGAM was a different problem entirely. With only 170 training points (prevalence 2.7%), the indicator variable has variance of just 0.026, which is p(1−p) for a near-zero prevalence class. Almost every pair in the distance matrix is (0,0), so gamma values across all lags hover near zero and the fitter returns the minimum allowed range (40 m) in both Runs A and B. At short lags on a class this sparse, even the min_pairs=30 threshold per bin is hard to meet, which compounds the problem. This is a lower-bound artifact, not a meaningful spatial range. I assigned SGAM a proxy radius of 190 m by ecological analogy to SGZ: both are seagrasses on sand, and *Amphibolis antarctica* shares spatial-pattern drivers with *Zostera* sp. at the scale of this bay. It is the least defensible radius in the table, and I flag it as such. But I think the reasoning is transparent enough that a reader can decide whether they'd make the same call.

The core of the multi-lag fitting, genericized — see [`methods/variogram_loocv.py`](../methods/variogram_loocv.py) for the full implementation:

```python
import numpy as np
from scipy.optimize import curve_fit

def gaussian_variogram(h, nugget, sill, range_a):
    """Gaussian variogram model."""
    return nugget + (sill - nugget) * (1 - np.exp(-3 * (h / range_a)**2))

def fit_indicator_variogram(coords, indicator, xlag, xltol, nlag, min_pairs=30):
    """
    Compute experimental indicator variogram and fit a model.

    coords: (N, 2) array of point coordinates
    indicator: (N,) binary array, 1 where point belongs to class
    xlag: lag spacing in metres
    xltol: lag tolerance in metres
    nlag: number of lags
    min_pairs: minimum pairs per bin to include in fit
    """
    # Pairwise distances (upper triangle only for efficiency)
    n = len(coords)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = coords[i_idx, 0] - coords[j_idx, 0]
    dy = coords[i_idx, 1] - coords[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)

    # Squared indicator differences
    diff_sq = (indicator[i_idx] - indicator[j_idx])**2

    # Bin into lags
    lag_centers = np.arange(1, nlag + 1) * xlag
    gamma = np.full(nlag, np.nan)
    npairs = np.zeros(nlag, dtype=int)

    for k, center in enumerate(lag_centers):
        mask = np.abs(dists - center) <= xltol
        npairs[k] = mask.sum()
        if npairs[k] >= min_pairs:
            gamma[k] = 0.5 * np.mean(diff_sq[mask])

    # Fit model to valid bins
    valid = ~np.isnan(gamma)
    if valid.sum() < 3:
        return None  # insufficient data for fitting

    upper_bound = lag_centers[valid][-1] * 1.25  # fitting bound
    p0 = [0.01, np.nanmax(gamma), lag_centers[valid][-1] * 0.5]
    bounds = ([0, 0, xlag], [np.nanmax(gamma), 1.0, upper_bound])

    try:
        popt, _ = curve_fit(
            gaussian_variogram, lag_centers[valid], gamma[valid],
            p0=p0, bounds=bounds, maxfev=5000
        )
        fitted_range = popt[2]
        bound_hit = fitted_range > 0.95 * upper_bound
        return {
            'lag_centers': lag_centers[valid],
            'gamma': gamma[valid],
            'npairs': npairs[valid],
            'nugget': popt[0], 'sill': popt[1], 'range': popt[2],
            'bound_hit': bound_hit, 'upper_bound': upper_bound
        }
    except RuntimeError:
        return None  # fit did not converge
```

### Per-class autocorrelation length stratifies by patch geometry

The radii in that table aren't random. ALG and NVB have longer correlation lengths (278 m and 414 m) than FMAT and SGZ (186 m and 190 m), and the split maps onto an ecological distinction that Ierodiaconou et al. (2018) describe in their habitat characterization.

ALG is granitic bedrock and boulder reef, spanning 2–22 m depth. Its spatial extent is locked to the reef geometry: where the rock is, the macroalgae are. NVB is bare sand covering broad swaths of the bay floor, its extent shaped by sediment transport regimes operating at the scale of the whole cove. Both are substrate-controlled habitats, and their indicator variogram ranges reflect that: the autocorrelation persists over hundreds of metres because the underlying substrate doesn't change at shorter scales.

FMAT and SGZ sit on the same sandy substrate but are biologically driven. FMAT is filamentous microalgae and diatom mats responding to local nutrient flux in the sheltered southern arm. SGZ is *Zostera* sp. seagrass, patchy, varying from sparse to dense, shaped by light availability and bioturbation. These habitats can shift over metres as conditions change. Their shorter variogram ranges (186 m, 190 m) reflect that finer-grained spatial structure.

This isn't just a statistical convenience. It means a single global exclusion radius is always a compromise: too small for substrate-controlled classes (leaking autocorrelation from ALG and NVB patches), too large for biologically-driven ones (discarding usable training data for FMAT and SGZ). Per-class radii grounded in indicator variograms are the principled alternative when the data support fitting them, which brings us to the question of when they don't.

---

## Pre-flight screening for spatial LOOCV

The preceding section ended with per-class radii as the principled choice "when the data support fitting them." The pre-flight screening I describe here is how I checked whether the data actually did.

The concern is training set depletion. When you hold out a single point and exclude everything within its class-specific radius, the remaining training set may be too small or too imbalanced to fit a reliable model for that fold. On a 0.72 km² site with radii up to 414 m, this is not hypothetical. I found two distinct failure modes worth documenting.

**Class-level structural failure: SGAM.** SGAM has 170 training points (2.7% prevalence), tightly clustered in the south-central part of the bay. With a 190 m exclusion radius, 98.8% of SGAM points fall within the exclusion zone of a typical held-out SGAM point. The expected number of same-class training examples remaining per fold is 2.0. This cannot be resolved by adjusting the radius; it is a consequence of a rare, spatially compact class on a small site. LOOCV F1 for SGAM is structurally unreliable, and I report it as such rather than excluding SGAM from evaluation. The failure is the finding.

**Point-level degenerate fold: NVB idx=3198.** NVB has the largest exclusion radius (414 m) and the most training points (3,036), but one point sits in such a dense NVB cluster that its 414 m exclusion leaves only 9 eligible training points for that fold. This is a point-level problem, not a class-level one; most NVB folds have over a thousand training points. I kept this fold in the evaluation and flagged it.

These two cases motivated a screening protocol I ran before any LOOCV model fitting:

1. For every held-out point, compute the eligible training set size after exclusion.
2. Flag any fold where total eligible points falls below 100 (`MIN_FOLD_ABSOLUTE`).
3. Flag any fold where any class has fewer than 10 representatives (`MIN_CLASS_REPS`).
4. Report degenerate folds; do not silently drop them.
5. Compute the ratio of exclusion zone area to study extent per class. If any class exclusion exceeds 50% of the study diagonal, the LOOCV protocol may not be viable for that class at that site.

The screening flagged 266 degenerate folds and 2,556 folds with low class representation, overwhelmingly driven by SGAM depletion. These numbers told me, before fitting a single model, that the aggregate LOOCV F1 would be dominated by undertrained folds rather than honest spatial evaluation. That context matters when interpreting the results below.

The screening implementation — also in [`methods/variogram_loocv.py`](../methods/variogram_loocv.py):

```python
def screen_loocv_folds(coords, labels, class_radii,
                       min_fold_absolute=100, min_class_reps=10):
    """
    Pre-flight screening for per-class spatial LOOCV.

    coords: (N, 2) array of training point coordinates
    labels: (N,) array of class labels
    class_radii: dict mapping class label to exclusion radius in metres
    min_fold_absolute: flag if fewer than this many eligible training points
    min_class_reps: flag if any class has fewer than this many representatives

    Returns list of dicts, one per held-out point, with fold diagnostics.
    """
    # Pre-compute full pairwise distance matrix
    # (VanderPlas pattern: NumPy broadcasting, float32 storage)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

    classes = np.unique(labels)
    n = len(coords)
    fold_reports = []

    for i in range(n):
        held_out_class = labels[i]
        radius = class_radii[held_out_class]

        # Exclude points within radius of held-out point
        excluded = dist_matrix[i] <= radius
        excluded[i] = True  # always exclude self
        eligible = ~excluded

        n_eligible = eligible.sum()

        # Per-class counts in eligible training set
        class_counts = {}
        for c in classes:
            class_counts[c] = ((labels == c) & eligible).sum()

        min_class = min(class_counts.values())
        worst_class = min(class_counts, key=class_counts.get)

        flags = []
        if n_eligible < min_fold_absolute:
            flags.append(f"MIN_FOLD_ABSOLUTE: {n_eligible} < {min_fold_absolute}")
        if min_class < min_class_reps:
            flags.append(f"MIN_CLASS_REPS: {worst_class}={min_class} < {min_class_reps}")

        fold_reports.append({
            'idx': i, 'held_out_class': held_out_class,
            'n_eligible': n_eligible, 'class_counts': class_counts,
            'flags': flags
        })

    return fold_reports
```

---

## The per-class picture and the LOOCV result

The headline numbers: StratifiedKFold F1 of 0.989, BlockKFold F1 of 0.635, a 0.35 gap. The per-class breakdown shows where that gap comes from.

| Class | StratifiedKFold F1 | BlockKFold F1 | Drop |
|-------|-------------------|---------------|------|
| ALG | 0.980 | 0.568 | 0.412 |
| FMAT | 0.991 | 0.313 | 0.678 |
| NVB | 0.991 | 0.618 | 0.373 |
| SGAM | 0.967 | 0.000 | 0.967 |
| SGZ | 0.984 | 0.096 | 0.888 |

The classes that collapse hardest under spatial CV are exactly the ones you'd expect from the ecology. SGAM (rare, tightly clustered) drops to zero: once you block spatially, there aren't enough SGAM examples in most folds to learn the class at all. SGZ and FMAT, the smaller biologically-driven patches, lose most of their apparent performance. NVB and ALG, the spatially extensive substrate-controlled classes, retain more, though ALG still drops substantially because reef patches, while large, are geographically concentrated along the bay margins.

Under StratifiedKFold, every class looks nearly perfect because random splits let each held-out point find a near-twin in training.

*Figure omitted from the source-only archive: Figure 3. Weighted F1 under three CV schemes: StratifiedKFold (k=5), BlockKFold at 150 m, and per-class variogram LOOCV. Blue bars include all five classes; orange bars exclude SGAM (4 classes). Error bars show ±1 standard deviation across folds (StratifiedKFold and BlockKFold only; LOOCV has no fold-level variance estimate). Annotation at bottom flags SGAM structural failure (n=170, 98.8% self-exclusion under LOOCV) and screening results (266 degenerate folds, 2,556 folds with low class representation)..*

I also ran per-class LOOCV using the radii from above. The aggregate weighted F1 was 0.224. On its face, that looks like a much harsher estimate of spatial optimism, but the screening results explain why. With 266 degenerate folds and 2,556 folds where at least one class had fewer than 10 training representatives, the aggregate F1 reflects models trained on depleted data, not a clean measure of spatial leakage. NVB's 414 m exclusion radius covers a large fraction of the 0.72 km² study area; 41% of folds had insufficient class representation.

The honest conclusion is that per-class LOOCV at variogram-derived radii is scientifically defensible in principle but operationally constrained by the study area and sample size at Refuge Cove. Two directions for future work suggest themselves. The first is a multi-radius grid: run LOOCV at exclusion radii of 100 m, 150 m, 190 m, 240 m and so on, observe how F1 evolves with radius, and identify the inflection point where training-data depletion takes over from spatial-leakage removal. The second is rotated block CV: average BlockKFold F1 across multiple grid orientations (e.g. every 30 degrees) to reduce sensitivity to the arbitrary alignment between block grid and habitat geometry. Both deserve their own treatment rather than being footnotes here.

---

## Part II — One negative result

### Object-based features in a tabular pipeline

Ierodiaconou et al. (2018) found that their combined pixel-based and object-based model (83.6% overall accuracy) significantly outperformed either approach alone. Naturally, I tried to replicate the benefit by computing backscatter statistics (mean, standard deviation, skewness) within image segments and feeding them as features to a LightGBM classifier alongside my pixel-based features.

It made things worse. CV suggested a small improvement, but the public leaderboard score dropped. The direction of the gap — CV up and LB down — pointed to overfitting, and the mechanism turned out to be pseudoreplication. All training points within a single segment receive identical OB feature values. The model sees what looks like independent confirmation from multiple points, but it's the same observation repeated. The effective sample size is the number of segments, not the number of points.

The paper's approach didn't have this problem. Ierodiaconou et al. used OB features for object-based classification, where the unit of analysis is the segment itself. Each segment is one observation with one label. My mistake was importing segment-level summaries into a point-level pipeline without adjusting for the change in support.

For anyone working with OB features in a point-based tabular model: either deduplicate to one representative point per segment, or use OB features only in a model where the segment is the prediction unit. The concept of OB features isn't dead; the application needs to match the support. I'm currently exploring random walker segmentation as an alternative that may produce segments better aligned with habitat boundaries, but that work is ongoing and I don't have results to share yet.

---

## What this contributes

Three things I hope are useful beyond this competition. First, spatial optimism bias on benthic ML at site scale is large. On this dataset the gap between StratifiedKFold and BlockKFold was 0.35 weighted F1. Without spatial CV, a substantial fraction of apparent feature engineering gains can come from leakage rather than learned structure.

Second, per-class indicator variogram radii are operationally defensible when the data support fitting them, but they require pre-flight screening. The SGAM structural failure and the NVB degenerate fold are not edge cases to be swept aside; they are the kind of thing that silently corrupts an aggregate metric if you don't look for it before fitting models.

Third, pseudoreplication from segment-level features fed to point-level tabular models is a quiet CV failure mode. The fix is straightforward once you see the problem, but I didn't see it until the leaderboard told me something was wrong.

---

## Code and data availability

All code shown in this post is standalone and uses generic variable names with no competition-specific feature identities. The snippets above cover the multi-lag variogram fitting and the pre-flight screening protocol. The full implementations are in [`methods/variogram_loocv.py`](../methods/variogram_loocv.py). The competition pipeline notebook (in the Code tab) uses verde BlockKFold at 150 m spacing; the per-class LOOCV code is included in the snippets in this post.

---

## Acknowledgements

Jake VanderPlas for the vectorized pairwise distance pattern (SciPy conference talks). Ierodiaconou et al. (2018) for the site characterization, habitat descriptions, and spatial autocorrelation analysis that grounded every decision in this post. The MLWG organizers for a competition framing that motivated this work.

Spatial cross-validation in this pipeline uses BlockKFold from Verde, part of the Fatiando a Terra project, which turned 16 this year. Thank you to Leonardo Uieda, Santiago Soler, and the Fatiando community for building and maintaining the tools that make honest spatial evaluation accessible in Python.

The indicator variogram methodology follows Dubrule (2003, §4.5.2). Michael Pyrcz's GeostatsPy notebooks and teaching materials shaped how I approached variogram fitting and interpretation.

---

## References

Ierodiaconou, D., Schimel, A.C.G., Kennedy, D., Monk, J., Gaylard, G., Young, M., Diesing, M., Rattray, A., 2018. Combining pixel and object based image analysis of ultra-high resolution multibeam bathymetry and backscatter for habitat mapping in shallow marine waters. *Marine Geophysical Research* 39, 271–288.

Uieda, L. (2018). Verde: Processing and gridding spatial data using Green's functions. *Journal of Open Source Software*, 3(29), 957. doi:10.21105/joss.00957

Dubrule, O., 2003. *Geostatistics for Seismic Data Integration in Earth Models*. SEG/EAGE Distinguished Instructor Short Course No. 6. Society of Exploration Geophysicists, Tulsa, OK. doi:10.1190/1.9781560801962
