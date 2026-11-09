# A principled null: wave-ray cumulative focusing didn't help

The most surprising — and disappointing — finding from my pipeline was a feature
that *should have worked*. I'm posting the post-mortem because the failure
carries a methodology lesson I think is worth more than the feature itself
would have been.

## The hypothesis

Habitat distribution at Refuge Cove is wave-exposure-mediated: macroalgal reef
sits at exposed rocky margins, while seagrass occupies sheltered pockets: a
relationship documented for this site by Ierodiaconou et al. (2018) and earlier
for the broader region by Rattray, Ierodiaconou & Womersley (2015).

My tabular feature set already encoded wave-arrival time via Fast Marching
through bathymetry-as-velocity, propagating waves from an offshore source eastward
across the survey. I wondered whether *cumulative* wave focusing, the integrated
history of how a wavefront converged or diverged on its path from open water to
each survey point, captured exposure information beyond what arrival time alone
could express.

The figure below shows the wave-ray field. Streamlines trace wave-propagation
direction from the eastern open-water boundary inward; the background colormap
shows ∇²T (the divergence of the wavefront gradient): green where wave rays
spread out (low local exposure) and purple where they focus (high local exposure).
The five habitat classes are overlaid.

*Figure omitted from the source-only archive: Wave-ray field over Refuge Cove. Streamlines (white) trace wave-propagation direction inward from the eastern open-water boundary. Background colormap shows ∇²T — green where wave rays diverge (low local exposure), purple where they focus (high local exposure). Coloured markers show the five habitat classes (ALG, FMAT, NVB, SGAM, SGZ)..*

Several class-region patterns in the figure look ecologically suggestive:
macroalgal reef along the western convergence margins where focused wave energy
arrives at the rocky coast, *Amphibolis* seagrass tucked into the southern
divergence pocket, no-visible-biota sand in the broad central low-energy basin,
though visual confirmation of "fit" is itself an unreliable guide and the
remaining classes occupy positions less cleanly mapped to the wave-exposure
narrative.

A brief note on the eastern cutoff visible in the figure. The wave field is computed
only over the survey domain west of a north-south source line placed 53.8 m east
of all sampling points. The eastern edge of the bathymetry raster itself is an
*acquisition* boundary (where the survey vessel stopped collecting data) not
a *physical* boundary of the cove. Treating the acquisition edge as a wave
source would produce wavefronts that follow data-coverage geometry rather than
real wave-propagation geometry.

Also, a note on what T_arrival represents. The wave field here is the solution to the eikonal equation |∇T| = 1/v, where T(x, y) is the time at which a wavefront reaches each point and v(x, y) is local wave-propagation velocity. Following the shallow-water approximation, v ∝ √(depth), so deeper water carries faster wavefronts and shoaling slows them down. The Fast Marching Method solves this on the bathymetry raster with the eastern source line as the initial condition. The result is a per-pixel arrival time that captures wave refraction around bathymetric features — the same physics that makes waves bend around headlands and focus into bays.

## The feature

For each survey point, integrate divergence backward along the wave-ray streamline
from that point to the source:

```
For each pixel (i, j) in the survey domain:
    x, y = (i, j)
    accumulator = 0
    while x, y inside survey:
        u, v = bilinear_sample(grad_T, x, y)
        d    = bilinear_sample(div_T,  x, y)
        accumulator += d
        x, y -= unit_vector(u, v)         # step backward toward source
    flow_accum[i, j] = accumulator
```

Implementation specifics (numba-JIT for speed):

```python
T_smooth = gaussian_filter(T_arrival, sigma=3)        # stabilize 2nd derivatives
dy, dx   = np.gradient(T_smooth, 0.25, 0.25)          # gradient components
div_T    = scipy.ndimage.laplace(T_smooth) / 0.25**2  # ∇²T (pixel-spaced)

@numba.njit(parallel=True)
def flow_accumulate(div_T, dx, dy, mask):
    H, W = div_T.shape
    accum = np.full((H, W), np.nan)
    for i in numba.prange(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            x, y = float(j), float(i)
            total = 0.0
            for _ in range(2 * (H + W)):           # safety cap
                u = bilinear(dx, x, y)
                v = bilinear(dy, x, y)
                if np.hypot(u, v) < 1e-9: break    # gradient stagnation
                total += bilinear(div_T, x, y)
                x -= u / np.hypot(u, v)            # step backward toward source
                y -= v / np.hypot(u, v)
                if not in_mask(mask, x, y): break
            accum[i, j] = total
    return accum

flow_accum = flow_accumulate(div_T, dx, dy, survey_mask)
```

## The verification

Two physical sanity checks before testing classification utility:

- **Monotonicity along streamlines.** Five sample streamlines from random
  survey-interior pixels showed strictly monotonic accumulation curves — no
  oscillations. Integration is numerically stable.
- **Reef-vs-sand separation at the population level.** Mean cumulative-divergence
  values were positive across all classes (consistent with mostly-divergence
  paths from open water), but the magnitude separated by exposure: macroalgal-class
  points (high-exposure rocky reef) had cumulative values roughly twice those of
  no-visible-biota sand points (low-exposure interior bay). The signal does
  discriminate exposed from sheltered habitats. Geometrically: exposed-coastal
  points are reached via long paths that cross more divergence-dominated open
  water, accumulating more positive divergence than the shorter sheltered paths
  to interior bay points.

Both gates passed. The feature carries real physical signal.

## The gate

I evaluated the candidate feature against a 1-SE gate: the new feature must
improve mean weighted F1 by at least one standard error of the baseline before
I'd consider it a genuine signal contribution rather than fold-level noise.

Concretely, with five spatial folds the baseline weighted F1 has SE around 0.036,
so the gate threshold is `baseline + 1 SE`. Improvements smaller than that are
indistinguishable from sampling variance across the fold split — keeping such
features inflates the model with feature-selection noise without improving
generalization.

The 1-SE rule is conservative compared to "any improvement passes" but less
conservative than requiring statistical significance via a paired test. It
matches the implicit decision rule used elsewhere in this competition's CV
analyses (e.g., the κ blend weight sweep that motivated my final blend ratio).

## The empirical test

A single LightGBM (n_estimators=1000, default hyperparameters elsewhere, with
modest upweighting of the macroalgal class), 5-fold spatial BlockKFold at 150 m
spacing:

| Configuration | Mean weighted F1 ± SE |
|---|---|
| Baseline (87 features, no cumulative-flow feature) | 0.7257 ± 0.0362 |
| Gate threshold (baseline + 1 SE) | **0.7619** |
| Baseline + cumulative-flow feature (88 features) | 0.7352 ± 0.0301 |

The feature failed to clear the 1-SE gate. It adds no useful classification signal
on top of the existing 87 features.

## Why it failed — and why the failure is informative

The discriminative content of cumulative wave focusing was already captured by
features I had: arrival time itself, coastline edge proximity, and the multi-scale
rugosity family. Cumulative path-integrated divergence and local arrival time are
physically distinct quantities: one is a path integral of second derivatives,
the other a scalar field; however, for the specific purpose of separating five
habitat classes at this site, they encode equivalent information.

The methodology lesson cuts both ways.

Earlier in my work, I had pruned divergence-derived features from my candidate
list on the basis of high distance correlation with arrival time. That decision
was made without testing whether the high correlation reflected redundancy of
*classification signal* or just redundancy of *value-level dependence*. I revived
the feature on physical-hypothesis grounds. The disciplined test (single-LGB
spatial CV, 1-SE gate) confirmed the original prune was correct, but it took
a hypothesis to know *why*.

Mean-distance-correlation pruning would have permanently retired this feature
without articulating the reason. Hypothesis-driven thinking revived it.
Classification-utility gating put it back to rest. Both halves of the loop are
necessary; neither alone is sufficient.

## Future work

A residual-only formulation — `flow_accum − f(arrival_time)` projecting out the
linearly-redundant component — might still extract usable signal from the
underlying wave physics. Out of scope for this competition; flagging as future
work for anyone interested in extending the feature engineering past published
methods.

A non-linear meta-learner architecture (a stacker with a tree-based meta-model
rather than logistic regression) could in principle gate the feature's signal
selectively where it matters. I tested stacking with a logistic-regression meta
across three architectural variants in this competition; all three failed for
reasons separate from this feature. Worth re-examining with non-linear meta in
a future iteration.

---

## References

- Ierodiaconou, D., Schimel, A.C.G., Kennedy, D., Monk, J., Gaylard, G., Young, M.,
  Diesing, M. & Rattray, A. (2018). Combining pixel and object based image analysis
  of ultra-high resolution multibeam bathymetry and backscatter for habitat mapping
  in shallow marine waters. *Marine Geophysical Research*, 39, 271–288.
- Rattray, A., Ierodiaconou, D. & Womersley, T. (2015). Wave exposure as a predictor
  of benthic habitat distribution on high energy temperate reefs. *Frontiers in
  Marine Science*, 2:8.
