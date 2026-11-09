# Testing two ecological hypotheses at Refuge Cove, and what came out of them

----------

## Setting up the question

When a habitat classifier hesitates between two classes at a test point, there are at least two ways to read the situation. The default reading is that the classifier needs better features; the contested point reveals an information gap in the feature set, and the response is to engineer a feature that closes the gap. There is another reading worth taking seriously: that the contested point is sitting at a real ecological transition zone, and what the model is doing, assigning probability between two classes, is partly a faithful representation of local ecological ambiguity.

This post is about testing that second reading on the GeoHab 2026 MLWG competition dataset.

Two ecological ideas had been forming in my mind through this competition, both grounded in Ierodiaconou et al. 2018, and both shaped by reflection on what contested points at Refuge Cove might actually mean.

The first idea came directly from the paper's Fig. 5, which shows mud accumulation in the southern arm of the cove near a freshwater creek outflow. The figure caption associates the muddy region with the SGZ (*Zostera*) habitat. Mud proportion in surface sediment is not directly measurable from acoustic data; backscatter at this site discriminates gravel and mean grain size but not mud content, and that gap suggests a candidate feature: distance to the creek mouth as a geometric proxy for the unmeasured mud signal. I built a distance-to-creek-mouth feature to proxy this signal, but when I tested it, the classification gains came from the opposite end of the cove — a northern cluster far from the creek — not from the south-margin geography where the mud-gradient mechanism predicts. The feature was doing spatial work, not creek-mechanism work.

The second idea came out of Fig. 9, which ranks variable importance across three model variants (pixel-based, object-based, and a combined hybrid). Bathymetry is the best predictor for NVB in the pixel-based variant, and remains a top predictor for NVB, FMAT, and SGZ in the object-based and combined variants, sometimes top, sometimes not, but always meaningful. The claim I attached to that finding was specific: that depth itself sets a habitat-viability boundary for the NVB↔photosynthetic transition, and that small bathymetric perturbations near that boundary, perhaps triggered by tidal cycles, could plausibly flip the class assignment at a contested point. That's the hypothesis I went on to test.

A note on where my curiosity for this comes from. I am not an ecologist and not an expert in any of the disciplines this post touches. The curiosity itself was shaped by discussions of evolutionary biology in palaeontology courses at university, and a deep and continuing interest in Stephen Jay Gould's books. None of that makes me qualified to argue ecology. It does make me think that contested points in a habitat classifier are worth looking at twice.

What this post records is what happened when I tested the second idea. The test I designed didn't fit the data. The test I redesigned returned a partial result: consistent at one contested point, structurally different at the other. What I think is worth sharing is not the result itself but an investigative angle that came out of the process.

----------

## The test I designed, and what it found at Step 0

The investigation method was: for each contested test point currently predicted near a NVB↔photosynthetic boundary, find the K nearest *training* points (in feature space) that the model predicts as the candidate other class, and report the bathymetric delta between the test point and its K-nearest other-class neighbors. The prediction that follows from the depth-threshold hypothesis is that those deltas should be small, in the pioneer direction (NVB-predicted points with shallower photosynthetic-predicted neighbors), and consistent across a sensitivity sweep over K values. I framed the directional prediction in pioneer-colonization terms, with NVB-predicted points being candidates for shallowing-driven photosynthetic colonization, though that framing is my own interpretive layer, not something the paper discusses.

The full protocol is in the appendix. Six contested test points (IDs 7, 14, 19, 23, 46, 55) made up the initial scope, identified across prior sessions of the competition as multi-class boundary points. The implementation used my model's predictions as source, K ∈ {3, 5, 9} as the sensitivity sweep, and a standardized Euclidean distance over 87 features that excluded any spatial coordinate or coordinate-derived feature.

The first step of the implementation was a scope check. For each of the six contested IDs, I looked up my model's predicted class probabilities, to confirm that the class boundaries were where the hypothesis predicted them to be: between NVB and at least one of {FMAT, SGZ}.

However, the probability table showed that none of the six contested points were sitting at a contested NVB↔photosynthetic boundary. The actual class boundaries were elsewhere: FMAT↔SGAM at one ID, FMAT↔ALG at two IDs, NVB↔ALG at two IDs, and one ID confidently predicted ALG with no meaningful second-class signal.

| ID | Top-1 class | Top-1 proba | Top-2 class | Top-2 proba |
|---|---|---|---|---|
| 7  | FMAT | 0.656 | SGAM | 0.344 |
| 14 | NVB  | 0.915 | ALG  | 0.085 |
| 19 | FMAT | 0.707 | ALG  | 0.293 |
| 23 | NVB  | 0.943 | ALG  | 0.056 |
| 46 | ALG  | 1.000 | —    | —     |
| 55 | FMAT | 0.661 | ALG  | 0.339 |

*Table 1. Predicted class probabilities at the six contested test points. None sit at the NVB↔photosynthetic boundary the depth-threshold hypothesis required.*

This was a setup-fit problem that I had not pre-checked. The hypothesis applied to a specific class transition. The contested test points in this competition's dataset, as my own model predicted them, weren't sitting at that transition. Two of the six were NVB-predicted (and could in principle serve as pioneer-direction tests if we accepted small photosynthetic probability signals in the tail), but four were not.

I halted the depth-threshold test there. Running the K-nearest analysis on points that didn't fit the hypothesis's setup would have produced numbers without a clean interpretation: at best, a small-N sample on points that were borderline cases, at worst, evidence misread to fit the wrong question. The honest move was to stop and look at what the model was actually uncertain about at those points.

----------

## The second hypothesis, designed against the actual class boundaries

What the Step 0 table suggested to me was that the contested points in this dataset weren't sitting at depth-gated boundaries. They were sitting at substrate boundaries (NVB↔ALG and FMAT↔ALG, where ALG is granitic reef versus soft sediment) and at a rugosity boundary (FMAT↔SGAM, where the difference is structural variance from *Amphibolis* root mounds versus filamentous mat). The mechanisms most likely doing the work at these specific contested points are not depth thresholds. They are substrate transitions and structural variance.

Within that, the two NVB-predicted contested points stood out as the only candidates where a coherent ecological hypothesis could still apply. Both were predicted NVB confidently by the model, and both had ALG as their secondary class, meaning the model saw them as bare sand or gravel but with some leakage of reef-like signal in the feature representation. Ierodiaconou's Fig. 9 ranks rugosity-SD as the most important variable for ALG identification, and rugosity is a substrate-via-texture signal: granitic boulders create high rugosity standard deviation, and sand sheets are smooth. So the natural reframe of the question for these two points was: do they sit at rugosity-elevated positions in the feature distribution of otherwise confidently-NVB training points? If yes, the NVB↔ALG ambiguity at these points has a substrate-mediated story that's testable in the existing feature set.

ALG in the Ierodiaconou classification wraps the entire perimeter of Refuge Cove, north margin, east margin, south tip, wherever granitic reef substrate exists at the cove edges. The hypothesis: contested NVB-predicted points sit at high-rugosity positions relative to confidently-NVB training points, because the NVB↔ALG boundary at this site is a substrate boundary mediated by roughness.

*Figure omitted from the source-only archive: Multi-Scale Topographic Position (MSTP) RGB composite over Refuge Cove, with model-predicted classification boundaries and the two NVB-predicted test points (IDs 14 and 23) marked. Both points sit at the transition between the smooth interior (NVB-dominated) and the textured margins (ALG-dominated). The transition is spatially broad around ID 14 and tight around ID 23, consistent with the distance contrast in the K-nearest analysis..*

*Figure 1. Multi-Scale Topographic Position (MSTP) RGB composite over Refuge Cove, with model-predicted classification boundaries and the two NVB-predicted test points (IDs 14 and 23) marked. Both points sit at the transition between the smooth interior (NVB-dominated) and the textured margins (ALG-dominated). The transition is spatially broad around ID 14 and tight around ID 23, consistent with the distance contrast in the K-nearest analysis (ID 14: 1.3–1.8 z-units from nearest ALG neighbors; ID 23: 0.4–0.5 z-units). MSTP encodes relative topographic position across multiple spatial scales, conceptually similar to the Bathymetric Position Index but evaluated across a scale range rather than at a single window size.*

This was a different hypothesis than the one I had set out to test. Before running anything, I needed to do a check that the depth-threshold investigation had not done: verify that the contested points actually sat at the conditions the new hypothesis predicted, before designing the experiment around them. I'll call that a *pre-flight check*, because the language matters: in the depth-threshold case, the equivalent of this check would have caught the setup-fit problem before the implementation started.

The pre-flight check was simple. For the two NVB-predicted contested points, look up their values for two features, a fixed-window rugosity SD measure and a multi-scale roughness measure (which returns the roughness value at the optimal spatial scale per pixel), and compare each value against the distribution of those features across training points where my model predicts NVB with probability ≥ 0.85. The threshold: at least one of the two contested points had to be at or above the 75th percentile of confidently-NVB training points on at least one rugosity feature. The rule was deliberately loose because failing the check meant abandoning the hypothesis entirely; I wanted to abandon only if the hypothesis genuinely had no setup-fit, not because I'd set the bar arbitrarily high.

The result was clear. Both contested points sat at the 95th percentile of confidently-NVB training points on the fixed-window rugosity SD, and at the 92nd–94th percentile on the multi-scale roughness measure. The pre-flight check passed at both points, with margin.

The experiment that followed used the same method as the depth investigation, but with the feature space restricted to a three-feature subspace specific to the rugosity hypothesis: the two rugosity features above and raw bathymetry as a control axis. Restricting to those three features means "K-nearest" measures similarity in rugosity-and-depth space, not in the full 87-feature space the model itself sees. That restriction is what makes the result interpretable as evidence about the substrate-via-rugosity mechanism specifically, rather than evidence about whichever combination of features happens to dominate standardized Euclidean distance over the full feature set.

The candidate other class for the K-nearest filter was ALG only, since ALG was the actual secondary class for both NVB-predicted points in the Step 0 table. The K sweep was the same as the depth investigation: {3, 5, 9}. The full protocol is in the appendix.

The K-nearest comparison found no consistent directional signal on either rugosity feature across both points and all K values. Neither rugosity SD nor multi-scale roughness showed the same sign at both contested points; the direction of the delta depended on which point and which roughness scale. The control axis (raw bathymetry) was the only feature with a consistent sign: ALG-predicted neighbors were shallower than both contested points at every K.

What the experiment did find was a distance contrast. The standardized Euclidean distance from each contested point to its K-th nearest ALG-predicted neighbor, measured in the three-feature subspace, was markedly different between the two points. At ID 14, that distance ranged from 1.3 to 1.8 standardized units across K values, moderate separation. At ID 23, it ranged from 0.4 to 0.5, close enough that the point is nearly indistinguishable from ALG-predicted training points in rugosity-and-depth space. The model predicts ID 23 as NVB, but it does so on the basis of features outside this subspace.

*Figure omitted from the source-only archive: ALG-predicted (blue) and NVB-predicted (red) training points in the rugosity SD – multi-scale roughness – depth subspace. ID 23 sits inside the overlap zone between the two clouds. ID 14 sits at the edge of the NVB cluster, further from the ALG cloud. The NVB cluster is concentrated at low rugosity; ALG points are dispersed across higher values..*

*Figure 2. ALG-predicted (blue) and NVB-predicted (red) training points in the rugosity SD – multi-scale roughness – depth subspace. ID 23 sits inside the overlap zone between the two clouds. ID 14 sits at the edge of the NVB cluster, further from the ALG cloud. The NVB cluster is concentrated at low rugosity; ALG points are dispersed across higher values.*

Two different stories emerge, one per contested point:

For ID 14, the substrate-via-rugosity hypothesis is partially consistent with the data. The point sits at the outer edge of the NVB cloud in rugosity-depth space (Figure 2), at a spatially broad transition zone visible in the MSTP composite (Figure 1). Its ALG-predicted neighbors are moderately distant in the three-feature subspace. A shift along the rugosity axes could plausibly move the model's prediction from NVB to ALG, though the directional signal on individual rugosity features was not clean.

For ID 23, the rugosity-depth subspace doesn't explain the prediction. The point sits in a region where ALG-predicted and NVB-predicted training points overlap (Figure 2), at a spatially tight margin transition (Figure 1), with ALG-predicted neighbors at 0.4 standardized units away. The model is distinguishing NVB from ALG at this point using features outside the three-axis subspace: backscatter intensity, texture features, focal means at other scales, or some combination. The substrate-via-rugosity story does not apply here.

----------

## What this work doesn't say

Several scope limitations are worth being explicit about, both because they constrain what can be claimed from the result and because they give other workers a clear sense of where the framework here would need to be re-checked at their own sites.

The result rests on the contested points produced by my model on this dataset. The set of contested points is a function of the feature set, the training procedure, and the model architecture I used; a different pipeline on the same data, or any pipeline on a different site, would produce a different set of contested points. What I'm sharing is an investigative angle, one that can be applied to whatever class boundaries your model produces, not a finding about contested points in general.

Sample size for the second investigation is two. Two NVB-predicted contested points cleared the pre-flight check; the rugosity analysis ran on those two and produced no consistent directional signal on rugosity across both points, but a clear distance contrast: one point moderately separated from ALG-predicted neighbors, the other nearly indistinguishable. With N=2, these are observations on a small sample, and they're presented that way.

The second investigation tested a substrate-mediated rugosity hypothesis, not a comprehensive enumeration of mechanisms that could explain NVB↔ALG boundary ambiguity. The three-feature subspace was chosen to operationalize the specific hypothesis. Other mechanisms, such as backscatter-mediated discrimination, texture-mediated discrimination, and focal-mean signals at scales not represented in the three-feature subspace, weren't tested. The finding that the second contested point's K-nearest ALG-predicted neighbors are extremely close in the rugosity-depth subspace is consistent with one of those other mechanisms operating, but doesn't identify which. A wave-exposure axis was considered but dropped from the analysis: the feature available in the model (a traveltime from the eikonal solution) does not encode energy density, which is the physical quantity the exposure hypothesis requires. The divergence of the traveltime field would be a more appropriate feature for testing an exposure mechanism, and is flagged as future work.

A brief note on a related observation worth carrying forward. Multi-Scale Topographic Position composites (Lindsay 2019, available in the WhiteboxTools toolbox) visually correspond to the segmentation in Ierodiaconou's classification, suggesting they encode substrate-relevant signal at this site. The visual mechanism is clear; whether that signal can be cleanly extracted in tabular form for a tree-based classifier is a separate question, and one I haven't resolved here. A feature can encode a mechanism visibly to a human reader and still behave differently when fed as a column to a model: the model may already capture the same signal through other features, or the tabular extraction may lose information present in the raster, or both. Either way, the visual correspondence is worth flagging for other workers.

----------

## What this contributes

The connection between the two investigations isn't either of the two hypotheses tested. The first hypothesis (depth-threshold) found that the contested points weren't sitting at depth-gated boundaries. The second hypothesis (substrate-via-rugosity) found a partial fit at one contested point and a structural redirection at the other. Neither is a clean ecological-mechanism finding. Together they are something slightly different: an example of an investigative angle that I think is worth offering to other workers facing contested points in their own habitat ML pipelines.

The angle is three steps:

First, before designing a mechanism test against contested points, look at what your model is actually uncertain about at those points. The class-boundary structure in your dataset may not match the mechanism you're hoping to test. If it doesn't, the test you design against an idealized hypothesis will not have setup-fit, and the mismatch will surface at implementation time as it did here. Looking at the actual class boundaries first is cheap, for me it was a probability-table lookup on six points, and it can prevent a half-day or more of wasted analysis on a hypothesis the data wasn't equipped to test.

Second, when a hypothesis presents itself that fits the actual class boundaries in your dataset, pre-check the conditions the hypothesis needs before running the experiment. The pre-check should be cheap, have a clear pass/fail rule, and be honest enough that failing it means abandoning the hypothesis rather than weakening the threshold to make it pass. For me the pre-check was two percentile lookups against the confidently-predicted-class training distribution. It took under five minutes of analysis time and gave a clear go-decision for the experiment.

Third, when the experiment returns a result that doesn't reduce to a single clean direction, the per-point asymmetry may itself be the finding. The second investigation here returned no consistent directional signal on rugosity, but a clear distance contrast: one contested point moderately separated from ALG-predicted neighbors, the other nearly indistinguishable. The temptation in writing up such a result is to choose the cleaner half and report it. The honest move is to report both halves and let the asymmetry stand. That's what I've tried to do here, and what I'd suggest other workers do in equivalent situations.

----------

## Methodological appendix

This appendix records the minimum protocol detail a reader would need to follow what was done. A notebook reproducing the figures and the analyses will be added to the project repository post-deadline.

**Site and dataset.** Refuge Cove, a sheltered embayment on the south coast of Victoria, Australia, mapped by Ierodiaconou et al. 2018 with multibeam echosounder bathymetry and backscatter at 0.25 m horizontal resolution. Ground-truth labels at 6,256 training points and 98 test points across five classes: ALG (macroalgae-dominated reef), FMAT (filamentous mat on soft sediment), NVB (no visible biota), SGAM (*Amphibolis antarctica* seagrass), and SGZ (*Zostera* sp. seagrass). The dataset is the basis for the GeoHab 2026 MLWG Kaggle competition.

**Model and feature set.** All predictions in this post come from the LightGBM tabular component of my competition model, trained on an 87-feature set. LightGBM with `n_estimators=1000`, no early stopping, `class_weight={ALG:2.0, others:1.0}`, `seed=42`. Out-of-fold predictions on the training set come from a spatially blocked cross-validation, five folds, single seed.

**The first investigation (depth-threshold).** For each contested test point, the K nearest training points (in feature space) predicted as the candidate other class are identified, and the bathymetric delta is reported. Standardized Euclidean distance over the 87-feature set, with all spatial-coordinate-derived features deliberately excluded. NaN-aware via `nan_euclidean_distances` in scikit-learn. K sweep ∈ {3, 5, 9}. The investigation halted at Step 0 when the scope check showed that none of the six contested test points were sitting at NVB↔photosynthetic boundaries in the model's predictions.

**The second investigation (substrate-via-rugosity).** Pre-flight check first. For the two NVB-predicted contested test points, percentile rank within the confidently-NVB training distribution (NVB prediction probability ≥ 0.85, n = 3,240) was computed for two features: a fixed-window rugosity standard deviation and a multi-scale roughness measure (returning the roughness value at the optimal spatial scale per pixel). Pre-flight passed with both contested points at the 95th percentile on the fixed-window rugosity SD and the 92nd–94th percentile on the multi-scale roughness measure.

The K-nearest experiment that followed used the same method, but in a three-feature subspace specific to the substrate-via-rugosity hypothesis: the two rugosity features above and raw bathymetry as a control axis. K sweep ∈ {3, 5, 9}. Candidate other-class restricted to ALG (the actual secondary class for both NVB-predicted points). All other parameters as above.

**Code and data availability.** The model and feature engineering are documented separately in the competition's pipeline notebook (linked on the competition page). Feature CSVs and the model's OOF predictions are included with the project repository. A notebook reproducing the figures and analyses in this post will be added post-deadline.

**References.** Ierodiaconou, D. et al. 2018. *Combining pixel and object based image analysis of ultra-high resolution multibeam bathymetry and backscatter for habitat mapping in shallow marine waters*. Marine Geophysical Research 39:271–288.
