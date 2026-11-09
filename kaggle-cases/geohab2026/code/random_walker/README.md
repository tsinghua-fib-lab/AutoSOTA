# Random Walker Segmentation for Benthic Habitat Feature Engineering

Semi-supervised segmentation as a feature source for tabular ML pipelines, with explicit treatment of label leakage.

## Motivation

Pixel-based (PB) classifiers for benthic habitat mapping extract features at point locations but lack spatial context: they don't know whether a point sits in the middle of a homogeneous patch or on a class boundary. Object-based (OB) approaches address this by computing zonal statistics within segments, but most open-source segmentation methods (Felzenszwalb, SLIC, hierarchical watershed) are unsupervised and don't use class labels.

Random walker segmentation (Grady 2006) offers a different path. Given labeled seed points, it solves for the probability that a random walker starting from each unlabeled pixel reaches each seed class first. The output is a per-pixel probability field P(c, y, x) that encodes gradient-connected spatial context, exactly the signal tabular classifiers struggle to learn from point features alone.

The catch: this creates a subtle but severe label leakage trap when the probabilities are used as features in a supervised pipeline.

## The Leakage Trap

Seed pixels have P(correct_class) = 1.0 by construction. Nearby pixels inherit near-1.0 probabilities via short random-walk paths. Extracting P(c) at training-point locations gives the model a near-perfect lookup of the training label, catastrophically inflating cross-validation scores. Test points, which were not seeds, get genuine probabilities, so the feature definition is asymmetric between train and test.

## Escape Routes

Three approaches to extract leak-safe features, ordered by computational cost:

**(a) Leave-one-out seeding.** For each training point, rerun random walker with that point excluded from seeds. Mathematically clean. Computationally brutal (N random walks for N training points).

**(b) Spatial-block-wise seeding.** Split training points into K spatial blocks (matching the spatial CV protocol). For each block, run random walker using only other blocks' points as seeds and query P at the held-out block. K random walks per pass. Practical with conjugate gradient solver and pyamg backend.

**(c) Boundary-derived features.** The segmentation's class boundaries (where argmax transitions between classes) depend on gradient structure and seed layout, not on any individual training point's label. Removing one seed doesn't materially shift boundaries when neighboring same-class seeds keep them anchored. Two features are leak-safe by construction:

- `dist_to_nearest_boundary`: continuous distance in meters to the nearest class transition
- `boundary_density_in_window`: fraction of pixels within a window that are on a class boundary

Both have identical definitions at train and test time.

## Why Random Walker for Benthic Mapping

Random walker handles gradational boundaries gracefully, which makes it well suited for soft-sediment habitats where classes grade continuously (e.g., filamentous mat to unvegetated sediment to seagrass). Edge-based and region-merging methods either fail to find boundaries in these zones (no sharp edges to detect) or produce arbitrary boundaries from noise. The probabilistic formulation naturally represents the uncertainty at transitions.

Comparison with alternatives tested on the same dataset:

- **Felzenszwalb:** fast, parameter-light, but graph edge weights from pixel differences can be dominated by a single noisy channel
- **SLIC + RAG:** superpixel compactness constraint produces segments unrelated to habitat structure
- **HiGRA hierarchical watershed:** no seeds needed, multi-scale for free, but segment boundaries follow attribute hierarchy rather than class-informed gradients
- **Marker-controlled watershed:** gives boundary control via edge maps, but produces hard segments without probability output

## Input Channel Selection

Random walker with `multichannel=True` accepts multiple raster channels. Not all features are suitable inputs: spatially incoherent or positional features (raw coordinates, distance-to-shore) produce poor probability fields. Input channels should be selected via a separability analysis to identify spatially coherent features that encode habitat-relevant gradients.

## Connection to Destriping

Random walker uses gradient structure to route probability flow between seeds. Residual acquisition artifacts (e.g., sonar striping in backscatter) contaminate gradients and route probability along stripe directions rather than habitat boundaries. Destriping the input channels before segmentation is not optional.

## Dependencies

- scikit-image (`skimage.segmentation.random_walker`)
- pyamg (for conjugate gradient solver, recommended for large rasters)
- numpy, rasterio

## References

- Grady, L. (2006). Random walks for image segmentation. *IEEE TPAMI* 28(11):1768–1783.
- Gouillart, E. et al. (2014). scikit-image: Image processing in Python. *PeerJ* 2:e453.
- Ierodiaconou, D. et al. (2018). Combining pixel and object based approaches for habitat mapping using MBES. *Marine Geophysical Research* 39:271–288.

## Status

Exploration notebook completed during GeoHab 2026 MLWG competition. Route (c) boundary-derived features implemented and tested. Routes (a) and (b) documented but not executed due to competition timeline.

## Parent Notebook

This work extends `marker_controlled_watershed_2025_update.ipynb`, a geological lineament mapping workflow using Gabor-filtered orientation maps and watershed segmentation on curvature surfaces.
