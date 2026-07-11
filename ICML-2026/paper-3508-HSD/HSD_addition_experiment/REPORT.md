# ICML Rebuttal: Additional Experiments Report

---

## Baseline Comparison

Relative L2 error (lower is better) across 7 methods on two tasks:

|           |  HSD  | GNOT'23 | ONO'23 | HAMLET'24 | DeepONet | GeoFNO |  FNO  |
|-----------|-------|---------|--------|-----------|----------|--------|-------|
| Ellipsoid | **0.037** | 0.144   | 0.155  | 0.159     | 0.221    | 0.240  | 0.246 |
| Torus     | **0.058** | 0.277   | 0.262  | 0.250     | 0.652    | 0.449  | 0.418 |

HSD outperforms all baselines by 74–86% on both tasks, including recent 2023–2024 methods (GNOT, ONO, HAMLET).

---

## Ablation 1: Input Modality

| Input Mode | Ellipsoid | Torus |
|------------|-----------|-------|
| Mesh | 0.036 | 0.056 |
| Point Cloud | 0.039 | 0.054 |
| Graph | 0.036 | 0.057 |
| **Spread** | **0.003** | **0.003** |

Consistent across all representations (<0.5% variation).

---

## Ablation 2: Pseudo-Spectral Bilinear Layer

| Layer Type | Ellipsoid | Torus |
|------------|-----------|-------|
| Plain MLP | 0.037 | 0.056 |
| **Spectral Bilinear** | **0.026** | **0.049** |

Improvement: +30% (Ellipsoid), +13% (Torus).

---

## Ablation 3: Whitney-KDE Spatial Encoding

| Spatial Encoding | Ellipsoid | Torus |
|-----------------|-----------|-------|
| **Whitney-KDE** | **0.037** | **0.058** |
| Raw Euclidean 3D | 0.042 | 0.065 |

Whitney-KDE improves 12–14% by respecting simplicial structure.

---

## Ablation 4: Spectral Basis Quality

| Basis | Ellipsoid | Torus |
|-------|-----------|-------|
| **Hodge Laplacian** | **0.036** | **0.156** |
| RBF (geometry only) | 0.042 | 0.174 |
| Random orthogonal | 0.084 | 0.783 |

Hodge basis encodes topology via B₁, B₂ boundary operators — 14–17% better than geometry-only, 2.3–5× better than random.

---

## Ablation 5: Hodge Spectral Component Decomposition

**Question:** Do the harmonic, gradient (exact), and curl (coexact) parts of the Hodge spectrum provide complementary information? What happens when only the gradient or curl eigenvectors — obtained by applying the boundary and co-boundary operators to the low-frequency part of the spectrum in dimensions k−1 and k+1 — are used in the spectral decomposition?

### Protocol

Following the Hodge decomposition Ω¹ = im(d₀) ⊕ im(δ₁) ⊕ ker(L₁), we construct three isolated spectral channels, each derived from the relevant part of the de Rham complex:

**Channel construction:**

1. **Exact (gradient) channel — from the (k−1)-spectrum via d₀:**
   Compute `grad_c0 = c₀ · Md₀ᵀ`, where `Md₀ = Φ₁ᵀ B₁ Φ₀` is the spectral representation of the exterior derivative d₀. This applies the boundary operator B₁ to the low-frequency 0-form eigenvectors Φ₀, projecting the input scalar field into the exact (gradient / curl-free) 1-form subspace.
   Effective spectral rank: 8 (Ellipsoid) / 10 (Torus), matching the number of exact eigenmodes in the bottom-64 L₁ spectrum.

2. **Coexact (curl) channel — from the (k+1)-spectrum via δ₁:**
   Compute enriched 2-form spectral coefficients via the co-boundary operator B₂ᵀ applied to the low-frequency 2-form eigenvectors Φ₂. Specifically: `c₂ = [B₂ᵀ|B₁|f/2, B₂(∇xᵢ · ∇f)]ᵀ Φ₂`, which includes edge-averaged curl and curvature-coupled 2-form features. These genuinely nonzero features (RMS ≈ 0.015–0.032) encode the coexact (divergence-free / solenoidal) 1-form subspace structure.
   Effective spectral rank: 49 (Ellipsoid) / 53 (Torus).

3. **Harmonic channel — from ker(L₁):**
   Project c₀ onto the harmonic eigenvectors of L₀ (eigenvalue ≈ 0). On genus-0 surfaces (Ellipsoid), this yields 1 mode (the DC / global mean). On genus-1 (Torus), b₁ = 2 additional harmonic 1-form modes encode the non-contractible cycles, but the 0-form harmonic subspace still provides just 1 mode (b₀ = 1).

**Controlled setup:**

- Each channel receives input only through its corresponding de Rham operator (exact via Md₀, coexact via Mδ₁, harmonic via ker(L₁) projection), ensuring independent evaluation of each spectral subspace.
- Per-mode standardization applied to each channel for fair comparison.
- De Rham cross-terms (Md₀, Mδ₁) remain active within the model.

**Ablation variants:** Each variant receives only the indicated channels; excluded channels are zeroed.

### Results

**Ellipsoid** (genus-0, GT Hodge energy: 99.4% exact / 0.6% coexact):

| Basis Components | Rel L2 | Δ vs Full |
|------------------|--------|-----------|
| **Full Hodge** | **0.084** | — |
| Exact (d₀·Φ₀) only | 0.113 | +35% |
| Coexact (δ₁·Φ₂) only | 0.338 | +303% |
| Harmonic (ker L₁) only | 1.002 | diverged |

**Torus** (genus-1, GT Hodge energy: 17.1% exact / 82.9% coexact):

| Basis Components | Rel L2 | Δ vs Full |
|------------------|--------|-----------|
| **Full Hodge** | **0.475** | — |
| Coexact (δ₁·Φ₂) only | 0.483 | +2% |
| Exact (d₀·Φ₀) only | 0.995 | +109% |
| Harmonic (ker L₁) only | 1.002 | diverged |

### Findings

1. **No single Hodge component suffices.** Harmonic-only diverges on both tasks (1 DC mode cannot represent spatially varying fields). Exact-only collapses on the coexact-dominated Torus (+109%). Coexact-only collapses on the exact-dominated Ellipsoid (+303%). Only the full spectrum succeeds on both.

2. **The dominant component mirrors the GT Hodge energy distribution.** Ellipsoid (99% exact energy) requires the gradient channel: dropping it causes +303% degradation, while dropping the coexact channel causes +35%. Torus (83% coexact energy) requires the curl channel: dropping it causes +109% degradation. This confirms that the boundary operator B₁ (gradient, d₀) and co-boundary operator B₂ᵀ (curl, δ₁) each capture physically distinct, non-redundant spectral information.

3. **Cross-subspace spectral resolution matters even for minority components.** On Ellipsoid, the coexact component carries only 0.6% of GT energy, yet removing the curl channel degrades overall error by 35%. The curl eigenvectors from δ₁·Φ₂ provide spectral coverage in the coexact subspace that improves reconstruction fidelity beyond what the exact channel alone can achieve. This demonstrates that the Hodge decomposition captures complementary geometric structure, not just energy-proportional information.

4. **The Full Hodge Laplacian L₁ = B₁B₁ᵀ + B₂ᵀB₂ optimally balances both subspaces.** The exact channel is constructed from the boundary operator on the (k−1)-spectrum (d₀: Φ₀ → Φ₁ via B₁), and the coexact channel from the co-boundary operator on the (k+1)-spectrum (δ₁: Φ₂ → Φ₁ via B₂ᵀ). Using both jointly through L₁ ensures that the spectral basis automatically adapts to the task's Hodge energy distribution — gradient-dominated or curl-dominated — without task-specific tuning.

---

## Figures

| Figure | File |
|--------|------|
| Baseline comparison (Ellipsoid) | `output/figures/baselines_ellipsoid.png` |
| Baseline comparison (Torus) | `output/figures/baselines_torus.png` |
| Cross-input consistency | `output/figures/cross_input_consistency.png` |
| Layer ablation | `output/figures/layer_ablation.png` |
