/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.CoveringNumber
import SLT.HansonWright
import Mathlib.Analysis.InnerProductSpace.LinearMap
import Mathlib.Analysis.InnerProductSpace.Rayleigh

/-!
# Random Matrix Theory Basics

This file starts the random-matrix infrastructure for HDP, Section 4.4.

## Main definitions

* `RMT.matrixOperatorNorm`: the Euclidean `2 → 2` operator norm of a rectangular matrix.
* `RMT.randomMatrix`: a random rectangular matrix assembled from scalar entries.
* `RMT.matrixLargestSingularValue`: the largest singular value `s₁(A)`.
* `RMT.matrixSmallestSingularValue`: the smallest domain-indexed singular value `sₙ(A)`.
* `RMT.maxFamilySubGaussianPsi2Norm`: maximum scalar ψ₂ scale over a finite family.
* `RMT.maxMatrixRowSubGaussianPsi2Norm`: maximum vector ψ₂ scale over the rows.
* `RMT.HasIndependentMeanZeroSubGaussianEntries`: the entry assumptions in HDP Theorem 4.4.3.
* `RMT.HasIndependentVarianceOneSubGaussianEntries`: the entry assumptions in Exercise 4.42.
* `RMT.HasIndependentMeanZeroSubGaussianUpperTriangle`: the symmetric entry assumptions in
  HDP Corollary 4.4.7.
* `RMT.HasIndependentMeanZeroIsotropicSubGaussianRows`: the row assumptions in
  HDP Theorem 4.6.1.

## Main results

* `RMT.norm_subgaussian_matrices_hdp`: the exact proposition stated by HDP Theorem 4.4.3.
* `RMT.norm_subgaussian_matrices_expectation_hdp`: the exact proposition stated by
  HDP Remark 4.4.4.
* `RMT.norm_random_matrices_lower_bound_hdp`: the exact proposition stated by Exercise 4.42.
* `RMT.norm_symmetric_subgaussian_matrices_hdp`: the exact proposition stated by
  HDP Corollary 4.4.7.
* `RMT.two_sided_subgaussian_matrices_hdp`: the exact proposition stated by
  HDP Theorem 4.6.1.
* `RMT.two_sided_subgaussian_matrices_expectation_hdp`: the exact proposition stated by
  HDP Remark 4.6.2.
-/

open MeasureTheory ProbabilityTheory Real Filter
open scoped BigOperators NNReal Topology

noncomputable section

namespace RMT

variable {Ω : Type*} [MeasurableSpace Ω]

/-! ## Deterministic rectangular matrices -/

/-- The Euclidean `2 → 2` operator norm of a rectangular real matrix. -/
def matrixOperatorNorm {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  ‖A.toEuclideanLin.toContinuousLinearMap‖

omit [MeasurableSpace Ω] in
/-- The zero-based singular values of a real matrix, as singular values of its Euclidean map. -/
def matrixSingularValue {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (k : ℕ) : ℝ :=
  A.toEuclideanLin.singularValues k

omit [MeasurableSpace Ω] in
/-- The largest singular value `s₁(A)` of a real matrix. -/
def matrixLargestSingularValue {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  matrixSingularValue A 0

omit [MeasurableSpace Ω] in
/-- The smallest domain-indexed singular value `sₙ(A)` of an `m × n` real matrix. -/
def matrixSmallestSingularValue {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  matrixSingularValue A (n - 1)

omit [MeasurableSpace Ω] in
lemma matrixSingularValue_nonneg {m n k : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    0 ≤ matrixSingularValue A k :=
  A.toEuclideanLin.singularValues_nonneg k

omit [MeasurableSpace Ω] in
lemma matrixLargestSingularValue_nonneg {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    0 ≤ matrixLargestSingularValue A :=
  matrixSingularValue_nonneg A

omit [MeasurableSpace Ω] in
lemma matrixSmallestSingularValue_nonneg {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    0 ≤ matrixSmallestSingularValue A :=
  matrixSingularValue_nonneg A

omit [MeasurableSpace Ω] in
lemma matrixSmallestSingularValue_le_largest {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    matrixSmallestSingularValue A ≤ matrixLargestSingularValue A := by
  unfold matrixSmallestSingularValue matrixLargestSingularValue matrixSingularValue
  exact A.toEuclideanLin.singularValues_antitone (Nat.zero_le (n - 1))

/-- A random rectangular matrix assembled from scalar entries. -/
def randomMatrix {m n : ℕ} (A : Fin m → Fin n → Ω → ℝ) (ω : Ω) :
    Matrix (Fin m) (Fin n) ℝ :=
  fun i j => A i j ω

omit [MeasurableSpace Ω] in
/-- The empirical covariance deviation `m⁻¹ AᵀA - Iₙ` appearing in HDP Theorem 4.6.1. -/
def sampleCovarianceDeviation {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  ((m : ℝ)⁻¹) • (A.conjTranspose * A) - 1

omit [MeasurableSpace Ω] in
/-- Operator norm of the empirical covariance deviation `m⁻¹ AᵀA - Iₙ`. -/
def sampleCovarianceDeviationOperatorNorm {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    ℝ :=
  matrixOperatorNorm (sampleCovarianceDeviation A)

omit [MeasurableSpace Ω] in
/-- The Euclidean linear map associated to `AᵀA` is `A† ∘ A`. -/
lemma conjTranspose_mul_self_toEuclideanLin {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    (A.conjTranspose * A).toEuclideanLin =
      LinearMap.adjoint A.toEuclideanLin ∘ₗ A.toEuclideanLin := by
  rw [← Matrix.toEuclideanLin_conjTranspose_eq_adjoint A]
  simpa [Matrix.toEuclideanLin_eq_toLin_orthonormal] using Matrix.toLin_mul
    (v₁ := (EuclideanSpace.basisFun (Fin n) ℝ).toBasis)
    (v₂ := (EuclideanSpace.basisFun (Fin m) ℝ).toBasis)
    (v₃ := (EuclideanSpace.basisFun (Fin n) ℝ).toBasis) A.conjTranspose A

omit [MeasurableSpace Ω] in
lemma inner_conjTranspose_mul_self_toEuclideanLin {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (x : EuclideanSpace ℝ (Fin n)) :
    inner ℝ ((A.conjTranspose * A).toEuclideanLin x) x =
      ‖A.toEuclideanLin x‖ ^ 2 := by
  rw [conjTranspose_mul_self_toEuclideanLin]
  rw [LinearMap.comp_apply]
  rw [LinearMap.adjoint_inner_left]
  rw [real_inner_self_eq_norm_sq]

omit [MeasurableSpace Ω] in
lemma inner_sampleCovarianceDeviation_toEuclideanLin {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (x : EuclideanSpace ℝ (Fin n)) :
    inner ℝ ((sampleCovarianceDeviation A).toEuclideanLin x) x =
      (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2 := by
  unfold sampleCovarianceDeviation
  change inner ℝ
      (WithLp.toLp 2 (((((m : ℝ)⁻¹ • (A.conjTranspose * A) - 1).mulVec x.ofLp)))) x =
    (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2
  rw [Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec]
  rw [PiLp.inner_apply]
  simp only [Pi.sub_apply, Pi.smul_apply, RCLike.inner_apply, conj_trivial, smul_eq_mul]
  have hgram :
      ∑ i : Fin n, (m : ℝ)⁻¹ *
          (((A.conjTranspose * A).mulVec x.ofLp) i) * x.ofLp i =
        (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 := by
    calc
      ∑ i : Fin n, (m : ℝ)⁻¹ *
          (((A.conjTranspose * A).mulVec x.ofLp) i) * x.ofLp i
          = (m : ℝ)⁻¹ * ∑ i : Fin n,
              (((A.conjTranspose * A).mulVec x.ofLp) i) * x.ofLp i := by
              rw [Finset.mul_sum]
              apply Finset.sum_congr rfl
              intro i _
              ring
      _ = (m : ℝ)⁻¹ * inner ℝ ((A.conjTranspose * A).toEuclideanLin x) x := by
              congr 1
              rw [show ((A.conjTranspose * A).toEuclideanLin x) =
                WithLp.toLp 2 (((A.conjTranspose * A).mulVec x.ofLp)) by rfl]
              rw [PiLp.inner_apply]
              simp only [RCLike.inner_apply, conj_trivial]
              apply Finset.sum_congr rfl
              intro i _
              ring
      _ = (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 := by
              rw [inner_conjTranspose_mul_self_toEuclideanLin]
  have hx_sq :
      ∑ i : Fin n, x.ofLp i * x.ofLp i = ‖x‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
    apply Finset.sum_congr rfl
    intro i _
    ring
  calc
    ∑ x_1 : Fin n,
        x.ofLp x_1 *
        ((m : ℝ)⁻¹ * (((A.conjTranspose * A).mulVec x.ofLp) x_1) - x.ofLp x_1)
        = ∑ i : Fin n,
            (m : ℝ)⁻¹ * (((A.conjTranspose * A).mulVec x.ofLp) i) * x.ofLp i -
          ∑ i : Fin n, x.ofLp i * x.ofLp i := by
            rw [← Finset.sum_sub_distrib]
            apply Finset.sum_congr rfl
            intro i _
            ring
    _ = (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2 := by
          rw [hgram, hx_sq]
    _ = (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2 := rfl

omit [MeasurableSpace Ω] in
lemma sampleCovarianceDeviation_isSymm {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    (sampleCovarianceDeviation A).IsSymm := by
  have hgram : (A.conjTranspose * A).IsSymm := by
    rw [← Matrix.isHermitian_iff_isSymm]
    exact Matrix.isHermitian_conjTranspose_mul_self A
  refine Matrix.IsSymm.ext ?_
  intro i j
  have hji : (A.conjTranspose * A) j i = (A.conjTranspose * A) i j :=
    Matrix.IsSymm.apply hgram i j
  have hji' : (A.transpose * A) j i = (A.transpose * A) i j := by
    simpa using hji
  have hone : (1 : Matrix (Fin n) (Fin n) ℝ) j i = (1 : Matrix (Fin n) (Fin n) ℝ) i j := by
    by_cases hij : i = j
    · subst j
      rfl
    · have hji_ne : j ≠ i := fun h => hij h.symm
      simp [Matrix.one_apply_ne hij, Matrix.one_apply_ne hji_ne]
  change (((m : ℝ)⁻¹ • (A.conjTranspose * A) -
      (1 : Matrix (Fin n) (Fin n) ℝ)) j i) =
    (((m : ℝ)⁻¹ • (A.conjTranspose * A) -
      (1 : Matrix (Fin n) (Fin n) ℝ)) i j)
  calc
    (((m : ℝ)⁻¹ • (A.conjTranspose * A) -
        (1 : Matrix (Fin n) (Fin n) ℝ)) j i)
        = (m : ℝ)⁻¹ * (A.transpose * A) j i -
            (1 : Matrix (Fin n) (Fin n) ℝ) j i := by simp
    _ = (m : ℝ)⁻¹ * (A.transpose * A) i j -
            (1 : Matrix (Fin n) (Fin n) ℝ) i j := by rw [hji', hone]
    _ = (((m : ℝ)⁻¹ • (A.conjTranspose * A) -
        (1 : Matrix (Fin n) (Fin n) ℝ)) i j) := by simp

omit [MeasurableSpace Ω] in
lemma abs_inner_toEuclideanLin_le_matrixOperatorNorm_mul_norm_sq {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (x : EuclideanSpace ℝ (Fin n)) :
    |inner ℝ (A.toEuclideanLin x) x| ≤ matrixOperatorNorm A * ‖x‖ ^ 2 := by
  have hAx :
      ‖A.toEuclideanLin x‖ ≤ matrixOperatorNorm A * ‖x‖ := by
    simpa [matrixOperatorNorm] using A.toEuclideanLin.toContinuousLinearMap.le_opNorm x
  calc
    |inner ℝ (A.toEuclideanLin x) x| ≤ ‖A.toEuclideanLin x‖ * ‖x‖ :=
      abs_real_inner_le_norm _ _
    _ ≤ (matrixOperatorNorm A * ‖x‖) * ‖x‖ :=
      mul_le_mul_of_nonneg_right hAx (norm_nonneg x)
    _ = matrixOperatorNorm A * ‖x‖ ^ 2 := by ring

omit [MeasurableSpace Ω] in
lemma abs_inv_mul_sq_matrixSingularValue_sub_one_le_sampleCovarianceDeviationOperatorNorm
    {m n k : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (hk : k < n) :
    |(m : ℝ)⁻¹ * matrixSingularValue A k ^ 2 - 1| ≤
      sampleCovarianceDeviationOperatorNorm A := by
  let T := A.toEuclideanLin
  have hfin : Module.finrank ℝ (EuclideanSpace ℝ (Fin n)) = n := by
    simp
  have hkfin : k < Module.finrank ℝ (EuclideanSpace ℝ (Fin n)) := by
    simpa [hfin] using hk
  have heig :
      Module.End.HasEigenvalue (LinearMap.adjoint T ∘ₗ T) (T.singularValues k ^ 2) :=
    T.hasEigenvalue_adjoint_comp_self_sq_singularValues hkfin
  rcases heig.exists_hasEigenvector with ⟨v, hv⟩
  let r : ℝ := ‖v‖
  let y : EuclideanSpace ℝ (Fin n) := r⁻¹ • v
  have hv_ne : v ≠ 0 := hv.2
  have hr_pos : 0 < r := by
    dsimp [r]
    exact norm_pos_iff.mpr hv_ne
  have hy_norm : ‖y‖ = 1 := by
    dsimp [y]
    rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hr_pos.le)]
    exact inv_mul_cancel₀ hr_pos.ne'
  have hy_eig : (LinearMap.adjoint T ∘ₗ T) y = (T.singularValues k ^ 2) • y := by
    calc
      (LinearMap.adjoint T ∘ₗ T) y
          = r⁻¹ • ((LinearMap.adjoint T ∘ₗ T) v) := by
            dsimp [y]
            rw [map_smul, map_smul]
      _ = r⁻¹ • ((T.singularValues k ^ 2) • v) := by
            rw [hv.apply_eq_smul]
      _ = (T.singularValues k ^ 2) • y := by
            dsimp [y]
            rw [smul_smul, smul_smul]
            congr 1
            ring
  have hinner_adjoint :
      inner ℝ ((LinearMap.adjoint T ∘ₗ T) y) y = ‖T y‖ ^ 2 := by
    rw [LinearMap.comp_apply, LinearMap.adjoint_inner_left, real_inner_self_eq_norm_sq]
  have hTy_sq : ‖T y‖ ^ 2 = matrixSingularValue A k ^ 2 := by
    calc
      ‖T y‖ ^ 2 = inner ℝ ((LinearMap.adjoint T ∘ₗ T) y) y := hinner_adjoint.symm
      _ = T.singularValues k ^ 2 := by
        rw [hy_eig, inner_smul_left, real_inner_self_eq_norm_sq, hy_norm]
        simp
      _ = matrixSingularValue A k ^ 2 := by
        rfl
  have hquad :=
    abs_inner_toEuclideanLin_le_matrixOperatorNorm_mul_norm_sq
      (sampleCovarianceDeviation A) y
  rw [inner_sampleCovarianceDeviation_toEuclideanLin, hTy_sq, hy_norm] at hquad
  simpa [sampleCovarianceDeviationOperatorNorm] using hquad

omit [MeasurableSpace Ω] in
lemma matrixSingularValue_sq_bounds_of_sampleCovarianceDeviationOperatorNorm_le
    {m n k : ℕ} (hm : 0 < m) (A : Matrix (Fin m) (Fin n) ℝ)
    {δ : ℝ} (hδ : sampleCovarianceDeviationOperatorNorm A ≤ δ) (hk : k < n) :
    (m : ℝ) * (1 - δ) ≤ matrixSingularValue A k ^ 2 ∧
      matrixSingularValue A k ^ 2 ≤ (m : ℝ) * (1 + δ) := by
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have habs :=
    (abs_inv_mul_sq_matrixSingularValue_sub_one_le_sampleCovarianceDeviationOperatorNorm
      (m := m) (n := n) (k := k) A hk).trans hδ
  have hbounds := abs_le.mp habs
  constructor
  · have hlower : 1 - δ ≤ (m : ℝ)⁻¹ * matrixSingularValue A k ^ 2 := by
      linarith
    calc
      (m : ℝ) * (1 - δ) ≤
          (m : ℝ) * ((m : ℝ)⁻¹ * matrixSingularValue A k ^ 2) :=
        mul_le_mul_of_nonneg_left hlower hmR_pos.le
      _ = matrixSingularValue A k ^ 2 := by
        field_simp [hmR_pos.ne']
  · have hupper : (m : ℝ)⁻¹ * matrixSingularValue A k ^ 2 ≤ 1 + δ := by
      linarith
    calc
      matrixSingularValue A k ^ 2 =
          (m : ℝ) * ((m : ℝ)⁻¹ * matrixSingularValue A k ^ 2) := by
        field_simp [hmR_pos.ne']
      _ ≤ (m : ℝ) * (1 + δ) :=
        mul_le_mul_of_nonneg_left hupper hmR_pos.le

omit [MeasurableSpace Ω] in
lemma matrixSingularValue_bounds_of_sampleCovarianceDeviationOperatorNorm_le_max
    {m n k : ℕ} (hm : 0 < m) (A : Matrix (Fin m) (Fin n) ℝ)
    {B : ℝ} (hB : 0 ≤ B)
    (hdev : sampleCovarianceDeviationOperatorNorm A ≤
      max (B / √(m : ℝ)) ((B / √(m : ℝ)) ^ 2))
    (hk : k < n) :
    √(m : ℝ) - B ≤ matrixSingularValue A k ∧
      matrixSingularValue A k ≤ √(m : ℝ) + B := by
  let a : ℝ := √(m : ℝ)
  let s : ℝ := matrixSingularValue A k
  let ε : ℝ := B / a
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have ha_pos : 0 < a := by
    dsimp [a]
    exact sqrt_pos.mpr hmR_pos
  have ha_nonneg : 0 ≤ a := ha_pos.le
  have ha_sq : a ^ 2 = (m : ℝ) := by
    dsimp [a]
    exact sq_sqrt hmR_pos.le
  have hε_nonneg : 0 ≤ ε := div_nonneg hB ha_nonneg
  have hsq_bounds :=
    matrixSingularValue_sq_bounds_of_sampleCovarianceDeviationOperatorNorm_le
      (m := m) (n := n) (k := k) hm A hdev hk
  have hsq_lower : (m : ℝ) * (1 - max ε (ε ^ 2)) ≤ s ^ 2 := by
    simpa [s, ε, a] using hsq_bounds.1
  have hsq_upper : s ^ 2 ≤ (m : ℝ) * (1 + max ε (ε ^ 2)) := by
    simpa [s, ε, a] using hsq_bounds.2
  have hs_nonneg : 0 ≤ s := by
    dsimp [s]
    exact matrixSingularValue_nonneg A
  have hupper_sq : s ^ 2 ≤ (a + B) ^ 2 := by
    by_cases hBa : B ≤ a
    · have hε_le_one : ε ≤ 1 := by
        dsimp [ε]
        exact div_le_one_of_le₀ hBa ha_nonneg
      have hε_sq_le : ε ^ 2 ≤ ε := by nlinarith [hε_nonneg, hε_le_one]
      have hmax : max ε (ε ^ 2) = ε := max_eq_left hε_sq_le
      have hmul : (m : ℝ) * max ε (ε ^ 2) ≤ (m : ℝ) * ε :=
        mul_le_mul_of_nonneg_left (by rw [hmax]) hmR_pos.le
      calc
        s ^ 2 ≤ (m : ℝ) * (1 + max ε (ε ^ 2)) := hsq_upper
        _ ≤ a ^ 2 * (1 + ε) := by
              rw [ha_sq]
              nlinarith
        _ = a ^ 2 + a * B := by
              dsimp [ε]
              field_simp [ha_pos.ne']
        _ ≤ (a + B) ^ 2 := by
              nlinarith [ha_nonneg, hB, sq_nonneg B]
    · have ha_le_B : a ≤ B := le_of_not_ge hBa
      have hε_ge_one : 1 ≤ ε := by
        dsimp [ε]
        exact (one_le_div ha_pos).mpr ha_le_B
      have hε_le_sq : ε ≤ ε ^ 2 := by nlinarith [hε_nonneg, hε_ge_one]
      have hmax : max ε (ε ^ 2) = ε ^ 2 := max_eq_right hε_le_sq
      calc
        s ^ 2 ≤ (m : ℝ) * (1 + max ε (ε ^ 2)) := hsq_upper
        _ ≤ a ^ 2 * (1 + ε ^ 2) := by
              rw [ha_sq, hmax]
        _ = a ^ 2 + B ^ 2 := by
              dsimp [ε]
              field_simp [ha_pos.ne']
        _ ≤ (a + B) ^ 2 := by
              nlinarith [ha_nonneg, hB]
  have hupper : s ≤ a + B :=
    le_of_sq_le_sq hupper_sq (add_nonneg ha_nonneg hB)
  have hlower : a - B ≤ s := by
    by_cases hBa : B ≤ a
    · have hε_le_one : ε ≤ 1 := by
        dsimp [ε]
        exact div_le_one_of_le₀ hBa ha_nonneg
      have hε_sq_le : ε ^ 2 ≤ ε := by nlinarith [hε_nonneg, hε_le_one]
      have hmax : max ε (ε ^ 2) = ε := max_eq_left hε_sq_le
      have hlower_sq : (a - B) ^ 2 ≤ s ^ 2 := by
        calc
          (a - B) ^ 2 ≤ a ^ 2 * (1 - ε) := by
                dsimp [ε]
                field_simp [ha_pos.ne']
                nlinarith [hB, hBa, ha_nonneg]
          _ = (m : ℝ) * (1 - max ε (ε ^ 2)) := by
                rw [ha_sq, hmax]
          _ ≤ s ^ 2 := hsq_lower
      exact le_of_sq_le_sq hlower_sq hs_nonneg
    · have hle_zero : a - B ≤ 0 := by
        exact sub_nonpos.mpr (le_of_not_ge hBa)
      exact hle_zero.trans hs_nonneg
  exact ⟨by simpa [a, s] using hlower, by simpa [a, s] using hupper⟩

omit [MeasurableSpace Ω] in
lemma singular_value_sandwich_of_sampleCovarianceDeviationOperatorNorm_le_max
    {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (A : Matrix (Fin m) (Fin n) ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hdev : sampleCovarianceDeviationOperatorNorm A ≤
      max (B / √(m : ℝ)) ((B / √(m : ℝ)) ^ 2)) :
    √(m : ℝ) - B ≤ matrixSmallestSingularValue A ∧
      matrixSmallestSingularValue A ≤ matrixLargestSingularValue A ∧
        matrixLargestSingularValue A ≤ √(m : ℝ) + B := by
  have hsmall :=
    matrixSingularValue_bounds_of_sampleCovarianceDeviationOperatorNorm_le_max
      (m := m) (n := n) (k := n - 1) hm A hB hdev (by omega)
  have hlarge :=
    matrixSingularValue_bounds_of_sampleCovarianceDeviationOperatorNorm_le_max
      (m := m) (n := n) (k := 0) hm A hB hdev hn
  exact ⟨by simpa [matrixSmallestSingularValue] using hsmall.1,
    matrixSmallestSingularValue_le_largest A,
    by simpa [matrixLargestSingularValue] using hlarge.2⟩

omit [MeasurableSpace Ω] in
/-- The `j`-th column of a real matrix as a Euclidean vector. -/
def matrixColumnVector {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (j : Fin n) :
    EuclideanSpace ℝ (Fin m) :=
  WithLp.toLp 2 fun i => A i j

omit [MeasurableSpace Ω] in
/-- The `i`-th row of a real matrix as a Euclidean vector. -/
def matrixRowVector {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) :
    EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 fun j => A i j

omit [MeasurableSpace Ω] in
lemma matrixColumnVector_randomMatrix_eq_randomVector {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (ω : Ω) (j : Fin n) :
    matrixColumnVector (randomMatrix A ω) j =
      HansonWright.randomVector (fun i => A i j) ω := by
  rfl

omit [MeasurableSpace Ω] in
lemma matrixRowVector_randomMatrix_eq_randomVector {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (ω : Ω) (i : Fin m) :
    matrixRowVector (randomMatrix A ω) i =
      HansonWright.randomVector (fun j => A i j) ω := by
  rfl

omit [MeasurableSpace Ω] in
@[simp]
lemma randomMatrix_apply {m n : ℕ} (A : Fin m → Fin n → Ω → ℝ) (ω : Ω)
    (i : Fin m) (j : Fin n) :
    randomMatrix A ω i j = A i j ω :=
  rfl

/-- If all unit-vector bilinear forms of a rectangular matrix are bounded by `B`, then its
operator norm is bounded by `B`. -/
lemma matrixOperatorNorm_le_of_forall_unit_abs_inner_le {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) {B : ℝ} (hB : 0 ≤ B)
    (h : ∀ x : EuclideanSpace ℝ (Fin n), ∀ y : EuclideanSpace ℝ (Fin m),
      ‖x‖ = 1 → ‖y‖ = 1 → |inner ℝ (A.toEuclideanLin x) y| ≤ B) :
    matrixOperatorNorm A ≤ B := by
  unfold matrixOperatorNorm
  refine ContinuousLinearMap.opNorm_le_of_re_inner_le hB ?_
  intro x y hx hy
  exact (le_abs_self _).trans (h x y hx hy)

/-! ## Matrix-specific nets -/

/-- The Euclidean unit sphere in `ℝ^n`. -/
def euclideanUnitSphere (n : ℕ) : Set (EuclideanSpace ℝ (Fin n)) :=
  Metric.sphere 0 1

/-- The Euclidean unit ball in `ℝ^n`. -/
def euclideanUnitBall (n : ℕ) : Set (EuclideanSpace ℝ (Fin n)) :=
  Metric.closedBall 0 1

/--
A pair of finite nets used to discretize the bilinear form
`(x, y) ↦ ⟪A x, y⟫` for an `m × n` matrix.  The underlying net notion is the generic
`IsENet` from `SLT.CoveringNumber`; this structure only packages the domain and codomain nets
needed for rectangular matrix operator norms.
-/
structure MatrixBilinearNet (m n : ℕ) (ε : ℝ) where
  domainNet : Finset (EuclideanSpace ℝ (Fin n))
  codomainNet : Finset (EuclideanSpace ℝ (Fin m))
  domain_isNet : IsENet domainNet ε (euclideanUnitSphere n)
  codomain_isNet : IsENet codomainNet ε (euclideanUnitSphere m)

/--
A centered matrix bilinear net: the net points themselves lie in the corresponding unit balls.
This is the form naturally produced by finite coverings of the unit balls and used in the
Section 4.4 net argument.
-/
structure CenteredMatrixBilinearNet (m n : ℕ) (ε : ℝ) extends
    MatrixBilinearNet m n ε where
  domain_subset_unitBall : (domainNet : Set (EuclideanSpace ℝ (Fin n))) ⊆ euclideanUnitBall n
  codomain_subset_unitBall : (codomainNet : Set (EuclideanSpace ℝ (Fin m))) ⊆
    euclideanUnitBall m

omit [MeasurableSpace Ω] in
lemma norm_eq_one_of_mem_euclideanUnitSphere {n : ℕ}
    {x : EuclideanSpace ℝ (Fin n)} (hx : x ∈ euclideanUnitSphere n) :
    ‖x‖ = 1 := by
  simpa [euclideanUnitSphere, dist_eq_norm] using hx

omit [MeasurableSpace Ω] in
lemma norm_le_one_of_mem_euclideanUnitBall {n : ℕ}
    {x : EuclideanSpace ℝ (Fin n)} (hx : x ∈ euclideanUnitBall n) :
    ‖x‖ ≤ 1 := by
  simpa [euclideanUnitBall, dist_eq_norm] using hx

omit [MeasurableSpace Ω] in
lemma euclideanUnitSphere_nonempty_of_pos {n : ℕ} (hn : 0 < n) :
    (euclideanUnitSphere n).Nonempty := by
  let i : Fin n := ⟨0, hn⟩
  refine ⟨EuclideanSpace.single i (1 : ℝ), ?_⟩
  simp [euclideanUnitSphere]

omit [MeasurableSpace Ω] in
lemma MatrixBilinearNet.exists_domain_near {m n : ℕ} {ε : ℝ}
    (N : MatrixBilinearNet m n ε) {x : EuclideanSpace ℝ (Fin n)}
    (hx : x ∈ euclideanUnitSphere n) :
    ∃ x₀ ∈ N.domainNet, ‖x - x₀‖ ≤ ε := by
  obtain ⟨x₀, hx₀, hball⟩ := Set.mem_iUnion₂.mp (N.domain_isNet hx)
  exact ⟨x₀, hx₀, by simpa [dist_eq_norm] using Metric.mem_closedBall.mp hball⟩

omit [MeasurableSpace Ω] in
lemma MatrixBilinearNet.exists_codomain_near {m n : ℕ} {ε : ℝ}
    (N : MatrixBilinearNet m n ε) {y : EuclideanSpace ℝ (Fin m)}
    (hy : y ∈ euclideanUnitSphere m) :
    ∃ y₀ ∈ N.codomainNet, ‖y - y₀‖ ≤ ε := by
  obtain ⟨y₀, hy₀, hball⟩ := Set.mem_iUnion₂.mp (N.codomain_isNet hy)
  exact ⟨y₀, hy₀, by simpa [dist_eq_norm] using Metric.mem_closedBall.mp hball⟩

omit [MeasurableSpace Ω] in
lemma MatrixBilinearNet.domainNet_nonempty_of_pos {m n : ℕ} {ε : ℝ}
    (N : MatrixBilinearNet m n ε) (hn : 0 < n) :
    N.domainNet.Nonempty := by
  obtain ⟨x, hx⟩ := euclideanUnitSphere_nonempty_of_pos (n := n) hn
  obtain ⟨x₀, hx₀, _⟩ := N.exists_domain_near hx
  exact ⟨x₀, hx₀⟩

omit [MeasurableSpace Ω] in
lemma MatrixBilinearNet.codomainNet_nonempty_of_pos {m n : ℕ} {ε : ℝ}
    (N : MatrixBilinearNet m n ε) (hm : 0 < m) :
    N.codomainNet.Nonempty := by
  obtain ⟨y, hy⟩ := euclideanUnitSphere_nonempty_of_pos (n := m) hm
  obtain ⟨y₀, hy₀, _⟩ := N.exists_codomain_near hy
  exact ⟨y₀, hy₀⟩

omit [MeasurableSpace Ω] in
lemma exists_centeredMatrixBilinearNet {m n : ℕ} {ε : ℝ} (hε : 0 < ε) :
    ∃ N : CenteredMatrixBilinearNet m n ε,
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (ε / 2) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n))) ∧
      (N.codomainNet.card : WithTop ℕ) ≤
        coveringNumber (ε / 2) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m))) := by
  obtain ⟨domainNet, hdomain_net_ball, hdomain_subset, hdomain_card⟩ :=
    exists_enet_subset_from_half (A := EuclideanSpace ℝ (Fin n)) (eps := ε)
      (s := (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n)))) hε
      (euclideanBall_totallyBounded (ι := Fin n) 1)
      (euclideanBall_nonempty (ι := Fin n) (by norm_num : (0 : ℝ) ≤ 1))
  obtain ⟨codomainNet, hcodomain_net_ball, hcodomain_subset, hcodomain_card⟩ :=
    exists_enet_subset_from_half (A := EuclideanSpace ℝ (Fin m)) (eps := ε)
      (s := (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m)))) hε
      (euclideanBall_totallyBounded (ι := Fin m) 1)
      (euclideanBall_nonempty (ι := Fin m) (by norm_num : (0 : ℝ) ≤ 1))
  refine ⟨
    { domainNet := domainNet
      codomainNet := codomainNet
      domain_isNet := ?_
      codomain_isNet := ?_
      domain_subset_unitBall := ?_
      codomain_subset_unitBall := ?_ },
    hdomain_card, hcodomain_card⟩
  · intro x hx
    exact hdomain_net_ball (by
      have hxnorm := norm_eq_one_of_mem_euclideanUnitSphere hx
      simpa [euclideanBall, Metric.mem_closedBall, dist_eq_norm] using hxnorm.le)
  · intro y hy
    exact hcodomain_net_ball (by
      have hynorm := norm_eq_one_of_mem_euclideanUnitSphere hy
      simpa [euclideanBall, Metric.mem_closedBall, dist_eq_norm] using hynorm.le)
  · simpa [euclideanUnitBall, euclideanBall] using hdomain_subset
  · simpa [euclideanUnitBall, euclideanBall] using hcodomain_subset

omit [MeasurableSpace Ω] in
lemma exists_quarter_centeredMatrixBilinearNet (m n : ℕ) :
    ∃ N : CenteredMatrixBilinearNet m n (1 / 4),
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n))) ∧
      (N.codomainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m))) := by
  obtain ⟨N, hdomain, hcodomain⟩ :=
    exists_centeredMatrixBilinearNet (m := m) (n := n)
      (by norm_num : 0 < (1 / 4 : ℝ))
  refine ⟨N, ?_, ?_⟩
  · simpa [show (4 : ℝ)⁻¹ / 2 = (8 : ℝ)⁻¹ by norm_num] using hdomain
  · simpa [show (4 : ℝ)⁻¹ / 2 = (8 : ℝ)⁻¹ by norm_num] using hcodomain

omit [MeasurableSpace Ω] in
lemma finset_card_le_seventeen_pow_of_card_le_euclideanBall_coveringNumber {d : ℕ}
    (hd : 0 < d) {t : Finset (EuclideanSpace ℝ (Fin d))}
    (ht :
      (t.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin d)))) :
    (t.card : ℝ) ≤ 17 ^ d := by
  haveI : Nonempty (Fin d) := ⟨⟨0, hd⟩⟩
  have hcover_top :
      coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin d))) < ⊤ :=
    coveringNumber_lt_top_of_totallyBounded (by norm_num : 0 < (1 / 8 : ℝ))
      (euclideanBall_totallyBounded (ι := Fin d) 1)
  let hcover_ne_top :
      coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin d))) ≠ ⊤ :=
    ne_top_of_lt hcover_top
  have ht_nat :
      t.card ≤
        (coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin d)))).untop
          hcover_ne_top := by
    rw [← WithTop.coe_le_coe]
    rw [WithTop.coe_untop]
    exact ht
  have hcover_bound :
      (((coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin d)))).untop
          hcover_ne_top : ℕ) : ℝ) ≤ 17 ^ d := by
    have h :=
      coveringNumber_euclideanBall_le (ι := Fin d) (R := 1) (eps := 1 / 8)
        (by norm_num : (0 : ℝ) ≤ 1) (by norm_num : 0 < (1 / 8 : ℝ))
    norm_num at h
    simpa [Fintype.card_fin] using h
  have ht_real :
      (t.card : ℝ) ≤
        (((coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin d)))).untop
          hcover_ne_top : ℕ) : ℝ) := by
    exact_mod_cast ht_nat
  exact ht_real.trans hcover_bound

omit [MeasurableSpace Ω] in
lemma centeredMatrixBilinearNet_domain_card_le_seventeen_pow {m n : ℕ} (hn : 0 < n)
    (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN :
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n)))) :
    (N.domainNet.card : ℝ) ≤ 17 ^ n :=
  finset_card_le_seventeen_pow_of_card_le_euclideanBall_coveringNumber hn hN

omit [MeasurableSpace Ω] in
lemma centeredMatrixBilinearNet_codomain_card_le_seventeen_pow {m n : ℕ} (hm : 0 < m)
    (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN :
      (N.codomainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m)))) :
    (N.codomainNet.card : ℝ) ≤ 17 ^ m :=
  finset_card_le_seventeen_pow_of_card_le_euclideanBall_coveringNumber hm hN

omit [MeasurableSpace Ω] in
lemma centeredMatrixBilinearNet_card_product_le_seventeen_pow_add {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN_domain :
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n))))
    (hN_codomain :
      (N.codomainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m)))) :
    (((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ)) ≤ 17 ^ (m + n) := by
  have hdomain :=
    centeredMatrixBilinearNet_domain_card_le_seventeen_pow hn N hN_domain
  have hcodomain :=
    centeredMatrixBilinearNet_codomain_card_le_seventeen_pow hm N hN_codomain
  calc
    (((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ))
        = (N.domainNet.card : ℝ) * (N.codomainNet.card : ℝ) := by norm_num
    _ ≤ 17 ^ n * 17 ^ m := by
          exact mul_le_mul hdomain hcodomain (Nat.cast_nonneg _)
            (pow_nonneg (by norm_num : (0 : ℝ) ≤ 17) n)
    _ = 17 ^ (m + n) := by
          rw [pow_add]
          ring

/--
Deterministic net reduction for rectangular operator norms. If all bilinear forms
`⟪Ax, y⟫` are bounded by `u` on a centered `ε`-net of the two unit spheres, then the operator
norm is bounded by `u / (1 - 2ε)`.
-/
lemma matrixOperatorNorm_le_of_centered_bilinear_net {m n : ℕ} {ε u : ℝ}
    (A : Matrix (Fin m) (Fin n) ℝ) (N : CenteredMatrixBilinearNet m n ε)
    (hε_nonneg : 0 ≤ ε) (hε : 2 * ε < 1) (hu : 0 ≤ u)
    (hnet : ∀ x ∈ N.domainNet, ∀ y ∈ N.codomainNet,
      |inner ℝ (A.toEuclideanLin x) y| ≤ u) :
    matrixOperatorNorm A ≤ u / (1 - 2 * ε) := by
  let M := matrixOperatorNorm A
  have hM_nonneg : 0 ≤ M := norm_nonneg _
  have hden : 0 < 1 - 2 * ε := by linarith
  have hstep : M ≤ u + 2 * ε * M := by
    unfold M
    refine matrixOperatorNorm_le_of_forall_unit_abs_inner_le A
      (add_nonneg hu (mul_nonneg (mul_nonneg (by norm_num) hε_nonneg) hM_nonneg)) ?_
    intro x y hx hy
    have hx_sphere : x ∈ euclideanUnitSphere n := by
      simpa [euclideanUnitSphere, dist_eq_norm] using hx
    have hy_sphere : y ∈ euclideanUnitSphere m := by
      simpa [euclideanUnitSphere, dist_eq_norm] using hy
    obtain ⟨x₀, hx₀_mem, hx₀_dist⟩ := N.toMatrixBilinearNet.exists_domain_near hx_sphere
    obtain ⟨y₀, hy₀_mem, hy₀_dist⟩ := N.toMatrixBilinearNet.exists_codomain_near hy_sphere
    have hx₀_norm : ‖x₀‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hx₀_mem)
    have hy₀_norm : ‖y₀‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.codomain_subset_unitBall hy₀_mem)
    have hAx₀y₀ : |inner ℝ (A.toEuclideanLin x₀) y₀| ≤ u :=
      hnet x₀ hx₀_mem y₀ hy₀_mem
    have hA_dx : ‖A.toEuclideanLin (x - x₀)‖ ≤ M * ‖x - x₀‖ := by
      simpa [M, matrixOperatorNorm] using
        A.toEuclideanLin.toContinuousLinearMap.le_opNorm (x - x₀)
    have hA_x : ‖A.toEuclideanLin x‖ ≤ M * ‖x‖ := by
      simpa [M, matrixOperatorNorm] using A.toEuclideanLin.toContinuousLinearMap.le_opNorm x
    have hterm_x :
        |inner ℝ (A.toEuclideanLin (x - x₀)) y₀| ≤ M * ε := by
      calc
        |inner ℝ (A.toEuclideanLin (x - x₀)) y₀|
            ≤ ‖A.toEuclideanLin (x - x₀)‖ * ‖y₀‖ := abs_real_inner_le_norm _ _
        _ ≤ (M * ‖x - x₀‖) * 1 := by
              exact mul_le_mul hA_dx hy₀_norm (norm_nonneg _) (mul_nonneg hM_nonneg (norm_nonneg _))
        _ = M * ‖x - x₀‖ := by ring
        _ ≤ M * ε := mul_le_mul_of_nonneg_left hx₀_dist hM_nonneg
    have hterm_y :
        |inner ℝ (A.toEuclideanLin x) (y - y₀)| ≤ M * ε := by
      calc
        |inner ℝ (A.toEuclideanLin x) (y - y₀)|
            ≤ ‖A.toEuclideanLin x‖ * ‖y - y₀‖ := abs_real_inner_le_norm _ _
        _ ≤ (M * ‖x‖) * ε := by
              exact mul_le_mul hA_x hy₀_dist (norm_nonneg _)
                (mul_nonneg hM_nonneg (norm_nonneg _))
        _ = M * ε := by rw [hx]; ring
    have hdecomp :
        inner ℝ (A.toEuclideanLin x) y =
          inner ℝ (A.toEuclideanLin x₀) y₀ +
            inner ℝ (A.toEuclideanLin (x - x₀)) y₀ +
              inner ℝ (A.toEuclideanLin x) (y - y₀) := by
      have hx_decomp : x₀ + (x - x₀) = x := by abel
      have hy_decomp : y₀ + (y - y₀) = y := by abel
      calc
        inner ℝ (A.toEuclideanLin x) y =
            inner ℝ (A.toEuclideanLin x) (y₀ + (y - y₀)) := by rw [hy_decomp]
        _ = inner ℝ (A.toEuclideanLin x) y₀ +
              inner ℝ (A.toEuclideanLin x) (y - y₀) := by rw [inner_add_right]
        _ = inner ℝ (A.toEuclideanLin (x₀ + (x - x₀))) y₀ +
              inner ℝ (A.toEuclideanLin x) (y - y₀) := by rw [hx_decomp]
        _ = inner ℝ (A.toEuclideanLin x₀ + A.toEuclideanLin (x - x₀)) y₀ +
              inner ℝ (A.toEuclideanLin x) (y - y₀) := by rw [map_add]
        _ = inner ℝ (A.toEuclideanLin x₀) y₀ +
              inner ℝ (A.toEuclideanLin (x - x₀)) y₀ +
                inner ℝ (A.toEuclideanLin x) (y - y₀) := by rw [inner_add_left]
    calc
      |inner ℝ (A.toEuclideanLin x) y|
          ≤ |inner ℝ (A.toEuclideanLin x₀) y₀| +
              |inner ℝ (A.toEuclideanLin (x - x₀)) y₀| +
                |inner ℝ (A.toEuclideanLin x) (y - y₀)| := by
            rw [hdecomp]
            have h₁ := abs_add_le
              (inner ℝ (A.toEuclideanLin x₀) y₀ +
                inner ℝ (A.toEuclideanLin (x - x₀)) y₀)
              (inner ℝ (A.toEuclideanLin x) (y - y₀))
            have h₂ := abs_add_le (inner ℝ (A.toEuclideanLin x₀) y₀)
              (inner ℝ (A.toEuclideanLin (x - x₀)) y₀)
            linarith
      _ ≤ u + M * ε + M * ε := by linarith
      _ = u + 2 * ε * M := by ring
  have hmul : (1 - 2 * ε) * M ≤ u := by nlinarith
  have hmul' : M * (1 - 2 * ε) ≤ u := by nlinarith
  exact (le_div_iff₀ hden).mpr (by simpa [M] using hmul')

/-- The standard `ε = 1/4` matrix-net reduction used in HDP's proof of Theorem 4.4.3. -/
lemma matrixOperatorNorm_le_two_mul_of_quarter_centered_bilinear_net {m n : ℕ} {u : ℝ}
    (A : Matrix (Fin m) (Fin n) ℝ) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hu : 0 ≤ u)
    (hnet : ∀ x ∈ N.domainNet, ∀ y ∈ N.codomainNet,
      |inner ℝ (A.toEuclideanLin x) y| ≤ u) :
    matrixOperatorNorm A ≤ 2 * u := by
  have h :=
    matrixOperatorNorm_le_of_centered_bilinear_net (A := A) (N := N)
      (by norm_num : 0 ≤ (1 / 4 : ℝ)) (by norm_num : 2 * (1 / 4 : ℝ) < 1) hu hnet
  calc
    matrixOperatorNorm A ≤ u / (1 - 2 * (1 / 4 : ℝ)) := h
    _ = 2 * u := by ring

omit [MeasurableSpace Ω] in
lemma matrix_toEuclideanLin_isSymmetric_of_isSymm {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsSymm) :
    A.toEuclideanLin.IsSymmetric := by
  rw [LinearMap.isSymmetric_iff_isSelfAdjoint, LinearMap.isSelfAdjoint_iff']
  rw [← Matrix.toEuclideanLin_conjTranspose_eq_adjoint A]
  have hconj : A.conjTranspose = A := by
    ext i j
    simpa [Matrix.conjTranspose] using hA.apply i j
  rw [hconj]

omit [MeasurableSpace Ω] in
lemma matrix_toEuclideanCLM_isSymmetric_of_isSymm {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsSymm) :
    (A.toEuclideanLin.toContinuousLinearMap).IsSymmetric := by
  have hlin := matrix_toEuclideanLin_isSymmetric_of_isSymm hA
  exact LinearMap.IsSymmetric.isSelfAdjoint
    (A := A.toEuclideanLin.toContinuousLinearMap) (by simpa using hlin) |>.isSymmetric

omit [MeasurableSpace Ω] in
lemma matrixOperatorNorm_le_of_forall_unit_abs_quadratic_le {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ) (hA_symm : A.IsSymm) {B : ℝ} (hB : 0 ≤ B)
    (h : ∀ x : EuclideanSpace ℝ (Fin n), ‖x‖ = 1 →
      |inner ℝ (A.toEuclideanLin x) x| ≤ B) :
    matrixOperatorNorm A ≤ B := by
  let T := A.toEuclideanLin.toContinuousLinearMap
  have hT_symm : T.IsSymmetric := by
    simpa [T] using matrix_toEuclideanCLM_isSymmetric_of_isSymm hA_symm
  have hray : ∀ x, |T.rayleighQuotient x| ≤ B := by
    intro x
    by_cases hx : x = 0
    · simp [T, ContinuousLinearMap.rayleighQuotient, hx, hB]
    · let c : ℝ := ‖x‖⁻¹
      have hc : c ≠ 0 := by
        simp [c, norm_ne_zero_iff.mpr hx]
      let y : EuclideanSpace ℝ (Fin n) := c • x
      have hy_norm : ‖y‖ = 1 := by
        simp [y, c, norm_smul, norm_ne_zero_iff.mpr hx]
      have hy_ray : T.rayleighQuotient y = T.rayleighQuotient x := by
        simpa [y, c] using T.rayleigh_smul x (c := c) hc
      calc
        |T.rayleighQuotient x| = |T.rayleighQuotient y| := by rw [hy_ray]
        _ = |inner ℝ (A.toEuclideanLin y) y| := by
          simp [T, ContinuousLinearMap.rayleighQuotient,
            ContinuousLinearMap.reApplyInnerSelf_apply, hy_norm]
        _ ≤ B := h y hy_norm
  unfold matrixOperatorNorm
  change ‖T‖ ≤ B
  rw [T.norm_eq_iSup_rayleighQuotient hT_symm]
  exact ciSup_le hray

/--
Deterministic net reduction for symmetric operator norms. If a centered `ε`-net of the unit
sphere controls all quadratic forms `⟪Ax, x⟫`, then it controls the Euclidean operator norm.
-/
lemma matrixOperatorNorm_le_of_centered_quadratic_net {n : ℕ} {ε u : ℝ}
    (A : Matrix (Fin n) (Fin n) ℝ) (hA_symm : A.IsSymm)
    (N : CenteredMatrixBilinearNet n n ε) (hε_nonneg : 0 ≤ ε) (hε : 2 * ε < 1)
    (hu : 0 ≤ u)
    (hnet : ∀ x ∈ N.domainNet, |inner ℝ (A.toEuclideanLin x) x| ≤ u) :
    matrixOperatorNorm A ≤ u / (1 - 2 * ε) := by
  let M := matrixOperatorNorm A
  have hM_nonneg : 0 ≤ M := norm_nonneg _
  have hden : 0 < 1 - 2 * ε := by linarith
  have hstep : M ≤ u + 2 * ε * M := by
    unfold M
    refine matrixOperatorNorm_le_of_forall_unit_abs_quadratic_le A hA_symm
      (add_nonneg hu (mul_nonneg (mul_nonneg (by norm_num) hε_nonneg) hM_nonneg)) ?_
    intro x hx
    have hx_sphere : x ∈ euclideanUnitSphere n := by
      simpa [euclideanUnitSphere, dist_eq_norm] using hx
    obtain ⟨x₀, hx₀_mem, hx₀_dist⟩ := N.toMatrixBilinearNet.exists_domain_near hx_sphere
    have hx₀_norm : ‖x₀‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hx₀_mem)
    have hAx₀x₀ : |inner ℝ (A.toEuclideanLin x₀) x₀| ≤ u :=
      hnet x₀ hx₀_mem
    have hA_dx : ‖A.toEuclideanLin (x - x₀)‖ ≤ M * ‖x - x₀‖ := by
      simpa [M, matrixOperatorNorm] using
        A.toEuclideanLin.toContinuousLinearMap.le_opNorm (x - x₀)
    have hA_x : ‖A.toEuclideanLin x‖ ≤ M * ‖x‖ := by
      simpa [M, matrixOperatorNorm] using A.toEuclideanLin.toContinuousLinearMap.le_opNorm x
    have hterm_x :
        |inner ℝ (A.toEuclideanLin (x - x₀)) x₀| ≤ M * ε := by
      calc
        |inner ℝ (A.toEuclideanLin (x - x₀)) x₀|
            ≤ ‖A.toEuclideanLin (x - x₀)‖ * ‖x₀‖ := abs_real_inner_le_norm _ _
        _ ≤ (M * ‖x - x₀‖) * 1 := by
              exact mul_le_mul hA_dx hx₀_norm (norm_nonneg _)
                (mul_nonneg hM_nonneg (norm_nonneg _))
        _ = M * ‖x - x₀‖ := by ring
        _ ≤ M * ε := mul_le_mul_of_nonneg_left hx₀_dist hM_nonneg
    have hterm_y :
        |inner ℝ (A.toEuclideanLin x) (x - x₀)| ≤ M * ε := by
      calc
        |inner ℝ (A.toEuclideanLin x) (x - x₀)|
            ≤ ‖A.toEuclideanLin x‖ * ‖x - x₀‖ := abs_real_inner_le_norm _ _
        _ ≤ (M * ‖x‖) * ε := by
              exact mul_le_mul hA_x hx₀_dist (norm_nonneg _)
                (mul_nonneg hM_nonneg (norm_nonneg _))
        _ = M * ε := by rw [hx]; ring
    have hdecomp :
        inner ℝ (A.toEuclideanLin x) x =
          inner ℝ (A.toEuclideanLin x₀) x₀ +
            inner ℝ (A.toEuclideanLin (x - x₀)) x₀ +
              inner ℝ (A.toEuclideanLin x) (x - x₀) := by
      have hx_decomp : x₀ + (x - x₀) = x := by abel
      calc
        inner ℝ (A.toEuclideanLin x) x =
            inner ℝ (A.toEuclideanLin x) (x₀ + (x - x₀)) := by rw [hx_decomp]
        _ = inner ℝ (A.toEuclideanLin x) x₀ +
              inner ℝ (A.toEuclideanLin x) (x - x₀) := by rw [inner_add_right]
        _ = inner ℝ (A.toEuclideanLin (x₀ + (x - x₀))) x₀ +
              inner ℝ (A.toEuclideanLin x) (x - x₀) := by rw [hx_decomp]
        _ = inner ℝ (A.toEuclideanLin x₀ + A.toEuclideanLin (x - x₀)) x₀ +
              inner ℝ (A.toEuclideanLin x) (x - x₀) := by rw [map_add]
        _ = inner ℝ (A.toEuclideanLin x₀) x₀ +
              inner ℝ (A.toEuclideanLin (x - x₀)) x₀ +
                inner ℝ (A.toEuclideanLin x) (x - x₀) := by rw [inner_add_left]
    calc
      |inner ℝ (A.toEuclideanLin x) x|
          ≤ |inner ℝ (A.toEuclideanLin x₀) x₀| +
              |inner ℝ (A.toEuclideanLin (x - x₀)) x₀| +
                |inner ℝ (A.toEuclideanLin x) (x - x₀)| := by
            rw [hdecomp]
            have h₁ := abs_add_le
              (inner ℝ (A.toEuclideanLin x₀) x₀ +
                inner ℝ (A.toEuclideanLin (x - x₀)) x₀)
              (inner ℝ (A.toEuclideanLin x) (x - x₀))
            have h₂ := abs_add_le (inner ℝ (A.toEuclideanLin x₀) x₀)
              (inner ℝ (A.toEuclideanLin (x - x₀)) x₀)
            linarith
      _ ≤ u + M * ε + M * ε := by linarith
      _ = u + 2 * ε * M := by ring
  have hmul' : M * (1 - 2 * ε) ≤ u := by nlinarith
  exact (le_div_iff₀ hden).mpr (by simpa [M] using hmul')

/-- The standard `ε = 1/4` symmetric matrix-net reduction used in HDP Corollary 4.4.7. -/
lemma matrixOperatorNorm_le_two_mul_of_quarter_centered_quadratic_net {n : ℕ} {u : ℝ}
    (A : Matrix (Fin n) (Fin n) ℝ) (hA_symm : A.IsSymm)
    (N : CenteredMatrixBilinearNet n n (1 / 4)) (hu : 0 ≤ u)
    (hnet : ∀ x ∈ N.domainNet, |inner ℝ (A.toEuclideanLin x) x| ≤ u) :
    matrixOperatorNorm A ≤ 2 * u := by
  have h :=
    matrixOperatorNorm_le_of_centered_quadratic_net (A := A) (hA_symm := hA_symm) (N := N)
      (by norm_num : 0 ≤ (1 / 4 : ℝ)) (by norm_num : 2 * (1 / 4 : ℝ) < 1) hu hnet
  calc
    matrixOperatorNorm A ≤ u / (1 - 2 * (1 / 4 : ℝ)) := h
    _ = 2 * u := by ring

omit [MeasurableSpace Ω] in
lemma matrixOperatorNorm_event_subset_quarter_centered_quadratic_net {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {u : ℝ}
    (N : CenteredMatrixBilinearNet n n (1 / 4))
    (hA_symm : ∀ ω, (randomMatrix A ω).IsSymm) (hu : 0 ≤ u) :
    {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)} ⊆
      ⋃ x ∈ N.domainNet,
        {ω | u < |inner ℝ ((randomMatrix A ω).toEuclideanLin x) x|} := by
  intro ω hω
  by_contra hω_union
  have hnet : ∀ x ∈ N.domainNet,
      |inner ℝ ((randomMatrix A ω).toEuclideanLin x) x| ≤ u := by
    intro x hx
    have hnot_x :
        ω ∉ {ω | u < |inner ℝ ((randomMatrix A ω).toEuclideanLin x) x|} := by
      intro hmem
      exact hω_union (Set.mem_iUnion₂.mpr ⟨x, hx, hmem⟩)
    exact le_of_not_gt (by simpa using hnot_x)
  have hnorm :
      matrixOperatorNorm (randomMatrix A ω) ≤ 2 * u :=
    matrixOperatorNorm_le_two_mul_of_quarter_centered_quadratic_net
      (randomMatrix A ω) (hA_symm ω) N hu hnet
  exact (not_le_of_gt hω) hnorm

omit [MeasurableSpace Ω] in
lemma sampleCovarianceDeviationOperatorNorm_le_two_mul_of_quarter_centered_quadratic_net
    {m n : ℕ} {u : ℝ}
    (A : Matrix (Fin m) (Fin n) ℝ) (N : CenteredMatrixBilinearNet n n (1 / 4))
    (hu : 0 ≤ u)
    (hnet : ∀ x ∈ N.domainNet,
      |(m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2| ≤ u) :
    sampleCovarianceDeviationOperatorNorm A ≤ 2 * u := by
  unfold sampleCovarianceDeviationOperatorNorm
  refine matrixOperatorNorm_le_two_mul_of_quarter_centered_quadratic_net
    (sampleCovarianceDeviation A) (sampleCovarianceDeviation_isSymm A) N hu ?_
  intro x hx
  rw [inner_sampleCovarianceDeviation_toEuclideanLin]
  exact hnet x hx

omit [MeasurableSpace Ω] in
lemma sampleCovarianceDeviationOperatorNorm_event_subset_quarter_centered_quadratic_net
    {m n : ℕ} {A : Fin m → Fin n → Ω → ℝ} {u : ℝ}
    (N : CenteredMatrixBilinearNet n n (1 / 4)) (hu : 0 ≤ u) :
    {ω | 2 * u < sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)} ⊆
      ⋃ x ∈ N.domainNet,
        {ω |
          u < |(m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2|} := by
  intro ω hω
  by_contra hω_union
  have hnet : ∀ x ∈ N.domainNet,
      |(m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2| ≤ u := by
    intro x hx
    have hnot_x :
        ω ∉ {ω |
          u < |(m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2|} := by
      intro hmem
      exact hω_union (Set.mem_iUnion₂.mpr ⟨x, hx, hmem⟩)
    exact le_of_not_gt (by simpa using hnot_x)
  have hnorm :
      sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ≤ 2 * u :=
    sampleCovarianceDeviationOperatorNorm_le_two_mul_of_quarter_centered_quadratic_net
      (randomMatrix A ω) N hu hnet
  exact (not_le_of_gt hω) hnorm

omit [MeasurableSpace Ω] in
abbrev MatrixBilinearPair (m n : ℕ) :=
  EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)

omit [MeasurableSpace Ω] in
def CenteredMatrixBilinearNet.pairs {m n : ℕ} {ε : ℝ}
    (N : CenteredMatrixBilinearNet m n ε) : Finset (MatrixBilinearPair m n) :=
  N.domainNet ×ˢ N.codomainNet

omit [MeasurableSpace Ω] in
def CenteredMatrixBilinearNet.signedPairs {m n : ℕ} {ε : ℝ}
    (N : CenteredMatrixBilinearNet m n ε) : Finset (MatrixBilinearPair m n × Bool) :=
  N.pairs ×ˢ (Finset.univ : Finset Bool)

omit [MeasurableSpace Ω] in
def CenteredMatrixBilinearNet.signedDomain {m n : ℕ} {ε : ℝ}
    (N : CenteredMatrixBilinearNet m n ε) :
    Finset (EuclideanSpace ℝ (Fin n) × Bool) :=
  N.domainNet ×ˢ (Finset.univ : Finset Bool)

omit [MeasurableSpace Ω] in
def signedBilinearValue {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    MatrixBilinearPair m n × Bool → ℝ :=
  fun q =>
    if q.2 then inner ℝ (A.toEuclideanLin q.1.1) q.1.2
    else -inner ℝ (A.toEuclideanLin q.1.1) q.1.2

omit [MeasurableSpace Ω] in
def signedSampleCovarianceDeviationValue {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    EuclideanSpace ℝ (Fin n) × Bool → ℝ :=
  fun q =>
    let z : ℝ := (m : ℝ)⁻¹ * ‖A.toEuclideanLin q.1‖ ^ 2 - ‖q.1‖ ^ 2
    if q.2 then z else -z

omit [MeasurableSpace Ω] in
lemma matrixOperatorNorm_le_two_mul_signed_sup_of_quarter_centered_bilinear_net {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN : N.pairs.Nonempty) :
    matrixOperatorNorm A ≤
      2 * (N.signedPairs.sup'
        (hN.product ⟨true, Finset.mem_univ true⟩)
        (signedBilinearValue A)) := by
  let s : Finset (MatrixBilinearPair m n) := N.pairs
  let signed : Finset (MatrixBilinearPair m n × Bool) := N.signedPairs
  let signedValue : MatrixBilinearPair m n × Bool → ℝ := signedBilinearValue A
  have hsigned : signed.Nonempty := by
    exact hN.product ⟨true, Finset.mem_univ true⟩
  let absMax : ℝ := s.sup' hN fun p => |inner ℝ (A.toEuclideanLin p.1) p.2|
  have habsMax_nonneg : 0 ≤ absMax := by
    obtain ⟨p, hp⟩ := hN
    have hp_le : |inner ℝ (A.toEuclideanLin p.1) p.2| ≤ absMax :=
      Finset.le_sup' (s := s) (f := fun p => |inner ℝ (A.toEuclideanLin p.1) p.2|) hp
    exact (abs_nonneg _).trans hp_le
  have hnorm_abs : matrixOperatorNorm A ≤ 2 * absMax := by
    refine matrixOperatorNorm_le_two_mul_of_quarter_centered_bilinear_net
      (A := A) (N := N) habsMax_nonneg ?_
    intro x hx y hy
    have hp : ((x, y) : MatrixBilinearPair m n) ∈ s := by
      change (x, y) ∈ N.domainNet ×ˢ N.codomainNet
      exact Finset.mem_product.mpr ⟨hx, hy⟩
    simpa [absMax, s] using
      (Finset.le_sup' (s := s) (f := fun p => |inner ℝ (A.toEuclideanLin p.1) p.2|) hp)
  have habs_le_signed : absMax ≤ signed.sup' hsigned signedValue := by
    refine Finset.sup'_le (s := s) (H := hN)
      (f := fun p => |inner ℝ (A.toEuclideanLin p.1) p.2|)
      (a := signed.sup' hsigned signedValue) ?_
    intro p hp
    let z : ℝ := inner ℝ (A.toEuclideanLin p.1) p.2
    by_cases hz : 0 ≤ z
    · have hp_signed : (p, true) ∈ signed := by
        change (p, true) ∈ s ×ˢ (Finset.univ : Finset Bool)
        exact Finset.mem_product.mpr ⟨hp, Finset.mem_univ true⟩
      calc
        |inner ℝ (A.toEuclideanLin p.1) p.2| = inner ℝ (A.toEuclideanLin p.1) p.2 :=
          abs_of_nonneg hz
        _ = signedValue (p, true) := by
          simp [signedValue, signedBilinearValue]
        _ ≤ signed.sup' hsigned signedValue := Finset.le_sup' signedValue hp_signed
    · have hzneg : z < 0 := lt_of_not_ge hz
      have hp_signed : (p, false) ∈ signed := by
        change (p, false) ∈ s ×ˢ (Finset.univ : Finset Bool)
        exact Finset.mem_product.mpr ⟨hp, Finset.mem_univ false⟩
      calc
        |inner ℝ (A.toEuclideanLin p.1) p.2| = -inner ℝ (A.toEuclideanLin p.1) p.2 :=
          abs_of_neg hzneg
        _ = signedValue (p, false) := by
          simp [signedValue, signedBilinearValue]
        _ ≤ signed.sup' hsigned signedValue := Finset.le_sup' signedValue hp_signed
  calc
    matrixOperatorNorm A ≤ 2 * absMax := hnorm_abs
    _ ≤ 2 * signed.sup' hsigned signedValue :=
      mul_le_mul_of_nonneg_left habs_le_signed (by norm_num : (0 : ℝ) ≤ 2)

omit [MeasurableSpace Ω] in
lemma sampleCovarianceDeviationOperatorNorm_le_two_mul_signed_sup_of_quarter_centered_net
    {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (N : CenteredMatrixBilinearNet n n (1 / 4)) (hN : N.domainNet.Nonempty) :
    sampleCovarianceDeviationOperatorNorm A ≤
      2 * (N.signedDomain.sup'
        (hN.product ⟨true, Finset.mem_univ true⟩)
        (signedSampleCovarianceDeviationValue A)) := by
  let signed : Finset (EuclideanSpace ℝ (Fin n) × Bool) := N.signedDomain
  let signedValue : EuclideanSpace ℝ (Fin n) × Bool → ℝ :=
    signedSampleCovarianceDeviationValue A
  have hsigned : signed.Nonempty :=
    hN.product ⟨true, Finset.mem_univ true⟩
  let u : ℝ := signed.sup' hsigned signedValue
  have hnet : ∀ x ∈ N.domainNet,
      |(m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2| ≤ u := by
    intro x hx
    let z : ℝ := (m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2
    by_cases hz : 0 ≤ z
    · have hp_signed : (x, true) ∈ signed := by
        change (x, true) ∈ N.domainNet ×ˢ (Finset.univ : Finset Bool)
        exact Finset.mem_product.mpr ⟨hx, Finset.mem_univ true⟩
      calc
        |(m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2| = z := by
          exact abs_of_nonneg hz
        _ = signedValue (x, true) := by
          simp [signedValue, signedSampleCovarianceDeviationValue, z]
        _ ≤ u := Finset.le_sup' signedValue hp_signed
    · have hzneg : z < 0 := lt_of_not_ge hz
      have hp_signed : (x, false) ∈ signed := by
        change (x, false) ∈ N.domainNet ×ˢ (Finset.univ : Finset Bool)
        exact Finset.mem_product.mpr ⟨hx, Finset.mem_univ false⟩
      calc
        |(m : ℝ)⁻¹ * ‖A.toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2| = -z := by
          exact abs_of_neg hzneg
        _ = signedValue (x, false) := by
          simp [signedValue, signedSampleCovarianceDeviationValue, z]
        _ ≤ u := Finset.le_sup' signedValue hp_signed
  have hu : 0 ≤ u := by
    obtain ⟨x, hx⟩ := hN
    exact (abs_nonneg _).trans (hnet x hx)
  calc
    sampleCovarianceDeviationOperatorNorm A ≤ 2 * u :=
      sampleCovarianceDeviationOperatorNorm_le_two_mul_of_quarter_centered_quadratic_net
        A N hu hnet
    _ = 2 * (N.signedDomain.sup'
        (hN.product ⟨true, Finset.mem_univ true⟩)
        (signedSampleCovarianceDeviationValue A)) := by rfl

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.pairs_nonempty_of_pos {m n : ℕ} {ε : ℝ}
    (N : CenteredMatrixBilinearNet m n ε) (hm : 0 < m) (hn : 0 < n) :
    N.pairs.Nonempty := by
  have hdomain : N.domainNet.Nonempty :=
    N.toMatrixBilinearNet.domainNet_nonempty_of_pos hn
  have hcodomain : N.codomainNet.Nonempty :=
    N.toMatrixBilinearNet.codomainNet_nonempty_of_pos hm
  simpa [CenteredMatrixBilinearNet.pairs] using hdomain.product hcodomain

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.signedPairs_card_ge_two_of_pos {m n : ℕ} {ε : ℝ}
    (N : CenteredMatrixBilinearNet m n ε) (hm : 0 < m) (hn : 0 < n) :
    2 ≤ N.signedPairs.card := by
  have hpairs : N.pairs.Nonempty := N.pairs_nonempty_of_pos hm hn
  have hpairs_one : 1 ≤ N.pairs.card := Nat.succ_le_of_lt (Finset.card_pos.mpr hpairs)
  have hmul : 1 * 2 ≤ N.pairs.card * 2 := Nat.mul_le_mul_right 2 hpairs_one
  simpa [CenteredMatrixBilinearNet.signedPairs, Finset.card_product] using hmul

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.signedDomain_card_ge_two_of_pos {m n : ℕ} {ε : ℝ}
    (N : CenteredMatrixBilinearNet m n ε) (hn : 0 < n) :
    2 ≤ N.signedDomain.card := by
  have hdomain : N.domainNet.Nonempty :=
    N.toMatrixBilinearNet.domainNet_nonempty_of_pos hn
  have hdomain_one : 1 ≤ N.domainNet.card := Nat.succ_le_of_lt (Finset.card_pos.mpr hdomain)
  have hmul : 1 * 2 ≤ N.domainNet.card * 2 := Nat.mul_le_mul_right 2 hdomain_one
  simpa [CenteredMatrixBilinearNet.signedDomain, Finset.card_product] using hmul

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.signedDomain_card_real_le_two_mul_seventeen_pow {m n : ℕ}
    (hn : 0 < n) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN_domain :
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n)))) :
    (N.signedDomain.card : ℝ) ≤ 2 * 17 ^ n := by
  have hdomain := centeredMatrixBilinearNet_domain_card_le_seventeen_pow hn N hN_domain
  calc
    (N.signedDomain.card : ℝ)
        = 2 * (N.domainNet.card : ℝ) := by
          simp [CenteredMatrixBilinearNet.signedDomain, Finset.card_product,
            Nat.cast_mul, mul_comm]
    _ ≤ 2 * 17 ^ n := mul_le_mul_of_nonneg_left hdomain (by norm_num)

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.log_signedDomain_card_le {m n : ℕ}
    (hn : 0 < n) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN_domain :
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n)))) :
    log (N.signedDomain.card : ℝ) ≤ 19 * (n : ℝ) := by
  have hsigned_ge2 := N.signedDomain_card_ge_two_of_pos hn
  have hcard_pos : 0 < (N.signedDomain.card : ℝ) := by
    exact_mod_cast (lt_of_lt_of_le (by norm_num : 0 < 2) hsigned_ge2)
  have hcard_le := N.signedDomain_card_real_le_two_mul_seventeen_pow hn hN_domain
  have hn_ge1 : 1 ≤ (n : ℝ) := by exact_mod_cast hn
  calc
    log (N.signedDomain.card : ℝ)
        ≤ log (2 * (17 : ℝ) ^ n) := log_le_log hcard_pos hcard_le
    _ = log (2 : ℝ) + log ((17 : ℝ) ^ n) := by
          rw [log_mul (by norm_num : (2 : ℝ) ≠ 0)
            (pow_ne_zero n (by norm_num : (17 : ℝ) ≠ 0))]
    _ = log (2 : ℝ) + (n : ℝ) * log (17 : ℝ) := by rw [log_pow]
    _ ≤ 2 + (n : ℝ) * 17 := by
          exact add_le_add (log_two_lt_d9.le.trans (by norm_num))
            (mul_le_mul_of_nonneg_left (log_le_self (by norm_num : (0 : ℝ) ≤ 17))
              (Nat.cast_nonneg n))
    _ ≤ 19 * (n : ℝ) := by nlinarith

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.signedPairs_card_real_le_two_mul_seventeen_pow {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN_domain :
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n))))
    (hN_codomain :
      (N.codomainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m)))) :
    (N.signedPairs.card : ℝ) ≤ 2 * 17 ^ (m + n) := by
  have hcard :=
    centeredMatrixBilinearNet_card_product_le_seventeen_pow_add hm hn N hN_domain hN_codomain
  calc
    (N.signedPairs.card : ℝ)
        = 2 * ((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ) := by
          simp [CenteredMatrixBilinearNet.signedPairs, CenteredMatrixBilinearNet.pairs,
            Finset.card_product, Nat.cast_mul, mul_comm]
    _ ≤ 2 * 17 ^ (m + n) :=
          mul_le_mul_of_nonneg_left hcard (by norm_num : (0 : ℝ) ≤ 2)

omit [MeasurableSpace Ω] in
lemma CenteredMatrixBilinearNet.sqrt_two_log_signedPairs_card_le {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n) (N : CenteredMatrixBilinearNet m n (1 / 4))
    (hN_domain :
      (N.domainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin n))))
    (hN_codomain :
      (N.codomainNet.card : WithTop ℕ) ≤
        coveringNumber (1 / 8) (euclideanBall 1 : Set (EuclideanSpace ℝ (Fin m)))) :
    √(2 * log (N.signedPairs.card : ℝ)) ≤ 7 * (√(m : ℝ) + √(n : ℝ)) := by
  have hsigned_ge2 := N.signedPairs_card_ge_two_of_pos hm hn
  have hcard_pos : 0 < (N.signedPairs.card : ℝ) := by
    exact_mod_cast (lt_of_lt_of_le (by norm_num : 0 < 2) hsigned_ge2)
  have hcard_ge1 : 1 ≤ (N.signedPairs.card : ℝ) := by
    exact_mod_cast (le_trans (by norm_num : 1 ≤ 2) hsigned_ge2)
  have hlog_nonneg : 0 ≤ log (N.signedPairs.card : ℝ) := log_nonneg hcard_ge1
  have hcard_le :=
    N.signedPairs_card_real_le_two_mul_seventeen_pow hm hn hN_domain hN_codomain
  have hd_ge1 : 1 ≤ ((m + n : ℕ) : ℝ) := by
    exact_mod_cast Nat.succ_le_of_lt (Nat.add_pos_left hm n)
  have hlog_le : log (N.signedPairs.card : ℝ) ≤
      19 * ((m + n : ℕ) : ℝ) := by
    calc
      log (N.signedPairs.card : ℝ)
          ≤ log (2 * (17 : ℝ) ^ (m + n)) := log_le_log hcard_pos hcard_le
      _ = log (2 : ℝ) + log ((17 : ℝ) ^ (m + n)) := by
            rw [log_mul (by norm_num : (2 : ℝ) ≠ 0)
              (pow_ne_zero (m + n) (by norm_num : (17 : ℝ) ≠ 0))]
      _ = log (2 : ℝ) + ((m + n : ℕ) : ℝ) * log (17 : ℝ) := by
            rw [log_pow]
      _ ≤ 2 + ((m + n : ℕ) : ℝ) * 17 := by
            exact add_le_add (log_le_self (by norm_num : 0 ≤ (2 : ℝ)))
              (mul_le_mul_of_nonneg_left
                (log_le_self (by norm_num : 0 ≤ (17 : ℝ))) (Nat.cast_nonneg _))
      _ ≤ 19 * ((m + n : ℕ) : ℝ) := by
            nlinarith
  have hdim_le_sq :
      ((m + n : ℕ) : ℝ) ≤ (√(m : ℝ) + √(n : ℝ)) ^ 2 := by
    have hsqrt_m_sq : √(m : ℝ) ^ 2 = (m : ℝ) :=
      Real.sq_sqrt (Nat.cast_nonneg m)
    have hsqrt_n_sq : √(n : ℝ) ^ 2 = (n : ℝ) :=
      Real.sq_sqrt (Nat.cast_nonneg n)
    have hmn : 0 ≤ 2 * √(m : ℝ) * √(n : ℝ) := by positivity
    calc
      ((m + n : ℕ) : ℝ) = (m : ℝ) + (n : ℝ) := by
        norm_num [Nat.cast_add]
      _ ≤ (√(m : ℝ) + √(n : ℝ)) ^ 2 := by
        nlinarith [hsqrt_m_sq, hsqrt_n_sq, hmn]
  have harg_nonneg : 0 ≤ 2 * log (N.signedPairs.card : ℝ) := by positivity
  have htarget_nonneg : 0 ≤ 7 * (√(m : ℝ) + √(n : ℝ)) := by positivity
  have hsquare :
      (√(2 * log (N.signedPairs.card : ℝ))) ^ 2 ≤
        (7 * (√(m : ℝ) + √(n : ℝ))) ^ 2 := by
    rw [sq_sqrt harg_nonneg]
    calc
      2 * log (N.signedPairs.card : ℝ)
          ≤ 38 * ((m + n : ℕ) : ℝ) := by nlinarith
      _ ≤ 38 * (√(m : ℝ) + √(n : ℝ)) ^ 2 :=
            mul_le_mul_of_nonneg_left hdim_le_sq (by norm_num : 0 ≤ (38 : ℝ))
      _ ≤ (7 * (√(m : ℝ) + √(n : ℝ))) ^ 2 := by
            nlinarith [sq_nonneg (√(m : ℝ) + √(n : ℝ))]
  exact (sq_le_sq₀ (sqrt_nonneg _) htarget_nonneg).1 hsquare

/-! ## Entrywise sub-Gaussian scales -/

/-- Maximum MGF-ψ₂ scale over a finite scalar random family. It is `0` for an empty index type. -/
def maxFamilySubGaussianPsi2Norm {ι : Type*} [Fintype ι]
    (X : ι → Ω → ℝ) (μ : Measure Ω) : ℝ := by
  classical
  exact if h : (Finset.univ : Finset ι).Nonempty then
    (Finset.univ : Finset ι).sup' h fun i => subGaussianPsi2Norm (X i) μ
  else 0

lemma subGaussianPsi2Norm_le_maxFamilySubGaussianPsi2Norm {ι : Type*} [Fintype ι]
    (X : ι → Ω → ℝ) (μ : Measure Ω) (i : ι) :
    subGaussianPsi2Norm (X i) μ ≤ maxFamilySubGaussianPsi2Norm X μ := by
  classical
  unfold maxFamilySubGaussianPsi2Norm
  rw [dif_pos (show (Finset.univ : Finset ι).Nonempty from ⟨i, Finset.mem_univ i⟩)]
  exact Finset.le_sup' (f := fun i => subGaussianPsi2Norm (X i) μ) (Finset.mem_univ i)

/-- Maximum entrywise MGF-ψ₂ scale for a random rectangular matrix. -/
def maxMatrixEntrySubGaussianPsi2Norm {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) : ℝ :=
  maxFamilySubGaussianPsi2Norm (fun p : Fin m × Fin n => A p.1 p.2) μ

lemma subGaussianPsi2Norm_le_maxMatrixEntrySubGaussianPsi2Norm {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) (i : Fin m) (j : Fin n) :
    subGaussianPsi2Norm (A i j) μ ≤ maxMatrixEntrySubGaussianPsi2Norm A μ := by
  exact subGaussianPsi2Norm_le_maxFamilySubGaussianPsi2Norm
    (fun p : Fin m × Fin n => A p.1 p.2) μ (i, j)

/-- A positive vector ψ₂/MGF scale: every one-dimensional unit projection is sub-Gaussian. -/
def HasSubGaussianVectorPsi2Bound {n : ℕ} (X : Ω → EuclideanSpace ℝ (Fin n))
    (μ : Measure Ω) (K : ℝ) : Prop :=
  0 < K ∧
    ∀ x : EuclideanSpace ℝ (Fin n), ‖x‖ = 1 →
      HasSubgaussianMGF (fun ω => inner ℝ (X ω) x) ⟨K ^ 2, sq_nonneg K⟩ μ

/-- The HDP vector ψ₂ scale, defined as the infimum over admissible projection scales. -/
def subGaussianVectorPsi2Norm {n : ℕ} (X : Ω → EuclideanSpace ℝ (Fin n))
    (μ : Measure Ω) : ℝ :=
  sInf {K : ℝ | HasSubGaussianVectorPsi2Bound X μ K}

/-- Finiteness of the vector ψ₂ scale. -/
def HasFiniteSubGaussianVectorPsi2Norm {n : ℕ} (X : Ω → EuclideanSpace ℝ (Fin n))
    (μ : Measure Ω) : Prop :=
  ∃ K : ℝ, HasSubGaussianVectorPsi2Bound X μ K

lemma subGaussianVectorPsi2Norm_nonneg {n : ℕ}
    {X : Ω → EuclideanSpace ℝ (Fin n)} {μ : Measure Ω}
    (hfin : HasFiniteSubGaussianVectorPsi2Norm X μ) :
    0 ≤ subGaussianVectorPsi2Norm X μ := by
  exact le_csInf hfin fun K hK => hK.1.le

lemma exists_hasSubGaussianVectorPsi2Bound_lt {n : ℕ}
    {X : Ω → EuclideanSpace ℝ (Fin n)} {μ : Measure Ω} {R : ℝ}
    (hfin : HasFiniteSubGaussianVectorPsi2Norm X μ)
    (hR : subGaussianVectorPsi2Norm X μ < R) :
    ∃ K : ℝ, HasSubGaussianVectorPsi2Bound X μ K ∧ K < R := by
  let S : Set ℝ := {K : ℝ | HasSubGaussianVectorPsi2Bound X μ K}
  have hne : S.Nonempty := hfin
  by_contra h
  push Not at h
  have hR_le : R ≤ sInf S := le_csInf hne fun K hK => h K hK
  exact not_lt_of_ge hR_le (by simpa [subGaussianVectorPsi2Norm, S] using hR)

lemma hasSubgaussianMGF_inner_of_subGaussianVectorPsi2Norm_lt {n : ℕ}
    {X : Ω → EuclideanSpace ℝ (Fin n)} {μ : Measure Ω} {R : ℝ}
    (hfin : HasFiniteSubGaussianVectorPsi2Norm X μ)
    (hR : subGaussianVectorPsi2Norm X μ < R)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    HasSubgaussianMGF (fun ω => inner ℝ (X ω) x) ⟨R ^ 2, sq_nonneg R⟩ μ := by
  obtain ⟨K, hK, hKR⟩ := exists_hasSubGaussianVectorPsi2Bound_lt hfin hR
  have hRpos : 0 < R := lt_trans hK.1 hKR
  have hsq : K ^ 2 ≤ R ^ 2 :=
    (sq_le_sq₀ hK.1.le hRpos.le).2 hKR.le
  exact hasSubgaussianMGF_mono_param (hK.2 x hx) (by
    change K ^ 2 ≤ R ^ 2
    exact hsq)

lemma hasSubgaussianMGF_inner_of_subGaussianVectorPsi2Norm_le {n : ℕ}
    {X : Ω → EuclideanSpace ℝ (Fin n)} {μ : Measure Ω} {R : ℝ}
    (hfin : HasFiniteSubGaussianVectorPsi2Norm X μ)
    (hR : subGaussianVectorPsi2Norm X μ ≤ R)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    HasSubgaussianMGF (fun ω => inner ℝ (X ω) x) ⟨R ^ 2, sq_nonneg R⟩ μ where
  integrable_exp_mul t := by
    have hnorm_nonneg : 0 ≤ subGaussianVectorPsi2Norm X μ :=
      subGaussianVectorPsi2Norm_nonneg hfin
    have hlt : subGaussianVectorPsi2Norm X μ < R + 1 := by linarith
    exact (hasSubgaussianMGF_inner_of_subGaussianVectorPsi2Norm_lt hfin hlt hx).integrable_exp_mul t
  mgf_le t := by
    have hlim_const :
        Tendsto (fun _ : ℝ => mgf (fun ω => inner ℝ (X ω) x) μ t)
          (𝓝[>] (0 : ℝ)) (𝓝 (mgf (fun ω => inner ℝ (X ω) x) μ t)) :=
      tendsto_const_nhds
    have hlim_bound :
        Tendsto
          (fun ε : ℝ =>
            exp ((((⟨(R + ε) ^ 2, sq_nonneg (R + ε)⟩ : ℝ≥0) : ℝ) * t ^ 2) / 2))
          (𝓝[>] (0 : ℝ))
          (𝓝 (exp ((((⟨R ^ 2, sq_nonneg R⟩ : ℝ≥0) : ℝ) * t ^ 2) / 2))) := by
      have hcont :
          ContinuousAt (fun ε : ℝ => exp (((R + ε) ^ 2) * t ^ 2 / 2)) 0 := by
        fun_prop
      simpa only [NNReal.coe_mk, add_zero] using hcont.tendsto.mono_left nhdsWithin_le_nhds
    have hev :
        (fun _ : ℝ => mgf (fun ω => inner ℝ (X ω) x) μ t) ≤ᶠ[𝓝[>] (0 : ℝ)]
          fun ε =>
            exp ((((⟨(R + ε) ^ 2, sq_nonneg (R + ε)⟩ : ℝ≥0) : ℝ) * t ^ 2) / 2) := by
      filter_upwards [self_mem_nhdsWithin] with ε hε
      have hεpos : 0 < ε := by simpa using hε
      have hlt : subGaussianVectorPsi2Norm X μ < R + ε := by linarith
      exact (hasSubgaussianMGF_inner_of_subGaussianVectorPsi2Norm_lt hfin hlt hx).mgf_le t
    exact le_of_tendsto_of_tendsto hlim_const hlim_bound hev

/-- The random `i`-th row of a scalar random matrix as a Euclidean vector. -/
def randomMatrixRowVector {m n : ℕ} (A : Fin m → Fin n → Ω → ℝ) (i : Fin m) :
    Ω → EuclideanSpace ℝ (Fin n) :=
  fun ω => matrixRowVector (randomMatrix A ω) i

/-- Maximum vector ψ₂ scale of the rows of a random matrix. It is `0` for no rows. -/
def maxMatrixRowSubGaussianPsi2Norm {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) : ℝ := by
  classical
  exact if h : (Finset.univ : Finset (Fin m)).Nonempty then
    (Finset.univ : Finset (Fin m)).sup' h fun i =>
      subGaussianVectorPsi2Norm (randomMatrixRowVector A i) μ
  else 0

lemma subGaussianVectorPsi2Norm_le_maxMatrixRowSubGaussianPsi2Norm {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) (i : Fin m) :
    subGaussianVectorPsi2Norm (randomMatrixRowVector A i) μ ≤
      maxMatrixRowSubGaussianPsi2Norm A μ := by
  classical
  unfold maxMatrixRowSubGaussianPsi2Norm
  rw [dif_pos (show (Finset.univ : Finset (Fin m)).Nonempty from ⟨i, Finset.mem_univ i⟩)]
  exact Finset.le_sup'
    (f := fun i => subGaussianVectorPsi2Norm (randomMatrixRowVector A i) μ)
    (Finset.mem_univ i)

lemma row_inner_hasSubgaussianMGF_of_max_row_psi2 {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hfinite : ∀ i, HasFiniteSubGaussianVectorPsi2Norm (randomMatrixRowVector A i) μ)
    (i : Fin m) {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    HasSubgaussianMGF (fun ω => inner ℝ (randomMatrixRowVector A i ω) x)
      ⟨K ^ 2, sq_nonneg K⟩ μ := by
  have hnorm_le :
      subGaussianVectorPsi2Norm (randomMatrixRowVector A i) μ ≤ K := by
    rw [hK_def]
    exact subGaussianVectorPsi2Norm_le_maxMatrixRowSubGaussianPsi2Norm A μ i
  exact hasSubgaussianMGF_inner_of_subGaussianVectorPsi2Norm_le (hfinite i) hnorm_le hx

/--
If `K` is the maximum entrywise ψ₂ scale, then each entry has an MGF sub-Gaussian
certificate at scale `K`. This uses the closed-infimum lemma for `subGaussianPsi2Norm`.
-/
lemma entry_hasSubgaussianMGF_of_max {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (hK_def : K = maxMatrixEntrySubGaussianPsi2Norm A μ)
    (hfinite : ∀ i j, HasFiniteSubGaussianPsi2Norm (A i j) μ)
    (i : Fin m) (j : Fin n) :
    HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ := by
  have hnorm_le :
      subGaussianPsi2Norm (A i j) μ ≤ K := by
    rw [hK_def]
    exact subGaussianPsi2Norm_le_maxMatrixEntrySubGaussianPsi2Norm A μ i j
  exact hasSubgaussianMGF_of_subGaussianPsi2Norm_le (hfinite i j) hnorm_le

/--
If `K` is the maximum entrywise ψ₂ scale, then each entry has an MGF sub-Gaussian
certificate at the slightly larger scale `2K`. The factor `2` is the standard margin for
turning the infimum defining `subGaussianPsi2Norm` into a strict admissible bound.
-/
lemma entry_hasSubgaussianMGF_two_mul_max {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (hK_def : K = maxMatrixEntrySubGaussianPsi2Norm A μ) (hK : 0 < K)
    (hfinite : ∀ i j, HasFiniteSubGaussianPsi2Norm (A i j) μ)
    (i : Fin m) (j : Fin n) :
    HasSubgaussianMGF (A i j) ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ := by
  have hnorm_le :
      subGaussianPsi2Norm (A i j) μ ≤ K := by
    rw [hK_def]
    exact subGaussianPsi2Norm_le_maxMatrixEntrySubGaussianPsi2Norm A μ i j
  have hnorm_lt : subGaussianPsi2Norm (A i j) μ < 2 * K := by
    nlinarith
  exact hasSubgaussianMGF_of_subGaussianPsi2Norm_lt (hfinite i j) hnorm_lt

/-! ## Fixed bilinear forms of random matrices -/

omit [MeasurableSpace Ω] in
lemma inner_randomMatrix_eq_sum_entries {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (ω : Ω)
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) :
    inner ℝ ((randomMatrix A ω).toEuclideanLin x) y =
      ∑ p : Fin m × Fin n, (x p.2 * y p.1) * A p.1 p.2 ω := by
  change inner ℝ (WithLp.toLp 2 ((randomMatrix A ω).mulVec x.ofLp)) y =
    ∑ p : Fin m × Fin n, (x p.2 * y p.1) * A p.1 p.2 ω
  rw [PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial, Matrix.mulVec, dotProduct, randomMatrix_apply]
  have hprod :
      (∑ p : Fin m × Fin n, (x.ofLp p.2 * y.ofLp p.1) * A p.1 p.2 ω) =
        ∑ i : Fin m, ∑ j : Fin n, (x.ofLp j * y.ofLp i) * A i j ω := by
    simpa using
      (Finset.sum_product' (Finset.univ : Finset (Fin m)) (Finset.univ : Finset (Fin n))
        (fun i j => (x.ofLp j * y.ofLp i) * A i j ω))
  rw [hprod]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

omit [MeasurableSpace Ω] in
lemma inner_matrix_eq_sum_entries {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) :
    inner ℝ (A.toEuclideanLin x) y =
      ∑ p : Fin m × Fin n, (x p.2 * y p.1) * A p.1 p.2 := by
  let Aconst : Fin m → Fin n → Unit → ℝ := fun i j _ => A i j
  have h := inner_randomMatrix_eq_sum_entries (Ω := Unit) Aconst () x y
  have hAconst : randomMatrix Aconst () = A := by
    ext i j
    rfl
  simpa [Aconst, hAconst] using h

omit [MeasurableSpace Ω] in
abbrev UpperTriangleIndex (n : ℕ) :=
  {p : Fin n × Fin n // p.1 ≤ p.2}

omit [MeasurableSpace Ω] in
def upperTriangleQuadraticCoeff {n : ℕ} (x : EuclideanSpace ℝ (Fin n))
    (p : UpperTriangleIndex n) : ℝ :=
  if p.val.1 = p.val.2 then x p.val.2 * x p.val.1
  else 2 * (x p.val.2 * x p.val.1)

omit [MeasurableSpace Ω] in
lemma inner_randomMatrix_eq_sum_upperTriangle {n : ℕ}
    (A : Fin n → Fin n → Ω → ℝ) (hsym : ∀ ω i j, A i j ω = A j i ω)
    (ω : Ω) (x : EuclideanSpace ℝ (Fin n)) :
    inner ℝ ((randomMatrix A ω).toEuclideanLin x) x =
      ∑ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω := by
  classical
  let f : Fin n × Fin n → ℝ := fun p => (x p.2 * x p.1) * A p.1 p.2 ω
  let U := UpperTriangleIndex n
  let D := {p : U // p.val.1 = p.val.2}
  let O := {p : U // p.val.1 ≠ p.val.2}
  let L := {p : Fin n × Fin n // ¬p.1 ≤ p.2}
  let g : U → ℝ := fun p => upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω
  have hsplit_full :
      (∑ p : U, f p.val) + (∑ p : L, f p.val) =
        ∑ p : Fin n × Fin n, f p :=
    Fintype.sum_subtype_add_sum_subtype (fun p : Fin n × Fin n => p.1 ≤ p.2) f
  have hsplit_upper_f :
      (∑ p : D, f p.val.val) + (∑ p : O, f p.val.val) = ∑ p : U, f p.val :=
    Fintype.sum_subtype_add_sum_subtype (fun p : U => p.val.1 = p.val.2)
      (fun p : U => f p.val)
  have hsplit_upper_g :
      (∑ p : D, g p.val) + (∑ p : O, g p.val) = ∑ p : U, g p :=
    Fintype.sum_subtype_add_sum_subtype (fun p : U => p.val.1 = p.val.2) g
  let swapLower : L ≃ O :=
  { 
    toFun q :=
      let hlt : q.val.2 < q.val.1 := lt_of_not_ge q.property
      ⟨⟨(q.val.2, q.val.1), le_of_lt hlt⟩, ne_of_lt hlt⟩
    invFun p :=
      ⟨(p.val.val.2, p.val.val.1), by
        intro hle
        exact p.property (le_antisymm p.val.property hle)⟩
    left_inv q := by
      ext <;> rfl
    right_inv p := by
      ext <;> rfl }
  have hlower : (∑ p : L, f p.val) = ∑ p : O, f p.val.val := by
    refine Fintype.sum_equiv swapLower (fun p : L => f p.val) (fun p : O => f p.val.val) ?_
    intro p
    simp [swapLower, f]
    rw [hsym ω p.val.1 p.val.2]
    ring
  have hdiag : (∑ p : D, g p.val) = ∑ p : D, f p.val.val := by
    apply Finset.sum_congr rfl
    intro p _
    simp [g, f, upperTriangleQuadraticCoeff, p.property]
  have hoff : (∑ p : O, g p.val) = 2 * ∑ p : O, f p.val.val := by
    calc
      (∑ p : O, g p.val) = ∑ p : O, 2 * f p.val.val := by
        apply Finset.sum_congr rfl
        intro p _
        simp [g, f, upperTriangleQuadraticCoeff, p.property]
        ring
      _ = 2 * ∑ p : O, f p.val.val := by
        rw [Finset.mul_sum]
  calc
    inner ℝ ((randomMatrix A ω).toEuclideanLin x) x
        = ∑ p : Fin n × Fin n, f p := by
          simpa [f] using inner_randomMatrix_eq_sum_entries A ω x x
    _ = (∑ p : U, f p.val) + (∑ p : L, f p.val) := hsplit_full.symm
    _ = (∑ p : U, f p.val) + (∑ p : O, f p.val.val) := by rw [hlower]
    _ = ((∑ p : D, f p.val.val) + (∑ p : O, f p.val.val)) +
          (∑ p : O, f p.val.val) := by rw [hsplit_upper_f]
    _ = (∑ p : D, g p.val) + (∑ p : O, g p.val) := by
          rw [hdiag, hoff]
          ring
    _ = ∑ p : U, g p := hsplit_upper_g
    _ = ∑ p : UpperTriangleIndex n,
          upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω := rfl

lemma inner_randomMatrix_measurable {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} (hA_meas : ∀ i j, Measurable (A i j))
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) :
    Measurable fun ω => inner ℝ ((randomMatrix A ω).toEuclideanLin x) y := by
  have hsum :
      Measurable fun ω => ∑ p : Fin m × Fin n, (x p.2 * y p.1) * A p.1 p.2 ω := by
    apply Finset.measurable_sum
    intro p _
    exact (hA_meas p.1 p.2).const_mul _
  convert hsum using 1
  funext ω
  exact inner_randomMatrix_eq_sum_entries A ω x y

-- The Euclidean operator norm is bounded by the `ℓ¹` norm of the matrix entries.
omit [MeasurableSpace Ω] in
lemma matrixOperatorNorm_le_sum_abs_entries {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    matrixOperatorNorm A ≤ ∑ p : Fin m × Fin n, |A p.1 p.2| := by
  refine matrixOperatorNorm_le_of_forall_unit_abs_inner_le A
    (Finset.sum_nonneg fun p _ => abs_nonneg (A p.1 p.2)) ?_
  intro x y hx hy
  rw [inner_matrix_eq_sum_entries]
  calc
    |∑ p : Fin m × Fin n, (x p.2 * y p.1) * A p.1 p.2|
        ≤ ∑ p : Fin m × Fin n, |(x p.2 * y p.1) * A p.1 p.2| :=
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ p : Fin m × Fin n, |A p.1 p.2| := by
      apply Finset.sum_le_sum
      intro p _
      have hx_coord : |x p.2| ≤ 1 := by
        have h := PiLp.norm_apply_le (x := x) p.2
        simpa [Real.norm_eq_abs, hx] using h
      have hy_coord : |y p.1| ≤ 1 := by
        have h := PiLp.norm_apply_le (x := y) p.1
        simpa [Real.norm_eq_abs, hy] using h
      have hxy : |x p.2| * |y p.1| ≤ 1 * 1 := by
        exact mul_le_mul hx_coord hy_coord (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)
      calc
        |(x p.2 * y p.1) * A p.1 p.2|
            = |x p.2| * |y p.1| * |A p.1 p.2| := by
              rw [abs_mul, abs_mul]
        _ ≤ 1 * 1 * |A p.1 p.2| := by
              exact mul_le_mul_of_nonneg_right hxy (abs_nonneg _)
        _ = |A p.1 p.2| := by ring

omit [MeasurableSpace Ω] in
lemma matrixColumnVector_eq_toEuclideanLin_single {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (j : Fin n) :
    matrixColumnVector A j =
      A.toEuclideanLin (WithLp.toLp 2 (Pi.single j (1 : ℝ))) := by
  rw [Matrix.toLpLin_toLp]
  ext i
  simp [matrixColumnVector]

omit [MeasurableSpace Ω] in
lemma norm_matrixColumnVector_le_matrixOperatorNorm {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (j : Fin n) :
    ‖matrixColumnVector A j‖ ≤ matrixOperatorNorm A := by
  let e : EuclideanSpace ℝ (Fin n) := WithLp.toLp 2 (Pi.single j (1 : ℝ))
  have he : ‖e‖ = 1 := by
    simp [e]
  calc
    ‖matrixColumnVector A j‖ = ‖A.toEuclideanLin e‖ := by
      rw [matrixColumnVector_eq_toEuclideanLin_single A j]
    _ = ‖A.toEuclideanLin.toContinuousLinearMap e‖ := rfl
    _ ≤ ‖A.toEuclideanLin.toContinuousLinearMap‖ * ‖e‖ :=
      A.toEuclideanLin.toContinuousLinearMap.le_opNorm e
    _ = matrixOperatorNorm A := by
      simp [matrixOperatorNorm, he]

omit [MeasurableSpace Ω] in
lemma matrixOperatorNorm_conjTranspose {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    matrixOperatorNorm A.conjTranspose = matrixOperatorNorm A := by
  unfold matrixOperatorNorm
  rw [Matrix.toEuclideanLin_conjTranspose_eq_adjoint]
  rw [LinearMap.adjoint_toContinuousLinearMap]
  exact ContinuousLinearMap.adjoint.norm_map A.toEuclideanLin.toContinuousLinearMap

omit [MeasurableSpace Ω] in
lemma norm_matrixRowVector_le_matrixOperatorNorm {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) :
    ‖matrixRowVector A i‖ ≤ matrixOperatorNorm A := by
  calc
    ‖matrixRowVector A i‖ = ‖matrixColumnVector A.conjTranspose i‖ := by
      simp [matrixRowVector, matrixColumnVector]
    _ ≤ matrixOperatorNorm A.conjTranspose :=
      norm_matrixColumnVector_le_matrixOperatorNorm A.conjTranspose i
    _ = matrixOperatorNorm A := matrixOperatorNorm_conjTranspose A

omit [MeasurableSpace Ω] in
lemma avg_row_column_norm_le_matrixOperatorNorm {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) (j : Fin n) :
    (‖matrixRowVector A i‖ + ‖matrixColumnVector A j‖) / 2 ≤ matrixOperatorNorm A := by
  have hrow := norm_matrixRowVector_le_matrixOperatorNorm A i
  have hcol := norm_matrixColumnVector_le_matrixOperatorNorm A j
  nlinarith

omit [MeasurableSpace Ω] in
lemma row_column_norm_average_le_matrixOperatorNorm_of_le {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) (j : Fin n) {r c : ℝ}
    (hrow : r ≤ ‖matrixRowVector A i‖) (hcol : c ≤ ‖matrixColumnVector A j‖) :
    (r + c) / 2 ≤ matrixOperatorNorm A := by
  have hdet := avg_row_column_norm_le_matrixOperatorNorm A i j
  linarith

omit [MeasurableSpace Ω] in
lemma randomVector_norm_sq_eq_sum_sq {n : ℕ} (X : Fin n → Ω → ℝ) (ω : Ω) :
    ‖HansonWright.randomVector X ω‖ ^ 2 = ∑ i, X i ω ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq]
  rfl

omit [MeasurableSpace Ω] in
lemma quadraticForm_one_eq_sum_sq {n : ℕ} (x : Fin n → ℝ) :
    HansonWright.quadraticForm (1 : Matrix (Fin n) (Fin n) ℝ) x =
      ∑ i, x i ^ 2 := by
  classical
  unfold HansonWright.quadraticForm
  calc
    ∑ i, ∑ j, (1 : Matrix (Fin n) (Fin n) ℝ) i j * x i * x j
        = ∑ i, (1 : Matrix (Fin n) (Fin n) ℝ) i i * x i * x i := by
          apply Finset.sum_congr rfl
          intro i _
          exact Finset.sum_eq_single i
            (fun j _ hji => by simp [Matrix.one_apply_ne hji.symm])
            (fun hi => (hi (Finset.mem_univ i)).elim)
    _ = ∑ i, x i ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          simp [pow_two]

omit [MeasurableSpace Ω] in
lemma randomQuadraticForm_one_eq_norm_sq {n : ℕ}
    (X : Fin n → Ω → ℝ) (ω : Ω) :
    HansonWright.randomQuadraticForm (1 : Matrix (Fin n) (Fin n) ℝ) X ω =
      ‖HansonWright.randomVector X ω‖ ^ 2 := by
  rw [HansonWright.randomQuadraticForm, quadraticForm_one_eq_sum_sq,
    randomVector_norm_sq_eq_sum_sq]

omit [MeasurableSpace Ω] in
lemma frobeniusNormSq_one {n : ℕ} :
    HansonWright.frobeniusNormSq (1 : Matrix (Fin n) (Fin n) ℝ) = n := by
  classical
  unfold HansonWright.frobeniusNormSq
  calc
    ∑ i, ∑ j, ((1 : Matrix (Fin n) (Fin n) ℝ) i j) ^ 2
        = ∑ i, ((1 : Matrix (Fin n) (Fin n) ℝ) i i) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          exact Finset.sum_eq_single i
            (fun j _ hji => by simp [Matrix.one_apply_ne hji.symm])
            (fun hi => (hi (Finset.mem_univ i)).elim)
    _ = ∑ _i : Fin n, (1 : ℝ) := by
          apply Finset.sum_congr rfl
          intro i _
          simp
    _ = n := by simp

omit [MeasurableSpace Ω] in
lemma frobeniusNorm_one_sq {n : ℕ} :
    HansonWright.frobeniusNorm (1 : Matrix (Fin n) (Fin n) ℝ) ^ 2 = n := by
  rw [HansonWright.frobeniusNorm_sq, frobeniusNormSq_one]

omit [MeasurableSpace Ω] in
lemma toEuclideanCLM_one {n : ℕ} :
    Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ) (1 : Matrix (Fin n) (Fin n) ℝ) =
      ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n)) := by
  ext x i
  simp

omit [MeasurableSpace Ω] in
lemma operatorNorm_one_le {n : ℕ} :
    HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ) ≤ 1 := by
  unfold HansonWright.operatorNorm
  rw [toEuclideanCLM_one]
  exact ContinuousLinearMap.norm_id_le

omit [MeasurableSpace Ω] in
lemma operatorNorm_one_pos_of_pos {n : ℕ} (hn : 0 < n) :
    0 < HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ) := by
  unfold HansonWright.operatorNorm
  let i : Fin n := ⟨0, hn⟩
  let e : EuclideanSpace ℝ (Fin n) := EuclideanSpace.single i (1 : ℝ)
  have he : ‖e‖ = 1 := by simp [e]
  have happly :
      Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ)
          (1 : Matrix (Fin n) (Fin n) ℝ) e = e := by
    rw [toEuclideanCLM_one]
    rfl
  have hle :=
    (Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ)
      (1 : Matrix (Fin n) (Fin n) ℝ)).le_opNorm e
  rw [happly, he, mul_one] at hle
  exact lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1) hle

omit [MeasurableSpace Ω] in
lemma one_le_operatorNorm_one_of_pos {n : ℕ} (hn : 0 < n) :
    1 ≤ HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ) := by
  unfold HansonWright.operatorNorm
  let i : Fin n := ⟨0, hn⟩
  let e : EuclideanSpace ℝ (Fin n) := EuclideanSpace.single i (1 : ℝ)
  have he : ‖e‖ = 1 := by simp [e]
  have happly :
      Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ)
          (1 : Matrix (Fin n) (Fin n) ℝ) e = e := by
    rw [toEuclideanCLM_one]
    rfl
  have hle :=
    (Matrix.toEuclideanCLM (n := Fin n) (𝕜 := ℝ)
      (1 : Matrix (Fin n) (Fin n) ℝ)).le_opNorm e
  rw [happly, he, mul_one] at hle
  exact hle

omit [MeasurableSpace Ω] in
lemma operatorNorm_one_eq_of_pos {n : ℕ} (hn : 0 < n) :
    HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ) = 1 :=
  le_antisymm operatorNorm_one_le (one_le_operatorNorm_one_of_pos hn)

omit [MeasurableSpace Ω] in
/-- A fixed explicit Hanson-Wright constant large enough for the local estimates used here. -/
def hdpHansonWrightConstant : ℝ :=
  64 * exp 1 ^ 2

omit [MeasurableSpace Ω] in
lemma hdpHansonWrightConstant_pos :
    0 < hdpHansonWrightConstant := by
  unfold hdpHansonWrightConstant
  positivity

omit [MeasurableSpace Ω] in
lemma hdpHansonWrightConstant_domain :
    4 * exp 1 ≤ hdpHansonWrightConstant := by
  unfold hdpHansonWrightConstant
  have hexp_ge_one : 1 ≤ exp 1 := one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
  have hmul := mul_le_mul_of_nonneg_left hexp_ge_one
    (by positivity : 0 ≤ 4 * exp 1)
  nlinarith

omit [MeasurableSpace Ω] in
lemma hdpHansonWrightConstant_diag_quad :
    8 * exp 1 ^ 3 ≤ hdpHansonWrightConstant := by
  unfold hdpHansonWrightConstant
  have hexp_le_eight : exp 1 ≤ 8 := by
    nlinarith [exp_one_lt_three.le]
  have hmul := mul_le_mul_of_nonneg_left hexp_le_eight
    (by positivity : 0 ≤ 8 * exp 1 ^ 2)
  nlinarith

omit [MeasurableSpace Ω] in
lemma hdpHansonWrightConstant_offdiag_domain :
    16 * exp 1 ≤ hdpHansonWrightConstant ^ 2 := by
  have hC_pos : 0 < hdpHansonWrightConstant := hdpHansonWrightConstant_pos
  have hexp_ge_one : 1 ≤ exp 1 := one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
  have hC_ge_sixteen : 16 * exp 1 ≤ hdpHansonWrightConstant := by
    unfold hdpHansonWrightConstant
    have hmul := mul_le_mul_of_nonneg_left hexp_ge_one
      (by positivity : 0 ≤ 16 * exp 1)
    nlinarith
  have hC_ge_one : 1 ≤ hdpHansonWrightConstant := by
    unfold hdpHansonWrightConstant
    nlinarith [hexp_ge_one, sq_nonneg (exp 1)]
  have hC_sq_ge : hdpHansonWrightConstant ≤ hdpHansonWrightConstant ^ 2 := by
    have hmul := mul_le_mul hC_ge_one (le_refl hdpHansonWrightConstant)
      hC_pos.le hC_pos.le
    simpa [pow_two] using hmul
  exact hC_ge_sixteen.trans hC_sq_ge

omit [MeasurableSpace Ω] in
lemma hdpHansonWrightConstant_offdiag_quad :
    64 * exp 1 ^ 2 ≤ hdpHansonWrightConstant := by
  rfl

omit [MeasurableSpace Ω] in
/-- The absolute constant used in the vector norm lower tail for Exercise 4.42. -/
def randomVectorLowerTailConstant : ℝ :=
  1 / (64 * hdpHansonWrightConstant * exp 1 ^ 2)

omit [MeasurableSpace Ω] in
lemma randomVectorLowerTailConstant_pos :
    0 < randomVectorLowerTailConstant := by
  unfold randomVectorLowerTailConstant hdpHansonWrightConstant
  positivity

/--
For fixed deterministic vectors `x` and `y`, the bilinear form `⟪A x, y⟫` is a finite
independent linear combination of the entries of `A`, hence has a sub-Gaussian MGF certificate.
The variance proxy is kept in explicit coefficient-sum form; later net estimates bound it by
`K²` on unit balls.
-/
lemma inner_randomMatrix_hasSubgaussianMGF_explicit {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (h_indep : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ)
    (hA_subG : ∀ i j, HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) :
    HasSubgaussianMGF
      (fun ω => inner ℝ ((randomMatrix A ω).toEuclideanLin x) y)
      (∑ p : Fin m × Fin n,
        Real.toNNReal ((x p.2 * y p.1) ^ 2) *
          Real.toNNReal (K ^ 2)) μ := by
  let cK : ℝ≥0 := Real.toNNReal (K ^ 2)
  have hcK : cK = (⟨K ^ 2, sq_nonneg K⟩ : ℝ≥0) := by
    ext
    exact Real.coe_toNNReal (K ^ 2) (sq_nonneg K)
  have hA_subG' :
      ∀ p ∈ (Finset.univ : Finset (Fin m × Fin n)),
        HasSubgaussianMGF (fun ω => A p.1 p.2 ω) cK μ := by
    intro p _
    rw [hcK]
    exact hA_subG p.1 p.2
  have hsum :=
    HansonWright.hasSubgaussianMGF_finset_sum_const_mul_of_iIndepFun
      (μ := μ) h_indep (s := Finset.univ) hA_subG'
      (fun p : Fin m × Fin n => x p.2 * y p.1)
  refine hsum.congr (ae_of_all _ fun ω => ?_)
  simpa using (inner_randomMatrix_eq_sum_entries A ω x y).symm

omit [MeasurableSpace Ω] in
lemma sum_sq_bilinear_coeff_eq_norm_mul {m n : ℕ}
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) :
    (∑ p : Fin m × Fin n, (x p.2 * y p.1) ^ 2) = ‖x‖ ^ 2 * ‖y‖ ^ 2 := by
  have hprod :
      (∑ p : Fin m × Fin n, (x.ofLp p.2 * y.ofLp p.1) ^ 2) =
        ∑ i : Fin m, ∑ j : Fin n, (x.ofLp j * y.ofLp i) ^ 2 := by
    simpa using
      (Finset.sum_product' (Finset.univ : Finset (Fin m)) (Finset.univ : Finset (Fin n))
        (fun i j => (x.ofLp j * y.ofLp i) ^ 2))
  rw [hprod]
  calc
    (∑ i : Fin m, ∑ j : Fin n, (x.ofLp j * y.ofLp i) ^ 2)
        = ∑ i : Fin m, (y.ofLp i) ^ 2 * ∑ j : Fin n, (x.ofLp j) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro j _
          ring
    _ = (∑ i : Fin m, (y.ofLp i) ^ 2) * ∑ j : Fin n, (x.ofLp j) ^ 2 := by
          rw [Finset.sum_mul]
    _ = ‖y‖ ^ 2 * ‖x‖ ^ 2 := by
          rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
    _ = ‖x‖ ^ 2 * ‖y‖ ^ 2 := by ring

omit [MeasurableSpace Ω] in
lemma upperTriangleQuadraticCoeff_sq_le_four_mul {n : ℕ}
    (x : EuclideanSpace ℝ (Fin n)) (p : UpperTriangleIndex n) :
    upperTriangleQuadraticCoeff x p ^ 2 ≤ 4 * (x p.val.2 * x p.val.1) ^ 2 := by
  by_cases hp : p.val.1 = p.val.2
  · simp [upperTriangleQuadraticCoeff, hp]
    nlinarith [sq_nonneg (x p.val.2 * x p.val.1)]
  · simp [upperTriangleQuadraticCoeff, hp]
    ring_nf
    exact le_rfl

omit [MeasurableSpace Ω] in
lemma sum_upperTriangleQuadraticCoeff_sq_le_four_norm_pow_four {n : ℕ}
    (x : EuclideanSpace ℝ (Fin n)) :
    (∑ p : UpperTriangleIndex n, upperTriangleQuadraticCoeff x p ^ 2) ≤ 4 * ‖x‖ ^ 4 := by
  classical
  let imageUpper : Finset (Fin n × Fin n) :=
    (Finset.univ : Finset (UpperTriangleIndex n)).image (fun p => p.val)
  have hpoint :
      ∀ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p ^ 2 ≤ 4 * (x p.val.2 * x p.val.1) ^ 2 :=
    upperTriangleQuadraticCoeff_sq_le_four_mul x
  have hsum_point :
      (∑ p : UpperTriangleIndex n, upperTriangleQuadraticCoeff x p ^ 2) ≤
        ∑ p : UpperTriangleIndex n, 4 * (x p.val.2 * x p.val.1) ^ 2 :=
    Finset.sum_le_sum fun p _ => hpoint p
  have hsum_image :
      (∑ q ∈ imageUpper, 4 * (x q.2 * x q.1) ^ 2) =
        ∑ p : UpperTriangleIndex n, 4 * (x p.val.2 * x p.val.1) ^ 2 := by
    simp [imageUpper, Finset.sum_image]
  have hsum_subset :
      (∑ p : UpperTriangleIndex n, 4 * (x p.val.2 * x p.val.1) ^ 2) ≤
        ∑ q : Fin n × Fin n, 4 * (x q.2 * x q.1) ^ 2 := by
    rw [← hsum_image]
    exact Finset.sum_le_univ_sum_of_nonneg (s := imageUpper)
      (f := fun q : Fin n × Fin n => 4 * (x q.2 * x q.1) ^ 2) fun q => by positivity
  have hfull :
      (∑ q : Fin n × Fin n, 4 * (x q.2 * x q.1) ^ 2) = 4 * ‖x‖ ^ 4 := by
    calc
      (∑ q : Fin n × Fin n, 4 * (x q.2 * x q.1) ^ 2)
          = 4 * ∑ q : Fin n × Fin n, (x q.2 * x q.1) ^ 2 := by
            rw [Finset.mul_sum]
      _ = 4 * (‖x‖ ^ 2 * ‖x‖ ^ 2) := by
            rw [sum_sq_bilinear_coeff_eq_norm_mul x x]
      _ = 4 * ‖x‖ ^ 4 := by ring
  exact hsum_point.trans (hsum_subset.trans_eq hfull)

omit [MeasurableSpace Ω] in
lemma sum_upperTriangleQuadraticCoeff_sq_le_four_of_norm_le_one {n : ℕ}
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) :
    (∑ p : UpperTriangleIndex n, upperTriangleQuadraticCoeff x p ^ 2) ≤ 4 := by
  have hsum := sum_upperTriangleQuadraticCoeff_sq_le_four_norm_pow_four x
  have hx4 : ‖x‖ ^ 4 ≤ 1 := by
    have hx2 : ‖x‖ ^ 2 ≤ 1 := by
      nlinarith [norm_nonneg x, hx]
    have hsq_nonneg : 0 ≤ ‖x‖ ^ 2 := sq_nonneg ‖x‖
    have hpow : ‖x‖ ^ 4 = (‖x‖ ^ 2) ^ 2 := by ring
    rw [hpow]
    nlinarith [sq_nonneg (‖x‖ ^ 2), hx2, hsq_nonneg]
  calc
    (∑ p : UpperTriangleIndex n, upperTriangleQuadraticCoeff x p ^ 2) ≤
        4 * ‖x‖ ^ 4 := hsum
    _ ≤ 4 * 1 := mul_le_mul_of_nonneg_left hx4 (by norm_num : (0 : ℝ) ≤ 4)
    _ = 4 := by ring

omit [MeasurableSpace Ω] in
lemma upperTriangleQuadraticCoeff_variance_proxy_le {n : ℕ}
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) (K : ℝ) :
    (((∑ p : UpperTriangleIndex n,
      Real.toNNReal (upperTriangleQuadraticCoeff x p ^ 2) *
        Real.toNNReal (K ^ 2)) : ℝ≥0) : ℝ) ≤ (2 * K) ^ 2 := by
  rw [NNReal.coe_sum]
  simp only [NNReal.coe_mul]
  calc
    (∑ p : UpperTriangleIndex n,
        ↑(Real.toNNReal (upperTriangleQuadraticCoeff x p ^ 2)) *
          ↑(Real.toNNReal (K ^ 2)))
        = ∑ p : UpperTriangleIndex n, upperTriangleQuadraticCoeff x p ^ 2 * K ^ 2 := by
          apply Finset.sum_congr rfl
          intro p _
          rw [Real.coe_toNNReal _ (sq_nonneg (upperTriangleQuadraticCoeff x p)),
            Real.coe_toNNReal _ (sq_nonneg K)]
    _ = (∑ p : UpperTriangleIndex n, upperTriangleQuadraticCoeff x p ^ 2) * K ^ 2 := by
          rw [Finset.sum_mul]
    _ ≤ 4 * K ^ 2 := by
          exact mul_le_mul_of_nonneg_right
            (sum_upperTriangleQuadraticCoeff_sq_le_four_of_norm_le_one hx) (sq_nonneg K)
    _ = (2 * K) ^ 2 := by ring

lemma upperTriangleQuadraticForm_hasSubgaussianMGF_explicit {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (h_indep : iIndepFun (fun p : UpperTriangleIndex n => A p.val.1 p.val.2) μ)
    (hA_subG : ∀ p : UpperTriangleIndex n,
      HasSubgaussianMGF (A p.val.1 p.val.2) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (x : EuclideanSpace ℝ (Fin n)) :
    HasSubgaussianMGF
      (fun ω => ∑ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω)
      (∑ p : UpperTriangleIndex n,
        Real.toNNReal (upperTriangleQuadraticCoeff x p ^ 2) *
          Real.toNNReal (K ^ 2)) μ := by
  let cK : ℝ≥0 := Real.toNNReal (K ^ 2)
  have hcK : cK = (⟨K ^ 2, sq_nonneg K⟩ : ℝ≥0) := by
    ext
    exact Real.coe_toNNReal (K ^ 2) (sq_nonneg K)
  have hA_subG' :
      ∀ p ∈ (Finset.univ : Finset (UpperTriangleIndex n)),
        HasSubgaussianMGF (fun ω => A p.val.1 p.val.2 ω) cK μ := by
    intro p _
    rw [hcK]
    exact hA_subG p
  have hsum :=
    HansonWright.hasSubgaussianMGF_finset_sum_const_mul_of_iIndepFun
      (μ := μ) h_indep (s := Finset.univ) hA_subG'
      (fun p : UpperTriangleIndex n => upperTriangleQuadraticCoeff x p)
  simpa [cK] using hsum

lemma upperTriangleQuadraticForm_hasSubgaussianMGF_of_norm_le_one {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (h_indep : iIndepFun (fun p : UpperTriangleIndex n => A p.val.1 p.val.2) μ)
    (hA_subG : ∀ p : UpperTriangleIndex n,
      HasSubgaussianMGF (A p.val.1 p.val.2) ⟨K ^ 2, sq_nonneg K⟩ μ)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) :
    HasSubgaussianMGF
      (fun ω => ∑ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω)
      ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ := by
  have h :=
    upperTriangleQuadraticForm_hasSubgaussianMGF_explicit
      (A := A) (μ := μ) (K := K) h_indep hA_subG x
  refine HansonWright.hasSubgaussianMGF_mono_param h ?_
  change (((∑ p : UpperTriangleIndex n,
      Real.toNNReal (upperTriangleQuadraticCoeff x p ^ 2) *
        Real.toNNReal (K ^ 2)) : ℝ≥0) : ℝ) ≤ (2 * K) ^ 2
  exact upperTriangleQuadraticCoeff_variance_proxy_le hx K

omit [MeasurableSpace Ω] in
lemma bilinear_coeff_variance_proxy_eq {m n : ℕ}
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) (K : ℝ) :
    (((∑ p : Fin m × Fin n,
      Real.toNNReal ((x p.2 * y p.1) ^ 2) * Real.toNNReal (K ^ 2)) : ℝ≥0) : ℝ) =
      K ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2 := by
  rw [NNReal.coe_sum]
  simp only [NNReal.coe_mul]
  calc
    (∑ p : Fin m × Fin n,
        ↑(Real.toNNReal ((x p.2 * y p.1) ^ 2)) * ↑(Real.toNNReal (K ^ 2)))
        = ∑ p : Fin m × Fin n, (x p.2 * y p.1) ^ 2 * K ^ 2 := by
          apply Finset.sum_congr rfl
          intro p _
          rw [Real.coe_toNNReal _ (sq_nonneg (x p.2 * y p.1)),
            Real.coe_toNNReal _ (sq_nonneg K)]
    _ = (∑ p : Fin m × Fin n, (x p.2 * y p.1) ^ 2) * K ^ 2 := by
          rw [Finset.sum_mul]
    _ = (‖x‖ ^ 2 * ‖y‖ ^ 2) * K ^ 2 := by
          rw [sum_sq_bilinear_coeff_eq_norm_mul]
    _ = K ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2 := by ring

/-- Fixed bilinear forms of independent entrywise sub-Gaussian random matrices are
sub-Gaussian with variance proxy `K² ‖x‖² ‖y‖²`. -/
lemma inner_randomMatrix_hasSubgaussianMGF {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (h_indep : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ)
    (hA_subG : ∀ i j, HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (x : EuclideanSpace ℝ (Fin n)) (y : EuclideanSpace ℝ (Fin m)) :
    HasSubgaussianMGF
      (fun ω => inner ℝ ((randomMatrix A ω).toEuclideanLin x) y)
      ⟨K ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2,
        mul_nonneg (mul_nonneg (sq_nonneg K) (sq_nonneg ‖x‖)) (sq_nonneg ‖y‖)⟩ μ := by
  have h :=
    inner_randomMatrix_hasSubgaussianMGF_explicit
      (A := A) (μ := μ) (K := K) h_indep hA_subG x y
  refine HansonWright.hasSubgaussianMGF_mono_param h ?_
  change (((∑ p : Fin m × Fin n,
      Real.toNNReal ((x p.2 * y p.1) ^ 2) * Real.toNNReal (K ^ 2)) : ℝ≥0) : ℝ) ≤
    K ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2
  rw [bilinear_coeff_variance_proxy_eq]

/-- Unit-ball specialization of `inner_randomMatrix_hasSubgaussianMGF`. -/
lemma inner_randomMatrix_hasSubgaussianMGF_of_norm_le_one {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (h_indep : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ)
    (hA_subG : ∀ i j, HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ)
    {x : EuclideanSpace ℝ (Fin n)} {y : EuclideanSpace ℝ (Fin m)}
    (hx : ‖x‖ ≤ 1) (hy : ‖y‖ ≤ 1) :
    HasSubgaussianMGF
      (fun ω => inner ℝ ((randomMatrix A ω).toEuclideanLin x) y)
      ⟨K ^ 2, sq_nonneg K⟩ μ := by
  have h :=
    inner_randomMatrix_hasSubgaussianMGF (A := A) (μ := μ) (K := K)
      h_indep hA_subG x y
  refine HansonWright.hasSubgaussianMGF_mono_param h ?_
  change K ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2 ≤ K ^ 2
  have hx2 : ‖x‖ ^ 2 ≤ 1 := by
    nlinarith [norm_nonneg x, hx]
  have hy2 : ‖y‖ ^ 2 ≤ 1 := by
    nlinarith [norm_nonneg y, hy]
  have hxy : ‖x‖ ^ 2 * ‖y‖ ^ 2 ≤ 1 := by
    nlinarith [sq_nonneg ‖x‖, sq_nonneg ‖y‖, hx2, hy2]
  calc
    K ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2 = K ^ 2 * (‖x‖ ^ 2 * ‖y‖ ^ 2) := by ring
    _ ≤ K ^ 2 * 1 := mul_le_mul_of_nonneg_left hxy (sq_nonneg K)
    _ = K ^ 2 := by ring

/-! ## Two-sided tails for fixed bilinear forms -/

/-- Two-sided tail bound from an MGF sub-Gaussian certificate. -/
lemma HasSubgaussianMGF.measure_abs_ge_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {σ u : ℝ} (hσ : 0 < σ)
    (hX : HasSubgaussianMGF X ⟨σ ^ 2, sq_nonneg σ⟩ μ) (hu : 0 < u) :
    (μ {ω | u ≤ |X ω|}).toReal ≤ 2 * exp (-u ^ 2 / (2 * σ ^ 2)) := by
  have hpos :
      (μ {ω | u ≤ X ω}).toReal ≤ exp (-u ^ 2 / (2 * σ ^ 2)) :=
    chernoff_bound_subGaussian hσ hu
      (fun t => by
        calc
          cgf X μ t ≤ ((⟨σ ^ 2, sq_nonneg σ⟩ : ℝ≥0) : ℝ) * t ^ 2 / 2 :=
            hX.cgf_le t
          _ = t ^ 2 * σ ^ 2 / 2 := by
            simp [mul_comm])
      (fun t => hX.integrable_exp_mul t)
  have hneg :
      (μ {ω | u ≤ -X ω}).toReal ≤ exp (-u ^ 2 / (2 * σ ^ 2)) :=
    chernoff_bound_subGaussian hσ hu
      (fun t => by
        change cgf (-X) μ t ≤ t ^ 2 * σ ^ 2 / 2
        calc
          cgf (-X) μ t ≤ ((⟨σ ^ 2, sq_nonneg σ⟩ : ℝ≥0) : ℝ) * t ^ 2 / 2 :=
            hX.neg.cgf_le t
          _ = t ^ 2 * σ ^ 2 / 2 := by
            simp [mul_comm])
      (fun t => hX.neg.integrable_exp_mul t)
  have h_subset :
      {ω | u ≤ |X ω|} ⊆ {ω | u ≤ X ω} ∪ {ω | u ≤ -X ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    rcases le_or_gt 0 (X ω) with hXω | hXω
    · left
      simpa [abs_of_nonneg hXω] using hω
    · right
      simpa [abs_of_neg hXω] using hω
  calc
    (μ {ω | u ≤ |X ω|}).toReal
        ≤ (μ ({ω | u ≤ X ω} ∪ {ω | u ≤ -X ω})).toReal := by
          exact ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono h_subset)
    _ ≤ (μ {ω | u ≤ X ω}).toReal + (μ {ω | u ≤ -X ω}).toReal := by
          rw [← ENNReal.toReal_add (measure_ne_top μ _) (measure_ne_top μ _)]
          exact ENNReal.toReal_mono
            (ENNReal.add_ne_top.mpr ⟨measure_ne_top μ _, measure_ne_top μ _⟩)
            (measure_union_le _ _)
    _ ≤ exp (-u ^ 2 / (2 * σ ^ 2)) + exp (-u ^ 2 / (2 * σ ^ 2)) :=
          add_le_add hpos hneg
    _ = 2 * exp (-u ^ 2 / (2 * σ ^ 2)) := by ring

/-- Fixed bilinear forms of a random matrix have the standard two-sided sub-Gaussian tail
when the test vectors lie in the Euclidean unit balls. -/
lemma inner_randomMatrix_tail_of_norm_le_one {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (hK : 0 < K)
    (h_indep : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ)
    (hA_subG : ∀ i j, HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ)
    {x : EuclideanSpace ℝ (Fin n)} {y : EuclideanSpace ℝ (Fin m)}
    (hx : ‖x‖ ≤ 1) (hy : ‖y‖ ≤ 1) (hu : 0 < u) :
    (μ {ω | u ≤ |inner ℝ ((randomMatrix A ω).toEuclideanLin x) y|}).toReal ≤
      2 * exp (-u ^ 2 / (2 * K ^ 2)) := by
  exact HasSubgaussianMGF.measure_abs_ge_le hK
    (inner_randomMatrix_hasSubgaussianMGF_of_norm_le_one
      (A := A) (μ := μ) (K := K) h_indep hA_subG hx hy) hu

/-- Fixed upper-triangle quadratic forms have the standard two-sided sub-Gaussian tail on the
Euclidean unit ball. -/
lemma upperTriangleQuadraticForm_tail_of_norm_le_one {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (hK : 0 < K)
    (h_indep : iIndepFun (fun p : UpperTriangleIndex n => A p.val.1 p.val.2) μ)
    (hA_subG : ∀ p : UpperTriangleIndex n,
      HasSubgaussianMGF (A p.val.1 p.val.2) ⟨K ^ 2, sq_nonneg K⟩ μ)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) (hu : 0 < u) :
    (μ {ω |
      u ≤ |∑ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω|}).toReal ≤
      2 * exp (-u ^ 2 / (8 * K ^ 2)) := by
  have hK2 : 0 < 2 * K := by nlinarith
  have htail :=
    HasSubgaussianMGF.measure_abs_ge_le hK2
      (upperTriangleQuadraticForm_hasSubgaussianMGF_of_norm_le_one
        (A := A) (μ := μ) (K := K) h_indep hA_subG hx) hu
  have hden : 2 * (2 * K) ^ 2 = 8 * K ^ 2 := by ring
  simpa [hden] using htail

/-- Finite-net tail bound for the upper-triangle quadratic forms. -/
lemma upperTriangleQuadraticForm_net_tail {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (N : CenteredMatrixBilinearNet n n (1 / 4)) (hK : 0 < K)
    (h_indep : iIndepFun (fun p : UpperTriangleIndex n => A p.val.1 p.val.2) μ)
    (hA_subG : ∀ p : UpperTriangleIndex n,
      HasSubgaussianMGF (A p.val.1 p.val.2) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hu : 0 < u) :
    (μ (⋃ x ∈ N.domainNet,
      {ω |
        u < |∑ p : UpperTriangleIndex n,
          upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω|})).toReal ≤
      2 * (N.domainNet.card : ℝ) * exp (-u ^ 2 / (8 * K ^ 2)) := by
  let tailEvent : EuclideanSpace ℝ (Fin n) → Set Ω :=
    fun x => {ω |
      u < |∑ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω|}
  have htail_each :
      ∀ x ∈ N.domainNet, μ.real (tailEvent x) ≤
        2 * exp (-u ^ 2 / (8 * K ^ 2)) := by
    intro x hx
    have hx_norm : ‖x‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hx)
    have hmono :
        μ.real (tailEvent x) ≤
          (μ {ω |
            u ≤ |∑ p : UpperTriangleIndex n,
              upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω|}).toReal := by
      refine ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono ?_)
      intro ω hω
      exact le_of_lt (by simpa [tailEvent] using hω)
    exact hmono.trans
      (upperTriangleQuadraticForm_tail_of_norm_le_one
        (A := A) (μ := μ) (K := K) hK h_indep hA_subG hx_norm hu)
  calc
    (μ (⋃ x ∈ N.domainNet,
      {ω |
        u < |∑ p : UpperTriangleIndex n,
          upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω|})).toReal
        = μ.real (⋃ x ∈ N.domainNet, tailEvent x) := by rfl
    _ ≤ ∑ x ∈ N.domainNet, μ.real (tailEvent x) :=
          measureReal_biUnion_finset_le N.domainNet tailEvent
    _ ≤ ∑ x ∈ N.domainNet, 2 * exp (-u ^ 2 / (8 * K ^ 2)) := by
          exact Finset.sum_le_sum fun x hx => htail_each x hx
    _ = (N.domainNet.card : ℝ) * (2 * exp (-u ^ 2 / (8 * K ^ 2))) := by
          simp [nsmul_eq_mul]
    _ = 2 * (N.domainNet.card : ℝ) * exp (-u ^ 2 / (8 * K ^ 2)) := by ring

/--
Symmetric finite-net operator-norm tail, assuming the deterministic upper-triangle
representation of every quadratic form.
-/
lemma matrixOperatorNorm_tail_of_quarter_centered_quadratic_net {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (N : CenteredMatrixBilinearNet n n (1 / 4))
    (hA_symm : ∀ ω, (randomMatrix A ω).IsSymm)
    (h_repr : ∀ ω x,
      inner ℝ ((randomMatrix A ω).toEuclideanLin x) x =
        ∑ p : UpperTriangleIndex n,
          upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω)
    (hK : 0 < K)
    (h_indep : iIndepFun (fun p : UpperTriangleIndex n => A p.val.1 p.val.2) μ)
    (hA_subG : ∀ p : UpperTriangleIndex n,
      HasSubgaussianMGF (A p.val.1 p.val.2) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hu : 0 < u) :
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * (N.domainNet.card : ℝ) * exp (-u ^ 2 / (8 * K ^ 2)) := by
  let tailEvent : EuclideanSpace ℝ (Fin n) → Set Ω :=
    fun x => {ω |
      u < |∑ p : UpperTriangleIndex n,
        upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω|}
  have hsubset :
      {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)} ⊆
        ⋃ x ∈ N.domainNet, tailEvent x := by
    intro ω hω
    have hdet :=
      matrixOperatorNorm_event_subset_quarter_centered_quadratic_net
        (A := A) (u := u) N hA_symm hu.le hω
    rcases Set.mem_iUnion₂.mp hdet with ⟨x, hx, hmem⟩
    exact Set.mem_iUnion₂.mpr ⟨x, hx, by simpa [tailEvent, h_repr ω x] using hmem⟩
  have hnet :=
    upperTriangleQuadraticForm_net_tail
      (A := A) (μ := μ) (K := K) (u := u) N hK h_indep hA_subG hu
  calc
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal
        ≤ μ.real (⋃ x ∈ N.domainNet, tailEvent x) := by
          simpa [Measure.real] using
            ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hsubset)
    _ ≤ 2 * (N.domainNet.card : ℝ) * exp (-u ^ 2 / (8 * K ^ 2)) := by
          simpa [tailEvent, Measure.real] using hnet

lemma matrixOperatorNorm_tail_le_of_upperTriangle_subgaussian {n : ℕ} (hn : 0 < n)
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (hA_symm : ∀ ω, (randomMatrix A ω).IsSymm)
    (h_repr : ∀ ω x,
      inner ℝ ((randomMatrix A ω).toEuclideanLin x) x =
        ∑ p : UpperTriangleIndex n,
          upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω)
    (hK : 0 < K)
    (h_indep : iIndepFun (fun p : UpperTriangleIndex n => A p.val.1 p.val.2) μ)
    (hA_subG : ∀ p : UpperTriangleIndex n,
      HasSubgaussianMGF (A p.val.1 p.val.2) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hu : 0 < u) :
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * 17 ^ n * exp (-u ^ 2 / (8 * K ^ 2)) := by
  obtain ⟨N, hN_domain, _hN_codomain⟩ := exists_quarter_centeredMatrixBilinearNet n n
  have htail :=
    matrixOperatorNorm_tail_of_quarter_centered_quadratic_net
      (A := A) (μ := μ) (K := K) (u := u) N hA_symm h_repr hK h_indep hA_subG hu
  have hcard :=
    centeredMatrixBilinearNet_domain_card_le_seventeen_pow hn N hN_domain
  have hfactor :
      2 * (N.domainNet.card : ℝ) * exp (-u ^ 2 / (8 * K ^ 2)) ≤
        2 * 17 ^ n * exp (-u ^ 2 / (8 * K ^ 2)) := by
    have hleft : 2 * (N.domainNet.card : ℝ) ≤ 2 * 17 ^ n :=
      mul_le_mul_of_nonneg_left hcard (by norm_num : (0 : ℝ) ≤ 2)
    exact mul_le_mul_of_nonneg_right hleft (le_of_lt (exp_pos _))
  exact htail.trans hfactor

/--
Finite-net tail bound for the operator norm. For a centered quarter-net of the two Euclidean
unit spheres, the event `‖A‖ > 2u` is contained in the union of the fixed bilinear-form tail
events over all net pairs.
-/
lemma matrixOperatorNorm_tail_of_quarter_centered_bilinear_net {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (N : CenteredMatrixBilinearNet m n (1 / 4)) (hK : 0 < K)
    (h_indep : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ)
    (hA_subG : ∀ i j, HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hu : 0 < u) :
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * ((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ) *
        exp (-u ^ 2 / (2 * K ^ 2)) := by
  classical
  let s : Finset (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)) :=
    N.domainNet ×ˢ N.codomainNet
  let tailEvent : EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m) → Set Ω :=
    fun p => {ω | u < |inner ℝ ((randomMatrix A ω).toEuclideanLin p.1) p.2|}
  have hsubset :
      {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)} ⊆
        ⋃ p ∈ s, tailEvent p := by
    intro ω hω
    by_contra hω_union
    have hnet : ∀ x ∈ N.domainNet, ∀ y ∈ N.codomainNet,
        |inner ℝ ((randomMatrix A ω).toEuclideanLin x) y| ≤ u := by
      intro x hx y hy
      have hnot_pair : ω ∉ tailEvent (x, y) := by
        intro hmem
        exact hω_union
          (Set.mem_iUnion₂.mpr ⟨(x, y), Finset.mem_product.mpr ⟨hx, hy⟩, hmem⟩)
      exact le_of_not_gt (by simpa [tailEvent] using hnot_pair)
    have hnorm :
        matrixOperatorNorm (randomMatrix A ω) ≤ 2 * u :=
      matrixOperatorNorm_le_two_mul_of_quarter_centered_bilinear_net
        (randomMatrix A ω) N hu.le hnet
    exact (not_le_of_gt hω) hnorm
  have htail_each :
      ∀ p ∈ s, μ.real (tailEvent p) ≤ 2 * exp (-u ^ 2 / (2 * K ^ 2)) := by
    intro p hp
    obtain ⟨hp_dom, hp_cod⟩ := Finset.mem_product.mp hp
    have hp_dom_norm : ‖p.1‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hp_dom)
    have hp_cod_norm : ‖p.2‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.codomain_subset_unitBall hp_cod)
    have hmono :
        μ.real (tailEvent p) ≤
          (μ {ω | u ≤ |inner ℝ ((randomMatrix A ω).toEuclideanLin p.1) p.2|}).toReal := by
      refine ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono ?_)
      intro ω hω
      exact le_of_lt (by simpa [tailEvent] using hω)
    exact hmono.trans
      (inner_randomMatrix_tail_of_norm_le_one
        (A := A) (μ := μ) (K := K) hK h_indep hA_subG
        hp_dom_norm hp_cod_norm hu)
  calc
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal
        ≤ μ.real (⋃ p ∈ s, tailEvent p) := by
          simpa [Measure.real] using
            ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hsubset)
    _ ≤ ∑ p ∈ s, μ.real (tailEvent p) :=
          measureReal_biUnion_finset_le s tailEvent
    _ ≤ ∑ p ∈ s, 2 * exp (-u ^ 2 / (2 * K ^ 2)) := by
          exact Finset.sum_le_sum fun p hp => htail_each p hp
    _ = ((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ) *
          (2 * exp (-u ^ 2 / (2 * K ^ 2))) := by
          simp [s, Finset.card_product, nsmul_eq_mul]
    _ = 2 * ((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ) *
          exp (-u ^ 2 / (2 * K ^ 2)) := by ring

lemma matrixOperatorNorm_tail_le_of_entrywise_subgaussian {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ] {K u : ℝ}
    (hK : 0 < K)
    (h_indep : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ)
    (hA_subG : ∀ i j, HasSubgaussianMGF (A i j) ⟨K ^ 2, sq_nonneg K⟩ μ)
    (hu : 0 < u) :
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * 17 ^ (m + n) * exp (-u ^ 2 / (2 * K ^ 2)) := by
  obtain ⟨N, hN_domain, hN_codomain⟩ := exists_quarter_centeredMatrixBilinearNet m n
  have htail :=
    matrixOperatorNorm_tail_of_quarter_centered_bilinear_net
      (A := A) (μ := μ) (K := K) (u := u) N hK h_indep hA_subG hu
  have hcard :=
    centeredMatrixBilinearNet_card_product_le_seventeen_pow_add hm hn N hN_domain hN_codomain
  have hfactor :
      2 * ((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ) *
          exp (-u ^ 2 / (2 * K ^ 2)) ≤
        2 * 17 ^ (m + n) * exp (-u ^ 2 / (2 * K ^ 2)) := by
    have hleft :
        2 * ((N.domainNet.card * N.codomainNet.card : ℕ) : ℝ) ≤
          2 * 17 ^ (m + n) :=
      mul_le_mul_of_nonneg_left hcard (by norm_num : (0 : ℝ) ≤ 2)
    exact mul_le_mul_of_nonneg_right hleft (le_of_lt (exp_pos _))
  exact htail.trans hfactor

/-! ## HDP entry hypotheses -/

/-- Entry hypotheses in HDP Theorem 4.4.3: independent, mean-zero, sub-Gaussian entries. -/
structure HasIndependentMeanZeroSubGaussianEntries {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) : Prop where
  measurable : ∀ i j, Measurable (A i j)
  independent : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ
  mean_zero : ∀ i j, ∫ ω, A i j ω ∂μ = 0
  finite_psi2 : ∀ i j, HasFiniteSubGaussianPsi2Norm (A i j) μ

/-- Entry hypotheses in HDP Exercise 4.42: independent, variance-one, sub-Gaussian entries. -/
structure HasIndependentVarianceOneSubGaussianEntries {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) : Prop where
  measurable : ∀ i j, Measurable (A i j)
  independent : iIndepFun (fun p : Fin m × Fin n => A p.1 p.2) μ
  finite_psi2 : ∀ i j, HasFiniteSubGaussianPsi2Norm (A i j) μ
  second_moment_one : ∀ i j, ∫ ω, (A i j ω) ^ 2 ∂μ = 1

/-- Entry hypotheses in HDP Corollary 4.4.7: a symmetric matrix whose on-and-above-diagonal
entries are independent, mean-zero, and sub-Gaussian. -/
structure HasIndependentMeanZeroSubGaussianUpperTriangle {n : ℕ}
    (A : Fin n → Fin n → Ω → ℝ) (μ : Measure Ω) : Prop where
  symmetric : ∀ ω i j, A i j ω = A j i ω
  measurable_upper : ∀ i j, i ≤ j → Measurable (A i j)
  independent_upper :
    iIndepFun (fun p : {p : Fin n × Fin n // p.1 ≤ p.2} =>
      A p.val.1 p.val.2) μ
  mean_zero_upper : ∀ i j, i ≤ j → ∫ ω, A i j ω ∂μ = 0
  finite_psi2_upper : ∀ i j, i ≤ j → HasFiniteSubGaussianPsi2Norm (A i j) μ

/--
Row hypotheses in HDP Theorem 4.6.1: independent, mean-zero, sub-Gaussian, isotropic rows.

The random row `Aᵢ` is represented by `randomMatrixRowVector A i`. Isotropy is stated in
the one-dimensional marginal form `E ⟪Aᵢ, x⟫² = ‖x‖²`.
-/
structure HasIndependentMeanZeroIsotropicSubGaussianRows {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) : Prop where
  measurable : ∀ i j, Measurable (A i j)
  independent_rows : iIndepFun (fun i : Fin m => randomMatrixRowVector A i) μ
  mean_zero : ∀ i (x : EuclideanSpace ℝ (Fin n)),
    ∫ ω, inner ℝ (randomMatrixRowVector A i ω) x ∂μ = 0
  isotropic : ∀ i (x : EuclideanSpace ℝ (Fin n)),
    ∫ ω, (inner ℝ (randomMatrixRowVector A i ω) x) ^ 2 ∂μ = ‖x‖ ^ 2
  finite_row_psi2 : ∀ i, HasFiniteSubGaussianVectorPsi2Norm (randomMatrixRowVector A i) μ

omit [MeasurableSpace Ω] in
lemma randomVector_rowProjection_eq_toEuclideanLin {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (ω : Ω) (x : EuclideanSpace ℝ (Fin n)) :
    HansonWright.randomVector
        (fun i : Fin m => fun ω => inner ℝ (randomMatrixRowVector A i ω) x) ω =
      (randomMatrix A ω).toEuclideanLin x := by
  ext i
  change inner ℝ (WithLp.toLp 2 fun j => A i j ω) x =
    (WithLp.toLp 2 ((randomMatrix A ω).mulVec x.ofLp)) i
  rw [PiLp.inner_apply]
  simp only [RCLike.inner_apply, conj_trivial, Matrix.mulVec, dotProduct, randomMatrix_apply]
  apply Finset.sum_congr rfl
  intro j _
  ring

lemma HasIndependentMeanZeroIsotropicSubGaussianRows.row_projection_iIndepFun {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    (x : EuclideanSpace ℝ (Fin n)) :
    iIndepFun (fun i : Fin m => fun ω => inner ℝ (randomMatrixRowVector A i ω) x) μ := by
  simpa [Function.comp_def] using
    hA.independent_rows.comp (fun _ v => inner ℝ v x) (fun _ => by fun_prop)

lemma HasIndependentMeanZeroIsotropicSubGaussianRows.row_projection_hasSubgaussianMGF_two_mul_max
    {m n : ℕ} {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} {K : ℝ}
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (i : Fin m) {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    HasSubgaussianMGF (fun ω => inner ℝ (randomMatrixRowVector A i ω) x)
      ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ := by
  have hbase :=
    row_inner_hasSubgaussianMGF_of_max_row_psi2
      (A := A) (μ := μ) hK_def hA.finite_row_psi2 i hx
  refine hasSubgaussianMGF_mono_param hbase ?_
  change K ^ 2 ≤ (2 * K) ^ 2
  nlinarith [sq_nonneg K]

lemma HasIndependentMeanZeroIsotropicSubGaussianRows.row_projection_second_moment_one
    {m n : ℕ} {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    (i : Fin m) {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    ∫ ω, (inner ℝ (randomMatrixRowVector A i ω) x) ^ 2 ∂μ = 1 := by
  simpa [hx] using hA.isotropic i x

lemma HasIndependentMeanZeroSubGaussianUpperTriangle.measurable {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω}
    (hA : HasIndependentMeanZeroSubGaussianUpperTriangle A μ) (i j : Fin n) :
    Measurable (A i j) := by
  by_cases hij : i ≤ j
  · exact hA.measurable_upper i j hij
  · have hji : j ≤ i := le_of_not_ge hij
    have h_eq : A i j = A j i := by
      funext ω
      exact hA.symmetric ω i j
    simpa [h_eq] using hA.measurable_upper j i hji

lemma HasIndependentMeanZeroSubGaussianUpperTriangle.mean_zero {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω}
    (hA : HasIndependentMeanZeroSubGaussianUpperTriangle A μ) (i j : Fin n) :
    ∫ ω, A i j ω ∂μ = 0 := by
  by_cases hij : i ≤ j
  · exact hA.mean_zero_upper i j hij
  · have hji : j ≤ i := le_of_not_ge hij
    calc
      ∫ ω, A i j ω ∂μ = ∫ ω, A j i ω ∂μ := by
        exact integral_congr_ae (ae_of_all μ fun ω => hA.symmetric ω i j)
      _ = 0 := hA.mean_zero_upper j i hji

lemma HasIndependentMeanZeroSubGaussianUpperTriangle.finite_psi2 {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω}
    (hA : HasIndependentMeanZeroSubGaussianUpperTriangle A μ) (i j : Fin n) :
    HasFiniteSubGaussianPsi2Norm (A i j) μ := by
  by_cases hij : i ≤ j
  · exact hA.finite_psi2_upper i j hij
  · have hji : j ≤ i := le_of_not_ge hij
    obtain ⟨K, hK_pos, hK_mgf⟩ := hA.finite_psi2_upper j i hji
    refine ⟨K, hK_pos, ?_⟩
    exact hK_mgf.congr (ae_of_all μ fun ω => (hA.symmetric ω i j).symm)

lemma HasIndependentMeanZeroSubGaussianUpperTriangle.randomMatrix_isSymm {n : ℕ}
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω}
    (hA : HasIndependentMeanZeroSubGaussianUpperTriangle A μ) (ω : Ω) :
    (randomMatrix A ω).IsSymm := by
  refine Matrix.IsSymm.ext ?_
  intro i j
  exact (hA.symmetric ω i j).symm

lemma matrixOperatorNorm_tail_le_of_upperTriangle_max_psi2 {n : ℕ} (hn : 0 < n)
    {A : Fin n → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroSubGaussianUpperTriangle A μ)
    (h_repr : ∀ ω x,
      inner ℝ ((randomMatrix A ω).toEuclideanLin x) x =
        ∑ p : UpperTriangleIndex n,
          upperTriangleQuadraticCoeff x p * A p.val.1 p.val.2 ω)
    {K u : ℝ}
    (hK_def : K = maxMatrixEntrySubGaussianPsi2Norm A μ) (hK : 0 < K) (hu : 0 < u) :
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * 17 ^ n * exp (-u ^ 2 / (32 * K ^ 2)) := by
  have hK2 : 0 < 2 * K := by nlinarith
  have htail :=
    matrixOperatorNorm_tail_le_of_upperTriangle_subgaussian
      (A := A) (μ := μ) (K := 2 * K) (u := u) hn
      (fun ω => hA.randomMatrix_isSymm ω) h_repr hK2 hA.independent_upper
      (fun p =>
        entry_hasSubgaussianMGF_two_mul_max
          (A := A) (μ := μ) hK_def hK hA.finite_psi2 p.val.1 p.val.2)
      hu
  have hden : 8 * (2 * K) ^ 2 = 32 * K ^ 2 := by ring
  simpa [hden] using htail

lemma matrixOperatorNorm_tail_le_of_max_psi2 {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroSubGaussianEntries A μ) {K u : ℝ}
    (hK_def : K = maxMatrixEntrySubGaussianPsi2Norm A μ) (hK : 0 < K) (hu : 0 < u) :
    (μ {ω | 2 * u < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * 17 ^ (m + n) * exp (-u ^ 2 / (8 * K ^ 2)) := by
  have hK2 : 0 < 2 * K := by nlinarith
  have htail :=
    matrixOperatorNorm_tail_le_of_entrywise_subgaussian
      (A := A) (μ := μ) (K := 2 * K) (u := u) hm hn hK2 hA.independent
      (fun i j => entry_hasSubgaussianMGF_two_mul_max hK_def hK hA.finite_psi2 i j) hu
  have hden : 2 * (2 * K) ^ 2 = 8 * K ^ 2 := by ring
  simpa [hden] using htail

lemma probability_compl_real_ge_one_sub {μ : Measure Ω} [IsProbabilityMeasure μ]
    (s : Set Ω) :
    1 - (μ s).toReal ≤ (μ sᶜ).toReal := by
  classical
  have hmeasure :
      μ Set.univ ≤ μ s + μ sᶜ := by
    calc
      μ Set.univ = μ (s ∪ sᶜ) := by rw [Set.union_compl_self]
      _ ≤ μ s + μ sᶜ := measure_union_le s sᶜ
  have hfinite : μ s + μ sᶜ ≠ ⊤ :=
    ENNReal.add_ne_top.mpr ⟨measure_ne_top μ s, measure_ne_top μ sᶜ⟩
  have hreal := ENNReal.toReal_mono hfinite hmeasure
  rw [IsProbabilityMeasure.measure_univ, ENNReal.toReal_one] at hreal
  rw [ENNReal.toReal_add (measure_ne_top μ s) (measure_ne_top μ sᶜ)] at hreal
  linarith

lemma probability_le_of_tail_real_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {B δ : ℝ}
    (htail : (μ {ω | B < X ω}).toReal ≤ δ) :
    (μ {ω | X ω ≤ B}).toReal ≥ 1 - δ := by
  have hcompl := probability_compl_real_ge_one_sub (μ := μ) {ω | B < X ω}
  have hset : {ω | B < X ω}ᶜ = {ω | X ω ≤ B} := by
    ext ω
    simp [not_lt]
  rw [hset] at hcompl
  linarith

lemma probability_ge_of_lower_tail_real_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {B δ : ℝ}
    (htail : (μ {ω | X ω < B}).toReal ≤ δ) :
    (μ {ω | B ≤ X ω}).toReal ≥ 1 - δ := by
  have hcompl := probability_compl_real_ge_one_sub (μ := μ) {ω | X ω < B}
  have hset : {ω | X ω < B}ᶜ = {ω | B ≤ X ω} := by
    ext ω
    simp [not_lt]
  rw [hset] at hcompl
  linarith

lemma integral_randomQuadraticForm_one_eq_nat {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {X : Fin n → Ω → ℝ}
    (hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ)
    (hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1) :
    ∫ ω, HansonWright.randomQuadraticForm (1 : Matrix (Fin n) (Fin n) ℝ) X ω ∂μ =
      (n : ℝ) := by
  have hfun :
      (fun ω => HansonWright.randomQuadraticForm (1 : Matrix (Fin n) (Fin n) ℝ) X ω) =
        fun ω => ∑ i, X i ω ^ 2 := by
    funext ω
    rw [HansonWright.randomQuadraticForm, quadraticForm_one_eq_sum_sq]
  rw [hfun]
  calc
    ∫ ω, ∑ i, X i ω ^ 2 ∂μ = ∑ i, ∫ ω, X i ω ^ 2 ∂μ := by
      rw [integral_finsetSum]
      intro i _
      exact hX_sq_int i
    _ = ∑ _i : Fin n, (1 : ℝ) := by
      apply Finset.sum_congr rfl
      intro i _
      exact hsecond i
    _ = (n : ℝ) := by simp

lemma centeredQuadraticForm_one_eq_norm_sq_sub_nat {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {X : Fin n → Ω → ℝ}
    (hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ)
    (hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1) (ω : Ω) :
    HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin n) (Fin n) ℝ) X ω =
      ‖HansonWright.randomVector X ω‖ ^ 2 - (n : ℝ) := by
  unfold HansonWright.centeredQuadraticForm
  rw [integral_randomQuadraticForm_one_eq_nat hX_sq_int hsecond,
    randomQuadraticForm_one_eq_norm_sq]

lemma one_div_four_exp_sq_le_sq_of_subgaussian_second_moment_one
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X : Ω → ℝ} {K : ℝ}
    (hK : 0 < K)
    (hX_subG : HasSubgaussianMGF X ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ)
    (hsecond : ∫ ω, X ω ^ 2 ∂μ = 1) :
    1 / (4 * exp 1 ^ 2) ≤ K ^ 2 := by
  have hCpos : 0 < (2 * K) ^ 2 := by positivity
  have hsq_le :=
    HansonWright.integral_sq_le_of_hasSubgaussianMGF_of_le
      (μ := μ) hX_subG (C0 := (2 * K) ^ 2) hCpos (by rfl)
  have hone_le : 1 ≤ exp 1 * ((2 * K) ^ 2 * exp 1) := by
    calc
      1 = ∫ ω, X ω ^ 2 ∂μ := hsecond.symm
      _ ≤ exp 1 * ((2 * K) ^ 2 * exp 1) := hsq_le
  have hone_le' : 1 ≤ 4 * exp 1 ^ 2 * K ^ 2 := by
    calc
      1 ≤ exp 1 * ((2 * K) ^ 2 * exp 1) := hone_le
      _ = 4 * exp 1 ^ 2 * K ^ 2 := by ring
  have hden_pos : 0 < 4 * exp 1 ^ 2 := by positivity
  have hone_le'' : 1 ≤ K ^ 2 * (4 * exp 1 ^ 2) := by
    nlinarith
  exact (div_le_iff₀ hden_pos).mpr hone_le''

lemma HasIndependentMeanZeroIsotropicSubGaussianRows.max_row_psi2_sq_lower
    {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ) (hK : 0 < K) :
    1 / (4 * exp 1 ^ 2) ≤ K ^ 2 := by
  let i0 : Fin m := ⟨0, hm⟩
  let j0 : Fin n := ⟨0, hn⟩
  let x : EuclideanSpace ℝ (Fin n) := EuclideanSpace.single j0 (1 : ℝ)
  have hx : ‖x‖ = 1 := by
    simp [x]
  have hbase :
      HasSubgaussianMGF (fun ω => inner ℝ (randomMatrixRowVector A i0 ω) x)
        ⟨K ^ 2, sq_nonneg K⟩ μ :=
    row_inner_hasSubgaussianMGF_of_max_row_psi2
      (A := A) (μ := μ) hK_def hA.finite_row_psi2 i0 hx
  have hsubG :
      HasSubgaussianMGF (fun ω => inner ℝ (randomMatrixRowVector A i0 ω) x)
        ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ := by
    refine hasSubgaussianMGF_mono_param hbase ?_
    change K ^ 2 ≤ (2 * K) ^ 2
    nlinarith [sq_nonneg K]
  exact one_div_four_exp_sq_le_sq_of_subgaussian_second_moment_one hK hsubG
    (hA.row_projection_second_moment_one i0 hx)

lemma randomVector_bad_event_subset_quadratic_tail {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {X : Fin n → Ω → ℝ}
    (hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ)
    (hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1) {t : ℝ}
    (ht_nonneg : 0 ≤ t) (ht_le : t ≤ √(n : ℝ)) :
    {ω | ‖HansonWright.randomVector X ω‖ < √(n : ℝ) - t} ⊆
      {ω | √(n : ℝ) * t ≤
        |HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin n) (Fin n) ℝ) X ω|} := by
  intro ω hω
  let q : ℝ := ‖HansonWright.randomVector X ω‖
  let a : ℝ := √(n : ℝ)
  have hq_nonneg : 0 ≤ q := norm_nonneg _
  have ha_nonneg : 0 ≤ a := sqrt_nonneg _
  have hdiff_nonneg : 0 ≤ a - t := sub_nonneg.mpr ht_le
  have hq_sq_lt : q ^ 2 < (a - t) ^ 2 := by
    exact (sq_lt_sq₀ hq_nonneg hdiff_nonneg).mpr hω
  have ha_sq : a ^ 2 = (n : ℝ) := by
    exact sq_sqrt (Nat.cast_nonneg n)
  have hdev : a * t ≤ (n : ℝ) - q ^ 2 := by
    have hle : q ^ 2 ≤ (a - t) ^ 2 := le_of_lt hq_sq_lt
    nlinarith [ha_sq, ht_nonneg, ht_le]
  have hcenter :
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin n) (Fin n) ℝ) X ω =
        q ^ 2 - (n : ℝ) := by
    simpa [q] using centeredQuadraticForm_one_eq_norm_sq_sub_nat hX_sq_int hsecond ω
  have hnonpos : q ^ 2 - (n : ℝ) ≤ 0 := by
    nlinarith [hdev, mul_nonneg ha_nonneg ht_nonneg]
  change √(n : ℝ) * t ≤
    |HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin n) (Fin n) ℝ) X ω|
  rw [hcenter]
  rw [abs_of_nonpos hnonpos]
  linarith

lemma randomVector_quadratic_tail_rhs_le {n : ℕ} (hn : 0 < n)
    {K t : ℝ} (hK : 0 < K) (ht : 0 < t)
    (ht_le : t ≤ √(n : ℝ)) (hK_lower : 1 / (4 * exp 1 ^ 2) ≤ K ^ 2)
    {O F : ℝ} (hO_pos : 0 < O) (hO_le : O ≤ 1) (hF_sq : F ^ 2 = (n : ℝ)) :
    2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
        min ((√(n : ℝ) * t) ^ 2 / ((2 * K) ^ 4 * F ^ 2))
          ((√(n : ℝ) * t) / ((2 * K) ^ 2 * O))) ≤
      2 * exp (-(randomVectorLowerTailConstant * t ^ 2 / K ^ 4)) := by
  let d : ℝ := t ^ 2 / (16 * exp 1 ^ 2 * K ^ 4)
  have hnR_pos : 0 < (n : ℝ) := by exact_mod_cast hn
  have hsqrt_sq : (√(n : ℝ)) ^ 2 = (n : ℝ) := sq_sqrt hnR_pos.le
  have hKsq_pos : 0 < K ^ 2 := sq_pos_of_pos hK
  have hK4_pos : 0 < K ^ 4 := pow_pos hK 4
  have hexp_sq_ge_one : 1 ≤ exp 1 ^ 2 := by
    have hone_exp : 1 ≤ exp 1 := by
      exact one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
    nlinarith
  have hfirst :
      d ≤ (√(n : ℝ) * t) ^ 2 / ((2 * K) ^ 4 * F ^ 2) := by
    have hfirst_eq :
        (√(n : ℝ) * t) ^ 2 / ((2 * K) ^ 4 * F ^ 2) =
          t ^ 2 / (16 * K ^ 4) := by
      rw [hF_sq]
      field_simp [hK.ne', ne_of_gt hnR_pos]
      nlinarith [hsqrt_sq]
    rw [hfirst_eq]
    dsimp [d]
    rw [div_le_div_iff₀ (by positivity : (0 : ℝ) < 16 * exp 1 ^ 2 * K ^ 4)
      (by positivity : (0 : ℝ) < 16 * K ^ 4)]
    have hden_le : 16 * K ^ 4 ≤ 16 * exp 1 ^ 2 * K ^ 4 := by
      nlinarith [hexp_sq_ge_one, hK4_pos.le]
    exact mul_le_mul_of_nonneg_left hden_le (sq_nonneg t)
  have hsecond :
      d ≤ (√(n : ℝ) * t) / ((2 * K) ^ 2 * O) := by
    have hden_pos : 0 < (2 * K) ^ 2 * O := by positivity
    have hbasic : t ^ 2 / (4 * K ^ 2) ≤ (√(n : ℝ) * t) / ((2 * K) ^ 2 * O) := by
      have htO_le : t * O ≤ √(n : ℝ) := by
        calc
          t * O ≤ t * 1 := mul_le_mul_of_nonneg_left hO_le ht.le
          _ = t := by ring
          _ ≤ √(n : ℝ) := ht_le
      rw [div_le_div_iff₀ (by positivity : (0 : ℝ) < 4 * K ^ 2) hden_pos]
      ring_nf
      have ht_mul : t * (t * O) ≤ t * √(n : ℝ) :=
        mul_le_mul_of_nonneg_left htO_le ht.le
      have hscaled : 4 * K ^ 2 * (t * (t * O)) ≤ 4 * K ^ 2 * (t * √(n : ℝ)) :=
        mul_le_mul_of_nonneg_left ht_mul (by positivity)
      nlinarith [hscaled]
    have hd_le_basic : d ≤ t ^ 2 / (4 * K ^ 2) := by
      rw [div_le_div_iff₀ (by positivity : (0 : ℝ) < 16 * exp 1 ^ 2 * K ^ 4)
        (by positivity : (0 : ℝ) < 4 * K ^ 2)]
      have hKlower' : 1 ≤ 4 * exp 1 ^ 2 * K ^ 2 := by
        have hden_pos' : 0 < 4 * exp 1 ^ 2 := by positivity
        have htmp := (div_le_iff₀ hden_pos').mp hK_lower
        nlinarith
      have hden_le : 4 * K ^ 2 ≤ 16 * exp 1 ^ 2 * K ^ 4 := by
        have hmul := mul_le_mul_of_nonneg_left hKlower'
          (by positivity : 0 ≤ 4 * K ^ 2)
        nlinarith [hmul]
      exact mul_le_mul_of_nonneg_left hden_le (sq_nonneg t)
    exact hd_le_basic.trans hbasic
  have hmin : d ≤
      min ((√(n : ℝ) * t) ^ 2 / ((2 * K) ^ 4 * F ^ 2))
        ((√(n : ℝ) * t) / ((2 * K) ^ 2 * O)) :=
    le_min hfirst hsecond
  have hC : 0 < hdpHansonWrightConstant := by
    unfold hdpHansonWrightConstant
    positivity
  have hcoef_nonpos : -(1 / (4 * hdpHansonWrightConstant)) ≤ 0 := by
    have hpos : 0 ≤ 1 / (4 * hdpHansonWrightConstant) := by positivity
    linarith
  have harg :
      -(1 / (4 * hdpHansonWrightConstant)) *
          min ((√(n : ℝ) * t) ^ 2 / ((2 * K) ^ 4 * F ^ 2))
            ((√(n : ℝ) * t) / ((2 * K) ^ 2 * O)) ≤
        -(randomVectorLowerTailConstant * t ^ 2 / K ^ 4) := by
    calc
      -(1 / (4 * hdpHansonWrightConstant)) *
          min ((√(n : ℝ) * t) ^ 2 / ((2 * K) ^ 4 * F ^ 2))
            ((√(n : ℝ) * t) / ((2 * K) ^ 2 * O))
          ≤ -(1 / (4 * hdpHansonWrightConstant)) * d :=
            mul_le_mul_of_nonpos_left hmin hcoef_nonpos
      _ = -(randomVectorLowerTailConstant * t ^ 2 / K ^ 4) := by
        unfold randomVectorLowerTailConstant
        dsimp [d]
        field_simp [hK.ne', hC.ne', (exp_pos 1).ne']
        ring
  exact mul_le_mul_of_nonneg_left (exp_le_exp.mpr harg) (by norm_num)

theorem randomVector_norm_lower_tail_of_subgaussian_mgf_bound {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} (hn : 0 < n) {X : Fin n → Ω → ℝ} {K : ℝ}
    (hK : 0 < K) (h_indep : iIndepFun X μ)
    (hX_subG : ∀ i, HasSubgaussianMGF (X i) ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ)
    (hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1) :
    ∀ t : ℝ, 0 < t →
      (μ {ω | √(n : ℝ) - t ≤ ‖HansonWright.randomVector X ω‖}).toReal ≥
        1 - 2 * exp (-(randomVectorLowerTailConstant * t ^ 2 / K ^ 4)) := by
  let C0 : ℝ := hdpHansonWrightConstant
  have hC0_pos : 0 < C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    positivity
  have hexp_pos : 0 < exp 1 := exp_pos 1
  have hexp_ge_one : 1 ≤ exp 1 := one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
  have hexp_le_eight : exp 1 ≤ 8 := by
    nlinarith [exp_one_lt_three.le]
  have hC0_domain : 4 * exp 1 ≤ C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    have hmul := mul_le_mul_of_nonneg_left hexp_ge_one
      (by positivity : 0 ≤ 4 * exp 1)
    nlinarith
  have hC0_diag_quad : 8 * exp 1 ^ 3 ≤ C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    have hmul := mul_le_mul_of_nonneg_left hexp_le_eight
      (by positivity : 0 ≤ 8 * exp 1 ^ 2)
    nlinarith
  have hC0_ge_offdiag : 16 * exp 1 ≤ C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    have hmul := mul_le_mul_of_nonneg_left hexp_ge_one
      (by positivity : 0 ≤ 16 * exp 1)
    nlinarith
  have hC0_ge_one : 1 ≤ C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    nlinarith [hexp_ge_one, sq_nonneg (exp 1)]
  have hC0_sq_ge : C0 ≤ C0 ^ 2 := by
    have hC0_nonneg : 0 ≤ C0 := hC0_pos.le
    have hmul := mul_le_mul hC0_ge_one (le_refl C0) hC0_nonneg hC0_nonneg
    simpa [pow_two] using hmul
  have hC0_offdiag_domain : 16 * exp 1 ≤ C0 ^ 2 :=
    hC0_ge_offdiag.trans hC0_sq_ge
  have hC0_offdiag_quad : 64 * exp 1 ^ 2 ≤ C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    nlinarith [sq_nonneg (exp 1)]
  have hK2 : 0 < 2 * K := by positivity
  have hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ := by
    intro i
    exact integrable_pow_of_integrable_exp_mul
      (X := X i) (t := 1) one_ne_zero
      (by simpa using (hX_subG i).integrable_exp_mul 1)
      (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2
  have hF_pos : 0 < HansonWright.frobeniusNorm (1 : Matrix (Fin n) (Fin n) ℝ) := by
    have hsq := frobeniusNorm_one_sq (n := n)
    have hnR_pos : 0 < (n : ℝ) := by exact_mod_cast hn
    have hnonneg :=
      HansonWright.frobeniusNorm_nonneg (1 : Matrix (Fin n) (Fin n) ℝ)
    nlinarith
  have hF_sq :
      HansonWright.frobeniusNorm (1 : Matrix (Fin n) (Fin n) ℝ) ^ 2 = (n : ℝ) :=
    frobeniusNorm_one_sq
  have hOp_pos : 0 < HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ) :=
    operatorNorm_one_pos_of_pos hn
  have hOp_le : HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ) ≤ 1 :=
    operatorNorm_one_le
  let i0 : Fin n := ⟨0, hn⟩
  have hK_lower : 1 / (4 * exp 1 ^ 2) ≤ K ^ 2 :=
    one_div_four_exp_sq_le_sq_of_subgaussian_second_moment_one hK
      (hX_subG i0) (hsecond i0)
  intro t ht
  by_cases ht_big : √(n : ℝ) ≤ t
  · have hset :
        {ω | √(n : ℝ) - t ≤ ‖HansonWright.randomVector X ω‖} = Set.univ := by
      ext ω
      constructor
      · intro
        trivial
      · intro
        exact (sub_nonpos.mpr ht_big).trans (norm_nonneg _)
    rw [hset, IsProbabilityMeasure.measure_univ, ENNReal.toReal_one]
    have hnonneg : 0 ≤ 2 * exp (-(randomVectorLowerTailConstant * t ^ 2 / K ^ 4)) := by
      positivity
    linarith
  · have ht_le : t ≤ √(n : ℝ) := le_of_not_ge ht_big
    have htail :=
      HansonWright.hanson_wright_inequality (μ := μ)
        (A := (1 : Matrix (Fin n) (Fin n) ℝ)) (X := X)
        (K := 2 * K) (C := C0) (t := √(n : ℝ) * t)
        hK2 hC0_pos hC0_domain hC0_diag_quad hC0_offdiag_domain hC0_offdiag_quad
        hF_pos hOp_pos h_indep hX_subG (by positivity)
    have htail_bound :
        (μ {ω | √(n : ℝ) * t ≤
          |HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin n) (Fin n) ℝ) X ω|}).toReal ≤
          2 * exp (-(randomVectorLowerTailConstant * t ^ 2 / K ^ 4)) := by
      have htail' := htail.trans
        (randomVector_quadratic_tail_rhs_le (n := n) hn hK ht ht_le hK_lower
          (O := HansonWright.operatorNorm (1 : Matrix (Fin n) (Fin n) ℝ))
          (F := HansonWright.frobeniusNorm (1 : Matrix (Fin n) (Fin n) ℝ))
          hOp_pos hOp_le hF_sq)
      simpa [C0] using htail'
    have hbad_subset :=
      randomVector_bad_event_subset_quadratic_tail (μ := μ) (X := X)
        hX_sq_int hsecond ht.le ht_le
    have hbad_tail :
        (μ {ω | ‖HansonWright.randomVector X ω‖ < √(n : ℝ) - t}).toReal ≤
          2 * exp (-(randomVectorLowerTailConstant * t ^ 2 / K ^ 4)) := by
      exact (ENNReal.toReal_mono (measure_ne_top μ _)
        (measure_mono hbad_subset)).trans htail_bound
    exact probability_ge_of_lower_tail_real_le
      (μ := μ) (X := fun ω => ‖HansonWright.randomVector X ω‖)
      (B := √(n : ℝ) - t) hbad_tail

theorem fixed_unit_vector_matrix_apply_lower_tail_of_row_subgaussian {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ) (hK : 0 < K)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    ∀ t : ℝ, 0 < t →
      (μ {ω | √(m : ℝ) - t ≤ ‖(randomMatrix A ω).toEuclideanLin x‖}).toReal ≥
        1 - 2 * exp (-(randomVectorLowerTailConstant * t ^ 2 / K ^ 4)) := by
  let X : Fin m → Ω → ℝ :=
    fun i => fun ω => inner ℝ (randomMatrixRowVector A i ω) x
  have htail :=
    randomVector_norm_lower_tail_of_subgaussian_mgf_bound
      (μ := μ) (n := m) hm (X := X) (K := K) hK
      (by
        dsimp [X]
        exact hA.row_projection_iIndepFun x)
      (fun i => by
        dsimp [X]
        exact hA.row_projection_hasSubgaussianMGF_two_mul_max hK_def i hx)
      (fun i => by
        dsimp [X]
        exact hA.row_projection_second_moment_one i hx)
  intro t ht
  simpa [X, randomVector_rowProjection_eq_toEuclideanLin] using htail t ht

theorem fixed_unit_vector_matrix_apply_sq_sub_dim_tail_of_row_subgaussian {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K u : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hu : 0 ≤ u)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    (μ {ω | u ≤ |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)|}).toReal ≤
      2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
        min (u ^ 2 /
          (K ^ 4 * HansonWright.frobeniusNorm (1 : Matrix (Fin m) (Fin m) ℝ) ^ 2))
          (u / (K ^ 2 * HansonWright.operatorNorm (1 : Matrix (Fin m) (Fin m) ℝ)))) := by
  let X : Fin m → Ω → ℝ :=
    fun i => fun ω => inner ℝ (randomMatrixRowVector A i ω) x
  have h_indep : iIndepFun X μ := by
    dsimp [X]
    exact hA.row_projection_iIndepFun x
  have hX_subG :
      ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ := by
    intro i
    dsimp [X]
    exact row_inner_hasSubgaussianMGF_of_max_row_psi2
      (A := A) (μ := μ) hK_def hA.finite_row_psi2 i hx
  have hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ := by
    intro i
    exact integrable_pow_of_integrable_exp_mul
      (X := X i) (t := 1) one_ne_zero
      (by simpa using (hX_subG i).integrable_exp_mul 1)
      (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2
  have hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1 := by
    intro i
    dsimp [X]
    exact hA.row_projection_second_moment_one i hx
  have hF_pos : 0 <
      HansonWright.frobeniusNorm (1 : Matrix (Fin m) (Fin m) ℝ) := by
    have hsq := frobeniusNorm_one_sq (n := m)
    have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
    have hnonneg :=
      HansonWright.frobeniusNorm_nonneg (1 : Matrix (Fin m) (Fin m) ℝ)
    nlinarith
  have hOp_pos : 0 < HansonWright.operatorNorm (1 : Matrix (Fin m) (Fin m) ℝ) :=
    operatorNorm_one_pos_of_pos hm
  have htail :=
    HansonWright.hanson_wright_inequality (μ := μ)
      (A := (1 : Matrix (Fin m) (Fin m) ℝ)) (X := X)
      (K := K) (C := hdpHansonWrightConstant) (t := u)
      hK hdpHansonWrightConstant_pos hdpHansonWrightConstant_domain
      hdpHansonWrightConstant_diag_quad hdpHansonWrightConstant_offdiag_domain
      hdpHansonWrightConstant_offdiag_quad hF_pos hOp_pos h_indep hX_subG hu
  have hcenter_eq : ∀ ω,
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X ω =
        ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) := by
    intro ω
    calc
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X ω
          = ‖HansonWright.randomVector X ω‖ ^ 2 - (m : ℝ) := by
              exact centeredQuadraticForm_one_eq_norm_sq_sub_nat hX_sq_int hsecond ω
      _ = ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) := by
              rw [show HansonWright.randomVector X ω =
                (randomMatrix A ω).toEuclideanLin x by
                  dsimp [X]
                  exact randomVector_rowProjection_eq_toEuclideanLin A ω x]
  have hset :
      {ω | u ≤ |HansonWright.centeredQuadraticForm μ
          (1 : Matrix (Fin m) (Fin m) ℝ) X ω|} =
        {ω | u ≤ |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)|} := by
    ext ω
    simp [hcenter_eq ω]
  rw [← hset]
  exact htail

lemma fixed_unit_vector_matrix_apply_sq_sub_dim_cgf_le_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K l : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hl : |l| ≤ (2 * hdpHansonWrightConstant * K ^ 2)⁻¹)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    cgf (fun ω => ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)) μ l ≤
      hdpHansonWrightConstant * l ^ 2 * K ^ 4 * (m : ℝ) := by
  let X : Fin m → Ω → ℝ :=
    fun i => fun ω => inner ℝ (randomMatrixRowVector A i ω) x
  have h_indep : iIndepFun X μ := by
    dsimp [X]
    exact hA.row_projection_iIndepFun x
  have hX_subG :
      ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ := by
    intro i
    dsimp [X]
    exact row_inner_hasSubgaussianMGF_of_max_row_psi2
      (A := A) (μ := μ) hK_def hA.finite_row_psi2 i hx
  have hHW : HansonWright.HasHansonWrightMGF μ (1 : Matrix (Fin m) (Fin m) ℝ)
      X K hdpHansonWrightConstant :=
    HansonWright.hasHansonWrightMGF_of_subgaussian
      hK hdpHansonWrightConstant_pos hdpHansonWrightConstant_domain
      hdpHansonWrightConstant_diag_quad hdpHansonWrightConstant_offdiag_domain
      hdpHansonWrightConstant_offdiag_quad (operatorNorm_one_pos_of_pos hm) h_indep hX_subG
  have hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ := by
    intro i
    exact integrable_pow_of_integrable_exp_mul
      (X := X i) (t := 1) one_ne_zero
      (by simpa using (hX_subG i).integrable_exp_mul 1)
      (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2
  have hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1 := by
    intro i
    dsimp [X]
    exact hA.row_projection_second_moment_one i hx
  have hcenter_eq : ∀ ω,
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X ω =
        ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) := by
    intro ω
    calc
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X ω
          = ‖HansonWright.randomVector X ω‖ ^ 2 - (m : ℝ) := by
              exact centeredQuadraticForm_one_eq_norm_sq_sub_nat hX_sq_int hsecond ω
      _ = ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) := by
              rw [show HansonWright.randomVector X ω =
                (randomMatrix A ω).toEuclideanLin x by
                  dsimp [X]
                  exact randomVector_rowProjection_eq_toEuclideanLin A ω x]
  have hdomain :
      |l| ≤ (2 * hdpHansonWrightConstant * K ^ 2 *
          HansonWright.operatorNorm (1 : Matrix (Fin m) (Fin m) ℝ))⁻¹ := by
    simpa [operatorNorm_one_eq_of_pos hm, mul_assoc] using hl
  have hcgf := hHW.1 l hdomain
  have hfun :
      (fun ω => ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)) =
        HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X := by
    funext ω
    exact (hcenter_eq ω).symm
  calc
    cgf (fun ω => ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)) μ l
        = cgf (HansonWright.centeredQuadraticForm μ
            (1 : Matrix (Fin m) (Fin m) ℝ) X) μ l := by rw [hfun]
    _ ≤ hdpHansonWrightConstant * l ^ 2 * K ^ 4 *
        HansonWright.frobeniusNorm (1 : Matrix (Fin m) (Fin m) ℝ) ^ 2 := hcgf
    _ = hdpHansonWrightConstant * l ^ 2 * K ^ 4 * (m : ℝ) := by
        rw [frobeniusNorm_one_sq (n := m)]

lemma fixed_unit_vector_matrix_apply_sq_sub_dim_integrable_exp_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K l : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hl : |l| ≤ (2 * hdpHansonWrightConstant * K ^ 2)⁻¹)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    Integrable (fun ω => exp (l *
      (‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)))) μ := by
  let X : Fin m → Ω → ℝ :=
    fun i => fun ω => inner ℝ (randomMatrixRowVector A i ω) x
  have h_indep : iIndepFun X μ := by
    dsimp [X]
    exact hA.row_projection_iIndepFun x
  have hX_subG :
      ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ := by
    intro i
    dsimp [X]
    exact row_inner_hasSubgaussianMGF_of_max_row_psi2
      (A := A) (μ := μ) hK_def hA.finite_row_psi2 i hx
  have hHW : HansonWright.HasHansonWrightMGF μ (1 : Matrix (Fin m) (Fin m) ℝ)
      X K hdpHansonWrightConstant :=
    HansonWright.hasHansonWrightMGF_of_subgaussian
      hK hdpHansonWrightConstant_pos hdpHansonWrightConstant_domain
      hdpHansonWrightConstant_diag_quad hdpHansonWrightConstant_offdiag_domain
      hdpHansonWrightConstant_offdiag_quad (operatorNorm_one_pos_of_pos hm) h_indep hX_subG
  have hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ := by
    intro i
    exact integrable_pow_of_integrable_exp_mul
      (X := X i) (t := 1) one_ne_zero
      (by simpa using (hX_subG i).integrable_exp_mul 1)
      (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2
  have hsecond : ∀ i, ∫ ω, X i ω ^ 2 ∂μ = 1 := by
    intro i
    dsimp [X]
    exact hA.row_projection_second_moment_one i hx
  have hcenter_eq : ∀ ω,
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X ω =
        ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) := by
    intro ω
    calc
      HansonWright.centeredQuadraticForm μ (1 : Matrix (Fin m) (Fin m) ℝ) X ω
          = ‖HansonWright.randomVector X ω‖ ^ 2 - (m : ℝ) := by
              exact centeredQuadraticForm_one_eq_norm_sq_sub_nat hX_sq_int hsecond ω
      _ = ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) := by
              rw [show HansonWright.randomVector X ω =
                (randomMatrix A ω).toEuclideanLin x by
                  dsimp [X]
                  exact randomVector_rowProjection_eq_toEuclideanLin A ω x]
  have hdomain :
      |l| ≤ (2 * hdpHansonWrightConstant * K ^ 2 *
          HansonWright.operatorNorm (1 : Matrix (Fin m) (Fin m) ℝ))⁻¹ := by
    simpa [operatorNorm_one_eq_of_pos hm, mul_assoc] using hl
  have hint := hHW.2 l hdomain
  convert hint using 1
  funext ω
  rw [hcenter_eq ω]

lemma fixed_vector_matrix_apply_sq_sub_scaled_dim_cgf_le_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K l : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hl : |l| ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2))
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) :
    cgf (fun ω => (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2)
        μ l ≤ hdpHansonWrightConstant * l ^ 2 * K ^ 4 / (m : ℝ) := by
  by_cases hx0 : x = 0
  · have hfun :
        (fun ω => (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2) =
          fun _ => (0 : ℝ) := by
      funext ω
      simp [hx0]
    rw [hfun]
    unfold cgf mgf
    simp
    unfold hdpHansonWrightConstant
    positivity
  · let r : ℝ := ‖x‖
    let y : EuclideanSpace ℝ (Fin n) := r⁻¹ • x
    let a : ℝ := r ^ 2 / (m : ℝ)
    let θ : ℝ := l * a
    have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
    have hr_pos : 0 < r := by
      dsimp [r]
      exact norm_pos_iff.mpr hx0
    have hr_nonneg : 0 ≤ r := hr_pos.le
    have ha_nonneg : 0 ≤ a := by
      dsimp [a]
      positivity
    have hr_sq_le_one : r ^ 2 ≤ 1 := by
      dsimp [r]
      nlinarith [norm_nonneg x, hx]
    have hy : ‖y‖ = 1 := by
      dsimp [y]
      rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hr_pos.le)]
      exact inv_mul_cancel₀ hr_pos.ne'
    have hx_eq : x = r • y := by
      dsimp [y]
      rw [smul_smul, mul_inv_cancel₀ hr_pos.ne', one_smul]
    have hscaled_eq : ∀ ω,
        (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2 =
          a * (‖(randomMatrix A ω).toEuclideanLin y‖ ^ 2 - (m : ℝ)) := by
      intro ω
      let T := (randomMatrix A ω).toEuclideanLin
      have hTx : T x = r • T y := by
        rw [hx_eq, map_smul]
      have hrnorm : ‖(r : ℝ)‖ = r := Real.norm_of_nonneg hr_nonneg
      dsimp [a]
      rw [hTx, norm_smul, hrnorm]
      change (m : ℝ)⁻¹ * (r * ‖T y‖) ^ 2 - r ^ 2 =
        r ^ 2 / (m : ℝ) * (‖T y‖ ^ 2 - (m : ℝ))
      field_simp [hmR_pos.ne']
    have hθ_domain : |θ| ≤ (2 * hdpHansonWrightConstant * K ^ 2)⁻¹ := by
      have ha_le : a ≤ (m : ℝ)⁻¹ := by
        dsimp [a]
        rw [div_eq_mul_inv]
        simpa [one_mul] using
          mul_le_mul_of_nonneg_right hr_sq_le_one (inv_nonneg.mpr hmR_pos.le)
      calc
        |θ| = |l| * a := by
          dsimp [θ]
          rw [abs_mul, abs_of_nonneg ha_nonneg]
        _ ≤ ((m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2)) * a :=
          mul_le_mul_of_nonneg_right hl ha_nonneg
        _ ≤ ((m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2)) * (m : ℝ)⁻¹ := by
          have hcoef_nonneg :
              0 ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2) := by
            unfold hdpHansonWrightConstant
            positivity
          exact mul_le_mul_of_nonneg_left ha_le hcoef_nonneg
        _ = (2 * hdpHansonWrightConstant * K ^ 2)⁻¹ := by
          field_simp [hmR_pos.ne', hK.ne', hdpHansonWrightConstant_pos.ne']
    have hcgf_unit :=
      fixed_unit_vector_matrix_apply_sq_sub_dim_cgf_le_hdp
        (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hθ_domain hy
    have hcgf_eq :
        cgf (fun ω => (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2)
            μ l =
          cgf (fun ω => ‖(randomMatrix A ω).toEuclideanLin y‖ ^ 2 - (m : ℝ)) μ θ := by
      unfold cgf mgf
      congr 1
      apply integral_congr_ae
      filter_upwards with ω
      congr 1
      rw [hscaled_eq ω]
      dsimp [θ]
      ring
    have hmain_le :
        hdpHansonWrightConstant * θ ^ 2 * K ^ 4 * (m : ℝ) ≤
          hdpHansonWrightConstant * l ^ 2 * K ^ 4 / (m : ℝ) := by
      have hr_four_le_one : r ^ 4 ≤ 1 := by
        nlinarith [hr_sq_le_one, sq_nonneg (r ^ 2)]
      calc
        hdpHansonWrightConstant * θ ^ 2 * K ^ 4 * (m : ℝ)
            = hdpHansonWrightConstant * l ^ 2 * K ^ 4 * (r ^ 4 / (m : ℝ)) := by
              dsimp [θ, a]
              field_simp [hmR_pos.ne']
        _ ≤ hdpHansonWrightConstant * l ^ 2 * K ^ 4 * (1 / (m : ℝ)) := by
          have hcoef_nonneg : 0 ≤ hdpHansonWrightConstant * l ^ 2 * K ^ 4 := by
            unfold hdpHansonWrightConstant
            positivity
          exact mul_le_mul_of_nonneg_left
            (div_le_div_of_nonneg_right hr_four_le_one hmR_pos.le) hcoef_nonneg
        _ = hdpHansonWrightConstant * l ^ 2 * K ^ 4 / (m : ℝ) := by ring
    rw [hcgf_eq]
    exact hcgf_unit.trans hmain_le

lemma fixed_vector_matrix_apply_sq_sub_scaled_dim_integrable_exp_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K l : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hl : |l| ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2))
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) :
    Integrable (fun ω => exp (l *
      ((m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2))) μ := by
  by_cases hx0 : x = 0
  · haveI : IsFiniteMeasure μ := inferInstance
    convert integrable_const (μ := μ) (c := (1 : ℝ)) using 1
    funext ω
    simp [hx0]
  · let r : ℝ := ‖x‖
    let y : EuclideanSpace ℝ (Fin n) := r⁻¹ • x
    let a : ℝ := r ^ 2 / (m : ℝ)
    let θ : ℝ := l * a
    have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
    have hr_pos : 0 < r := by
      dsimp [r]
      exact norm_pos_iff.mpr hx0
    have hr_nonneg : 0 ≤ r := hr_pos.le
    have ha_nonneg : 0 ≤ a := by
      dsimp [a]
      positivity
    have hr_sq_le_one : r ^ 2 ≤ 1 := by
      dsimp [r]
      nlinarith [norm_nonneg x, hx]
    have hy : ‖y‖ = 1 := by
      dsimp [y]
      rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hr_pos.le)]
      exact inv_mul_cancel₀ hr_pos.ne'
    have hx_eq : x = r • y := by
      dsimp [y]
      rw [smul_smul, mul_inv_cancel₀ hr_pos.ne', one_smul]
    have hscaled_eq : ∀ ω,
        (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2 =
          a * (‖(randomMatrix A ω).toEuclideanLin y‖ ^ 2 - (m : ℝ)) := by
      intro ω
      let T := (randomMatrix A ω).toEuclideanLin
      have hTx : T x = r • T y := by
        rw [hx_eq, map_smul]
      have hrnorm : ‖(r : ℝ)‖ = r := Real.norm_of_nonneg hr_nonneg
      dsimp [a]
      rw [hTx, norm_smul, hrnorm]
      change (m : ℝ)⁻¹ * (r * ‖T y‖) ^ 2 - r ^ 2 =
        r ^ 2 / (m : ℝ) * (‖T y‖ ^ 2 - (m : ℝ))
      field_simp [hmR_pos.ne']
    have hθ_domain : |θ| ≤ (2 * hdpHansonWrightConstant * K ^ 2)⁻¹ := by
      have ha_le : a ≤ (m : ℝ)⁻¹ := by
        dsimp [a]
        rw [div_eq_mul_inv]
        simpa [one_mul] using
          mul_le_mul_of_nonneg_right hr_sq_le_one (inv_nonneg.mpr hmR_pos.le)
      calc
        |θ| = |l| * a := by
          dsimp [θ]
          rw [abs_mul, abs_of_nonneg ha_nonneg]
        _ ≤ ((m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2)) * a :=
          mul_le_mul_of_nonneg_right hl ha_nonneg
        _ ≤ ((m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2)) * (m : ℝ)⁻¹ := by
          have hcoef_nonneg :
              0 ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2) := by
            unfold hdpHansonWrightConstant
            positivity
          exact mul_le_mul_of_nonneg_left ha_le hcoef_nonneg
        _ = (2 * hdpHansonWrightConstant * K ^ 2)⁻¹ := by
          field_simp [hmR_pos.ne', hK.ne', hdpHansonWrightConstant_pos.ne']
    have hint :=
      fixed_unit_vector_matrix_apply_sq_sub_dim_integrable_exp_hdp
        (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hθ_domain hy
    convert hint using 1
    funext ω
    rw [hscaled_eq ω]
    dsimp [θ]
    ring_nf

lemma fixed_vector_matrix_apply_sq_sub_scaled_dim_integrable_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    {x : EuclideanSpace ℝ (Fin n)} :
    Integrable
      (fun ω => (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2) μ := by
  by_cases hx0 : x = 0
  · convert integrable_const (μ := μ) (c := (0 : ℝ)) using 1
    funext ω
    simp [hx0]
  · let r : ℝ := ‖x‖
    let y : EuclideanSpace ℝ (Fin n) := r⁻¹ • x
    let a : ℝ := r ^ 2 / (m : ℝ)
    have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
    have hr_pos : 0 < r := by
      dsimp [r]
      exact norm_pos_iff.mpr hx0
    have hr_nonneg : 0 ≤ r := hr_pos.le
    have hy : ‖y‖ = 1 := by
      dsimp [y]
      rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hr_pos.le)]
      exact inv_mul_cancel₀ hr_pos.ne'
    have hx_eq : x = r • y := by
      dsimp [y]
      rw [smul_smul, mul_inv_cancel₀ hr_pos.ne', one_smul]
    let X : Fin m → Ω → ℝ :=
      fun i => fun ω => inner ℝ (randomMatrixRowVector A i ω) y
    have hX_subG :
        ∀ i, HasSubgaussianMGF (X i) ⟨K ^ 2, sq_nonneg K⟩ μ := by
      intro i
      dsimp [X]
      exact row_inner_hasSubgaussianMGF_of_max_row_psi2
        (A := A) (μ := μ) hK_def hA.finite_row_psi2 i hy
    have hX_sq_int : ∀ i, Integrable (fun ω => X i ω ^ 2) μ := by
      intro i
      exact integrable_pow_of_integrable_exp_mul
        (X := X i) (t := 1) one_ne_zero
        (by simpa using (hX_subG i).integrable_exp_mul 1)
        (by simpa using (hX_subG i).integrable_exp_mul (-1)) 2
    have hunit_int :
        Integrable (fun ω => ‖(randomMatrix A ω).toEuclideanLin y‖ ^ 2 - (m : ℝ)) μ := by
      have hsum_int : Integrable (fun ω => ∑ i, X i ω ^ 2) μ :=
        integrable_finsetSum _ fun i _ => hX_sq_int i
      refine (hsum_int.sub (integrable_const (c := (m : ℝ)))).congr ?_
      filter_upwards with ω
      rw [show (randomMatrix A ω).toEuclideanLin y = HansonWright.randomVector X ω by
        dsimp [X]
        exact (randomVector_rowProjection_eq_toEuclideanLin A ω y).symm]
      rw [randomVector_norm_sq_eq_sum_sq]
      rfl
    have hscaled_eq : ∀ ω,
        (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2 =
          a * (‖(randomMatrix A ω).toEuclideanLin y‖ ^ 2 - (m : ℝ)) := by
      intro ω
      let T := (randomMatrix A ω).toEuclideanLin
      have hTx : T x = r • T y := by
        rw [hx_eq, map_smul]
      have hrnorm : ‖(r : ℝ)‖ = r := Real.norm_of_nonneg hr_nonneg
      dsimp [a]
      rw [hTx, norm_smul, hrnorm]
      change (m : ℝ)⁻¹ * (r * ‖T y‖) ^ 2 - r ^ 2 =
        r ^ 2 / (m : ℝ) * (‖T y‖ ^ 2 - (m : ℝ))
      field_simp [hmR_pos.ne']
    refine (hunit_int.const_mul a).congr ?_
    filter_upwards with ω
    exact (hscaled_eq ω).symm

lemma fixed_vector_matrix_apply_sq_sub_scaled_dim_measurable {m n : ℕ}
    {A : Fin m → Fin n → Ω → ℝ}
    (hA_meas : ∀ i j, Measurable (A i j))
    (x : EuclideanSpace ℝ (Fin n)) :
    Measurable
      (fun ω => (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - ‖x‖ ^ 2) := by
  let X : Fin m → Ω → ℝ :=
    fun i => fun ω => inner ℝ (randomMatrixRowVector A i ω) x
  have hX_meas : ∀ i, Measurable (X i) := by
    intro i
    have hsum :
        Measurable fun ω => ∑ j : Fin n, x j * A i j ω := by
      apply Finset.measurable_sum
      intro j _
      exact (hA_meas i j).const_mul _
    convert hsum using 1
    funext ω
    dsimp [X, randomMatrixRowVector]
    rw [PiLp.inner_apply]
    simp only [RCLike.inner_apply, conj_trivial]
    apply Finset.sum_congr rfl
    intro j _
    simp [matrixRowVector]
  have hsum_meas : Measurable fun ω => ∑ i, X i ω ^ 2 := by
    apply Finset.measurable_sum
    intro i _
    exact (hX_meas i).pow_const 2
  convert (hsum_meas.const_mul ((m : ℝ)⁻¹)).sub_const (‖x‖ ^ 2) using 1
  funext ω
  rw [show (randomMatrix A ω).toEuclideanLin x = HansonWright.randomVector X ω by
    dsimp [X]
    exact (randomVector_rowProjection_eq_toEuclideanLin A ω x).symm]
  rw [randomVector_norm_sq_eq_sum_sq]

theorem fixed_unit_vector_matrix_apply_sq_sub_dim_tail_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K u : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hu : 0 ≤ u)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ = 1) :
    (μ {ω | u ≤ |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ)|}).toReal ≤
      2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
        min (u ^ 2 / (K ^ 4 * (m : ℝ))) (u / K ^ 2)) := by
  have htail :=
    fixed_unit_vector_matrix_apply_sq_sub_dim_tail_of_row_subgaussian
      (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hu hx
  simpa [frobeniusNorm_one_sq (n := m), operatorNorm_one_eq_of_pos hm] using htail

theorem fixed_vector_matrix_apply_sq_sub_scaled_dim_tail_hdp {m n : ℕ}
    (hm : 0 < m) {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K u : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hu : 0 < u)
    {x : EuclideanSpace ℝ (Fin n)} (hx : ‖x‖ ≤ 1) :
    (μ {ω |
      u ≤ |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2|}).toReal ≤
      2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
        min (u ^ 2 / (K ^ 4 * (m : ℝ))) (u / K ^ 2)) := by
  by_cases hx0 : x = 0
  · have hset :
        {ω : Ω |
          u ≤ |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2|} =
          ∅ := by
      ext ω
      simp [hx0, not_le_of_gt hu]
    rw [hset, measure_empty, ENNReal.toReal_zero]
    positivity
  · let r : ℝ := ‖x‖
    let y : EuclideanSpace ℝ (Fin n) := r⁻¹ • x
    have hr_pos : 0 < r := by
      dsimp [r]
      exact norm_pos_iff.mpr hx0
    have hy : ‖y‖ = 1 := by
      dsimp [y]
      rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hr_pos.le)]
      exact inv_mul_cancel₀ hr_pos.ne'
    have hx_eq : x = r • y := by
      dsimp [y]
      rw [smul_smul, mul_inv_cancel₀ hr_pos.ne', one_smul]
    have hr_sq_le_one : r ^ 2 ≤ 1 := by
      dsimp [r]
      nlinarith [norm_nonneg x, hx]
    have hsubset :
        {ω |
          u ≤ |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2|} ⊆
          {ω | u ≤ |‖(randomMatrix A ω).toEuclideanLin y‖ ^ 2 - (m : ℝ)|} := by
      intro ω hω
      let T := (randomMatrix A ω).toEuclideanLin
      have hTx : T x = r • T y := by
        rw [hx_eq, map_smul]
      have hdiff :
          ‖T x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2 =
            r ^ 2 * (‖T y‖ ^ 2 - (m : ℝ)) := by
        rw [hTx]
        have hrnorm : ‖(r : ℝ)‖ = r := Real.norm_of_nonneg hr_pos.le
        rw [norm_smul, hrnorm]
        change (r * ‖T y‖) ^ 2 - (m : ℝ) * r ^ 2 =
          r ^ 2 * (‖T y‖ ^ 2 - (m : ℝ))
        ring
      have hdiff_abs :
          |‖T x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2| =
            r ^ 2 * |‖T y‖ ^ 2 - (m : ℝ)| := by
        rw [hdiff, abs_mul, abs_of_nonneg (sq_nonneg r)]
      calc
        u ≤ |‖T x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2| := hω
        _ = r ^ 2 * |‖T y‖ ^ 2 - (m : ℝ)| := hdiff_abs
        _ ≤ |‖T y‖ ^ 2 - (m : ℝ)| := by
          calc
            r ^ 2 * |‖T y‖ ^ 2 - (m : ℝ)|
                ≤ 1 * |‖T y‖ ^ 2 - (m : ℝ)| :=
                  mul_le_mul_of_nonneg_right hr_sq_le_one (abs_nonneg _)
            _ = |‖T y‖ ^ 2 - (m : ℝ)| := one_mul _
    have htail :=
      fixed_unit_vector_matrix_apply_sq_sub_dim_tail_hdp
        (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hu.le hy
    exact (ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hsubset)).trans htail

lemma sampleCovarianceDeviationOperatorNorm_tail_of_quarter_centered_quadratic_net
    {m n : ℕ} (hm : 0 < m)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (N : CenteredMatrixBilinearNet n n (1 / 4))
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K u : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hu : 0 < u) :
    (μ {ω | 2 * u < sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * (N.domainNet.card : ℝ) *
        exp (-(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
  let tailEvent : EuclideanSpace ℝ (Fin n) → Set Ω :=
    fun x => {ω |
      (m : ℝ) * u <
        |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2|}
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have hthreshold_pos : 0 < (m : ℝ) * u := mul_pos hmR_pos hu
  have hsubset :
      {ω | 2 * u < sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)} ⊆
        ⋃ x ∈ N.domainNet, tailEvent x := by
    intro ω hω
    have hdet :=
      sampleCovarianceDeviationOperatorNorm_event_subset_quarter_centered_quadratic_net
        (A := A) (u := u) N hu.le hω
    rcases Set.mem_iUnion₂.mp hdet with ⟨x, hx, hmem⟩
    let a : ℝ := ‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2
    let b : ℝ := ‖x‖ ^ 2
    have hmem' : u < |(m : ℝ)⁻¹ * a - b| := by
      simpa [a, b] using hmem
    have hmul : (m : ℝ) * u < (m : ℝ) * |(m : ℝ)⁻¹ * a - b| :=
      mul_lt_mul_of_pos_left hmem' hmR_pos
    have hscale : (m : ℝ) * |(m : ℝ)⁻¹ * a - b| =
        |a - (m : ℝ) * b| := by
      have hinner :
          (m : ℝ) * ((m : ℝ)⁻¹ * a - b) = a - (m : ℝ) * b := by
        field_simp [hmR_pos.ne']
      have hmul_abs :
          |(m : ℝ) * ((m : ℝ)⁻¹ * a - b)| =
            (m : ℝ) * |(m : ℝ)⁻¹ * a - b| := by
        rw [abs_mul, abs_of_pos hmR_pos]
      rw [← hmul_abs, hinner]
    exact Set.mem_iUnion₂.mpr ⟨x, hx, by
      have : (m : ℝ) * u < |a - (m : ℝ) * b| := by simpa [hscale] using hmul
      simpa [tailEvent, a, b] using this⟩
  have htail_each :
      ∀ x ∈ N.domainNet, μ.real (tailEvent x) ≤
        2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
    intro x hx
    have hx_norm : ‖x‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hx)
    have hmono :
        μ.real (tailEvent x) ≤
          (μ {ω |
            (m : ℝ) * u ≤
              |‖(randomMatrix A ω).toEuclideanLin x‖ ^ 2 - (m : ℝ) * ‖x‖ ^ 2|}).toReal := by
      refine ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono ?_)
      intro ω hω
      exact le_of_lt (by simpa [tailEvent] using hω)
    exact hmono.trans
      (fixed_vector_matrix_apply_sq_sub_scaled_dim_tail_hdp
        (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hthreshold_pos hx_norm)
  calc
    (μ {ω | 2 * u < sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)}).toReal
        ≤ μ.real (⋃ x ∈ N.domainNet, tailEvent x) := by
          simpa [Measure.real] using
            ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hsubset)
    _ ≤ ∑ x ∈ N.domainNet, μ.real (tailEvent x) :=
          measureReal_biUnion_finset_le N.domainNet tailEvent
    _ ≤ ∑ x ∈ N.domainNet,
          2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
            min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
          exact Finset.sum_le_sum fun x hx => htail_each x hx
    _ = (N.domainNet.card : ℝ) *
          (2 * exp (-(1 / (4 * hdpHansonWrightConstant)) *
            min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2))) := by
          simp [nsmul_eq_mul]
    _ = 2 * (N.domainNet.card : ℝ) *
          exp (-(1 / (4 * hdpHansonWrightConstant)) *
            min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
          ring

lemma sampleCovarianceDeviationOperatorNorm_tail_le_of_row_subgaussian
    {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K u : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hu : 0 < u) :
    (μ {ω | 2 * u < sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)}).toReal ≤
      2 * 17 ^ n *
        exp (-(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
  obtain ⟨N, hN_domain, _hN_codomain⟩ := exists_quarter_centeredMatrixBilinearNet n n
  have htail :=
    sampleCovarianceDeviationOperatorNorm_tail_of_quarter_centered_quadratic_net
      (m := m) (n := n) hm (A := A) (μ := μ) N hA hK_def hK hu
  have hcard :=
    centeredMatrixBilinearNet_domain_card_le_seventeen_pow hn N hN_domain
  have hfactor :
      2 * (N.domainNet.card : ℝ) *
          exp (-(1 / (4 * hdpHansonWrightConstant)) *
            min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) ≤
        2 * 17 ^ n *
          exp (-(1 / (4 * hdpHansonWrightConstant)) *
            min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
    have hleft : 2 * (N.domainNet.card : ℝ) ≤ 2 * 17 ^ n :=
      mul_le_mul_of_nonneg_left hcard (by norm_num : (0 : ℝ) ≤ 2)
    exact mul_le_mul_of_nonneg_right hleft (le_of_lt (exp_pos _))
  exact htail.trans hfactor

omit [MeasurableSpace Ω] in
lemma seventeen_pow_le_exp_seventeen_mul (d : ℕ) :
    (17 : ℝ) ^ d ≤ exp (17 * (d : ℝ)) := by
  have h17 : (17 : ℝ) ≤ exp 17 := by
    calc
      (17 : ℝ) ≤ 17 + 1 := by norm_num
      _ ≤ exp 17 := by simpa [add_comm] using Real.add_one_le_exp 17
  induction d with
  | zero =>
      simp
  | succ d ih =>
      calc
        (17 : ℝ) ^ (d + 1) = (17 : ℝ) ^ d * 17 := by rw [pow_succ]
        _ ≤ exp (17 * (d : ℝ)) * exp 17 :=
          mul_le_mul ih h17 (by norm_num : (0 : ℝ) ≤ 17) (le_of_lt (exp_pos _))
        _ = exp (17 * (d : ℝ) + 17) := by rw [← exp_add]
        _ = exp (17 * ((d + 1 : ℕ) : ℝ)) := by
          congr 1
          norm_num [Nat.cast_add]
          ring

omit [MeasurableSpace Ω] in
def sampleCovarianceDeviationHdpScale (m n : ℕ) (K t : ℝ) : ℝ :=
  K ^ 2 *
    (√((4 * hdpHansonWrightConstant * (17 * (n : ℝ) + t ^ 2)) / (m : ℝ)) +
      (4 * hdpHansonWrightConstant * (17 * (n : ℝ) + t ^ 2)) / (m : ℝ))

omit [MeasurableSpace Ω] in
def sampleCovarianceDeviationScaleConstant : ℝ :=
  68 * hdpHansonWrightConstant

omit [MeasurableSpace Ω] in
def twoSidedSubgaussianMatricesConstant : ℝ :=
  4 * sampleCovarianceDeviationScaleConstant

omit [MeasurableSpace Ω] in
lemma two_mul_scale_le_max_of_scale_bounds {D C k0 X Y : ℝ}
    (hD_nonneg : 0 ≤ D) (hC_nonneg : 0 ≤ C) (hk0_pos : 0 < k0)
    (hX_nonneg : 0 ≤ X) (hY_nonneg : 0 ≤ Y) (hY_le : Y ≤ X / k0)
    (hC_D : 4 * D ≤ C) (hCk0 : 1 ≤ C * k0) (hC_sq : 4 * D ≤ C ^ 2 * k0) :
    2 * D * X + 2 * D * Y * X ≤ max (C * X) ((C * X) ^ 2) := by
  by_cases hcase : C * X ≤ 1
  · have hCX_nonneg : 0 ≤ C * X := mul_nonneg hC_nonneg hX_nonneg
    have hmax : max (C * X) ((C * X) ^ 2) = C * X := by
      exact max_eq_left (by nlinarith [hCX_nonneg, hcase])
    have hk0Y_le_X : k0 * Y ≤ X := by
      calc
        k0 * Y = Y * k0 := by ring
        _ ≤ X := (le_div_iff₀ hk0_pos).mp hY_le
    have hCk0Y_le_CX : C * k0 * Y ≤ C * X := by
      calc
        C * k0 * Y = C * (k0 * Y) := by ring
        _ ≤ C * X := mul_le_mul_of_nonneg_left hk0Y_le_X hC_nonneg
    have hY_le_one : Y ≤ 1 := by
      calc
        Y ≤ (C * k0) * Y := by
          exact le_mul_of_one_le_left hY_nonneg hCk0
        _ = C * k0 * Y := by ring
        _ ≤ C * X := hCk0Y_le_CX
        _ ≤ 1 := hcase
    rw [hmax]
    have hterm : 2 * D * Y * X ≤ 2 * D * X := by
      have hcoef : 0 ≤ 2 * D * X := by positivity
      calc
        2 * D * Y * X = (2 * D * X) * Y := by ring
        _ ≤ (2 * D * X) * 1 := mul_le_mul_of_nonneg_left hY_le_one hcoef
        _ = 2 * D * X := by ring
    calc
      2 * D * X + 2 * D * Y * X
          ≤ 2 * D * X + 2 * D * X := add_le_add (le_refl (2 * D * X)) hterm
      _ = 4 * D * X := by ring
      _ ≤ C * X := mul_le_mul_of_nonneg_right hC_D hX_nonneg
  · have hcase' : 1 ≤ C * X := le_of_not_ge hcase
    have hCX_nonneg : 0 ≤ C * X := by linarith
    have hmax : max (C * X) ((C * X) ^ 2) = (C * X) ^ 2 := by
      exact max_eq_right (by nlinarith [hCX_nonneg, hcase'])
    rw [hmax]
    have hX_le_CX2 : X ≤ C * X ^ 2 := by
      have hmul := mul_le_mul_of_nonneg_right hcase' hX_nonneg
      simpa [mul_assoc, pow_two] using hmul
    have hfirst : 2 * D * X ≤ (C ^ 2 / 2) * X ^ 2 := by
      calc
        2 * D * X ≤ 2 * D * (C * X ^ 2) :=
          mul_le_mul_of_nonneg_left hX_le_CX2 (mul_nonneg (by positivity) hD_nonneg)
        _ = (2 * D * C) * X ^ 2 := by ring
        _ ≤ (C ^ 2 / 2) * X ^ 2 := by
          have hcoef : 2 * D * C ≤ C ^ 2 / 2 := by
            nlinarith [hC_D, hC_nonneg]
          exact mul_le_mul_of_nonneg_right hcoef (sq_nonneg X)
    have hYmul_le : Y * X ≤ (X / k0) * X :=
      mul_le_mul_of_nonneg_right hY_le hX_nonneg
    have hcoef_second : 2 * D / k0 ≤ C ^ 2 / 2 := by
      rw [div_le_iff₀ hk0_pos]
      nlinarith [hC_sq]
    have hsecond : 2 * D * Y * X ≤ (C ^ 2 / 2) * X ^ 2 := by
      calc
        2 * D * Y * X = 2 * D * (Y * X) := by ring
        _ ≤ 2 * D * ((X / k0) * X) :=
          mul_le_mul_of_nonneg_left hYmul_le (mul_nonneg (by positivity) hD_nonneg)
        _ = (2 * D / k0) * X ^ 2 := by
          field_simp [hk0_pos.ne']
        _ ≤ (C ^ 2 / 2) * X ^ 2 :=
          mul_le_mul_of_nonneg_right hcoef_second (sq_nonneg X)
    calc
      2 * D * X + 2 * D * Y * X
          ≤ (C ^ 2 / 2) * X ^ 2 + (C ^ 2 / 2) * X ^ 2 :=
            add_le_add hfirst hsecond
      _ = (C * X) ^ 2 := by ring

omit [MeasurableSpace Ω] in
lemma sampleCovarianceDeviationHdpScale_le_max_hdp_radius {m n : ℕ} (hm : 0 < m)
    {K t : ℝ} (hK : 0 < K) (ht : 0 ≤ t)
    (hK_lower : 1 / (4 * exp 1 ^ 2) ≤ K ^ 2) :
    2 * sampleCovarianceDeviationHdpScale m n K t ≤
      max ((twoSidedSubgaussianMatricesConstant * K ^ 2 * (√(n : ℝ) + t)) / √(m : ℝ))
        (((twoSidedSubgaussianMatricesConstant * K ^ 2 * (√(n : ℝ) + t)) /
          √(m : ℝ)) ^ 2) := by
  let C0 : ℝ := hdpHansonWrightConstant
  let D : ℝ := sampleCovarianceDeviationScaleConstant
  let C : ℝ := twoSidedSubgaussianMatricesConstant
  let k0 : ℝ := 1 / (4 * exp 1 ^ 2)
  let S : ℝ := √(n : ℝ) + t
  let Y : ℝ := S / √(m : ℝ)
  let X : ℝ := K ^ 2 * Y
  let q : ℝ := 4 * C0 * (17 * (n : ℝ) + t ^ 2)
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have hsqrt_m_pos : 0 < √(m : ℝ) := sqrt_pos.mpr hmR_pos
  have hsqrt_m_ne : √(m : ℝ) ≠ 0 := hsqrt_m_pos.ne'
  have hsqrt_m_sq : √(m : ℝ) ^ 2 = (m : ℝ) := sq_sqrt hmR_pos.le
  have hC0_pos : 0 < C0 := by
    dsimp [C0]
    exact hdpHansonWrightConstant_pos
  have hC0_nonneg : 0 ≤ C0 := hC0_pos.le
  have hD_ge_one : 1 ≤ D := by
    dsimp [D, sampleCovarianceDeviationScaleConstant, C0]
    dsimp [hdpHansonWrightConstant]
    nlinarith [one_le_exp (by norm_num : (0 : ℝ) ≤ 1), sq_nonneg (exp 1)]
  have hD_nonneg : 0 ≤ D := le_trans (by norm_num : (0 : ℝ) ≤ 1) hD_ge_one
  have hexp_ge_one : 1 ≤ exp 1 := one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
  have hexp_sq_ge_one : 1 ≤ exp 1 ^ 2 := by
    nlinarith [hexp_ge_one, exp_pos 1]
  have hexp_sq_pos : 0 < exp 1 ^ 2 := by positivity
  have hD_ge_exp_sq : exp 1 ^ 2 ≤ D := by
    dsimp [D, sampleCovarianceDeviationScaleConstant, C0, hdpHansonWrightConstant]
    nlinarith [sq_nonneg (exp 1)]
  have hexp_ne : exp 1 ≠ 0 := (exp_pos 1).ne'
  have hC_nonneg : 0 ≤ C := by
    dsimp [C, twoSidedSubgaussianMatricesConstant]
    positivity
  have hk0_pos : 0 < k0 := by
    dsimp [k0]
    positivity
  have hS_nonneg : 0 ≤ S := by
    dsimp [S]
    positivity
  have hY_nonneg : 0 ≤ Y := by
    dsimp [Y]
    positivity
  have hX_nonneg : 0 ≤ X := by
    dsimp [X]
    positivity
  have hY_le : Y ≤ X / k0 := by
    have hone_le : 1 ≤ K ^ 2 / k0 := by
      rw [le_div_iff₀ hk0_pos]
      simpa [k0] using hK_lower
    calc
      Y ≤ (K ^ 2 / k0) * Y := le_mul_of_one_le_left hY_nonneg hone_le
      _ = X / k0 := by
        dsimp [X]
        field_simp [hk0_pos.ne']
  have hC_D : 4 * D ≤ C := by
    dsimp [C, twoSidedSubgaussianMatricesConstant]
    linarith
  have hCk0 : 1 ≤ C * k0 := by
    have h_eq : C * k0 = D / exp 1 ^ 2 := by
      dsimp [C, k0, twoSidedSubgaussianMatricesConstant]
      field_simp [hexp_ne]
      ring
    rw [h_eq]
    rw [le_div_iff₀ hexp_sq_pos]
    simpa using hD_ge_exp_sq
  have hC_sq : 4 * D ≤ C ^ 2 * k0 := by
    have h_eq : C ^ 2 * k0 = 4 * D ^ 2 / exp 1 ^ 2 := by
      dsimp [C, k0, twoSidedSubgaussianMatricesConstant]
      field_simp [hexp_ne]
      ring
    rw [h_eq]
    rw [le_div_iff₀ hexp_sq_pos]
    nlinarith [hD_ge_exp_sq, hD_nonneg]
  have hS_sq_base : (n : ℝ) + t ^ 2 ≤ S ^ 2 := by
    have hsqrt_n_sq : √(n : ℝ) ^ 2 = (n : ℝ) :=
      sq_sqrt (Nat.cast_nonneg n)
    have hnt : 0 ≤ 2 * √(n : ℝ) * t := by positivity
    dsimp [S]
    nlinarith [hsqrt_n_sq, hnt]
  have hsum_le : 17 * (n : ℝ) + t ^ 2 ≤ 17 * S ^ 2 := by
    have ht_sq_nonneg : 0 ≤ t ^ 2 := sq_nonneg t
    calc
      17 * (n : ℝ) + t ^ 2 ≤ 17 * ((n : ℝ) + t ^ 2) := by
        nlinarith
      _ ≤ 17 * S ^ 2 := mul_le_mul_of_nonneg_left hS_sq_base (by norm_num)
  have hq_le : q ≤ D * S ^ 2 := by
    have h68_le_D : 68 * C0 ≤ D := by
      dsimp [D, sampleCovarianceDeviationScaleConstant, C0]
      linarith
    calc
      q = 4 * C0 * (17 * (n : ℝ) + t ^ 2) := rfl
      _ ≤ 4 * C0 * (17 * S ^ 2) :=
        mul_le_mul_of_nonneg_left hsum_le (by positivity)
      _ = 68 * C0 * S ^ 2 := by ring
      _ ≤ D * S ^ 2 := mul_le_mul_of_nonneg_right h68_le_D (sq_nonneg S)
  have hq_nonneg : 0 ≤ q := by
    dsimp [q, C0]
    positivity
  have hq_div_le : q / (m : ℝ) ≤ D * Y ^ 2 := by
    calc
      q / (m : ℝ) ≤ D * S ^ 2 / (m : ℝ) :=
        div_le_div_of_nonneg_right hq_le hmR_pos.le
      _ = D * Y ^ 2 := by
        dsimp [Y]
        field_simp [hsqrt_m_ne]
        rw [hsqrt_m_sq]
  have hsqrt_le : √(q / (m : ℝ)) ≤ D * Y := by
    have hDY_nonneg : 0 ≤ D * Y := mul_nonneg hD_nonneg hY_nonneg
    rw [sqrt_le_left hDY_nonneg]
    calc
      q / (m : ℝ) ≤ D * Y ^ 2 := hq_div_le
      _ ≤ (D * Y) ^ 2 := by
        have hD_le_Dsq : D ≤ D ^ 2 := by
          have hmul := mul_le_mul hD_ge_one (le_refl D) hD_nonneg hD_nonneg
          simpa [pow_two] using hmul
        calc
          D * Y ^ 2 ≤ D ^ 2 * Y ^ 2 :=
            mul_le_mul_of_nonneg_right hD_le_Dsq (sq_nonneg Y)
          _ = (D * Y) ^ 2 := by ring
  have hscale_pre :
      2 * sampleCovarianceDeviationHdpScale m n K t ≤ 2 * D * X + 2 * D * Y * X := by
    have hinner :
        sampleCovarianceDeviationHdpScale m n K t ≤ D * X + D * Y * X := by
      calc
        sampleCovarianceDeviationHdpScale m n K t
            = K ^ 2 * (√(q / (m : ℝ)) + q / (m : ℝ)) := by
              dsimp [sampleCovarianceDeviationHdpScale, q, C0]
        _ ≤ K ^ 2 * (D * Y + D * Y ^ 2) :=
            mul_le_mul_of_nonneg_left (add_le_add hsqrt_le hq_div_le) (sq_nonneg K)
        _ = D * X + D * Y * X := by
            dsimp [X]
            ring
    calc
      2 * sampleCovarianceDeviationHdpScale m n K t
          ≤ 2 * (D * X + D * Y * X) :=
            mul_le_mul_of_nonneg_left hinner (by norm_num)
      _ = 2 * D * X + 2 * D * Y * X := by ring
  have hmax :=
    two_mul_scale_le_max_of_scale_bounds
      (D := D) (C := C) (k0 := k0) (X := X) (Y := Y)
      hD_nonneg hC_nonneg hk0_pos hX_nonneg hY_nonneg hY_le hC_D hCk0 hC_sq
  calc
    2 * sampleCovarianceDeviationHdpScale m n K t
        ≤ 2 * D * X + 2 * D * Y * X := hscale_pre
    _ ≤ max (C * X) ((C * X) ^ 2) := hmax
    _ = max ((twoSidedSubgaussianMatricesConstant * K ^ 2 * (√(n : ℝ) + t)) /
          √(m : ℝ))
        (((twoSidedSubgaussianMatricesConstant * K ^ 2 * (√(n : ℝ) + t)) /
          √(m : ℝ)) ^ 2) := by
        dsimp [C, X, Y, S]
        ring_nf

lemma expected_max_of_cgf_bound_at {ι : Type*} {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : ι → Ω → ℝ} {s : Finset ι} (hs : s.Nonempty) (hs_card : 2 ≤ s.card)
    {l a : ℝ} (hl : 0 < l)
    (hX_meas : ∀ i ∈ s, Measurable (X i))
    (hX_int_basic : ∀ i ∈ s, Integrable (X i) μ)
    (hX_cgf : ∀ i ∈ s, cgf (X i) μ l ≤ a)
    (hX_int_exp : ∀ i ∈ s, Integrable (fun ω => exp (l * X i ω)) μ) :
    ∫ ω, s.sup' hs (fun i => X i ω) ∂μ ≤ (1 / l) * (log s.card + a) := by
  set d := s.card with hd_def
  have hd_pos : 0 < d := Nat.lt_of_lt_of_le (by norm_num : 0 < 2) hs_card
  have hsup_int : Integrable (fun ω => s.sup' hs (fun i => X i ω)) μ := by
    refine Integrable.mono (integrable_finsetSum s (fun i hi => (hX_int_basic i hi).abs)) ?_ ?_
    · convert (Finset.measurable_sup' hs (fun i hi => hX_meas i hi)).aestronglyMeasurable using 1
      funext ω
      simp only [Finset.sup'_apply]
    · refine ae_of_all μ (fun ω => ?_)
      simp only [Real.norm_eq_abs]
      have h_sum_nonneg : 0 ≤ ∑ i ∈ s, |X i ω| :=
        Finset.sum_nonneg fun i _ => abs_nonneg _
      rw [abs_of_nonneg h_sum_nonneg]
      convert abs_sup'_le_sum hs (fun i => X i ω) using 2
  have hMGF_bound : ∀ i ∈ s, ∫ ω, exp (l * X i ω) ∂μ ≤ exp a := by
    intro i hi
    have hcgf := hX_cgf i hi
    unfold ProbabilityTheory.cgf at hcgf
    have h_exp_bound := exp_le_exp.mpr hcgf
    unfold ProbabilityTheory.mgf at h_exp_bound
    simp only [exp_log (integral_exp_pos (hX_int_exp i hi))] at h_exp_bound
    exact h_exp_bound
  have hsum_int : Integrable (fun ω => ∑ i ∈ s, exp (l * X i ω)) μ :=
    integrable_finsetSum s fun i hi => hX_int_exp i hi
  have hsup_exp_bound :
      ∫ ω, exp (l * s.sup' hs (fun i => X i ω)) ∂μ ≤ d * exp a := by
    calc
      ∫ ω, exp (l * s.sup' hs (fun i => X i ω)) ∂μ
          ≤ ∫ ω, ∑ i ∈ s, exp (l * X i ω) ∂μ := by
            apply integral_mono_of_nonneg
            · exact ae_of_all μ fun _ => (exp_pos _).le
            · exact hsum_int
            · exact ae_of_all μ fun ω => exp_mul_sup'_le_sum hs (fun i => X i ω) l
      _ = ∑ i ∈ s, ∫ ω, exp (l * X i ω) ∂μ :=
            integral_finsetSum s fun i hi => hX_int_exp i hi
      _ ≤ ∑ i ∈ s, exp a := Finset.sum_le_sum fun i hi => hMGF_bound i hi
      _ = d * exp a := by simp [Finset.sum_const, hd_def]
  have hsup_exp_int : Integrable (fun ω => exp (l * s.sup' hs (fun i => X i ω))) μ := by
    apply Integrable.mono' hsum_int
    · apply Measurable.aestronglyMeasurable
      apply Measurable.exp
      apply Measurable.const_mul
      convert Finset.measurable_sup' hs (fun i hi => hX_meas i hi) using 1
      funext ω
      simp only [Finset.sup'_apply]
    · refine ae_of_all μ (fun ω => ?_)
      simp only [Real.norm_eq_abs]
      rw [abs_of_pos (exp_pos _)]
      exact exp_mul_sup'_le_sum hs (fun i => X i ω) l
  have hsup_exp_pos : 0 < ∫ ω, exp (l * s.sup' hs (fun i => X i ω)) ∂μ :=
    integral_exp_pos hsup_exp_int
  have hlog_bound :
      log (∫ ω, exp (l * s.sup' hs (fun i => X i ω)) ∂μ) ≤ log d + a := by
    calc
      log (∫ ω, exp (l * s.sup' hs (fun i => X i ω)) ∂μ)
          ≤ log (d * exp a) := log_le_log hsup_exp_pos hsup_exp_bound
      _ = log d + log (exp a) := by
            rw [log_mul (by exact_mod_cast hd_pos.ne') (exp_pos a).ne']
      _ = log d + a := by rw [log_exp]
  calc
    ∫ ω, s.sup' hs (fun i => X i ω) ∂μ
        ≤ (1 / l) * log (∫ ω, exp (l * s.sup' hs (fun i => X i ω)) ∂μ) :=
          mean_le_log_mgf hsup_int hl hsup_exp_int
    _ ≤ (1 / l) * (log d + a) := by
          exact mul_le_mul_of_nonneg_left hlog_bound (le_of_lt (one_div_pos.mpr hl))
    _ = (1 / l) * (log s.card + a) := by rw [hd_def]

omit [MeasurableSpace Ω] in
lemma sampleCovarianceDeviation_tail_prefactor_le {m n : ℕ} (hm : 0 < m)
    {K t : ℝ} (hK : 0 < K) :
    2 * 17 ^ n *
        exp (-(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * sampleCovarianceDeviationHdpScale m n K t) ^ 2 /
              (K ^ 4 * (m : ℝ)))
            (((m : ℝ) * sampleCovarianceDeviationHdpScale m n K t) / K ^ 2)) ≤
      2 * exp (-(t ^ 2)) := by
  let q : ℝ := 4 * hdpHansonWrightConstant * (17 * (n : ℝ) + t ^ 2)
  let r : ℝ := q / (m : ℝ)
  let s : ℝ := √r
  let u : ℝ := sampleCovarianceDeviationHdpScale m n K t
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have hC_pos : 0 < hdpHansonWrightConstant := hdpHansonWrightConstant_pos
  have hq_nonneg : 0 ≤ q := by
    dsimp [q]
    have hsum : 0 ≤ 17 * (n : ℝ) + t ^ 2 := by positivity
    positivity
  have hr_nonneg : 0 ≤ r := div_nonneg hq_nonneg hmR_pos.le
  have hs_nonneg : 0 ≤ s := by
    dsimp [s]
    exact sqrt_nonneg r
  have hs_sq : s ^ 2 = r := by
    dsimp [s]
    exact sq_sqrt hr_nonneg
  have hu_eq : u = K ^ 2 * (s + r) := by
    dsimp [u, sampleCovarianceDeviationHdpScale, q, r, s]
  have hq_eq_mr : q = (m : ℝ) * r := by
    dsimp [r]
    field_simp [hmR_pos.ne']
  have hfirst :
      q ≤ ((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ)) := by
    have hv_nonneg : 0 ≤ s + r := add_nonneg hs_nonneg hr_nonneg
    have hs_le : s ≤ s + r := by linarith
    have hs_sq_le : s ^ 2 ≤ (s + r) ^ 2 :=
      (sq_le_sq₀ hs_nonneg hv_nonneg).mpr hs_le
    calc
      q = (m : ℝ) * r := hq_eq_mr
      _ = (m : ℝ) * s ^ 2 := by rw [hs_sq]
      _ ≤ (m : ℝ) * (s + r) ^ 2 := mul_le_mul_of_nonneg_left hs_sq_le hmR_pos.le
      _ = ((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ)) := by
        rw [hu_eq]
        field_simp [hK.ne', hmR_pos.ne']
  have hsecond :
      q ≤ ((m : ℝ) * u) / K ^ 2 := by
    calc
      q = (m : ℝ) * r := hq_eq_mr
      _ ≤ (m : ℝ) * (s + r) := by
        have : r ≤ s + r := by linarith
        exact mul_le_mul_of_nonneg_left this hmR_pos.le
      _ = ((m : ℝ) * u) / K ^ 2 := by
        rw [hu_eq]
        field_simp [hK.ne']
  have hmin : q ≤
      min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2) :=
    le_min hfirst hsecond
  have hcoef_nonpos : -(1 / (4 * hdpHansonWrightConstant)) ≤ 0 := by
    have hpos : 0 ≤ 1 / (4 * hdpHansonWrightConstant) := by positivity
    linarith
  have harg :
      -(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2) ≤
        -(17 * (n : ℝ) + t ^ 2) := by
    calc
      -(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)
          ≤ -(1 / (4 * hdpHansonWrightConstant)) * q :=
            mul_le_mul_of_nonpos_left hmin hcoef_nonpos
      _ = -(17 * (n : ℝ) + t ^ 2) := by
        dsimp [q]
        field_simp [hC_pos.ne']
  have htail_exp :
      exp (-(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) ≤
        exp (-(17 * (n : ℝ) + t ^ 2)) :=
    exp_le_exp.mpr harg
  have hcard : (17 : ℝ) ^ n ≤ exp (17 * (n : ℝ)) :=
    seventeen_pow_le_exp_seventeen_mul n
  calc
    2 * 17 ^ n *
        exp (-(1 / (4 * hdpHansonWrightConstant)) *
          min (((m : ℝ) * sampleCovarianceDeviationHdpScale m n K t) ^ 2 /
              (K ^ 4 * (m : ℝ)))
            (((m : ℝ) * sampleCovarianceDeviationHdpScale m n K t) / K ^ 2))
        = 2 * (17 : ℝ) ^ n *
            exp (-(1 / (4 * hdpHansonWrightConstant)) *
              min (((m : ℝ) * u) ^ 2 / (K ^ 4 * (m : ℝ))) (((m : ℝ) * u) / K ^ 2)) := by
          simp [u]
    _ ≤ 2 * exp (17 * (n : ℝ)) * exp (-(17 * (n : ℝ) + t ^ 2)) := by
          have hleft : 2 * (17 : ℝ) ^ n ≤ 2 * exp (17 * (n : ℝ)) :=
            mul_le_mul_of_nonneg_left hcard (by norm_num : (0 : ℝ) ≤ 2)
          exact mul_le_mul hleft htail_exp (le_of_lt (exp_pos _))
            (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2)
              (le_of_lt (exp_pos _)))
    _ = 2 * exp (-(t ^ 2)) := by
          calc
            2 * exp (17 * (n : ℝ)) * exp (-(17 * (n : ℝ) + t ^ 2))
                = 2 * (exp (17 * (n : ℝ)) * exp (-(17 * (n : ℝ) + t ^ 2))) := by
                    ring
            _ = 2 * exp (17 * (n : ℝ) + (-(17 * (n : ℝ) + t ^ 2))) := by
                    rw [← exp_add]
            _ = 2 * exp (-(t ^ 2)) := by
                    congr 1
                    ring_nf

lemma sampleCovarianceDeviationOperatorNorm_hdp_bound_of_pos {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ) (hK : 0 < K) :
    ∀ t : ℝ, 0 ≤ t →
      (μ {ω |
        sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ≤
          2 * sampleCovarianceDeviationHdpScale m n K t}).toReal ≥
        1 - 2 * exp (-(t ^ 2)) := by
  intro t _ht
  let u : ℝ := sampleCovarianceDeviationHdpScale m n K t
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have hnR_pos : 0 < (n : ℝ) := by exact_mod_cast hn
  have hq_pos :
      0 < 4 * hdpHansonWrightConstant * (17 * (n : ℝ) + t ^ 2) := by
    have h17n_pos : 0 < 17 * (n : ℝ) := mul_pos (by norm_num) hnR_pos
    have hsum_pos : 0 < 17 * (n : ℝ) + t ^ 2 := by nlinarith [sq_nonneg t]
    exact mul_pos (mul_pos (by norm_num) hdpHansonWrightConstant_pos) hsum_pos
  have hdiv_pos :
      0 < (4 * hdpHansonWrightConstant * (17 * (n : ℝ) + t ^ 2)) / (m : ℝ) :=
    div_pos hq_pos hmR_pos
  have hu : 0 < u := by
    dsimp [u, sampleCovarianceDeviationHdpScale]
    exact mul_pos (sq_pos_of_pos hK) (add_pos (sqrt_pos.mpr hdiv_pos) hdiv_pos)
  have htail :=
    sampleCovarianceDeviationOperatorNorm_tail_le_of_row_subgaussian
      (m := m) (n := n) hm hn (A := A) (μ := μ) hA
      (K := K) (u := u) hK_def hK hu
  have htail_bound :
      (μ {ω | 2 * u < sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)}).toReal ≤
        2 * exp (-(t ^ 2)) := by
    exact htail.trans
      (by
        simpa [u] using
          sampleCovarianceDeviation_tail_prefactor_le (m := m) (n := n) hm (K := K) (t := t) hK)
  have hprob :=
    probability_le_of_tail_real_le
      (μ := μ) (X := fun ω => sampleCovarianceDeviationOperatorNorm (randomMatrix A ω))
      (B := 2 * u) (δ := 2 * exp (-(t ^ 2))) htail_bound
  simpa [u] using hprob

lemma singular_value_sandwich_probability_of_sampleCovarianceDeviationHdpScale
    {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K B t : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ)
    (hK : 0 < K) (hB : 0 ≤ B) (ht : 0 ≤ t)
    (hscale :
      2 * sampleCovarianceDeviationHdpScale m n K t ≤
        max (B / √(m : ℝ)) ((B / √(m : ℝ)) ^ 2)) :
    (μ {ω |
      √(m : ℝ) - B ≤ matrixSmallestSingularValue (randomMatrix A ω) ∧
        matrixSmallestSingularValue (randomMatrix A ω) ≤
          matrixLargestSingularValue (randomMatrix A ω) ∧
        matrixLargestSingularValue (randomMatrix A ω) ≤ √(m : ℝ) + B}).toReal ≥
      1 - 2 * exp (-(t ^ 2)) := by
  let covEvent : Set Ω :=
    {ω | sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ≤
      2 * sampleCovarianceDeviationHdpScale m n K t}
  let svEvent : Set Ω :=
    {ω |
      √(m : ℝ) - B ≤ matrixSmallestSingularValue (randomMatrix A ω) ∧
        matrixSmallestSingularValue (randomMatrix A ω) ≤
          matrixLargestSingularValue (randomMatrix A ω) ∧
        matrixLargestSingularValue (randomMatrix A ω) ≤ √(m : ℝ) + B}
  have hcov_prob :
      (μ covEvent).toReal ≥ 1 - 2 * exp (-(t ^ 2)) := by
    simpa [covEvent] using
      sampleCovarianceDeviationOperatorNorm_hdp_bound_of_pos
        (m := m) (n := n) hm hn (A := A) (μ := μ) hA hK_def hK t ht
  have hsubset : covEvent ⊆ svEvent := by
    intro ω hω
    have hdev :
        sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ≤
          max (B / √(m : ℝ)) ((B / √(m : ℝ)) ^ 2) :=
      hω.trans hscale
    exact singular_value_sandwich_of_sampleCovarianceDeviationOperatorNorm_le_max
      (m := m) (n := n) hm hn (randomMatrix A ω) hB hdev
  have hmono : (μ covEvent).toReal ≤ (μ svEvent).toReal :=
    ENNReal.toReal_mono (measure_ne_top μ svEvent) (measure_mono hsubset)
  have hsv : (μ svEvent).toReal ≥ 1 - 2 * exp (-(t ^ 2)) := by
    linarith
  simpa [svEvent] using hsv

omit [MeasurableSpace Ω] in
lemma cast_add_add_sq_le_sq_sqrt_add_sqrt_add {m n : ℕ} {t : ℝ} (ht : 0 ≤ t) :
    ((m + n : ℕ) : ℝ) + t ^ 2 ≤ (√(m : ℝ) + √(n : ℝ) + t) ^ 2 := by
  have hsqrt_m_sq : √(m : ℝ) ^ 2 = (m : ℝ) :=
    Real.sq_sqrt (Nat.cast_nonneg m)
  have hsqrt_n_sq : √(n : ℝ) ^ 2 = (n : ℝ) :=
    Real.sq_sqrt (Nat.cast_nonneg n)
  have hmn : 0 ≤ 2 * √(m : ℝ) * √(n : ℝ) := by positivity
  have hmt : 0 ≤ 2 * √(m : ℝ) * t := by positivity
  have hnt : 0 ≤ 2 * √(n : ℝ) * t := by positivity
  calc
    ((m + n : ℕ) : ℝ) + t ^ 2 = (m : ℝ) + (n : ℝ) + t ^ 2 := by
      norm_num [Nat.cast_add]
    _ ≤ (m : ℝ) + (n : ℝ) + t ^ 2 +
          (2 * √(m : ℝ) * √(n : ℝ) + 2 * √(m : ℝ) * t + 2 * √(n : ℝ) * t) := by
        nlinarith
    _ = (√(m : ℝ) + √(n : ℝ) + t) ^ 2 := by
        nlinarith [hsqrt_m_sq, hsqrt_n_sq]

omit [MeasurableSpace Ω] in
lemma rectangular_net_tail_prefactor_le {m n : ℕ} {K t : ℝ}
    (hK : 0 < K) (ht : 0 < t) :
    2 * 17 ^ (m + n) *
        exp (-(12 * K * (√(m : ℝ) + √(n : ℝ) + t)) ^ 2 / (8 * K ^ 2)) ≤
      2 * exp (-(t ^ 2)) := by
  let S : ℝ := √(m : ℝ) + √(n : ℝ) + t
  have hKsq : K ^ 2 ≠ 0 := pow_ne_zero 2 hK.ne'
  have hexponent :
      -(12 * K * S) ^ 2 / (8 * K ^ 2) = -(18 * S ^ 2) := by
    field_simp [hKsq]
    ring
  have hSsq : ((m + n : ℕ) : ℝ) + t ^ 2 ≤ S ^ 2 := by
    simpa [S] using
      cast_add_add_sq_le_sq_sqrt_add_sqrt_add (m := m) (n := n) (t := t) ht.le
  have harg : 17 * ((m + n : ℕ) : ℝ) - 18 * S ^ 2 ≤ -(t ^ 2) := by
    have hd : 0 ≤ ((m + n : ℕ) : ℝ) := Nat.cast_nonneg _
    have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
    nlinarith
  have hnet :
      (17 : ℝ) ^ (m + n) * exp (-(18 * S ^ 2)) ≤ exp (-(t ^ 2)) := by
    calc
      (17 : ℝ) ^ (m + n) * exp (-(18 * S ^ 2))
          ≤ exp (17 * ((m + n : ℕ) : ℝ)) * exp (-(18 * S ^ 2)) := by
            exact mul_le_mul_of_nonneg_right
              (seventeen_pow_le_exp_seventeen_mul (m + n)) (le_of_lt (exp_pos _))
      _ = exp (17 * ((m + n : ℕ) : ℝ) - 18 * S ^ 2) := by
            rw [← exp_add]
            ring_nf
      _ ≤ exp (-(t ^ 2)) := exp_le_exp.mpr harg
  calc
    2 * 17 ^ (m + n) * exp (-(12 * K * S) ^ 2 / (8 * K ^ 2))
        = 2 * ((17 : ℝ) ^ (m + n) * exp (-(18 * S ^ 2))) := by
          rw [hexponent]
          ring
    _ ≤ 2 * exp (-(t ^ 2)) :=
          mul_le_mul_of_nonneg_left hnet (by norm_num : (0 : ℝ) ≤ 2)

omit [MeasurableSpace Ω] in
lemma cast_add_sq_le_sq_sqrt_add {n : ℕ} {t : ℝ} (ht : 0 ≤ t) :
    (n : ℝ) + t ^ 2 ≤ (√(n : ℝ) + t) ^ 2 := by
  have hsqrt_n_sq : √(n : ℝ) ^ 2 = (n : ℝ) :=
    Real.sq_sqrt (Nat.cast_nonneg n)
  have hnt : 0 ≤ 2 * √(n : ℝ) * t := by positivity
  calc
    (n : ℝ) + t ^ 2 ≤ (n : ℝ) + t ^ 2 + 2 * √(n : ℝ) * t := by
      nlinarith
    _ = (√(n : ℝ) + t) ^ 2 := by
      nlinarith [hsqrt_n_sq]

omit [MeasurableSpace Ω] in
lemma symmetric_net_tail_prefactor_le {n : ℕ} {K t : ℝ}
    (hK : 0 < K) (ht : 0 < t) :
    2 * 17 ^ n *
        exp (-(32 * K * (√(n : ℝ) + t)) ^ 2 / (32 * K ^ 2)) ≤
      4 * exp (-(t ^ 2)) := by
  let S : ℝ := √(n : ℝ) + t
  have hKsq : K ^ 2 ≠ 0 := pow_ne_zero 2 hK.ne'
  have hexponent :
      -(32 * K * S) ^ 2 / (32 * K ^ 2) = -(32 * S ^ 2) := by
    field_simp [hKsq]
  have hSsq : (n : ℝ) + t ^ 2 ≤ S ^ 2 := by
    simpa [S] using cast_add_sq_le_sq_sqrt_add (n := n) (t := t) ht.le
  have harg : 17 * (n : ℝ) - 32 * S ^ 2 ≤ -(t ^ 2) := by
    have hn_nonneg : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
    have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
    nlinarith
  have hnet :
      (17 : ℝ) ^ n * exp (-(32 * S ^ 2)) ≤ exp (-(t ^ 2)) := by
    calc
      (17 : ℝ) ^ n * exp (-(32 * S ^ 2))
          ≤ exp (17 * (n : ℝ)) * exp (-(32 * S ^ 2)) := by
            exact mul_le_mul_of_nonneg_right
              (seventeen_pow_le_exp_seventeen_mul n) (le_of_lt (exp_pos _))
      _ = exp (17 * (n : ℝ) - 32 * S ^ 2) := by
            rw [← exp_add]
            ring_nf
      _ ≤ exp (-(t ^ 2)) := exp_le_exp.mpr harg
  calc
    2 * 17 ^ n * exp (-(32 * K * (√(n : ℝ) + t)) ^ 2 / (32 * K ^ 2))
        = 2 * ((17 : ℝ) ^ n * exp (-(32 * S ^ 2))) := by
          rw [hexponent]
          ring
    _ ≤ 2 * exp (-(t ^ 2)) :=
          mul_le_mul_of_nonneg_left hnet (by norm_num : (0 : ℝ) ≤ 2)
    _ ≤ 4 * exp (-(t ^ 2)) := by
          exact mul_le_mul_of_nonneg_right (by norm_num : (2 : ℝ) ≤ 4)
            (le_of_lt (exp_pos _))

/-! ## Exact HDP Section 4.4 propositions -/

/--
HDP Theorem 4.4.3, "Norm of matrices with subgaussian entries".

For an `m × n` random matrix with independent, mean-zero, sub-Gaussian entries and
`K = max_{i,j} ‖Aᵢⱼ‖_{ψ₂}`, there is a positive absolute constant `C` such that for every
`t > 0`,
`‖A‖ ≤ C K (√m + √n + t)` with probability at least `1 - 2 exp (-t²)`.
-/
def norm_subgaussian_matrices_hdp {m n : ℕ} (A : Fin m → Fin n → Ω → ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  HasIndependentMeanZeroSubGaussianEntries A μ →
    ∃ C : ℝ, 0 < C ∧
      ∀ K : ℝ, K = maxMatrixEntrySubGaussianPsi2Norm A μ → 0 < K →
        ∀ t : ℝ, 0 < t →
          (μ {ω |
            matrixOperatorNorm (randomMatrix A ω) ≤
              C * K * (√(m : ℝ) + √(n : ℝ) + t)}).toReal ≥
            1 - 2 * exp (-(t ^ 2))

/-- HDP Theorem 4.4.3 for positive dimensions, proved from the rectangular net tail bound. -/
theorem norm_subgaussian_matrices_hdp_of_pos {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] :
    norm_subgaussian_matrices_hdp A μ := by
  intro hA
  refine ⟨24, by norm_num, ?_⟩
  intro K hK_def hK t ht
  let S : ℝ := √(m : ℝ) + √(n : ℝ) + t
  have hSpos : 0 < S := by
    have hm_sqrt : 0 ≤ √(m : ℝ) := Real.sqrt_nonneg _
    have hn_sqrt : 0 ≤ √(n : ℝ) := Real.sqrt_nonneg _
    nlinarith
  have hu : 0 < 12 * K * S :=
    mul_pos (mul_pos (by norm_num : (0 : ℝ) < 12) hK) hSpos
  have htail :=
    matrixOperatorNorm_tail_le_of_max_psi2 (m := m) (n := n) hm hn
      (A := A) (μ := μ) hA (K := K) (u := 12 * K * S) hK_def hK hu
  have htail_bound :
      (μ {ω | 24 * K * S < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
        2 * exp (-(t ^ 2)) := by
    have hpref :=
      rectangular_net_tail_prefactor_le (m := m) (n := n) (K := K) (t := t) hK ht
    have htail' := htail.trans hpref
    have htail'' :
        (μ {ω | K * (2 * (12 * S)) < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
          2 * exp (-(t ^ 2)) := by
      simpa [S, mul_assoc, mul_left_comm, mul_comm] using htail'
    have hthreshold : K * (2 * (12 * S)) = 24 * K * S := by ring
    simpa [hthreshold] using htail''
  have hprob :=
    probability_le_of_tail_real_le
      (μ := μ) (X := fun ω => matrixOperatorNorm (randomMatrix A ω))
      (B := 24 * K * S) (δ := 2 * exp (-(t ^ 2))) htail_bound
  simpa [S, mul_assoc, mul_left_comm, mul_comm] using hprob

/--
HDP Remark 4.4.4, "Expectation".

The high-probability bound from Theorem 4.4.3 yields
`E ‖A‖ ≤ C K (√m + √n)`.
-/
def norm_subgaussian_matrices_expectation_hdp {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  HasIndependentMeanZeroSubGaussianEntries A μ →
    ∃ C : ℝ, 0 < C ∧
      ∀ K : ℝ, K = maxMatrixEntrySubGaussianPsi2Norm A μ → 0 < K →
        ∫ ω, matrixOperatorNorm (randomMatrix A ω) ∂μ ≤
          C * K * (√(m : ℝ) + √(n : ℝ))

/-- HDP Remark 4.4.4 for positive dimensions, proved by integrating the finite-net maximum. -/
theorem norm_subgaussian_matrices_expectation_hdp_of_pos {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] :
    norm_subgaussian_matrices_expectation_hdp A μ := by
  intro hA
  refine ⟨28, by norm_num, ?_⟩
  intro K hK_def hK
  obtain ⟨N, hN_domain, hN_codomain⟩ := exists_quarter_centeredMatrixBilinearNet m n
  let signed : Finset (MatrixBilinearPair m n × Bool) := N.signedPairs
  let Z : MatrixBilinearPair m n × Bool → Ω → ℝ :=
    fun q ω => signedBilinearValue (randomMatrix A ω) q
  have hpairs_nonempty : N.pairs.Nonempty := N.pairs_nonempty_of_pos hm hn
  have hsigned_nonempty : signed.Nonempty := by
    change (N.pairs ×ˢ (Finset.univ : Finset Bool)).Nonempty
    exact hpairs_nonempty.product ⟨true, Finset.mem_univ true⟩
  have hsigned_card : 2 ≤ signed.card := by
    simpa [signed] using N.signedPairs_card_ge_two_of_pos hm hn
  have hK2 : 0 < 2 * K := by nlinarith
  have hentry :
      ∀ i j, HasSubgaussianMGF (A i j) ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ :=
    fun i j => entry_hasSubgaussianMGF_two_mul_max hK_def hK hA.finite_psi2 i j
  have hZ_mgf :
      ∀ q ∈ signed, HasSubgaussianMGF (Z q) ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ := by
    intro q hq
    have hq_signed : q ∈ N.pairs ×ˢ (Finset.univ : Finset Bool) := by
      simpa [signed, CenteredMatrixBilinearNet.signedPairs] using hq
    have hq_pair : q.1 ∈ N.pairs := (Finset.mem_product.mp hq_signed).1
    have hq_pair' : q.1 ∈ N.domainNet ×ˢ N.codomainNet := by
      simpa [CenteredMatrixBilinearNet.pairs] using hq_pair
    obtain ⟨hx_mem, hy_mem⟩ := Finset.mem_product.mp hq_pair'
    have hx_norm : ‖q.1.1‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hx_mem)
    have hy_norm : ‖q.1.2‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.codomain_subset_unitBall hy_mem)
    have hbase :=
      inner_randomMatrix_hasSubgaussianMGF_of_norm_le_one
        (A := A) (μ := μ) (K := 2 * K) hA.independent hentry hx_norm hy_norm
    by_cases hsign : q.2
    · simpa [Z, signedBilinearValue, hsign] using hbase
    · convert hbase.neg using 1
      funext ω
      simp [Z, signedBilinearValue, hsign]
  have hZ_meas : ∀ q ∈ signed, Measurable (Z q) := by
    intro q hq
    have hbase := inner_randomMatrix_measurable hA.measurable q.1.1 q.1.2
    by_cases hsign : q.2
    · simpa [Z, signedBilinearValue, hsign] using hbase
    · simpa [Z, signedBilinearValue, hsign] using hbase.neg
  have hZ_int : ∀ q ∈ signed, Integrable (Z q) μ := fun q hq => (hZ_mgf q hq).integrable
  have hZ_exp : ∀ q ∈ signed, ∀ t, Integrable (fun ω => exp (t * Z q ω)) μ :=
    fun q hq t => (hZ_mgf q hq).integrable_exp_mul t
  have hZ_cgf :
      ∀ q ∈ signed, ∀ t, cgf (Z q) μ t ≤ t ^ 2 * (2 * K) ^ 2 / 2 := by
    intro q hq t
    calc
      cgf (Z q) μ t
          ≤ ((⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ : ℝ≥0) : ℝ) * t ^ 2 / 2 :=
            (hZ_mgf q hq).cgf_le t
      _ = t ^ 2 * (2 * K) ^ 2 / 2 := by
            ring
  have hmax :=
    expected_max_subGaussian (μ := μ) (X := Z) (σ := 2 * K) hK2
      (s := signed) hsigned_nonempty hsigned_card hZ_meas hZ_int hZ_cgf hZ_exp
  have hsup_int :
      Integrable (fun ω => signed.sup' hsigned_nonempty (fun q => Z q ω)) μ := by
    refine Integrable.mono
      (integrable_finsetSum signed (fun q hq => (hZ_int q hq).abs)) ?_ ?_
    · convert (Finset.measurable_sup' hsigned_nonempty (fun q hq => hZ_meas q hq)).aestronglyMeasurable using 1
      funext ω
      simp only [Finset.sup'_apply]
    · refine ae_of_all μ (fun ω => ?_)
      simp only [Real.norm_eq_abs]
      have h_sum_nonneg : 0 ≤ ∑ q ∈ signed, |Z q ω| :=
        Finset.sum_nonneg fun q _ => abs_nonneg _
      rw [abs_of_nonneg h_sum_nonneg]
      convert abs_sup'_le_sum hsigned_nonempty (fun q => Z q ω) using 2
  have hnorm_nonneg : 0 ≤ᵐ[μ] fun ω => matrixOperatorNorm (randomMatrix A ω) :=
    ae_of_all μ fun ω => norm_nonneg _
  have hnorm_le_sup :
      (fun ω => matrixOperatorNorm (randomMatrix A ω)) ≤ᵐ[μ]
        fun ω => 2 * signed.sup' hsigned_nonempty (fun q => Z q ω) := by
    refine ae_of_all μ (fun ω => ?_)
    have hdet :=
      matrixOperatorNorm_le_two_mul_signed_sup_of_quarter_centered_bilinear_net
        (A := randomMatrix A ω) (N := N) hpairs_nonempty
    simpa [signed, Z] using hdet
  have hsqrt_log :
      √(2 * log (signed.card : ℝ)) ≤ 7 * (√(m : ℝ) + √(n : ℝ)) := by
    simpa [signed] using
      N.sqrt_two_log_signedPairs_card_le hm hn hN_domain hN_codomain
  calc
    ∫ ω, matrixOperatorNorm (randomMatrix A ω) ∂μ
        ≤ ∫ ω, 2 * signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ :=
          integral_mono_of_nonneg hnorm_nonneg (hsup_int.const_mul 2) hnorm_le_sup
    _ = 2 * ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ := by
          rw [integral_const_mul]
    _ ≤ 2 * ((2 * K) * √(2 * log (signed.card : ℝ))) :=
          mul_le_mul_of_nonneg_left hmax (by norm_num : (0 : ℝ) ≤ 2)
    _ = 4 * K * √(2 * log (signed.card : ℝ)) := by ring
    _ ≤ 4 * K * (7 * (√(m : ℝ) + √(n : ℝ))) := by
          exact mul_le_mul_of_nonneg_left hsqrt_log (by positivity)
    _ = 28 * K * (√(m : ℝ) + √(n : ℝ)) := by ring

/--
HDP Exercise 4.42, "Norm of random matrices: a lower bound".

For independent variance-one sub-Gaussian entries and `K = max_{i,j} ‖Aᵢⱼ‖_{ψ₂}`, there is
a positive absolute constant `c` such that for every `t > 0`,
`‖A‖ ≥ (√m + √n - t) / 2` with probability at least
`1 - 2 exp (-c t² / K⁴)`.
-/
def norm_random_matrices_lower_bound_hdp {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  HasIndependentVarianceOneSubGaussianEntries A μ →
    ∃ c : ℝ, 0 < c ∧
      ∀ K : ℝ, K = maxMatrixEntrySubGaussianPsi2Norm A μ → 0 < K →
        ∀ t : ℝ, 0 < t →
          (μ {ω |
            (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
              matrixOperatorNorm (randomMatrix A ω)}).toReal ≥
            1 - 2 * exp (-(c * t ^ 2 / K ^ 4))

/-- HDP Exercise 4.42 for positive dimensions, proved from one row or one column. -/
theorem norm_random_matrices_lower_bound_hdp_of_pos {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] :
    norm_random_matrices_lower_bound_hdp A μ := by
  intro hA
  refine ⟨randomVectorLowerTailConstant / 4,
    div_pos randomVectorLowerTailConstant_pos (by norm_num), ?_⟩
  intro K hK_def hK t ht
  have hentry :
      ∀ i j, HasSubgaussianMGF (A i j) ⟨(2 * K) ^ 2, sq_nonneg (2 * K)⟩ μ :=
    fun i j => entry_hasSubgaussianMGF_two_mul_max hK_def hK hA.finite_psi2 i j
  by_cases hmn : m ≤ n
  · let i0 : Fin m := ⟨0, hm⟩
    have hrow_indep : iIndepFun (fun j : Fin n => A i0 j) μ := by
      exact hA.independent.precomp (g := fun j : Fin n => (i0, j)) (by
        intro j₁ j₂ h
        simpa using congrArg Prod.snd h)
    have hrow_tail :=
      randomVector_norm_lower_tail_of_subgaussian_mgf_bound (μ := μ) (n := n) hn
        (X := fun j : Fin n => A i0 j) (K := K) hK hrow_indep
        (fun j => hentry i0 j) (fun j => hA.second_moment_one i0 j)
        (t / 2) (by linarith)
    have hsqrt_le : √(m : ℝ) ≤ √(n : ℝ) := by
      exact sqrt_le_sqrt (by exact_mod_cast hmn)
    have hthreshold :
        (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤ √(n : ℝ) - t / 2 := by
      linarith
    have hsubset :
        {ω | √(n : ℝ) - t / 2 ≤
          ‖HansonWright.randomVector (fun j : Fin n => A i0 j) ω‖} ⊆
        {ω |
          (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
            matrixOperatorNorm (randomMatrix A ω)} := by
      intro ω hω
      have hrow_le := norm_matrixRowVector_le_matrixOperatorNorm (randomMatrix A ω) i0
      calc
        (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤ √(n : ℝ) - t / 2 := hthreshold
        _ ≤ ‖HansonWright.randomVector (fun j : Fin n => A i0 j) ω‖ := hω
        _ = ‖matrixRowVector (randomMatrix A ω) i0‖ := by
              rw [matrixRowVector_randomMatrix_eq_randomVector]
        _ ≤ matrixOperatorNorm (randomMatrix A ω) := hrow_le
    have hmono :
        (μ {ω | √(n : ℝ) - t / 2 ≤
          ‖HansonWright.randomVector (fun j : Fin n => A i0 j) ω‖}).toReal ≤
        (μ {ω |
          (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
            matrixOperatorNorm (randomMatrix A ω)}).toReal :=
      ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hsubset)
    calc
      (μ {ω |
          (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
            matrixOperatorNorm (randomMatrix A ω)}).toReal
          ≥ (μ {ω | √(n : ℝ) - t / 2 ≤
              ‖HansonWright.randomVector (fun j : Fin n => A i0 j) ω‖}).toReal := hmono
      _ ≥ 1 - 2 *
            exp (-(randomVectorLowerTailConstant * (t / 2) ^ 2 / K ^ 4)) := hrow_tail
      _ = 1 - 2 * exp (-((randomVectorLowerTailConstant / 4) * t ^ 2 / K ^ 4)) := by
            ring_nf
  · have hnm : n ≤ m := le_of_lt (Nat.lt_of_not_ge hmn)
    let j0 : Fin n := ⟨0, hn⟩
    have hcol_indep : iIndepFun (fun i : Fin m => A i j0) μ := by
      exact hA.independent.precomp (g := fun i : Fin m => (i, j0)) (by
        intro i₁ i₂ h
        simpa using congrArg Prod.fst h)
    have hcol_tail :=
      randomVector_norm_lower_tail_of_subgaussian_mgf_bound (μ := μ) (n := m) hm
        (X := fun i : Fin m => A i j0) (K := K) hK hcol_indep
        (fun i => hentry i j0) (fun i => hA.second_moment_one i j0)
        (t / 2) (by linarith)
    have hsqrt_le : √(n : ℝ) ≤ √(m : ℝ) := by
      exact sqrt_le_sqrt (by exact_mod_cast hnm)
    have hthreshold :
        (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤ √(m : ℝ) - t / 2 := by
      linarith
    have hsubset :
        {ω | √(m : ℝ) - t / 2 ≤
          ‖HansonWright.randomVector (fun i : Fin m => A i j0) ω‖} ⊆
        {ω |
          (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
            matrixOperatorNorm (randomMatrix A ω)} := by
      intro ω hω
      have hcol_le := norm_matrixColumnVector_le_matrixOperatorNorm (randomMatrix A ω) j0
      calc
        (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤ √(m : ℝ) - t / 2 := hthreshold
        _ ≤ ‖HansonWright.randomVector (fun i : Fin m => A i j0) ω‖ := hω
        _ = ‖matrixColumnVector (randomMatrix A ω) j0‖ := by
              rw [matrixColumnVector_randomMatrix_eq_randomVector]
        _ ≤ matrixOperatorNorm (randomMatrix A ω) := hcol_le
    have hmono :
        (μ {ω | √(m : ℝ) - t / 2 ≤
          ‖HansonWright.randomVector (fun i : Fin m => A i j0) ω‖}).toReal ≤
        (μ {ω |
          (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
            matrixOperatorNorm (randomMatrix A ω)}).toReal :=
      ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hsubset)
    calc
      (μ {ω |
          (√(m : ℝ) + √(n : ℝ) - t) / 2 ≤
            matrixOperatorNorm (randomMatrix A ω)}).toReal
          ≥ (μ {ω | √(m : ℝ) - t / 2 ≤
              ‖HansonWright.randomVector (fun i : Fin m => A i j0) ω‖}).toReal := hmono
      _ ≥ 1 - 2 *
            exp (-(randomVectorLowerTailConstant * (t / 2) ^ 2 / K ^ 4)) := hcol_tail
      _ = 1 - 2 * exp (-((randomVectorLowerTailConstant / 4) * t ^ 2 / K ^ 4)) := by
            ring_nf

/--
HDP Corollary 4.4.7, "Norm of symmetric matrices with subgaussian entries".

For an `n × n` symmetric random matrix whose on-and-above-diagonal entries are independent,
mean-zero, and sub-Gaussian, and `K = max_{i,j} ‖Aᵢⱼ‖_{ψ₂}`, there is a positive absolute
constant `C` such that for every `t > 0`,
`‖A‖ ≤ C K (√n + t)` with probability at least `1 - 4 exp (-t²)`.
-/
def norm_symmetric_subgaussian_matrices_hdp {n : ℕ}
    (A : Fin n → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  HasIndependentMeanZeroSubGaussianUpperTriangle A μ →
    ∃ C : ℝ, 0 < C ∧
      ∀ K : ℝ, K = maxMatrixEntrySubGaussianPsi2Norm A μ → 0 < K →
        ∀ t : ℝ, 0 < t →
          (μ {ω |
            matrixOperatorNorm (randomMatrix A ω) ≤
              C * K * (√(n : ℝ) + t)}).toReal ≥
            1 - 4 * exp (-(t ^ 2))

/-- HDP Corollary 4.4.7 for positive dimensions, proved from the symmetric quadratic net. -/
theorem norm_symmetric_subgaussian_matrices_hdp_of_pos {n : ℕ} (hn : 0 < n)
    (A : Fin n → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] :
    norm_symmetric_subgaussian_matrices_hdp A μ := by
  intro hA
  refine ⟨64, by norm_num, ?_⟩
  intro K hK_def hK t ht
  let S : ℝ := √(n : ℝ) + t
  have hSpos : 0 < S := by
    have hn_sqrt : 0 ≤ √(n : ℝ) := Real.sqrt_nonneg _
    nlinarith
  have hu : 0 < 32 * K * S :=
    mul_pos (mul_pos (by norm_num : (0 : ℝ) < 32) hK) hSpos
  have htail :=
    matrixOperatorNorm_tail_le_of_upperTriangle_max_psi2
      (n := n) hn (A := A) (μ := μ) hA
      (fun ω x => inner_randomMatrix_eq_sum_upperTriangle A hA.symmetric ω x)
      (K := K) (u := 32 * K * S) hK_def hK hu
  have htail_bound :
      (μ {ω | 64 * K * S < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
        4 * exp (-(t ^ 2)) := by
    have hpref :=
      symmetric_net_tail_prefactor_le (n := n) (K := K) (t := t) hK ht
    have htail' := htail.trans hpref
    have htail'' :
        (μ {ω | K * (2 * (32 * S)) < matrixOperatorNorm (randomMatrix A ω)}).toReal ≤
          4 * exp (-(t ^ 2)) := by
      simpa [S, mul_assoc, mul_left_comm, mul_comm] using htail'
    have hthreshold : K * (2 * (32 * S)) = 64 * K * S := by ring
    simpa [hthreshold] using htail''
  have hprob :=
    probability_le_of_tail_real_le
      (μ := μ) (X := fun ω => matrixOperatorNorm (randomMatrix A ω))
      (B := 64 * K * S) (δ := 4 * exp (-(t ^ 2))) htail_bound
  simpa [S, mul_assoc, mul_left_comm, mul_comm] using hprob

/-! ## Exact HDP Section 4.6 propositions -/

/--
HDP Theorem 4.6.1, "Two-sided bound on subgaussian matrices".

For an `m × n` random matrix with independent, mean-zero, sub-Gaussian, isotropic rows
`Aᵢ`, and `K = max_i ‖Aᵢ‖_{ψ₂}`, there is a positive absolute constant `C` such that for
every `t ≥ 0`, with probability at least `1 - 2 exp (-t²)`,

`√m - C K² (√n + t) ≤ sₙ(A) ≤ s₁(A) ≤ √m + C K² (√n + t)`.
-/
def two_sided_subgaussian_matrices_hdp {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  HasIndependentMeanZeroIsotropicSubGaussianRows A μ →
    ∃ C : ℝ, 0 < C ∧
      ∀ K : ℝ, K = maxMatrixRowSubGaussianPsi2Norm A μ → 0 < K →
        ∀ t : ℝ, 0 ≤ t →
          (μ {ω |
            √(m : ℝ) - C * K ^ 2 * (√(n : ℝ) + t) ≤
                matrixSmallestSingularValue (randomMatrix A ω) ∧
              matrixSmallestSingularValue (randomMatrix A ω) ≤
                matrixLargestSingularValue (randomMatrix A ω) ∧
              matrixLargestSingularValue (randomMatrix A ω) ≤
                √(m : ℝ) + C * K ^ 2 * (√(n : ℝ) + t)}).toReal ≥
            1 - 2 * exp (-(t ^ 2))

omit [MeasurableSpace Ω] in
lemma twoSidedSubgaussianMatricesConstant_pos :
    0 < twoSidedSubgaussianMatricesConstant := by
  have hD_pos : 0 < sampleCovarianceDeviationScaleConstant := by
    dsimp [sampleCovarianceDeviationScaleConstant, hdpHansonWrightConstant]
    positivity
  dsimp [twoSidedSubgaussianMatricesConstant]
  positivity

/-- HDP Theorem 4.6.1 for positive dimensions. -/
theorem two_sided_subgaussian_matrices_hdp_of_pos {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] :
    two_sided_subgaussian_matrices_hdp A μ := by
  intro hA
  refine ⟨twoSidedSubgaussianMatricesConstant, twoSidedSubgaussianMatricesConstant_pos, ?_⟩
  intro K hK_def hK t ht
  let B : ℝ := twoSidedSubgaussianMatricesConstant * K ^ 2 * (√(n : ℝ) + t)
  have hB_nonneg : 0 ≤ B := by
    have hS_nonneg : 0 ≤ √(n : ℝ) + t := by positivity
    dsimp [B]
    exact mul_nonneg
      (mul_nonneg twoSidedSubgaussianMatricesConstant_pos.le (sq_nonneg K)) hS_nonneg
  have hK_lower :
      1 / (4 * exp 1 ^ 2) ≤ K ^ 2 :=
    hA.max_row_psi2_sq_lower hm hn hK_def hK
  have hscale :
      2 * sampleCovarianceDeviationHdpScale m n K t ≤
        max (B / √(m : ℝ)) ((B / √(m : ℝ)) ^ 2) := by
    simpa [B, mul_assoc, mul_left_comm, mul_comm] using
      sampleCovarianceDeviationHdpScale_le_max_hdp_radius
        (m := m) (n := n) hm (K := K) (t := t) hK ht hK_lower
  simpa [B, mul_assoc, mul_left_comm, mul_comm] using
    singular_value_sandwich_probability_of_sampleCovarianceDeviationHdpScale
      (m := m) (n := n) hm hn (A := A) (μ := μ) hA hK_def hK hB_nonneg ht hscale

theorem sampleCovarianceDeviationOperatorNorm_expectation_hdp_bound_of_pos
    {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    {A : Fin m → Fin n → Ω → ℝ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    (hA : HasIndependentMeanZeroIsotropicSubGaussianRows A μ)
    {K : ℝ} (hK_def : K = maxMatrixRowSubGaussianPsi2Norm A μ) (hK : 0 < K) :
    ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ ≤
      152 * hdpHansonWrightConstant * K ^ 2 *
        (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ)) := by
  obtain ⟨N, hN_domain, _hN_codomain⟩ := exists_quarter_centeredMatrixBilinearNet n n
  let signed : Finset (EuclideanSpace ℝ (Fin n) × Bool) := N.signedDomain
  let Z : EuclideanSpace ℝ (Fin n) × Bool → Ω → ℝ :=
    fun q ω => signedSampleCovarianceDeviationValue (randomMatrix A ω) q
  have hdomain_nonempty : N.domainNet.Nonempty :=
    N.toMatrixBilinearNet.domainNet_nonempty_of_pos hn
  have hsigned_nonempty : signed.Nonempty := by
    exact hdomain_nonempty.product ⟨true, Finset.mem_univ true⟩
  have hsigned_card : 2 ≤ signed.card := by
    simpa [signed] using N.signedDomain_card_ge_two_of_pos hn
  let R : ℝ := log (signed.card : ℝ)
  let C0 : ℝ := hdpHansonWrightConstant
  let V : ℝ := C0 * K ^ 4 / (m : ℝ)
  let L : ℝ := (m : ℝ) / (2 * C0 * K ^ 2)
  have hmR_pos : 0 < (m : ℝ) := by exact_mod_cast hm
  have hC0_pos : 0 < C0 := by
    dsimp [C0]
    exact hdpHansonWrightConstant_pos
  have hC0_nonneg : 0 ≤ C0 := hC0_pos.le
  have hC0_ge_one : 1 ≤ C0 := by
    dsimp [C0, hdpHansonWrightConstant]
    nlinarith [one_le_exp (by norm_num : (0 : ℝ) ≤ 1), sq_nonneg (exp 1)]
  have hV_pos : 0 < V := by
    dsimp [V]
    positivity
  have hV_nonneg : 0 ≤ V := hV_pos.le
  have hL_pos : 0 < L := by
    dsimp [L]
    positivity
  have hL_nonneg : 0 ≤ L := hL_pos.le
  have hR_pos : 0 < R := by
    dsimp [R]
    exact log_pos (by exact_mod_cast hsigned_card)
  have hR_nonneg : 0 ≤ R := hR_pos.le
  have hR_le : R ≤ 19 * (n : ℝ) := by
    simpa [R, signed] using N.log_signedDomain_card_le hn hN_domain
  have hZ_meas : ∀ q ∈ signed, Measurable (Z q) := by
    intro q hq
    have hbase :=
      fixed_vector_matrix_apply_sq_sub_scaled_dim_measurable
        (A := A) hA.measurable q.1
    by_cases hsign : q.2
    · simpa [Z, signedSampleCovarianceDeviationValue, hsign] using hbase
    · simpa [Z, signedSampleCovarianceDeviationValue, hsign] using hbase.neg
  have hZ_int : ∀ q ∈ signed, Integrable (Z q) μ := by
    intro q hq
    have hbase :=
      fixed_vector_matrix_apply_sq_sub_scaled_dim_integrable_hdp
        (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def (x := q.1)
    by_cases hsign : q.2
    · simpa [Z, signedSampleCovarianceDeviationValue, hsign] using hbase
    · convert hbase.neg using 1
      funext ω
      simp [Z, signedSampleCovarianceDeviationValue, hsign]
  have hZ_cgf_at :
      ∀ q ∈ signed, ∀ l : ℝ, 0 < l → l ≤ L →
        cgf (Z q) μ l ≤ V * l ^ 2 := by
    intro q hq l hl_pos hl_le
    have hq_domain : q.1 ∈ N.domainNet := by
      have hq' : q ∈ N.domainNet ×ˢ (Finset.univ : Finset Bool) := by
        simpa [signed, CenteredMatrixBilinearNet.signedDomain] using hq
      exact (Finset.mem_product.mp hq').1
    have hx_norm : ‖q.1‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hq_domain)
    have hl_abs : |l| ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2) := by
      simpa [L, C0, abs_of_nonneg hl_pos.le] using hl_le
    by_cases hsign : q.2
    · have hbase :=
        fixed_vector_matrix_apply_sq_sub_scaled_dim_cgf_le_hdp
          (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hl_abs hx_norm
      simpa [Z, V, C0, signedSampleCovarianceDeviationValue, hsign, mul_assoc,
        mul_left_comm, mul_comm, div_eq_mul_inv] using hbase
    · have hl_abs_neg : |-l| ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2) := by
        simpa [abs_neg] using hl_abs
      have hbase :=
        fixed_vector_matrix_apply_sq_sub_scaled_dim_cgf_le_hdp
          (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hl_abs_neg hx_norm
      have hcgf_neg :
          cgf (Z q) μ l =
            cgf
              (fun ω => (m : ℝ)⁻¹ * ‖(randomMatrix A ω).toEuclideanLin q.1‖ ^ 2 -
                ‖q.1‖ ^ 2) μ (-l) := by
        unfold cgf mgf
        congr 1
        apply integral_congr_ae
        filter_upwards with ω
        congr 1
        simp [Z, signedSampleCovarianceDeviationValue, hsign]
        ring
      rw [hcgf_neg]
      simpa [V, C0, mul_assoc, mul_left_comm, mul_comm, div_eq_mul_inv] using hbase
  have hZ_int_exp_at :
      ∀ q ∈ signed, ∀ l : ℝ, 0 < l → l ≤ L →
        Integrable (fun ω => exp (l * Z q ω)) μ := by
    intro q hq l hl_pos hl_le
    have hq_domain : q.1 ∈ N.domainNet := by
      have hq' : q ∈ N.domainNet ×ˢ (Finset.univ : Finset Bool) := by
        simpa [signed, CenteredMatrixBilinearNet.signedDomain] using hq
      exact (Finset.mem_product.mp hq').1
    have hx_norm : ‖q.1‖ ≤ 1 :=
      norm_le_one_of_mem_euclideanUnitBall (N.domain_subset_unitBall hq_domain)
    have hl_abs : |l| ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2) := by
      simpa [L, C0, abs_of_nonneg hl_pos.le] using hl_le
    by_cases hsign : q.2
    · have hbase :=
        fixed_vector_matrix_apply_sq_sub_scaled_dim_integrable_exp_hdp
          (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hl_abs hx_norm
      simpa [Z, signedSampleCovarianceDeviationValue, hsign] using hbase
    · have hl_abs_neg : |-l| ≤ (m : ℝ) / (2 * hdpHansonWrightConstant * K ^ 2) := by
        simpa [abs_neg] using hl_abs
      have hbase :=
        fixed_vector_matrix_apply_sq_sub_scaled_dim_integrable_exp_hdp
          (m := m) (n := n) hm (A := A) (μ := μ) hA hK_def hK hl_abs_neg hx_norm
      convert hbase using 1
      funext ω
      simp [Z, signedSampleCovarianceDeviationValue, hsign]
      ring
  have hsup_int : Integrable (fun ω => signed.sup' hsigned_nonempty (fun q => Z q ω)) μ := by
    refine Integrable.mono
      (integrable_finsetSum signed (fun q hq => (hZ_int q hq).abs)) ?_ ?_
    · convert (Finset.measurable_sup' hsigned_nonempty (fun q hq => hZ_meas q hq)).aestronglyMeasurable using 1
      funext ω
      simp only [Finset.sup'_apply]
    · refine ae_of_all μ (fun ω => ?_)
      simp only [Real.norm_eq_abs]
      have h_sum_nonneg : 0 ≤ ∑ q ∈ signed, |Z q ω| :=
        Finset.sum_nonneg fun q _ => abs_nonneg _
      rw [abs_of_nonneg h_sum_nonneg]
      convert abs_sup'_le_sum hsigned_nonempty (fun q => Z q ω) using 2
  have hcov_nonneg :
      0 ≤ᵐ[μ] fun ω => sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) :=
    ae_of_all μ fun _ => norm_nonneg _
  have hcov_le_sup :
      (fun ω => sampleCovarianceDeviationOperatorNorm (randomMatrix A ω)) ≤ᵐ[μ]
        fun ω => 2 * signed.sup' hsigned_nonempty (fun q => Z q ω) := by
    refine ae_of_all μ (fun ω => ?_)
    have hdet :=
      sampleCovarianceDeviationOperatorNorm_le_two_mul_signed_sup_of_quarter_centered_net
        (A := randomMatrix A ω) (N := N) hdomain_nonempty
    simpa [signed, Z, CenteredMatrixBilinearNet.signedDomain] using hdet
  have hcov_le_expect_sup :
      ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ ≤
        2 * ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ := by
    calc
      ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ
          ≤ ∫ ω, 2 * signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ :=
            integral_mono_of_nonneg hcov_nonneg (hsup_int.const_mul 2) hcov_le_sup
      _ = 2 * ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ := by
            rw [integral_const_mul]
  have hsup_bound_at : ∀ l : ℝ, 0 < l → l ≤ L →
      ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ ≤
        (1 / l) * (R + V * l ^ 2) := by
    intro l hl_pos hl_le
    have hmax :=
      expected_max_of_cgf_bound_at
        (μ := μ) (X := Z) (s := signed) hsigned_nonempty hsigned_card
        (l := l) (a := V * l ^ 2) hl_pos hZ_meas hZ_int
        (fun q hq => hZ_cgf_at q hq l hl_pos hl_le)
        (fun q hq => hZ_int_exp_at q hq l hl_pos hl_le)
    simpa [R] using hmax
  have hratio_nonneg : 0 ≤ (n : ℝ) / (m : ℝ) := by positivity
  have hsqrt_ratio_nonneg : 0 ≤ √((n : ℝ) / (m : ℝ)) := sqrt_nonneg _
  have htarget_nonneg :
      0 ≤ 152 * C0 * K ^ 2 * (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ)) := by
    positivity
  by_cases hsmall : R ≤ V * L ^ 2
  · let l : ℝ := √(R / V)
    have hl_pos : 0 < l := by
      dsimp [l]
      exact sqrt_pos.mpr (div_pos hR_pos hV_pos)
    have hl_sq : l ^ 2 = R / V := by
      dsimp [l]
      exact sq_sqrt (div_nonneg hR_nonneg hV_nonneg)
    have hR_eq : R = V * l ^ 2 := by
      rw [hl_sq]
      field_simp [hV_pos.ne']
    have hl_le : l ≤ L := by
      rw [sqrt_le_left hL_nonneg]
      calc
        R / V ≤ (V * L ^ 2) / V :=
          div_le_div_of_nonneg_right hsmall hV_nonneg
        _ = L ^ 2 := by field_simp [hV_pos.ne']
    have hsup_bound := hsup_bound_at l hl_pos hl_le
    have hsup_sqrt :
        ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ ≤ 2 * √(V * R) := by
      have hVl_nonneg : 0 ≤ V * l := mul_nonneg hV_nonneg hl_pos.le
      have hsqrt_eq : √(V * R) = V * l := by
        have hVR_eq : V * R = (V * l) ^ 2 := by
          rw [hR_eq]
          ring
        rw [hVR_eq, sqrt_sq hVl_nonneg]
      calc
        ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ
            ≤ (1 / l) * (R + V * l ^ 2) := hsup_bound
        _ = 2 * √(V * R) := by
          rw [← hR_eq, hsqrt_eq]
          field_simp [hl_pos.ne']
          nlinarith [hR_eq]
    have hcov_sqrt :
        ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ ≤
          4 * √(V * R) := by
      calc
        ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ
            ≤ 2 * ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ :=
              hcov_le_expect_sup
        _ ≤ 2 * (2 * √(V * R)) :=
              mul_le_mul_of_nonneg_left hsup_sqrt (by norm_num)
        _ = 4 * √(V * R) := by ring
    have hsqrt_bound :
        √(V * R) ≤ 38 * C0 * K ^ 2 * √((n : ℝ) / (m : ℝ)) := by
      have hright_nonneg : 0 ≤ 38 * C0 * K ^ 2 * √((n : ℝ) / (m : ℝ)) := by
        positivity
      rw [sqrt_le_left hright_nonneg]
      calc
        V * R ≤ (C0 * K ^ 4 / (m : ℝ)) * (19 * (n : ℝ)) :=
          mul_le_mul_of_nonneg_left hR_le hV_nonneg
        _ ≤ (38 * C0 * K ^ 2 * √((n : ℝ) / (m : ℝ))) ^ 2 := by
          have hmain : (19 : ℝ) ≤ C0 * 38 ^ 2 := by
            nlinarith [hC0_ge_one]
          have hnm_sqrt_sq : (√((n : ℝ) / (m : ℝ))) ^ 2 = (n : ℝ) / (m : ℝ) :=
            sq_sqrt hratio_nonneg
          rw [mul_pow, mul_pow, hnm_sqrt_sq]
          field_simp [hmR_pos.ne']
          exact hmain
    calc
      ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ
          ≤ 4 * √(V * R) := hcov_sqrt
      _ ≤ 4 * (38 * C0 * K ^ 2 * √((n : ℝ) / (m : ℝ))) :=
            mul_le_mul_of_nonneg_left hsqrt_bound (by norm_num)
      _ = 152 * C0 * K ^ 2 * √((n : ℝ) / (m : ℝ)) := by ring
      _ ≤ 152 * C0 * K ^ 2 * (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ)) := by
            exact mul_le_mul_of_nonneg_left (le_add_of_nonneg_right hratio_nonneg)
              (by positivity)
      _ = 152 * hdpHansonWrightConstant * K ^ 2 *
          (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ)) := by
            dsimp [C0]
  · have hlarge : V * L ^ 2 < R := lt_of_not_ge hsmall
    have hsup_bound := hsup_bound_at L hL_pos le_rfl
    have hsup_linear :
        ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ ≤ 2 * R / L := by
      calc
        ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ
            ≤ (1 / L) * (R + V * L ^ 2) := hsup_bound
        _ ≤ (1 / L) * (R + R) := by
              have hadd : R + V * L ^ 2 ≤ R + R := by linarith
              exact mul_le_mul_of_nonneg_left hadd
                (le_of_lt (one_div_pos.mpr hL_pos))
        _ = 2 * R / L := by ring
    have hcov_linear :
        ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ ≤
          4 * R / L := by
      calc
        ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ
            ≤ 2 * ∫ ω, signed.sup' hsigned_nonempty (fun q => Z q ω) ∂μ :=
              hcov_le_expect_sup
        _ ≤ 2 * (2 * R / L) := mul_le_mul_of_nonneg_left hsup_linear (by norm_num)
        _ = 4 * R / L := by ring
    calc
      ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ
          ≤ 4 * R / L := hcov_linear
      _ = 8 * C0 * K ^ 2 * R / (m : ℝ) := by
            dsimp [L]
            field_simp [hmR_pos.ne', hK.ne', hC0_pos.ne']
            ring
      _ ≤ 8 * C0 * K ^ 2 * (19 * (n : ℝ)) / (m : ℝ) := by
            exact div_le_div_of_nonneg_right
              (mul_le_mul_of_nonneg_left hR_le (by positivity)) hmR_pos.le
      _ ≤ 152 * C0 * K ^ 2 * ((n : ℝ) / (m : ℝ)) := by
            have hnum : (8 : ℝ) * 19 ≤ 152 := by norm_num
            field_simp [hmR_pos.ne']
            exact hnum
      _ ≤ 152 * C0 * K ^ 2 * (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ)) := by
            exact mul_le_mul_of_nonneg_left (le_add_of_nonneg_left hsqrt_ratio_nonneg)
              (by positivity)
      _ = 152 * hdpHansonWrightConstant * K ^ 2 *
          (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ)) := by
            dsimp [C0]

/--
HDP Remark 4.6.2, "Expectation".

The expectation form of Theorem 4.6.1 states

`E ‖m⁻¹ AᵀA - Iₙ‖ ≤ C K² (√(n / m) + n / m)`.
-/
def two_sided_subgaussian_matrices_expectation_hdp {m n : ℕ}
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  HasIndependentMeanZeroIsotropicSubGaussianRows A μ →
    ∃ C : ℝ, 0 < C ∧
      ∀ K : ℝ, K = maxMatrixRowSubGaussianPsi2Norm A μ → 0 < K →
        ∫ ω, sampleCovarianceDeviationOperatorNorm (randomMatrix A ω) ∂μ ≤
          C * K ^ 2 * (√((n : ℝ) / (m : ℝ)) + (n : ℝ) / (m : ℝ))

/-- HDP Remark 4.6.2 for positive dimensions. -/
theorem two_sided_subgaussian_matrices_expectation_hdp_of_pos {m n : ℕ}
    (hm : 0 < m) (hn : 0 < n)
    (A : Fin m → Fin n → Ω → ℝ) (μ : Measure Ω) [IsProbabilityMeasure μ] :
    two_sided_subgaussian_matrices_expectation_hdp A μ := by
  intro hA
  refine ⟨152 * hdpHansonWrightConstant, ?_, ?_⟩
  · exact mul_pos (by norm_num) hdpHansonWrightConstant_pos
  · intro K hK_def hK
    exact sampleCovarianceDeviationOperatorNorm_expectation_hdp_bound_of_pos
      (m := m) (n := n) hm hn (A := A) (μ := μ) hA hK_def hK

end RMT
