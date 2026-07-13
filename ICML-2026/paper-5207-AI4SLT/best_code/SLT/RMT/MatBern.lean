/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.RMT.Lieb
import Mathlib.Probability.Independence.Basic

/-!
# Matrix Bernstein Inequality

This file records the HDP Theorem 5.4.1 statement in the local random-matrix notation.

## Main definitions

* `RMT.matrixBernsteinSum`: the random sum `∑ᵢ Xᵢ`.
* `RMT.matrixBernsteinTraceMgf`: the trace MGF `E tr exp(λ∑ᵢ Xᵢ)`.
* `RMT.matrixBernsteinIndividualMgf`: the individual matrix MGF `E exp(λXᵢ)`.
* `RMT.matrixBernsteinLogMgfSum`: the log-MGF sum `∑ᵢ log E exp(λXᵢ)` from HDP (5.17).
* `RMT.matrixBernsteinPiFamily`: the canonical coordinate family on a finite product matrix
  space.
* `RMT.matrixBernsteinVarianceMatrix`: the matrix variance proxy `∑ᵢ E Xᵢ²`.
* `RMT.matrixBernsteinVarianceProxy`: the scalar variance proxy `‖∑ᵢ E Xᵢ²‖`.
* `RMT.matrixLargestEigenvalue`: the largest sorted eigenvalue of a Hermitian matrix.
* `RMT.matrixBernsteinUpperTailEvent` / `RMT.matrixBernsteinLowerTailEvent`: the one-sided
  spectral tail events used in the proof of HDP Theorem 5.4.1.
* `RMT.matrixBernsteinMgfRate`: the scalar rate `g(λ)` in HDP Lemma 5.4.10.
* `RMT.matrixBernsteinOptimizingLambda`: the value of `λ` chosen in the proof of Theorem 5.4.1.
* `RMT.matrixBernsteinOneSidedTailBound`: the one-sided Bernstein right-hand side.
* `RMT.matrixBernsteinTailBound`: the Bernstein right-hand side from HDP Theorem 5.4.1.
* `RMT.matrix_bernstein_inequality_hdp`: the exact proposition stated by HDP Theorem 5.4.1.

## Main results

* `RMT.matrixBernstein_upperTail_le_of_trace_mgf_bound`: the fixed-`λ` Markov step for the upper
  spectral tail.
* `RMT.matrixBernstein_upperTail_le_of_trace_mgf_bound_at_optimizingLambda`: the optimized
  fixed-`λ` Markov step with the exact HDP Bernstein exponent.
* `RMT.matrixBernstein_upperTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae`:
  the optimized one-sided upper tail, assuming only the trace-MGF estimate remains.
* `RMT.matrixBernstein_norm_tail_le_of_optimized_trace_mgf_bounds_of_summands_isHermitian_ae`:
  the nondegenerate HDP norm-tail bound from the two optimized trace-MGF estimates.
* `RMT.matrixBernsteinTraceMgf_le_of_trace_exp_variance_bound`: converts the variance-matrix
  trace bound into the scalar `n exp(g(λ)σ²)` trace-MGF bound.
* `RMT.matrixBernsteinTraceMgf_le_of_lieb_recursion_and_individual_mgf_bounds`: converts the
  Lieb-recursion trace bound and individual matrix MGF Loewner bounds into the variance trace
  bound.
* `RMT.matrixBernstein_lieb_recursion_of_independent`: the arbitrary finite-family HDP Step 2
  recursion for independent summands.
* `RMT.matrixBernstein_norm_tail_le_of_lieb_recursions_and_individual_mgf_bounds_at_optimizingLambda`:
  the nondegenerate HDP norm-tail bound from the exact Step 2 recursion and Lemma 5.4.10 bounds.
* `RMT.matrixBernstein_norm_tail_le_of_trace_exp_variance_bounds_at_optimizingLambda`: the
  nondegenerate HDP norm-tail bound from the two variance-matrix trace bounds.
* `RMT.matrixBernstein_norm_tail_le_of_one_sided_eigenvalue_tails_ae`: the final two-sided
  union-bound step under the a.e. Hermitian event inclusion.
* `RMT.matrix_bernstein_inequality_hdp_of_no_summands`: proves the exact HDP proposition for
  `N = 0`.
* `RMT.matrix_bernstein_inequality_hdp_all`: proves the exact HDP proposition for every finite
  `N`.
-/

open MeasureTheory ProbabilityTheory Real
open scoped BigOperators MatrixOrder Matrix.Norms.L2Operator Topology

noncomputable section

namespace RMT

variable {Ω : Type*} [MeasurableSpace Ω]

local instance {n : ℕ} : MeasurableSpace (Matrix (Fin n) (Fin n) ℝ) := borel _
local instance {n : ℕ} : BorelSpace (Matrix (Fin n) (Fin n) ℝ) := ⟨rfl⟩

lemma measurableSet_matrix_isHermitian {n : ℕ} :
    MeasurableSet {A : Matrix (Fin n) (Fin n) ℝ | A.IsHermitian} := by
  have hclosed :
      IsClosed {A : Matrix (Fin n) (Fin n) ℝ | Matrix.transpose A = A} :=
    isClosed_eq continuous_id.matrix_transpose continuous_id
  simpa [Matrix.IsHermitian, Matrix.conjTranspose_eq_transpose_of_trivial] using
    hclosed.measurableSet

lemma measurableSet_matrix_norm_le {n : ℕ} {K : ℝ} :
    MeasurableSet {A : Matrix (Fin n) (Fin n) ℝ | ‖A‖ ≤ K} :=
  (isClosed_le continuous_norm continuous_const).measurableSet

/-! ## HDP Theorem 5.4.1 data -/

/-- Independence of the matrix-valued family `Xᵢ` in HDP Theorem 5.4.1. -/
def MatrixBernsteinIndependent {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (μ : Measure Ω) : Prop :=
  letI : MeasurableSpace (Matrix (Fin n) (Fin n) ℝ) := borel _
  @iIndepFun Ω (Fin N) _ (fun _ : Fin N => Matrix (Fin n) (Fin n) ℝ)
    (fun _ => inferInstance) X μ

lemma MatrixBernsteinIndependent.neg {n N : ℕ}
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {μ : Measure Ω}
    (hX : MatrixBernsteinIndependent X μ) :
    MatrixBernsteinIndependent (fun i ω => -X i ω) μ := by
  unfold MatrixBernsteinIndependent at hX ⊢
  simpa [Function.comp_def] using
    hX.comp (fun _ A => -A) (fun _ => continuous_neg.borel_measurable)

/-- The random matrix sum `∑ᵢ Xᵢ` appearing in HDP Theorem 5.4.1. -/
def matrixBernsteinSum {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (ω : Ω) :
    Matrix (Fin n) (Fin n) ℝ :=
  ∑ i : Fin N, X i ω

/--
The canonical coordinate family on a finite product of matrix spaces.

This is the product-law model of an independent finite family.
-/
def matrixBernsteinPiFamily {n N : ℕ} :
    Fin N →
      (Fin N → Matrix (Fin n) (Fin n) ℝ) →
        Matrix (Fin n) (Fin n) ℝ :=
  fun i p => p i

@[simp]
lemma matrixBernsteinPiFamily_apply {n N : ℕ}
    (i : Fin N) (p : Fin N → Matrix (Fin n) (Fin n) ℝ) :
    matrixBernsteinPiFamily i p = p i := rfl

/-- The trace MGF `E tr exp(λ∑ᵢ Xᵢ)` appearing in the proof of HDP Theorem 5.4.1. -/
def matrixBernsteinTraceMgf {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (l : ℝ) : ℝ :=
  ∫ ω, Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) ∂μ

/-- The individual matrix MGF `E exp(λXᵢ)` appearing in HDP Lemma 5.4.10. -/
def matrixBernsteinIndividualMgf {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (l : ℝ) (i : Fin N) :
    Matrix (Fin n) (Fin n) ℝ :=
  ∫ ω, NormedSpace.exp (l • X i ω) ∂μ

lemma matrixBernsteinIndividualMgf_piFamily {n N : ℕ}
    (μs : Fin N → Measure (Matrix (Fin n) (Fin n) ℝ))
    [∀ i, IsProbabilityMeasure (μs i)] (l : ℝ) (i : Fin N) :
    matrixBernsteinIndividualMgf (Measure.pi μs) matrixBernsteinPiFamily l i =
      ∫ A : Matrix (Fin n) (Fin n) ℝ, NormedSpace.exp (l • A) ∂μs i := by
  let Mat : Type := Matrix (Fin n) (Fin n) ℝ
  letI : NormedAlgebra ℚ Mat := NormedAlgebra.restrictScalars ℚ ℝ Mat
  let f : Mat → Mat := fun A => NormedSpace.exp (l • A)
  have hmp : MeasurePreserving (fun p : Fin N → Mat => p i) (Measure.pi μs) (μs i) :=
    measurePreserving_eval (μ := μs) i
  have hf_meas :
      AEStronglyMeasurable f (Measure.map (fun p : Fin N → Mat => p i) (Measure.pi μs)) := by
    exact (NormedSpace.exp_continuous.comp (continuous_const_smul l)).aestronglyMeasurable
  calc
    matrixBernsteinIndividualMgf (Measure.pi μs) matrixBernsteinPiFamily l i =
        ∫ p : Fin N → Mat, f (p i) ∂Measure.pi μs := by
          rfl
    _ = ∫ A : Mat, f A ∂Measure.map (fun p : Fin N → Mat => p i) (Measure.pi μs) := by
          exact (integral_map (measurable_pi_apply i).aemeasurable hf_meas).symm
    _ = ∫ A : Mat, f A ∂μs i := by
          rw [hmp.map_eq]

/-- The logarithmic matrix-MGF sum `∑ᵢ log(E exp(λXᵢ))` from HDP equation (5.17). -/
def matrixBernsteinLogMgfSum {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (l : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  ∑ i : Fin N, CFC.log (matrixBernsteinIndividualMgf μ X l i)

lemma matrixBernsteinLogMgfSum_piFamily {n N : ℕ}
    (μs : Fin N → Measure (Matrix (Fin n) (Fin n) ℝ))
    [∀ i, IsProbabilityMeasure (μs i)] (l : ℝ) :
    matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l =
      ∑ i : Fin N,
        CFC.log (∫ A : Matrix (Fin n) (Fin n) ℝ, NormedSpace.exp (l • A) ∂μs i) := by
  simp [matrixBernsteinLogMgfSum, matrixBernsteinIndividualMgf_piFamily]

/--
The largest sorted eigenvalue of a Hermitian real matrix.

The definition returns `0` on non-Hermitian inputs; the HDP proof only uses it on the a.e.
Hermitian random sums.
-/
def matrixLargestEigenvalue {n : ℕ} (hn : 0 < n)
    (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  if hA : A.IsHermitian then
    hA.eigenvaluesToEuclideanLin ⟨0, by simpa using hn⟩
  else 0

/-- The upper-tail event `λmax(∑ᵢ Xᵢ) ≥ t` from the proof of HDP Theorem 5.4.1. -/
def matrixBernsteinUpperTailEvent {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Set Ω :=
  {ω | matrixLargestEigenvalue hn (matrixBernsteinSum X ω) ≥ t}

/-- The lower-tail event `λmax(-∑ᵢ Xᵢ) ≥ t` from the proof of HDP Theorem 5.4.1. -/
def matrixBernsteinLowerTailEvent {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Set Ω :=
  {ω | matrixLargestEigenvalue hn (-(matrixBernsteinSum X ω)) ≥ t}

omit [MeasurableSpace Ω] in
@[simp]
lemma matrixBernsteinLowerTailEvent_eq_upperTailEvent_neg {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    matrixBernsteinLowerTailEvent hn X t =
      matrixBernsteinUpperTailEvent hn (fun i ω => -X i ω) t := by
  ext ω
  simp [matrixBernsteinLowerTailEvent, matrixBernsteinUpperTailEvent, matrixBernsteinSum]

/-- The matrix variance proxy `∑ᵢ E Xᵢ²` from HDP Theorem 5.4.1. -/
def matrixBernsteinVarianceMatrix {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  ∑ i : Fin N, ∫ ω, (X i ω) ^ 2 ∂μ

/-- The scalar variance proxy `σ² = ‖∑ᵢ E Xᵢ²‖` from HDP Theorem 5.4.1. -/
def matrixBernsteinVarianceProxy {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ‖matrixBernsteinVarianceMatrix μ X‖

lemma matrixBernsteinVarianceSummand_sq_posSemidef_ae {n N : ℕ}
    {μ : Measure Ω} {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} (i : Fin N)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian) :
    ∀ᵐ ω ∂μ, ((X i ω) ^ 2).PosSemidef := by
  filter_upwards [hX_symmetric] with ω hω
  have hsymm : (X i ω).IsSymm := by
    simpa using hω
  simpa [pow_two, hsymm.eq] using Matrix.posSemidef_self_mul_conjTranspose (X i ω)

lemma matrixBernsteinVarianceSummand_integral_isHermitian {n N : ℕ}
    {μ : Measure Ω} {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} (i : Fin N)
    (h_square_integrable : Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian) :
    (∫ ω, (X i ω) ^ 2 ∂μ).IsHermitian :=
  (Matrix.integral_posSemidef_of_ae h_square_integrable
    (matrixBernsteinVarianceSummand_sq_posSemidef_ae i hX_symmetric)).1

lemma matrixBernsteinVarianceMatrix_isHermitian_ae {n N : ℕ} {μ : Measure Ω}
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian) :
    (matrixBernsteinVarianceMatrix μ X).IsHermitian := by
  classical
  let f : Fin N → Matrix (Fin n) (Fin n) ℝ := fun i => ∫ ω, (X i ω) ^ 2 ∂μ
  have hf : ∀ i, (f i).IsHermitian := fun i =>
    matrixBernsteinVarianceSummand_integral_isHermitian i
      (h_square_integrable i) (hX_symmetric i)
  have hsum : ∀ s : Finset (Fin N), (Finset.sum s f).IsHermitian := by
    intro s
    refine Finset.induction_on s ?base ?step
    · simp [f]
    · intro i s his hs
      rw [Finset.sum_insert his]
      exact (hf i).add hs
  simpa [matrixBernsteinVarianceMatrix, f] using hsum Finset.univ

omit [MeasurableSpace Ω] in
@[simp]
lemma matrixBernsteinSum_neg {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (ω : Ω) :
    matrixBernsteinSum (fun i ω => -X i ω) ω = -matrixBernsteinSum X ω := by
  simp [matrixBernsteinSum]

@[simp]
lemma matrixBernsteinVarianceMatrix_neg {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) :
    matrixBernsteinVarianceMatrix μ (fun i ω => -X i ω) =
      matrixBernsteinVarianceMatrix μ X := by
  simp [matrixBernsteinVarianceMatrix]

@[simp]
lemma matrixBernsteinVarianceProxy_neg {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) :
    matrixBernsteinVarianceProxy μ (fun i ω => -X i ω) =
      matrixBernsteinVarianceProxy μ X := by
  simp [matrixBernsteinVarianceProxy, matrixBernsteinVarianceMatrix]

/-- The right-hand side in HDP Theorem 5.4.1. -/
def matrixBernsteinTailBound (n : ℕ) (σ_sq K t : ℝ) : ℝ :=
  2 * (n : ℝ) * Real.exp (-(t ^ 2 / 2) / (σ_sq + K * t / 3))

/-- The one-sided right-hand side for `λmax` in the proof of HDP Theorem 5.4.1. -/
def matrixBernsteinOneSidedTailBound (n : ℕ) (σ_sq K t : ℝ) : ℝ :=
  (n : ℝ) * Real.exp (-(t ^ 2 / 2) / (σ_sq + K * t / 3))

/--
The scalar rate

`g(λ) = (λ²/2)/(1 - |λ|K/3)`

from HDP Lemma 5.4.10.
-/
def matrixBernsteinMgfRate (K l : ℝ) : ℝ :=
  (l ^ 2 / 2) / (1 - |l| * K / 3)

/-- The value of `λ` chosen in the proof of HDP Theorem 5.4.1. -/
def matrixBernsteinOptimizingLambda (σ_sq K t : ℝ) : ℝ :=
  t / (σ_sq + K * t / 3)

lemma matrixBernsteinVarianceProxy_nonneg {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ matrixBernsteinVarianceProxy μ X :=
  norm_nonneg _

lemma matrixBernsteinMgfRate_nonneg {K l : ℝ}
    (hden : 0 ≤ 1 - |l| * K / 3) :
    0 ≤ matrixBernsteinMgfRate K l := by
  exact div_nonneg (div_nonneg (sq_nonneg l) (by norm_num)) hden

omit [MeasurableSpace Ω] in
lemma real_exp_le_taylor_four_add_abs_fifth_of_abs_le_three {u : ℝ}
    (hu : |u| ≤ 3) :
    Real.exp u ≤ (∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ)) + |u| ^ 5 / 60 := by
  have hcomplex :
      ‖Complex.exp (u : ℂ) -
          ∑ m ∈ Finset.range 5, (u : ℂ) ^ m / (m.factorial : ℂ)‖ ≤
        ‖(u : ℂ)‖ ^ 5 / (Nat.factorial 5 : ℝ) * 2 := by
    apply Complex.exp_bound'
    rw [Complex.norm_real, Real.norm_eq_abs]
    norm_num
    nlinarith [abs_nonneg u, hu]
  have hcast :
      ((Real.exp u - ∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ) : ℝ) : ℂ) =
        Complex.exp (u : ℂ) -
          ∑ m ∈ Finset.range 5, (u : ℂ) ^ m / (m.factorial : ℂ) := by
    simp [Complex.ofReal_exp, Complex.ofReal_sum, Complex.ofReal_pow]
  have hreal_abs :
      |Real.exp u - ∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ)| ≤
        |u| ^ 5 / (Nat.factorial 5 : ℝ) * 2 := by
    rw [← Real.norm_eq_abs]
    rw [← Complex.norm_real
      (Real.exp u - ∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ))]
    rw [hcast]
    simpa [Complex.norm_real, Real.norm_eq_abs] using hcomplex
  have hupper := (abs_sub_le_iff.mp hreal_abs).1
  calc
    Real.exp u ≤ (∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ)) +
        |u| ^ 5 / (Nat.factorial 5 : ℝ) * 2 := by
          linarith
    _ = (∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ)) + |u| ^ 5 / 60 := by
          ring

omit [MeasurableSpace Ω] in
/-- The scalar Bernstein envelope used in HDP Lemma 5.4.10. -/
lemma real_exp_le_bernstein_quadratic_of_abs_lt_three {u : ℝ} (hu : |u| < 3) :
    Real.exp u ≤ 1 + u + (u ^ 2 / 2) / (1 - |u| / 3) := by
  have htaylor := real_exp_le_taylor_four_add_abs_fifth_of_abs_le_three (le_of_lt hu)
  have hpoly :
      (∑ m ∈ Finset.range 5, u ^ m / (m.factorial : ℝ)) + |u| ^ 5 / 60 ≤
        1 + u + (u ^ 2 / 2) / (1 - |u| / 3) := by
    by_cases hu_nonneg : 0 ≤ u
    · have habs : |u| = u := abs_of_nonneg hu_nonneg
      have hdenpos : 0 < 1 - u / 3 := by
        have hu_lt : u < 3 := by simpa [habs] using hu
        linarith
      rw [habs]
      norm_num [Finset.sum_range_succ, Nat.factorial]
      have htail :
          u ^ 2 / 2 + u ^ 3 / 6 + u ^ 4 / 24 + u ^ 5 / 60 ≤
            (u ^ 2 / 2) / (1 - u / 3) := by
        rw [le_div_iff₀ hdenpos]
        ring_nf
        have hquad : 0 ≤ 5 - u + 2 * u ^ 2 := by
          nlinarith [sq_nonneg (u - 1 / 4)]
        nlinarith [mul_nonneg (sq_nonneg (u ^ 2)) hquad]
      linarith
    · have hu_nonpos : u ≤ 0 := le_of_lt (not_le.mp hu_nonneg)
      have habs : |u| = -u := abs_of_nonpos hu_nonpos
      have hneg_lt : -u < 3 := by
        have h := (abs_lt.mp hu).1
        linarith
      have hdenpos : 0 < 1 - (-u) / 3 := by linarith
      rw [habs]
      norm_num [Finset.sum_range_succ, Nat.factorial]
      let v : ℝ := -u
      have hv_nonneg : 0 ≤ v := by simp [v, hu_nonpos]
      have hv_lt : v < 3 := by simpa [v] using hneg_lt
      have hv_le : v ≤ 3 := hv_lt.le
      have hdenpos_v : 0 < 1 - v / 3 := by linarith
      have htail_v :
          v ^ 2 / 2 - v ^ 3 / 6 + v ^ 4 / 24 + v ^ 5 / 60 ≤
            (v ^ 2 / 2) / (1 - v / 3) := by
        rw [le_div_iff₀ hdenpos_v]
        ring_nf
        have hv_sq_le_three_mul : v ^ 2 ≤ 3 * v := by
          nlinarith [mul_nonneg hv_nonneg (sub_nonneg.mpr hv_le)]
        have hv_sq_le : v ^ 2 ≤ 9 := by nlinarith
        have hv_cube_nonneg : 0 ≤ v ^ 3 := by
          nlinarith [mul_nonneg hv_nonneg (sq_nonneg v)]
        have hfactor : 0 ≤ 120 - 35 * v - v ^ 2 + 2 * v ^ 3 := by
          nlinarith
        nlinarith [mul_nonneg (mul_nonneg hv_nonneg (sq_nonneg v)) hfactor]
      have htail_raw :
          (-u) ^ 2 / 2 - (-u) ^ 3 / 6 + (-u) ^ 4 / 24 + (-u) ^ 5 / 60 ≤
            ((-u) ^ 2 / 2) / (1 - (-u) / 3) := by
        simpa [v] using htail_v
      have htail_simpl :
          u ^ 2 / 2 + u ^ 3 / 6 + u ^ 4 / 24 + (-u) ^ 5 / 60 ≤
            (u ^ 2 / 2) / (1 - -u / 3) := by
        ring_nf at htail_raw ⊢
        exact htail_raw
      nlinarith [htail_simpl]
  exact htaylor.trans hpoly

omit [MeasurableSpace Ω] in
/--
The scalar envelope in HDP Lemma 5.4.10, with the global Bernstein denominator
`1 - |λ|K/3`.
-/
lemma matrixBernstein_scalar_mgfRate_envelope {K l x : ℝ}
    (hden : 0 < 1 - |l| * K / 3) (hx : |x| ≤ K) :
    Real.exp (l * x) ≤ 1 + l * x + matrixBernsteinMgfRate K l * x ^ 2 := by
  have h_abs_lx_le : |l * x| ≤ |l| * K := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hx (abs_nonneg l)
  have h_lK_lt : |l| * K < 3 := by nlinarith
  have hlocal_lt : |l * x| < 3 := h_abs_lx_le.trans_lt h_lK_lt
  have hbase := real_exp_le_bernstein_quadratic_of_abs_lt_three hlocal_lt
  have hlocal_den_pos : 0 < 1 - |l * x| / 3 := by nlinarith
  have hden_le : 1 - |l| * K / 3 ≤ 1 - |l * x| / 3 := by nlinarith
  have hnum_nonneg : 0 ≤ (l * x) ^ 2 / 2 :=
    div_nonneg (sq_nonneg _) (by norm_num)
  have hquad_le :
      ((l * x) ^ 2 / 2) / (1 - |l * x| / 3) ≤
        ((l * x) ^ 2 / 2) / (1 - |l| * K / 3) :=
    div_le_div_of_nonneg_left hnum_nonneg hden hden_le
  have hquad_eq :
      ((l * x) ^ 2 / 2) / (1 - |l| * K / 3) =
        matrixBernsteinMgfRate K l * x ^ 2 := by
    simp [matrixBernsteinMgfRate]
    ring
  nlinarith

lemma matrixBernsteinOptimizingLambda_den_nonneg {σ_sq K t : ℝ}
    (hσ : 0 ≤ σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    0 ≤ σ_sq + K * t / 3 := by
  exact add_nonneg hσ (div_nonneg (mul_nonneg hK ht) (by norm_num))

lemma matrixBernsteinOptimizingLambda_nonneg {σ_sq K t : ℝ}
    (hσ : 0 ≤ σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    0 ≤ matrixBernsteinOptimizingLambda σ_sq K t := by
  exact div_nonneg ht (matrixBernsteinOptimizingLambda_den_nonneg hσ hK ht)

omit [MeasurableSpace Ω] in
/--
At HDP's optimizing value of `λ`, the denominator in `g(λ)` simplifies to
`σ² / (σ² + Kt/3)`.
-/
lemma matrixBernsteinMgfRate_den_optimizingLambda_eq
    {σ_sq K t : ℝ} (hσ : 0 < σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    1 - |matrixBernsteinOptimizingLambda σ_sq K t| * K / 3 =
      σ_sq / (σ_sq + K * t / 3) := by
  have hd_pos : 0 < σ_sq + K * t / 3 := by
    exact add_pos_of_pos_of_nonneg hσ (div_nonneg (mul_nonneg hK ht) (by norm_num))
  have hl_nonneg : 0 ≤ matrixBernsteinOptimizingLambda σ_sq K t :=
    matrixBernsteinOptimizingLambda_nonneg hσ.le hK ht
  rw [abs_of_nonneg hl_nonneg]
  simp [matrixBernsteinOptimizingLambda]
  field_simp [hd_pos.ne']
  ring

omit [MeasurableSpace Ω] in
/-- The optimized `λ` is in the non-singular domain of HDP's scalar MGF rate. -/
lemma matrixBernsteinMgfRate_den_optimizingLambda_pos
    {σ_sq K t : ℝ} (hσ : 0 < σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    0 < 1 - |matrixBernsteinOptimizingLambda σ_sq K t| * K / 3 := by
  rw [matrixBernsteinMgfRate_den_optimizingLambda_eq hσ hK ht]
  have hd_pos : 0 < σ_sq + K * t / 3 := by
    exact add_pos_of_pos_of_nonneg hσ (div_nonneg (mul_nonneg hK ht) (by norm_num))
  exact div_pos hσ hd_pos

omit [MeasurableSpace Ω] in
/--
The scalar exponent identity obtained by inserting HDP's optimizing value of `λ`.

This is the algebraic heart of the last step in the one-sided Bernstein tail. The hypothesis
`0 < σ_sq` keeps the optimizer strictly inside the MGF domain; the degenerate variance case
requires a limiting argument and is kept separate.
-/
lemma matrixBernstein_chernoffExponent_optimizingLambda
    {σ_sq K t : ℝ} (hσ : 0 < σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    -(matrixBernsteinOptimizingLambda σ_sq K t * t) +
        matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) * σ_sq =
      -(t ^ 2 / 2) / (σ_sq + K * t / 3) := by
  let d : ℝ := σ_sq + K * t / 3
  have hd_pos : 0 < d := by
    exact add_pos_of_pos_of_nonneg hσ (div_nonneg (mul_nonneg hK ht) (by norm_num))
  have hl_nonneg : 0 ≤ matrixBernsteinOptimizingLambda σ_sq K t :=
    matrixBernsteinOptimizingLambda_nonneg hσ.le hK ht
  have hden_eq := matrixBernsteinMgfRate_den_optimizingLambda_eq
    (σ_sq := σ_sq) (K := K) (t := t) hσ hK ht
  have hden_ne : 1 - |matrixBernsteinOptimizingLambda σ_sq K t| * K / 3 ≠ 0 := by
    rw [hden_eq]
    exact div_ne_zero hσ.ne' hd_pos.ne'
  simp [matrixBernsteinMgfRate, matrixBernsteinOptimizingLambda]
  rw [abs_of_nonneg]
  rotate_left
  · exact div_nonneg ht hd_pos.le
  field_simp [hd_pos.ne', hden_ne]
  ring

omit [MeasurableSpace Ω] in
/--
The optimized trace-MGF Markov right-hand side is exactly the one-sided Bernstein right-hand side.
-/
lemma matrixBernstein_optimizedTraceMgfBound_eq_oneSided
    (n : ℕ) {σ_sq K t : ℝ} (hσ : 0 < σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t) :
    Real.exp (-(matrixBernsteinOptimizingLambda σ_sq K t * t)) *
        ((n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) *
              σ_sq)) =
      matrixBernsteinOneSidedTailBound n σ_sq K t := by
  have hexp :=
    matrixBernstein_chernoffExponent_optimizingLambda
      (σ_sq := σ_sq) (K := K) (t := t) hσ hK ht
  calc
    Real.exp (-(matrixBernsteinOptimizingLambda σ_sq K t * t)) *
        ((n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) *
              σ_sq))
        = (n : ℝ) *
            (Real.exp (-(matrixBernsteinOptimizingLambda σ_sq K t * t)) *
              Real.exp
                (matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) *
                  σ_sq)) := by
          ring
    _ = (n : ℝ) *
          Real.exp
            (-(matrixBernsteinOptimizingLambda σ_sq K t * t) +
              matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) *
                σ_sq) := by
          rw [Real.exp_add]
    _ = matrixBernsteinOneSidedTailBound n σ_sq K t := by
          rw [hexp]
          rfl

lemma matrixBernsteinTailBound_nonneg (n : ℕ) (σ_sq K t : ℝ) :
    0 ≤ matrixBernsteinTailBound n σ_sq K t := by
  have htwo : (0 : ℝ) ≤ 2 := by norm_num
  have hn : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
  have hexp : 0 ≤ Real.exp (-(t ^ 2 / 2) / (σ_sq + K * t / 3)) :=
    le_of_lt (Real.exp_pos _)
  exact mul_nonneg (mul_nonneg htwo hn) hexp

lemma matrixBernsteinTailBound_eq_two_mul_oneSided (n : ℕ) (σ_sq K t : ℝ) :
    matrixBernsteinTailBound n σ_sq K t =
      2 * matrixBernsteinOneSidedTailBound n σ_sq K t := by
  simp only [matrixBernsteinTailBound, matrixBernsteinOneSidedTailBound]
  ring

/-- The a.e. version of the event inclusion corresponding to HDP equation (5.15). -/
def matrixBernsteinNormEventAESubsetUpperLower {n N : ℕ} (μ : Measure Ω)
    (hn : 0 < n) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ t : ℝ, 0 ≤ t →
    ∀ᵐ ω ∂μ,
      ω ∈ {ω | ‖matrixBernsteinSum X ω‖ ≥ t} →
        ω ∈ matrixBernsteinUpperTailEvent hn X t ∪ matrixBernsteinLowerTailEvent hn X t

lemma probability_real_le_of_ae_subset_union {μ : Measure Ω} [IsProbabilityMeasure μ]
    {s u v : Set Ω} {a b : ℝ}
    (hs : ∀ᵐ ω ∂μ, ω ∈ s → ω ∈ u ∪ v)
    (hu : (μ u).toReal ≤ a) (hv : (μ v).toReal ≤ b) :
    (μ s).toReal ≤ a + b := by
  have hmeasure : μ s ≤ μ u + μ v :=
    (measure_mono_ae hs).trans (measure_union_le u v)
  have hfinite : μ u + μ v ≠ ⊤ :=
    ENNReal.add_ne_top.mpr ⟨measure_ne_top μ u, measure_ne_top μ v⟩
  have hreal := ENNReal.toReal_mono hfinite hmeasure
  rw [ENNReal.toReal_add (measure_ne_top μ u) (measure_ne_top μ v)] at hreal
  linarith

lemma integral_matrix_le_of_ae_le {n : ℕ} {μ : Measure Ω}
    {A B : Ω → Matrix (Fin n) (Fin n) ℝ}
    (hA_int : Integrable A μ) (hB_int : Integrable B μ)
    (hle : ∀ᵐ ω ∂μ, A ω ≤ B ω) :
    (∫ ω, A ω ∂μ) ≤ ∫ ω, B ω ∂μ := by
  rw [Matrix.le_iff]
  have hdiff_int : Integrable (fun ω => B ω - A ω) μ := hB_int.sub hA_int
  have hdiff_psd : ∀ᵐ ω ∂μ, (B ω - A ω).PosSemidef :=
    hle.mono fun _ hω => Matrix.le_iff.mp hω
  have hpsd := Matrix.integral_posSemidef_of_ae hdiff_int hdiff_psd
  rwa [integral_sub hB_int hA_int] at hpsd

/--
The individual matrix exponential in HDP Lemma 5.4.10 is integrable when the summand is
integrable and a.s. norm-bounded.

This is a finite-dimensional compactness argument: the continuous map `A ↦ exp(λA)` is bounded
on the closed ball containing the summand a.s.
-/
lemma matrixBernsteinIndividual_exp_integrable_of_integrable_norm_bound {n N : ℕ}
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {K l : ℝ} (i : Fin N)
    (h_X_int : Integrable (X i) μ)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ := by
  letI : NormedAlgebra ℚ (Matrix (Fin n) (Fin n) ℝ) :=
    NormedAlgebra.restrictScalars ℚ ℝ (Matrix (Fin n) (Fin n) ℝ)
  let s : Set (Matrix (Fin n) (Fin n) ℝ) :=
    Metric.closedBall (0 : Matrix (Fin n) (Fin n) ℝ) K
  have hs : IsCompact s := by
    simpa [s] using
      (isCompact_closedBall (0 : Matrix (Fin n) (Fin n) ℝ) K)
  have hcont :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => NormedSpace.exp (l • A)) :=
    NormedSpace.exp_continuous.comp (continuous_const_smul l)
  rcases hs.exists_bound_of_continuousOn hcont.continuousOn with ⟨C, hC⟩
  have hmeas :
      AEStronglyMeasurable (fun ω => NormedSpace.exp (l • X i ω)) μ :=
    hcont.comp_aestronglyMeasurable h_X_int.aestronglyMeasurable
  exact Integrable.of_bound hmeas C
    (h_norm_bound.mono fun ω hω => by
      exact hC (X i ω) (by simpa [s, Metric.mem_closedBall, dist_eq_norm] using hω))

/-- A nonempty a.s. norm-bounded family has a nonnegative bound. -/
lemma matrixBernstein_norm_bound_nonneg_of_index {n N : ℕ} {μ : Measure Ω}
    [IsProbabilityMeasure μ]
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {K : ℝ} (i : Fin N)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    0 ≤ K := by
  rcases h_norm_bound.exists with ⟨ω, hω⟩
  exact (norm_nonneg (X i ω)).trans hω

/--
The trace integrand in one application of HDP Lemma 5.4.9 is integrable under the usual
integrability and a.s. norm-bound hypotheses.
-/
lemma matrixBernstein_lieb_trace_integrable_of_integrable_norm_bound {n N : ℕ}
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {K l : ℝ}
    (H : Matrix (Fin n) (Fin n) ℝ) (i : Fin N)
    (h_X_int : Integrable (X i) μ)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    Integrable
      (fun ω => Matrix.trace (NormedSpace.exp (H + l • X i ω))) μ := by
  letI : NormedAlgebra ℚ (Matrix (Fin n) (Fin n) ℝ) :=
    NormedAlgebra.restrictScalars ℚ ℝ (Matrix (Fin n) (Fin n) ℝ)
  let s : Set (Matrix (Fin n) (Fin n) ℝ) :=
    Metric.closedBall (0 : Matrix (Fin n) (Fin n) ℝ) K
  have hs : IsCompact s := by
    simpa [s] using
      (isCompact_closedBall (0 : Matrix (Fin n) (Fin n) ℝ) K)
  have hinside :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => H + l • A) :=
    continuous_const.add (continuous_const_smul l)
  have hexp :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => NormedSpace.exp A) :=
    NormedSpace.exp_continuous
  have htrace :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => Matrix.trace A) := by
    fun_prop
  have hcont :
      Continuous
        (fun A : Matrix (Fin n) (Fin n) ℝ =>
          Matrix.trace (NormedSpace.exp (H + l • A))) :=
    htrace.comp (hexp.comp hinside)
  rcases hs.exists_bound_of_continuousOn hcont.continuousOn with ⟨C, hC⟩
  have hmeas :
      AEStronglyMeasurable
        (fun ω => Matrix.trace (NormedSpace.exp (H + l • X i ω))) μ :=
    hcont.comp_aestronglyMeasurable h_X_int.aestronglyMeasurable
  exact Integrable.of_bound hmeas C
    (h_norm_bound.mono fun ω hω => by
      exact hC (X i ω) (by simpa [s, Metric.mem_closedBall, dist_eq_norm] using hω))

/--
One unconditional application of HDP Lemma 5.4.9 to a single matrix Bernstein summand.

This is the one-step Lieb estimate used repeatedly in HDP Step 2 before conditioning is used to
turn it into the full recursion.
-/
theorem matrixBernstein_lieb_one_step_of_integrable_norm_bound {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (H : Matrix (Fin n) (Fin n) ℝ) (i : Fin N)
    (hH : H.IsHermitian)
    (h_X_int : Integrable (X i) μ)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    ∫ ω, Matrix.trace (NormedSpace.exp (H + l • X i ω)) ∂μ ≤
      Matrix.trace
        (NormedSpace.exp (H + CFC.log (matrixBernsteinIndividualMgf μ X l i))) := by
  have hl_self : IsSelfAdjoint l := by
    rw [isSelfAdjoint_iff]
    simp
  have hZ_symmetric : ∀ᵐ ω ∂μ, (l • X i ω).IsHermitian :=
    hX_symmetric.mono fun _ hω => hω.smul hl_self
  have h_trace_int :
      Integrable
        (fun ω => Matrix.trace (NormedSpace.exp (H + l • X i ω))) μ :=
    matrixBernstein_lieb_trace_integrable_of_integrable_norm_bound
      (X := X) (K := K) (l := l) H i h_X_int h_norm_bound
  have h_exp_int :
      Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ :=
    matrixBernsteinIndividual_exp_integrable_of_integrable_norm_bound
      (X := X) (K := K) (l := l) i h_X_int h_norm_bound
  simpa [matrixBernsteinIndividualMgf] using
    lieb_inequality_random_matrices_hdp_5_4_9
      (μ := μ) (H := H) (Z := fun ω => l • X i ω)
      hH hZ_symmetric h_trace_int h_exp_int

/-- Individual matrix MGFs are positive definite when the summand is Hermitian a.e. -/
lemma matrixBernsteinIndividualMgf_posDef_of_ae_isHermitian {n N : ℕ}
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {l : ℝ} (i : Fin N)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_exp_int : Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ) :
    (matrixBernsteinIndividualMgf μ X l i).PosDef := by
  have hl_self : IsSelfAdjoint l := by
    rw [isSelfAdjoint_iff]
    simp
  have hZ_symmetric : ∀ᵐ ω ∂μ, (l • X i ω).IsHermitian :=
    hX_symmetric.mono fun _ hω => hω.smul hl_self
  simpa [matrixBernsteinIndividualMgf] using
    integral_exp_posDef_of_ae_isHermitian
      (μ := μ) (Z := fun ω => l • X i ω) hZ_symmetric h_exp_int

/--
Finite product-space HDP Step 2 with an initial Hermitian shift.

This is the product-law version of the recursive argument leading to HDP (5.17): split off one
coordinate, apply the induction hypothesis to the tail with shift `H + λX₀`, then apply
Lemma 5.4.9 once more to `X₀`.
-/
theorem matrixBernstein_lieb_recursion_pi_with_shift {n N : ℕ}
    (μs : Fin N → Measure (Matrix (Fin n) (Fin n) ℝ))
    [∀ i, IsProbabilityMeasure (μs i)] {K l : ℝ}
    (H : Matrix (Fin n) (Fin n) ℝ) (hH : H.IsHermitian)
    (h_integrable :
      ∀ i, Integrable (fun A : Matrix (Fin n) (Fin n) ℝ => A) (μs i))
    (h_symmetric : ∀ i, ∀ᵐ A ∂μs i, A.IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ A ∂μs i, ‖A‖ ≤ K)
    (htrace_int :
      Integrable
        (fun p : Fin N → Matrix (Fin n) (Fin n) ℝ =>
          Matrix.trace
            (NormedSpace.exp (H + l • matrixBernsteinSum matrixBernsteinPiFamily p)))
        (Measure.pi μs)) :
    ∫ p : Fin N → Matrix (Fin n) (Fin n) ℝ,
        Matrix.trace
          (NormedSpace.exp (H + l • matrixBernsteinSum matrixBernsteinPiFamily p))
        ∂Measure.pi μs ≤
      Matrix.trace
        (NormedSpace.exp
          (H + matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l)) := by
  induction N generalizing H with
  | zero =>
      simp [matrixBernsteinSum, matrixBernsteinLogMgfSum]
  | succ N ih =>
      let Mat : Type := Matrix (Fin n) (Fin n) ℝ
      let μtail : Fin N → Measure Mat := fun i => μs i.succ
      let tailLog : Mat :=
        matrixBernsteinLogMgfSum (Measure.pi μtail) matrixBernsteinPiFamily l
      let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (N + 1) => Mat) (0 : Fin (N + 1))
      let ν : Measure (Mat × (Fin N → Mat)) := (μs 0).prod (Measure.pi μtail)
      let g : Mat × (Fin N → Mat) → ℝ :=
        fun z =>
          Matrix.trace
            (NormedSpace.exp
              (H + l • (z.1 + matrixBernsteinSum matrixBernsteinPiFamily z.2)))
      haveI : ∀ i, IsProbabilityMeasure (μtail i) := by
        intro i
        dsimp [μtail]
        infer_instance
      have hmp : MeasurePreserving e (Measure.pi μs) ν := by
        simpa [e, ν, μtail] using
          (measurePreserving_piFinSuccAbove (μ := μs) (0 : Fin (N + 1)))
      have hlog_split :
          matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l =
            CFC.log (∫ A : Mat, NormedSpace.exp (l • A) ∂μs 0) + tailLog := by
        dsimp [tailLog]
        rw [matrixBernsteinLogMgfSum_piFamily (μs := μs) (l := l),
          matrixBernsteinLogMgfSum_piFamily (μs := μtail) (l := l),
          Fin.sum_univ_succ]
      have htail_integrable :
          ∀ i, Integrable (fun A : Mat => A) (μtail i) := by
        intro i
        exact h_integrable i.succ
      have htail_symmetric : ∀ i, ∀ᵐ A ∂μtail i, A.IsHermitian := by
        intro i
        exact h_symmetric i.succ
      have htail_norm_bound : ∀ i, ∀ᵐ A ∂μtail i, ‖A‖ ≤ K := by
        intro i
        exact h_norm_bound i.succ
      have htailLog : tailLog.IsHermitian := by
        classical
        dsimp [tailLog, matrixBernsteinLogMgfSum]
        refine Finset.induction_on (Finset.univ : Finset (Fin N)) ?base ?step
        · simp
        · intro i s his hs
          rw [Finset.sum_insert his]
          have hlog :
              (CFC.log
                (matrixBernsteinIndividualMgf (Measure.pi μtail)
                  matrixBernsteinPiFamily l i)).IsHermitian :=
            (show IsSelfAdjoint
                (CFC.log
                  (matrixBernsteinIndividualMgf (Measure.pi μtail)
                    matrixBernsteinPiFamily l i)) from
              IsSelfAdjoint.log).isHermitian
          exact hlog.add hs
      have hg_meas : AEStronglyMeasurable g ν := by
        letI : NormedAlgebra ℚ Mat := NormedAlgebra.restrictScalars ℚ ℝ Mat
        have hsum_tail :
            Continuous (fun q : Fin N → Mat => matrixBernsteinSum matrixBernsteinPiFamily q) := by
          unfold matrixBernsteinSum matrixBernsteinPiFamily
          fun_prop
        have hsum :
            Continuous (fun z : Mat × (Fin N → Mat) =>
              z.1 + matrixBernsteinSum matrixBernsteinPiFamily z.2) :=
          continuous_fst.add (hsum_tail.comp continuous_snd)
        have hinside :
            Continuous (fun z : Mat × (Fin N → Mat) =>
              H + l • (z.1 + matrixBernsteinSum matrixBernsteinPiFamily z.2)) :=
          continuous_const.add ((continuous_const_smul l).comp hsum)
        have hexp :
            Continuous (fun z : Mat × (Fin N → Mat) =>
              NormedSpace.exp
                (H + l • (z.1 + matrixBernsteinSum matrixBernsteinPiFamily z.2))) :=
          NormedSpace.exp_continuous.comp hinside
        have htrace : Continuous (fun A : Mat => Matrix.trace A) := by
          fun_prop
        exact (htrace.comp hexp).aestronglyMeasurable
      have hcomp_int : Integrable (g ∘ e) (Measure.pi μs) := by
        convert htrace_int using 1
        ext p
        simp [g, e, Mat, matrixBernsteinSum, Fin.sum_univ_succ, Fin.tail_def, smul_add]
      have hν_int : Integrable g ν :=
        (hmp.integrable_comp hg_meas).1 hcomp_int
      have hsection_int :
          ∀ᵐ A ∂μs 0, Integrable (fun q : Fin N → Mat => g (A, q)) (Measure.pi μtail) :=
        hν_int.prod_right_ae
      have hright_int :
          Integrable
            (fun A : Mat =>
              Matrix.trace (NormedSpace.exp (H + tailLog + l • A))) (μs 0) := by
        let Xid : Fin 1 → Mat → Mat := fun _ A => A
        exact matrixBernstein_lieb_trace_integrable_of_integrable_norm_bound
          (μ := μs 0) (X := Xid) (K := K) (l := l)
          (H := H + tailLog) (i := 0)
          (by simpa [Xid] using h_integrable 0)
          (by simpa [Xid] using h_norm_bound 0)
      have hinner_le :
          (fun A : Mat => ∫ q : Fin N → Mat, g (A, q) ∂Measure.pi μtail) ≤ᵐ[μs 0]
            fun A : Mat => Matrix.trace (NormedSpace.exp (H + tailLog + l • A)) := by
        filter_upwards [h_symmetric 0, hsection_int] with A hA hA_int
        have hshift : (H + l • A).IsHermitian := by
          have hl_self : IsSelfAdjoint l := by
            rw [isSelfAdjoint_iff]
            simp
          exact hH.add (hA.smul hl_self)
        have htail :=
          ih (H := H + l • A) (μs := μtail) hshift
            htail_integrable htail_symmetric htail_norm_bound
            (by
              refine hA_int.congr ?_
              filter_upwards with q
              simp [g, smul_add, add_assoc])
        simpa [g, tailLog, smul_add, add_assoc, add_comm, add_left_comm] using htail
      have hfirst :
          ∫ A : Mat, ∫ q : Fin N → Mat, g (A, q) ∂Measure.pi μtail ∂μs 0 ≤
            ∫ A : Mat,
              Matrix.trace (NormedSpace.exp (H + tailLog + l • A)) ∂μs 0 :=
        integral_mono_ae hν_int.integral_prod_left hright_int hinner_le
      have hsecond :
          ∫ A : Mat,
              Matrix.trace (NormedSpace.exp (H + tailLog + l • A)) ∂μs 0 ≤
            Matrix.trace
              (NormedSpace.exp
                (H + tailLog + CFC.log (∫ A : Mat, NormedSpace.exp (l • A) ∂μs 0))) := by
        let Xid : Fin 1 → Mat → Mat := fun _ A => A
        have hraw :=
          matrixBernstein_lieb_one_step_of_integrable_norm_bound
            (μ := μs 0) (X := Xid) (K := K) (l := l)
            (H := H + tailLog) (i := (0 : Fin 1))
            (hH.add htailLog)
            (by simpa [Xid] using h_integrable 0)
            (by simpa [Xid] using h_symmetric 0)
            (by simpa [Xid] using h_norm_bound 0)
        simpa [Xid, matrixBernsteinIndividualMgf, Mat] using hraw
      calc
        ∫ p : Fin (N + 1) → Mat,
            Matrix.trace
              (NormedSpace.exp
                (H + l • matrixBernsteinSum matrixBernsteinPiFamily p)) ∂Measure.pi μs =
            ∫ z : Mat × (Fin N → Mat), g z ∂ν := by
              calc
                ∫ p : Fin (N + 1) → Mat,
                    Matrix.trace
                      (NormedSpace.exp
                        (H + l • matrixBernsteinSum matrixBernsteinPiFamily p))
                    ∂Measure.pi μs =
                    ∫ p : Fin (N + 1) → Mat, g (e p) ∂Measure.pi μs := by
                      simp [g, e, Mat, matrixBernsteinSum, Fin.sum_univ_succ,
                        Fin.tail_def, smul_add]
                _ = ∫ z : Mat × (Fin N → Mat), g z ∂ν := hmp.integral_comp' g
        _ = ∫ A : Mat, ∫ q : Fin N → Mat, g (A, q) ∂Measure.pi μtail ∂μs 0 := by
              rw [integral_prod g hν_int]
        _ ≤ ∫ A : Mat,
              Matrix.trace (NormedSpace.exp (H + tailLog + l • A)) ∂μs 0 := hfirst
        _ ≤ Matrix.trace
              (NormedSpace.exp
                (H + tailLog + CFC.log (∫ A : Mat, NormedSpace.exp (l • A) ∂μs 0))) :=
              hsecond
        _ = Matrix.trace
              (NormedSpace.exp
                (H + matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l)) := by
              rw [hlog_split]
              congr 2
              ac_rfl

/--
Arbitrary finite-family HDP Step 2 recursion.

The proof transports the independent family to its joint law on the finite product space, applies
the product-space recursion, and transports the trace MGF and individual matrix MGFs back.
-/
theorem matrixBernstein_lieb_recursion_of_independent {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (h_independent : MatrixBernsteinIndependent X μ)
    (h_integrable : ∀ i, Integrable (X i) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (htrace_int :
      Integrable
        (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))) μ) :
    matrixBernsteinTraceMgf μ X l ≤
      Matrix.trace (NormedSpace.exp (matrixBernsteinLogMgfSum μ X l)) := by
  let Mat : Type := Matrix (Fin n) (Fin n) ℝ
  letI : NormedAlgebra ℚ Mat := NormedAlgebra.restrictScalars ℚ ℝ Mat
  let μs : Fin N → Measure Mat := fun i => μ.map (X i)
  let φ : Ω → Fin N → Mat := fun ω i => X i ω
  let g : (Fin N → Mat) → ℝ :=
    fun p => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum matrixBernsteinPiFamily p))
  have hX_ae : ∀ i, AEMeasurable (X i) μ := by
    intro i
    exact (h_integrable i).aestronglyMeasurable.aemeasurable
  haveI : ∀ i, IsProbabilityMeasure (μs i) := by
    intro i
    dsimp [μs]
    exact Measure.isProbabilityMeasure_map (hX_ae i)
  have hφ : AEMeasurable φ μ :=
    aemeasurable_pi_lambda _ hX_ae
  have hmap_eq : μ.map φ = Measure.pi μs := by
    unfold MatrixBernsteinIndependent at h_independent
    simpa [φ, μs, Mat] using
      (iIndepFun.map_fun_eq_pi_map hX_ae h_independent)
  have h_integrable_map :
      ∀ i, Integrable (fun A : Mat => A) (μs i) := by
    intro i
    have h_id : AEStronglyMeasurable (fun A : Mat => A) (μ.map (X i)) :=
      aestronglyMeasurable_id
    simpa [μs, Function.comp_def] using
      (integrable_map_measure h_id (hX_ae i)).2 (h_integrable i)
  have h_symmetric_map : ∀ i, ∀ᵐ A ∂μs i, A.IsHermitian := by
    intro i
    simpa [μs] using
      (ae_map_iff (hX_ae i) (p := fun A : Mat => A.IsHermitian)
        measurableSet_matrix_isHermitian).2 (hX_symmetric i)
  have h_norm_bound_map : ∀ i, ∀ᵐ A ∂μs i, ‖A‖ ≤ K := by
    intro i
    simpa [μs] using
      (ae_map_iff (hX_ae i) (p := fun A : Mat => ‖A‖ ≤ K)
        measurableSet_matrix_norm_le).2 (h_norm_bound i)
  have hg : AEStronglyMeasurable g (μ.map φ) := by
    letI : NormedAlgebra ℚ Mat := NormedAlgebra.restrictScalars ℚ ℝ Mat
    have hsum :
        Continuous (fun p : Fin N → Mat => matrixBernsteinSum matrixBernsteinPiFamily p) := by
      unfold matrixBernsteinSum matrixBernsteinPiFamily
      fun_prop
    have hinside :
        Continuous (fun p : Fin N → Mat =>
          l • matrixBernsteinSum matrixBernsteinPiFamily p) :=
      (continuous_const_smul l).comp hsum
    have hexp :
        Continuous (fun p : Fin N → Mat =>
          NormedSpace.exp (l • matrixBernsteinSum matrixBernsteinPiFamily p)) :=
      NormedSpace.exp_continuous.comp hinside
    have htrace : Continuous (fun A : Mat => Matrix.trace A) := by
      fun_prop
    exact (htrace.comp hexp).aestronglyMeasurable
  have htrace_comp_int : Integrable (g ∘ φ) μ := by
    convert htrace_int using 1
    ext ω
    simp [g, φ, matrixBernsteinSum]
  have hmap_int : Integrable g (μ.map φ) :=
    (integrable_map_measure hg hφ).2 htrace_comp_int
  have hprod_int : Integrable g (Measure.pi μs) := by
    rwa [← hmap_eq]
  have hprod_int_shift :
      Integrable
        (fun p : Fin N → Mat =>
          Matrix.trace
            (NormedSpace.exp ((0 : Mat) + l • matrixBernsteinSum matrixBernsteinPiFamily p)))
        (Measure.pi μs) := by
    simpa [g] using hprod_int
  have hprod_bound :
      matrixBernsteinTraceMgf (Measure.pi μs) matrixBernsteinPiFamily l ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l)) := by
    simpa [matrixBernsteinTraceMgf] using
      (matrixBernstein_lieb_recursion_pi_with_shift
        (μs := μs) (K := K) (l := l)
        (H := (0 : Mat)) (by simp)
        h_integrable_map h_symmetric_map h_norm_bound_map hprod_int_shift)
  have hmgf_eq :
      ∀ i,
        matrixBernsteinIndividualMgf (Measure.pi μs) matrixBernsteinPiFamily l i =
          matrixBernsteinIndividualMgf μ X l i := by
    intro i
    let f : Mat → Mat := fun A => NormedSpace.exp (l • A)
    have hf_meas : AEStronglyMeasurable f (μ.map (X i)) :=
      (NormedSpace.exp_continuous.comp (continuous_const_smul l)).aestronglyMeasurable
    calc
      matrixBernsteinIndividualMgf (Measure.pi μs) matrixBernsteinPiFamily l i =
          ∫ A : Mat, NormedSpace.exp (l • A) ∂μs i := by
            rw [matrixBernsteinIndividualMgf_piFamily]
      _ = ∫ ω, NormedSpace.exp (l • X i ω) ∂μ := by
            simpa [μs, f, Mat] using integral_map (hX_ae i) hf_meas
      _ = matrixBernsteinIndividualMgf μ X l i := rfl
  have hlog_eq :
      matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l =
        matrixBernsteinLogMgfSum μ X l := by
    unfold matrixBernsteinLogMgfSum
    exact Finset.sum_congr rfl fun i _ => by rw [hmgf_eq i]
  have htrace_eq :
      matrixBernsteinTraceMgf μ X l =
        matrixBernsteinTraceMgf (Measure.pi μs) matrixBernsteinPiFamily l := by
    calc
      matrixBernsteinTraceMgf μ X l = ∫ ω, g (φ ω) ∂μ := by
          simp [matrixBernsteinTraceMgf, g, φ, matrixBernsteinSum]
      _ = ∫ p : Fin N → Mat, g p ∂μ.map φ := by
          exact (integral_map hφ hg).symm
      _ = ∫ p : Fin N → Mat, g p ∂Measure.pi μs := by
          rw [hmap_eq]
      _ = matrixBernsteinTraceMgf (Measure.pi μs) matrixBernsteinPiFamily l := by
          simp [matrixBernsteinTraceMgf, g]
  calc
    matrixBernsteinTraceMgf μ X l =
        matrixBernsteinTraceMgf (Measure.pi μs) matrixBernsteinPiFamily l := htrace_eq
    _ ≤ Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum (Measure.pi μs) matrixBernsteinPiFamily l)) :=
        hprod_bound
    _ = Matrix.trace (NormedSpace.exp (matrixBernsteinLogMgfSum μ X l)) := by
        rw [hlog_eq]

/--
Expectation-level form of HDP Lemma 5.4.10 from a pointwise quadratic envelope.

The hypotheses isolate the two deterministic pieces of the scalar-to-matrix MGF proof:
`exp(λX) ⪯ 1 + λX + gX²` pointwise a.e., and
`1 + g E X² ⪯ exp(g E X²)`.
-/
theorem matrixBernsteinIndividualMgf_le_of_quadratic_envelope {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (i : Fin N) {l g : ℝ}
    (h_exp_int : Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : Integrable (X i) μ)
    (h_X2_int : Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∫ ω, X i ω ∂μ = 0)
    (h_envelope :
      ∀ᵐ ω ∂μ,
        NormedSpace.exp (l • X i ω) ≤
          1 + l • X i ω + g • (X i ω) ^ 2)
    (hone_add_le_exp :
      1 + g • (∫ ω, (X i ω) ^ 2 ∂μ) ≤
        NormedSpace.exp (g • ∫ ω, (X i ω) ^ 2 ∂μ)) :
    matrixBernsteinIndividualMgf μ X l i ≤
      NormedSpace.exp (g • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  let Q : Ω → Matrix (Fin n) (Fin n) ℝ :=
    fun ω => 1 + l • X i ω + g • (X i ω) ^ 2
  have hQ_int : Integrable Q μ := by
    exact ((integrable_const (c := (1 : Matrix (Fin n) (Fin n) ℝ))).add
      (h_X_int.smul l)).add (h_X2_int.smul g)
  have hintegral_le :
      (∫ ω, NormedSpace.exp (l • X i ω) ∂μ) ≤ ∫ ω, Q ω ∂μ :=
    integral_matrix_le_of_ae_le h_exp_int hQ_int
      (by simpa [Q] using h_envelope)
  have hQ_integral :
      (∫ ω, Q ω ∂μ) = 1 + g • ∫ ω, (X i ω) ^ 2 ∂μ := by
    calc
      (∫ ω, Q ω ∂μ)
          = ∫ ω, (1 : Matrix (Fin n) (Fin n) ℝ) + l • X i ω +
              g • (X i ω) ^ 2 ∂μ := by
            rfl
      _ = (∫ ω, (1 : Matrix (Fin n) (Fin n) ℝ) + l • X i ω ∂μ) +
            ∫ ω, g • (X i ω) ^ 2 ∂μ := by
            simpa using
              (integral_add
                (μ := μ)
                (f := fun ω =>
                  (1 : Matrix (Fin n) (Fin n) ℝ) + l • X i ω)
                (g := fun ω => g • (X i ω) ^ 2)
                ((integrable_const (c := (1 : Matrix (Fin n) (Fin n) ℝ))).add
                  (h_X_int.smul l)) (h_X2_int.smul g))
      _ = (∫ ω, (1 : Matrix (Fin n) (Fin n) ℝ) ∂μ) +
            ∫ ω, l • X i ω ∂μ + ∫ ω, g • (X i ω) ^ 2 ∂μ := by
            simpa [add_assoc] using
              congrArg (fun A => A + ∫ ω, g • (X i ω) ^ 2 ∂μ)
                (integral_add
                  (μ := μ)
                  (f := fun _ => (1 : Matrix (Fin n) (Fin n) ℝ))
                  (g := fun ω => l • X i ω)
                  (integrable_const (c := (1 : Matrix (Fin n) (Fin n) ℝ)))
                  (h_X_int.smul l))
      _ = 1 + l • (∫ ω, X i ω ∂μ) + g • ∫ ω, (X i ω) ^ 2 ∂μ := by
            rw [integral_const, probReal_univ, one_smul, integral_smul, integral_smul]
      _ = 1 + g • ∫ ω, (X i ω) ^ 2 ∂μ := by
            rw [h_mean_zero]
            simp
  calc
    matrixBernsteinIndividualMgf μ X l i =
        ∫ ω, NormedSpace.exp (l • X i ω) ∂μ := rfl
    _ ≤ ∫ ω, Q ω ∂μ := hintegral_le
    _ = 1 + g • ∫ ω, (X i ω) ^ 2 ∂μ := hQ_integral
    _ ≤ NormedSpace.exp (g • ∫ ω, (X i ω) ^ 2 ∂μ) := hone_add_le_exp

omit [MeasurableSpace Ω] in
/-- Matrix version of the scalar bound `1 + x ≤ exp x` on positive semidefinite matrices. -/
lemma one_add_le_exp_of_posSemidef {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ}
    (hA : A.PosSemidef) :
    1 + A ≤ NormedSpace.exp A := by
  have hscalar : ∀ x : ℝ, |x| ≤ ‖A‖ → 1 + x ≤ Real.exp x := by
    intro x _
    simpa [add_comm] using Real.add_one_le_exp x
  have hcfc :
      hA.1.cfc (fun x : ℝ => 1 + x) ≤ hA.1.cfc Real.exp :=
    Matrix.loewner_functionalCalculus_order_hdp hA.1 (norm_nonneg A) hscalar le_rfl
  have hleft :
      hA.1.cfc (fun x : ℝ => 1 + x) = 1 + A := by
    rw [← hA.1.cfc_eq (fun x : ℝ => 1 + x)]
    rw [cfc_const_add (1 : ℝ) (fun x : ℝ => x) A]
    rw [cfc_id' ℝ A]
    simp
  have hright :
      hA.1.cfc Real.exp = NormedSpace.exp A := by
    rw [← hA.1.cfc_eq Real.exp]
    exact CFC.real_exp_eq_normedSpace_exp (a := A)
      (ha := (show IsSelfAdjoint A from hA.1))
  simpa [hleft, hright] using hcfc

omit [MeasurableSpace Ω] in
lemma one_add_smul_le_exp_smul_of_posSemidef {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} {g : ℝ}
    (hA : A.PosSemidef) (hg : 0 ≤ g) :
    1 + g • A ≤ NormedSpace.exp (g • A) :=
  one_add_le_exp_of_posSemidef (hA.smul hg)

/--
Expectation-level HDP Lemma 5.4.10 from a pointwise quadratic envelope and the usual
positive-rate condition.
-/
theorem matrixBernsteinIndividualMgf_le_of_quadratic_envelope_nonneg {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (i : Fin N) {l g : ℝ}
    (hg : 0 ≤ g)
    (h_exp_int : Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : Integrable (X i) μ)
    (h_X2_int : Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_envelope :
      ∀ᵐ ω ∂μ,
        NormedSpace.exp (l • X i ω) ≤
          1 + l • X i ω + g • (X i ω) ^ 2) :
    matrixBernsteinIndividualMgf μ X l i ≤
      NormedSpace.exp (g • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  have hV_pos :
      (∫ ω, (X i ω) ^ 2 ∂μ).PosSemidef :=
    Matrix.integral_posSemidef_of_ae h_X2_int
      (matrixBernsteinVarianceSummand_sq_posSemidef_ae i hX_symmetric)
  exact matrixBernsteinIndividualMgf_le_of_quadratic_envelope μ X i
    h_exp_int h_X_int h_X2_int h_mean_zero h_envelope
    (one_add_smul_le_exp_smul_of_posSemidef hV_pos hg)

omit [MeasurableSpace Ω] in
/--
Scalar-to-matrix quadratic envelope used in HDP Lemma 5.4.10.

If the scalar inequality holds on the spectral interval controlled by `‖A‖ ≤ K`, then functional
calculus gives `exp(λA) ⪯ 1 + λA + gA²`.
-/
lemma matrix_exp_le_quadratic_of_spectral_envelope {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} {K l g : ℝ}
    (hA : A.IsHermitian) (hK : 0 ≤ K) (hAnorm : ‖A‖ ≤ K)
    (hscalar : ∀ x : ℝ, |x| ≤ K → Real.exp (l * x) ≤ 1 + l * x + g * x ^ 2) :
    NormedSpace.exp (l • A) ≤ 1 + l • A + g • A ^ 2 := by
  have hcfc :
      hA.cfc (fun x : ℝ => Real.exp (l * x)) ≤
        hA.cfc (fun x : ℝ => 1 + l * x + g * x ^ 2) :=
    Matrix.loewner_functionalCalculus_order_hdp hA hK hscalar hAnorm
  have hleft :
      hA.cfc (fun x : ℝ => Real.exp (l * x)) = NormedSpace.exp (l • A) := by
    rw [← hA.cfc_eq (fun x : ℝ => Real.exp (l * x))]
    rw [cfc_comp_const_mul (r := l) (f := Real.exp) (a := A)
      (ha := (show IsSelfAdjoint A from hA))]
    exact CFC.real_exp_eq_normedSpace_exp (a := l • A)
      (ha := (show IsSelfAdjoint (l • A) from
        hA.smul (by rw [isSelfAdjoint_iff]; simp)))
  have hright :
      hA.cfc (fun x : ℝ => 1 + l * x + g * x ^ 2) =
        1 + l • A + g • A ^ 2 := by
    rw [← hA.cfc_eq (fun x : ℝ => 1 + l * x + g * x ^ 2)]
    rw [cfc_add (a := A) (f := fun x : ℝ => 1 + l * x) (g := fun x : ℝ => g * x ^ 2)]
    · rw [cfc_const_add (1 : ℝ) (fun x : ℝ => l * x) A
        (ha := (show IsSelfAdjoint A from hA))]
      rw [cfc_const_mul_id (r := l) (a := A) (ha := (show IsSelfAdjoint A from hA))]
      rw [cfc_const_mul (r := g) (f := fun x : ℝ => x ^ 2) (a := A)]
      rw [cfc_pow_id (R := ℝ) A 2 (ha := (show IsSelfAdjoint A from hA))]
      simp [pow_two]
  simpa [hleft, hright] using hcfc

/-- A.e. pointwise quadratic envelope for one matrix Bernstein summand. -/
lemma matrixBernstein_quadratic_envelope_ae_of_spectral_envelope {n N : ℕ}
    {μ : Measure Ω} {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (i : Fin N) {K l g : ℝ} (hK : 0 ≤ K)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (hscalar : ∀ x : ℝ, |x| ≤ K → Real.exp (l * x) ≤ 1 + l * x + g * x ^ 2) :
    ∀ᵐ ω ∂μ,
      NormedSpace.exp (l • X i ω) ≤
        1 + l • X i ω + g • (X i ω) ^ 2 := by
  filter_upwards [hX_symmetric, h_norm_bound] with ω hω_symm hω_norm
  exact matrix_exp_le_quadratic_of_spectral_envelope hω_symm hK hω_norm hscalar

/--
HDP Lemma 5.4.10 reduced to its scalar quadratic envelope.

For a fixed summand, once the scalar inequality
`exp(λx) ≤ 1 + λx + g x²` is known on `[-K,K]`, the matrix MGF satisfies
`E exp(λXᵢ) ⪯ exp(g E Xᵢ²)`.
-/
theorem matrixBernsteinIndividualMgf_le_of_scalar_quadratic_envelope_nonneg {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (i : Fin N) {K l g : ℝ}
    (hg : 0 ≤ g) (hK : 0 ≤ K)
    (h_exp_int : Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : Integrable (X i) μ)
    (h_X2_int : Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (hscalar : ∀ x : ℝ, |x| ≤ K → Real.exp (l * x) ≤ 1 + l * x + g * x ^ 2) :
    matrixBernsteinIndividualMgf μ X l i ≤
      NormedSpace.exp (g • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  exact matrixBernsteinIndividualMgf_le_of_quadratic_envelope_nonneg μ X i
    hg h_exp_int h_X_int h_X2_int h_mean_zero hX_symmetric
    (matrixBernstein_quadratic_envelope_ae_of_spectral_envelope i hK hX_symmetric
      h_norm_bound hscalar)

/-- HDP Lemma 5.4.10 specialized to the Bernstein rate `g(λ)`, modulo the scalar envelope. -/
theorem matrixBernsteinIndividualMgf_le_of_mgfRate_scalar_envelope {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (i : Fin N) {K l : ℝ}
    (hden : 0 ≤ 1 - |l| * K / 3) (hK : 0 ≤ K)
    (h_exp_int : Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : Integrable (X i) μ)
    (h_X2_int : Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (hscalar :
      ∀ x : ℝ, |x| ≤ K →
        Real.exp (l * x) ≤ 1 + l * x + matrixBernsteinMgfRate K l * x ^ 2) :
    matrixBernsteinIndividualMgf μ X l i ≤
      NormedSpace.exp
        (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  exact matrixBernsteinIndividualMgf_le_of_scalar_quadratic_envelope_nonneg
    μ X i (matrixBernsteinMgfRate_nonneg hden) hK
    h_exp_int h_X_int h_X2_int h_mean_zero hX_symmetric h_norm_bound hscalar

/-- HDP Lemma 5.4.10 for one summand, with the scalar Bernstein envelope discharged. -/
theorem matrixBernsteinIndividualMgf_le_of_mgfRate {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (i : Fin N) {K l : ℝ}
    (hden : 0 < 1 - |l| * K / 3) (hK : 0 ≤ K)
    (h_exp_int : Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : Integrable (X i) μ)
    (h_X2_int : Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    matrixBernsteinIndividualMgf μ X l i ≤
      NormedSpace.exp
        (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  exact matrixBernsteinIndividualMgf_le_of_mgfRate_scalar_envelope μ X i
    hden.le hK h_exp_int h_X_int h_X2_int h_mean_zero hX_symmetric h_norm_bound
    (fun x hx => matrixBernstein_scalar_mgfRate_envelope (K := K) (l := l) hden hx)

/--
All-summands version of the individual MGF bound at the HDP Bernstein rate, modulo the scalar
envelope.
-/
theorem matrixBernsteinIndividualMgf_bounds_of_mgfRate_scalar_envelope {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (hden : 0 ≤ 1 - |l| * K / 3) (hK : 0 ≤ K)
    (h_exp_int : ∀ i, Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : ∀ i, Integrable (X i) μ)
    (h_X2_int : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (hscalar :
      ∀ x : ℝ, |x| ≤ K →
        Real.exp (l * x) ≤ 1 + l * x + matrixBernsteinMgfRate K l * x ^ 2) :
    ∀ i,
      matrixBernsteinIndividualMgf μ X l i ≤
        NormedSpace.exp
          (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  intro i
  exact matrixBernsteinIndividualMgf_le_of_mgfRate_scalar_envelope μ X i
    hden hK (h_exp_int i) (h_X_int i) (h_X2_int i) (h_mean_zero i)
    (hX_symmetric i) (h_norm_bound i) hscalar

/-- All-summands version of HDP Lemma 5.4.10, with the scalar envelope discharged. -/
theorem matrixBernsteinIndividualMgf_bounds_of_mgfRate {n N : ℕ}
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (hden : 0 < 1 - |l| * K / 3) (hK : 0 ≤ K)
    (h_exp_int : ∀ i, Integrable (fun ω => NormedSpace.exp (l • X i ω)) μ)
    (h_X_int : ∀ i, Integrable (X i) μ)
    (h_X2_int : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    ∀ i,
      matrixBernsteinIndividualMgf μ X l i ≤
        NormedSpace.exp
          (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ) := by
  intro i
  exact matrixBernsteinIndividualMgf_le_of_mgfRate μ X i
    hden hK (h_exp_int i) (h_X_int i) (h_X2_int i) (h_mean_zero i)
    (hX_symmetric i) (h_norm_bound i)

omit [MeasurableSpace Ω] in
/-- Matrix exponentials of Hermitian real matrices have nonnegative trace. -/
lemma trace_exp_nonneg_of_isHermitian {n : ℕ} {A : Matrix (Fin n) (Fin n) ℝ}
    (hA : A.IsHermitian) :
    0 ≤ Matrix.trace (NormedSpace.exp A) :=
  (Matrix.exp_posDef_of_isHermitian hA).posSemidef.trace_nonneg

/-- The trace integrand in the matrix Laplace transform is nonnegative a.e. -/
lemma matrixBernstein_trace_exp_nonneg_ae_of_sum_isHermitian_ae {n N : ℕ}
    {μ : Measure Ω} {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (hXsum : ∀ᵐ ω ∂μ, (matrixBernsteinSum X ω).IsHermitian) (l : ℝ) :
    0 ≤ᵐ[μ] fun ω =>
      Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) := by
  filter_upwards [hXsum] with ω hω
  have hl : IsSelfAdjoint l := by
    rw [isSelfAdjoint_iff]
    simp
  exact trace_exp_nonneg_of_isHermitian (hω.smul hl)

/--
HDP Step 1 for the upper spectral tail at a fixed value of `λ`.

If `tr exp(λS)` pointwise dominates `exp(λt)` on the event `λmax(S) ≥ t` and its expectation is
bounded by `B`, then Markov's inequality gives
`P{λmax(S) ≥ t} ≤ exp(-λt) B`.
-/
theorem matrixBernstein_upperTail_le_of_trace_mgf_bound {n N : ℕ}
    (hn : 0 < n) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {l t B : ℝ}
    (htrace_nonneg : 0 ≤ᵐ[μ]
      fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)))
    (htrace_int : Integrable
      (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))) μ)
    (hdom : ∀ᵐ ω ∂μ,
      ω ∈ matrixBernsteinUpperTailEvent hn X t →
        Real.exp (l * t) ≤
          Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)))
    (hmgf : matrixBernsteinTraceMgf μ X l ≤ B) :
    (μ (matrixBernsteinUpperTailEvent hn X t)).toReal ≤ Real.exp (-(l * t)) * B := by
  let f : Ω → ℝ := fun ω =>
    Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))
  let ε : ℝ := Real.exp (l * t)
  let s : Set Ω := matrixBernsteinUpperTailEvent hn X t
  let r : Set Ω := {ω | ε ≤ f ω}
  have hεpos : 0 < ε := Real.exp_pos _
  have hs_le_r : μ s ≤ μ r := by
    apply measure_mono_ae
    filter_upwards [hdom] with ω hω hωs
    exact hω hωs
  have hs_real_le_r : (μ s).toReal ≤ μ.real r := by
    rw [measureReal_def]
    exact ENNReal.toReal_mono (measure_ne_top μ r) hs_le_r
  have hmarkov : ε * μ.real r ≤ ∫ ω, f ω ∂μ :=
    mul_meas_ge_le_integral_of_nonneg (μ := μ) (f := f)
      (by simpa [f] using htrace_nonneg) (by simpa [f] using htrace_int) ε
  have hs_mul_le : ε * (μ s).toReal ≤ ∫ ω, f ω ∂μ := by
    have hmul := mul_le_mul_of_nonneg_left hs_real_le_r (le_of_lt hεpos)
    exact hmul.trans hmarkov
  have hs_le_integral : (μ s).toReal ≤ ε⁻¹ * (∫ ω, f ω ∂μ) :=
    (le_inv_mul_iff₀ hεpos).mpr hs_mul_le
  have hs_le_B : (μ s).toReal ≤ ε⁻¹ * B := by
    exact hs_le_integral.trans
      (mul_le_mul_of_nonneg_left (by simpa [f, matrixBernsteinTraceMgf] using hmgf)
        (inv_nonneg.mpr (le_of_lt hεpos)))
  simpa [s, ε, f, Real.exp_neg, mul_comm, mul_left_comm, mul_assoc] using hs_le_B

/--
The optimized one-sided upper-tail step in HDP Theorem 5.4.1, assuming the trace-MGF estimate at
the optimizer.

The positive-variance hypothesis is the nondegenerate case in which the optimizer lies strictly
inside the scalar MGF domain.
-/
theorem matrixBernstein_upperTail_le_of_trace_mgf_bound_at_optimizingLambda {n N : ℕ}
    (hn : 0 < n) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {σ_sq K t : ℝ}
    (hσ : 0 < σ_sq) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (htrace_nonneg : 0 ≤ᵐ[μ]
      fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda σ_sq K t • matrixBernsteinSum X ω)))
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda σ_sq K t • matrixBernsteinSum X ω))) μ)
    (hdom : ∀ᵐ ω ∂μ,
      ω ∈ matrixBernsteinUpperTailEvent hn X t →
        Real.exp (matrixBernsteinOptimizingLambda σ_sq K t * t) ≤
          Matrix.trace
            (NormedSpace.exp
              (matrixBernsteinOptimizingLambda σ_sq K t • matrixBernsteinSum X ω)))
    (hmgf :
      matrixBernsteinTraceMgf μ X (matrixBernsteinOptimizingLambda σ_sq K t) ≤
        (n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) *
              σ_sq)) :
    (μ (matrixBernsteinUpperTailEvent hn X t)).toReal ≤
      matrixBernsteinOneSidedTailBound n σ_sq K t := by
  have htail :=
    matrixBernstein_upperTail_le_of_trace_mgf_bound
      hn X μ
      (l := matrixBernsteinOptimizingLambda σ_sq K t)
      (t := t)
      (B := (n : ℝ) *
        Real.exp
          (matrixBernsteinMgfRate K (matrixBernsteinOptimizingLambda σ_sq K t) * σ_sq))
      htrace_nonneg htrace_int hdom hmgf
  rwa [matrixBernstein_optimizedTraceMgfBound_eq_oneSided n hσ hK ht] at htail

/--
The final two-sided union-bound step in HDP Theorem 5.4.1, with the event inclusion required only
almost surely.
-/
theorem matrixBernstein_norm_tail_le_of_one_sided_eigenvalue_tails_ae {n N : ℕ}
    (hn : 0 < n) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {σ_sq K t : ℝ} (ht : 0 ≤ t)
    (hsubset : matrixBernsteinNormEventAESubsetUpperLower μ hn X)
    (hupper :
      (μ (matrixBernsteinUpperTailEvent hn X t)).toReal ≤
        matrixBernsteinOneSidedTailBound n σ_sq K t)
    (hlower :
      (μ (matrixBernsteinLowerTailEvent hn X t)).toReal ≤
        matrixBernsteinOneSidedTailBound n σ_sq K t) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n σ_sq K t := by
  have hprob :=
    probability_real_le_of_ae_subset_union (μ := μ)
      (s := {ω | ‖matrixBernsteinSum X ω‖ ≥ t})
      (u := matrixBernsteinUpperTailEvent hn X t)
      (v := matrixBernsteinLowerTailEvent hn X t)
      (a := matrixBernsteinOneSidedTailBound n σ_sq K t)
      (b := matrixBernsteinOneSidedTailBound n σ_sq K t)
      (hsubset t ht) hupper hlower
  calc
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal
        ≤ matrixBernsteinOneSidedTailBound n σ_sq K t +
          matrixBernsteinOneSidedTailBound n σ_sq K t := hprob
    _ = matrixBernsteinTailBound n σ_sq K t := by
          rw [matrixBernsteinTailBound_eq_two_mul_oneSided]
          ring

@[simp]
lemma matrixLargestEigenvalue_of_isHermitian {n : ℕ} (hn : 0 < n)
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) :
    matrixLargestEigenvalue hn A =
      hA.eigenvaluesToEuclideanLin ⟨0, by simpa using hn⟩ := by
  unfold matrixLargestEigenvalue
  rw [dif_pos hA]

omit [MeasurableSpace Ω] in
/-- The trace exponential dominates the exponential of the largest eigenvalue. -/
lemma exp_largestEigenvalue_le_trace_exp {n : ℕ} (hn : 0 < n)
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) :
    Real.exp (matrixLargestEigenvalue hn A) ≤ Matrix.trace (NormedSpace.exp A) := by
  let i0 : Fin (Fintype.card (Fin n)) := ⟨0, by simpa using hn⟩
  have htrace : Matrix.trace (NormedSpace.exp A) = Matrix.spectralTrace hA Real.exp := by
    rw [← CFC.real_exp_eq_normedSpace_exp (a := A)
      (ha := (show IsSelfAdjoint A from hA))]
    exact Matrix.trace_cfc_eq_spectralTrace hA Real.exp
  rw [htrace]
  rw [matrixLargestEigenvalue_of_isHermitian hn hA]
  unfold Matrix.spectralTrace
  exact Finset.single_le_sum (fun _ _ => le_of_lt (Real.exp_pos _)) (Finset.mem_univ i0)

omit [MeasurableSpace Ω] in
/-- Every Hermitian eigenvalue is bounded above by the L2 operator norm. -/
lemma eigenvaluesToEuclideanLin_le_norm {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian)
    (i : Fin (Fintype.card (Fin n))) :
    hA.eigenvaluesToEuclideanLin i ≤ ‖A‖ := by
  let E := EuclideanSpace ℝ (Fin n)
  let L : E →ₗ[ℝ] E := A.toEuclideanLin
  let T : E →L[ℝ] E := L.toContinuousLinearMap
  let hLsymm : L.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hA
  let x : E := hLsymm.eigenvectorBasis finrank_euclideanSpace i
  have hrayleigh :
      T.rayleighQuotient x = hA.eigenvaluesToEuclideanLin i := by
    simpa [LinearMap.rayleighQuotient_toContinuousLinearMap, T, L,
      Matrix.IsHermitian.eigenvaluesToEuclideanLin, x] using
      hLsymm.rayleighQuotient_eigenvectorBasis finrank_euclideanSpace i
  have hle_abs :
      hA.eigenvaluesToEuclideanLin i ≤ |hA.eigenvaluesToEuclideanLin i| :=
    le_abs_self _
  have hle_T : |hA.eigenvaluesToEuclideanLin i| ≤ ‖T‖ := by
    simpa [hrayleigh] using T.rayleighQuotient_le_norm x
  rw [Matrix.l2_opNorm_def]
  change hA.eigenvaluesToEuclideanLin i ≤ ‖T‖
  exact hle_abs.trans hle_T

omit [MeasurableSpace Ω] in
/-- `tr exp(A) ≤ n exp(‖A‖)` for real Hermitian `n × n` matrices. -/
lemma trace_exp_le_card_mul_exp_norm_of_isHermitian {n : ℕ}
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) :
    Matrix.trace (NormedSpace.exp A) ≤ (n : ℝ) * Real.exp ‖A‖ := by
  have htrace :
      Matrix.trace (NormedSpace.exp A) = Matrix.spectralTrace hA Real.exp := by
    rw [← CFC.real_exp_eq_normedSpace_exp (a := A)
      (ha := (show IsSelfAdjoint A from hA))]
    exact Matrix.trace_cfc_eq_spectralTrace hA Real.exp
  rw [htrace]
  unfold Matrix.spectralTrace
  calc
    (∑ i : Fin (Fintype.card (Fin n)), Real.exp (hA.eigenvaluesToEuclideanLin i))
        ≤ ∑ _i : Fin (Fintype.card (Fin n)), Real.exp ‖A‖ := by
          exact Finset.sum_le_sum fun i _ =>
            Real.exp_le_exp.mpr (eigenvaluesToEuclideanLin_le_norm hA i)
    _ = (n : ℝ) * Real.exp ‖A‖ := by
          simp

omit [MeasurableSpace Ω] in
/--
The deterministic dimension-factor step used after the matrix MGF recursion:
`tr exp(gV) ≤ n exp(g‖V‖)` for `g ≥ 0` and Hermitian `V`.
-/
lemma trace_exp_smul_le_card_mul_exp_mul_norm_of_isHermitian {n : ℕ}
    {V : Matrix (Fin n) (Fin n) ℝ} (hV : V.IsHermitian) {g : ℝ} (hg : 0 ≤ g) :
    Matrix.trace (NormedSpace.exp (g • V)) ≤ (n : ℝ) * Real.exp (g * ‖V‖) := by
  have hg_self : IsSelfAdjoint g := by
    rw [isSelfAdjoint_iff]
    simp
  have hle := trace_exp_le_card_mul_exp_norm_of_isHermitian (hV.smul hg_self)
  simpa [norm_smul, Real.norm_eq_abs, abs_of_nonneg hg] using hle

/--
The deterministic dimension-factor reduction in the trace-MGF proof.

The remaining hypothesis is the Lieb-recursion/individual-MGF estimate
`E tr exp(λS) ≤ tr exp(g(λ) V)`, where `V = ∑ᵢ E Xᵢ²`.
-/
theorem matrixBernsteinTraceMgf_le_of_trace_exp_variance_bound {n N : ℕ}
    (μ : Measure Ω) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (hV : (matrixBernsteinVarianceMatrix μ X).IsHermitian)
    (hg : 0 ≤ matrixBernsteinMgfRate K l)
    (htrace_bound :
      matrixBernsteinTraceMgf μ X l ≤
        Matrix.trace
          (NormedSpace.exp (matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ X))) :
    matrixBernsteinTraceMgf μ X l ≤
      (n : ℝ) * Real.exp (matrixBernsteinMgfRate K l * matrixBernsteinVarianceProxy μ X) := by
  exact htrace_bound.trans
    (by
      simpa [matrixBernsteinVarianceProxy] using
        trace_exp_smul_le_card_mul_exp_mul_norm_of_isHermitian
          (n := n) hV (g := matrixBernsteinMgfRate K l) hg)

/-- The log-MGF sum from HDP (5.17) is Hermitian when each individual matrix MGF is positive. -/
lemma matrixBernsteinLogMgfSum_isHermitian {n N : ℕ} (μ : Measure Ω)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) (l : ℝ)
    (hindividual_pos : ∀ i, (matrixBernsteinIndividualMgf μ X l i).PosDef) :
    (matrixBernsteinLogMgfSum μ X l).IsHermitian := by
  classical
  unfold matrixBernsteinLogMgfSum
  refine Finset.induction_on (Finset.univ : Finset (Fin N)) ?base ?step
  · simp
  · intro i s his hs
    rw [Finset.sum_insert his]
    have hM : IsSelfAdjoint (matrixBernsteinIndividualMgf μ X l i) :=
      (hindividual_pos i).isHermitian
    have hlog : (CFC.log (matrixBernsteinIndividualMgf μ X l i)).IsHermitian :=
      (show IsSelfAdjoint (CFC.log (matrixBernsteinIndividualMgf μ X l i)) from
        IsSelfAdjoint.log).isHermitian
    exact hlog.add hs

/--
HDP Step 3, deterministic log-sum form.

If every individual matrix MGF satisfies
`E exp(λXᵢ) ⪯ exp(g(λ) E Xᵢ²)`, then the logarithmic sum appearing after the Lieb recursion is
bounded by `g(λ) ∑ᵢ E Xᵢ²`.
-/
theorem matrixBernsteinLogMgfSum_le_mgfRate_smul_varianceMatrix {n N : ℕ}
    (μ : Measure Ω) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (hindividual_pos : ∀ i, (matrixBernsteinIndividualMgf μ X l i).PosDef)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (hindividual_mgf_bound :
      ∀ i,
        matrixBernsteinIndividualMgf μ X l i ≤
          NormedSpace.exp
            (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ)) :
    matrixBernsteinLogMgfSum μ X l ≤
      matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ X := by
  classical
  have hterm :
      ∀ i,
        CFC.log (matrixBernsteinIndividualMgf μ X l i) ≤
          matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ := by
    intro i
    have hVi : (∫ ω, (X i ω) ^ 2 ∂μ).IsHermitian :=
      matrixBernsteinVarianceSummand_integral_isHermitian i
        (h_square_integrable i) (hX_symmetric i)
    have hg_self : IsSelfAdjoint (matrixBernsteinMgfRate K l) := by
      rw [isSelfAdjoint_iff]
      simp
    have hlog_le :
        CFC.log (matrixBernsteinIndividualMgf μ X l i) ≤
          CFC.log
            (NormedSpace.exp
              (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ)) :=
      Matrix.cfc_log_le_log_of_le_posDef
        (X := matrixBernsteinIndividualMgf μ X l i)
        (Y := NormedSpace.exp
          (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ))
        (hindividual_pos i) (hindividual_mgf_bound i)
    have hlog_exp :
        CFC.log
            (NormedSpace.exp
              (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ)) =
          matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ := by
      exact CFC.log_exp
        (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ)
        (ha := (show IsSelfAdjoint
          (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ) from
          hVi.smul hg_self))
    rwa [hlog_exp] at hlog_le
  have hsum :
      (∑ i : Fin N, CFC.log (matrixBernsteinIndividualMgf μ X l i)) ≤
        ∑ i : Fin N, matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ :=
    Finset.sum_le_sum fun i _ => hterm i
  calc
    matrixBernsteinLogMgfSum μ X l =
        ∑ i : Fin N, CFC.log (matrixBernsteinIndividualMgf μ X l i) := by
          rfl
    _ ≤ ∑ i : Fin N, matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ := hsum
    _ = matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ X := by
          simp [matrixBernsteinVarianceMatrix, Finset.smul_sum]

/--
Combines the Lieb-recursion trace bound (HDP (5.17)) with the individual matrix MGF Loewner
bounds (HDP Lemma 5.4.10) to obtain the variance-matrix trace bound.
-/
theorem matrixBernsteinTraceMgf_le_of_lieb_recursion_and_individual_mgf_bounds {n N : ℕ}
    (μ : Measure Ω) (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ) {K l : ℝ}
    (hindividual_pos : ∀ i, (matrixBernsteinIndividualMgf μ X l i).PosDef)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (hlieb_recursion :
      matrixBernsteinTraceMgf μ X l ≤
        Matrix.trace (NormedSpace.exp (matrixBernsteinLogMgfSum μ X l)))
    (hindividual_mgf_bound :
      ∀ i,
        matrixBernsteinIndividualMgf μ X l i ≤
          NormedSpace.exp
            (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ)) :
    matrixBernsteinTraceMgf μ X l ≤
      Matrix.trace
        (NormedSpace.exp (matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ X)) := by
  have hlog_herm :
      (matrixBernsteinLogMgfSum μ X l).IsHermitian :=
    matrixBernsteinLogMgfSum_isHermitian μ X l hindividual_pos
  have hV : (matrixBernsteinVarianceMatrix μ X).IsHermitian :=
    matrixBernsteinVarianceMatrix_isHermitian_ae h_square_integrable hX_symmetric
  have hg_self : IsSelfAdjoint (matrixBernsteinMgfRate K l) := by
    rw [isSelfAdjoint_iff]
    simp
  have hlog_le :
      matrixBernsteinLogMgfSum μ X l ≤
        matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ X :=
    matrixBernsteinLogMgfSum_le_mgfRate_smul_varianceMatrix μ X
      hindividual_pos h_square_integrable hX_symmetric hindividual_mgf_bound
  exact hlieb_recursion.trans
    (by
      simpa using
        Matrix.trace_exp_monotone_of_le hlog_herm (hV.smul hg_self) hlog_le)

omit [MeasurableSpace Ω] in
/-- The largest eigenvalue of `l • A` is at least `l` times the largest eigenvalue of `A`. -/
lemma mul_largestEigenvalue_le_largestEigenvalue_smul {n : ℕ} (hn : 0 < n)
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) (l : ℝ) :
    l * matrixLargestEigenvalue hn A ≤ matrixLargestEigenvalue hn (l • A) := by
  classical
  let E := EuclideanSpace ℝ (Fin n)
  let L : E →ₗ[ℝ] E := A.toEuclideanLin
  let B : Matrix (Fin n) (Fin n) ℝ := l • A
  let LB : E →ₗ[ℝ] E := B.toEuclideanLin
  let i0 : Fin (Fintype.card (Fin n)) := ⟨0, by simpa using hn⟩
  let hLsymm : L.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hA
  have hl_self : IsSelfAdjoint l := by
    rw [isSelfAdjoint_iff]
    simp
  have hB : B.IsHermitian := hA.smul hl_self
  let hBsymm : LB.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hB
  let x0 : E := hLsymm.eigenvectorBasis finrank_euclideanSpace i0
  have hx0 : x0 ≠ 0 := by
    simpa [x0] using (hLsymm.eigenvectorBasis finrank_euclideanSpace).toBasis.ne_zero i0
  have htopB : hBsymm.trailingEigenSubspace finrank_euclideanSpace i0 = ⊤ := by
    apply Submodule.eq_top_of_finrank_eq
    rw [hBsymm.finrank_trailingEigenSubspace finrank_euclideanSpace i0]
    rw [finrank_euclideanSpace]
    simp [i0]
  have hxmem : x0 ∈ hBsymm.trailingEigenSubspace finrank_euclideanSpace i0 := by
    rw [htopB]
    trivial
  have hle := hBsymm.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace
    finrank_euclideanSpace i0 hxmem hx0
  have hle_eig : LinearMap.rayleighQuotient LB x0 ≤ hB.eigenvaluesToEuclideanLin i0 := by
    simpa [Matrix.IsHermitian.eigenvaluesToEuclideanLin, hBsymm] using hle
  have hle' : LinearMap.rayleighQuotient LB x0 ≤ matrixLargestEigenvalue hn B := by
    simpa [matrixLargestEigenvalue_of_isHermitian hn hB, i0] using hle_eig
  have hrayleighB : LinearMap.rayleighQuotient LB x0 = l * LinearMap.rayleighQuotient L x0 := by
    unfold LinearMap.rayleighQuotient
    simp [LB, B, L, inner_smul_left]
    ring
  have hrayleighA : LinearMap.rayleighQuotient L x0 = hA.eigenvaluesToEuclideanLin i0 := by
    simpa [L, x0, Matrix.IsHermitian.eigenvaluesToEuclideanLin] using
      hLsymm.rayleighQuotient_eigenvectorBasis finrank_euclideanSpace i0
  calc
    l * matrixLargestEigenvalue hn A = l * hA.eigenvaluesToEuclideanLin i0 := by
      rw [matrixLargestEigenvalue_of_isHermitian hn hA]
    _ = LinearMap.rayleighQuotient LB x0 := by rw [hrayleighB, hrayleighA]
    _ ≤ matrixLargestEigenvalue hn B := hle'
    _ = matrixLargestEigenvalue hn (l • A) := rfl

omit [MeasurableSpace Ω] in
/-- Pointwise spectral domination used in HDP Step 1. -/
lemma exp_mul_le_trace_exp_smul_of_largestEigenvalue_ge {n : ℕ} (hn : 0 < n)
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) {l t : ℝ}
    (hl : 0 ≤ l) (hAt : matrixLargestEigenvalue hn A ≥ t) :
    Real.exp (l * t) ≤ Matrix.trace (NormedSpace.exp (l • A)) := by
  have hlA : l * t ≤ matrixLargestEigenvalue hn (l • A) :=
    (mul_le_mul_of_nonneg_left hAt hl).trans
      (mul_largestEigenvalue_le_largestEigenvalue_smul hn hA l)
  have hl_self : IsSelfAdjoint l := by
    rw [isSelfAdjoint_iff]
    simp
  exact (Real.exp_le_exp.mpr hlA).trans
    (exp_largestEigenvalue_le_trace_exp hn (hA.smul hl_self))

/--
For a real Hermitian matrix, the operator norm is bounded by the larger of the top eigenvalues of
`A` and `-A`.
-/
lemma matrix_norm_le_max_largestEigenvalue_neg {n : ℕ} (hn : 0 < n)
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) :
    ‖A‖ ≤ max (matrixLargestEigenvalue hn A) (matrixLargestEigenvalue hn (-A)) := by
  classical
  let E := EuclideanSpace ℝ (Fin n)
  let L : E →ₗ[ℝ] E := A.toEuclideanLin
  let T : E →L[ℝ] E := L.toContinuousLinearMap
  let Lneg : E →ₗ[ℝ] E := (-A).toEuclideanLin
  let Tneg : E →L[ℝ] E := Lneg.toContinuousLinearMap
  let i0 : Fin (Fintype.card (Fin n)) := ⟨0, by simpa using hn⟩
  let hLsymm : L.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hA
  let hLnegSymm : Lneg.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.mpr hA.neg
  let lmax : ℝ := hA.eigenvaluesToEuclideanLin i0
  let lminNeg : ℝ := hA.neg.eigenvaluesToEuclideanLin i0
  let M : ℝ := max lmax lminNeg
  have htop : hLsymm.trailingEigenSubspace finrank_euclideanSpace i0 = ⊤ := by
    apply Submodule.eq_top_of_finrank_eq
    rw [hLsymm.finrank_trailingEigenSubspace finrank_euclideanSpace i0]
    rw [finrank_euclideanSpace]
    simp [i0]
  have htop_neg : hLnegSymm.trailingEigenSubspace finrank_euclideanSpace i0 = ⊤ := by
    apply Submodule.eq_top_of_finrank_eq
    rw [hLnegSymm.finrank_trailingEigenSubspace finrank_euclideanSpace i0]
    rw [finrank_euclideanSpace]
    simp [i0]
  have hle_nonzero : ∀ {x : E}, x ≠ 0 → T.rayleighQuotient x ≤ lmax := by
    intro x hx
    have hxmem : x ∈ hLsymm.trailingEigenSubspace finrank_euclideanSpace i0 := by
      rw [htop]
      trivial
    have hle := hLsymm.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace
      finrank_euclideanSpace i0 hxmem hx
    convert hle using 1
    · simp [LinearMap.rayleighQuotient_toContinuousLinearMap, T, L]
    · simp [lmax, Matrix.IsHermitian.eigenvaluesToEuclideanLin]
  have hle_neg_nonzero : ∀ {x : E}, x ≠ 0 → -T.rayleighQuotient x ≤ lminNeg := by
    intro x hx
    have hxmem : x ∈ hLnegSymm.trailingEigenSubspace finrank_euclideanSpace i0 := by
      rw [htop_neg]
      trivial
    have hle := hLnegSymm.rayleighQuotient_le_eigenvalues_of_mem_trailingEigenSubspace
      finrank_euclideanSpace i0 hxmem hx
    have hle' : Tneg.rayleighQuotient x ≤ lminNeg := by
      convert hle using 1
      · simp [LinearMap.rayleighQuotient_toContinuousLinearMap, Tneg, Lneg]
      · simp [lminNeg, Matrix.IsHermitian.eigenvaluesToEuclideanLin, Lneg]
    have hTneg_eq : Tneg = -T := by
      ext y
      simp [Tneg, Lneg, T, L]
    rw [hTneg_eq, ContinuousLinearMap.rayleighQuotient_neg_apply] at hle'
    exact hle'
  have hM_nonneg : 0 ≤ M := by
    let x0 : E := hLsymm.eigenvectorBasis finrank_euclideanSpace i0
    have hx0 : x0 ≠ 0 := by
      simpa [x0] using (hLsymm.eigenvectorBasis finrank_euclideanSpace).toBasis.ne_zero i0
    have h1 := hle_nonzero (x := x0) hx0
    have h2 := hle_neg_nonzero (x := x0) hx0
    by_cases hx : 0 ≤ T.rayleighQuotient x0
    · exact hx.trans (h1.trans (le_max_left _ _))
    · have hx' : 0 ≤ -T.rayleighQuotient x0 := by linarith
      exact hx'.trans (h2.trans (le_max_right _ _))
  have hbound : ∀ x : E, |T.rayleighQuotient x| ≤ M := by
    intro x
    by_cases hx : x = 0
    · simp [hx, ContinuousLinearMap.rayleighQuotient_apply_zero, hM_nonneg, M]
    · have hupper : T.rayleighQuotient x ≤ M := (hle_nonzero hx).trans (le_max_left _ _)
      have hnegupper : -T.rayleighQuotient x ≤ M := (hle_neg_nonzero hx).trans
        (le_max_right _ _)
      exact abs_le.mpr ⟨by linarith, hupper⟩
  have hTsymm : (T : E →ₗ[ℝ] E).IsSymmetric := by
    simpa [T, L]
  have hnormT : ‖T‖ ≤ M := by
    rw [ContinuousLinearMap.norm_eq_iSup_rayleighQuotient T hTsymm]
    exact ciSup_le (by simpa [M] using hbound)
  rw [Matrix.l2_opNorm_def]
  change ‖T‖ ≤ max (matrixLargestEigenvalue hn A) (matrixLargestEigenvalue hn (-A))
  simpa [matrixLargestEigenvalue_of_isHermitian hn hA,
    matrixLargestEigenvalue_of_isHermitian hn hA.neg, T, L, M, lmax, lminNeg] using hnormT

/-- Pointwise form of HDP (5.15) for a Hermitian matrix. -/
lemma matrix_norm_ge_imp_largestEigenvalue_or_neg {n : ℕ} (hn : 0 < n)
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) {t : ℝ}
    (hAt : ‖A‖ ≥ t) :
    matrixLargestEigenvalue hn A ≥ t ∨ matrixLargestEigenvalue hn (-A) ≥ t := by
  have htmax : t ≤ max (matrixLargestEigenvalue hn A) (matrixLargestEigenvalue hn (-A)) :=
    hAt.trans (matrix_norm_le_max_largestEigenvalue_neg hn hA)
  exact le_max_iff.mp htmax

/-- HDP (5.15), assuming the random sum is Hermitian almost surely. -/
lemma matrixBernsteinNormEventAESubsetUpperLower_of_sum_isHermitian_ae {n N : ℕ}
    {μ : Measure Ω} (hn : 0 < n) {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (hXsum : ∀ᵐ ω ∂μ, (matrixBernsteinSum X ω).IsHermitian) :
    matrixBernsteinNormEventAESubsetUpperLower μ hn X := by
  intro t _ht
  filter_upwards [hXsum] with ω hω hnorm
  exact matrix_norm_ge_imp_largestEigenvalue_or_neg hn hω hnorm

lemma matrixBernsteinSum_isHermitian_ae {n N : ℕ}
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {μ : Measure Ω}
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian) :
    ∀ᵐ ω ∂μ, (matrixBernsteinSum X ω).IsHermitian := by
  classical
  have h_all : ∀ᵐ ω ∂μ, ∀ i, (X i ω).IsHermitian :=
    Filter.eventually_all.mpr hX_symmetric
  filter_upwards [h_all] with ω hω
  unfold matrixBernsteinSum
  refine Finset.induction_on (Finset.univ : Finset (Fin N)) ?base ?step
  · simp
  · intro i s his hs
    rw [Finset.sum_insert his]
    exact (hω i).add hs

/-- The upper-tail event implies the trace-exponential lower bound a.e. -/
lemma matrixBernstein_upperTail_trace_exp_dominates_ae_of_sum_isHermitian_ae {n N : ℕ}
    {μ : Measure Ω} (hn : 0 < n) {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (hXsum : ∀ᵐ ω ∂μ, (matrixBernsteinSum X ω).IsHermitian)
    {l t : ℝ} (hl : 0 ≤ l) :
    ∀ᵐ ω ∂μ,
      ω ∈ matrixBernsteinUpperTailEvent hn X t →
        Real.exp (l * t) ≤
          Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) := by
  filter_upwards [hXsum] with ω hω htail
  exact exp_mul_le_trace_exp_smul_of_largestEigenvalue_ge hn hω hl htail

/-- The upper-tail trace-exponential domination follows from a.e. Hermitian summands. -/
lemma matrixBernstein_upperTail_trace_exp_dominates_ae_of_summands_isHermitian_ae {n N : ℕ}
    {μ : Measure Ω} (hn : 0 < n) {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    {l t : ℝ} (hl : 0 ≤ l) :
    ∀ᵐ ω ∂μ,
      ω ∈ matrixBernsteinUpperTailEvent hn X t →
        Real.exp (l * t) ≤
          Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) :=
  matrixBernstein_upperTail_trace_exp_dominates_ae_of_sum_isHermitian_ae hn
    (matrixBernsteinSum_isHermitian_ae (X := X) (μ := μ) hX_symmetric) hl

/-- The trace Laplace-transform integrand is nonnegative a.e. under a.e. Hermitian summands. -/
lemma matrixBernstein_trace_exp_nonneg_ae_of_summands_isHermitian_ae {n N : ℕ}
    {μ : Measure Ω} {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian) (l : ℝ) :
    0 ≤ᵐ[μ] fun ω =>
      Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) :=
  matrixBernstein_trace_exp_nonneg_ae_of_sum_isHermitian_ae
    (matrixBernsteinSum_isHermitian_ae (X := X) (μ := μ) hX_symmetric) l

/-- The a.s. summand norm bound gives an a.s. norm bound for the finite matrix sum. -/
lemma matrixBernsteinSum_norm_le_ae {n N : ℕ} {μ : Measure Ω}
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {K : ℝ}
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    ∀ᵐ ω ∂μ, ‖matrixBernsteinSum X ω‖ ≤ (N : ℝ) * K := by
  classical
  have h_all : ∀ᵐ ω ∂μ, ∀ i, ‖X i ω‖ ≤ K :=
    Filter.eventually_all.mpr h_norm_bound
  filter_upwards [h_all] with ω hω
  calc
    ‖matrixBernsteinSum X ω‖ = ‖∑ i : Fin N, X i ω‖ := rfl
    _ ≤ ∑ i : Fin N, ‖X i ω‖ := norm_sum_le _ _
    _ ≤ ∑ _i : Fin N, K := Finset.sum_le_sum fun i _ => hω i
    _ = (N : ℝ) * K := by
      simp [Finset.sum_const, nsmul_eq_mul]

/--
The trace-exponential integrand is a.e. bounded when the summands are a.e. Hermitian and
uniformly norm-bounded.
-/
lemma matrixBernstein_trace_exp_abs_le_ae_of_summands_isHermitian_norm_bound {n N : ℕ}
    {μ : Measure Ω} {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {K l : ℝ}
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    ∀ᵐ ω ∂μ,
      |Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))| ≤
        (n : ℝ) * Real.exp (|l| * ((N : ℝ) * K)) := by
  have hsum_symm := matrixBernsteinSum_isHermitian_ae (X := X) (μ := μ) hX_symmetric
  have hsum_norm := matrixBernsteinSum_norm_le_ae (X := X) (μ := μ) h_norm_bound
  filter_upwards [hsum_symm, hsum_norm] with ω hω_symm hω_norm
  have hl : IsSelfAdjoint l := by
    rw [isSelfAdjoint_iff]
    simp
  have hnonneg :
      0 ≤ Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) :=
    trace_exp_nonneg_of_isHermitian (hω_symm.smul hl)
  have htrace_le :
      Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) ≤
        (n : ℝ) * Real.exp ‖l • matrixBernsteinSum X ω‖ :=
    trace_exp_le_card_mul_exp_norm_of_isHermitian (hω_symm.smul hl)
  have hnorm_smul :
      ‖l • matrixBernsteinSum X ω‖ ≤ |l| * ((N : ℝ) * K) := by
    calc
      ‖l • matrixBernsteinSum X ω‖ = |l| * ‖matrixBernsteinSum X ω‖ := by
        simp [norm_smul, Real.norm_eq_abs]
      _ ≤ |l| * ((N : ℝ) * K) :=
        mul_le_mul_of_nonneg_left hω_norm (abs_nonneg l)
  have htrace_bound :
      Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω)) ≤
        (n : ℝ) * Real.exp (|l| * ((N : ℝ) * K)) :=
    htrace_le.trans
      (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hnorm_smul) (Nat.cast_nonneg n))
  simpa [abs_of_nonneg hnonneg] using htrace_bound

/--
The trace Laplace transform in HDP Step 1 is integrable under the theorem's integrability,
a.e. symmetry, and uniform norm-bound hypotheses.
-/
lemma matrixBernstein_trace_exp_integrable_of_summands_integrable_norm_bound_isHermitian_ae
    {n N : ℕ} {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ} {K l : ℝ}
    (h_integrable : ∀ i, Integrable (X i) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    Integrable
      (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))) μ := by
  classical
  have hsum_integrable : Integrable (fun ω => matrixBernsteinSum X ω) μ := by
    unfold matrixBernsteinSum
    exact integrable_finsetSum Finset.univ fun i _ => h_integrable i
  letI : NormedAlgebra ℚ (Matrix (Fin n) (Fin n) ℝ) :=
    NormedAlgebra.restrictScalars ℚ ℝ (Matrix (Fin n) (Fin n) ℝ)
  have hsmul :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => l • A) :=
    continuous_const_smul l
  have hexp :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => NormedSpace.exp A) :=
    NormedSpace.exp_continuous
  have htrace :
      Continuous (fun A : Matrix (Fin n) (Fin n) ℝ => Matrix.trace A) := by
    fun_prop
  have hmeas :
      AEStronglyMeasurable
        (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))) μ :=
    (htrace.comp (hexp.comp hsmul)).comp_aestronglyMeasurable
      hsum_integrable.aestronglyMeasurable
  exact Integrable.of_bound hmeas ((n : ℝ) * Real.exp (|l| * ((N : ℝ) * K)))
    (matrixBernstein_trace_exp_abs_le_ae_of_summands_isHermitian_norm_bound
      (X := X) hX_symmetric h_norm_bound)

/-- HDP (5.15), derived from the a.e. Hermitian summand hypothesis. -/
lemma matrixBernsteinNormEventAESubsetUpperLower_of_summands_isHermitian_ae {n N : ℕ}
    {μ : Measure Ω} (hn : 0 < n) {X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ}
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian) :
    matrixBernsteinNormEventAESubsetUpperLower μ hn X :=
  matrixBernsteinNormEventAESubsetUpperLower_of_sum_isHermitian_ae hn
    (matrixBernsteinSum_isHermitian_ae (X := X) (μ := μ) hX_symmetric)

/-! ## Optimized one-sided tails from trace-MGF estimates -/

/--
The optimized one-sided upper-tail estimate, with the Hermitian-summand hypothesis supplying the
two pointwise/a.e. positivity facts used in HDP Step 1.

After this reduction, the only substantive remaining obligation is the trace-MGF estimate
`E tr exp(λS) ≤ n exp(g(λ) σ²)`.
-/
theorem matrixBernstein_upperTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ)
    (hmgf :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        (n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) *
              matrixBernsteinVarianceProxy μ X)) :
    (μ (matrixBernsteinUpperTailEvent hn X t)).toReal ≤
      matrixBernsteinOneSidedTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  have hlambda_nonneg :
      0 ≤ matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t :=
    matrixBernsteinOptimizingLambda_nonneg
      (matrixBernsteinVarianceProxy_nonneg μ X) hK ht
  exact matrixBernstein_upperTail_le_of_trace_mgf_bound_at_optimizingLambda
    hn X μ hσ hK ht
    (matrixBernstein_trace_exp_nonneg_ae_of_summands_isHermitian_ae
      (μ := μ) hX_symmetric
      (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))
    htrace_int
    (matrixBernstein_upperTail_trace_exp_dominates_ae_of_summands_isHermitian_ae
      (μ := μ) hn hX_symmetric hlambda_nonneg)
    hmgf

/--
The optimized upper-tail estimate from the variance-matrix trace bound
`E tr exp(λS) ≤ tr exp(g(λ) V)`, where `V = ∑ᵢ E Xᵢ²`.

This is the HDP step after the Lieb recursion and before replacing `tr exp(gV)` by
`n exp(g‖V‖)`.
-/
theorem matrixBernstein_upperTail_le_of_trace_exp_variance_bound_at_optimizingLambda
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ)
    (htrace_bound :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) •
              matrixBernsteinVarianceMatrix μ X))) :
    (μ (matrixBernsteinUpperTailEvent hn X t)).toReal ≤
      matrixBernsteinOneSidedTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  have hV : (matrixBernsteinVarianceMatrix μ X).IsHermitian :=
    matrixBernsteinVarianceMatrix_isHermitian_ae h_square_integrable hX_symmetric
  have hg :
      0 ≤ matrixBernsteinMgfRate K
        (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) :=
    matrixBernsteinMgfRate_nonneg
      (le_of_lt
        (matrixBernsteinMgfRate_den_optimizingLambda_pos
          (σ_sq := matrixBernsteinVarianceProxy μ X) hσ hK ht))
  have hmgf :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        (n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) *
              matrixBernsteinVarianceProxy μ X) :=
    matrixBernsteinTraceMgf_le_of_trace_exp_variance_bound
      μ X hV hg htrace_bound
  exact matrixBernstein_upperTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae
    hn X μ hσ hK ht hX_symmetric htrace_int hmgf

/--
The optimized one-sided lower-tail estimate, obtained by applying the previous upper-tail estimate
to the negated summands.
-/
theorem matrixBernstein_lowerTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ)
    (hmgf_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        (n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) *
              matrixBernsteinVarianceProxy μ X)) :
    (μ (matrixBernsteinLowerTailEvent hn X t)).toReal ≤
      matrixBernsteinOneSidedTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let Xneg : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ := fun i ω => -X i ω
  have hXneg_symmetric : ∀ i, ∀ᵐ ω ∂μ, (Xneg i ω).IsHermitian := by
    intro i
    exact (hX_symmetric i).mono fun _ hω => hω.neg
  have hσ_neg : 0 < matrixBernsteinVarianceProxy μ Xneg := by
    simpa [Xneg] using hσ
  have htail :
      (μ (matrixBernsteinUpperTailEvent hn Xneg t)).toReal ≤
        matrixBernsteinOneSidedTailBound n (matrixBernsteinVarianceProxy μ Xneg) K t :=
    matrixBernstein_upperTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae
      hn Xneg μ hσ_neg hK ht hXneg_symmetric
      (by simpa [Xneg] using htrace_int_neg)
      (by simpa [Xneg] using hmgf_neg)
  simpa [Xneg] using htail

/--
The optimized lower-tail estimate from the variance-matrix trace bound for the signed family
`-X`.
-/
theorem matrixBernstein_lowerTail_le_of_trace_exp_variance_bound_at_optimizingLambda
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ)
    (htrace_bound_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) •
              matrixBernsteinVarianceMatrix μ (fun i ω => -X i ω)))) :
    (μ (matrixBernsteinLowerTailEvent hn X t)).toReal ≤
      matrixBernsteinOneSidedTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let Xneg : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ := fun i ω => -X i ω
  have hσ_neg : 0 < matrixBernsteinVarianceProxy μ Xneg := by
    simpa [Xneg] using hσ
  have h_square_integrable_neg : ∀ i, Integrable (fun ω => (Xneg i ω) ^ 2) μ := by
    intro i
    simpa [Xneg, pow_two] using h_square_integrable i
  have hXneg_symmetric : ∀ i, ∀ᵐ ω ∂μ, (Xneg i ω).IsHermitian := by
    intro i
    exact (hX_symmetric i).mono fun _ hω => hω.neg
  have htail :
      (μ (matrixBernsteinUpperTailEvent hn Xneg t)).toReal ≤
        matrixBernsteinOneSidedTailBound n (matrixBernsteinVarianceProxy μ Xneg) K t :=
    matrixBernstein_upperTail_le_of_trace_exp_variance_bound_at_optimizingLambda
      hn Xneg μ hσ_neg hK ht h_square_integrable_neg hXneg_symmetric
      (by simpa [Xneg] using htrace_int_neg)
      (by simpa [Xneg] using htrace_bound_neg)
  simpa [Xneg] using htail

/--
The nondegenerate HDP norm-tail bound after Step 1, assuming the optimized trace-MGF estimates for
both `S` and `-S`.

The hypothesis `0 < σ²` is the nondegenerate branch where HDP's optimizer is strictly inside the
scalar MGF domain.
-/
theorem matrixBernstein_norm_tail_le_of_optimized_trace_mgf_bounds_of_summands_isHermitian_ae
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ)
    (htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ)
    (hmgf :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        (n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) *
              matrixBernsteinVarianceProxy μ X))
    (hmgf_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        (n : ℝ) *
          Real.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) *
              matrixBernsteinVarianceProxy μ X)) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  exact matrixBernstein_norm_tail_le_of_one_sided_eigenvalue_tails_ae
    hn X μ ht
    (matrixBernsteinNormEventAESubsetUpperLower_of_summands_isHermitian_ae hn hX_symmetric)
    (matrixBernstein_upperTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae
      hn X μ hσ hK ht hX_symmetric htrace_int hmgf)
    (matrixBernstein_lowerTail_le_of_optimized_trace_mgf_bound_of_summands_isHermitian_ae
      hn X μ hσ hK ht hX_symmetric htrace_int_neg hmgf_neg)

/--
The nondegenerate HDP norm-tail bound after the Lieb recursion, assuming the two variance-matrix
trace bounds for `S` and `-S`.
-/
theorem matrixBernstein_norm_tail_le_of_trace_exp_variance_bounds_at_optimizingLambda
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ)
    (htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ)
    (htrace_bound :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) •
              matrixBernsteinVarianceMatrix μ X)))
    (htrace_bound_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) •
              matrixBernsteinVarianceMatrix μ (fun i ω => -X i ω)))) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  exact matrixBernstein_norm_tail_le_of_one_sided_eigenvalue_tails_ae
    hn X μ ht
    (matrixBernsteinNormEventAESubsetUpperLower_of_summands_isHermitian_ae hn hX_symmetric)
    (matrixBernstein_upperTail_le_of_trace_exp_variance_bound_at_optimizingLambda
      hn X μ hσ hK ht h_square_integrable hX_symmetric htrace_int htrace_bound)
    (matrixBernstein_lowerTail_le_of_trace_exp_variance_bound_at_optimizingLambda
      hn X μ hσ hK ht h_square_integrable hX_symmetric htrace_int_neg htrace_bound_neg)

/--
The nondegenerate HDP norm-tail bound from the exact remaining Step 2 and Step 3 hypotheses:

* the Lieb-recursion trace bounds for `X` and `-X`, and
* the individual matrix MGF Loewner bounds of HDP Lemma 5.4.10 for `X` and `-X`.
-/
theorem matrixBernstein_norm_tail_le_of_lieb_recursions_and_individual_mgf_bounds_at_optimizingLambda
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ)
    (htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ)
    (hindividual_pos :
      ∀ i,
        (matrixBernsteinIndividualMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) i).PosDef)
    (hindividual_pos_neg :
      ∀ i,
        (matrixBernsteinIndividualMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) i).PosDef)
    (hlieb_recursion :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ X
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))))
    (hlieb_recursion_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ (fun i ω => -X i ω)
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))))
    (hindividual_mgf_bound :
      ∀ i,
        matrixBernsteinIndividualMgf μ X
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) i ≤
          NormedSpace.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) •
              ∫ ω, (X i ω) ^ 2 ∂μ))
    (hindividual_mgf_bound_neg :
      ∀ i,
        matrixBernsteinIndividualMgf μ (fun i ω => -X i ω)
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) i ≤
          NormedSpace.exp
            (matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) •
              ∫ ω, ((-X i ω) ^ 2) ∂μ)) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let l : ℝ := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t
  let Xneg : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ := fun i ω => -X i ω
  have h_square_integrable_neg : ∀ i, Integrable (fun ω => (Xneg i ω) ^ 2) μ := by
    intro i
    simpa [Xneg, pow_two] using h_square_integrable i
  have hXneg_symmetric : ∀ i, ∀ᵐ ω ∂μ, (Xneg i ω).IsHermitian := by
    intro i
    exact (hX_symmetric i).mono fun _ hω => hω.neg
  have htrace_bound :
      matrixBernsteinTraceMgf μ X l ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ X)) :=
    matrixBernsteinTraceMgf_le_of_lieb_recursion_and_individual_mgf_bounds
      μ X hindividual_pos h_square_integrable hX_symmetric
      (by simpa [l] using hlieb_recursion)
      (by simpa [l] using hindividual_mgf_bound)
  have htrace_bound_neg :
      matrixBernsteinTraceMgf μ Xneg l ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinMgfRate K l • matrixBernsteinVarianceMatrix μ Xneg)) :=
    matrixBernsteinTraceMgf_le_of_lieb_recursion_and_individual_mgf_bounds
      μ Xneg
      (by simpa [Xneg, l] using hindividual_pos_neg)
      h_square_integrable_neg hXneg_symmetric
      (by simpa [Xneg, l] using hlieb_recursion_neg)
      (by simpa [Xneg, l] using hindividual_mgf_bound_neg)
  exact matrixBernstein_norm_tail_le_of_trace_exp_variance_bounds_at_optimizingLambda
    hn X μ hσ hK ht h_square_integrable hX_symmetric htrace_int htrace_int_neg
    (by simpa [l] using htrace_bound)
    (by simpa [Xneg, l] using htrace_bound_neg)

/--
The nondegenerate HDP norm-tail bound after the Lieb recursion and the scalar Bernstein envelope.

This discharges the individual matrix-MGF hypotheses in
`matrixBernstein_norm_tail_le_of_lieb_recursions_and_individual_mgf_bounds_at_optimizingLambda`
using the functional-calculus reduction for HDP Lemma 5.4.10. The only remaining Step 3 input is
the scalar inequality
`exp(λx) ≤ 1 + λx + g(λ)x²` on `[-K, K]`.
-/
theorem matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_scalar_envelope_at_optimizingLambda
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (h_integrable : ∀ i, Integrable (X i) μ)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ)
    (htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ)
    (h_exp_int :
      ∀ i,
        Integrable
          (fun ω =>
            NormedSpace.exp
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
                X i ω)) μ)
    (h_exp_int_neg :
      ∀ i,
        Integrable
          (fun ω =>
            NormedSpace.exp
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
                (-X i ω))) μ)
    (hlieb_recursion :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ X
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))))
    (hlieb_recursion_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ (fun i ω => -X i ω)
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))))
    (hscalar :
      ∀ x : ℝ, |x| ≤ K →
        Real.exp (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t * x) ≤
          1 +
            matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t * x +
              matrixBernsteinMgfRate K
                (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) *
                x ^ 2) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let l : ℝ := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t
  let Xneg : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ := fun i ω => -X i ω
  have hden : 0 ≤ 1 - |l| * K / 3 := by
    exact le_of_lt
      (matrixBernsteinMgfRate_den_optimizingLambda_pos
        (σ_sq := matrixBernsteinVarianceProxy μ X) hσ hK ht)
  have hXneg_symmetric : ∀ i, ∀ᵐ ω ∂μ, (Xneg i ω).IsHermitian := by
    intro i
    exact (hX_symmetric i).mono fun _ hω => hω.neg
  have h_integrable_neg : ∀ i, Integrable (Xneg i) μ := by
    intro i
    simpa [Xneg] using (h_integrable i).neg
  have h_square_integrable_neg : ∀ i, Integrable (fun ω => (Xneg i ω) ^ 2) μ := by
    intro i
    simpa [Xneg, pow_two] using h_square_integrable i
  have h_mean_zero_neg : ∀ i, ∫ ω, Xneg i ω ∂μ = 0 := by
    intro i
    simp only [Xneg]
    rw [integral_neg]
    simp [h_mean_zero i]
  have h_norm_bound_neg : ∀ i, ∀ᵐ ω ∂μ, ‖Xneg i ω‖ ≤ K := by
    intro i
    exact (h_norm_bound i).mono fun _ hω => by simpa [Xneg] using hω
  have hindividual_pos :
      ∀ i, (matrixBernsteinIndividualMgf μ X l i).PosDef := by
    intro i
    exact matrixBernsteinIndividualMgf_posDef_of_ae_isHermitian i
      (hX_symmetric i) (by simpa [l] using h_exp_int i)
  have hindividual_pos_neg :
      ∀ i, (matrixBernsteinIndividualMgf μ Xneg l i).PosDef := by
    intro i
    exact matrixBernsteinIndividualMgf_posDef_of_ae_isHermitian i
      (X := Xneg) (hXneg_symmetric i) (by simpa [Xneg, l] using h_exp_int_neg i)
  have hindividual_mgf_bound :
      ∀ i,
        matrixBernsteinIndividualMgf μ X l i ≤
          NormedSpace.exp
            (matrixBernsteinMgfRate K l • ∫ ω, (X i ω) ^ 2 ∂μ) := by
    exact matrixBernsteinIndividualMgf_bounds_of_mgfRate_scalar_envelope
      μ X hden hK
      (by intro i; simpa [l] using h_exp_int i)
      h_integrable h_square_integrable h_mean_zero hX_symmetric h_norm_bound
      (by simpa [l] using hscalar)
  have hindividual_mgf_bound_neg :
      ∀ i,
        matrixBernsteinIndividualMgf μ Xneg l i ≤
          NormedSpace.exp
            (matrixBernsteinMgfRate K l • ∫ ω, (Xneg i ω) ^ 2 ∂μ) := by
    exact matrixBernsteinIndividualMgf_bounds_of_mgfRate_scalar_envelope
      μ Xneg hden hK
      (by intro i; simpa [Xneg, l] using h_exp_int_neg i)
      h_integrable_neg h_square_integrable_neg h_mean_zero_neg hXneg_symmetric
      h_norm_bound_neg (by simpa [Xneg, l] using hscalar)
  exact matrixBernstein_norm_tail_le_of_lieb_recursions_and_individual_mgf_bounds_at_optimizingLambda
    hn X μ hσ hK ht h_square_integrable hX_symmetric htrace_int htrace_int_neg
    (by simpa [l] using hindividual_pos)
    (by simpa [Xneg, l] using hindividual_pos_neg)
    hlieb_recursion hlieb_recursion_neg
    (by simpa [l] using hindividual_mgf_bound)
    (by simpa [Xneg, l] using hindividual_mgf_bound_neg)

/--
The nondegenerate HDP norm-tail bound after the Lieb recursion and HDP Lemma 5.4.10.

The scalar Bernstein envelope and all integrability facts used in the trace and matrix MGFs are
discharged from the theorem hypotheses. The remaining explicit inputs are the two Lieb-recursion
trace estimates.
-/
theorem matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_at_optimizingLambda
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (hK : 0 ≤ K) (ht : 0 ≤ t)
    (h_integrable : ∀ i, Integrable (X i) μ)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (hlieb_recursion :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ X
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))))
    (hlieb_recursion_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ (fun i ω => -X i ω)
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t)))) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let l : ℝ := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t
  let Xneg : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ := fun i ω => -X i ω
  have htrace_int : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum X ω))) μ := by
    exact
      matrixBernstein_trace_exp_integrable_of_summands_integrable_norm_bound_isHermitian_ae
        (X := X) (K := K)
        (l := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t)
        h_integrable hX_symmetric h_norm_bound
  have h_integrable_neg : ∀ i, Integrable (Xneg i) μ := by
    intro i
    simpa [Xneg] using (h_integrable i).neg
  have hXneg_symmetric : ∀ i, ∀ᵐ ω ∂μ, (Xneg i ω).IsHermitian := by
    intro i
    exact (hX_symmetric i).mono fun _ hω => hω.neg
  have h_norm_bound_neg : ∀ i, ∀ᵐ ω ∂μ, ‖Xneg i ω‖ ≤ K := by
    intro i
    exact (h_norm_bound i).mono fun _ hω => by simpa [Xneg] using hω
  have h_exp_int :
      ∀ i,
        Integrable
          (fun ω =>
            NormedSpace.exp
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
                X i ω)) μ := by
    intro i
    exact matrixBernsteinIndividual_exp_integrable_of_integrable_norm_bound
      (X := X) (K := K)
      (l := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t)
      i (h_integrable i) (h_norm_bound i)
  have h_exp_int_neg :
      ∀ i,
        Integrable
          (fun ω =>
            NormedSpace.exp
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
                (-X i ω))) μ := by
    intro i
    have htmp :
        Integrable
          (fun ω => NormedSpace.exp (l • Xneg i ω)) μ :=
      matrixBernsteinIndividual_exp_integrable_of_integrable_norm_bound
        (X := Xneg) (K := K) (l := l) i
        (h_integrable_neg i) (h_norm_bound_neg i)
    simpa [Xneg, l] using htmp
  have htrace_int_neg : Integrable
      (fun ω =>
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t •
              matrixBernsteinSum (fun i ω => -X i ω) ω))) μ := by
    have htmp :
        Integrable
          (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum Xneg ω))) μ :=
      matrixBernstein_trace_exp_integrable_of_summands_integrable_norm_bound_isHermitian_ae
        (X := Xneg) (K := K) (l := l)
        h_integrable_neg hXneg_symmetric h_norm_bound_neg
    simpa [Xneg, l] using htmp
  exact
    matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_scalar_envelope_at_optimizingLambda
      hn X μ hσ hK ht h_integrable h_square_integrable h_mean_zero hX_symmetric
      h_norm_bound htrace_int htrace_int_neg h_exp_int h_exp_int_neg hlieb_recursion
      hlieb_recursion_neg
      (fun x hx =>
        matrixBernstein_scalar_mgfRate_envelope
          (K := K)
          (l := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t)
          (matrixBernsteinMgfRate_den_optimizingLambda_pos
            (σ_sq := matrixBernsteinVarianceProxy μ X) hσ hK ht)
          hx)

/--
Nonempty-index version of the preceding nondegenerate HDP bound.

Here `K ≥ 0` is derived from the a.s. norm bound, matching HDP's statement where nonnegativity of
`K` is implicit in `‖Xᵢ‖ ≤ K`.
-/
theorem matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_at_optimizingLambda_nonempty
    {n N : ℕ} (hn : 0 < n) (hN : 0 < N)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (ht : 0 ≤ t)
    (h_integrable : ∀ i, Integrable (X i) μ)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K)
    (hlieb_recursion :
      matrixBernsteinTraceMgf μ X
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ X
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t))))
    (hlieb_recursion_neg :
      matrixBernsteinTraceMgf μ (fun i ω => -X i ω)
          (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t) ≤
        Matrix.trace
          (NormedSpace.exp
            (matrixBernsteinLogMgfSum μ (fun i ω => -X i ω)
              (matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t)))) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let i0 : Fin N := ⟨0, by simpa using hN⟩
  have hK : 0 ≤ K :=
    matrixBernstein_norm_bound_nonneg_of_index i0 (h_norm_bound i0)
  exact matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_at_optimizingLambda
    hn X μ hσ hK ht h_integrable h_square_integrable h_mean_zero hX_symmetric
    h_norm_bound hlieb_recursion hlieb_recursion_neg

/--
The nondegenerate arbitrary finite-family case of HDP Theorem 5.4.1.

The two Lieb-recursion hypotheses in
`matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_at_optimizingLambda_nonempty` are
discharged from independence by the arbitrary finite-family Lieb recursion above, once for `X` and
once for `-X`.
-/
theorem matrixBernstein_norm_tail_le_of_pos_variance
    {n N : ℕ} (hn : 0 < n) (hN : 0 < N)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (h_independent : MatrixBernsteinIndependent X μ)
    (hσ : 0 < matrixBernsteinVarianceProxy μ X) (ht : 0 ≤ t)
    (h_integrable : ∀ i, Integrable (X i) μ)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (h_mean_zero : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (h_norm_bound : ∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  let l : ℝ := matrixBernsteinOptimizingLambda (matrixBernsteinVarianceProxy μ X) K t
  let Xneg : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ := fun i ω => -X i ω
  have h_independent_neg : MatrixBernsteinIndependent Xneg μ :=
    h_independent.neg
  have h_integrable_neg : ∀ i, Integrable (Xneg i) μ := by
    intro i
    simpa [Xneg] using (h_integrable i).neg
  have hXneg_symmetric : ∀ i, ∀ᵐ ω ∂μ, (Xneg i ω).IsHermitian := by
    intro i
    exact (hX_symmetric i).mono fun _ hω => hω.neg
  have h_norm_bound_neg : ∀ i, ∀ᵐ ω ∂μ, ‖Xneg i ω‖ ≤ K := by
    intro i
    exact (h_norm_bound i).mono fun _ hω => by simpa [Xneg] using hω
  have htrace_int :
      Integrable
        (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum X ω))) μ :=
    matrixBernstein_trace_exp_integrable_of_summands_integrable_norm_bound_isHermitian_ae
      (X := X) (K := K) (l := l) h_integrable hX_symmetric h_norm_bound
  have htrace_int_neg :
      Integrable
        (fun ω => Matrix.trace (NormedSpace.exp (l • matrixBernsteinSum Xneg ω))) μ :=
    matrixBernstein_trace_exp_integrable_of_summands_integrable_norm_bound_isHermitian_ae
      (X := Xneg) (K := K) (l := l)
      h_integrable_neg hXneg_symmetric h_norm_bound_neg
  have hlieb_recursion :
      matrixBernsteinTraceMgf μ X l ≤
        Matrix.trace (NormedSpace.exp (matrixBernsteinLogMgfSum μ X l)) :=
    matrixBernstein_lieb_recursion_of_independent
      μ X h_independent h_integrable hX_symmetric h_norm_bound htrace_int
  have hlieb_recursion_neg :
      matrixBernsteinTraceMgf μ Xneg l ≤
        Matrix.trace (NormedSpace.exp (matrixBernsteinLogMgfSum μ Xneg l)) :=
    matrixBernstein_lieb_recursion_of_independent
      μ Xneg h_independent_neg h_integrable_neg hXneg_symmetric h_norm_bound_neg
      htrace_int_neg
  exact
    matrixBernstein_norm_tail_le_of_lieb_recursions_and_mgfRate_at_optimizingLambda_nonempty
      hn hN X μ hσ ht h_integrable h_square_integrable h_mean_zero hX_symmetric
      h_norm_bound
      (by simpa [l] using hlieb_recursion)
      (by simpa [Xneg, l] using hlieb_recursion_neg)

/-- Zero variance forces every Hermitian summand to vanish almost surely. -/
lemma matrixBernstein_ae_eq_zero_of_varianceProxy_eq_zero
    {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (hσ : matrixBernsteinVarianceProxy μ X = 0) :
    ∀ i, ∀ᵐ ω ∂μ, X i ω = 0 := by
  have hV_zero : matrixBernsteinVarianceMatrix μ X = 0 := by
    apply norm_eq_zero.mp
    simpa [matrixBernsteinVarianceProxy] using hσ
  have htrace_int :
      ∀ i, Integrable (fun ω => Matrix.trace ((X i ω) ^ 2)) μ := by
    intro i
    have hcomp :
        Integrable
          (fun ω =>
            Matrix.traceRightMulContinuousLinearMap
              (1 : Matrix (Fin n) (Fin n) ℝ) ((X i ω) ^ 2)) μ :=
      (Matrix.traceRightMulContinuousLinearMap
        (1 : Matrix (Fin n) (Fin n) ℝ)).integrable_comp (h_square_integrable i)
    simpa [Matrix.traceRightMulContinuousLinearMap_apply] using hcomp
  have htrace_integral :
      ∀ i,
        Matrix.trace (∫ ω, (X i ω) ^ 2 ∂μ) =
          ∫ ω, Matrix.trace ((X i ω) ^ 2) ∂μ := by
    intro i
    calc
      Matrix.trace (∫ ω, (X i ω) ^ 2 ∂μ) =
          Matrix.traceRightMulContinuousLinearMap
            (1 : Matrix (Fin n) (Fin n) ℝ) (∫ ω, (X i ω) ^ 2 ∂μ) := by
            simp [Matrix.traceRightMulContinuousLinearMap_apply]
      _ =
          ∫ ω,
            Matrix.traceRightMulContinuousLinearMap
              (1 : Matrix (Fin n) (Fin n) ℝ) ((X i ω) ^ 2) ∂μ := by
            exact (ContinuousLinearMap.integral_comp_comm
              (Matrix.traceRightMulContinuousLinearMap
                (1 : Matrix (Fin n) (Fin n) ℝ)) (h_square_integrable i)).symm
      _ = ∫ ω, Matrix.trace ((X i ω) ^ 2) ∂μ := by
            simp [Matrix.traceRightMulContinuousLinearMap_apply]
  have htrace_nonneg :
      ∀ i, 0 ≤ᵐ[μ] fun ω => Matrix.trace ((X i ω) ^ 2) := by
    intro i
    filter_upwards [hX_symmetric i] with ω hω
    have hsymm : (X i ω).IsSymm := by
      simpa using hω
    have hpsd : ((X i ω) ^ 2).PosSemidef := by
      simpa [pow_two, hsymm.eq] using Matrix.posSemidef_self_mul_conjTranspose (X i ω)
    exact hpsd.trace_nonneg
  have htraceV_sum :
      Matrix.trace (matrixBernsteinVarianceMatrix μ X) =
        ∑ i : Fin N, ∫ ω, Matrix.trace ((X i ω) ^ 2) ∂μ := by
    calc
      Matrix.trace (matrixBernsteinVarianceMatrix μ X) =
          ∑ i : Fin N, Matrix.trace (∫ ω, (X i ω) ^ 2 ∂μ) := by
            simp [matrixBernsteinVarianceMatrix]
      _ = ∑ i : Fin N, ∫ ω, Matrix.trace ((X i ω) ^ 2) ∂μ := by
            exact Finset.sum_congr rfl fun i _ => htrace_integral i
  have hscalar_sum_zero :
      (∑ i : Fin N, ∫ ω, Matrix.trace ((X i ω) ^ 2) ∂μ) = 0 := by
    rw [← htraceV_sum, hV_zero]
    simp
  have hscalar_integral_zero :
      ∀ i, ∫ ω, Matrix.trace ((X i ω) ^ 2) ∂μ = 0 := by
    intro i
    exact
      (Finset.sum_eq_zero_iff_of_nonneg
        (fun j _ => integral_nonneg_of_ae (htrace_nonneg j))).1
        hscalar_sum_zero i (Finset.mem_univ i)
  intro i
  have htrace_ae_zero :
      (fun ω => Matrix.trace ((X i ω) ^ 2)) =ᵐ[μ] 0 :=
    (integral_eq_zero_iff_of_nonneg_ae (htrace_nonneg i) (htrace_int i)).1
      (hscalar_integral_zero i)
  filter_upwards [hX_symmetric i, htrace_ae_zero] with ω hω htr
  have hsymm : (X i ω).IsSymm := by
    simpa using hω
  have htr_mul :
      Matrix.trace ((X i ω) * (X i ω).conjTranspose) = 0 := by
    simpa [pow_two, hsymm.eq] using htr
  exact Matrix.trace_mul_conjTranspose_self_eq_zero_iff.mp htr_mul

/-- Zero variance forces the matrix Bernstein sum to vanish almost surely. -/
lemma matrixBernsteinSum_ae_eq_zero_of_varianceProxy_eq_zero
    {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian)
    (hσ : matrixBernsteinVarianceProxy μ X = 0) :
    ∀ᵐ ω ∂μ, matrixBernsteinSum X ω = 0 := by
  have hX_zero :
      ∀ i, ∀ᵐ ω ∂μ, X i ω = 0 :=
    matrixBernstein_ae_eq_zero_of_varianceProxy_eq_zero
      X μ h_square_integrable hX_symmetric hσ
  filter_upwards [Filter.eventually_all.mpr hX_zero] with ω hω
  simp [matrixBernsteinSum, hω]

/-- The zero-variance branch of HDP Theorem 5.4.1 for any finite family. -/
theorem matrixBernstein_norm_tail_le_of_zero_variance
    {n N : ℕ} (hn : 0 < n)
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] {K t : ℝ}
    (hσ : matrixBernsteinVarianceProxy μ X = 0) (ht : 0 ≤ t)
    (h_square_integrable : ∀ i, Integrable (fun ω => (X i ω) ^ 2) μ)
    (hX_symmetric : ∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian) :
    (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
      matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t := by
  have hsum_zero :
      ∀ᵐ ω ∂μ, matrixBernsteinSum X ω = 0 :=
    matrixBernsteinSum_ae_eq_zero_of_varianceProxy_eq_zero
      X μ h_square_integrable hX_symmetric hσ
  by_cases htzero : t = 0
  · subst t
    have hprob_le_one :
        (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ (0 : ℝ)}).toReal ≤ 1 := by
      have hmeasure :
          μ {ω | ‖matrixBernsteinSum X ω‖ ≥ (0 : ℝ)} ≤ μ Set.univ :=
        measure_mono (Set.subset_univ _)
      have hreal :=
        ENNReal.toReal_mono (measure_ne_top μ Set.univ) hmeasure
      simp [IsProbabilityMeasure.measure_univ] at hreal ⊢
    have htail_ge_one :
        1 ≤ matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K 0 := by
      have hn_one : (1 : ℝ) ≤ n := by
        exact_mod_cast Nat.succ_le_of_lt hn
      rw [hσ, matrixBernsteinTailBound]
      simp
      nlinarith
    exact hprob_le_one.trans htail_ge_one
  · have htpos : 0 < t := lt_of_le_of_ne ht (Ne.symm htzero)
    have hnot_event :
        ∀ᵐ ω ∂μ, ω ∉ {ω | ‖matrixBernsteinSum X ω‖ ≥ t} := by
      filter_upwards [hsum_zero] with ω hω
      simp [hω, not_le.mpr htpos]
    have hmeasure_zero : μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t} = 0 :=
      (measure_eq_zero_iff_ae_notMem).2 hnot_event
    rw [hmeasure_zero]
    simpa using matrixBernsteinTailBound_nonneg n (matrixBernsteinVarianceProxy μ X) K t

/-! ## Exact HDP Theorem 5.4.1 proposition -/

/--
HDP Theorem 5.4.1, "Matrix Bernstein inequality".

Let `X₁, ..., X_N` be independent, mean-zero, `n × n` symmetric random matrices, such that
`‖Xᵢ‖ ≤ K` almost surely for all `i`. With
`σ² = ‖∑ᵢ E Xᵢ²‖`, for every `t ≥ 0`,

`P{ ‖∑ᵢ Xᵢ‖ ≥ t } ≤ 2n exp (-(t²/2)/(σ² + Kt/3))`.

The explicit integrability hypotheses make the Bochner integrals in the formal statement non-junk.
The hypothesis `0 < n` records the positive matrix dimension implicit in the textbook statement.
-/
def matrix_bernstein_inequality_hdp {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : ℝ) : Prop :=
  0 < n →
    MatrixBernsteinIndependent X μ →
      (∀ i, Integrable (X i) μ) →
        (∀ i, Integrable (fun ω => (X i ω) ^ 2) μ) →
          (∀ i, ∫ ω, X i ω ∂μ = 0) →
            (∀ i, ∀ᵐ ω ∂μ, (X i ω).IsHermitian) →
              (∀ i, ∀ᵐ ω ∂μ, ‖X i ω‖ ≤ K) →
                ∀ t : ℝ, 0 ≤ t →
                  (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ t}).toReal ≤
                    matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K t

/--
The exact HDP theorem is trivial for an empty family of summands.

This handles the boundary case `N = 0` directly, so the nondegenerate Bernstein proof can focus
on the genuinely recursive case.
-/
theorem matrix_bernstein_inequality_hdp_of_no_summands {n : ℕ}
    (X : Fin 0 → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : ℝ) :
    matrix_bernstein_inequality_hdp X μ K := by
  intro hn _hindep _h_integrable _h_square_integrable _h_mean_zero _h_symmetric
    _h_norm_bound t ht
  have hsum_zero : ∀ ω, matrixBernsteinSum X ω = 0 := by
    intro ω
    simp [matrixBernsteinSum]
  have hvar_zero : matrixBernsteinVarianceProxy μ X = 0 := by
    simp [matrixBernsteinVarianceProxy, matrixBernsteinVarianceMatrix]
  by_cases htzero : t = 0
  · subst t
    have hprob_le_one :
        (μ {ω | ‖matrixBernsteinSum X ω‖ ≥ (0 : ℝ)}).toReal ≤ 1 := by
      have hmeasure :
          μ {ω | ‖matrixBernsteinSum X ω‖ ≥ (0 : ℝ)} ≤ μ Set.univ :=
        measure_mono (Set.subset_univ _)
      have hreal :=
        ENNReal.toReal_mono (measure_ne_top μ Set.univ) hmeasure
      simp [IsProbabilityMeasure.measure_univ] at hreal ⊢
    have htail_ge_one :
        1 ≤ matrixBernsteinTailBound n (matrixBernsteinVarianceProxy μ X) K 0 := by
      have hn_one : (1 : ℝ) ≤ n := by
        exact_mod_cast Nat.succ_le_of_lt hn
      rw [hvar_zero, matrixBernsteinTailBound]
      simp
      nlinarith
    exact hprob_le_one.trans htail_ge_one
  · have htpos : 0 < t := lt_of_le_of_ne ht (Ne.symm htzero)
    have hset_empty : {ω | ‖matrixBernsteinSum X ω‖ ≥ t} = ∅ := by
      ext ω
      simp [hsum_zero ω, not_le.mpr htpos]
    rw [hset_empty]
    simpa using matrixBernsteinTailBound_nonneg n (matrixBernsteinVarianceProxy μ X) K t

/--
The exact HDP Theorem 5.4.1 for every finite family of summands.

The proof splits off the empty family, then for nonempty families combines the general
positive-variance Bernstein proof with the already proved zero-variance branch.
-/
theorem matrix_bernstein_inequality_hdp_all {n N : ℕ}
    (X : Fin N → Ω → Matrix (Fin n) (Fin n) ℝ)
    (μ : Measure Ω) [IsProbabilityMeasure μ] (K : ℝ) :
    matrix_bernstein_inequality_hdp X μ K := by
  cases N with
  | zero =>
      exact matrix_bernstein_inequality_hdp_of_no_summands X μ K
  | succ N =>
      intro hn hindep h_integrable h_square_integrable h_mean_zero h_symmetric
        h_norm_bound t ht
      by_cases hσpos : 0 < matrixBernsteinVarianceProxy μ X
      · exact matrixBernstein_norm_tail_le_of_pos_variance
          (N := N + 1) hn (Nat.succ_pos N) X μ hindep hσpos ht h_integrable
          h_square_integrable h_mean_zero h_symmetric h_norm_bound
      · have hσzero : matrixBernsteinVarianceProxy μ X = 0 := by
          exact le_antisymm (le_of_not_gt hσpos) (matrixBernsteinVarianceProxy_nonneg μ X)
        exact matrixBernstein_norm_tail_le_of_zero_variance
          hn X μ hσzero ht h_square_integrable h_symmetric

end RMT
