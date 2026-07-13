/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MeasureInfrastructure
import SLT.SubGaussian
import Mathlib.Probability.Moments.Basic
import Mathlib.Probability.Moments.Variance
import Mathlib.Probability.IdentDistrib
import Mathlib.Tactic

/-!
# Probability Utility Lemma Library

Shared probability-theoretic lemmas for concentration inequalities and
statistical learning theory.  Provides Chernoff optimisation, MGF/CGF
properties for independent sums, and correlation inequalities.

## Main results

* `subgaussTailOneSided`: One-sided sub-Gaussian tail bound via Chernoff optimisation.
* `subgaussTailTwoSided`: Two-sided sub-Gaussian tail bound.
* `mgfAddIndependent`: MGF factorisation for independent random variables.
* `cgfAddIndependent`: CGF subadditivity for independent sums.
* `varianceAddIndependent`: Var(X+Y) = Var(X) + Var(Y) for independent X, Y.
* `cauchySchwarzExpectation`: Cauchy-Schwarz inequality for expectations.
* `expectation_nonneg_of_nonneg`: E[X] ≥ 0 when X ≥ 0 a.e.
* `expectation_mono`: E[X] ≤ E[Y] when X ≤ Y a.e.
* `additivityOfExpectation`: E[X+Y] = E[X] + E[Y].
-/

open MeasureTheory ProbabilityTheory Real Set Finset
open scoped ENNReal NNReal BigOperators Topology

noncomputable section

variable {Ω : Type*} [MeasurableSpace Ω]

/-!
## Sub-Gaussian Tail Bounds
-/

/-- One-sided sub-Gaussian tail bound: if cgf(X, t) ≤ t²σ²/2 for all t,
    then P(X ≥ u) ≤ exp(-u²/(2σ²)) for u > 0.
    This is the standard Chernoff optimisation for sub-Gaussian variables. -/
theorem subgaussTailOneSided {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {σ u : ℝ} (hσ : 0 < σ) (hu : 0 < u)
    (h_cgf : ∀ t : ℝ, ProbabilityTheory.cgf X μ t ≤ t^2 * σ^2 / 2)
    (h_int : ∀ t : ℝ, Integrable (fun ω => exp (t * X ω)) μ) :
    (μ {ω | u ≤ X ω}).toReal ≤ exp (-u^2 / (2 * σ^2)) := by
  set t_opt := u / σ^2 with ht_def
  have ht_nonneg : 0 ≤ t_opt := div_nonneg (by linarith) (sq_nonneg σ)
  have h_chernoff : (μ {ω | u ≤ X ω}).toReal ≤
      exp (-t_opt * u + ProbabilityTheory.cgf X μ t_opt) :=
    ProbabilityTheory.measure_ge_le_exp_cgf u ht_nonneg (h_int t_opt)
  have h_cgf_bound : ProbabilityTheory.cgf X μ t_opt ≤ t_opt^2 * σ^2 / 2 := h_cgf t_opt
  have h_opt_simp : -t_opt * u + t_opt^2 * σ^2 / 2 = -u^2 / (2 * σ^2) := by
    rw [ht_def]
    field_simp [hσ.ne.symm]
    ring
  calc (μ {ω | u ≤ X ω}).toReal
    _ ≤ exp (-t_opt * u + ProbabilityTheory.cgf X μ t_opt) := h_chernoff
    _ ≤ exp (-t_opt * u + (t_opt^2 * σ^2 / 2)) :=
      exp_le_exp.mpr (by gcongr; exact h_cgf_bound)
    _ = exp (-u^2 / (2 * σ^2)) := by rw [h_opt_simp]

/-- Two-sided sub-Gaussian tail bound: P(|X| ≥ u) ≤ 2·exp(-u²/(2σ²))
    when E[X] = 0 and cgf(±X, t) ≤ t²σ²/2 for all t. -/
theorem subgaussTailTwoSided {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {σ u : ℝ} (hσ : 0 < σ) (hu : 0 < u)
    (h_cgf_pos : ∀ t : ℝ, ProbabilityTheory.cgf X μ t ≤ t^2 * σ^2 / 2)
    (h_cgf_neg : ∀ t : ℝ, ProbabilityTheory.cgf (-X) μ t ≤ t^2 * σ^2 / 2)
    (h_int : ∀ t : ℝ, Integrable (fun ω => exp (t * X ω)) μ) :
    (μ {ω | u ≤ |X ω|}).toReal ≤ 2 * exp (-u^2 / (2 * σ^2)) := by
  have h_set_decomp : {ω | u ≤ |X ω|} ⊆ {ω | u ≤ X ω} ∪ {ω | u ≤ -X ω} := by
    intro ω hω
    simp only [mem_setOf_eq, mem_union_iff]
    by_cases hx : 0 ≤ X ω
    · left; rw [abs_of_nonneg hx] at hω; exact hω
    · right; rw [abs_of_neg (not_le.mp hx)] at hω; exact hω
  have h_union_meas : μ {ω | u ≤ |X ω|} ≤ μ ({ω | u ≤ X ω} ∪ {ω | u ≤ -X ω}) :=
    measure_mono h_set_decomp
  have h_union_bound : μ ({ω | u ≤ X ω} ∪ {ω | u ≤ -X ω}) ≤
      μ {ω | u ≤ X ω} + μ {ω | u ≤ -X ω} := measure_union_le _ _
  have h_toReal : (μ {ω | u ≤ |X ω|}).toReal ≤
      (μ {ω | u ≤ X ω} + μ {ω | u ≤ -X ω}).toReal :=
    ENNReal.toReal_mono (measure_ne_top _ _) (le_trans h_union_meas h_union_bound)
  have h_sum_toReal : (μ {ω | u ≤ X ω} + μ {ω | u ≤ -X ω}).toReal =
      (μ {ω | u ≤ X ω}).toReal + (μ {ω | u ≤ -X ω}).toReal :=
    ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _)
  rw [h_sum_toReal] at h_toReal
  have h_bound_pos : (μ {ω | u ≤ X ω}).toReal ≤ exp (-u^2 / (2 * σ^2)) :=
    subgaussTailOneSided hσ hu h_cgf_pos h_int
  have h_bound_neg : (μ {ω | u ≤ -X ω}).toReal ≤ exp (-u^2 / (2 * σ^2)) := by
    have h_int_neg : ∀ t : ℝ, Integrable (fun ω => exp (t * (-X ω))) μ := by
      intro t
      simpa [mul_comm, ← mul_neg] using h_int (-t)
    exact subgaussTailOneSided hσ hu h_cgf_neg h_int_neg
  linarith

/-!
## MGF and CGF Properties for Independent Sums
-/

/-- The MGF of a sum of independent random variables is the product
    of the individual MGFs. -/
theorem mgfAddIndependent {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ) (t : ℝ)
    (h_int_X : Integrable (fun ω => exp (t * X ω)) μ)
    (h_int_Y : Integrable (fun ω => exp (t * Y ω)) μ) :
    mgf (X + Y) μ t = mgf X μ t * mgf Y μ t := by
  rw [mgf_eq_integral, mgf_eq_integral, mgf_eq_integral]
  calc
    ∫ ω, exp (t * (X ω + Y ω)) ∂μ = ∫ ω, exp (t * X ω) * exp (t * Y ω) ∂μ := by
      refine integral_congr_ae ?_
      filter_upwards with ω
      rw [mul_add, exp_add]
    _ = (∫ ω, exp (t * X ω) ∂μ) * (∫ ω, exp (t * Y ω) ∂μ) :=
      h_indep.integral_mul_of_indep
        (h_int_X.aestronglyMeasurable) (h_int_Y.aestronglyMeasurable)

/-- CGF subadditivity for independent sums:
    cgf(X+Y) ≤ cgf(X) + cgf(Y) when X ⟂ Y. -/
theorem cgfAddIndependent {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ) (t : ℝ)
    (h_int_X : Integrable (fun ω => exp (t * X ω)) μ)
    (h_int_Y : Integrable (fun ω => exp (t * Y ω)) μ) :
    ProbabilityTheory.cgf (X + Y) μ t ≤
    ProbabilityTheory.cgf X μ t + ProbabilityTheory.cgf Y μ t := by
  have h_mgf_pos_X : 0 < mgf X μ t := ProbabilityTheory.mgf_pos h_int_X t
  have h_mgf_pos_Y : 0 < mgf Y μ t := ProbabilityTheory.mgf_pos h_int_Y t
  rw [ProbabilityTheory.cgf_eq_log_mgf, ProbabilityTheory.cgf_eq_log_mgf,
    ProbabilityTheory.cgf_eq_log_mgf]
  rw [mgfAddIndependent h_indep t h_int_X h_int_Y]
  rw [Real.log_mul (ne_of_gt h_mgf_pos_X) (ne_of_gt h_mgf_pos_Y)]

/-- CGF subadditivity for a sum of n independent variables.
    Proved by induction. -/
theorem cgfSumIndependent {μ : Measure Ω} [IsProbabilityMeasure μ]
    (X : ℕ → Ω → ℝ) (n : ℕ)
    (h_pairwise_indep : ∀ i j, i < n → j < n → i ≠ j → IndepFun (X i) (X j) μ)
    (t : ℝ) (h_int : ∀ i, i < n → Integrable (fun ω => exp (t * X i ω)) μ) :
    ProbabilityTheory.cgf (∑ i in range n, X i) μ t ≤
    ∑ i in range n, ProbabilityTheory.cgf (X i) μ t := by
  induction' n with k ih
  · simp
  · rw [sum_range_succ, sum_range_succ]
    have h_sum_int : Integrable (fun ω => exp (t * (∑ i in range k, X i ω))) μ := by
      have : (fun ω => exp (t * (∑ i in range k, X i ω))) =
          ∏ i in range k, fun ω => exp (t * X i ω) := by
        ext ω
        simp [exp_add, mul_add, add_comm, add_left_comm, add_assoc]
      rw [this]
      refine Integrable.finset_prod _ (fun i hi => ?_)
      rw [mem_range] at hi
      exact h_int i (lt_of_lt_of_le hi (by omega))
    have h_indep_sumk : IndepFun (∑ i in range k, X i) (X k) μ := by
      refine h_pairwise_indep 0 k (by omega) (by omega) (by omega) |>.finset_sum ?_ ?_
      · exact Finset.range k
      · intro i hi; rw [mem_range] at hi; exact h_pairwise_indep i k (by omega) (by omega) (by omega)
    have h_indep_XY : IndepFun (X k) (∑ i in range k, X i) μ :=
      h_indep_sumk.symm
    apply le_trans (cgfAddIndependent h_indep_XY t (h_int k (by omega)) h_sum_int)
    rw [add_comm]
    apply add_le_add_left
    apply ih
    · intro i j hi hj hij
      exact h_pairwise_indep i j (lt_of_lt_of_le hi (by omega)) (lt_of_lt_of_le hj (by omega)) hij
    · intro i hi
      exact h_int i (lt_of_lt_of_le hi (by omega))

/-!
## Variance Properties for Independent Sums
-/

/-- Var(X+Y) = Var(X) + Var(Y) when X ⟂ Y. -/
theorem varianceAddIndependent {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X Y : Ω → ℝ} (h_indep : IndepFun X Y μ)
    (h_int_X : Integrable X μ) (h_int_Y : Integrable Y μ)
    (h_int_X_sq : Integrable (X ^ 2) μ) (h_int_Y_sq : Integrable (Y ^ 2) μ) :
    ProbabilityTheory.Variance (X + Y) μ =
    ProbabilityTheory.Variance X μ + ProbabilityTheory.Variance Y μ := by
  have h_var_add := ProbabilityTheory.variance_add h_indep h_int_X h_int_Y
  have h_cov_zero : ProbabilityTheory.covariance X Y μ = 0 :=
    h_indep.covariance_eq_zero h_int_X h_int_Y
  rw [h_var_add, h_cov_zero, add_zero]

/-!
## Cauchy-Schwarz Inequality for Expectations
-/

/-- Cauchy-Schwarz inequality: |E[XY]| ≤ √(E[X²]·E[Y²]). -/
theorem cauchySchwarzExpectation {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X Y : Ω → ℝ} (h_int_X_sq : Integrable (X ^ 2) μ) (h_int_Y_sq : Integrable (Y ^ 2) μ)
    (h_int_XY : Integrable (X * Y) μ) :
    |∫ ω, X ω * Y ω ∂μ| ≤
    Real.sqrt ((∫ ω, X ω ^ 2 ∂μ) * (∫ ω, Y ω ^ 2 ∂μ)) := by
  have h_sq : (∫ ω, X ω * Y ω ∂μ)^2 ≤
      (∫ ω, X ω ^ 2 ∂μ) * (∫ ω, Y ω ^ 2 ∂μ) :=
    integral_mul_self_le_integral_mul_self_sq_sq h_int_X_sq h_int_Y_sq
  calc
    |∫ ω, X ω * Y ω ∂μ| = Real.sqrt ((∫ ω, X ω * Y ω ∂μ)^2) := by
      rw [Real.sqrt_sq_eq_abs]
    _ ≤ Real.sqrt ((∫ ω, X ω ^ 2 ∂μ) * (∫ ω, Y ω ^ 2 ∂μ)) :=
      Real.sqrt_le_sqrt h_sq

/-!
## Basic Expectation Properties
-/

/-- E[X] ≥ 0 when X ≥ 0 almost everywhere. -/
theorem expectation_nonneg_of_nonneg {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} (h_nonneg : 0 ≤ᵐ[μ] X) (h_int : Integrable X μ) : 0 ≤ ∫ ω, X ω ∂μ := by
  refine integral_nonneg ?_
  intro ω
  exact h_nonneg ω

/-- E[X] ≤ E[Y] when X ≤ Y almost everywhere. -/
theorem expectation_mono {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X Y : Ω → ℝ} (h_le : X ≤ᵐ[μ] Y) (h_int_X : Integrable X μ) (h_int_Y : Integrable Y μ) :
    ∫ ω, X ω ∂μ ≤ ∫ ω, Y ω ∂μ :=
  integral_mono h_int_X h_int_Y h_le

/-- Linearity of expectation: E[X + Y] = E[X] + E[Y]. -/
theorem additivityOfExpectation {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X Y : Ω → ℝ} (h_int_X : Integrable X μ) (h_int_Y : Integrable Y μ) :
    ∫ ω, (X ω + Y ω) ∂μ = (∫ ω, X ω ∂μ) + (∫ ω, Y ω ∂μ) :=
  integral_add h_int_X h_int_Y

/-- E[c·X] = c·E[X]. -/
theorem homogeneityOfExpectation {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} (c : ℝ) (h_int : Integrable X μ) :
    ∫ ω, c * X ω ∂μ = c * ∫ ω, X ω ∂μ :=
  integral_mul_left c h_int

end
