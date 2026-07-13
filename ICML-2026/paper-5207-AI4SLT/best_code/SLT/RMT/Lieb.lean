/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.MatrixInfra.MatCalc
import Mathlib.Analysis.Convex.Integral
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Topology.Algebra.Module.ContinuousLinearMap.PiProd

/-!
# Lieb Inequality for Random Matrices

This file records the setup for HDP Lemma 5.4.9 in the local matrix notation.

The deterministic Lieb concavity theorem is proved in `SLT.MatrixInfra.MatCalc` as
`Matrix.lieb_inequality_hdp_5_4_8`. The reductions below isolate the random-matrix Jensen step
that turns deterministic Lieb concavity into HDP Lemma 5.4.9.

## Main definitions

* `RMT.lieb_inequality_random_matrices_hdp`: the proposition form of HDP Lemma 5.4.9.

## Main results

* `RMT.concaveOn_integral_le_of_continuousWithinAt`: Jensen's inequality for a concave function
  on a non-closed convex domain.
* `RMT.integral_exp_posDef_of_ae_isHermitian`: `E exp Z` is positive definite for Hermitian
  random `Z`.
* `RMT.lieb_inequality_random_matrices_hdp_of_deterministic_lieb`: the reduction from
  deterministic Lieb concavity to the random-matrix statement.
* `RMT.lieb_inequality_random_matrices_hdp_5_4_9`: HDP Lemma 5.4.9.
-/

open MeasureTheory
open scoped MatrixOrder Matrix.Norms.L2Operator
open scoped Topology
open NormedSpace

noncomputable section

namespace RMT

variable {Ω : Type*} [MeasurableSpace Ω]
variable {n : Type*}

private def matrixEntryLinearMap (i j : n) : Matrix n n ℝ →L[ℝ] ℝ :=
  (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : n => ℝ) j).comp
    (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : n => n → ℝ) i)

@[simp] private theorem matrixEntryLinearMap_apply (i j : n) (M : Matrix n n ℝ) :
    matrixEntryLinearMap i j M = M i j := rfl

variable [Fintype n]

/--
Jensen's inequality for a concave function on a possibly non-closed convex domain.

Mathlib's packaged integral Jensen theorem assumes the domain is closed. For HDP Lemma 5.4.9 the
positive-definite cone is relatively open, so this variant uses closure of the hypograph and only
requires continuity within the domain at the mean.
-/
theorem concaveOn_integral_le_of_continuousWithinAt
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {s : Set E} {g : E → ℝ} {f : Ω → E}
    (hg : ConcaveOn ℝ s g)
    (hgc : ContinuousWithinAt g s (∫ x, f x ∂μ))
    (hfs : ∀ᵐ x ∂μ, f x ∈ s)
    (hfi : Integrable f μ)
    (hgi : Integrable (g ∘ f) μ) :
    (∫ x, g (f x) ∂μ) ≤ g (∫ x, f x ∂μ) := by
  let hypograph : Set (E × ℝ) := {p | p.1 ∈ s ∧ p.2 ≤ g p.1}
  have hhyp_conv : Convex ℝ hypograph := by
    simpa [hypograph] using hg.convex_hypograph
  have hpair_ae : ∀ᵐ x ∂μ, (f x, g (f x)) ∈ hypograph := by
    filter_upwards [hfs] with x hx
    exact ⟨hx, le_rfl⟩
  have hpair_int : Integrable (fun x => (f x, g (f x))) μ :=
    hfi.prodMk hgi
  have hclosure :
      (∫ x, (f x, g (f x)) ∂μ) ∈ closure hypograph := by
    exact hhyp_conv.closure.integral_mem isClosed_closure
      (hpair_ae.mono fun _ hx => subset_closure hx) hpair_int
  have hpair_eq :
      (∫ x, (f x, g (f x)) ∂μ) =
        (∫ x, f x ∂μ, ∫ x, g (f x) ∂μ) :=
    integral_pair hfi hgi
  rw [hpair_eq] at hclosure
  by_contra hnot
  have hlt : g (∫ x, f x ∂μ) < ∫ x, g (f x) ∂μ :=
    lt_of_not_ge hnot
  let m : ℝ := (g (∫ x, f x ∂μ) + ∫ x, g (f x) ∂μ) / 2
  have hgxm : g (∫ x, f x ∂μ) < m := by
    dsimp [m]
    linarith
  have hmy : m < ∫ x, g (f x) ∂μ := by
    dsimp [m]
    linarith
  have hcont_eventually :
      {z : E | g z < m} ∈ 𝓝[s] (∫ x, f x ∂μ) :=
    hgc (isOpen_Iio.mem_nhds hgxm)
  rcases mem_nhdsWithin_iff_exists_mem_nhds_inter.mp hcont_eventually with
    ⟨U, hU, hUs⟩
  let V : Set (E × ℝ) := U ×ˢ Set.Ioi m
  have hV_nhds : V ∈ 𝓝 (∫ x, f x ∂μ, ∫ x, g (f x) ∂μ) :=
    prod_mem_nhds hU (isOpen_Ioi.mem_nhds hmy)
  have hV_disjoint : Disjoint V hypograph := by
    rw [Set.disjoint_left]
    intro p hpV hph
    have hgm : g p.1 < m := hUs ⟨hpV.1, hph.1⟩
    have hmp : m < p.2 := hpV.2
    have hple : p.2 ≤ g p.1 := hph.2
    linarith
  have hV_inter_nonempty : (V ∩ hypograph).Nonempty :=
    mem_closure_iff_nhds.mp hclosure V hV_nhds
  rcases hV_inter_nonempty with ⟨p, hpV, hph⟩
  exact hV_disjoint.le_bot ⟨hpV, hph⟩

private def matrixQuadraticFormLinearMap (x : n → ℝ) : Matrix n n ℝ →L[ℝ] ℝ :=
  ∑ i, ∑ j, (x i * x j) • matrixEntryLinearMap i j

private theorem matrixQuadraticFormLinearMap_apply (x : n → ℝ) (M : Matrix n n ℝ) :
    matrixQuadraticFormLinearMap x M = dotProduct (star x) (Matrix.mulVec M x) := by
  simp [matrixQuadraticFormLinearMap, dotProduct, Matrix.mulVec]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  change x i * x j * M i j = x i * (M i j * x j)
  ring

variable [DecidableEq n]

/-! ## Pointwise reductions for the Jensen step -/

/-- If `Z` is Hermitian a.e., then `exp Z` lies in the positive-definite domain a.e. -/
theorem ae_exp_mem_positiveDefiniteSymmetricMatrices
    {μ : Measure Ω} {Z : Ω → Matrix n n ℝ}
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian) :
    ∀ᵐ ω ∂μ, exp (Z ω) ∈ Matrix.positiveDefiniteSymmetricMatrices n := by
  filter_upwards [hZ_symm] with ω hZω
  exact Matrix.exp_posDef_of_isHermitian hZω

private theorem integral_exp_isHermitian_of_ae_isHermitian
    {μ : Measure Ω} {Z : Ω → Matrix n n ℝ}
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian)
    (hZ_exp_int : Integrable (fun ω => exp (Z ω)) μ) :
    (∫ ω, exp (Z ω) ∂μ).IsHermitian := by
  refine Matrix.IsHermitian.ext fun i j => ?_
  have hentry : ∀ᵐ ω ∂μ,
      matrixEntryLinearMap j i (exp (Z ω)) = matrixEntryLinearMap i j (exp (Z ω)) := by
    filter_upwards [hZ_symm] with ω hZω
    have hExp : (exp (Z ω)).IsHermitian := Matrix.IsHermitian.exp hZω
    simpa using hExp.apply i j
  calc
    star ((∫ ω, exp (Z ω) ∂μ) j i) =
        matrixEntryLinearMap j i (∫ ω, exp (Z ω) ∂μ) := by
      simp
    _ = ∫ ω, matrixEntryLinearMap j i (exp (Z ω)) ∂μ := by
      exact (ContinuousLinearMap.integral_comp_comm (matrixEntryLinearMap j i) hZ_exp_int).symm
    _ = ∫ ω, matrixEntryLinearMap i j (exp (Z ω)) ∂μ := by
      exact integral_congr_ae hentry
    _ = matrixEntryLinearMap i j (∫ ω, exp (Z ω) ∂μ) := by
      exact ContinuousLinearMap.integral_comp_comm (matrixEntryLinearMap i j) hZ_exp_int
    _ = (∫ ω, exp (Z ω) ∂μ) i j := by
      simp

/--
If `Z` is Hermitian a.e. and `exp Z` is integrable, then `E exp Z` is positive definite.
This supplies the positive-definite argument needed by the logarithm in HDP Lemma 5.4.9.
-/
theorem integral_exp_posDef_of_ae_isHermitian
    {μ : Measure Ω} [IsProbabilityMeasure μ] {Z : Ω → Matrix n n ℝ}
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian)
    (hZ_exp_int : Integrable (fun ω => exp (Z ω)) μ) :
    (∫ ω, exp (Z ω) ∂μ).PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos
    (integral_exp_isHermitian_of_ae_isHermitian hZ_symm hZ_exp_int) ?_
  intro x hx
  let q : Ω → ℝ := fun ω => dotProduct (star x) (Matrix.mulVec (exp (Z ω)) x)
  have hq_int : Integrable q μ := by
    simpa [q, matrixQuadraticFormLinearMap_apply] using
      (matrixQuadraticFormLinearMap x).integrable_comp hZ_exp_int
  have hq_nonneg : 0 ≤ᵐ[μ] q := by
    filter_upwards [hZ_symm] with ω hZω
    exact (Matrix.exp_posDef_of_isHermitian hZω).posSemidef.dotProduct_mulVec_nonneg x
  have hq_pos_ae : ∀ᵐ ω ∂μ, 0 < q ω := by
    filter_upwards [hZ_symm] with ω hZω
    exact (Matrix.exp_posDef_of_isHermitian hZω).dotProduct_mulVec_pos hx
  have hsupport_pos : 0 < μ (Function.support q) := by
    have hsupport_ae : Set.univ ≤ᵐ[μ] Function.support q := by
      filter_upwards [hq_pos_ae] with ω hω _
      exact ne_of_gt hω
    have h_univ_le : μ Set.univ ≤ μ (Function.support q) := measure_mono_ae hsupport_ae
    have h_univ_pos : 0 < μ Set.univ := by
      simp
    exact h_univ_pos.trans_le h_univ_le
  have hq_integral_pos : 0 < ∫ ω, q ω ∂μ := by
    exact (integral_pos_iff_support_of_nonneg_ae hq_nonneg hq_int).2 hsupport_pos
  calc
    0 < ∫ ω, q ω ∂μ := hq_integral_pos
    _ = ∫ ω, matrixQuadraticFormLinearMap x (exp (Z ω)) ∂μ := by
      exact integral_congr_ae (Filter.Eventually.of_forall fun ω =>
        (matrixQuadraticFormLinearMap_apply x (exp (Z ω))).symm)
    _ = matrixQuadraticFormLinearMap x (∫ ω, exp (Z ω) ∂μ) := by
      exact ContinuousLinearMap.integral_comp_comm (matrixQuadraticFormLinearMap x) hZ_exp_int
    _ = dotProduct (star x) (Matrix.mulVec (∫ ω, exp (Z ω) ∂μ) x) := by
      rw [matrixQuadraticFormLinearMap_apply]

/-- If `Z` is Hermitian a.e., then `E exp Z` lies in the positive-definite domain. -/
theorem integral_exp_mem_positiveDefiniteSymmetricMatrices
    {μ : Measure Ω} [IsProbabilityMeasure μ] {Z : Ω → Matrix n n ℝ}
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian)
    (hZ_exp_int : Integrable (fun ω => exp (Z ω)) μ) :
    (∫ ω, exp (Z ω) ∂μ) ∈ Matrix.positiveDefiniteSymmetricMatrices n :=
  integral_exp_posDef_of_ae_isHermitian hZ_symm hZ_exp_int

/--
If `Z` is Hermitian a.e., the integrand in HDP Lemma 5.4.9 is a.e. the Lieb trace function
evaluated at `exp Z`.
-/
theorem ae_liebTraceFunction_exp_eq
    {μ : Measure Ω} (H : Matrix n n ℝ) {Z : Ω → Matrix n n ℝ}
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian) :
    ∀ᵐ ω ∂μ,
      Matrix.liebTraceFunction H (exp (Z ω)) = Matrix.trace (exp (H + Z ω)) := by
  filter_upwards [hZ_symm] with ω hZω
  exact Matrix.liebTraceFunction_exp_eq H hZω

/-! ## HDP Lemma 5.4.9 -/

/--
Reduction of HDP Lemma 5.4.9 to the Jensen inequality for the Lieb trace function.

The hypothesis `h_lieb_jensen` is exactly the inequality obtained by applying Jensen to the
concave function `Matrix.liebTraceFunction H` and the random positive-definite matrix `exp ∘ Z`.
-/
theorem lieb_inequality_random_matrices_hdp_of_liebTraceFunction_integral_le
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : Matrix n n ℝ) (Z : Ω → Matrix n n ℝ)
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian)
    (h_lieb_jensen :
      ∫ ω, Matrix.liebTraceFunction H (exp (Z ω)) ∂μ ≤
        Matrix.liebTraceFunction H (∫ ω, exp (Z ω) ∂μ)) :
    ∫ ω, Matrix.trace (exp (H + Z ω)) ∂μ ≤
      Matrix.trace (exp (H + CFC.log (∫ ω, exp (Z ω) ∂μ))) := by
  calc
    ∫ ω, Matrix.trace (exp (H + Z ω)) ∂μ =
        ∫ ω, Matrix.liebTraceFunction H (exp (Z ω)) ∂μ := by
          exact integral_congr_ae ((ae_liebTraceFunction_exp_eq H hZ_symm).mono
            fun _ hω => hω.symm)
    _ ≤ Matrix.liebTraceFunction H (∫ ω, exp (Z ω) ∂μ) := h_lieb_jensen
    _ = Matrix.trace (exp (H + CFC.log (∫ ω, exp (Z ω) ∂μ))) := rfl

/--
The Jensen step for HDP Lemma 5.4.9 follows from deterministic Lieb concavity.

The only topological point is that `E exp Z` is positive definite, so the Lieb trace function is
continuous there relative to the positive-definite cone.
-/
theorem liebTraceFunction_integral_le_of_deterministic_lieb
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : Matrix n n ℝ) (Z : Ω → Matrix n n ℝ)
    (hH : H.IsHermitian)
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian)
    (h_trace_integrable : Integrable (fun ω => Matrix.trace (exp (H + Z ω))) μ)
    (h_exp_integrable : Integrable (fun ω => exp (Z ω)) μ) :
    ∫ ω, Matrix.liebTraceFunction H (exp (Z ω)) ∂μ ≤
      Matrix.liebTraceFunction H (∫ ω, exp (Z ω) ∂μ) := by
  have hmean_mem :
      (∫ ω, exp (Z ω) ∂μ) ∈ Matrix.positiveDefiniteSymmetricMatrices n :=
    integral_exp_mem_positiveDefiniteSymmetricMatrices hZ_symm h_exp_integrable
  have hcont : ContinuousWithinAt (Matrix.liebTraceFunction H)
      (Matrix.positiveDefiniteSymmetricMatrices n) (∫ ω, exp (Z ω) ∂μ) :=
    Matrix.continuousWithinAt_liebTraceFunction H hmean_mem
  have h_lieb_int :
      Integrable (fun ω => Matrix.liebTraceFunction H (exp (Z ω))) μ :=
    Integrable.congr h_trace_integrable
      ((ae_liebTraceFunction_exp_eq H hZ_symm).mono fun _ hω => hω.symm)
  exact concaveOn_integral_le_of_continuousWithinAt
    (Matrix.lieb_inequality_hdp_5_4_8 H hH) hcont
    (ae_exp_mem_positiveDefiniteSymmetricMatrices hZ_symm)
    h_exp_integrable (by simpa [Function.comp_def] using h_lieb_int)

/--
HDP Lemma 5.4.9, Lieb inequality for random matrices.

If `H` is fixed symmetric and `Z` is a random symmetric matrix, then

`E tr exp(H + Z) ≤ tr exp(H + log(E exp Z))`.

The integrability hypotheses make the two Bochner integrals in the formal statement non-junk.
-/
def lieb_inequality_random_matrices_hdp
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : Matrix n n ℝ) (Z : Ω → Matrix n n ℝ) : Prop :=
  H.IsHermitian →
    (∀ᵐ ω ∂μ, (Z ω).IsHermitian) →
      Integrable (fun ω => Matrix.trace (exp (H + Z ω))) μ →
        Integrable (fun ω => exp (Z ω)) μ →
          ∫ ω, Matrix.trace (exp (H + Z ω)) ∂μ ≤
            Matrix.trace (exp (H + CFC.log (∫ ω, exp (Z ω) ∂μ)))

/--
HDP Lemma 5.4.9 follows from deterministic Lieb concavity and Jensen's inequality on the
positive-definite cone.
-/
theorem lieb_inequality_random_matrices_hdp_of_deterministic_lieb
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : Matrix n n ℝ) (Z : Ω → Matrix n n ℝ) :
    lieb_inequality_random_matrices_hdp μ H Z := by
  intro hH hZ_symm h_trace_integrable h_exp_integrable
  exact lieb_inequality_random_matrices_hdp_of_liebTraceFunction_integral_le μ H Z hZ_symm
    (liebTraceFunction_integral_le_of_deterministic_lieb μ H Z hH hZ_symm
      h_trace_integrable h_exp_integrable)

/--
HDP Lemma 5.4.9, Lieb inequality for random matrices.

If `H` is fixed symmetric and `Z` is a random symmetric matrix, then

`E tr exp(H + Z) ≤ tr exp(H + log(E exp Z))`.

The integrability hypotheses make the two Bochner integrals in the formal statement non-junk.
-/
theorem lieb_inequality_random_matrices_hdp_5_4_9
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : Matrix n n ℝ) (Z : Ω → Matrix n n ℝ)
    (hH : H.IsHermitian)
    (hZ_symm : ∀ᵐ ω ∂μ, (Z ω).IsHermitian)
    (h_trace_integrable : Integrable (fun ω => Matrix.trace (exp (H + Z ω))) μ)
    (h_exp_integrable : Integrable (fun ω => exp (Z ω)) μ) :
    ∫ ω, Matrix.trace (exp (H + Z ω)) ∂μ ≤
      Matrix.trace (exp (H + CFC.log (∫ ω, exp (Z ω) ∂μ))) :=
  lieb_inequality_random_matrices_hdp_of_deterministic_lieb μ H Z
    hH hZ_symm h_trace_integrable h_exp_integrable

end RMT
