/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import Mathlib.MeasureTheory.Function.LpSeminorm.Basic
import Mathlib.MeasureTheory.Function.LpSpace.Basic
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Convolution
import Mathlib.Analysis.Calculus.BumpFunction.FiniteDimension
import Mathlib.Analysis.Calculus.BumpFunction.InnerProduct
import Mathlib.Analysis.SpecialFunctions.SmoothTransition
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Module.FiniteDimension
import SLT.GaussianMeasure

/-!
# Strong Gaussian Sobolev Space Definitions

This file defines the Gaussian Sobolev space W^{1,2}(γ) for C¹ function space and related infrastructure
for the density theorem: C_c^∞(ℝⁿ) is dense in W^{1,2}(γ) for C¹.

## Main Definitions

* `GaussianSobolevNorm` - The W^{1,2}(γ) norm: ‖f‖_{L²(γ)}² + ‖∇f‖_{L²(γ)}²
* `MemW12Gaussian` - Predicate for f ∈ W^{1,2}(γ)
* `smoothCutoff` - Standard smooth cutoff χ with χ=1 on |x|≤1, χ=0 on |x|≥2
* `smoothCutoffR` - Rescaled cutoff χ_R(x) = χ(‖x‖/R)
* `stdMollifier` - Standard mollifier ρ with ∫ρ=1, supp ⊆ B(0,1)
* `mollify` - Convolution f ⋆ ρ_ε

-/

open MeasureTheory ProbabilityTheory Filter Topology GaussianMeasure
open scoped ENNReal NNReal Topology

namespace GaussianSobolev

variable {n : ℕ}

/-- Abbreviation for n-dimensional Euclidean space -/
abbrev E (n : ℕ) := EuclideanSpace ℝ (Fin n)

/-! ### Standard Gaussian Measure -/

/-- The standard 1D Gaussian measure has IsOpenPosMeasure: it assigns positive mass
    to all nonempty open sets. This follows from `volume ≪ gaussianReal 0 1`
    (via `gaussianReal_absolutelyContinuous'`) and the fact that Lebesgue measure on ℝ
    is IsOpenPosMeasure. -/
instance : (gaussianReal 0 1).IsOpenPosMeasure :=
  (gaussianReal_absolutelyContinuous' 0 one_ne_zero).isOpenPosMeasure

/-- The standard Gaussian on EuclideanSpace has IsOpenPosMeasure. -/
instance stdGaussianE_isOpenPosMeasure : (stdGaussianE n).IsOpenPosMeasure := by
  unfold stdGaussianE GaussianMeasure.stdGaussianPi
  apply Continuous.isOpenPosMeasure_map
  · exact (EuclideanSpace.equiv (Fin n) ℝ).symm.continuous
  · exact (EuclideanSpace.equiv (Fin n) ℝ).symm.surjective

/-! ### Gaussian Sobolev Space -/

/-- The Gaussian Sobolev norm squared: ‖f‖_{L²(γ)}² + ‖∇f‖_{L²(γ)}² -/
noncomputable def GaussianSobolevNormSq (n : ℕ) (f : E n → ℝ)
    (γ : Measure (E n)) : ℝ≥0∞ :=
  eLpNorm f 2 γ ^ 2 + eLpNorm (fun x ↦ ‖fderiv ℝ f x‖) 2 γ ^ 2

/-- The Gaussian Sobolev norm: (‖f‖_{L²(γ)}² + ‖∇f‖_{L²(γ)}²)^{1/2} -/
noncomputable def GaussianSobolevNorm (n : ℕ) (f : E n → ℝ)
    (γ : Measure (E n)) : ℝ≥0∞ :=
  (GaussianSobolevNormSq n f γ) ^ (1/2 : ℝ)

/-- Membership in W^{1,2}(γ): f ∈ L²(γ) and ∇f ∈ L²(γ) -/
def MemW12Gaussian (n : ℕ) (f : E n → ℝ) (γ : Measure (E n)) : Prop :=
  MemLp f 2 γ ∧ MemLp (fun x ↦ fderiv ℝ f x) 2 γ

lemma memW12Gaussian_iff_sobolevNormSq_lt_top {f : E n → ℝ}
    {γ : Measure (E n)} :
    MemW12Gaussian n f γ ↔
      GaussianSobolevNormSq n f γ < ⊤ ∧
      AEStronglyMeasurable f γ ∧
      AEStronglyMeasurable (fun x ↦ fderiv ℝ f x) γ := by
  simp only [MemW12Gaussian, GaussianSobolevNormSq, MemLp]
  -- Use: eLpNorm (‖g ·‖) p μ = eLpNorm g p μ
  have heq : eLpNorm (fun x ↦ ‖fderiv ℝ f x‖) 2 γ = eLpNorm (fun x ↦ fderiv ℝ f x) 2 γ :=
    MeasureTheory.eLpNorm_norm (fun x ↦ fderiv ℝ f x)
  rw [heq]
  constructor
  · rintro ⟨⟨hf_meas, hf_norm⟩, hdf_meas, hdf_norm⟩
    refine ⟨?_, hf_meas, hdf_meas⟩
    exact ENNReal.add_lt_top.mpr ⟨ENNReal.pow_lt_top hf_norm, ENNReal.pow_lt_top hdf_norm⟩
  · rintro ⟨hsum, hf_meas, hdf_meas⟩
    have ⟨h1, h2⟩ := ENNReal.add_lt_top.mp hsum
    refine ⟨⟨hf_meas, ?_⟩, hdf_meas, ?_⟩
    · exact (ENNReal.pow_lt_top_iff.mp h1).resolve_right (by decide)
    · exact (ENNReal.pow_lt_top_iff.mp h2).resolve_right (by decide)

/-! ### Smooth Cutoff Functions -/

/-- Standard smooth cutoff function: χ(x) = 1 for |x| ≤ 1, χ(x) = 0 for |x| ≥ 2. -/
noncomputable def smoothCutoff : ℝ → ℝ :=
  (ContDiffBumpBase.ofInnerProductSpace ℝ).toFun 2

lemma smoothCutoff_eq_smoothTransition (x : ℝ) :
    smoothCutoff x = Real.smoothTransition (2 - |x|) := by
  have hden : (2 : ℝ) - 1 = 1 := by norm_num
  simp [smoothCutoff, ContDiffBumpBase.ofInnerProductSpace, Real.norm_eq_abs, hden]

lemma smoothCutoff_contDiff : ContDiff ℝ (⊤ : ℕ∞) smoothCutoff := by
  rw [contDiff_iff_contDiffAt]
  intro x
  have hmem : ((2 : ℝ), x) ∈ (Set.Ioi (1 : ℝ) ×ˢ (Set.univ : Set ℝ)) := by
    refine ⟨?_, by simp⟩
    norm_num
  have hnhds : (Set.Ioi (1 : ℝ) ×ˢ (Set.univ : Set ℝ)) ∈ 𝓝 ((2 : ℝ), x) := by
    have hopen : IsOpen (Set.Ioi (1 : ℝ) ×ˢ (Set.univ : Set ℝ)) :=
      IsOpen.prod isOpen_Ioi isOpen_univ
    exact hopen.mem_nhds hmem
  have h_at :
      ContDiffAt ℝ (⊤ : ℕ∞)
        (Function.uncurry (ContDiffBumpBase.ofInnerProductSpace ℝ).toFun) ((2 : ℝ), x) :=
    (ContDiffBumpBase.ofInnerProductSpace ℝ).smooth.contDiffAt hnhds
  have h_comp : ContDiffAt ℝ (⊤ : ℕ∞) (fun y : ℝ => ((2 : ℝ), y)) x :=
    (contDiffAt_const (c := (2 : ℝ))).prodMk contDiffAt_id
  change ContDiffAt ℝ (⊤ : ℕ∞)
    (fun y : ℝ => Function.uncurry (ContDiffBumpBase.ofInnerProductSpace ℝ).toFun ((2 : ℝ), y)) x
  exact h_at.comp x h_comp

lemma smoothCutoff_eq_one_of_le {x : ℝ} (hx : |x| ≤ 1) : smoothCutoff x = 1 := by
  have h : (1 : ℝ) ≤ 2 - |x| := by linarith
  simpa [smoothCutoff_eq_smoothTransition] using
    (Real.smoothTransition.one_of_one_le (x := 2 - |x|) h)

lemma smoothCutoff_eq_zero_of_ge {x : ℝ} (hx : 2 ≤ |x|) : smoothCutoff x = 0 := by
  have h : 2 - |x| ≤ 0 := by linarith
  simpa [smoothCutoff_eq_smoothTransition] using
    (Real.smoothTransition.zero_of_nonpos (x := 2 - |x|) h)

lemma smoothCutoff_nonneg (x : ℝ) : 0 ≤ smoothCutoff x := by
  simpa [smoothCutoff_eq_smoothTransition] using
    (Real.smoothTransition.nonneg (x := 2 - |x|))

lemma smoothCutoff_le_one (x : ℝ) : smoothCutoff x ≤ 1 := by
  simpa [smoothCutoff_eq_smoothTransition] using
    (Real.smoothTransition.le_one (x := 2 - |x|))

lemma smoothCutoff_eq_one_iff_abs_le {x : ℝ} : smoothCutoff x = 1 ↔ |x| ≤ 1 := by
  have h' : (1 : ℝ) ≤ 2 - |x| ↔ |x| ≤ 1 := by
    constructor
    · intro h
      linarith
    · intro h
      have h' : 1 + |x| ≤ 2 := by linarith
      exact (le_sub_iff_add_le).2 h'
  simp [smoothCutoff_eq_smoothTransition, h']

/-- The smooth cutoff is strictly less than 1 for |x| > 1.

    Proof strategy:
    - If 1 < |x|, then 2 - |x| < 1
    - `Real.smoothTransition` is strictly less than 1 on inputs < 1 -/
lemma smoothCutoff_lt_one_of_abs_gt_one {x : ℝ} (hx1 : 1 < |x|) :
    smoothCutoff x < 1 := by
  have hlt : 2 - |x| < 1 := by linarith
  simpa [smoothCutoff_eq_smoothTransition] using
    (Real.smoothTransition.lt_one_of_lt_one (x := 2 - |x|) hlt)

lemma smoothCutoff_support : Function.support smoothCutoff ⊆ Set.Icc (-2) 2 := by
  intro x hx
  simp only [Function.mem_support, ne_eq] at hx
  by_contra h
  simp only [Set.mem_Icc, not_and_or, not_le] at h
  cases h with
  | inl h =>
    have : 2 ≤ |x| := by simp [abs_of_neg (lt_trans h (neg_lt_zero.mpr two_pos))]; linarith
    exact hx (smoothCutoff_eq_zero_of_ge this)
  | inr h =>
    have : 2 ≤ |x| := by simp [abs_of_pos (lt_trans two_pos h)]; linarith
    exact hx (smoothCutoff_eq_zero_of_ge this)

/-- Rescaled smooth cutoff: χ_R(x) = χ(‖x‖/R) where χ is the standard cutoff -/
noncomputable def smoothCutoffR (R : ℝ) (x : E n) : ℝ :=
  smoothCutoff (‖x‖ / R)

lemma smoothCutoffR_eq_one_of_norm_le {R : ℝ} (hR : 0 < R) {x : E n}
    (hx : ‖x‖ ≤ R) : smoothCutoffR R x = 1 := by
  unfold smoothCutoffR
  apply smoothCutoff_eq_one_of_le
  rw [abs_of_nonneg (div_nonneg (norm_nonneg _) hR.le)]
  exact div_le_one_of_le₀ hx hR.le

lemma smoothCutoffR_eq_zero_of_norm_ge {R : ℝ} (hR : 0 < R) {x : E n}
    (hx : 2 * R ≤ ‖x‖) : smoothCutoffR R x = 0 := by
  unfold smoothCutoffR
  apply smoothCutoff_eq_zero_of_ge
  rw [abs_of_nonneg (div_nonneg (norm_nonneg _) hR.le)]
  calc 2 = 2 * R / R := by field_simp
    _ ≤ ‖x‖ / R := div_le_div_of_nonneg_right hx hR.le

lemma smoothCutoffR_nonneg (R : ℝ) (x : E n) : 0 ≤ smoothCutoffR R x :=
  smoothCutoff_nonneg _

lemma smoothCutoffR_le_one (R : ℝ) (x : E n) : smoothCutoffR R x ≤ 1 :=
  smoothCutoff_le_one _

lemma smoothCutoffR_contDiff {R : ℝ} (hR : 0 < R) : ContDiff ℝ (⊤ : ℕ∞) (smoothCutoffR R : E n → ℝ) := by
  rw [contDiff_iff_contDiffAt]
  intro x
  by_cases hx : ‖x‖ < R
  · -- Case: ‖x‖ < R, so smoothCutoffR R is locally constant = 1
    have h_eq : ∀ᶠ y in nhds x, smoothCutoffR R y = 1 := by
      have hmem : Metric.ball x (R - ‖x‖) ∈ nhds x := Metric.ball_mem_nhds x (by linarith)
      filter_upwards [hmem] with y hy
      apply smoothCutoffR_eq_one_of_norm_le hR
      rw [Metric.mem_ball, dist_eq_norm] at hy
      have h2 : ‖y‖ ≤ ‖x‖ + ‖y - x‖ := by
        calc ‖y‖ = ‖x + (y - x)‖ := by congr 1; abel
          _ ≤ ‖x‖ + ‖y - x‖ := norm_add_le x (y - x)
      linarith
    have h_const : ContDiffAt ℝ (⊤ : ℕ∞) (fun _ : E n => (1 : ℝ)) x := contDiffAt_const
    apply h_const.congr_of_eventuallyEq
    filter_upwards [h_eq] with y hy
    simp [hy]
  · -- Case: ‖x‖ ≥ R, so x ≠ 0 and norm is smooth at x (EuclideanSpace has InnerProductSpace)
    push Not at hx
    have hx_ne : x ≠ 0 := by
      intro h
      rw [h, norm_zero] at hx
      linarith
    unfold smoothCutoffR
    -- Compose: smoothCutoff ∘ (· / R) ∘ norm
    -- norm is smooth at x ≠ 0 for InnerProductSpace
    have h_norm : ContDiffAt ℝ (⊤ : ℕ∞) (fun y : E n => ‖y‖) x := contDiffAt_norm ℝ hx_ne
    have h_div : ContDiff ℝ (⊤ : ℕ∞) (fun t : ℝ => t / R) := contDiff_id.div_const R
    have h_comp1 : ContDiffAt ℝ (⊤ : ℕ∞) (fun y : E n => ‖y‖ / R) x :=
      h_div.contDiffAt.comp x h_norm
    exact smoothCutoff_contDiff.contDiffAt.comp x h_comp1

lemma smoothCutoffR_support_subset {R : ℝ} (hR : 0 < R) :
    Function.support (smoothCutoffR R : E n → ℝ) ⊆ Metric.closedBall 0 (2 * R) := by
  intro x hx
  simp only [Metric.mem_closedBall, dist_zero_right]
  by_contra h
  push Not at h
  have : smoothCutoffR R x = 0 := smoothCutoffR_eq_zero_of_norm_ge hR h.le
  simp only [Function.mem_support, this, ne_eq, not_true_eq_false] at hx

lemma smoothCutoffR_hasCompactSupport {R : ℝ} (hR : 0 < R) :
    HasCompactSupport (smoothCutoffR R : E n → ℝ) := by
  rw [hasCompactSupport_def]
  apply IsCompact.of_isClosed_subset (isCompact_closedBall 0 (2 * R)) isClosed_closure
  calc closure (Function.support (smoothCutoffR R))
      ⊆ closure (Metric.closedBall 0 (2 * R)) := closure_mono (smoothCutoffR_support_subset hR)
    _ = Metric.closedBall 0 (2 * R) := IsClosed.closure_eq Metric.isClosed_closedBall

/-- For any fixed x, smoothCutoffR R x → 1 as R → ∞.
    Proof: For R > ‖x‖, we have ‖x‖/R < 1, so smoothCutoff(‖x‖/R) = 1. -/
lemma smoothCutoffR_tendsto_one (x : E n) :
    Filter.Tendsto (fun R => smoothCutoffR R x) Filter.atTop (nhds 1) := by
  -- For R > ‖x‖ + 1, smoothCutoffR R x = 1 (eventually constant)
  apply tendsto_atTop_of_eventually_const (i₀ := ‖x‖ + 1)
  intro R hR
  have hR_pos : 0 < R := by
    calc 0 < ‖x‖ + 1 := by positivity
      _ ≤ R := hR
  have hxR : ‖x‖ < R := by linarith
  exact smoothCutoffR_eq_one_of_norm_le hR_pos (le_of_lt hxR)

/-! ### Standard Mollifier -/

/-- Standard mollifier kernel on ℝ: c · exp(-1/(1-x²)) for |x| < 1, 0 otherwise -/
noncomputable def stdMollifierKernel (x : ℝ) : ℝ :=
  if |x| < 1 then Real.exp (-1 / (1 - x^2)) else 0

/-- Normalization constant for the mollifier so that ∫ ρ = 1 -/
noncomputable def stdMollifierConst : ℝ :=
  (∫ x in Set.Ioo (-1 : ℝ) 1, stdMollifierKernel x)⁻¹

/-- Standard mollifier: normalized smooth function with supp ⊆ B(0,1) and ∫ρ = 1 -/
noncomputable def stdMollifier (x : ℝ) : ℝ :=
  stdMollifierConst * stdMollifierKernel x

lemma stdMollifierKernel_nonneg (x : ℝ) : 0 ≤ stdMollifierKernel x := by
  unfold stdMollifierKernel
  split_ifs with h
  · exact Real.exp_nonneg _
  · rfl

lemma stdMollifierKernel_pos_of_abs_lt_one {x : ℝ} (hx : |x| < 1) : 0 < stdMollifierKernel x := by
  unfold stdMollifierKernel
  simp only [hx, ↓reduceIte]
  exact Real.exp_pos _

/-- {x | |x| < 1} equals Metric.ball 0 1 in ℝ -/
lemma abs_lt_one_eq_ball : {x : ℝ | |x| < 1} = Metric.ball 0 1 := by
  ext x
  simp only [Set.mem_setOf_eq, Metric.mem_ball, dist_zero_right, Real.norm_eq_abs]

/-- The mollifier kernel is measurable -/
lemma stdMollifierKernel_measurable : Measurable stdMollifierKernel := by
  unfold stdMollifierKernel
  have h_meas_set : MeasurableSet {x : ℝ | |x| < 1} := by
    rw [abs_lt_one_eq_ball]; exact measurableSet_ball
  refine Measurable.ite h_meas_set ?_ ?_
  · apply Measurable.exp
    apply Measurable.div measurable_const
    exact measurable_const.sub (measurable_id.pow_const 2)
  · exact measurable_const

/-- On (-1, 1), stdMollifierKernel equals exp(-1/(1-x²)) -/
lemma stdMollifierKernel_eq_on_Ioo {x : ℝ} (hx : x ∈ Set.Ioo (-1 : ℝ) 1) :
    stdMollifierKernel x = Real.exp (-1 / (1 - x^2)) := by
  unfold stdMollifierKernel
  simp only [Set.mem_Ioo] at hx
  have h : |x| < 1 := abs_lt.mpr hx
  simp only [h, ↓reduceIte]

/-- The mollifier kernel is bounded by 1 -/
lemma stdMollifierKernel_le_one (x : ℝ) : stdMollifierKernel x ≤ 1 := by
  unfold stdMollifierKernel
  split_ifs with h
  · rw [Real.exp_le_one_iff]
    apply div_nonpos_of_nonpos_of_nonneg (by norm_num : (-1 : ℝ) ≤ 0)
    have hx2 : x^2 < 1 := by
      have h1 : |x|^2 = x^2 := sq_abs x
      have h2 : |x|^2 < 1 := by
        have habs_nn : 0 ≤ |x| := abs_nonneg x
        calc |x|^2 = |x| * |x| := sq |x|
          _ < 1 * 1 := by nlinarith
          _ = 1 := one_mul 1
      linarith [h1, h2]
    linarith
  · exact zero_le_one

/-- The mollifier kernel is integrable on (-1,1) -/
lemma stdMollifierKernel_integrableOn_Ioo : IntegrableOn stdMollifierKernel (Set.Ioo (-1) 1) := by
  refine IntegrableOn.of_bound ?_ stdMollifierKernel_measurable.aestronglyMeasurable.restrict 1 ?_
  · rw [Real.volume_Ioo]
    simp only [sub_neg_eq_add, one_add_one_eq_two]
    exact ENNReal.ofReal_lt_top
  · filter_upwards with x
    rw [Real.norm_eq_abs, abs_of_nonneg (stdMollifierKernel_nonneg x)]
    exact stdMollifierKernel_le_one x

/-- The integral of the mollifier kernel over (-1,1) is positive -/
lemma stdMollifierKernel_integral_pos :
    0 < ∫ x in Set.Ioo (-1 : ℝ) 1, stdMollifierKernel x := by
  rw [MeasureTheory.setIntegral_pos_iff_support_of_nonneg_ae]
  · -- The support intersects Ioo (-1) 1 with positive measure
    have h_supp : Set.Ioo (-1 : ℝ) 1 ⊆ Function.support stdMollifierKernel := by
      intro x hx
      simp only [Function.mem_support, ne_eq]
      have : 0 < stdMollifierKernel x := by
        simp only [Set.mem_Ioo] at hx
        exact stdMollifierKernel_pos_of_abs_lt_one (abs_lt.mpr hx)
      linarith
    calc volume (Function.support stdMollifierKernel ∩ Set.Ioo (-1) 1)
        = volume (Set.Ioo (-1 : ℝ) 1) := by
          congr 1
          ext x
          constructor
          · exact fun ⟨_, h2⟩ => h2
          · exact fun h => ⟨h_supp h, h⟩
      _ > 0 := by
          rw [Real.volume_Ioo]
          simp only [sub_neg_eq_add, one_add_one_eq_two, ENNReal.ofReal_pos]
          norm_num
  · -- Nonneg a.e.
    filter_upwards with x
    exact stdMollifierKernel_nonneg x
  · -- Integrable
    exact stdMollifierKernel_integrableOn_Ioo

lemma stdMollifierConst_pos : 0 < stdMollifierConst := by
  unfold stdMollifierConst
  exact inv_pos.mpr stdMollifierKernel_integral_pos

lemma stdMollifierConst_nonneg : 0 ≤ stdMollifierConst :=
  le_of_lt stdMollifierConst_pos

lemma stdMollifier_nonneg (x : ℝ) : 0 ≤ stdMollifier x := by
  unfold stdMollifier
  exact mul_nonneg stdMollifierConst_nonneg (stdMollifierKernel_nonneg x)

lemma stdMollifier_support : Function.support stdMollifier ⊆ Set.Ioo (-1) 1 := by
  intro x hx
  simp only [Function.mem_support, ne_eq] at hx
  unfold stdMollifier stdMollifierKernel at hx
  split_ifs at hx with h
  · simp only [Set.mem_Ioo, abs_lt] at h ⊢
    exact h
  · simp at hx

lemma stdMollifier_hasCompactSupport : HasCompactSupport stdMollifier := by
  rw [hasCompactSupport_def]
  apply IsCompact.of_isClosed_subset (isCompact_Icc (a := -1) (b := 1)) isClosed_closure
  calc closure (Function.support stdMollifier)
      ⊆ closure (Set.Ioo (-1) 1) := closure_mono stdMollifier_support
    _ = Set.Icc (-1) 1 := closure_Ioo (ne_of_lt (neg_lt_self one_pos))

/-! ### Smoothness of the Standard Mollifier

The key insight is that stdMollifierKernel x = expNegInvGlue (1 - x²).
Since expNegInvGlue is smooth (from Mathlib) and x ↦ 1 - x² is smooth (polynomial),
their composition is smooth.
-/

/-- stdMollifierKernel equals expNegInvGlue composed with 1 - x² -/
lemma stdMollifierKernel_eq_expNegInvGlue (x : ℝ) :
    stdMollifierKernel x = expNegInvGlue (1 - x^2) := by
  unfold stdMollifierKernel
  by_cases h : |x| < 1
  · -- When |x| < 1, we have 1 - x² > 0, so expNegInvGlue gives exp(...)
    simp only [h, ↓reduceIte]
    have habs_sq : |x|^2 < 1 := by nlinarith [h, abs_nonneg x]
    have hx2 : x^2 < 1 := by rwa [sq_abs] at habs_sq
    have hpos : 0 < 1 - x^2 := by linarith
    -- expNegInvGlue y = if y ≤ 0 then 0 else exp(-y⁻¹)
    simp only [expNegInvGlue, not_le.mpr hpos, ↓reduceIte]
    congr 1
    field_simp
  · -- When |x| ≥ 1, we have 1 - x² ≤ 0, so expNegInvGlue gives 0
    simp only [h, ↓reduceIte]
    have habs_sq : 1 ≤ |x|^2 := by
      push Not at h
      nlinarith [h, abs_nonneg x]
    have hx2 : 1 ≤ x^2 := by rwa [sq_abs] at habs_sq
    have hnonpos : 1 - x^2 ≤ 0 := by linarith
    exact (expNegInvGlue.zero_of_nonpos hnonpos).symm

/-- The mollifier kernel is smooth (C^∞). -/
lemma stdMollifierKernel_contDiff : ContDiff ℝ (⊤ : ℕ∞) stdMollifierKernel := by
  have h : stdMollifierKernel = expNegInvGlue ∘ (fun x => 1 - x^2) := by
    ext x; exact stdMollifierKernel_eq_expNegInvGlue x
  rw [h]
  have h_all : ∀ n : ℕ, ContDiff ℝ n (expNegInvGlue ∘ (fun x => 1 - x^2)) := by
    intro n
    apply ContDiff.comp (expNegInvGlue.contDiff (n := n))
    exact contDiff_const.sub (contDiff_id.pow 2)
  exact contDiff_infty.2 h_all

/-- The standard mollifier is smooth (C^∞) -/
lemma stdMollifier_contDiff : ContDiff ℝ (⊤ : ℕ∞) stdMollifier := by
  unfold stdMollifier
  exact contDiff_const.mul stdMollifierKernel_contDiff

lemma stdMollifierKernel_eq_zero_outside {x : ℝ} (hx : x ∉ Set.Ioo (-1 : ℝ) 1) :
    stdMollifierKernel x = 0 := by
  unfold stdMollifierKernel
  simp only [Set.mem_Ioo, not_and_or, not_lt] at hx
  have habs : ¬ |x| < 1 := by
    simp only [abs_lt, not_and_or, not_lt]
    cases hx with
    | inl h => left; exact h
    | inr h => right; exact h
  simp only [habs, ↓reduceIte]

lemma stdMollifier_eq_zero_outside {x : ℝ} (hx : x ∉ Set.Ioo (-1 : ℝ) 1) :
    stdMollifier x = 0 := by
  unfold stdMollifier
  simp only [stdMollifierKernel_eq_zero_outside hx, mul_zero]

lemma stdMollifier_measurable : Measurable stdMollifier := by
  exact measurable_const.mul stdMollifierKernel_measurable

lemma stdMollifier_integrable : Integrable stdMollifier := by
  have h : IntegrableOn stdMollifier (Set.Ioo (-1) 1) := by
    refine IntegrableOn.of_bound ?_ ?_ stdMollifierConst ?_
    · rw [Real.volume_Ioo]; simp
    · exact stdMollifier_measurable.aestronglyMeasurable.restrict
    · filter_upwards with x
      rw [Real.norm_eq_abs, abs_of_nonneg (stdMollifier_nonneg x)]
      unfold stdMollifier
      calc stdMollifierConst * stdMollifierKernel x
          ≤ stdMollifierConst * 1 := by
            apply mul_le_mul_of_nonneg_left (stdMollifierKernel_le_one x) stdMollifierConst_nonneg
        _ = stdMollifierConst := mul_one _
  have h_eq : stdMollifier = Set.indicator (Set.Ioo (-1) 1) stdMollifier := by
    ext x
    by_cases hx : x ∈ Set.Ioo (-1 : ℝ) 1
    · simp only [Set.indicator_apply, hx, ↓reduceIte]
    · simp only [Set.indicator_apply, hx, ↓reduceIte, stdMollifier_eq_zero_outside hx]
  rw [h_eq]
  exact IntegrableOn.integrable_indicator h measurableSet_Ioo

lemma stdMollifier_integral_eq_one : ∫ x, stdMollifier x = 1 := by
  -- The integral is just over the support Ioo (-1) 1
  have h_eq : ∫ x, stdMollifier x = ∫ x in Set.Ioo (-1 : ℝ) 1, stdMollifier x := by
    symm
    rw [← MeasureTheory.integral_indicator measurableSet_Ioo]
    congr 1
    ext x
    by_cases hx : x ∈ Set.Ioo (-1 : ℝ) 1
    · simp only [Set.indicator_apply, hx, ↓reduceIte]
    · simp only [Set.indicator_apply, hx, ↓reduceIte, stdMollifier_eq_zero_outside hx]
  rw [h_eq]
  -- Unfold and rewrite as constant times kernel
  simp only [stdMollifier]
  -- Pull the constant out of the set integral
  rw [MeasureTheory.integral_const_mul]
  -- Now show constant * kernel_integral = 1
  unfold stdMollifierConst
  rw [inv_mul_cancel₀]
  exact ne_of_gt stdMollifierKernel_integral_pos

/-- Scaled mollifier: ρ_ε(x) = ε^{-1} ρ(x/ε) -/
noncomputable def stdMollifierEps (ε : ℝ) (x : ℝ) : ℝ :=
  ε⁻¹ * stdMollifier (x / ε)

lemma stdMollifierEps_nonneg {ε : ℝ} (hε : 0 < ε) (x : ℝ) : 0 ≤ stdMollifierEps ε x := by
  unfold stdMollifierEps
  apply mul_nonneg (inv_nonneg.mpr hε.le) (stdMollifier_nonneg _)

lemma stdMollifierEps_eq_zero_of_abs_ge {ε : ℝ} (hε : 0 < ε) {x : ℝ} (hx : ε ≤ |x|) :
    stdMollifierEps ε x = 0 := by
  unfold stdMollifierEps
  have h : x / ε ∉ Set.Ioo (-1 : ℝ) 1 := by
    simp only [Set.mem_Ioo, not_and, not_lt]
    intro hleft
    have hx_pos : x > -ε := by
      have h1 : (-1 : ℝ) * ε < x := by
        calc (-1 : ℝ) * ε = -ε := by ring
          _ = (-ε / ε) * ε := by rw [div_mul_cancel₀]; exact ne_of_gt hε
          _ < (x / ε) * ε := by apply mul_lt_mul_of_pos_right; rw [neg_div]; simp [ne_of_gt hε]; exact hleft; exact hε
          _ = x := by rw [div_mul_cancel₀]; exact ne_of_gt hε
      linarith
    have hx_ge : x ≥ ε := by
      rcases abs_cases x with ⟨habs, _⟩ | ⟨habs, hxneg⟩
      · rw [habs] at hx; exact hx
      · rw [habs] at hx; linarith
    have h3 : ε / ε ≤ x / ε := by
      apply div_le_div_of_nonneg_right hx_ge hε.le
    simp only [div_self (ne_of_gt hε)] at h3
    exact h3
  rw [stdMollifier_eq_zero_outside h, mul_zero]

lemma stdMollifierEps_support {ε : ℝ} (hε : 0 < ε) :
    Function.support (stdMollifierEps ε) ⊆ Set.Ioo (-ε) ε := by
  intro x hx
  simp only [Function.mem_support] at hx
  by_contra h
  simp only [Set.mem_Ioo, not_and, not_lt] at h
  -- h : -ε < x → ε ≤ x
  -- We need |x| ≥ ε
  have habs : ε ≤ |x| := by
    by_cases hxneg : x < 0
    · -- x < 0, so |x| = -x, and we need ε ≤ -x, i.e., x ≤ -ε
      rw [abs_of_neg hxneg]
      by_contra hc; push Not at hc
      -- hc : -x < ε, i.e., -ε < x
      have : ε ≤ x := h (by linarith)
      linarith
    · -- x ≥ 0, so |x| = x, and h gives ε ≤ x when -ε < x
      push Not at hxneg
      rw [abs_of_nonneg hxneg]
      exact h (by linarith)
  exact hx (stdMollifierEps_eq_zero_of_abs_ge hε habs)

lemma stdMollifierEps_integral_eq_one {ε : ℝ} (hε : 0 < ε) :
    ∫ x, stdMollifierEps ε x = 1 := by
  unfold stdMollifierEps
  rw [MeasureTheory.integral_const_mul]
  -- Change of variables: x/ε = ε⁻¹ • x
  have h_eq : (fun x => stdMollifier (x / ε)) = (fun x => stdMollifier (ε⁻¹ • x)) := by
    ext x; simp [div_eq_inv_mul, smul_eq_mul]
  rw [h_eq]
  -- Use integral_comp_smul: ∫ f(R • x) = |R^{-dim}| • ∫ f(x)
  -- For dim = 1: ∫ f(ε⁻¹ • x) = |ε| • ∫ f(x)
  rw [MeasureTheory.Measure.integral_comp_smul_of_nonneg volume stdMollifier ε⁻¹ (hR := inv_nonneg.mpr hε.le)]
  have hdim : Module.finrank ℝ ℝ = 1 := Module.finrank_self ℝ
  simp only [hdim, pow_one, inv_inv, smul_eq_mul, stdMollifier_integral_eq_one, mul_one]
  field_simp

/-- The scaled mollifier is smooth (C^∞) -/
lemma stdMollifierEps_contDiff {ε : ℝ} (_hε : 0 < ε) : ContDiff ℝ (⊤ : ℕ∞) (stdMollifierEps ε) := by
  unfold stdMollifierEps
  apply ContDiff.mul contDiff_const
  exact stdMollifier_contDiff.comp (contDiff_id.div_const ε)

/-- n-dimensional mollifier: product of 1D mollifiers -/
noncomputable def stdMollifierPi (n : ℕ) (ε : ℝ) (x : E n) : ℝ :=
  ∏ i, stdMollifierEps ε (x i)

lemma stdMollifierPi_nonneg (ε : ℝ) (hε : 0 < ε) (x : E n) : 0 ≤ stdMollifierPi n ε x := by
  unfold stdMollifierPi stdMollifierEps
  apply Finset.prod_nonneg
  intro i _
  apply mul_nonneg
  · exact inv_nonneg.mpr hε.le
  · exact stdMollifier_nonneg _

lemma stdMollifierPi_hasCompactSupport {ε : ℝ} (hε : 0 < ε) :
    HasCompactSupport (stdMollifierPi n ε) := by
  -- The support is contained in the cube {x : ∀ i, |x i| < ε}
  -- which is contained in closedBall 0 (ε * √n)
  apply HasCompactSupport.intro' (K := Metric.closedBall (0 : E n) (ε * Real.sqrt n))
  · exact isCompact_closedBall 0 _
  · exact Metric.isClosed_closedBall
  · intro x hx
    simp only [Metric.mem_closedBall, dist_zero_right] at hx
    push Not at hx
    unfold stdMollifierPi
    -- If ‖x‖ > ε√n, by pigeonhole some |x i| ≥ ε
    -- Proof: if all |x i| < ε, then ‖x‖² = ∑ᵢ (x i)² < n * ε² = (ε√n)², contradiction
    by_cases hn : n = 0
    · -- For n = 0, E 0 is trivial and ‖x‖ = 0, contradicting hx : 0 < ‖x‖
      subst hn
      have hnorm_zero : ‖x‖ = 0 := by
        simp only [EuclideanSpace.norm_eq, Finset.univ_eq_empty, Finset.sum_empty, Real.sqrt_zero]
      simp only [Nat.cast_zero, Real.sqrt_zero, mul_zero, hnorm_zero] at hx
      exact absurd hx (lt_irrefl 0)
    · have hexists : ∃ i : Fin n, ε ≤ |x i| := by
        by_contra h
        push Not at h
        have hbound : ‖x‖^2 ≤ n * ε^2 := by
          have hnorm_sq : ‖x‖^2 = ∑ i : Fin n, ‖x i‖^2 := EuclideanSpace.norm_sq_eq x
          rw [hnorm_sq]
          calc ∑ i : Fin n, ‖x i‖^2
              ≤ ∑ i : Fin n, ε^2 := Finset.sum_le_sum (fun i _ => by
                  have hi := h i
                  simp only [Real.norm_eq_abs]
                  -- Since |x i| ≥ 0 and ε ≥ 0, use sq_le_sq₀
                  exact (sq_le_sq₀ (abs_nonneg _) hε.le).mpr (le_of_lt hi))
            _ = n * ε^2 := by simp [Finset.sum_const]
        have hsqrt : ‖x‖ ≤ ε * Real.sqrt n := by
          have h1 : ‖x‖^2 ≤ (ε * Real.sqrt n)^2 := by
            calc ‖x‖^2 ≤ n * ε^2 := hbound
              _ = ε^2 * n := by ring
              _ = (ε * Real.sqrt n)^2 := by
                  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n)]
          have heps_sqrt_nn : 0 ≤ ε * Real.sqrt n := by positivity
          have h2 := abs_le_of_sq_le_sq h1 heps_sqrt_nn
          rwa [abs_of_nonneg (norm_nonneg x)] at h2
        linarith
      obtain ⟨i, hi⟩ := hexists
      have hzero : stdMollifierEps ε (x i) = 0 := stdMollifierEps_eq_zero_of_abs_ge hε hi
      exact Finset.prod_eq_zero (Finset.mem_univ i) hzero

/-- The tsupport of the n-dimensional mollifier is contained in a closed ball of radius ε√n -/
lemma stdMollifierPi_tsupport_subset {ε : ℝ} (hε : 0 < ε) :
    tsupport (stdMollifierPi n ε) ⊆ Metric.closedBall (0 : E n) (ε * Real.sqrt n) := by
  -- The tsupport ⊆ closure K where K = closedBall 0 (ε * √n) was used in HasCompactSupport.intro'
  -- Since closedBall is closed, its closure is itself
  intro x hx
  by_contra hx_not_in
  simp only [Metric.mem_closedBall, dist_zero_right, not_le] at hx_not_in
  -- For x with ‖x‖ > ε√n, we show stdMollifierPi n ε = 0 in a neighborhood of x
  -- Hence x ∉ closure(support), i.e., x ∉ tsupport
  have h_zero_nhd : ∀ᶠ y in 𝓝 x, stdMollifierPi n ε y = 0 := by
    -- The set {y | ‖y‖ > ε * √n} is open and contains x
    have h_open : IsOpen {y : E n | ε * Real.sqrt n < ‖y‖} :=
      isOpen_lt continuous_const continuous_norm
    filter_upwards [h_open.mem_nhds hx_not_in] with y hy
    -- For y with ‖y‖ > ε * √n, use the same pigeonhole argument
    by_cases hn : n = 0
    · subst hn
      simp only [EuclideanSpace.norm_eq, Finset.univ_eq_empty, Finset.sum_empty,
        Real.sqrt_zero, Nat.cast_zero, mul_zero] at hy
      exact absurd hy (lt_irrefl _)
    · have hexists : ∃ i : Fin n, ε ≤ |y i| := by
        by_contra h
        push Not at h
        have hbound : ‖y‖^2 ≤ n * ε^2 := by
          have hnorm_sq : ‖y‖^2 = ∑ i : Fin n, ‖y i‖^2 := EuclideanSpace.norm_sq_eq y
          rw [hnorm_sq]
          calc ∑ i : Fin n, ‖y i‖^2
              ≤ ∑ i : Fin n, ε^2 := Finset.sum_le_sum (fun i _ => by
                  have hi := h i
                  simp only [Real.norm_eq_abs]
                  exact (sq_le_sq₀ (abs_nonneg _) hε.le).mpr (le_of_lt hi))
            _ = n * ε^2 := by simp [Finset.sum_const]
        have hsqrt : ‖y‖ ≤ ε * Real.sqrt n := by
          have h1 : ‖y‖^2 ≤ (ε * Real.sqrt n)^2 := by
            calc ‖y‖^2 ≤ n * ε^2 := hbound
              _ = ε^2 * n := by ring
              _ = (ε * Real.sqrt n)^2 := by
                  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n)]
          have heps_sqrt_nn : 0 ≤ ε * Real.sqrt n := by positivity
          have h2 := abs_le_of_sq_le_sq h1 heps_sqrt_nn
          rwa [abs_of_nonneg (norm_nonneg y)] at h2
        linarith
      obtain ⟨i, hi⟩ := hexists
      have hzero : stdMollifierEps ε (y i) = 0 := stdMollifierEps_eq_zero_of_abs_ge hε hi
      exact Finset.prod_eq_zero (Finset.mem_univ i) hzero
  have h_not_in_support : ∀ᶠ y in 𝓝 x, y ∉ Function.support (stdMollifierPi n ε) := by
    filter_upwards [h_zero_nhd] with y hy
    simp only [Function.mem_support, not_not, hy]
  rw [tsupport, mem_closure_iff_frequently] at hx
  exact hx.and_eventually h_not_in_support |>.exists.elim fun y hy =>
    hy.2 hy.1

/-- The n-dimensional mollifier is smooth (C^∞) -/
lemma stdMollifierPi_contDiff {ε : ℝ} (hε : 0 < ε) : ContDiff ℝ (⊤ : ℕ∞) (stdMollifierPi n ε) := by
  unfold stdMollifierPi
  -- The function can be written as f x = ∏ i, (g i) x where g i = stdMollifierEps ε ∘ coord i
  have h_eq : (fun x : E n => ∏ i, stdMollifierEps ε (x i)) =
      ∏ i : Fin n, (fun x : E n => stdMollifierEps ε (x i)) := by
    ext x
    simp only [Finset.prod_apply]
  rw [h_eq]
  apply contDiff_infty.2
  intro m
  apply contDiff_prod' (t := Finset.univ)
  intro i _
  -- Each factor is stdMollifierEps ε ∘ (fun x => x i), composition of smooth functions
  exact ((contDiff_infty.1 (stdMollifierEps_contDiff hε) m)).comp
    ((contDiff_piLp_apply (n := ⊤) 2 (i := i)).of_le le_top)

lemma stdMollifierPi_integral_eq_one {ε : ℝ} (hε : 0 < ε) :
    ∫ x, stdMollifierPi n ε x = 1 := by
  -- By Fubini, the integral of a product equals the product of integrals
  unfold stdMollifierPi
  -- Volume on E n = PiLp 2 (fun _ => ℝ) equals pushforward of volume on Fin n → ℝ via toLp
  have hvol : (volume : Measure (E n)) = Measure.map (MeasurableEquiv.toLp 2 (Fin n → ℝ)) volume :=
    (PiLp.volume_preserving_toLp (Fin n)).map_eq.symm
  rw [hvol]
  rw [integral_map_equiv (MeasurableEquiv.toLp 2 (Fin n → ℝ))]
  -- Now the integrand is ∏ i, stdMollifierEps ε (toLp 2 x i) = ∏ i, stdMollifierEps ε (x i)
  simp only [MeasurableEquiv.coe_toLp]
  -- Use Fubini (volume version): ∫ ∏ᵢ fᵢ(xᵢ) = (∫ fᵢ)^n
  rw [integral_fintype_prod_volume_eq_pow]
  -- Each 1D integral is 1
  simp only [stdMollifierEps_integral_eq_one hε, one_pow]

/-! ### Mollification (Convolution) -/

/-- Mollification of f by convolution with ρ_ε: (f ⋆ ρ_ε)(x) = ∫ f(x-y) ρ_ε(y) dy -/
noncomputable def mollify (ε : ℝ) (f : E n → ℝ) (x : E n) : ℝ :=
  ∫ y, f (x - y) * stdMollifierPi n ε y

lemma mollify_eq (ε : ℝ) (f : E n → ℝ) (x : E n) :
    mollify ε f x = ∫ y, f (x - y) * stdMollifierPi n ε y := rfl

/-! ### The Space W^{1,2}(γ) -/

/-- The Gaussian Sobolev space W^{1,2}(γ) as a set -/
def W12Gaussian (n : ℕ) (γ : Measure (E n)) : Set (E n → ℝ) :=
  {f | MemW12Gaussian n f γ}

/-- Smooth functions with compact support -/
def SmoothCompactSupport (n : ℕ) : Set (E n → ℝ) :=
  {f | ContDiff ℝ ⊤ f ∧ HasCompactSupport f}

end GaussianSobolev
