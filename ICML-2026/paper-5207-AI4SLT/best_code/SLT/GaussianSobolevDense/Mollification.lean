/-
Copyright (c) 2026 Yuanhe Zhang. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yuanhe Zhang, Jason D. Lee, Fanghui Liu
-/
import SLT.GaussianSobolevDense.Defs
import SLT.GaussianSobolevDense.Cutoff
import Mathlib.Analysis.Convolution
import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls

/-!
# Mollification Lemmas for Gaussian Sobolev Density

This file proves that mollification (convolution with smooth bump) preserves convergence
in W^{1,2}(γ) for functions with compact support.

## Main Results

* `mollify_smooth` - f ⋆ ρ_ε is smooth (C^∞)
* `mollify_compact_support` - if f has compact support, so does f ⋆ ρ_ε
* `mollify_L2_convergence_gaussian_continuous` - mollification L² convergence for continuous functions
* `mollify_L2_convergence_gaussian` - ‖g ⋆ ρ_ε - g‖_{L²(γ)} → 0 as ε → 0
* `tendsto_mollify_W12` - For fixed R, f^(R,ε) → f^(R) in W^{1,2}(γ) as ε → 0

## Implementation Notes

The key insight is that for g = f·χ_R with compact support:
1. g ⋆ ρ_ε is smooth by convolution with smooth compactly supported mollifier
2. Uniform convergence on compact sets + finite Gaussian measure → L²(γ) convergence
3. Derivative-convolution commutation gives gradient convergence

-/

open MeasureTheory ProbabilityTheory Filter Topology GaussianMeasure
open scoped ENNReal NNReal Topology Convolution Pointwise

namespace GaussianSobolev

variable {n : ℕ}

/-! ### Bridge to Mathlib Convolution -/

/-- The standard bilinear map for real-valued convolution (scalar multiplication). -/
noncomputable abbrev mulLR : ℝ →L[ℝ] ℝ →L[ℝ] ℝ := ContinuousLinearMap.mul ℝ ℝ

/-- Our mollify definition equals Mathlib's convolution with swapped arguments.
    mollify ε f x = ∫ y, f(x-y) * ρ_ε(y)
                  = ∫ t, f(t) * ρ_ε(x-t)   [with t = x - y]
                  = (f ⋆ ρ_ε)(x)          [Mathlib convolution] -/
lemma mollify_eq_convolution {n : ℕ} {ε : ℝ} (f : E n → ℝ) :
    mollify ε f = f ⋆[mulLR, volume] (stdMollifierPi n ε) := by
  ext x
  unfold mollify
  rw [convolution]
  conv_lhs =>
    arg 2; ext y
    rw [show f (x - y) * stdMollifierPi n ε y = mulLR (f (x - y)) (stdMollifierPi n ε y) by rfl]
  have h_eq : ∫ y : E n, mulLR (f (x - y)) (stdMollifierPi n ε y) =
      ∫ y : E n, mulLR (f y) (stdMollifierPi n ε (x - y)) := by
    have h := MeasureTheory.integral_sub_left_eq_self
      (fun y => mulLR (f y) (stdMollifierPi n ε (x - y))) volume x
    convert h using 1
    congr 1
    funext y
    simp only [sub_sub_cancel]
  rw [h_eq]

/-! ### Smoothness of Mollification -/

/-- Mollification of a locally integrable function with a smooth compactly supported
    mollifier is smooth (C^∞). -/
lemma mollify_smooth {n : ℕ} {g : E n → ℝ} (_hg : HasCompactSupport g)
    (hg_int : LocallyIntegrable g volume) {ε : ℝ} (hε : 0 < ε) :
    ContDiff ℝ (⊤ : ℕ∞) (mollify ε g) := by
  rw [mollify_eq_convolution g]
  apply HasCompactSupport.contDiff_convolution_right mulLR
  · exact stdMollifierPi_hasCompactSupport hε
  · exact hg_int
  · exact stdMollifierPi_contDiff hε

/-- Mollification preserves compact support up to ε√n-enlargement.

    The n-dimensional mollifier has support in the ball of radius ε√n (in l² norm),
    so supp(g ⋆ ρ_ε) ⊆ supp(g) + supp(ρ_ε) ⊆ B(0, 2R) + B(0, ε√n) ⊆ B(0, 2R + ε√n). -/
lemma mollify_compact_support {n : ℕ} {g : E n → ℝ} {R : ℝ} (_hR : 0 < R)
    (hg : tsupport g ⊆ Metric.closedBall (0 : E n) (2 * R))
    {ε : ℝ} (hε : 0 < ε) :
    tsupport (mollify ε g) ⊆ Metric.closedBall (0 : E n) (2 * R + ε * Real.sqrt n) := by
  rw [mollify_eq_convolution g]
  apply subset_trans (closure_mono (support_convolution_subset mulLR))
  have h1 : Function.support g + Function.support (stdMollifierPi n ε) ⊆
      tsupport g + tsupport (stdMollifierPi n ε) :=
    Set.add_subset_add subset_closure subset_closure
  apply subset_trans (closure_mono h1)
  have h_mol : tsupport (stdMollifierPi n ε) ⊆ Metric.closedBall (0 : E n) (ε * Real.sqrt n) :=
    stdMollifierPi_tsupport_subset hε
  apply subset_trans (closure_mono (Set.add_subset_add hg h_mol))
  have h_add : Metric.closedBall (0 : E n) (2 * R) + Metric.closedBall 0 (ε * Real.sqrt n) =
      Metric.closedBall 0 (2 * R + ε * Real.sqrt n) := by
    rw [closedBall_add_closedBall (by linarith : 0 ≤ 2 * R) (by positivity : 0 ≤ ε * Real.sqrt n)]
    simp
  rw [h_add, IsClosed.closure_eq Metric.isClosed_closedBall]

/-- Combined: mollification of cutoff function has compact support and is smooth. -/
lemma mollify_cutoff_smooth_compactSupport (f : E n → ℝ) {R ε : ℝ} (hR : 0 < R) (hε : 0 < ε)
    (hf_int : LocallyIntegrable f volume) :
    ContDiff ℝ (⊤ : ℕ∞) (mollify ε (f * smoothCutoffR R)) ∧
    HasCompactSupport (mollify ε (f * smoothCutoffR R)) := by
  constructor
  · apply mollify_smooth _ _ hε
    · exact HasCompactSupport.mul_left (smoothCutoffR_hasCompactSupport hR)
    · rw [MeasureTheory.locallyIntegrable_iff]
      intro K hK
      have hf_K : IntegrableOn f K volume := hf_int.integrableOn_isCompact hK
      apply Integrable.mono hf_K
      · exact (hf_int.aestronglyMeasurable.mul
          (smoothCutoffR_contDiff hR).continuous.aestronglyMeasurable).restrict
      · filter_upwards with x
        simp only [Pi.mul_apply, Real.norm_eq_abs]
        calc |f x * smoothCutoffR R x|
            = |f x| * |smoothCutoffR R x| := abs_mul _ _
          _ ≤ |f x| * 1 := by
              apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
              rw [abs_of_nonneg (smoothCutoffR_nonneg R x)]
              exact smoothCutoffR_le_one R x
          _ = |f x| := mul_one _
  · have h_supp : tsupport (f * smoothCutoffR R) ⊆ Metric.closedBall (0 : E n) (2 * R) := by
      apply closure_minimal _ Metric.isClosed_closedBall
      intro x hx
      rw [Function.mem_support, Pi.mul_apply] at hx
      have hχ : smoothCutoffR R x ≠ 0 := fun h => hx (by simp [h])
      exact smoothCutoffR_support_subset hR (Function.mem_support.mpr hχ)
    have h_mollify_supp := mollify_compact_support hR h_supp hε
    apply HasCompactSupport.of_support_subset_isCompact
      (isCompact_closedBall 0 (2 * R + ε * Real.sqrt n))
    intro x hx
    exact h_mollify_supp (subset_closure hx)

/-! ### Infrastructure for Mollification Convergence

Pointwise convergence → uniform on compact sets → L² convergence via finite measure. -/

/-! #### Pointwise Convergence -/

/-- For any positive ε, the mollifier ρ_ε is nonnegative and integrates to 1.
    These are the key properties of an approximate identity. -/
lemma stdMollifierPi_pos_integral {n : ℕ} {ε : ℝ} (hε : 0 < ε) :
    (∀ x, 0 ≤ stdMollifierPi n ε x) ∧ ∫ x, stdMollifierPi n ε x = 1 :=
  ⟨stdMollifierPi_nonneg ε hε, stdMollifierPi_integral_eq_one hε⟩

/-- The support of ρ_ε is contained in B(0, ε√n). -/
lemma stdMollifierPi_support_shrinks {n : ℕ} {ε : ℝ} (hε : 0 < ε) :
    Function.support (stdMollifierPi n ε) ⊆ Metric.closedBall (0 : E n) (ε * Real.sqrt n) :=
  subset_trans subset_closure (stdMollifierPi_tsupport_subset hε)

/-- For continuous g, mollify ε g x → g x as ε → 0. -/
lemma mollify_tendsto_pointwise {n : ℕ} {g : E n → ℝ} (hg_cont : Continuous g) (x : E n) :
    Filter.Tendsto (fun ε => mollify ε g x) (nhdsWithin 0 (Set.Ioi 0)) (nhds (g x)) := by
  rw [Metric.tendsto_nhdsWithin_nhds]
  intro δ hδ
  have hg_cont_x := hg_cont.continuousAt (x := x)
  rw [Metric.continuousAt_iff] at hg_cont_x
  obtain ⟨η, hη_pos, hη_cont⟩ := hg_cont_x δ hδ
  by_cases hn : n = 0
  · subst hn
    use 1
    constructor
    · exact one_pos
    · intro ε _hε_dist hε_pos
      rw [dist_eq_norm, Real.norm_eq_abs]
      simp only [mollify]
      have h_mol_one : ∀ y : E 0, stdMollifierPi 0 ε y = 1 := by
        intro y
        unfold stdMollifierPi
        rw [Finset.prod_eq_one]
        intro i _
        exact Fin.elim0 i
      conv_lhs =>
        arg 1; arg 1; arg 2; ext y
        rw [h_mol_one y, mul_one]
      have h_const : (fun y : E 0 => g (x - y)) = fun _ => g x := by
        ext y
        congr 1
        ext i; exact Fin.elim0 i
      rw [h_const, integral_const, smul_eq_mul]
      have h_vol_one : (volume : Measure (E 0)).real Set.univ = 1 := by
        haveI : Subsingleton (E 0) := by
          haveI : IsEmpty (Fin 0) := Fin.isEmpty
          haveI : Unique (Fin 0 → ℝ) := Pi.uniqueOfIsEmpty _
          exact (Equiv.subsingleton_congr (EuclideanSpace.equiv (Fin 0) ℝ).toEquiv).mpr
            Unique.instSubsingleton
        have h_univ_eq : (Set.univ : Set (E 0)) = {0} := Set.eq_singleton_iff_unique_mem.mpr
          ⟨Set.mem_univ 0, fun y _ => Subsingleton.elim y 0⟩
        let b : OrthonormalBasis (Fin 0) ℝ (E 0) := EuclideanSpace.basisFun (Fin 0) ℝ
        have h_parallel : parallelepiped (b : Fin 0 → E 0) = {0} := by
          ext z
          rw [mem_parallelepiped_iff]
          simp only [Set.mem_singleton_iff]
          constructor
          · intro ⟨_, _, h⟩
            simp only [Fintype.sum_empty] at h
            exact h
          · intro hz
            refine ⟨0, ?_, ?_⟩
            · simp only [Set.mem_Icc, le_refl, zero_le_one, and_self]
            · simp only [Fintype.sum_empty]; exact hz
        rw [Measure.real, h_univ_eq, ← h_parallel, b.volume_parallelepiped, ENNReal.toReal_one]
      rw [h_vol_one, one_mul, sub_self, abs_zero]
      exact hδ
  · obtain ⟨η', hη'_pos, hη'_cont⟩ := hg_cont_x (δ / 2) (by linarith)
    have hn' : 0 < n := Nat.pos_of_ne_zero hn
    have hsqrt_pos : 0 < Real.sqrt n := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hn')
    use η' / Real.sqrt n
    constructor
    · exact div_pos hη'_pos hsqrt_pos
    · intro ε hε_mem hε_dist
      rw [dist_eq_norm, Real.norm_eq_abs] at hε_dist ⊢
      have hε_pos : 0 < ε := hε_mem
      have hε_small : |ε| < η' / Real.sqrt n := by simp only [sub_zero] at hε_dist; exact hε_dist
      have hε_sqrt : ε * Real.sqrt n < η' := by
        have hε_abs : ε = |ε| := (abs_of_pos hε_pos).symm
        rw [hε_abs]
        calc |ε| * Real.sqrt n < (η' / Real.sqrt n) * Real.sqrt n := by
              apply mul_lt_mul_of_pos_right hε_small hsqrt_pos
          _ = η' := by field_simp
      simp only [mollify]
      have h_int_one := stdMollifierPi_integral_eq_one (n := n) hε_pos
      rw [show g x = ∫ y, g x * stdMollifierPi n ε y by
        rw [MeasureTheory.integral_const_mul, h_int_one, mul_one]]
      rw [← integral_sub]
      · have h_bound : |∫ y, g (x - y) * stdMollifierPi n ε y - g x * stdMollifierPi n ε y| ≤ δ / 2 := by
          calc |∫ y, g (x - y) * stdMollifierPi n ε y - g x * stdMollifierPi n ε y|
              = |∫ y, (g (x - y) - g x) * stdMollifierPi n ε y| := by
                congr 1; congr 1; ext y; ring
            _ ≤ ∫ y, |(g (x - y) - g x) * stdMollifierPi n ε y| := abs_integral_le_integral_abs
            _ = ∫ y, |g (x - y) - g x| * stdMollifierPi n ε y := by
                congr 1; ext y
                rw [abs_mul, abs_of_nonneg (stdMollifierPi_nonneg ε hε_pos y)]
            _ ≤ ∫ y, (δ / 2) * stdMollifierPi n ε y := by
                apply integral_mono_of_nonneg
                · filter_upwards with y
                  exact mul_nonneg (abs_nonneg _) (stdMollifierPi_nonneg ε hε_pos y)
                · exact Integrable.const_mul ((stdMollifierPi_contDiff hε_pos).continuous.integrable_of_hasCompactSupport
                    (stdMollifierPi_hasCompactSupport hε_pos)) _
                · filter_upwards with y
                  by_cases hy_supp : y ∈ tsupport (stdMollifierPi n ε)
                  · have hy_norm : ‖y‖ ≤ ε * Real.sqrt n := by
                      have := stdMollifierPi_tsupport_subset hε_pos hy_supp
                      rwa [Metric.mem_closedBall, dist_zero_right] at this
                    have hdist : dist (x - y) x < η' := by
                      simp only [dist_eq_norm]
                      rw [show x - y - x = -y by abel]
                      rw [norm_neg]
                      exact lt_of_le_of_lt hy_norm hε_sqrt
                    have := hη'_cont hdist
                    rw [dist_eq_norm, Real.norm_eq_abs] at this
                    apply mul_le_mul_of_nonneg_right (le_of_lt this)
                    exact stdMollifierPi_nonneg ε hε_pos y
                  · have hy_not_supp : y ∉ Function.support (stdMollifierPi n ε) :=
                      fun h => hy_supp (subset_closure h)
                    have hy_zero : stdMollifierPi n ε y = 0 := Function.notMem_support.mp hy_not_supp
                    simp only [hy_zero, mul_zero, le_refl]
            _ = (δ / 2) * ∫ y, stdMollifierPi n ε y := MeasureTheory.integral_const_mul _ _
            _ = (δ / 2) * 1 := by rw [h_int_one]
            _ = δ / 2 := mul_one _
        linarith [h_bound]
      · have hg_sub_cont : Continuous (fun y => g (x - y)) := hg_cont.comp (continuous_sub_left x)
        have hprod_cont : Continuous (fun y => g (x - y) * stdMollifierPi n ε y) :=
          hg_sub_cont.mul (stdMollifierPi_contDiff hε_pos).continuous
        have hprod_supp : HasCompactSupport (fun y => g (x - y) * stdMollifierPi n ε y) :=
          (stdMollifierPi_hasCompactSupport hε_pos).mul_left
        exact hprod_cont.integrable_of_hasCompactSupport hprod_supp
      · exact Integrable.const_mul ((stdMollifierPi_contDiff hε_pos).continuous.integrable_of_hasCompactSupport
          (stdMollifierPi_hasCompactSupport hε_pos)) (g x)

/-! #### Uniform Convergence on Compact Sets -/

set_option maxHeartbeats 400000 in
/-- For uniformly continuous functions on a compact set, mollification converges uniformly. -/
lemma mollify_tendsto_uniformly_on_compact {n : ℕ} {g : E n → ℝ} {K : Set (E n)}
    (hK : IsCompact K) (hg_cont : Continuous g) :
    TendstoUniformlyOn (fun ε => mollify ε g) g (nhdsWithin 0 (Set.Ioi 0)) K := by
  rw [Metric.tendstoUniformlyOn_iff]
  intro δ hδ
  have hg_unif : UniformContinuousOn g (K + Metric.closedBall 0 1) := by
    apply IsCompact.uniformContinuousOn_of_continuous
    · exact hK.add (isCompact_closedBall (0 : E n) 1)
    · exact hg_cont.continuousOn
  rw [Metric.uniformContinuousOn_iff] at hg_unif
  obtain ⟨η, hη_pos, hη_unif⟩ := hg_unif (δ / 2) (by linarith)
  have hsqrt_pos : 0 < Real.sqrt n + 1 := by linarith [Real.sqrt_nonneg n]
  have hε₀ : 0 < min (η / (Real.sqrt n + 1)) (1 / (Real.sqrt n + 1)) := by
    apply lt_min
    · exact div_pos hη_pos hsqrt_pos
    · exact div_pos one_pos hsqrt_pos
  apply mem_nhdsWithin.mpr
  use Metric.ball 0 (min (η / (Real.sqrt n + 1)) (1 / (Real.sqrt n + 1)))
  refine ⟨Metric.isOpen_ball, Metric.mem_ball_self hε₀, ?_⟩
  intro ε ⟨hε_ball, hε_pos_mem⟩
  simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs] at hε_ball
  simp only [Set.mem_Ioi] at hε_pos_mem
  have hε_pos : 0 < ε := hε_pos_mem
  have hε_small : ε < min (η / (Real.sqrt n + 1)) (1 / (Real.sqrt n + 1)) := lt_of_abs_lt hε_ball
  intro x hx
  have hε_lt_1 : ε < 1 := by
    have h := lt_of_lt_of_le hε_small (min_le_right _ _)
    have hsqrt_le : 1 ≤ Real.sqrt n + 1 := by linarith [Real.sqrt_nonneg n]
    calc ε < 1 / (Real.sqrt n + 1) := h
      _ ≤ 1 / 1 := by
        apply div_le_div_of_nonneg_left zero_le_one one_pos hsqrt_le
      _ = 1 := div_one 1
  have hε_η : ε * Real.sqrt n < η := by
    have h1 : ε < η / (Real.sqrt n + 1) := lt_of_lt_of_le hε_small (min_le_left _ _)
    have h2 : ε * (Real.sqrt n + 1) < η := by
      calc ε * (Real.sqrt n + 1) < (η / (Real.sqrt n + 1)) * (Real.sqrt n + 1) := by
            apply mul_lt_mul_of_pos_right h1
            linarith [Real.sqrt_nonneg n]
        _ = η := by field_simp
    calc ε * Real.sqrt n ≤ ε * (Real.sqrt n + 1) := by
          apply mul_le_mul_of_nonneg_left (by linarith) (le_of_lt hε_pos)
      _ < η := h2
  have hε_sqrtn_lt_1 : ε * Real.sqrt n < 1 := by
    have h1 : ε < 1 / (Real.sqrt n + 1) := lt_of_lt_of_le hε_small (min_le_right _ _)
    have h2 : ε * (Real.sqrt n + 1) < 1 := by
      calc ε * (Real.sqrt n + 1) < (1 / (Real.sqrt n + 1)) * (Real.sqrt n + 1) := by
            apply mul_lt_mul_of_pos_right h1 hsqrt_pos
        _ = 1 := by field_simp
    calc ε * Real.sqrt n ≤ ε * (Real.sqrt n + 1) := by
          apply mul_le_mul_of_nonneg_left (by linarith) (le_of_lt hε_pos)
      _ < 1 := h2
  rw [dist_eq_norm, Real.norm_eq_abs]
  simp only [mollify]
  have h_int_one := stdMollifierPi_integral_eq_one (n := n) hε_pos
  rw [show g x = ∫ y, g x * stdMollifierPi n ε y by
    rw [MeasureTheory.integral_const_mul, h_int_one, mul_one]]
  rw [← integral_sub]
  · have h_bound : |∫ y, g (x - y) * stdMollifierPi n ε y - g x * stdMollifierPi n ε y| ≤ δ / 2 := by
      calc |∫ y, g (x - y) * stdMollifierPi n ε y - g x * stdMollifierPi n ε y|
          = |∫ y, (g (x - y) - g x) * stdMollifierPi n ε y| := by
            congr 1; congr 1; ext y; ring
        _ ≤ ∫ y, |(g (x - y) - g x) * stdMollifierPi n ε y| := abs_integral_le_integral_abs
        _ = ∫ y, |g (x - y) - g x| * stdMollifierPi n ε y := by
            congr 1; ext y
            rw [abs_mul, abs_of_nonneg (stdMollifierPi_nonneg ε hε_pos y)]
        _ ≤ ∫ y, (δ / 2) * stdMollifierPi n ε y := by
            apply integral_mono_of_nonneg
            · exact ae_of_all _ fun y => mul_nonneg (abs_nonneg _) (stdMollifierPi_nonneg ε hε_pos y)
            · exact Integrable.const_mul ((stdMollifierPi_contDiff hε_pos).continuous.integrable_of_hasCompactSupport
                (stdMollifierPi_hasCompactSupport hε_pos)) _
            · apply ae_of_all _
              intro y
              by_cases hy_supp : y ∈ tsupport (stdMollifierPi n ε)
              · have hy_norm : ‖y‖ ≤ ε * Real.sqrt n := by
                  have := stdMollifierPi_tsupport_subset hε_pos hy_supp
                  rwa [Metric.mem_closedBall, dist_zero_right] at this
                have hdist : dist (x - y) x < η := by
                  simp only [dist_eq_norm]
                  rw [show x - y - x = -y by abel]
                  rw [norm_neg]
                  exact lt_of_le_of_lt hy_norm hε_η
                have hxy_mem : x - y ∈ K + Metric.closedBall 0 1 := by
                  rw [sub_eq_add_neg]
                  refine Set.add_mem_add hx ?_
                  simp only [Metric.mem_closedBall, dist_zero_right, norm_neg]
                  exact le_of_lt (lt_of_le_of_lt hy_norm hε_sqrtn_lt_1)
                have hx_mem : x ∈ K + Metric.closedBall 0 1 := by
                  rw [← add_zero x]
                  exact Set.add_mem_add hx (Metric.mem_closedBall_self (by linarith : (0 : ℝ) ≤ 1))
                have hg_close := hη_unif (x - y) hxy_mem x hx_mem hdist
                simp only [dist_eq_norm, Real.norm_eq_abs] at hg_close
                exact mul_le_mul_of_nonneg_right (le_of_lt hg_close) (stdMollifierPi_nonneg ε hε_pos y)
              · have hy_not_supp : y ∉ Function.support (stdMollifierPi n ε) :=
                  fun h => hy_supp (subset_closure h)
                have hy_zero : stdMollifierPi n ε y = 0 := Function.notMem_support.mp hy_not_supp
                simp only [hy_zero, mul_zero, le_refl]
        _ = (δ / 2) * ∫ y, stdMollifierPi n ε y := MeasureTheory.integral_const_mul _ _
        _ = (δ / 2) * 1 := by rw [h_int_one]
        _ = δ / 2 := mul_one _
    have h_neg : ∫ a, g x * stdMollifierPi n ε a - g (x - a) * stdMollifierPi n ε a =
                 -(∫ y, g (x - y) * stdMollifierPi n ε y - g x * stdMollifierPi n ε y) := by
      rw [← integral_neg]
      congr 1
      ext y
      ring
    rw [h_neg, abs_neg]
    linarith [h_bound]
  · exact Integrable.const_mul ((stdMollifierPi_contDiff hε_pos).continuous.integrable_of_hasCompactSupport
      (stdMollifierPi_hasCompactSupport hε_pos)) (g x)
  · have hg_sub_cont : Continuous (fun y => g (x - y)) := hg_cont.comp (continuous_sub_left x)
    have hprod_cont : Continuous (fun y => g (x - y) * stdMollifierPi n ε y) :=
      hg_sub_cont.mul (stdMollifierPi_contDiff hε_pos).continuous
    have hprod_supp : HasCompactSupport (fun y => g (x - y) * stdMollifierPi n ε y) :=
      (stdMollifierPi_hasCompactSupport hε_pos).mul_left
    exact hprod_cont.integrable_of_hasCompactSupport hprod_supp

/-! #### L² Convergence from Uniform Convergence -/

/-- Uniform convergence implies L² convergence for functions vanishing outside a
    finite-measure set. Bounds the L² norm by sup-norm times measure^(1/2). -/
lemma eLpNorm_tendsto_zero_of_tendstoUniformly_restrict {n : ℕ} {f : ℕ → E n → ℝ} {g : E n → ℝ}
    {s : Set (E n)} (hμ : volume s < ⊤)
    (hf_zero : ∀ i x, x ∉ s → f i x = 0) (hg_zero : ∀ x, x ∉ s → g x = 0)
    (h_unif : TendstoUniformly f g Filter.atTop) :
    Filter.Tendsto (fun i => eLpNorm (f i - g) 2 volume) Filter.atTop (nhds 0) := by
  have h_restrict : ∀ i, eLpNorm (f i - g) 2 volume = eLpNorm (f i - g) 2 (volume.restrict s) := by
    intro i
    symm
    apply eLpNorm_restrict_eq_of_support_subset
    intro x hx
    rw [Function.mem_support, Pi.sub_apply, sub_ne_zero] at hx
    by_contra hxs
    simp_all
  simp_rw [h_restrict]
  rw [ENNReal.tendsto_atTop_zero]
  intro ε hε
  rw [Metric.tendstoUniformly_iff] at h_unif
  by_cases hμ_zero : volume s = 0
  · use 0
    intro i _
    rw [Measure.restrict_zero_set hμ_zero, eLpNorm_measure_zero]
    exact zero_le
  · by_cases hε_top : ε = ⊤
    · use 0; intro i _; rw [hε_top]; exact le_top
    have h_sqrt_pos : 0 < (volume s).toReal.sqrt :=
      Real.sqrt_pos.mpr (ENNReal.toReal_pos hμ_zero hμ.ne)
    set δ := ε.toReal / (2 * (volume s).toReal.sqrt) with hδ_def
    have hδ_pos : 0 < δ := by
      apply div_pos
      · exact ENNReal.toReal_pos hε.ne' hε_top
      · linarith [h_sqrt_pos]
    have hN := h_unif δ hδ_pos
    simp only [Filter.eventually_atTop] at hN
    obtain ⟨N, hN⟩ := hN
    use N
    intro i hi
    have h_bound : eLpNorm (f i - g) 2 (volume.restrict s) ≤ ENNReal.ofReal δ * (volume s) ^ (1/2 : ℝ) := by
      calc eLpNorm (f i - g) 2 (volume.restrict s)
          ≤ eLpNorm (fun _ => δ) 2 (volume.restrict s) := by
            apply eLpNorm_mono_ae
            rw [Filter.eventually_iff_exists_mem]
            use Set.univ
            constructor
            · exact Filter.univ_mem
            · intro x _
              rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hδ_pos]
              specialize hN i hi x
              rw [dist_eq_norm, Real.norm_eq_abs] at hN
              rw [Pi.sub_apply, abs_sub_comm]
              exact le_of_lt hN
        _ = ENNReal.ofReal |δ| * (volume.restrict s Set.univ) ^ (1/2 : ℝ) := by
            rw [eLpNorm_const]
            · congr 1
              · rw [Real.enorm_eq_ofReal_abs]
            · norm_num
            · intro h
              rw [Measure.restrict_eq_zero] at h
              exact hμ_zero h
        _ = ENNReal.ofReal δ * (volume s) ^ (1/2 : ℝ) := by
            rw [abs_of_pos hδ_pos, Measure.restrict_apply_univ]
    calc eLpNorm (f i - g) 2 (volume.restrict s)
        ≤ ENNReal.ofReal δ * (volume s) ^ (1/2 : ℝ) := h_bound
      _ = ENNReal.ofReal (ε.toReal / (2 * (volume s).toReal.sqrt)) * (volume s) ^ (1/2 : ℝ) := by
          rw [hδ_def]
      _ ≤ ε := by
          have h_sqrt : (volume s) ^ (1/2 : ℝ) = ENNReal.ofReal ((volume s).toReal.sqrt) := by
            conv_lhs => rw [← ENNReal.ofReal_toReal hμ.ne]
            rw [ENNReal.ofReal_rpow_of_nonneg ENNReal.toReal_nonneg (by norm_num : (1:ℝ)/2 ≥ 0)]
            rw [Real.sqrt_eq_rpow]
          rw [h_sqrt]
          have hε_nonneg : 0 ≤ ε.toReal / (2 * (volume s).toReal.sqrt) := by
            apply div_nonneg ENNReal.toReal_nonneg
            linarith [h_sqrt_pos]
          rw [← ENNReal.ofReal_mul hε_nonneg]
          have h_simp : ε.toReal / (2 * (volume s).toReal.sqrt) * (volume s).toReal.sqrt = ε.toReal / 2 := by
            field_simp
          rw [h_simp]
          calc ENNReal.ofReal (ε.toReal / 2)
              ≤ ENNReal.ofReal ε.toReal := by
                  apply ENNReal.ofReal_le_ofReal
                  linarith [ENNReal.toReal_nonneg (a := ε)]
            _ ≤ ε := ENNReal.ofReal_toReal_le

/-- General version: for ANY measure μ with μ s < ⊤, uniform convergence implies L² convergence.
    This is essential for transferring L² convergence from Lebesgue to Gaussian measure. -/
lemma eLpNorm_tendsto_zero_of_tendstoUniformly_general {n : ℕ} {μ : Measure (E n)}
    {f : ℕ → E n → ℝ} {g : E n → ℝ}
    {s : Set (E n)} (hμ : μ s < ⊤)
    (hf_zero : ∀ i x, x ∉ s → f i x = 0) (hg_zero : ∀ x, x ∉ s → g x = 0)
    (h_unif : TendstoUniformly f g Filter.atTop) :
    Filter.Tendsto (fun i => eLpNorm (f i - g) 2 μ) Filter.atTop (nhds 0) := by
  have h_restrict : ∀ i, eLpNorm (f i - g) 2 μ = eLpNorm (f i - g) 2 (μ.restrict s) := by
    intro i
    symm
    apply eLpNorm_restrict_eq_of_support_subset
    intro x hx
    rw [Function.mem_support, Pi.sub_apply, sub_ne_zero] at hx
    by_contra hxs
    simp_all
  simp_rw [h_restrict]
  rw [ENNReal.tendsto_atTop_zero]
  intro ε hε
  rw [Metric.tendstoUniformly_iff] at h_unif
  by_cases hμ_zero : μ s = 0
  · use 0
    intro i _
    rw [Measure.restrict_eq_zero.mpr hμ_zero, eLpNorm_measure_zero]
    exact zero_le
  · by_cases hε_top : ε = ⊤
    · use 0; intro i _; rw [hε_top]; exact le_top
    have h_sqrt_pos : 0 < (μ s).toReal.sqrt :=
      Real.sqrt_pos.mpr (ENNReal.toReal_pos hμ_zero hμ.ne)
    set δ := ε.toReal / (2 * (μ s).toReal.sqrt) with hδ_def
    have hδ_pos : 0 < δ := by
      apply div_pos
      · exact ENNReal.toReal_pos hε.ne' hε_top
      · linarith [h_sqrt_pos]
    have hN := h_unif δ hδ_pos
    simp only [Filter.eventually_atTop] at hN
    obtain ⟨N, hN⟩ := hN
    use N
    intro i hi
    have h_bound : eLpNorm (f i - g) 2 (μ.restrict s) ≤ ENNReal.ofReal δ * (μ s) ^ (1/2 : ℝ) := by
      calc eLpNorm (f i - g) 2 (μ.restrict s)
          ≤ eLpNorm (fun _ => δ) 2 (μ.restrict s) := by
            apply eLpNorm_mono_ae
            rw [Filter.eventually_iff_exists_mem]
            use Set.univ
            constructor
            · exact Filter.univ_mem
            · intro x _
              rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hδ_pos]
              specialize hN i hi x
              rw [dist_eq_norm, Real.norm_eq_abs] at hN
              rw [Pi.sub_apply, abs_sub_comm]
              exact le_of_lt hN
        _ = ENNReal.ofReal |δ| * (μ.restrict s Set.univ) ^ (1/2 : ℝ) := by
            rw [eLpNorm_const]
            · congr 1
              · rw [Real.enorm_eq_ofReal_abs]
            · norm_num
            · intro h
              rw [Measure.restrict_eq_zero] at h
              exact hμ_zero h
        _ = ENNReal.ofReal δ * (μ s) ^ (1/2 : ℝ) := by
            rw [abs_of_pos hδ_pos, Measure.restrict_apply_univ]
    calc eLpNorm (f i - g) 2 (μ.restrict s)
        ≤ ENNReal.ofReal δ * (μ s) ^ (1/2 : ℝ) := h_bound
      _ = ENNReal.ofReal (ε.toReal / (2 * (μ s).toReal.sqrt)) * (μ s) ^ (1/2 : ℝ) := by
          rw [hδ_def]
      _ ≤ ε := by
          have h_sqrt : (μ s) ^ (1/2 : ℝ) = ENNReal.ofReal ((μ s).toReal.sqrt) := by
            conv_lhs => rw [← ENNReal.ofReal_toReal hμ.ne]
            rw [ENNReal.ofReal_rpow_of_nonneg ENNReal.toReal_nonneg (by norm_num : (1:ℝ)/2 ≥ 0)]
            rw [Real.sqrt_eq_rpow]
          rw [h_sqrt]
          have hε_nonneg : 0 ≤ ε.toReal / (2 * (μ s).toReal.sqrt) := by
            apply div_nonneg ENNReal.toReal_nonneg
            linarith [h_sqrt_pos]
          rw [← ENNReal.ofReal_mul hε_nonneg]
          have h_simp : ε.toReal / (2 * (μ s).toReal.sqrt) * (μ s).toReal.sqrt = ε.toReal / 2 := by
            field_simp
          rw [h_simp]
          calc ENNReal.ofReal (ε.toReal / 2)
              ≤ ENNReal.ofReal ε.toReal := by
                  apply ENNReal.ofReal_le_ofReal
                  linarith [ENNReal.toReal_nonneg (a := ε)]
            _ ≤ ε := ENNReal.ofReal_toReal_le

/-! ### L² Convergence for Continuous Compactly Supported Functions -/

set_option maxHeartbeats 400000 in
/-- Mollification of continuous compactly supported functions converges in L².
    This uses uniform convergence on compact sets + finite measure on the support. -/
lemma mollify_L2_convergence_continuous {n : ℕ} {g : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hg_cont : Continuous g) (hg_supp : tsupport g ⊆ Metric.closedBall 0 (2 * R)) :
    Filter.Tendsto (fun ε => eLpNorm (mollify ε g - g) 2 volume)
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  rw [ENNReal.tendsto_nhds_zero]
  intro δ hδ
  have hK : IsCompact (Metric.closedBall (0 : E n) (2 * R + 1)) := isCompact_closedBall 0 (2 * R + 1)
  have hμ : volume (Metric.closedBall (0 : E n) (2 * R + 1)) < ⊤ := by
    exact measure_closedBall_lt_top
  have h_unif := mollify_tendsto_uniformly_on_compact hK hg_cont
  rw [Metric.tendstoUniformlyOn_iff] at h_unif
  have hμ_toReal : (volume (Metric.closedBall (0 : E n) (2 * R + 1))).toReal ≥ 0 :=
    ENNReal.toReal_nonneg
  set μK := (volume (Metric.closedBall (0 : E n) (2 * R + 1))).toReal with hμK_def
  have h_denom_pos : 0 < 2 * Real.sqrt μK + 1 := by linarith [Real.sqrt_nonneg μK]
  by_cases hδ_top : δ = ⊤
  · simp only [hδ_top, le_top, Filter.eventually_iff_exists_mem]
    exact ⟨Set.univ, Filter.univ_mem, fun _ _ => trivial⟩
  set η := δ.toReal / (2 * Real.sqrt μK + 1) with hη_def
  have hη_pos : 0 < η := by
    apply div_pos
    · exact ENNReal.toReal_pos hδ.ne' hδ_top
    · exact h_denom_pos
  have h_evt := h_unif η hη_pos
  have h_evt_sqrt : ∀ᶠ (ε : ℝ) in nhdsWithin 0 (Set.Ioi 0), ε * Real.sqrt (n : ℝ) < 1 := by
    by_cases hn : n = 0
    · simp only [hn, Nat.cast_zero, Real.sqrt_zero, mul_zero, zero_lt_one, Filter.eventually_true]
    · have hn_pos : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn))
      have h_bound : 0 < 1 / Real.sqrt (n : ℝ) := div_pos one_pos hn_pos
      filter_upwards [Ioo_mem_nhdsGT h_bound] with ε hε
      calc ε * Real.sqrt n < (1 / Real.sqrt n) * Real.sqrt n := by
            apply mul_lt_mul_of_pos_right hε.2 hn_pos
        _ = 1 := div_mul_cancel₀ 1 (ne_of_gt hn_pos)
  have h_evt_pos : ∀ᶠ (ε : ℝ) in nhdsWithin 0 (Set.Ioi 0), 0 < ε := by
    filter_upwards [self_mem_nhdsWithin] with ε hε
    exact hε
  filter_upwards [h_evt, h_evt_sqrt, h_evt_pos] with ε hε_unif hε_sqrt hε_pos
  have h_supp_diff : ∀ x, x ∉ Metric.closedBall (0 : E n) (2 * R + 1) → (mollify ε g - g) x = 0 := by
    intro x hx
    simp only [Pi.sub_apply]
    have hg_x : g x = 0 := by
      by_contra hg_ne
      have hx_supp : x ∈ tsupport g := subset_closure (Function.mem_support.mpr hg_ne)
      have hx_ball : x ∈ Metric.closedBall 0 (2 * R) := hg_supp hx_supp
      have h_subset : Metric.closedBall (0 : E n) (2 * R) ⊆ Metric.closedBall 0 (2 * R + 1) := by
        apply Metric.closedBall_subset_closedBall
        linarith
      exact hx (h_subset hx_ball)
    rw [hg_x]
    simp only [mollify, sub_zero]
    apply integral_eq_zero_of_ae
    filter_upwards with (y : E n)
    by_cases hy : stdMollifierPi n ε y = 0
    · simp only [hy, mul_zero, Pi.zero_apply]
    · have hy_supp : y ∈ tsupport (stdMollifierPi n ε) :=
        subset_closure (Function.mem_support.mpr hy)
      have hy_norm : ‖y‖ ≤ ε * Real.sqrt (n : ℝ) := by
        have h := stdMollifierPi_tsupport_subset hε_pos hy_supp
        rwa [Metric.mem_closedBall, dist_zero_right] at h
      have hxy : x - y ∉ Metric.closedBall (0 : E n) (2 * R) := by
        simp only [Metric.mem_closedBall, dist_zero_right] at hx ⊢
        intro hxy_in
        have h_tri : ‖x‖ ≤ ‖x - y‖ + ‖y‖ := norm_le_norm_sub_add x y
        linarith
      have hg_xy : g (x - y) = 0 := by
        by_contra h_ne
        exact hxy (hg_supp (subset_closure (Function.mem_support.mpr h_ne)))
      simp only [hg_xy, zero_mul, Pi.zero_apply]
  have h_pointwise_bound : ∀ x ∈ Metric.closedBall (0 : E n) (2 * R + 1),
      dist (mollify ε g x) (g x) < η := fun x hx => by rw [dist_comm]; exact hε_unif x hx
  have h_restrict : eLpNorm (mollify ε g - g) 2 volume =
      eLpNorm (mollify ε g - g) 2 (volume.restrict (Metric.closedBall 0 (2 * R + 1))) := by
    symm
    apply eLpNorm_restrict_eq_of_support_subset
    intro x hx
    rw [Function.mem_support] at hx
    by_contra hx_not_in
    exact hx (h_supp_diff x hx_not_in)
  rw [h_restrict]
  calc eLpNorm (mollify ε g - g) 2 (volume.restrict (Metric.closedBall 0 (2 * R + 1)))
      ≤ eLpNorm (fun (_ : E n) => η) 2 (volume.restrict (Metric.closedBall 0 (2 * R + 1))) := by
        apply eLpNorm_mono_ae
        rw [Filter.eventually_iff_exists_mem]
        use Set.univ
        constructor
        · exact Filter.univ_mem
        · intro x _
          rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hη_pos]
          simp only [Pi.sub_apply]
          by_cases hx_in : x ∈ Metric.closedBall (0 : E n) (2 * R + 1)
          · specialize h_pointwise_bound x hx_in
            rw [dist_eq_norm, Real.norm_eq_abs] at h_pointwise_bound
            exact le_of_lt h_pointwise_bound
          · have h := h_supp_diff x hx_in
            simp only [Pi.sub_apply] at h
            rw [h, abs_zero]
            exact hη_pos.le
    _ = ENNReal.ofReal |η| * (volume.restrict (Metric.closedBall (0 : E n) (2 * R + 1)) Set.univ) ^ (1/2 : ℝ) := by
        rw [eLpNorm_const]
        · congr 1
          · rw [Real.enorm_eq_ofReal_abs]
        · norm_num
        · intro h
          rw [Measure.restrict_eq_zero] at h
          have hvol : volume (Metric.closedBall (0 : E n) (2 * R + 1)) = 0 := h
          have h_rad_pos : (0 : ℝ) < 2 * R + 1 := by linarith
          by_cases hn : n = 0
          · subst hn
            haveI : IsEmpty (Fin 0) := inferInstance
            haveI : Subsingleton (E 0) := inferInstance
            have h_univ : Metric.closedBall (0 : E 0) (2 * R + 1) = Set.univ := by
              ext x; simp [Metric.mem_closedBall, Subsingleton.elim x 0, dist_self, h_rad_pos.le]
            rw [h_univ] at hvol
            have h_ne : (volume : Measure (E 0)) ≠ 0 := NeZero.ne volume
            exact (MeasureTheory.Measure.measure_univ_ne_zero.mpr h_ne) hvol
          · haveI : Nonempty (Fin n) := ⟨⟨0, Nat.pos_of_ne_zero hn⟩⟩
            rw [EuclideanSpace.volume_closedBall] at hvol
            simp only [mul_eq_zero] at hvol
            rcases hvol with h_pow | h_const
            · have : (0 : ℝ≥0∞) < ENNReal.ofReal (2 * R + 1) ^ Fintype.card (Fin n) := by
                apply ENNReal.pow_pos
                exact ENNReal.ofReal_pos.mpr h_rad_pos
              rw [h_pow] at this
              exact (lt_irrefl 0 this).elim
            · have hgamma : 0 < Real.Gamma (Fintype.card (Fin n) / 2 + 1) := Real.Gamma_pos_of_pos (by positivity)
              have hsqrt : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr Real.pi_pos
              have hpow : 0 < Real.sqrt Real.pi ^ Fintype.card (Fin n) := pow_pos hsqrt _
              have h_pos : 0 < Real.sqrt Real.pi ^ Fintype.card (Fin n) / Real.Gamma (Fintype.card (Fin n) / 2 + 1) :=
                div_pos hpow hgamma
              have h_ne : (0 : ℝ≥0∞) < ENNReal.ofReal (Real.sqrt Real.pi ^ Fintype.card (Fin n) / Real.Gamma (Fintype.card (Fin n) / 2 + 1)) :=
                ENNReal.ofReal_pos.mpr h_pos
              rw [h_const] at h_ne
              exact (lt_irrefl 0 h_ne).elim
    _ = ENNReal.ofReal η * (volume (Metric.closedBall (0 : E n) (2 * R + 1))) ^ (1/2 : ℝ) := by
        rw [abs_of_pos hη_pos, Measure.restrict_apply_univ]
    _ ≤ δ := by
        rw [hη_def]
        by_cases hμK_zero : μK = 0
        · have h_vol_zero : (volume (Metric.closedBall (0 : E n) (2 * R + 1))) = 0 := by
            simp only [hμK_def, ENNReal.toReal_eq_zero_iff] at hμK_zero
            cases hμK_zero with
            | inl h => exact h
            | inr h => exact absurd h hμ.ne
          rw [h_vol_zero, ENNReal.zero_rpow_of_pos (by norm_num : (0:ℝ) < 1/2)]
          simp only [mul_zero, zero_le]
        · have hμK_pos : 0 < μK := lt_of_le_of_ne hμ_toReal (Ne.symm hμK_zero)
          have h_sqrt_pos : 0 < Real.sqrt μK := Real.sqrt_pos.mpr hμK_pos
          have h1 : (volume (Metric.closedBall (0 : E n) (2 * R + 1))) ^ (1/2 : ℝ) =
              ENNReal.ofReal (Real.sqrt μK) := by
            conv_lhs => rw [← ENNReal.ofReal_toReal hμ.ne]
            rw [ENNReal.ofReal_rpow_of_nonneg ENNReal.toReal_nonneg (by norm_num : (0:ℝ) ≤ 1/2)]
            congr 1
            rw [Real.sqrt_eq_rpow, hμK_def]
          rw [h1, ← ENNReal.ofReal_mul (le_of_lt hη_pos)]
          calc ENNReal.ofReal ((δ.toReal / (2 * Real.sqrt μK + 1)) * Real.sqrt μK)
              = ENNReal.ofReal (δ.toReal * Real.sqrt μK / (2 * Real.sqrt μK + 1)) := by ring_nf
            _ ≤ ENNReal.ofReal (δ.toReal * (2 * Real.sqrt μK + 1) / (2 * Real.sqrt μK + 1)) := by
                apply ENNReal.ofReal_le_ofReal
                apply div_le_div_of_nonneg_right _ (le_of_lt h_denom_pos)
                apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
                linarith [h_sqrt_pos]
            _ = ENNReal.ofReal δ.toReal := by
                congr 1
                rw [mul_div_assoc, div_self (ne_of_gt h_denom_pos), mul_one]
            _ ≤ δ := ENNReal.ofReal_toReal_le

/-! ### Gaussian L² Convergence for Continuous Functions -/

set_option maxHeartbeats 4000000 in
/-- For any probability measure μ, mollification of continuous compactly supported
    functions converges in L²(μ). Since stdGaussianE is a probability measure,
    μ s ≤ 1 < ⊤ for any set s, so we can use uniform convergence. -/
lemma mollify_L2_convergence_gaussian_continuous {n : ℕ} {g : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hg_cont : Continuous g) (hg_supp : tsupport g ⊆ Metric.closedBall 0 (2 * R)) :
    Filter.Tendsto (fun ε => eLpNorm (mollify ε g - g) 2 (stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  -- Define the enlarged ball that contains all supports
  set K := Metric.closedBall (0 : E n) (2 * R + 1) with hK_def
  -- stdGaussianE is a probability measure, so μ(K) ≤ 1 < ⊤
  have hμK_finite : stdGaussianE n K < ⊤ := by
    calc stdGaussianE n K ≤ stdGaussianE n Set.univ := measure_mono (Set.subset_univ _)
      _ = 1 := measure_univ
      _ < ⊤ := ENNReal.one_lt_top
  -- For small ε, mollify ε g - g is supported in K
  -- First, establish uniform convergence on K
  have h_unif_on_K : TendstoUniformlyOn (fun ε => mollify ε g) g (nhdsWithin 0 (Set.Ioi 0))
      (Metric.closedBall 0 (2 * R + 1)) :=
    mollify_tendsto_uniformly_on_compact (isCompact_closedBall 0 (2 * R + 1)) hg_cont
  -- The goal is Tendsto ... (nhdsWithin 0 (Ioi 0)) (nhds 0)
  rw [ENNReal.tendsto_nhds_zero]
  intro δ hδ
  -- From uniform convergence, get ε₀ such that for all 0 < ε < ε₀, mollify ε g is close to g on K
  rw [Metric.tendstoUniformlyOn_iff] at h_unif_on_K
  -- The norm is eLpNorm, so we need to convert δ
  -- Choose η such that η * (μK)^(1/2) ≤ δ
  -- Gaussian measure of a ball with positive radius is always positive
  have hμK_pos : 0 < stdGaussianE n K := by
    -- The ball contains 0 and has positive radius
    have hr_pos : 0 < 2 * R + 1 := by linarith
    have hr_ne : (2 * R + 1) ≠ 0 := ne_of_gt hr_pos
    have h_0_in_K : (0 : E n) ∈ interior K := by
      rw [hK_def, interior_closedBall 0 hr_ne]
      exact Metric.mem_ball_self hr_pos
    -- Interior is non-empty, so measure is positive
    have h_int_nonempty : (interior K).Nonempty := ⟨0, h_0_in_K⟩
    have h_int_open : IsOpen (interior K) := isOpen_interior
    -- Gaussian measure has full support, so positive on non-empty open sets
    haveI : (stdGaussianE n).IsOpenPosMeasure := GaussianSobolev.stdGaussianE_isOpenPosMeasure
    calc 0 < stdGaussianE n (interior K) := h_int_open.measure_pos _ h_int_nonempty
      _ ≤ stdGaussianE n K := measure_mono interior_subset
  -- Now we can proceed with the positive measure case
  -- Handle δ = ⊤ case first
  by_cases hδ_top : δ = ⊤
  · -- For δ = ⊤, the bound is trivially satisfied
    apply Filter.Eventually.of_forall
    intro ε
    rw [hδ_top]
    exact le_top
  · -- Now δ ≠ ⊤, so we can work with Real arithmetic
    have hδ_ne_top : δ ≠ ⊤ := hδ_top
    have h_sqrt_pos : 0 < (stdGaussianE n K).toReal.sqrt :=
      Real.sqrt_pos.mpr (ENNReal.toReal_pos (ne_of_gt hμK_pos) hμK_finite.ne)
    -- Convert δ to Real for division
    set δr := δ.toReal
    have hδr_pos : 0 < δr := ENNReal.toReal_pos hδ.ne' hδ_ne_top
    set η := δr / (2 * (stdGaussianE n K).toReal.sqrt + 1) with hη_def
    have hη_pos : 0 < η := by
      apply div_pos hδr_pos
      linarith [h_sqrt_pos]
    -- h_unif_on_K η hη_pos gives: ∀ᶠ ε in 𝓝[>] 0, ∀ x ∈ K, dist (g x) (mollify ε g x) < η
    -- We need ε * √n < 1 for the support bound, so use ε < 1/(√n + 1)
    set ε_bound := 1 / (Real.sqrt n + 1)
    have hε_bound_pos : 0 < ε_bound := by
      apply div_pos one_pos
      linarith [Real.sqrt_nonneg n]
    have h_Ioo : Set.Ioo (0 : ℝ) ε_bound ∈ 𝓝[>] (0 : ℝ) := Ioo_mem_nhdsGT hε_bound_pos
    have h_unif : ∀ᶠ (ε : ℝ) in 𝓝[>] 0, ∀ x ∈ K, dist (g x) (mollify ε g x) < η :=
      h_unif_on_K η hη_pos
    -- Combine: ε ∈ (0, ε_bound) and the uniform bound
    have h_combined : ∀ᶠ (ε : ℝ) in 𝓝[>] 0, (0 < ε ∧ ε < ε_bound) ∧
        (∀ x ∈ K, dist (g x) (mollify ε g x) < η) := by
      apply Filter.Eventually.and _ h_unif
      exact Filter.eventually_of_mem h_Ioo (fun x hx => ⟨hx.1, hx.2⟩)
    apply Filter.Eventually.mono h_combined
    intro ε ⟨⟨hε_pos, hε_lt_bound⟩, hε_unif⟩
    -- For x in K, |mollify ε g x - g x| < η
    have h_pointwise : ∀ x ∈ K, dist (mollify ε g x) (g x) < η := fun x hx => by
      rw [dist_comm]; exact hε_unif x hx
    -- For x ∉ K, mollify ε g x - g x = 0
    have h_supp_diff : ∀ x, x ∉ K → (mollify ε g - g) x = 0 := by
      intro x hx
      simp only [Pi.sub_apply]
      have h_gx : g x = 0 := by
        by_contra h_ne
        have hx_supp : x ∈ tsupport g := subset_closure (Function.mem_support.mpr h_ne)
        have hx_in : x ∈ Metric.closedBall 0 (2 * R) := hg_supp hx_supp
        have : x ∈ K := Metric.closedBall_subset_closedBall (by linarith) hx_in
        exact hx this
      have h_mol : mollify ε g x = 0 := by
        unfold mollify
        have h_integrand_zero : (fun y => g (x - y) * stdMollifierPi n ε y) = fun _ => 0 := by
          ext y
          by_cases hy_supp : y ∈ Function.support (stdMollifierPi n ε)
          · have hy_norm : ‖y‖ ≤ ε * Real.sqrt n := by
              have h_in_ball := stdMollifierPi_support_shrinks hε_pos hy_supp
              rwa [Metric.mem_closedBall, dist_zero_right] at h_in_ball
            have hxy : x - y ∉ tsupport g := by
              intro hxy_in
              have hxy_ball : x - y ∈ Metric.closedBall 0 (2 * R) := hg_supp hxy_in
              rw [Metric.mem_closedBall, dist_zero_right] at hxy_ball
              have hx_norm : ‖x‖ > 2 * R + 1 := by
                rw [hK_def, Metric.closedBall, Set.mem_setOf] at hx
                push Not at hx
                rw [dist_zero_right] at hx
                exact hx
              have hy_small : ‖y‖ < 1 := by
                have hsqrt_nonneg : 0 ≤ Real.sqrt n := Real.sqrt_nonneg n
                have h_denom_pos : 0 < Real.sqrt n + 1 := by linarith
                by_cases hn_zero : n = 0
                · simp only [hn_zero, Nat.cast_zero, Real.sqrt_zero] at hy_norm ⊢
                  calc ‖y‖ ≤ ε * 0 := hy_norm
                    _ = 0 := by ring
                    _ < 1 := by norm_num
                · have hsqrt_pos : 0 < Real.sqrt n := by
                    apply Real.sqrt_pos.mpr
                    exact Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn_zero)
                  calc ‖y‖ ≤ ε * Real.sqrt n := hy_norm
                    _ < (1 / (Real.sqrt n + 1)) * Real.sqrt n := by
                        apply mul_lt_mul_of_pos_right hε_lt_bound hsqrt_pos
                    _ = Real.sqrt n / (Real.sqrt n + 1) := by ring
                    _ < 1 := by
                        rw [div_lt_one h_denom_pos]
                        linarith
              have h_norm_diff : ‖x‖ - ‖y‖ ≤ ‖x - y‖ := by
                calc ‖x‖ - ‖y‖ = ‖x‖ + (- ‖y‖) := by ring
                  _ ≤ ‖x - y‖ + ‖y‖ + (-‖y‖) := by linarith [norm_sub_norm_le x y]
                  _ = ‖x - y‖ := by ring
              linarith
            have hg_xy : g (x - y) = 0 := by
              by_contra h_ne
              exact hxy (subset_closure (Function.mem_support.mpr h_ne))
            simp [hg_xy]
          · have h_mol_zero : stdMollifierPi n ε y = 0 := Function.notMem_support.mp hy_supp
            simp [h_mol_zero]
        rw [h_integrand_zero, integral_zero]
      linarith
    have h_norm_eq_restrict : eLpNorm (mollify ε g - g) 2 (stdGaussianE n) =
        eLpNorm (mollify ε g - g) 2 ((stdGaussianE n).restrict K) := by
      symm
      apply eLpNorm_restrict_eq_of_support_subset
      intro x hx
      rw [Function.mem_support] at hx
      by_contra hx_not_in
      exact hx (h_supp_diff x hx_not_in)
    rw [h_norm_eq_restrict]
    have h_supp_diff' : ∀ x, x ∉ K → mollify ε g x - g x = 0 := fun x hx => by
      have := h_supp_diff x hx
      simp only [Pi.sub_apply] at this
      exact this
    calc eLpNorm (mollify ε g - g) 2 ((stdGaussianE n).restrict K)
        ≤ eLpNorm (fun _ => η) 2 ((stdGaussianE n).restrict K) := by
          apply eLpNorm_mono_ae
          filter_upwards with x
          rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos hη_pos]
          simp only [Pi.sub_apply]
          by_cases hx_in : x ∈ K
          · specialize h_pointwise x hx_in
            rw [dist_eq_norm, Real.norm_eq_abs] at h_pointwise
            exact le_of_lt h_pointwise
          · rw [h_supp_diff' x hx_in, abs_zero]
            exact hη_pos.le
      _ = ENNReal.ofReal |η| * ((stdGaussianE n).restrict K Set.univ) ^ (1/2 : ℝ) := by
          have h_μ_ne_zero : (stdGaussianE n).restrict K ≠ 0 := by
            intro h_eq_zero
            have : (stdGaussianE n).restrict K Set.univ = 0 := by simp [h_eq_zero]
            rw [Measure.restrict_apply_univ] at this
            exact hμK_pos.ne' this
          have h_eq := eLpNorm_const η (by norm_num : (2 : ℝ≥0∞) ≠ 0) h_μ_ne_zero
          simp only [ENNReal.toReal_ofNat] at h_eq
          rw [h_eq]
          congr 1
          exact Real.enorm_eq_ofReal_abs η
      _ = ENNReal.ofReal η * (stdGaussianE n K) ^ (1/2 : ℝ) := by
          rw [abs_of_pos hη_pos, Measure.restrict_apply_univ]
      _ ≤ δ := by
          rw [hη_def]
          have h_denom_pos : 0 < 2 * (stdGaussianE n K).toReal.sqrt + 1 := by linarith [h_sqrt_pos]
          have h1 : (stdGaussianE n K) ^ (1/2 : ℝ) =
              ENNReal.ofReal ((stdGaussianE n K).toReal.sqrt) := by
            have h_rpow_ne_top : (stdGaussianE n K) ^ (1/2 : ℝ) ≠ ⊤ :=
              ENNReal.rpow_ne_top_of_nonneg (by norm_num : 0 ≤ (1:ℝ)/2) hμK_finite.ne
            rw [Real.sqrt_eq_rpow]
            rw [ENNReal.toReal_rpow]
            exact (ENNReal.ofReal_toReal h_rpow_ne_top).symm
          rw [h1]
          rw [← ENNReal.ofReal_mul (by linarith : 0 ≤ δr / (2 * (stdGaussianE n K).toReal.sqrt + 1))]
          calc ENNReal.ofReal (δr / (2 * (stdGaussianE n K).toReal.sqrt + 1) * (stdGaussianE n K).toReal.sqrt)
              ≤ ENNReal.ofReal δr := by
                apply ENNReal.ofReal_le_ofReal
                calc (δr / (2 * (stdGaussianE n K).toReal.sqrt + 1)) * (stdGaussianE n K).toReal.sqrt
                    = δr * ((stdGaussianE n K).toReal.sqrt / (2 * (stdGaussianE n K).toReal.sqrt + 1)) := by ring
                  _ ≤ δr * 1 := by
                      apply mul_le_mul_of_nonneg_left _ hδr_pos.le
                      rw [div_le_one h_denom_pos]
                      calc (stdGaussianE n K).toReal.sqrt
                          ≤ 2 * (stdGaussianE n K).toReal.sqrt := by linarith [h_sqrt_pos]
                        _ ≤ 2 * (stdGaussianE n K).toReal.sqrt + 1 := by linarith
                  _ = δr := mul_one _
            _ = δ := ENNReal.ofReal_toReal hδ_ne_top

/-- Mollification convergence in Gaussian L²: For differentiable compactly supported functions,
    mollification converges in L²(γ).

    **Key insight**: Differentiability implies continuity, so we can use
    `mollify_L2_convergence_gaussian_continuous` which only needs continuity.
    The proof uses uniform convergence on compact sets + finite Gaussian measure,
    avoiding the need for measure comparison constants.

    Note: The L² and gradient hypotheses are kept for compatibility but aren't
    needed for the proof (they would be needed for a rate-of-convergence bound). -/
lemma mollify_L2_convergence_gaussian {n : ℕ} {g : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hg_supp : tsupport g ⊆ Metric.closedBall 0 (2 * R))
    (_hg : MemLp g 2 (stdGaussianE n)) (hg_diff : Differentiable ℝ g)
    (_hg_grad : MemLp (fun x => fderiv ℝ g x) 2 (stdGaussianE n)) :
    Filter.Tendsto (fun ε => eLpNorm (mollify ε g - g) 2 (stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  have hg_cont : Continuous g := hg_diff.continuous
  exact mollify_L2_convergence_gaussian_continuous hR hg_cont hg_supp

/-! ### Main Mollification Theorem -/

/-- Helper: The support of f * smoothCutoffR R is contained in the closed ball of radius 2R.
    This follows from the compact support of smoothCutoffR. -/
lemma cutoff_product_support {f : E n → ℝ} {R : ℝ} (hR : 0 < R) :
    tsupport (f * smoothCutoffR R) ⊆ Metric.closedBall 0 (2 * R) := by
  apply subset_trans tsupport_mul_subset_right
  unfold tsupport
  rw [Metric.isClosed_closedBall.closure_subset_iff]
  exact smoothCutoffR_support_subset hR

/-- Helper: f * smoothCutoffR R is continuous if f is continuous.
    Uses the fact that smoothCutoffR is smooth hence continuous. -/
lemma cutoff_product_continuous {f : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hf_cont : Continuous f) : Continuous (f * smoothCutoffR R) := by
  apply Continuous.mul hf_cont
  exact (smoothCutoffR_contDiff hR).continuous

/-- Helper: f * smoothCutoffR R is differentiable if f is differentiable. -/
lemma cutoff_product_differentiable {f : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hf_diff : Differentiable ℝ f) : Differentiable ℝ (f * smoothCutoffR R) := by
  apply Differentiable.mul hf_diff
  exact (smoothCutoffR_contDiff hR).differentiable (WithTop.coe_ne_zero.mpr ENat.top_ne_zero)

/-- Helper: The gradient of f * smoothCutoffR R is continuous if f has continuous gradient
    and f is differentiable. -/
lemma cutoff_product_fderiv_continuous {f : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hf_diff : Differentiable ℝ f)
    (hf_grad_cont : Continuous (fun x => fderiv ℝ f x)) :
    Continuous (fun x => fderiv ℝ (f * smoothCutoffR R) x) := by
  have hχ_diff := (@smoothCutoffR_contDiff n R hR).differentiable (WithTop.coe_ne_zero.mpr ENat.top_ne_zero)
  have hχ_cont : Continuous (fun x => smoothCutoffR R x) := (@smoothCutoffR_contDiff n R hR).continuous
  have hχ_fderiv_cont : Continuous (fun x => fderiv ℝ (smoothCutoffR R) x) :=
    (@smoothCutoffR_contDiff n R hR).continuous_fderiv (WithTop.coe_ne_zero.mpr ENat.top_ne_zero)
  have h_eq : ∀ x, fderiv ℝ (f * smoothCutoffR R) x =
      (smoothCutoffR R x) • (fderiv ℝ f x) + (f x) • (fderiv ℝ (smoothCutoffR R) x) := by
    intro x
    ext v
    simp only [add_apply, smul_apply]
    rw [fderiv_mul (hf_diff x) (hχ_diff x)]
    simp only [add_apply, smul_apply, smul_eq_mul]
    ring
  simp_rw [h_eq]
  apply Continuous.add
  · apply Continuous.smul hχ_cont hf_grad_cont
  · apply Continuous.smul hf_diff.continuous hχ_fderiv_cont

/-- Helper: f * smoothCutoffR R is C¹ if f is differentiable with continuous gradient. -/
lemma cutoff_product_contDiff_one {f : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hf_diff : Differentiable ℝ f)
    (hf_grad_cont : Continuous (fun x => fderiv ℝ f x)) :
    ContDiff ℝ 1 (f * smoothCutoffR R) := by
  rw [contDiff_one_iff_fderiv]
  exact ⟨cutoff_product_differentiable hR hf_diff,
         cutoff_product_fderiv_continuous hR hf_diff hf_grad_cont⟩

/-- Helper: f * smoothCutoffR R has compact support contained in closedBall 0 (2R).
    This gives HasCompactSupport. -/
lemma cutoff_product_hasCompactSupport {f : E n → ℝ} {R : ℝ} (hR : 0 < R) :
    HasCompactSupport (f * smoothCutoffR R) := by
  apply HasCompactSupport.of_support_subset_isCompact (isCompact_closedBall 0 (2 * R))
  intro x hx
  exact cutoff_product_support hR (subset_closure hx)

/-- The derivative-convolution commutation lemma for our mollification.
    For C¹ functions g with compact support, fderiv (mollify ε g) = mollify ε (fderiv g)
    in the appropriate sense. -/
lemma fderiv_mollify_eq_mollify_fderiv_apply {g : E n → ℝ} {ε : ℝ} (hε : 0 < ε)
    (hg_c1 : ContDiff ℝ 1 g) (hg_supp : HasCompactSupport g) (x : E n) (v : E n) :
    fderiv ℝ (mollify ε g) x v = mollify ε (fun y => fderiv ℝ g y v) x := by
  rw [mollify_eq_convolution g]
  have hρ_int : LocallyIntegrable (stdMollifierPi n ε) volume :=
    (stdMollifierPi_contDiff hε).continuous.locallyIntegrable
  have h_fderiv := HasCompactSupport.hasFDerivAt_convolution_left mulLR hg_supp hg_c1 hρ_int x
  rw [HasFDerivAt.fderiv h_fderiv]
  simp only [mollify, convolution]
  have h_eq : ∀ t, (((ContinuousLinearMap.precompL (E n) mulLR) (fderiv ℝ g t))
      (stdMollifierPi n ε (x - t))) v = (fderiv ℝ g t v) * (stdMollifierPi n ε (x - t)) := by
    intro t
    rw [ContinuousLinearMap.precompL_apply, mulLR, ContinuousLinearMap.mul_apply']
  have hfderiv_supp : HasCompactSupport (fderiv ℝ g) := hg_supp.fderiv ℝ
  have hfderiv_cont : Continuous (fderiv ℝ g) := hg_c1.continuous_fderiv one_ne_zero
  have h_conv_exists : ConvolutionExistsAt (fderiv ℝ g) (stdMollifierPi n ε) x
      (ContinuousLinearMap.precompL (E n) mulLR) volume :=
    hfderiv_supp.convolutionExists_left (ContinuousLinearMap.precompL (E n) mulLR)
      hfderiv_cont hρ_int x
  have h_convex : MeasureTheory.Integrable
      (fun t => ((ContinuousLinearMap.precompL (E n) mulLR) (fderiv ℝ g t))
        (stdMollifierPi n ε (x - t))) volume := h_conv_exists.integrable
  rw [ContinuousLinearMap.integral_apply h_convex v]
  simp_rw [h_eq]
  rw [← MeasureTheory.integral_sub_left_eq_self
      (fun t => (fderiv ℝ g t) v * stdMollifierPi n ε (x - t)) volume x]
  congr 1
  ext y
  simp only [sub_sub_cancel]

/-- Gradient of mollification converges in L²(γ) for C¹ functions with compact support.

    **Key insight**: Uses ∇(f ⋆ ρ_ε) = (∇f) ⋆ ρ_ε (derivative-convolution commutation) and
    finite-dimensional compactness of the unit sphere to get uniform convergence in v. -/
lemma gradient_L2_convergence_gaussian {g : E n → ℝ} {R : ℝ} (hR : 0 < R)
    (hg_diff : Differentiable ℝ g)
    (hg_grad_cont : Continuous (fun x => fderiv ℝ g x))
    (hg_supp : tsupport g ⊆ Metric.closedBall 0 (2 * R)) :
    Filter.Tendsto (fun ε => eLpNorm (fun x => ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖) 2 (stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  have hg_c1 : ContDiff ℝ 1 g := contDiff_one_iff_fderiv.mpr ⟨hg_diff, hg_grad_cont⟩
  have hg_compact : HasCompactSupport g :=
    HasCompactSupport.of_support_subset_isCompact (isCompact_closedBall 0 (2 * R))
      (fun x hx => hg_supp (subset_closure hx))
  set K := Metric.closedBall (0 : E n) (2 * R + Real.sqrt n + 1) with hK_def
  have hK_compact : IsCompact K := isCompact_closedBall 0 (2 * R + Real.sqrt n + 1)
  have hμK_finite : stdGaussianE n K < ⊤ := by
    calc stdGaussianE n K ≤ stdGaussianE n Set.univ := measure_mono (Set.subset_univ _)
      _ = 1 := measure_univ
      _ < ⊤ := ENNReal.one_lt_top
  have h_supp_in_K : ∀ ε, 0 < ε → ε < 1 → ∀ x, x ∉ K →
      fderiv ℝ (mollify ε g) x = 0 ∧ fderiv ℝ g x = 0 := by
    intro ε hε_pos hε_lt_1 x hx
    constructor
    · have h_mol_supp : tsupport (mollify ε g) ⊆ Metric.closedBall 0 (2 * R + ε * Real.sqrt n) :=
        mollify_compact_support hR hg_supp hε_pos
      have h_mol_in_K : tsupport (mollify ε g) ⊆ K := by
        apply subset_trans h_mol_supp
        apply Metric.closedBall_subset_closedBall
        have h1 : ε * Real.sqrt n ≤ Real.sqrt n := by
          calc ε * Real.sqrt n ≤ 1 * Real.sqrt n := by nlinarith [Real.sqrt_nonneg n, hε_lt_1]
            _ = Real.sqrt n := one_mul _
        linarith [Real.sqrt_nonneg n]
      have hx_not_supp : x ∉ tsupport (mollify ε g) := fun h => hx (h_mol_in_K h)
      have h_zero_nbhd : mollify ε g =ᶠ[nhds x] (fun _ => (0 : ℝ)) := by
        rw [Filter.eventuallyEq_iff_exists_mem]
        use (tsupport (mollify ε g))ᶜ
        constructor
        · exact (isClosed_tsupport _).isOpen_compl.mem_nhds hx_not_supp
        · intro y hy
          exact (notMem_tsupport_iff_eventuallyEq.mp hy).self_of_nhds
      rw [h_zero_nbhd.fderiv_eq]
      exact fderiv_const_apply 0
    · have hx_not_in_2R : x ∉ Metric.closedBall 0 (2 * R) := by
        intro hx_in
        have hsqrt_nonneg : 0 ≤ Real.sqrt n := Real.sqrt_nonneg n
        have : x ∈ K := Metric.closedBall_subset_closedBall (by linarith) hx_in
        exact hx this
      have hx_not_supp : x ∉ tsupport g := fun h => hx_not_in_2R (hg_supp h)
      have h_zero_nbhd : g =ᶠ[nhds x] (fun _ => (0 : ℝ)) := by
        rw [Filter.eventuallyEq_iff_exists_mem]
        use (tsupport g)ᶜ
        constructor
        · exact (isClosed_tsupport _).isOpen_compl.mem_nhds hx_not_supp
        · intro y hy
          exact (notMem_tsupport_iff_eventuallyEq.mp hy).self_of_nhds
      rw [h_zero_nbhd.fderiv_eq]
      exact fderiv_const_apply 0
  have h_unif_conv_on_K : ∀ δ > 0, ∃ ε₀ > 0, ∀ ε, 0 < ε → ε < ε₀ → ∀ x ∈ K,
      ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖ < δ := by
    intro δ hδ
    set S := Metric.sphere (0 : E n) 1 with hS_def
    have hS_compact : IsCompact S := isCompact_sphere 0 1
    have h_joint_cont : Continuous (fun p : E n × E n => fderiv ℝ g p.1 p.2) := by
      exact (hg_grad_cont.comp continuous_fst).clm_apply continuous_snd
    have h_fv_cont : ∀ v, Continuous (fun y => fderiv ℝ g y v) := by
      intro v
      have h : Continuous (fun y : E n => (y, v)) :=
        Continuous.prodMk continuous_id (continuous_const (y := v))
      exact h_joint_cont.comp h
    have h_fderiv_unif_cont : UniformContinuousOn (fun x => fderiv ℝ g x) K := by
      exact hK_compact.uniformContinuousOn_of_continuous hg_grad_cont.continuousOn
    by_cases hn : n = 0
    · use 1, one_pos
      intro ε _ _ x _
      subst hn
      have : (fderiv ℝ (mollify ε g) x - fderiv ℝ g x) = 0 := by
        ext v
        have hv : v = 0 := Subsingleton.elim v 0
        simp [hv]
      rw [this, norm_zero]
      exact hδ
    · push Not at hn
      let b := EuclideanSpace.basisFun (Fin n) ℝ
      have h_fi_cont : ∀ i, Continuous (fun y => fderiv ℝ g y (b i)) := fun i => h_fv_cont (b i)
      set K' := K + Metric.closedBall (0 : E n) 1
      have hK'_compact : IsCompact K' := hK_compact.add (isCompact_closedBall 0 1)
      have hK_sub_K' : K ⊆ K' := by
        intro x hx
        rw [Set.mem_add]
        exact ⟨x, hx, 0, Metric.mem_closedBall_self zero_le_one, add_zero x⟩
      have h_unif_i : ∀ i, TendstoUniformlyOn (fun ε => mollify ε (fun y => fderiv ℝ g y (b i)))
          (fun y => fderiv ℝ g y (b i)) (nhdsWithin 0 (Set.Ioi 0)) K' := by
        intro i
        exact mollify_tendsto_uniformly_on_compact hK'_compact (h_fi_cont i)

      have h_eps_i : ∀ i, ∀ δ' > 0, ∀ᶠ ε in nhdsWithin (0 : ℝ) (Set.Ioi 0), ∀ x ∈ K',
          dist (fderiv ℝ g x (b i)) (mollify ε (fun y => fderiv ℝ g y (b i)) x) < δ' := by
        intro i δ' hδ'
        have h_i := h_unif_i i
        rw [Metric.tendstoUniformlyOn_iff] at h_i
        exact h_i δ' hδ'
      set δ' := δ / (Real.sqrt n + 1) with hδ'_def
      have hδ'_pos : δ' > 0 := div_pos hδ (by linarith [Real.sqrt_nonneg n])
      have h_finite_inter : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Set.Ioi 0),
          ∀ i, ∀ x ∈ K', dist (fderiv ℝ g x (b i)) (mollify ε (fun y => fderiv ℝ g y (b i)) x) < δ' := by
        have := Filter.eventually_all.mpr (fun i => h_eps_i i δ' hδ'_pos)
        filter_upwards [this] with ε hε i x hx
        exact hε i x hx
      rw [eventually_nhdsWithin_iff] at h_finite_inter
      obtain ⟨U, hU_nhds, hU_cond⟩ := Filter.eventually_iff_exists_mem.mp h_finite_inter
      obtain ⟨ε₀, hε₀_pos, hε₀_ball⟩ := Metric.nhds_basis_ball.mem_iff.mp hU_nhds
      use ε₀, hε₀_pos
      intro ε hε_pos hε_lt_ε₀ x hx
      have hε_in_U : ε ∈ U := by
        apply hε₀_ball
        simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos hε_pos]
        exact hε_lt_ε₀
      have hε_in_Ioi : ε ∈ Set.Ioi (0 : ℝ) := hε_pos
      have h_bound_basis : ∀ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| < δ' := by
        intro i
        have := hU_cond ε hε_in_U hε_in_Ioi i x (hK_sub_K' hx)
        rwa [Real.dist_eq, abs_sub_comm] at this
      have h_commute : ∀ i, fderiv ℝ (mollify ε g) x (b i) = mollify ε (fun y => fderiv ℝ g y (b i)) x := by
        intro i
        exact fderiv_mollify_eq_mollify_fderiv_apply hε_pos hg_c1 hg_compact x (b i)
      have h_bound_sqrt : Real.sqrt n * δ' < δ := by
        have h2 : (Real.sqrt n + 1) * δ' = δ := by rw [hδ'_def]; field_simp
        nlinarith [Real.sqrt_nonneg n, hδ'_pos]

      apply lt_of_le_of_lt
      · apply ContinuousLinearMap.opNorm_le_of_unit_norm (C := Real.sqrt n * δ')
        · apply mul_nonneg (Real.sqrt_nonneg n) (le_of_lt hδ'_pos)
        · intro v hv_norm
          rw [Real.norm_eq_abs]
          have hv_expand : v = ∑ i, (b.repr v i) • (b i) := (b.sum_repr v).symm
          have h_diff_apply : (fderiv ℝ (mollify ε g) x - fderiv ℝ g x) v =
              ∑ i, (b.repr v i) * ((fderiv ℝ (mollify ε g) x - fderiv ℝ g x) (b i)) := by
            conv_lhs => rw [hv_expand]
            simp only [map_sum, map_smul, smul_eq_mul]
          rw [h_diff_apply]
          have h_basis_diff : ∀ i, (fderiv ℝ (mollify ε g) x - fderiv ℝ g x) (b i) =
              mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i) := by
            intro i
            simp only [sub_apply, h_commute i]
          simp_rw [h_basis_diff]
          have h1 : |∑ i : Fin n, b.repr v i *
                  (mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i))|
              ≤ ∑ i : Fin n, |b.repr v i| *
                  |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| := by
            calc |∑ i, b.repr v i * (mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i))|
                ≤ ∑ i, |b.repr v i * (mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i))| :=
                  Finset.abs_sum_le_sum_abs _ _
              _ = ∑ i, |b.repr v i| * |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| := by
                  congr 1; ext i; exact abs_mul _ _
          have h2 : ∑ i : Fin n, |b.repr v i| *
                |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)|
              ≤ (∑ i, |b.repr v i| ^ 2).sqrt *
                (∑ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| ^ 2).sqrt :=
            Real.sum_mul_le_sqrt_mul_sqrt Finset.univ _ _
          have h_sum_eq_one : (∑ i : Fin n, |b.repr v i| ^ 2).sqrt = 1 := by
            have h_isometry : ‖b.repr v‖ = ‖v‖ := b.repr.norm_map v
            have h_norm_sq := EuclideanSpace.norm_sq_eq (b.repr v)
            have h_sum : (∑ i, |b.repr v i| ^ 2) = ‖b.repr v‖ ^ 2 := by
              rw [h_norm_sq]
              simp only [Real.norm_eq_abs, sq_abs]
            rw [h_sum, h_isometry, hv_norm, one_pow, Real.sqrt_one]
          have h_sum_bound : (∑ i : Fin n, |mollify ε (fun y => fderiv ℝ g y (b i)) x -
                fderiv ℝ g x (b i)| ^ 2).sqrt ≤ Real.sqrt n * δ' := by
            have h_le : ∑ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| ^ 2
                ≤ n * δ' ^ 2 := by
              calc ∑ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| ^ 2
                  ≤ ∑ _i : Fin n, δ' ^ 2 := by
                    apply Finset.sum_le_sum
                    intro i _
                    have h_abs_nonneg := abs_nonneg (mollify ε (fun y => fderiv ℝ g y (b i)) x -
                        fderiv ℝ g x (b i))
                    apply sq_le_sq' (by linarith) (le_of_lt (h_bound_basis i))
                _ = n * δ' ^ 2 := by rw [Finset.sum_const, Finset.card_fin]; simp [nsmul_eq_mul]
            calc (∑ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| ^ 2).sqrt
                ≤ (n * δ' ^ 2).sqrt := Real.sqrt_le_sqrt h_le
              _ = Real.sqrt n * δ' := by
                  rw [Real.sqrt_mul (Nat.cast_nonneg n), Real.sqrt_sq (le_of_lt hδ'_pos)]

          calc |∑ i, b.repr v i * (mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i))|
              ≤ ∑ i, |b.repr v i| * |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| := h1
            _ ≤ (∑ i, |b.repr v i| ^ 2).sqrt *
                (∑ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| ^ 2).sqrt := h2
            _ = 1 * (∑ i, |mollify ε (fun y => fderiv ℝ g y (b i)) x - fderiv ℝ g x (b i)| ^ 2).sqrt := by
                rw [h_sum_eq_one]
            _ ≤ 1 * (Real.sqrt n * δ') := by
                apply mul_le_mul_of_nonneg_left h_sum_bound zero_le_one
            _ = Real.sqrt n * δ' := one_mul _
      · exact h_bound_sqrt
  rw [ENNReal.tendsto_nhds_zero]
  intro δ hδ
  by_cases hδ_top : δ = ⊤
  · subst hδ_top
    exact Filter.Eventually.of_forall (fun _ => le_top)
  have hδ_ne_top : δ ≠ ⊤ := hδ_top
  have hδ2 : 0 < δ / 2 := by
    apply ENNReal.div_pos hδ.ne' (by norm_num : (2 : ℝ≥0∞) ≠ ⊤)
  have hδ2_ne_top : δ / 2 ≠ ⊤ := ENNReal.div_ne_top hδ_ne_top (by norm_num)
  set δr := (δ / 2).toReal with hδr_def
  have hδr_pos : 0 < δr := ENNReal.toReal_pos hδ2.ne' hδ2_ne_top
  obtain ⟨ε₀, hε₀_pos, hε₀_unif⟩ := h_unif_conv_on_K δr hδr_pos
  apply Filter.eventually_of_mem (U := Set.Ioo 0 (min ε₀ 1))
  · exact Ioo_mem_nhdsGT (lt_min hε₀_pos one_pos)
  · intro ε hε
    simp only [Set.mem_Ioo] at hε
    have hε_pos : 0 < ε := by linarith
    have hε_lt_ε₀ : ε < ε₀ := lt_of_lt_of_le hε.2 (min_le_left _ _)
    have hε_lt_1 : ε < 1 := lt_of_lt_of_le hε.2 (min_le_right _ _)
    have h_bound : ∀ x, ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖ ≤ δr := by
      intro x
      by_cases hx : x ∈ K
      · exact le_of_lt (hε₀_unif ε hε_pos hε_lt_ε₀ x hx)
      · obtain ⟨h1, h2⟩ := h_supp_in_K ε hε_pos hε_lt_1 x hx
        simp [h1, h2, hδr_pos.le]
    have hμ_ne_zero : stdGaussianE n ≠ 0 := by
      intro h
      have : (1 : ℝ≥0∞) = 0 := by
        calc (1 : ℝ≥0∞) = stdGaussianE n Set.univ := (measure_univ (μ := stdGaussianE n)).symm
          _ = (0 : Measure (E n)) Set.univ := by rw [h]
          _ = 0 := rfl
      exact one_ne_zero this
    calc eLpNorm (fun x => ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖) 2 (stdGaussianE n)
        ≤ eLpNorm (fun _ => δr) 2 (stdGaussianE n) := by
          apply eLpNorm_mono_ae
          filter_upwards with x
          rw [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)]
          rw [Real.norm_eq_abs, abs_of_pos hδr_pos]
          exact h_bound x
      _ = ‖δr‖ₑ * (stdGaussianE n Set.univ) ^ (1 / (2 : ℝ≥0∞).toReal) := by
          rw [eLpNorm_const _ (by norm_num : (2 : ℝ≥0∞) ≠ 0) hμ_ne_zero]
      _ = ENNReal.ofReal δr := by
          rw [Real.enorm_eq_ofReal (le_of_lt hδr_pos)]
          simp only [ENNReal.toReal_ofNat, one_div]
          rw [measure_univ, ENNReal.one_rpow, mul_one]
      _ = δ / 2 := by rw [hδr_def, ENNReal.ofReal_toReal hδ2_ne_top]
      _ ≤ δ := ENNReal.half_le_self

/-- Main mollification theorem: For fixed R, the mollified cutoff f^(R,ε) := (f·χ_R) ⋆ ρ_ε
    converges to f^(R) := f·χ_R in the W^{1,2}(γ) Sobolev norm as ε → 0. -/
lemma tendsto_mollify_W12 {n : ℕ} (f : E n → ℝ) {R : ℝ} (hR : 0 < R)
    (hf_diff : Differentiable ℝ f)
    (hf_grad_cont : Continuous (fun x => fderiv ℝ f x)) :
    Filter.Tendsto (fun ε =>
        GaussianSobolevNormSq n (mollify ε (f * smoothCutoffR R) - f * smoothCutoffR R) (stdGaussianE n))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
  set g := f * smoothCutoffR R
  have hg_supp : tsupport g ⊆ Metric.closedBall 0 (2 * R) := cutoff_product_support hR
  have hg_diff : Differentiable ℝ g := cutoff_product_differentiable hR hf_diff
  have hg_grad_cont : Continuous (fun x => fderiv ℝ g x) :=
    cutoff_product_fderiv_continuous hR hf_diff hf_grad_cont
  have hg_cont : Continuous g := hg_diff.continuous
  simp only [GaussianSobolevNormSq]
  have h0 : (0 : ℝ≥0∞) = 0 + 0 := by simp
  rw [h0]
  apply Filter.Tendsto.add
  · have h_L2 : Filter.Tendsto (fun ε => eLpNorm (mollify ε g - g) 2 (stdGaussianE n))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) :=
      mollify_L2_convergence_gaussian_continuous hR hg_cont hg_supp
    have h_sq : Filter.Tendsto (fun x : ℝ≥0∞ => x ^ (2 : ℕ)) (nhds 0) (nhds 0) := by
      have h_cont := ENNReal.continuous_pow 2
      have h := h_cont.tendsto (0 : ℝ≥0∞)
      simp only [zero_pow (by norm_num : 2 ≠ 0)] at h
      exact h
    exact Filter.Tendsto.comp h_sq h_L2
  · have h_fderiv_eq : ∀ ε > 0, ∀ x,
        ‖fderiv ℝ (mollify ε g - g) x‖ = ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖ := by
      intro ε hε x
      congr 1
      have hm_diff : DifferentiableAt ℝ (mollify ε g) x := by
        have hg_compact : HasCompactSupport g := cutoff_product_hasCompactSupport hR
        have hg_int : LocallyIntegrable g volume := hg_diff.continuous.locallyIntegrable
        exact (mollify_smooth hg_compact hg_int hε).differentiable
          (WithTop.coe_ne_zero.mpr ENat.top_ne_zero) |>.differentiableAt
      have hg_diff_at : DifferentiableAt ℝ g x := hg_diff x
      rw [fderiv_sub hm_diff hg_diff_at]

    -- Now use gradient_L2_convergence_gaussian
    have h_grad : Filter.Tendsto
        (fun ε => eLpNorm (fun x => ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖) 2 (stdGaussianE n))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) :=
      gradient_L2_convergence_gaussian hR hg_diff hg_grad_cont hg_supp

    -- Rewrite using h_fderiv_eq
    have h_eq : ∀ᶠ ε in nhdsWithin 0 (Set.Ioi 0),
        eLpNorm (fun x => ‖fderiv ℝ (mollify ε g - g) x‖) 2 (stdGaussianE n) =
        eLpNorm (fun x => ‖fderiv ℝ (mollify ε g) x - fderiv ℝ g x‖) 2 (stdGaussianE n) := by
      filter_upwards [self_mem_nhdsWithin] with ε hε
      congr 1
      ext x
      exact h_fderiv_eq ε hε x

    -- Tendsto with eventually equal functions
    have h_grad' : Filter.Tendsto
        (fun ε => eLpNorm (fun x => ‖fderiv ℝ (mollify ε g - g) x‖) 2 (stdGaussianE n))
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) :=
      h_grad.congr' (h_eq.mono (fun _ h => h.symm))

    -- Squared norm → 0 when norm → 0
    have h_sq : Filter.Tendsto (fun x : ℝ≥0∞ => x ^ (2 : ℕ)) (nhds 0) (nhds 0) := by
      have h_cont := ENNReal.continuous_pow 2
      have h := h_cont.tendsto (0 : ℝ≥0∞)
      simp only [zero_pow (by norm_num : 2 ≠ 0)] at h
      exact h

    exact Filter.Tendsto.comp h_sq h_grad'

end GaussianSobolev
